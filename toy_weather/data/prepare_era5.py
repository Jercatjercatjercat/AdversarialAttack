"""Crop five ERA5 fields, interpolate to 16x16, and save physical values.

Run from toy_weather: python -m data.prepare_era5
A local NetCDF/Zarr with the documented ERA5 names works via --input.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import numpy as np
import xarray as xr
from config import DATA, SOURCE, SOURCE_NAMES, VARIABLES, UNITS


def prepare_fields(ds, size=16, missing='error'):
    """xarray [time,latitude,longitude] fields -> physical [T,5,H,W] and times.

    Weather-specific operations: crop Europe, retain six-hour UTC times, convert
    pressure Pa to hPa and humidity kg/kg to g/kg. Interpolation resizes a grid; it does not invent observations.
    All channels share the same times/grid. Missing maps are rejected or explicitly
    dropped as whole states; individual missing pixels are never filled.
    """
    if size < 2:
        raise ValueError('Use size >= 2.')
    fields = ds[list(SOURCE_NAMES)]
    fields = fields.assign_coords(longitude=((fields.longitude + 180) % 360) - 180)
    fields = fields.sortby('longitude').sortby('latitude').sortby('time')
    for name in ('time', 'latitude', 'longitude'):
        if fields[name].dims != (name,) or len(np.unique(fields[name])) != len(fields[name]):
            raise ValueError('Expected unique, one-dimensional coordinates.')
    if not (float(fields.latitude.min()) <= 45 and float(fields.latitude.max()) >= 60
            and float(fields.longitude.min()) <= -10 and float(fields.longitude.max()) >= 10):
        raise ValueError('Source must cover 45–60 N, 10 W–10 E, including crop edges.')
    exact = ((fields.time.dt.hour % 6 == 0) & (fields.time.dt.minute == 0)
             & (fields.time.dt.second == 0) & (fields.time.dt.microsecond == 0)
             & (fields.time.dt.nanosecond == 0))
    fields = fields.isel(time=exact)
    # Keep a source-cell margin to interpolate without extrapolating the crop edges.
    fields = fields.sel(latitude=slice(42, 63), longitude=slice(-12, 12))
    lat, lon = np.linspace(45, 60, size), np.linspace(-10, 10, size)
    arrays = []
    for name, unit in zip(SOURCE_NAMES, UNITS):
        field = fields[name]
        if name == 'specific_humidity':
            # Moisture at one explicit pressure level, not an average of levels.
            field = field.sel(level=850, drop=True)
        if set(field.dims) != {'time', 'latitude', 'longitude'}:
            raise ValueError(f'{name}: explicitly select any extra levels/members first.')
        raw_unit = str(field.attrs.get('units', '')).lower().strip()
        accepted = {'K': ('k', 'kelvin'), 'm/s': ('m s**-1', 'm s-1', 'm/s'),
                    'hPa': ('pa', 'hpa', 'mbar'),
                    'g/kg': ('kg kg**-1', 'kg kg-1', 'kg/kg', 'g/kg')}
        if raw_unit not in accepted[unit]:
            raise ValueError(f'{name}: expected explicit {unit} units; got {raw_unit!r}.')
        if unit == 'hPa' and raw_unit == 'pa':
            field = field / 100.0
        if unit == 'g/kg' and raw_unit != 'g/kg':
            field = field * 1000.0
        field = field.interp(latitude=lat, longitude=lon).transpose('time', 'latitude', 'longitude')
        arrays.append(field.values.astype('float32'))
    values = np.stack(arrays, axis=1)  # [T,H,W] five times -> [T,5,H,W].
    times = fields.time.values.astype('datetime64[ns]')
    if np.isnat(times).any() or missing not in ('error', 'skip'):
        raise ValueError('Invalid timestamps or missing-state policy.')
    finite = np.isfinite(values).all(axis=(1, 2, 3))
    if not finite.all() and missing == 'error':
        raise ValueError('missing field values; use --missing skip to explicitly exclude whole states.')
    if missing == 'skip':
        values, times = values[finite], times[finite]
    return values, times, lat, lon


def save_prepared(values, times, lat, lon, output, source, boundaries, notes=None):
    """Save physical fields once, rather than duplicate overlapping input histories.

    Statistical fitting happens later on training inputs only. Boundaries are saved
    here so every training/evaluation script uses exactly the same calendar split.
    """
    values = np.asarray(values, dtype='float32')
    times = np.asarray(times, dtype='datetime64[ns]')
    if values.ndim != 4 or values.shape[:2] != (len(times), len(VARIABLES)):
        raise ValueError('Expected physical [T,5,H,W] and matching times.')
    if values.shape[-2:] != (len(lat), len(lon)) or not np.isfinite(values).all():
        raise ValueError('Grid sizes disagree or fields contain missing values.')
    if np.isnat(times).any() or np.any(np.diff(times) <= np.timedelta64(0, 'ns')):
        raise ValueError('Expected unique, increasing timestamps.')
    output = Path(output)
    if output.exists():
        raise FileExistsError('Choose a new --output to preserve the existing dataset.')
    metadata = dict(source=source, variables=list(VARIABLES), units=list(UNITS),
                    boundaries=boundaries, history_hours=[-6, 0], lead_hours=6,
                    test_status=('exploratory: 2022 was inspected in the first iteration'
                                 if boundaries[2].startswith('2022') else
                                 'custom split: independently establish whether this period is unseen'),
                    size=values.shape[-1], interpolation='linear; source 1.5-degree grid',
                    preparation_notes=notes or {})
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.partial.npz')
    np.savez_compressed(temporary, fields=values, times=np.asarray(times, dtype='datetime64[ns]').astype('int64'),
                        latitude=lat, longitude=lon, metadata=json.dumps(metadata))
    temporary.replace(output)
    print(f'Saved {output}: {values.shape}, {output.stat().st_size / 1024**2:.1f} MiB', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', help='Local NetCDF/Zarr; otherwise use public WeatherBench ERA5')
    parser.add_argument('--output', type=Path, default=DATA)
    parser.add_argument('--train-start', default='2015-01-01')
    parser.add_argument('--validation-start', default='2021-01-01')
    parser.add_argument('--test-start', default='2022-01-01')
    parser.add_argument('--end', default='2023-01-01', help='Exclusive end')
    parser.add_argument('--size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=6, help='Concurrent public month downloads')
    parser.add_argument('--missing', choices=['error', 'skip'], default='skip',
                        help='Skip whole invalid states (logged), or reject any missing map')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('Workers must be positive.')
    if args.output.exists():
        parser.error('Output exists; choose a new --output.')
    bounds = [args.train_start, args.validation_start, args.test_start, args.end]
    if not all(np.datetime64(a) < np.datetime64(b) for a, b in zip(bounds, bounds[1:])):
        parser.error('Dates must be strictly increasing.')
    if args.input:
        path = Path(args.input)
        ds = xr.open_zarr(path, consolidated=True, chunks=None) if path.is_dir() else xr.open_dataset(path)
    else:
        ds = xr.open_zarr(SOURCE, consolidated=True, chunks=None)
    collected, collected_times, reports = [], [], []
    with ds:
        available = ds.time.values.astype('datetime64[ns]')
        requested_end = np.datetime64(args.end, 'ns')-np.timedelta64(6, 'h')
        if available.min() > np.datetime64(args.train_start) or available.max() < requested_end:
            raise ValueError('Source does not cover the requested dates; choose covered boundaries.')
        # Month blocks keep humidity's global pressure-level chunks out of RAM together.
        # Cache completed PUBLIC blocks so an interrupted download can resume.
        periods = []
        month = np.datetime64(args.train_start, 'M')
        while month.astype('datetime64[D]') < np.datetime64(args.end):
            start = max(np.datetime64(args.train_start), month.astype('datetime64[D]'))
            end = min(np.datetime64(args.end), (month+np.timedelta64(1, 'M')).astype('datetime64[D]'))
            periods.append((start, end))
            month += np.timedelta64(1, 'M')
        cache = args.output.parent/'.era5_cache'
        if not args.input:
            cache.mkdir(parents=True, exist_ok=True)

        def load_month(period):
            # Each independent worker reads a distinct time block; shared source is read-only.
            start, end = period
            identity = f'{SOURCE}|{start}|{end}|{args.size}|{args.missing}|q850-v1'
            filename = cache/(hashlib.sha256(identity.encode()).hexdigest()[:20]+'.npz')
            if not args.input and filename.exists():
                with np.load(filename, allow_pickle=False) as cached:
                    return (cached['values'], cached['times'].astype('datetime64[ns]'),
                            cached['latitude'], cached['longitude'], json.loads(str(cached['report'])))
            subset = ds.sel(time=slice(str(start), str(end-np.timedelta64(1, 'ns'))))
            values, times, lat, lon = prepare_fields(subset, args.size, args.missing)
            expected = int(((subset.time.dt.hour % 6)==0).sum())
            report = dict(start=str(start), end=str(end), available_states=expected,
                          kept_states=len(times), dropped_states=expected-len(times))
            print(f'{start}..{end}: {len(times)} states kept, {report["dropped_states"]} dropped', flush=True)
            if not args.input:
                temporary = filename.with_suffix('.partial.npz')
                np.savez_compressed(temporary, values=values, times=times.astype('int64'),
                                    latitude=lat, longitude=lon, report=json.dumps(report))
                temporary.replace(filename)
            return values, times, lat, lon, report

        # Independent month readers overlap network waits, not model training.
        with ThreadPoolExecutor(max_workers=args.workers if not args.input else 1) as pool:
            for values, times, lat, lon, report in pool.map(load_month, periods):
                collected.append(values)
                collected_times.append(times)
                reports.append(report)
    notes = dict(missing_policy=args.missing, blocks=reports,
                 dropped_states=sum(r['dropped_states'] for r in reports))
    save_prepared(np.concatenate(collected), np.concatenate(collected_times), lat, lon,
                  args.output, args.input or SOURCE, bounds, notes)



if __name__ == '__main__':
    main()
