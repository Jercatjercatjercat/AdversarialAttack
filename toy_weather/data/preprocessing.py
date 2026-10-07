"""Prepare local ERA5 NetCDF/Zarr, or a clearly labelled non-ERA5 tutorial demo.

Run from toy_weather: python -m data.preprocessing --help
"""
import argparse
import json
from pathlib import Path
import numpy as np
import xarray as xr
from data.era5 import normalize, training_statistics


def prepare_fields(ds, variable="t2m", time="time", latitude="latitude",
                   longitude="longitude", bounds=(45, 60, -10, 10), size=16):
    """Raw DataArray [time,lat,lon] -> cropped [T,1,H,W] Kelvin and coordinates.

    Weather-specific coordinate handling; no temporal averaging or learned statistics.
    Bounds are south,north,west,east in degrees; date-line crossing is not supported.
    """
    if size < 2:
        raise ValueError("Grid size must be at least 2.")
    south, north, west, east = bounds
    if not (-90 <= south < north <= 90 and -180 <= west < east <= 180):
        raise ValueError("Use south<north, west<east, longitudes within [-180,180].")
    field = ds[variable].rename({time: "time", latitude: "latitude", longitude: "longitude"})
    # A singleton level/member can be removed; multiple levels/members need explicit selection.
    for dim in list(field.dims):
        if dim not in ("time", "latitude", "longitude"):
            if field.sizes[dim] != 1:
                raise ValueError(f"Select a single {dim} before preprocessing.")
            field = field.isel({dim: 0}, drop=True)
    field = field.transpose("time", "latitude", "longitude")
    if any(field[c].dims != (c,) for c in ("time", "latitude", "longitude")):
        raise ValueError("Expected a regular grid with one-dimensional coordinates.")
    units = str(field.attrs.get("units", "")).strip().lower()
    if units in ("c", "degc", "celsius", "degree_celsius", "degrees_celsius", "°c"):
        field = field + 273.15
    elif units not in ("k", "degk", "kelvin", "degrees_kelvin", "degree_kelvin"):
        raise ValueError("Temperature units must explicitly be Kelvin or Celsius.")
    field = field.assign_coords(longitude=((field.longitude + 180) % 360) - 180)
    # Sorting works for north-to-south ERA5 latitude and 0..360 longitude alike.
    field = field.sortby("latitude").sortby("longitude").sortby("time")
    if any(np.unique(field[c]).size != field.sizes[c] for c in field.dims):
        raise ValueError("Duplicate coordinates/times: select one consistent data stream.")
    if (south < float(field.latitude.min()) or north > float(field.latitude.max())
            or west < float(field.longitude.min()) or east > float(field.longitude.max())):
        raise ValueError("Requested crop extends outside the available data.")
    # Include one neighbouring source cell beyond each crop edge for interpolation.
    def surrounding_indices(values, lower, upper):
        start = max(0, np.searchsorted(values, lower) - 1)
        stop = min(len(values), np.searchsorted(values, upper, side="right") + 1)
        return slice(start, stop)
    field = field.isel(latitude=surrounding_indices(field.latitude.values, south, north),
                       longitude=surrounding_indices(field.longitude.values, west, east))
    # Keep exact UTC 00/06/12/18 timestamps; never mistake an hourly pair for six hours.
    six_hour = ((field.time.dt.hour % 6 == 0) & (field.time.dt.minute == 0)
                & (field.time.dt.second == 0))
    field = field.isel(time=six_hour)
    latitudes = np.linspace(south, north, size)
    longitudes = np.linspace(west, east, size)
    field = field.interp(latitude=latitudes, longitude=longitudes).load()
    values = field.values.astype("float32")[:, None]  # [T,H,W] -> [T,1,H,W].
    times = field.time.values.astype("datetime64[ns]")
    if not np.isfinite(values).all() or np.isnat(times).any():
        raise ValueError("Missing/invalid temperatures or timestamps; clean the source first.")
    return values, times, latitudes, longitudes


def chronological_pairs(values, times):
    """[T,1,H,W] maps -> disjoint time splits of exact six-hour pairs.

    Split raw timestamps first. Neighbouring pairs may share states WITHIN a split,
    but never across splits. Missing six-hour targets are skipped, not interpolated.
    """
    n = len(times)
    boundaries = (0, int(0.70 * n), int(0.85 * n), n)
    pairs, pair_times = {}, {}
    for name, start, stop in zip(("train", "validation", "test"), boundaries, boundaries[1:]):
        split_times = times[start:stop]
        valid = np.flatnonzero(np.diff(split_times) == np.timedelta64(6, "h"))
        if len(valid) == 0:
            raise ValueError(f"No six-hour pairs in {name}; supply a longer timeline.")
        maps = values[start:stop]
        pairs[name] = (maps[valid], maps[valid + 1])
        pair_times[name] = (split_times[valid], split_times[valid + 1])
    return pairs, pair_times


def save_prepared(ds, output, source="era5", **options):
    """xarray dataset -> normalized on-disk [N,1,H,W] pairs and train-only statistics."""
    values, times, latitudes, longitudes = prepare_fields(ds, **options)
    pairs, pair_times = chronological_pairs(values, times)
    stats = training_statistics(pairs["train"][0])
    metadata = {"source": source, "variable": options.get("variable", "t2m"),
                "lead_hours": 6, "units": "K", "grid_size": len(latitudes),
                "split_counts": {name: len(x) for name, (x, _) in pairs.items()}}
    saved = {"stats": json.dumps(stats), "metadata": json.dumps(metadata),
             "latitudes": latitudes, "longitudes": longitudes}
    for name, (x, y) in pairs.items():
        saved[name + "_x"] = normalize(x, stats).astype("float32")
        saved[name + "_y"] = normalize(y, stats).astype("float32")
        saved[name + "_input_times"], saved[name + "_target_times"] = pair_times[name]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **saved)
    print(f"Saved {output}: {metadata}")
    print(f"Training-only normalization: {stats}")
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Local .nc file or .zarr directory")
    source.add_argument("--tutorial", action="store_true", help="Download NCEP tutorial air temperature, NOT ERA5 T2m")
    parser.add_argument("--output", default="outputs/era5/prepared.npz")
    parser.add_argument("--variable", default="t2m")
    parser.add_argument("--time", default="time")
    parser.add_argument("--latitude", default="latitude")
    parser.add_argument("--longitude", default="longitude")
    parser.add_argument("--bounds", nargs=4, type=float, metavar=("S", "N", "W", "E"))
    parser.add_argument("--size", type=int, default=16)
    args = parser.parse_args()
    if not args.output.endswith(".npz"):
        parser.error("--output must end in .npz")
    if args.tutorial:
        # Download is opt-in; keep the tutorial cache inside this project.
        cache = Path(args.output).parent / "tutorial_cache"
        ds = xr.tutorial.open_dataset("air_temperature", cache_dir=str(cache))
        options = dict(variable="air", time="time", latitude="lat", longitude="lon",
                       bounds=args.bounds or (30, 50, -120, -90), size=args.size)
        label = "NCEP tutorial air temperature (NOT ERA5 T2m)"
    else:
        ds = xr.open_zarr(args.input, chunks=None) if args.input.rstrip("/").endswith(".zarr") else xr.open_dataset(args.input)
        options = dict(variable=args.variable, time=args.time, latitude=args.latitude,
                       longitude=args.longitude, bounds=args.bounds or (45, 60, -10, 10), size=args.size)
        label = "ERA5 local file (user-supplied)"
    try:
        save_prepared(ds, args.output, source=label, **options)
    finally:
        ds.close()


if __name__ == "__main__":
    main()
