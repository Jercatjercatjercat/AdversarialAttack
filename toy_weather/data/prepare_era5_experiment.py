"""Prepare a larger ERA5 experiment: train 2020, validate 2021, test 2022.

Run: python -m data.prepare_era5_experiment
Unlike the deleted one-off downloader, this prepares a complete reusable experiment
with explicit split dates. It reuses its local weather file on subsequent runs.
"""
import argparse
from pathlib import Path
import fsspec
import xarray as xr
from config import ROOT
from data.preprocessing import save_prepared

SOURCE = (
    "https://storage.googleapis.com/weatherbench2/datasets/era5/"
    "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
)


def fetch_subset(path):
    """Public six-hourly [T,lon,lat] K -> local [T,lat,lon] T2m maps, 2020--2022.

    Only temperature is selected. Global source chunks may still be transferred
    because the archive stores a whole global map in each spatial chunk.
    """
    print("Opening public ERA5 archive; selecting 2020--2022 temperature...", flush=True)
    store = fsspec.get_mapper(SOURCE)
    with xr.open_zarr(store, consolidated=True, chunks=None) as dataset:
        field = dataset["2m_temperature"].sel(time=slice("2020-01-01", "2022-12-31"))
        field = field.assign_coords(longitude=((field.longitude + 180) % 360) - 180)
        field = field.sortby("longitude").sortby("latitude")
        # Extra source cells surround the final 45..60 N, -10..10 E crop.
        field = field.sel(latitude=slice(42, 63), longitude=slice(-12, 12))
        field = field.transpose("time", "latitude", "longitude").rename("t2m")
        print(f"Loading selected data: {dict(field.sizes)}", flush=True)
        local = field.load().to_dataset()
        local.attrs.update(source="ERA5 via WeatherBench 2 (2020--2022)", source_url=SOURCE)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".partial.nc")
        local.to_netcdf(temporary)
        temporary.replace(path)
    print(f"Saved {path} ({path.stat().st_size / 1024**2:.2f} MiB)", flush=True)


def main():
    """Cache weather maps, then save normalized [N,1,16,16] pairs for fixed dates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(ROOT / "local_data" / "era5_t2m_2020_2022.nc"))
    parser.add_argument("--output", default=str(ROOT / "local_data" / "era5_2020_2022" / "prepared.npz"))
    parser.add_argument("--train-start", default="2020-01-01")
    parser.add_argument("--train-end", default="2021-01-01", help="Exclusive; shorten training without moving evaluation dates")
    args = parser.parse_args()
    path, output = Path(args.input), Path(args.output)
    if output.exists():
        parser.error("Prepared output already exists; choose another --output to preserve it.")
    if output.suffix != ".npz":
        parser.error("--output must end in .npz")
    if not path.exists():
        fetch_subset(path)
    else:
        print(f"Using existing local maps: {path}", flush=True)
    with xr.open_dataset(path) as ds:
        save_prepared(ds, output, source=ds.attrs.get("source", "ERA5 local file"),
                      validation_start="2021-01-01", test_start="2022-01-01",
                      train_start=args.train_start, train_end=args.train_end,
                      bounds=(45, 60, -10, 10), size=16)


if __name__ == "__main__":
    main()
