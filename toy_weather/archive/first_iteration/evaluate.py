"""Evaluate unseen test pairs and plot several forecasts in degrees Celsius."""
import argparse
from torch.utils.data import DataLoader
from config import run_directory
from runtime import configure, load_run, save_json
from metrics import evaluate_loader
from plotting import forecast_plot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("synthetic", "era5"), default="synthetic")
    parser.add_argument("--data")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--latitude-weighted", action="store_true")
    args = parser.parse_args()
    device = configure(42, args.device)
    output = run_directory(args.dataset, args.output_dir)
    model, datasets, stats, metadata = load_run(args.dataset, output, args.data, device)
    print(f"Source: {metadata['source']}")
    latitudes = metadata.get("latitudes") if args.latitude_weighted else None
    if args.latitude_weighted and latitudes is None:
        parser.error("Latitude weighting needs the geographic ERA5/tutorial grid.")
    metrics = evaluate_loader(model, DataLoader(datasets["test"], batch_size=64),
                              device, stats["std_k"], latitudes)
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")
    save_json(output / "test_metrics.json", metrics)
    for i in range(min(args.examples, len(datasets["test"]))):
        x, y = datasets["test"][i]
        prediction = model(x.unsqueeze(0).to(device))[0]
        forecast_plot(x, y, prediction, stats, output / f"test_example_{i}.png")


if __name__ == "__main__":
    main()
