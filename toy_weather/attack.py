"""Compare clean forecasts with FGSM/PGD forecasts under temperature-change limits."""
import argparse
import torch
from torch.utils.data import DataLoader
from attacks import fgsm, pgd
from config import run_directory
from runtime import configure, load_run, save_json
from data.era5 import denormalize
from plotting import field_panels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("synthetic", "era5"), default="synthetic")
    parser.add_argument("--method", choices=("fgsm", "pgd"), default="fgsm")
    parser.add_argument("--epsilon-k", type=float, nargs="+", default=[0.1, 0.5, 1.0])
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--data")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if any(e < 0 for e in args.epsilon_k) or args.steps < 1:
        parser.error("Use nonnegative temperature limits and positive steps.")
    device = configure(42, args.device)
    output = run_directory(args.dataset, args.output_dir)
    model, datasets, stats, metadata = load_run(args.dataset, output, args.data, device)
    print(f"Source: {metadata['source']}")
    loader = DataLoader(datasets["test"], batch_size=64)
    results = []
    for epsilon_k in args.epsilon_k:
        # x is normalized: a 0.1 K change becomes 0.1 / std_K normalized units.
        epsilon = epsilon_k / stats["std_k"]
        clean_sum, attacked_sum, count, maximum_change = 0.0, 0.0, 0, 0.0
        for batch, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            changed = (fgsm(model, x, y, epsilon) if args.method == "fgsm"
                       else pgd(model, x, y, epsilon, args.steps))
            with torch.no_grad():
                clean_prediction, attacked_prediction = model(x), model(changed)
                clean_sum += (clean_prediction - y).square().sum().item()
                attacked_sum += (attacked_prediction - y).square().sum().item()
                count += y.numel()
                maximum_change = max(maximum_change, (changed - x).abs().max().item() * stats["std_k"])
            if batch == 0:
                maps = [denormalize(t[0, 0].detach().cpu().numpy(), stats, celsius=True)
                        for t in (x, changed, y, clean_prediction, attacked_prediction)]
                field_panels([maps[0], maps[1], maps[1] - maps[0], maps[2], maps[3], maps[4]],
                    ["Original input", "Changed input", "Input change", "True future",
                     "Clean forecast", "Attacked forecast"],
                    output / f"{args.method}_{epsilon_k:g}K.png", differences=(2,))
        row = {"epsilon_k": epsilon_k, "clean_rmse_k": (clean_sum / count)**0.5 * stats["std_k"],
               "attacked_rmse_k": (attacked_sum / count)**0.5 * stats["std_k"],
               "max_input_change_k": maximum_change}
        if maximum_change > epsilon_k + 1e-5:
            raise RuntimeError("Attack exceeded its temperature-change limit.")
        results.append(row)
        print(f"{args.method.upper()} limit {epsilon_k:g} K | clean RMSE {row['clean_rmse_k']:.4f} K | "
              f"attacked RMSE {row['attacked_rmse_k']:.4f} K | max change {maximum_change:.4f} K")
    save_json(output / f"{args.method}_metrics.json", {"source": metadata["source"], "results": results})


if __name__ == "__main__":
    main()
