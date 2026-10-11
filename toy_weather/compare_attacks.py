"""Compare random changes, one-step attacks and PGD on ONE frozen checkpoint.

Linf budget r: max cell change <= r K. L2 budget r: input RMS <= r K, so
flattened L2 radius = r * sqrt(number of cells). These are different allowed sets;
equal RMS bounds help interpret size but do not make the threat models identical.
"""
import argparse
import csv
import hashlib
import math
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from config import run_directory
from runtime import configure, load_run, save_json
from data.era5 import denormalize
from perturbations import random_noise, fgsm, pgd
from plotting import field_panels, comparison_plot


def compare(model, test_data, stats, device, budgets, norms, methods,
            steps=10, random_repeats=3, seed=42, output=None):
    """Dataset yielding x,y [1,H,W] -> one results row per condition/repetition.

    Every condition sees all the SAME test pairs with the SAME frozen model.
    Random trials are averaged later, never selected for their worst performance.
    """
    model.eval()
    model.requires_grad_(False)
    std = stats["std_k"]
    cells = test_data[0][0].numel()
    rows = []
    loader = DataLoader(test_data, batch_size=64, shuffle=False)
    for norm in norms:
        for budget in budgets:
            radius_k = budget if norm == "linf" else budget * math.sqrt(cells)
            epsilon = radius_k / std
            for method in methods:
                repeats = random_repeats if method == "random" else 1
                for trial in range(repeats):
                    generator = torch.Generator().manual_seed(seed + trial)
                    clean_sum = attacked_sum = persistence_sum = change_sum = 0.0
                    max_cell = max_l2 = 0.0
                    count = 0
                    for batch, (x, y) in enumerate(loader):
                        x, y = x.to(device), y.to(device)  # [B,1,H,W].
                        if method == "random":
                            changed = random_noise(x, epsilon, norm, generator)
                        elif method == "fgsm":
                            changed = fgsm(model, x, y, epsilon, norm)
                        else:
                            changed = pgd(model, x, y, epsilon, steps=steps, norm=norm)
                        with torch.no_grad():
                            clean, attacked = model(x), model(changed)
                            clean_sum += (clean - y).square().sum().item()
                            attacked_sum += (attacked - y).square().sum().item()
                            persistence_sum += (x - y).square().sum().item()
                            delta_k = (changed - x) * std
                            change_sum += delta_k.square().sum().item()
                            max_cell = max(max_cell, delta_k.abs().max().item())
                            max_l2 = max(max_l2, delta_k.flatten(1).norm(dim=1).max().item())
                            count += y.numel()
                        if output is not None and batch == 0 and trial == 0:
                            maps = [denormalize(t[0, 0].detach().cpu().numpy(), stats, celsius=True)
                                    for t in (x, changed, y, clean, attacked)]
                            field_panels([maps[0], maps[1], maps[1] - maps[0], maps[2], maps[3], maps[4]],
                                ["Original input", "Changed input", "Input change", "True future",
                                 "Clean forecast", "Changed-input forecast"],
                                Path(output) / f"{method}_{norm}_{budget:g}K.png", differences=(2,))
                    measured = max_cell if norm == "linf" else max_l2
                    if measured > radius_k + max(1e-5, radius_k * 1e-5):
                        raise RuntimeError(f"{method} exceeded the {norm} budget.")
                    clean_rmse = math.sqrt(clean_sum / count) * std
                    changed_rmse = math.sqrt(attacked_sum / count) * std
                    label = "one_step_l2" if method == "fgsm" and norm == "l2" else method
                    row = {"method": label, "norm": norm, "budget_k": budget,
                           "constraint_radius_k": radius_k, "random_trial": trial if method == "random" else -1,
                           "test_pairs": len(test_data), "clean_rmse_k": clean_rmse,
                           "changed_rmse_k": changed_rmse, "rmse_increase_k": changed_rmse - clean_rmse,
                           "persistence_rmse_k": math.sqrt(persistence_sum / count) * std,
                           "max_cell_change_k": max_cell, "rms_input_change_k": math.sqrt(change_sum / count),
                           "max_sample_l2_k": max_l2}
                    rows.append(row)
                    print(f"{label:12s} {norm:4s} budget {budget:g} K | clean {clean_rmse:.4f} K | "
                          f"changed {changed_rmse:.4f} K | max cell {max_cell:.4f} K | "
                          f"input RMS {row['rms_input_change_k']:.4f} K", flush=True)
    return rows


def write_csv(path, rows):
    """Results dictionaries -> a plain comparison table, readable without Python."""
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(legacy=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("synthetic", "era5"), default="synthetic")
    parser.add_argument("--checkpoint-dir", "--output-dir", dest="checkpoint_dir")
    parser.add_argument("--results-dir")
    parser.add_argument("--data")
    parser.add_argument("--methods", nargs="+", choices=("random", "fgsm", "pgd"),
                        default=["fgsm"] if legacy else ["random", "fgsm", "pgd"])
    parser.add_argument("--method", choices=("random", "fgsm", "pgd"), help="Old single-method option")
    parser.add_argument("--norms", nargs="+", choices=("linf", "l2"), default=["linf"] if legacy else ["linf", "l2"])
    parser.add_argument("--budget-k", "--epsilon-k", dest="budgets", type=float, nargs="+",
                        default=[0.1, 0.5, 1.0] if legacy else [0.1, 0.5])
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--random-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42, help="Random noise seed, not model training seed")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if any(not math.isfinite(b) or b < 0 for b in args.budgets) or args.steps < 1 or args.random_repeats < 1:
        parser.error("Use finite nonnegative budgets and positive steps/repeat counts.")
    directory = run_directory(args.dataset, args.checkpoint_dir)
    methods = [args.method] if args.method else args.methods
    default_folder = ("legacy_" + "_".join(methods) + "_comparison") if legacy else "attack_comparison"
    output = Path(args.results_dir) if args.results_dir else directory / default_folder
    output.mkdir(parents=True, exist_ok=True)
    device = configure(args.seed, args.device)
    model, datasets, stats, metadata = load_run(args.dataset, directory, args.data, device)
    config = {"dataset": args.dataset, "source": metadata["source"], "data_metadata": metadata,
              "checkpoint": str((directory / "best_model.pt").resolve()),
              "checkpoint_sha256": hashlib.sha256((directory / "best_model.pt").read_bytes()).hexdigest(),
              "budgets_k": args.budgets, "norms": args.norms, "methods": methods,
              "steps": args.steps, "random_repeats": args.random_repeats, "noise_seed": args.seed,
              "l2_budget_definition": "RMS bound in K; L2 radius = budget * sqrt(cells)"}
    # Do not mix a new configuration into an existing results folder.
    config_path = output / "comparison_config.json"
    if config_path.exists():
        import json
        if json.loads(config_path.read_text()) != config:
            parser.error("Results folder contains a different comparison; choose a new --results-dir.")
    save_json(config_path, config)
    rows = compare(model, datasets["test"], stats, device, args.budgets, args.norms, methods,
                   args.steps, args.random_repeats, args.seed, output)
    save_json(output / "results.json", rows)
    write_csv(output / "results.csv", rows)
    comparison_plot(rows, output / "comparison.png")
    print(f"Saved comparison table and plots to {output}")


if __name__ == "__main__":
    main()
