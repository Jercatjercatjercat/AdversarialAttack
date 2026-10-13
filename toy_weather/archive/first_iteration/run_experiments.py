"""Repeat the SAME three-layer forecasting and attack experiment for model seeds.

Nothing is uploaded or pushed. Each model gets a separate folder. Dataset dates
and synthetic data seed stay fixed; only initialization/shuffling seeds change.
"""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from config import ROOT, OUTPUTS
from compare_attacks import write_csv
from runtime import save_json


def main():
    """Train/evaluate/compare scripts -> per-seed results and aggregate CSV tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("synthetic", "era5"), default="era5")
    parser.add_argument("--data", help="Prepared file; required for ERA5")
    parser.add_argument("--output-root", help="Fresh folder; defaults to outputs/experiments/<dataset>_comparison")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--budget-k", nargs="+", type=float, default=[0.1, 0.5])
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--random-repeats", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.dataset == "era5" and not args.data:
        parser.error("Pass --data with the prepared ERA5 file.")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Use distinct model seeds.")
    output = (Path(args.output_root) if args.output_root else OUTPUTS / "experiments" / (args.dataset + "_comparison")).resolve()
    if output.exists():
        parser.error("Output root already exists; use a fresh --output-root to preserve previous runs.")
    data_path = str(Path(args.data).resolve()) if args.data else None
    output.mkdir(parents=True)
    save_json(output / "experiment_config.json", vars(args))
    all_rows, clean_rows = [], []
    for seed in args.seeds:
        directory = output / f"seed_{seed}"
        common = ["--dataset", args.dataset, "--output-dir", str(directory), "--device", args.device]
        if data_path:
            common += ["--data", data_path]
        def run(script, options):
            # Reuse the active environment. Each script keeps its original simple role.
            subprocess.run([sys.executable, "-B", str(ROOT / script)] + options, cwd=ROOT, check=True)
        print(f"Training fixed three-layer CNN with model seed {seed}", flush=True)
        run("train.py", common + ["--seed", str(seed), "--data-seed", "42", "--epochs", str(args.epochs)])
        run("evaluate.py", common + ["--examples", "1"])
        run("check_input_gradient.py", common)
        run("compare_attacks.py", common + ["--budget-k"] + list(map(str, args.budget_k))
            + ["--steps", str(args.steps), "--random-repeats", str(args.random_repeats), "--seed", "42"])
        metrics = json.loads((directory / "test_metrics.json").read_text())
        clean_rows.append({"model_seed": seed, "cnn_rmse_k": metrics["rmse_k"],
                           "persistence_rmse_k": metrics["persistence_rmse_k"]})
        results = json.loads((directory / "attack_comparison" / "results.json").read_text())
        all_rows.extend({"model_seed": seed, **r} for r in results)
    write_csv(output / "all_results.csv", all_rows)
    write_csv(output / "clean_models.csv", clean_rows)
    # Average random trials WITHIN each trained model before summarizing model seeds.
    # Otherwise random gets 9 pseudo-independent observations while PGD gets only 3.
    groups = {}
    for row in all_rows:
        key = (row["method"], row["norm"], row["budget_k"])
        groups.setdefault(key, {}).setdefault(row["model_seed"], []).append(row)
    summary = []
    for (method, norm, budget), by_seed in groups.items():
        record = {"method": method, "norm": norm, "budget_k": budget, "model_count": len(by_seed)}
        for metric in ("clean_rmse_k", "changed_rmse_k", "rmse_increase_k", "max_cell_change_k", "rms_input_change_k"):
            per_model = [statistics.mean(r[metric] for r in trials) for trials in by_seed.values()]
            record[metric + "_mean"] = statistics.mean(per_model)
            record[metric + "_std_across_models"] = statistics.pstdev(per_model)
        summary.append(record)
    write_csv(output / "summary.csv", summary)
    save_json(output / "summary.json", summary)
    print(f"Saved systematic comparison tables to {output}", flush=True)


if __name__ == "__main__":
    main()
