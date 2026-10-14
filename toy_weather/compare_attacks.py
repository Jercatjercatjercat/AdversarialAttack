"""Compare random/one-step/PGD on one frozen model, changing one current variable."""
import argparse
import csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from config import DATA, OUTPUT
from perturbations import change_input
from metrics import evaluate_loader
from runtime import configure, load_run, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--checkpoint-dir', type=Path, default=OUTPUT)
    parser.add_argument('--results-dir', type=Path)
    parser.add_argument('--variable', default='t2m')
    parser.add_argument('--budget', type=float, nargs='+', default=[0.1, 0.5], help='Physical max-cell bound for Linf; physical RMS bound for L2')
    parser.add_argument('--norm', choices=['linf', 'l2'], nargs='+', default=['linf', 'l2'])
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--random-trials', type=int, default=3)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if any(v < 0 for v in args.budget) or min(args.steps, args.random_trials) < 1:
        parser.error('Nonnegative budgets and positive counts required.')
    device = configure(device=args.device)
    model, datasets, stats, metadata, lat, lon = load_run(args.checkpoint_dir, args.data, device)
    if args.variable not in metadata['variables']:
        parser.error('Variable was not predicted by this model.')
    output = args.results_dir or args.checkpoint_dir/'attack_comparison'
    if output.exists():
        parser.error('Results folder exists; choose a new --results-dir.')
    output.mkdir(parents=True)
    loader = DataLoader(datasets['test'], batch_size=64)
    variables, units = metadata['variables'], metadata['units']
    index = variables.index(args.variable)
    channel = (metadata['history']-1)*len(variables)+index
    std = np.asarray(stats['std'])
    clean = evaluate_loader(model, loader, stats, variables, units, device)
    save_json(output/'clean_metrics.json', clean)
    records = []
    for norm in args.norm:
        for budget in args.budget:
            radius = budget*(len(lat)*len(lon))**.5 if norm=='l2' else budget
            for method in ('random', 'one_step', 'pgd'):
                for trial in range(args.random_trials if method=='random' else 1):
                    generator = torch.Generator().manual_seed(100+trial)
                    totals = np.zeros(len(variables))
                    delta_squared, cells, max_change, count = 0.0, 0, 0.0, 0
                    for x, y in loader:
                        x, y = x.to(device), y.to(device)
                        changed = change_input(model, x, y, channel, radius/std[index], method, norm, args.steps, generator)
                        with torch.no_grad():
                            totals += (model(changed)-y).cpu().double().square().sum((0, 2, 3)).numpy()
                            delta = (changed[:, channel]-x[:, channel])*std[index]
                            delta_squared += delta.cpu().double().square().sum().item()
                            cells += delta.numel()
                            max_change = max(max_change, delta.abs().max().item())
                            count += y.shape[0]*y.shape[2]*y.shape[3]
                    record = dict(method=method, norm=norm, trial=trial, changed_variable=args.variable,
                                  budget=budget, budget_unit=units[index], physical_radius=radius,
                                  max_cell_change=max_change, rms_input_change=(delta_squared/cells)**.5)
                    for i, name in enumerate(variables):
                        record[f'{name}_rmse'] = float((totals[i]/count)**.5*std[i])
                    records.append(record)
                    summary = ' '.join(f"{name}={record[f'{name}_rmse']:.3f}{units[i]}" for i, name in enumerate(variables))
                    print(f'{method} {norm}, budget={budget}{units[index]}, trial={trial}: {summary}', flush=True)
    save_json(output/'results.json', records)
    save_json(output/'comparison_config.json', dict(prepared_sha256=metadata['prepared_sha256'],
              checkpoint_sha256=__import__('hashlib').sha256((args.checkpoint_dir/'best_model.pt').read_bytes()).hexdigest(),
              variable=args.variable, attack_time='current only', objective='mean normalized MSE over all output variables',
              random_trials=args.random_trials, steps=args.steps, budgets=args.budget, norms=args.norm,
              note='random rows are separate trials; persistence reference uses original inputs'))
    with (output/'results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)


if __name__ == '__main__':
    main()
