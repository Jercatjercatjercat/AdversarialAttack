"""Compare clean rollouts and attacks optimised through different forecast windows."""
import argparse
import csv
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Subset
from config import ROOT, DATA, OUTPUT
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.matplotlib'))
from runtime import configure, load_run, save_json
from attack_evaluation import settings, sample_indices, choose_condition, fingerprint
from rollouts import RolloutDataset
from rollout_evaluation import window_score, trajectory_errors, change_summary, validity_summary
from plot_rollouts import plot_results


def write_table(path, rows):
    """Parallel, labelled records -> CSV with actual newlines on all platforms."""
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--checkpoint-dir', type=Path, default=OUTPUT)
    parser.add_argument('--results-dir', type=Path, default=OUTPUT/'rollout_windows')
    parser.add_argument('--horizon-hours', type=int, default=120)
    parser.add_argument('--loss-windows', type=int, nargs='+', default=[6, 24, 72])
    parser.add_argument('--budget', type=float, default=.1, help='Linf cell cap / L2 RMS bound, in attacked-variable units')
    parser.add_argument('--norm', choices=['linf','l2'], default='linf')
    parser.add_argument('--variable', default='t2m')
    parser.add_argument('--attack-steps', type=int, nargs='+', default=[10, 20])
    parser.add_argument('--step-factors', type=float, nargs='+', default=[1., 2.])
    parser.add_argument('--validation-samples', type=int, default=32)
    parser.add_argument('--test-samples', type=int, default=0, help='0 = all eligible test trajectories')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--random-trials', type=int, default=3)
    parser.add_argument('--cell-cap', type=float, default=1.)
    parser.add_argument('--damage-threshold', type=float, default=.05, help='Descriptive physical RMSE-increase threshold, NOT statistical significance')
    parser.add_argument('--settings', type=Path, help='Reuse an existing validation selection instead of tuning again')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.horizon_hours<6 or args.horizon_hours%6 or any(w<6 or w%6 or w>args.horizon_hours for w in args.loss_windows):
        parser.error('Horizon/windows must be multiples of six hours, and windows cannot exceed the horizon.')
    if min(args.attack_steps+[args.batch_size,args.random_trials])<1 or min(args.step_factors)<=0 or min(args.validation_samples,args.test_samples)<0 or min(args.budget,args.cell_cap)<0 or args.damage_threshold<=0:
        parser.error('Invalid counts, budgets, step factors or threshold.')
    if args.results_dir.exists():
        parser.error('Choose a new results directory; completed experiments are never overwritten.')
    device = configure(device=args.device)
    model, datasets, stats, meta, _, _ = load_run(args.checkpoint_dir, args.data, device)
    if args.variable not in meta['variables']:
        parser.error('Attacked variable is absent from this checkpoint.')
    variable_index = meta['variables'].index(args.variable)
    checkpoint_hash = fingerprint(args.checkpoint_dir/'best_model.pt')
    prepared_hash = fingerprint(args.data)
    windows = sorted(set(args.loss_windows))
    horizon = args.horizon_hours//6
    validation, test = RolloutDataset(datasets['validation'], horizon), RolloutDataset(datasets['test'], horizon)
    vi, ti = sample_indices(validation, args.validation_samples), sample_indices(test, args.test_samples)
    cells = int(np.prod(datasets['test'][0][1].shape[-2:]))
    args.results_dir.mkdir(parents=True)
    save_json(args.results_dir/'run_config.json', {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    print(f'Complete {args.horizon_hours}h cases: validation {len(validation)}, test {len(test)}; selected {len(vi)}/{len(ti)}.', flush=True)
    rows, conditions = [], []
    if args.settings:
        import json
        selection = json.loads(args.settings.read_text())
        expected = dict(split='validation', checkpoint_sha256=checkpoint_hash, prepared_sha256=prepared_hash,
                        variable=args.variable, norm=args.norm, budget=args.budget, cell_cap=args.cell_cap,
                        loss_windows=windows, horizon_hours=args.horizon_hours)
        if any(selection.get(k)!=v for k,v in expected.items()):
            raise ValueError('Selection must be validation-only and match weights, data, windows and constraints.')
        conditions = selection['conditions']
    else:
        for window in windows:
            group = []
            for updates in args.attack_steps:
                for factor in args.step_factors:
                    c = dict(settings('pgd',args.norm,args.budget,cells,updates,factor,1,cell_cap=args.cell_cap), window_hours=window)
                    print(f'Validation: PGD loss={window}h, updates={updates}, factor={factor}', flush=True)
                    r = window_score(model,validation,vi,c,stats,variable_index,device,args.batch_size)
                    rows.append(r); group.append(r)
                    print(f"  normalized window MSE {r['objective_mse']:.6f}; {r['seconds']:.1f}s",flush=True)
            chosen = choose_condition(group,.99)
            fields = set(c)
            chosen = {k:v for k,v in chosen.items() if k in fields}
            conditions.append(dict(chosen,id=f'pgd_w{window:03d}h'))
            conditions.append(dict(settings('one_step',args.norm,args.budget,cells,cell_cap=args.cell_cap),
                                   window_hours=window,id=f'one_step_w{window:03d}h'))
        selection = dict(split='validation',checkpoint_sha256=checkpoint_hash,prepared_sha256=prepared_hash,
            variable=args.variable,norm=args.norm,budget=args.budget,cell_cap=args.cell_cap,
            loss_windows=windows,horizon_hours=args.horizon_hours,validation_indices=vi,
            criterion='Least effort retaining 99% of best observed per-window validation MSE gain',conditions=conditions)
        write_table(args.results_dir/'validation_sweep.csv',rows)
        save_json(args.results_dir/'validation_sweep.json',rows)
    # Write choices BEFORE computing any test errors. No test-driven setting selection.
    save_json(args.results_dir/'selected_settings.json',selection)
    leads = np.arange(1,horizon+1)*6
    arrays = dict(lead_hours=leads, initial_times=test.times[ti].astype(str),
                  base_indices=test.base_indices[ti])
    print('Test: clean rollout and original-state persistence',flush=True)
    clean, persistence, _, seconds = trajectory_errors(model,test,ti,stats,device,args.batch_size)
    arrays.update(clean=clean,persistence=persistence)
    records = [dict(id='clean',seconds=seconds)]
    random_condition = dict(settings('random',args.norm,args.budget,cells,cell_cap=args.cell_cap), window_hours=6)
    random_errors = []
    for trial in range(args.random_trials):
        print(f'Test: random trial {trial+1}',flush=True)
        error,_,delta,seconds = trajectory_errors(model,test,ti,stats,device,args.batch_size,random_condition,
                                                 variable_index,100+trial)
        random_errors.append(error)
        records.append(dict(id=f'random_{trial+1}',seconds=seconds,**change_summary(delta,random_condition)))
    arrays['random'] = np.mean(random_errors,axis=0)
    arrays['random_trials'] = np.stack(random_errors)
    for condition in conditions:
        print(f"Test: {condition['id']}; updates={condition['steps']}",flush=True)
        error,_,delta,seconds = trajectory_errors(model,test,ti,stats,device,args.batch_size,condition,variable_index)
        arrays[condition['id']] = error
        arrays[condition['id']+'_delta'] = delta
        records.append(dict(condition,seconds=seconds,
                            differentiated_forecast_steps_per_case=condition['steps']*(condition['window_hours']//6),
                            **change_summary(delta,condition)))
    np.savez_compressed(args.results_dir/'per_case_errors.npz',**arrays)
    validity, curves = [], []
    for label in ['clean','persistence','random']+[c['id'] for c in conditions]:
        error = arrays[label]
        for lead_index, lead in enumerate(leads):
            for v,name in enumerate(meta['variables']):
                rmse = float(np.sqrt(error[:,lead_index,v].mean()))
                baseline = float(np.sqrt(clean[:,lead_index,v].mean()))
                curves.append(dict(condition=label,lead_hours=int(lead),variable=name,unit=meta['units'][v],
                    rmse=rmse,clean_rmse=baseline,rmse_increase=rmse-baseline,
                    fraction_cases_worse=float(np.mean(error[:,lead_index,v]>clean[:,lead_index,v]))))
        if label not in ('clean','persistence'):
            validity.append(dict(condition=label,**validity_summary(error,clean,leads,variable_index,args.damage_threshold)))
    write_table(args.results_dir/'lead_time_metrics.csv',curves)
    write_table(args.results_dir/'attack_duration.csv',validity)
    save_json(args.results_dir/'conditions.json',records)
    save_json(args.results_dir/'summary.json',dict(samples=len(ti),validation_samples=len(selection['validation_indices']),
        first_initial_time=str(test.times[ti[0]]),last_initial_time=str(test.times[ti[-1]]),
        variable=args.variable,unit=meta['units'][variable_index],horizon_hours=args.horizon_hours,
        loss_windows=windows,initial_channel=(model.history-1)*model.variables+variable_index,
        objective='Mean normalized MSE over ALL six-hour leads inside the chosen loss window',
        attack_duration=validity,
        limitations=['One checkpoint/seed; previously inspected 2022 is exploratory.',
                     'Known future references used to construct attacks; not an operational claim.',
                     'Descriptive duration threshold is not statistical significance or a physical predictability limit.',
                     'CNN was trained one-step; long-horizon error can include substantial model drift.']))
    plot_results(args.results_dir,arrays,conditions,meta['variables'],meta['units'],args.variable,args.damage_threshold)
    if fingerprint(args.checkpoint_dir/'best_model.pt')!=checkpoint_hash or fingerprint(args.data)!=prepared_hash:
        raise RuntimeError('Weights or data changed during the comparison.')
    print(f'Saved frozen-model rollout comparison to {args.results_dir}',flush=True)


if __name__=='__main__':
    main()
