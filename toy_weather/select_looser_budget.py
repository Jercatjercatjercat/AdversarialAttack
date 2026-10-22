"""Choose a demonstration budget on validation, preserving every pilot result."""
import argparse
import math
from pathlib import Path
from config import DATA, OUTPUT
from runtime import configure, load_run, save_json
from attack_evaluation import settings, evaluate_condition, sample_indices, fingerprint
from rollouts import RolloutDataset
from rollout_evaluation import window_score


def choose_budget(rows, fixed_gap=.005, rollout_gap=.01):
    """Validation summaries -> smallest budget meeting BOTH physical-RMSE gaps.

    This is an educational visibility criterion, not statistical significance.
    A fallback is labelled explicitly if none of the predeclared budgets qualifies.
    """
    ordered = sorted(rows, key=lambda r:r['budget'])
    if not ordered or any(r['split']!='validation' for r in ordered):
        raise ValueError('Budget selection requires validation-only results.')
    eligible = [r for r in ordered if r['fixed_pgd_minus_fgsm_K']>=fixed_gap
                and r['rollout_pgd_minus_fgsm_K']>=rollout_gap]
    return dict(budget=(eligible[0] if eligible else ordered[-1])['budget'],
                criterion_met=bool(eligible), fixed_gap_K=fixed_gap,
                rollout_gap_K=rollout_gap)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,default=OUTPUT/'looser_budget_pilot')
    args=parser.parse_args()
    if args.results_dir.exists():parser.error('Choose a new results directory.')
    args.results_dir.mkdir(parents=True)
    protocol=dict(split='validation',budgets_K=[.25,.5,1.,2.],samples=32,
        fixed_steps=40,fixed_step_factor=1.,rollout_window_hours=72,
        rollout_steps=20,rollout_step_factor=2.,fixed_gap_K=.005,rollout_gap_K=.01,
        criterion='Smallest budget satisfying both gaps; largest tested budget if none qualifies, explicitly labelled.',
        note='Visibility criterion, not statistical significance. Test scores never select the budget.')
    save_json(args.results_dir/'protocol.json',protocol)
    device=configure()
    model,datasets,stats,meta,lat,lon=load_run(OUTPUT,DATA,device)
    checkpoint_hash=fingerprint(OUTPUT/'best_model.pt')
    data_hash=fingerprint(DATA)
    validation=datasets['validation']
    trajectories=RolloutDataset(validation,20)
    vi=sample_indices(validation,32)
    ri=sample_indices(trajectories,32)
    fixed,window,summary=[],[],[]
    for budget in protocol['budgets_K']:
        print(f'Validation budget ±{budget:g} K',flush=True)
        scores={}
        for method in ('one_step','pgd','mi_fgsm'):
            c=settings(method,'linf',budget,len(lat)*len(lon),
                       1 if method=='one_step' else 40,1.,cell_cap=budget)
            r=evaluate_condition(model,validation,vi,stats,meta,'t2m','t2m',c,device)
            fixed.append(r);scores[method]=r['t2m_rmse']
        rolling={}
        for method in ('one_step','pgd'):
            c=dict(settings(method,'linf',budget,len(lat)*len(lon),
                            1 if method=='one_step' else 20,2.,cell_cap=budget),window_hours=72)
            r=window_score(model,trajectories,ri,c,stats,0,device,32)
            window.append(r)
            rolling[method]=math.sqrt(r['objective_mse'])*stats['std'][0]
        row=dict(split='validation',budget=budget,
                 fixed_fgsm_K=scores['one_step'],fixed_pgd_K=scores['pgd'],fixed_mi_fgsm_K=scores['mi_fgsm'],
                 fixed_pgd_minus_fgsm_K=scores['pgd']-scores['one_step'],
                 rollout_fgsm_K=rolling['one_step'],rollout_pgd_K=rolling['pgd'],
                 rollout_pgd_minus_fgsm_K=rolling['pgd']-rolling['one_step'])
        summary.append(row)
        save_json(args.results_dir/'fixed_results.json',fixed)
        save_json(args.results_dir/'rollout_results.json',window)
        save_json(args.results_dir/'summary.json',summary)
        print(row,flush=True)
    selection=dict(choose_budget(summary),split='validation',
        checkpoint_sha256=checkpoint_hash,prepared_sha256=data_hash,
        fixed_validation_indices=vi,rollout_validation_indices=ri,protocol=protocol)
    save_json(args.results_dir/'selected_budget.json',selection)
    if fingerprint(OUTPUT/'best_model.pt')!=checkpoint_hash or fingerprint(DATA)!=data_hash:
        raise RuntimeError('Weights or data changed during validation pilot.')
    print('Selected:',selection['budget'],'K; visibility criterion met:',selection['criterion_met'],flush=True)


if __name__=='__main__':main()
