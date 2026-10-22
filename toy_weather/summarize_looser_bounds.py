"""Audit matching cases and summarize the completed looser-bound comparisons."""
import json
import csv
import argparse
from pathlib import Path
import numpy as np
from config import DATA, OUTPUT
from runtime import save_json
from attack_evaluation import fingerprint
from plot_rollouts import plt


def pilot_plot(pilot):
    """Validation scalar scores -> budget sensitivity; test scores are absent."""
    rows=json.loads((pilot/'summary.json').read_text())
    budgets=[r['budget'] for r in rows]
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),constrained_layout=True)
    axes[0].plot(budgets,[r['fixed_pgd_minus_fgsm_K'] for r in rows],'-o',color='#0072B2',label='PGD − FGSM')
    axes[0].plot(budgets,[r['fixed_mi_fgsm_K']-r['fixed_fgsm_K'] for r in rows],'--s',color='#D55E00',label='MI-FGSM − FGSM')
    axes[0].axhline(.005,color='#777777',ls=':',label='Visibility rule: 0.005 K')
    axes[0].set_title('Six-hour loss; 40 repeated updates')
    axes[1].plot(budgets,[r['rollout_pgd_minus_fgsm_K'] for r in rows],'-o',color='#0072B2',label='PGD − FGSM')
    axes[1].axhline(.01,color='#777777',ls=':',label='Visibility rule: 0.01 K')
    axes[1].set_title('72-hour average loss; 20 PGD updates')
    for ax in axes:
        ax.set(xlabel='Per-cell input limit (K)',ylabel='Additional RMSE above FGSM (K)',xticks=budgets)
        ax.axhline(0,color='#26383C',lw=.6);ax.legend(fontsize=9)
    fig.suptitle('Validation pilot only: all tested limits retained')
    fig.savefig(pilot/'budget_sensitivity.png',dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-dir',type=Path,default=OUTPUT/'looser_budget_pilot')
    parser.add_argument('--fixed-dir',type=Path,default=OUTPUT/'fixed_step_attacks_2K')
    parser.add_argument('--rollout-dir',type=Path,default=OUTPUT/'rollout_windows_2K')
    args=parser.parse_args()
    pilot=args.pilot_dir
    pilot_plot(pilot)
    selection=json.loads((pilot/'selected_budget.json').read_text())
    budget=selection['budget']
    fixed=args.fixed_dir
    rolling=args.rollout_dir
    config=json.loads((fixed/'comparison_config.json').read_text())
    old_config=json.loads((OUTPUT/'fixed_step_attacks/comparison_config.json').read_text())
    assert config['indices']==old_config['indices']
    assert config['checkpoint_sha256']==selection['checkpoint_sha256']==fingerprint(OUTPUT/'best_model.pt')
    assert config['prepared_sha256']==selection['prepared_sha256']==fingerprint(DATA)
    assert all(c['budget']==budget and c['cell_cap']==budget for c in config['conditions'])
    new_selection=json.loads((rolling/'selected_settings.json').read_text())
    assert new_selection['budget']==budget and new_selection['cell_cap']==budget
    assert new_selection['checkpoint_sha256']==selection['checkpoint_sha256']
    assert new_selection['prepared_sha256']==selection['prepared_sha256']
    for p in (fixed,rolling):
        save_json(p/'budget_selection_reference.json',dict(
            source=str(pilot/'selected_budget.json'),
            sha256=fingerprint(pilot/'selected_budget.json'),selection=selection))
    rows, paired=[], []
    with np.load(rolling/'per_case_errors.npz') as data, np.load(OUTPUT/'rollout_windows/per_case_errors.npz') as old:
        np.testing.assert_array_equal(data['initial_times'],old['initial_times'])
        np.testing.assert_array_equal(data['base_indices'],old['base_indices'])
        np.testing.assert_allclose(data['clean'],old['clean'],rtol=1e-6,atol=1e-6)
        np.testing.assert_allclose(data['persistence'],old['persistence'],rtol=1e-6,atol=1e-6)
        for c in new_selection['conditions']:
            k=c['window_hours']//6
            error=data[c['id']][:,:,0]
            assert np.isfinite(error).all()
            assert np.abs(data[c['id']+'_delta']).max()<=budget+1e-5
            rows.append(dict(condition=c['id'],method=c['method'],window_hours=c['window_hours'],
                steps=c['steps'],step_size_K=c['step_size'],
                window_rmse_K=float(np.sqrt(error[:,:k].mean())),
                clean_window_rmse_K=float(np.sqrt(data['clean'][:,:k,0].mean())),
                rmse_at_120h_K=float(np.sqrt(error[:,-1].mean())),
                clean_rmse_at_120h_K=float(np.sqrt(data['clean'][:,-1,0].mean()))))
        for window in (6,24,72):
            k=window//6
            pgd=data[f'pgd_w{window:03d}h'][:,:,0]
            fgsm=data[f'one_step_w{window:03d}h'][:,:,0]
            paired.append(dict(window_hours=window,
                pgd_minus_fgsm_window_rmse_K=float(np.sqrt(pgd[:,:k].mean())-np.sqrt(fgsm[:,:k].mean())),
                pgd_minus_fgsm_120h_rmse_K=float(np.sqrt(pgd[:,-1].mean())-np.sqrt(fgsm[:,-1].mean())),
                fraction_cases_pgd_higher_window_mse=float(np.mean(pgd[:,:k].mean(1)>fgsm[:,:k].mean(1)))))
    with (rolling/'window_objective_scores.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    with (rolling/'paired_method_comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(paired[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(paired)
    fixed_rows=json.loads((fixed/'results.json').read_text())
    fgsm=next(r['t2m_rmse'] for r in fixed_rows if r['method']=='one_step')
    compact=[dict(method=r['method'],steps=r['steps'],rmse_K=r['t2m_rmse'],
                  difference_from_fgsm_K=r['t2m_rmse']-fgsm) for r in fixed_rows if r['method']!='random']
    assert all(r['samples']==1458 and r['max_cell_change']<=budget+1e-5 for r in fixed_rows)
    result=dict(selected_budget_K=budget,multiplier_over_original=budget/.1,
        fixed_samples=1458,rollout_samples=1439,fixed_scores=compact,rollout_scores=rows,paired_scores=paired,
        checks='Same cases and clean forecasts as original; unchanged weights/data; input limits checked.',
        note='Known future targets; exploratory 2022; large budget is not physical-plausibility evidence.')
    save_json(pilot/'completed_test_summary.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
