"""Sweep attack effort on VALIDATION, save choices, then freeze them for testing."""
import argparse
from pathlib import Path
import os
from config import ROOT
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import DATA, OUTPUT
from runtime import configure,load_run,save_json
from attack_evaluation import settings,evaluate_condition,sample_indices,save_records,fingerprint,choose_condition


def group_key(condition):
    """One selection per method/norm/budget/k; comparisons do not blend budgets."""
    return (condition['method'],condition['norm'],condition['budget'],condition['k'])


def sweep_plot(records, variable, output, objective=None, objective_unit=None, gain_fraction=.99):
    """Compare step counts with each step-factor/restart setting on the same cases."""
    objective=objective or variable
    metric='objective_mse' if objective=='all' else f'{objective}_rmse'
    clean_metric='clean_objective_mse' if objective=='all' else f'{objective}_clean_rmse'
    keys=list(dict.fromkeys(group_key(r) for r in records if r['method'] not in ('random','one_step')))
    fig,axes=plt.subplots(len(keys),1,figsize=(8,3.4*len(keys)),squeeze=False)
    for ax,key in zip(axes[:,0],keys):
        rows=[r for r in records if group_key(r)==key]
        variants=list(dict.fromkeys((r['step_factor'],r['restarts']) for r in rows))
        for factor,restarts in variants:
            curve=sorted([r for r in rows if r['step_factor']==factor and r['restarts']==restarts],key=lambda r:r['steps'])
            ax.plot([r['steps'] for r in curve],[r[metric] for r in curve],marker='o',label=f'factor {factor}, attempts {restarts}')
        winner=choose_condition(rows,gain_fraction)
        ax.axhline(rows[0][clean_metric],color='black',linestyle='--',label='Clean input')
        ax.scatter([winner['steps']],[winner[metric]],marker='*',s=140,color='black',zorder=5,label='Selected effort')
        ax.set_title(f'{key[0]}, {key[1]}, budget {key[2]}, k={key[3]}')
        ax.set_xlabel('Gradient updates per attempt')
        ax.set_ylabel('Normalized objective MSE' if objective=='all' else f"{objective} RMSE ({objective_unit or rows[0]['unit']})")
        ax.legend(fontsize=8,ncol=2)
    fig.suptitle('Validation sweep: larger error means a stronger attack')
    fig.tight_layout(rect=(0,0,1,.97))
    fig.savefig(output,dpi=130)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=DATA)
    parser.add_argument('--checkpoint-dir',type=Path,default=OUTPUT)
    parser.add_argument('--results-dir',type=Path,default=OUTPUT/'attack_tuning')
    parser.add_argument('--variable',default='t2m')
    parser.add_argument('--objective',default=None,help='Forecast variable to worsen; default same as attacked variable, or all')
    parser.add_argument('--norm',nargs='+',choices=['l0','l1','l2','linf'],default=['linf','l2','l1','l0'])
    parser.add_argument('--budget',type=float,nargs='+',default=[.1],help='L0/Linf amplitude, L1 mean-absolute, L2 RMS bound in variable units')
    parser.add_argument('--steps',type=int,nargs='+',default=[1,5,10,20,40])
    parser.add_argument('--step-factors',type=float,nargs='+',default=[.5,1.,2.])
    parser.add_argument('--restarts',type=int,nargs='+',default=[1,3],help='Total attempts: one clean start plus random starts')
    parser.add_argument('--k',type=int,nargs='+',default=[4])
    parser.add_argument('--cell-cap',type=float,default=1.,help='Additional max-cell cap in physical units for every norm')
    parser.add_argument('--momentum',type=float,default=1.)
    parser.add_argument('--gain-fraction',type=float,default=.99,help='Least effort retaining this fraction of the best validation attack gain')
    parser.add_argument('--max-samples',type=int,default=64,help='Evenly spaced validation subset; 0 uses all validation')
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--random-trials',type=int,default=3)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    if any(v<0 for v in args.budget) or any(v<=0 for v in args.step_factors) or min(args.steps+args.restarts+args.k+[args.batch_size,args.random_trials])<1 or args.cell_cap<0 or not 0<=args.momentum<=1 or args.max_samples<0 or not 0<args.gain_fraction<=1:
        parser.error('Invalid budgets/counts/caps.')
    if args.results_dir.exists(): parser.error('Choose a new results folder.')
    device=configure(device=args.device)
    model,datasets,stats,meta,lat,lon=load_run(args.checkpoint_dir,args.data,device)
    objective=args.objective or args.variable
    if args.variable not in meta['variables'] or objective not in meta['variables']+['all']:
        parser.error('Unknown input/objective variable.')
    if max(args.k)>len(lat)*len(lon): parser.error('k exceeds the number of cells.')
    checkpoint_hash=fingerprint(args.checkpoint_dir/'best_model.pt')
    indices=sample_indices(datasets['validation'],args.max_samples)
    args.results_dir.mkdir(parents=True)
    selected={}
    rows=[]
    for norm in args.norm:
        for budget in args.budget:
            for k in (args.k if norm=='l0' else [4]):
                for trial in range(args.random_trials):
                    c=settings('random',norm,budget,len(lat)*len(lon),k=k,cell_cap=args.cell_cap)
                    r=evaluate_condition(model,datasets['validation'],indices,stats,meta,args.variable,objective,c,device,args.batch_size,100+trial)
                    rows.append(r)
                if norm!='l0':
                    c=settings('one_step',norm,budget,len(lat)*len(lon),cell_cap=args.cell_cap)
                    rows.append(evaluate_condition(model,datasets['validation'],indices,stats,meta,args.variable,objective,c,device,args.batch_size))
                    selected[group_key(c)]=c
                methods=['top_k'] if norm=='l0' else ['pgd','mi_fgsm'] if norm=='linf' else ['pgd']
                for method in methods:
                    group_rows=[]
                    for steps in args.steps:
                        for factor in args.step_factors:
                            for restarts in args.restarts:
                                c=settings(method,norm,budget,len(lat)*len(lon),steps,factor,restarts,k,args.cell_cap,args.momentum)
                                r=evaluate_condition(model,datasets['validation'],indices,stats,meta,args.variable,objective,c,device,args.batch_size)
                                rows.append(r)
                                group_rows.append(r)
                                print(f"{method} {norm} k={c['k']} steps={steps} factor={factor} attempts={restarts}: {args.variable} RMSE={r[f'{args.variable}_rmse']:.4f}",flush=True)
                    chosen=choose_condition(group_rows,args.gain_fraction)
                    selected[group_key(c)]={key:chosen[key] for key in c}
                    print('Selected:',selected[group_key(c)],flush=True)
                # Random seed/trial is not selected for being most damaging.
                selected[('random',norm,budget,k if norm=='l0' else None)]=settings('random',norm,budget,len(lat)*len(lon),k=k,cell_cap=args.cell_cap)
    if fingerprint(args.checkpoint_dir/'best_model.pt')!=checkpoint_hash:
        raise RuntimeError('Checkpoint changed during tuning.')
    save_records(args.results_dir,rows)
    selection=dict(split='validation',indices=indices,validation_samples=len(indices),
                   variable=args.variable,objective=objective,random_trials=args.random_trials,
                   checkpoint_sha256=checkpoint_hash,
                   prepared_sha256=meta['prepared_sha256'],conditions=list(selected.values()),
                   choice_rule='least gradient effort retaining gain_fraction of best validation attack gain; test data never queried',
                   gain_fraction=args.gain_fraction,
                   steps=args.steps,step_factors=args.step_factors,restarts=args.restarts,
                   note='subset tuning is exploratory; no claim of globally strongest attack')
    save_json(args.results_dir/'selected_settings.json',selection)
    strongest=[]
    for c in selection['conditions']:
        if c['method'] in ('random','one_step'):
            strongest.append(c)
        else:
            group=[r for r in rows if group_key(r)==group_key(c)]
            winner=choose_condition(group,1.)
            strongest.append({key:winner[key] for key in c})
    save_json(args.results_dir/'strongest_validation_settings.json',
              dict(selection,conditions=strongest,gain_fraction=1.,
                   choice_rule='strongest observed validation loss; least effort among exact ties'))
    sweep_plot(rows,args.variable,args.results_dir/'step_sweep.png',objective,
               meta['units'][meta['variables'].index(objective)] if objective!='all' else None,args.gain_fraction)
    print(f'Saved {len(rows)} validation conditions; settings frozen in {args.results_dir}/selected_settings.json')


if __name__=='__main__': main()
