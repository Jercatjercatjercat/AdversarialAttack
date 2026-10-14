"""Compare fixed attack settings, or test choices previously selected on validation."""
import argparse
import json
from pathlib import Path
from torch.utils.data import DataLoader,Subset
from config import DATA,OUTPUT
from metrics import evaluate_loader
from runtime import configure,load_run,save_json
from attack_evaluation import settings,evaluate_condition,sample_indices,save_records,fingerprint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=DATA)
    parser.add_argument('--checkpoint-dir',type=Path,default=OUTPUT)
    parser.add_argument('--results-dir',type=Path)
    parser.add_argument('--settings',type=Path,help='Validation-selected JSON freezes attack choices')
    parser.add_argument('--variable',default='t2m')
    parser.add_argument('--objective',default='all',help='Forecast variable to worsen, or all (historical default)')
    parser.add_argument('--budget',type=float,nargs='+',default=[.1,.5])
    parser.add_argument('--norm',nargs='+',choices=['l0','l1','l2','linf'],default=['linf','l2'])
    parser.add_argument('--methods',nargs='+',choices=['random','one_step','pgd','mi_fgsm','top_k'],default=['random','one_step','pgd'])
    parser.add_argument('--steps',type=int,nargs='+',default=[10])
    parser.add_argument('--step-factor',type=float,default=None,help='alpha=factor*radius/steps; omission preserves historical schedule')
    parser.add_argument('--restarts',type=int,default=1,help='Total attempts: one clean start plus random starts')
    parser.add_argument('--k',type=int,nargs='+',default=[4])
    parser.add_argument('--cell-cap',type=float,default=None)
    parser.add_argument('--momentum',type=float,default=1.)
    parser.add_argument('--random-trials',type=int,default=3)
    parser.add_argument('--split',choices=['validation','test'],default='test')
    parser.add_argument('--max-samples',type=int,default=0)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    if any(v<0 for v in args.budget) or min(args.steps+args.k+[args.restarts,args.random_trials,args.batch_size])<1 or args.max_samples<0 or not 0<=args.momentum<=1 or (args.cell_cap is not None and args.cell_cap<0) or (args.step_factor is not None and args.step_factor<=0):
        parser.error('Invalid budgets/counts/step sizes.')
    device=configure(device=args.device)
    model,datasets,stats,meta,lat,lon=load_run(args.checkpoint_dir,args.data,device)
    checkpoint_hash=fingerprint(args.checkpoint_dir/'best_model.pt')
    variable,objective,random_trials=args.variable,args.objective,args.random_trials
    conditions=[]
    if args.settings:
        selection=json.loads(args.settings.read_text())
        if selection['split']!='validation' or selection['checkpoint_sha256']!=checkpoint_hash or selection['prepared_sha256']!=meta['prepared_sha256']:
            parser.error('Settings must come from validation with these exact weights/data.')
        variable,objective,random_trials=selection['variable'],selection['objective'],selection['random_trials']
        conditions=selection['conditions']
    else:
        for norm in args.norm:
            for budget in args.budget:
                for k in (args.k if norm=='l0' else [4]):
                    for method in args.methods:
                        if method=='top_k' and norm!='l0' or method=='mi_fgsm' and norm!='linf' or method in ('one_step','pgd') and norm=='l0':
                            continue
                        for steps in (args.steps if method not in ('random','one_step') else [1]):
                            c=settings(method,norm,budget,len(lat)*len(lon),steps,args.step_factor or 1.,
                                       args.restarts if method not in ('random','one_step') else 1,
                                       k,args.cell_cap,args.momentum)
                            if args.step_factor is None and method not in ('random','one_step'):
                                c['step_size']=None; c['step_factor']=None
                            conditions.append(c)
    if variable not in meta['variables'] or objective not in meta['variables']+['all']:
        parser.error('Unknown variable/objective.')
    if not conditions: parser.error('No compatible method/norm combinations.')
    output=args.results_dir or args.checkpoint_dir/('tuned_attack_comparison' if args.settings else 'attack_comparison')
    if output.exists(): parser.error('Choose a new results folder to preserve existing evidence.')
    output.mkdir(parents=True)
    indices=sample_indices(datasets[args.split],args.max_samples)
    clean=evaluate_loader(model,DataLoader(Subset(datasets[args.split],indices),batch_size=args.batch_size),
                          stats,meta['variables'],meta['units'],device)
    save_json(output/'clean_metrics.json',clean)
    rows=[]
    for c in conditions:
        for trial in range(random_trials if c['method']=='random' else 1):
            r=evaluate_condition(model,datasets[args.split],indices,stats,meta,variable,objective,c,device,args.batch_size,100+trial)
            r['trial']=trial
            rows.append(r)
            print(f"{c['method']} {c['norm']} steps={c['steps']} k={c['k']}: {variable} RMSE={r[f'{variable}_rmse']:.4f}",flush=True)
    if fingerprint(args.checkpoint_dir/'best_model.pt')!=checkpoint_hash:
        raise RuntimeError('Checkpoint changed during evaluation; do not mix these results.')
    save_records(output,rows)
    save_json(output/'comparison_config.json',dict(split=args.split,indices=indices,variable=variable,
              objective=objective,checkpoint_sha256=checkpoint_hash,prepared_sha256=meta['prepared_sha256'],
              selected_settings_sha256=fingerprint(args.settings) if args.settings else None,
              conditions=conditions,random_trials=random_trials,batch_size=args.batch_size,
              note='No test-driven selection; random trials are separate; persistence uses original inputs'))


if __name__=='__main__': main()
