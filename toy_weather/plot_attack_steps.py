"""Plot fixed-step attacks without selecting a winner or rerunning the CNN."""
import argparse
import json
import os
from pathlib import Path
from config import ROOT, OUTPUT
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.matplotlib'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def plot_comparison(folder):
    """Saved scalar test scores -> overview bars and a detailed update-count curve.

    Each score pools errors over the SAME test examples. The right panel uses a
    tighter y-axis to expose tiny differences: it must not imply a large effect.
    No model weights, weather maps or attack settings are changed by plotting.
    """
    folder = Path(folder)
    rows = json.loads((folder/'results.json').read_text())
    config = json.loads((folder/'comparison_config.json').read_text())
    variable = config['variable']
    key = f'{variable}_rmse'
    clean = rows[0][f'{variable}_clean_rmse']
    unit = rows[0]['unit']
    fgsm = next(r[key] for r in rows if r['method']=='one_step')
    # Pool equally sized random trials in MSE space, then take one square root.
    random_rows = [r for r in rows if r['method']=='random']
    random = float(np.sqrt(np.mean([r[key]**2 for r in random_rows])))
    pgd = sorted([r for r in rows if r['method']=='pgd'], key=lambda r:r['steps'])
    momentum = sorted([r for r in rows if r['method']=='mi_fgsm'], key=lambda r:r['steps'])
    sizes = [r['steps'] for r in pgd]
    if sizes != [r['steps'] for r in momentum]:
        raise ValueError('Both repeated methods must use the same update counts.')
    if any(r['norm']!='linf' or r['restarts']!=1 for r in rows):
        raise ValueError('This figure expects Linf attacks with one attempt per case.')
    for a,b in zip(pgd,momentum):
        if a['step_size']!=b['step_size']:
            raise ValueError('PGD and MI-FGSM step sizes must match.')
        if not np.isclose(a['step_size'],a['budget']/a['steps']):
            raise ValueError('This figure expects step size = budget / updates.')
    if config['objective']!=variable or len({r['budget'] for r in rows})!=1:
        raise ValueError('Use one input budget and its own variable as objective.')
    plt.rcParams.update({'font.size':11, 'figure.facecolor':'#F7F5F0',
        'axes.facecolor':'#F7F5F0','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    labels=['Clean',f'Random ({len(random_rows)} trials)','FGSM (1 update)',
            f"PGD ({pgd[-1]['steps']} updates)",f"MI-FGSM ({momentum[-1]['steps']} updates)"]
    scores=[clean,random,fgsm,pgd[-1][key],momentum[-1][key]]
    axes[0].barh(labels,scores,color=['#26383C','#888888','#555555','#0072B2','#D55E00'])
    axes[0].invert_yaxis()
    for i,score in enumerate(scores):
        axes[0].text(score+.008,i,f'{score:.6f}',va='center',fontsize=10)
    axes[0].set(xlim=(0,max(scores)*1.18),xlabel=f'{variable} RMSE ({unit})',
                title='Overview: same initial-input limit')
    for data,label,color,marker in [(pgd,'PGD','#0072B2','o'),(momentum,'MI-FGSM','#D55E00','s')]:
        axes[1].plot(sizes,[r[key] for r in data],label=label,color=color,
                     marker=marker,lw=2,ls='-' if label=='PGD' else '--')
    axes[1].axhline(fgsm,color='#333333',ls=':',lw=2,label='FGSM: one update')
    # Full decimal ticks avoid a hidden offset such as '+8.5e-1'.
    formatter=ScalarFormatter(useOffset=False);formatter.set_scientific(False)
    axes[1].yaxis.set_major_formatter(formatter)
    axes[1].set(xticks=sizes,xticklabels=[f"{r['steps']}\nα={r['step_size']:g}" for r in pgd],
                xlabel=f'Updates; α = step size in {unit}',ylabel=f'{variable} RMSE ({unit})',
                title='Detail: zoomed vertical axis')
    axes[1].legend(loc='lower right',bbox_to_anchor=(1.,.15),fontsize=9)
    budget=pgd[0]['budget']
    fig.suptitle(f"Fixed-step comparison • {rows[0]['samples']:,} test cases • ±{budget:g} {unit} per cell")
    fig.supxlabel('Same clean start and temperature objective; α = budget / updates; momentum = 1. No test-driven selection.',fontsize=10)
    fig.savefig(folder/'step_comparison.png',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,default=OUTPUT/'fixed_step_attacks')
    args=parser.parse_args()
    plot_comparison(args.results_dir)
