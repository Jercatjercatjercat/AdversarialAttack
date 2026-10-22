"""Physical-unit figures for clean rollouts and attacks with different loss windows."""
from pathlib import Path
from itertools import cycle
import os
import numpy as np
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent/'.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = ['#566F69', '#887469', '#68758A', '#847B52']
# Forecast curves can nearly overlap. Colour PLUS dashes/markers makes their
# identities readable without relying on subtle colour differences alone.
WINDOW_STYLES = [('#0072B2', '-', 'o'), ('#D55E00', '--', 's'), ('#7B3294', '-.', '^')]
plt.rcParams.update({'font.size': 11, 'figure.facecolor': '#F7F5F0',
                     'axes.facecolor': '#F7F5F0', 'text.color': '#182C32',
                     'axes.labelcolor': '#182C32', 'xtick.color': '#182C32',
                     'ytick.color': '#182C32', 'axes.spines.top': False,
                     'axes.spines.right': False})


def plot_results(folder, arrays, conditions, names, units, variable, threshold):
    """Case MSE [condition,N,K,C] -> curves; a star marks the optimisation horizon."""
    folder = Path(folder)
    leads = arrays['lead_hours']
    index = names.index(variable)
    clean, persistence = arrays['clean'], arrays['persistence']
    clean_curve = np.sqrt(clean[:, :, index].mean(0))
    random_curve = np.sqrt(arrays['random'][:, :, index].mean(0))
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
    for i, (name, unit) in enumerate(zip(names, units)):
        ax = axes.flat[i]
        ax.plot(leads, np.sqrt(clean[:, :, i].mean(0)), color='#0072B2', lw=2, label='CNN: predicted inputs')
        ax.plot(leads, np.sqrt(persistence[:, :, i].mean(0)), color='#D55E00', ls='--', lw=2, label='No change: original state held fixed')
        ax.set(title=name, xlabel='Forecast lead (hours)', ylabel=f'RMSE ({unit})')
        ax.legend(fontsize=8)
    axes.flat[-1].axis('off')
    fig.suptitle('Clean forecasts: error as predictions are fed back into the CNN')
    fig.savefig(folder/'clean_rollout_errors.png', dpi=160)
    plt.close(fig)

    for method in ('one_step', 'pgd'):
        selected = [c for c in conditions if c['method']==method]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
        axes[0].plot(leads, clean_curve, color='#111111', lw=2.2, label='Clean')
        axes[0].plot(leads, random_curve, color='#777777', ls=':', lw=2, label='Random-trial mean')
        axes[0].plot(leads, np.sqrt(persistence[:, :, index].mean(0)), color='#888888', ls=(0,(6,3)), lw=1.5, label='No change: original state held fixed')
        axes[1].plot(leads, random_curve-clean_curve, color='#777777', ls=':', lw=2, label='Random')
        for (color, linestyle, marker), condition in zip(cycle(WINDOW_STYLES), selected):
            curve = np.sqrt(arrays[condition['id']][:, :, index].mean(0))
            label = f"Loss through {condition['window_hours']}h"
            for ax, values in zip(axes, (curve, curve-clean_curve)):
                ax.plot(leads, values, color=color, ls=linestyle, lw=2.2,
                        marker=marker, markevery=3, markersize=4, label=label)
                k = condition['window_hours']//6-1
                ax.scatter([leads[k]], [values[k]], color=color, edgecolor='white',
                           linewidth=.6, marker='*', s=130, zorder=5)
        axes[1].axhline(threshold, color='#67594F', ls='--', lw=1, label=f'Descriptive threshold: +{threshold:g} {units[index]}')
        axes[1].axhline(0, color='#26383C', lw=.6)
        axes[0].set(ylabel=f'{variable} forecast RMSE ({units[index]})', title='Total forecast error')
        axes[1].set(ylabel=f'Attacked RMSE − clean RMSE ({units[index]})', title='Extra error caused by initial changes')
        for ax in axes:
            ax.set_xlabel('Forecast lead (hours)'); ax.legend(fontsize=8)
        attack_name = ('FGSM (one input update)' if selected[0]['norm']=='linf' else 'One-step gradient attack') if method=='one_step' else 'PGD'
        bound = f"±{selected[0]['budget']:g} {units[index]} per cell" if selected[0]['norm']=='linf' else f"RMS ≤ {selected[0]['budget']:g} {units[index]}"
        fig.suptitle(f"{attack_name} • {bound} • star = end of loss window")
        fig.savefig(folder/f'{method}_loss_windows.png', dpi=160)
        plt.close(fig)

    # Identical y-axis limits make FGSM/PGD comparable without visual rescaling.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True, constrained_layout=True)
    for ax, method, title in zip(axes, ('one_step','pgd'), ('FGSM: one input update','PGD: repeated input updates')):
        ax.plot(leads, random_curve-clean_curve, color='#777777', ls=':', label='Random')
        selected = [c for c in conditions if c['method']==method]
        for (color, linestyle, marker), c in zip(cycle(WINDOW_STYLES),selected):
            values = np.sqrt(arrays[c['id']][:,:,index].mean(0))-clean_curve
            ax.plot(leads,values,color=color,ls=linestyle,marker=marker,markevery=3,
                    markersize=4,lw=2,label=f"Loss through {c['window_hours']}h")
            k=c['window_hours']//6-1
            ax.scatter([leads[k]],[values[k]],color=color,marker='*',s=110,zorder=5)
        ax.axhline(0,color='#26383C',lw=.7)
        ax.set(title=title,xlabel='Forecast lead (hours)')
        ax.legend(fontsize=8)
    axes[0].set_ylabel(f'Attacked RMSE − clean RMSE ({units[index]})')
    fig.suptitle(f"Same initial-input limit: {bound} • same cases and frozen CNN")
    fig.savefig(folder/'method_loss_window_comparison.png',dpi=180)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,4.5),constrained_layout=True)
    windows=sorted(set(c['window_hours'] for c in conditions))
    for (color,linestyle,marker),window in zip(cycle(WINDOW_STYLES),windows):
        pgd=next(c for c in conditions if c['method']=='pgd' and c['window_hours']==window)
        fgsm=next(c for c in conditions if c['method']=='one_step' and c['window_hours']==window)
        difference=np.sqrt(arrays[pgd['id']][:,:,index].mean(0))-np.sqrt(arrays[fgsm['id']][:,:,index].mean(0))
        ax.plot(leads,difference,color=color,ls=linestyle,marker=marker,markevery=3,lw=2,
                label=f'{window}h loss window')
    ax.axhline(0,color='#26383C',lw=.7)
    ax.set(xlabel='Forecast lead (hours)',ylabel=f'PGD RMSE − FGSM RMSE ({units[index]})',
           title=f'Additional error from PGD at {bound}')
    ax.legend()
    fig.savefig(folder/'pgd_minus_fgsm.png',dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    windows = sorted(set(c['window_hours'] for c in conditions))
    positions = np.arange(len(windows))
    for offset, method, color in [(-.18,'one_step',COLORS[0]),(.18,'pgd',COLORS[1])]:
        scores = []
        for window in windows:
            c = next(c for c in conditions if c['method']==method and c['window_hours']==window)
            scores.append(np.sqrt(arrays[c['id']][:, -1, index].mean()))
        ax.bar(positions+offset, scores, width=.34, label=method, color=color)
    ax.axhline(clean_curve[-1], color='#26383C', ls='--', label='Clean')
    ax.set(xticks=positions, xticklabels=[f'{w}h' for w in windows], xlabel='Attack optimisation window',
           ylabel=f"{variable} RMSE at {leads[-1]}h ({units[index]})", title='Attack evaluated beyond its optimisation window')
    ax.legend()
    fig.savefig(folder/'attack_window_comparison.png', dpi=160)
    plt.close(fig)

    # Extra-error bars reveal small differences hidden by total five-day model error.
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for offset, method, color in [(-.18,'one_step',COLORS[0]),(.18,'pgd',COLORS[1])]:
        increases = []
        for window in windows:
            c = next(c for c in conditions if c['method']==method and c['window_hours']==window)
            increases.append(np.sqrt(arrays[c['id']][:, -1, index].mean())-clean_curve[-1])
        ax.bar(positions+offset, increases, width=.34,
               label='FGSM (one update)' if method=='one_step' else 'PGD', color=color)
    ax.axhline(0, color='#26383C', lw=.7)
    ax.axhline(random_curve[-1]-clean_curve[-1], color='#827E76', ls=':', label='Random-trial mean')
    ax.set(xticks=positions, xticklabels=[f'{w}h' for w in windows], xlabel='Attack optimisation window',
           ylabel=f'Extra RMSE at {leads[-1]}h ({units[index]})', title='How much attack damage remains at five days?')
    ax.legend()
    fig.savefig(folder/'attack_window_extra_error.png', dpi=160)
    plt.close(fig)


def plot_existing_one_step(folder, output):
    """Saved 1458-case one-step results -> clean accuracy and matched Linf attacks.

    These are the ORIGINAL test cases, separate from the slightly shorter set
    with five complete future days. No new attack settings are chosen here.
    """
    import csv
    import json
    folder = Path(folder)
    clean = json.loads((folder/'test_metrics.json').read_text())['variables']['t2m']
    with (folder/'tuned_attacks_steps'/'results.csv').open() as f:
        rows = [r for r in csv.DictReader(f) if r['norm']=='linf']
    means = {method: np.mean([float(r['t2m_rmse']) for r in rows if r['method']==method])
             for method in ('random','one_step','pgd','mi_fgsm')}
    fig,axes = plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    labels=['No change','CNN']
    scores=[clean['persistence_rmse'],clean['rmse']]
    axes[0].barh(labels,scores,color=[COLORS[1],COLORS[0]])
    axes[0].invert_yaxis()
    for i,value in enumerate(scores): axes[0].text(value+.04,i,f'{value:.3f}',va='center')
    axes[0].set(xlabel='Temperature RMSE (K)',xlim=(0,3.1),title='Clean six-hour forecast')
    labels=['Clean','Random','FGSM','PGD','MI-FGSM']
    scores=[clean['rmse']]+[means[k] for k in ('random','one_step','pgd','mi_fgsm')]
    axes[1].barh(labels,scores,color=['#26383C',COLORS[0],COLORS[1],COLORS[2],COLORS[3]])
    axes[1].invert_yaxis()
    for i,value in enumerate(scores): axes[1].text(value+.012,i,f'{value:.3f}',va='center')
    axes[1].set(xlabel='Temperature RMSE (K)',xlim=(0,1.02),title='Same ±0.1 K per-cell input limit')
    fig.suptitle('Original 2022 test comparison • 1,458 cases • lower RMSE is better')
    fig.supxlabel('Selected PGD and MI-FGSM use one update here; this shows no momentum advantage.',fontsize=10)
    fig.savefig(output,dpi=180)
    plt.close(fig)
