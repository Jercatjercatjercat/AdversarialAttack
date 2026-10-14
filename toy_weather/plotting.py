"""Pictures distinguish observations, forecasts and changes; no model calculations."""
from pathlib import Path
import os
from config import ROOT
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from data.era5 import denormalize


def forecast_plot(current, target, prediction, stats, variables, units, path, timestamp=None):
    """Three normalized [C,H,W] maps -> rows of physical-unit forecast panels.

    Every variable gets its own colour scale/units. Temperature displays Celsius;
    error differences have the same numerical magnitude in Celsius and Kelvin.
    """
    maps = [denormalize(v, stats) for v in (current, target, prediction)]
    fig, axes = plt.subplots(len(variables), 4, figsize=(13, 2.7*len(variables)), squeeze=False)
    for i, name in enumerate(variables):
        physical = [v[i].copy() for v in maps]
        display_unit = units[i]
        if name == 't2m':
            physical = [v-273.15 for v in physical]
            display_unit = '°C'
        low, high = min(v.min() for v in physical), max(v.max() for v in physical)
        error = physical[2]-physical[1]
        limit = max(float(np.abs(error).max()), 1e-6)
        for j, (field, title) in enumerate(zip(physical+[error], ['Now', 'True +6h', 'CNN +6h', 'Prediction − truth'])):
            options = dict(cmap='coolwarm', vmin=-limit, vmax=limit) if j==3 else dict(cmap='viridis', vmin=low, vmax=high)
            im = axes[i, j].imshow(field, origin='lower', **options)
            axes[i, j].set_title(f'{name}: {title}')
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            fig.colorbar(im, ax=axes[i, j], shrink=.8, label=display_unit)
    if timestamp is not None:
        fig.suptitle(f'ERA5 input {timestamp} UTC → forecast six hours ahead')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def training_plot(history, variables, units, path):
    """Epoch records -> normalized loss and per-variable physical validation RMSE."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    ax = axes.flat[0]
    ax.plot([v['epoch'] for v in history], [v['train_mse'] for v in history], label='Training')
    ax.plot([v['epoch'] for v in history], [v['validation']['normalized_mse'] for v in history], label='Validation')
    ax.set_title('Mean normalized MSE')
    ax.legend()
    for ax, name, unit in zip(list(axes.flat)[1:], variables, units):
        epochs = [v['epoch'] for v in history]
        ax.plot(epochs, [v['validation']['variables'][name]['rmse'] for v in history], label='CNN')
        ax.plot(epochs, [v['validation']['variables'][name]['persistence_rmse'] for v in history], '--', label='No change')
        ax.set_title(f'{name}: validation RMSE ({unit})')
        ax.legend()
    for ax in list(axes.flat)[1+len(variables):]:
        ax.set_visible(False)
    for ax in axes.flat:
        ax.set_xlabel('Epoch')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def gradient_plot(gradient, variables, history, path):
    """Input gradient [L*C+clock,H,W] -> weather-channel sensitivity pictures.

    These values are derivatives of normalized MSE with respect to normalized
    weather; they are NOT temperatures or an applied attack. Clock channels omitted.
    """
    fig, axes = plt.subplots(history, len(variables), figsize=(3*len(variables), 3*history), squeeze=False)
    for t in range(history):
        for c, name in enumerate(variables):
            field = gradient[t*len(variables)+c]
            limit = max(float(np.abs(field).max()), 1e-12)
            im = axes[t, c].imshow(field, origin='lower', cmap='coolwarm', vmin=-limit, vmax=limit)
            axes[t, c].set_title(f'{name}, {"now" if t==history-1 else "−6h"}')
            fig.colorbar(im, ax=axes[t, c], shrink=.75)
    fig.suptitle('Loss sensitivity to normalized input weather (not an attack)')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
