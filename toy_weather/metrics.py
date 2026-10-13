"""Global squared-error totals first, square roots last; one score per variable."""
import numpy as np
import torch


def evaluate_loader(model, loader, stats, variables, units, device='cpu'):
    """Batches x [B,input,H,W], y [B,C,H,W] -> labelled scalar scores.

    Generic ML: aggregate all cells rather than average batch RMSEs. Weather-specific:
    convert differences to each variable's physical units and compare persistence.
    """
    squared = np.zeros(len(variables), dtype='float64')
    persistence = np.zeros_like(squared)
    count = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            prediction = model(x)
            squared += (prediction-y).double().square().sum((0, 2, 3)).cpu().numpy()
            persistence += (model.current_state(x)-y).double().square().sum((0, 2, 3)).cpu().numpy()
            count += y.shape[0]*y.shape[2]*y.shape[3]
    if not count:
        raise ValueError('Cannot evaluate an empty loader.')
    mse, persistence_mse = squared/count, persistence/count
    std = np.asarray(stats['std'])
    results = {'normalized_mse': float(mse.mean()), 'variables': {}}
    for i, (name, unit) in enumerate(zip(variables, units)):
        clean = float(np.sqrt(mse[i])*std[i])
        baseline = float(np.sqrt(persistence_mse[i])*std[i])
        results['variables'][name] = dict(unit=unit, mse=float(mse[i]*std[i]**2),
                                         rmse=clean, persistence_rmse=baseline,
                                         improvement_percent=100*(baseline-clean)/baseline if baseline else None)
    return results
