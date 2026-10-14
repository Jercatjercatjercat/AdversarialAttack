"""Shared scoring for validation tuning and frozen-setting test comparisons."""
import csv
import hashlib
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from perturbations import change_input, forecast_loss
from runtime import save_json


def norm_radius(budget, norm, cells):
    """Physical budget -> radius: Linf/L0 amplitude, L1 mean-absolute, L2 RMS."""
    return budget*cells if norm=='l1' else budget*cells**.5 if norm=='l2' else budget


def settings(method, norm, budget, cells, steps=1, factor=1., restarts=1, k=4,
             cell_cap=None, momentum=1.):
    """Explicit physical settings; factor chooses alpha = factor*radius/steps.

    L0's radius here is its amplitude cap, not its count. k separately limits count.
    one_step always uses the full radius; random has no gradient updates.
    """
    radius = norm_radius(budget, norm, cells)
    return dict(method=method, norm=norm, budget=budget, physical_radius=radius,
                steps=steps, step_factor=factor,
                step_size=radius*factor/steps if method not in ('one_step','random') else None,
                restarts=restarts, k=k if norm=='l0' else None, cell_cap=cell_cap,
                momentum=momentum if method=='mi_fgsm' else None)


def evaluate_condition(model, dataset, indices, stats, metadata, variable, objective,
                       condition, device, batch_size=32, seed=100):
    """Frozen model + selected cases -> pooled scores, measured changes and cost.

    Same cases/seed across settings. Scores include ALL forecast variables even
    when the attack objective is temperature alone. No checkpoint/weights are updated.
    """
    names = metadata['variables']
    variable_index = names.index(variable)
    channel = (metadata['history']-1)*len(names)+variable_index
    objective_index = None if objective=='all' else names.index(objective)
    std = np.asarray(stats['std'])
    generator = torch.Generator().manual_seed(seed)
    totals, clean_totals = np.zeros(len(names)), np.zeros(len(names))
    loss_total, clean_loss_total, count, samples = 0., 0., 0, 0
    sum_abs, sum_squared, max_cell, max_l1, max_l2, max_l0 = 0., 0., 0., 0., 0., 0
    began = time.monotonic()
    loader = DataLoader(Subset(dataset, list(indices)), batch_size=batch_size)
    for batch_index,(x,y) in enumerate(loader):
        # Extra restarts must not shift later batches' initial random changes.
        generator.manual_seed(seed+10007*batch_index)
        x,y = x.to(device),y.to(device)
        radius = condition['physical_radius']/std[variable_index]
        alpha = condition['step_size']
        cap = condition['cell_cap']
        changed = change_input(model, x, y, channel, radius, condition['method'],
            condition['norm'], condition['steps'], generator,
            step_size=alpha/std[variable_index] if alpha and radius else None,
            restarts=condition['restarts'], k=condition['k'] or 4,
            cell_cap=None if cap is None else cap/std[variable_index],
            momentum=condition['momentum'] if condition['momentum'] is not None else 1.,
            objective_index=objective_index)
        with torch.no_grad():
            prediction, clean = model(changed), model(x)
            totals += (prediction-y).cpu().double().square().sum((0,2,3)).numpy()
            clean_totals += (clean-y).cpu().double().square().sum((0,2,3)).numpy()
            loss_total += forecast_loss(prediction,y,objective_index).sum().item()
            clean_loss_total += forecast_loss(clean,y,objective_index).sum().item()
            delta = (changed[:,channel]-x[:,channel]).cpu().double()*std[variable_index]
            flat = delta.flatten(1)
            sum_abs += flat.abs().sum().item()
            sum_squared += flat.square().sum().item()
            max_cell = max(max_cell,flat.abs().max().item())
            max_l1 = max(max_l1,flat.abs().sum(1).max().item())
            max_l2 = max(max_l2,flat.norm(dim=1).max().item())
            max_l0 = max(max_l0,(flat!=0).sum(1).max().item())
            samples += y.shape[0]
            count += y.shape[0]*y.shape[2]*y.shape[3]
    if not samples:
        raise ValueError('Select at least one evaluation case.')
    record = dict(condition, variable=variable, unit=metadata['units'][variable_index],
                  objective=objective, seed=seed, samples=samples, objective_mse=loss_total/samples, clean_objective_mse=clean_loss_total/samples,
                  seconds=time.monotonic()-began, max_cell_change=max_cell,
                  mean_absolute_change=sum_abs/count, rms_input_change=(sum_squared/count)**.5,
                  max_sample_l0=int(max_l0), max_sample_l1=max_l1, max_sample_l2=max_l2,
                  gradient_updates_per_case=0 if condition['method']=='random' or condition['physical_radius']==0 else
                  1 if condition['method']=='one_step' else condition['steps']*condition['restarts'])
    for i,name in enumerate(names):
        record[f'{name}_rmse'] = float((totals[i]/count)**.5*std[i])
        record[f'{name}_clean_rmse'] = float((clean_totals[i]/count)**.5*std[i])
    # Actual float32 arithmetic can exceed a mathematical bound by tiny roundoff.
    cell_tolerance = 1e-5
    n_cells = dataset[0][1].shape[-2]*dataset[0][1].shape[-1]
    # Rounding errors add under L1 and combine quadratically under L2.
    tolerance = cell_tolerance*(n_cells if condition['norm']=='l1' else n_cells**.5 if condition['norm']=='l2' else 1)
    record['constraint_tolerance'] = tolerance
    record['cell_tolerance'] = cell_tolerance
    measured = max_cell if condition['norm'] in ('linf','l0') else max_l1 if condition['norm']=='l1' else max_l2
    if measured > condition['physical_radius']+tolerance:
        raise RuntimeError('Measured input change exceeded its norm limit.')
    if condition['norm']=='l0' and max_l0 > condition['k']:
        raise RuntimeError('Sparse attack changed more than k cells.')
    if condition['cell_cap'] is not None and max_cell > condition['cell_cap']+cell_tolerance:
        raise RuntimeError('Measured input change exceeded the separate cell cap.')
    return record


def sample_indices(dataset, maximum=0):
    """Stable evenly spaced indices; zero means use the complete split."""
    if maximum < 0:
        raise ValueError('Maximum samples cannot be negative.')
    count = min(maximum,len(dataset)) if maximum else len(dataset)
    return np.linspace(0,len(dataset)-1,count,dtype=int).tolist()


def save_records(folder, records):
    """Save machine-readable scores and a readable table without rounded arithmetic."""
    save_json(Path(folder)/'results.json',records)
    with (Path(folder)/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)


def fingerprint(path):
    """A file fingerprint prevents tuning settings being silently used with new weights."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choose_condition(rows, gain_fraction=.99):
    """Choose least gradient effort retaining a fraction of the best VALIDATION gain.

    Gain means attacked objective MSE minus clean objective MSE. Using raw loss
    instead could accept a weak attack simply because clean error is already large.
    Setting fraction=1 requires the strongest observed score. This is a selection
    convention, not proof that the sweep found the globally strongest attack.
    """
    if not 0 < gain_fraction <= 1:
        raise ValueError('Gain fraction must be in (0,1].')
    baseline=rows[0]['clean_objective_mse']
    maximum=max(r['objective_mse'] for r in rows)
    threshold=baseline+gain_fraction*max(0.,maximum-baseline)
    eligible=[r for r in rows if r['objective_mse']>=threshold-1e-12]
    return min(eligible,key=lambda r:(r['gradient_updates_per_case'],r['restarts'],r['steps'],r['step_factor']))
