"""Attack/evaluate complete trajectories while keeping the existing CNN frozen."""
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from perturbations import change_input
from rollouts import rollout, window_loss


def attack_batch(model, x, targets, clocks, condition, std, variable_index, seed):
    """Normalized x -> changed x of identical shape; attack current ONE variable.

    Budgets are saved in physical units, then divided by that variable's training
    standard deviation. Only the initial state is changed; no later re-injection.
    """
    channel = (model.history-1)*model.variables+variable_index
    steps = condition['window_hours']//6
    scale = float(std[variable_index])
    alpha = condition['step_size']
    cap = condition['cell_cap']
    def loss(candidate):
        return window_loss(model, candidate, targets, clocks, steps, variable_index)
    return change_input(model, x, targets, channel,
        condition['physical_radius']/scale, condition['method'], condition['norm'],
        steps=condition['steps'], generator=torch.Generator().manual_seed(seed),
        step_size=None if alpha is None else alpha/scale,
        restarts=condition['restarts'], cell_cap=None if cap is None else cap/scale,
        loss_fn=loss)


def window_score(model, dataset, indices, condition, stats, variable_index, device, batch_size):
    """Validation only: score exactly the chosen loss window, not the test future."""
    totals, clean_totals, count = 0., 0., 0
    began = time.monotonic()
    k = condition['window_hours']//6
    for batch, (x, targets, clocks) in enumerate(DataLoader(Subset(dataset, indices), batch_size=batch_size)):
        x, targets, clocks = x.to(device), targets[:, :k].to(device), clocks[:, :k].to(device)
        changed = attack_batch(model, x, targets, clocks, condition, stats['std'], variable_index, 100+10007*batch)
        with torch.no_grad():
            totals += window_loss(model, changed, targets, clocks, k, variable_index).sum().item()
            clean_totals += window_loss(model, x, targets, clocks, k, variable_index).sum().item()
        count += x.shape[0]
    return dict(condition, objective_mse=totals/count, clean_objective_mse=clean_totals/count,
                gradient_updates_per_case=condition['steps']*condition['restarts'],
                seconds=time.monotonic()-began, samples=count)


def trajectory_errors(model, dataset, indices, stats, device, batch_size,
                      condition=None, variable_index=0, seed=100):
    """All leads -> physical per-case MSE [N,K,C] and initial changes [N,H,W].

    Squared errors are averaged over cells BEFORE pooling cases/taking square roots.
    Predictions use previous predictions; observations are only verification targets.
    """
    errors, persistence_errors, changes = [], [], []
    scale = torch.tensor(stats['std'], dtype=torch.float64)[None, None, :]
    began = time.monotonic()
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size)
    channel = (model.history-1)*model.variables+variable_index
    for batch, (x, target, clocks) in enumerate(loader):
        x, target, clocks = x.to(device), target.to(device), clocks.to(device)
        changed = x if condition is None else attack_batch(
            model, x, target, clocks, condition, stats['std'], variable_index, seed+10007*batch)
        with torch.no_grad():
            prediction = rollout(model, changed, clocks)
            if not torch.isfinite(prediction).all():
                raise RuntimeError('Non-finite rollout: reduce the horizon and report the unstable lead.')
            error = (prediction-target).cpu().double().square().mean((-1, -2))*scale.square()
            errors.append(error.numpy())
            # Persistence uses the ORIGINAL current state, even for attacked forecasts.
            persistence = model.current_state(x)[:, None].expand_as(target)
            baseline = (persistence-target).cpu().double().square().mean((-1, -2))*scale.square()
            persistence_errors.append(baseline.numpy())
            delta = (changed[:, channel]-x[:, channel]).cpu().double()*stats['std'][variable_index]
            changes.append(delta.numpy())
        if (batch+1)%10 == 0 or batch+1 == len(loader):
            print(f'  evaluated {min((batch+1)*batch_size,len(indices))}/{len(indices)} cases', flush=True)
    return np.concatenate(errors), np.concatenate(persistence_errors), np.concatenate(changes), time.monotonic()-began


def change_summary(delta, condition):
    """Physical [N,H,W] -> actual disturbance sizes, checking the saved bounds."""
    flat = delta.reshape(len(delta), -1)
    maximum = float(np.abs(flat).max())
    l2 = float(np.linalg.norm(flat, axis=1).max())
    measured = maximum if condition['norm']=='linf' else l2
    tolerance = 1e-5*(1 if condition['norm']=='linf' else flat.shape[1]**.5)
    if measured > condition['physical_radius']+tolerance:
        raise RuntimeError('Initial perturbation exceeds its norm budget.')
    cap = condition['cell_cap']
    if cap is not None and maximum > cap+1e-5:
        raise RuntimeError('Initial perturbation exceeds the cell cap.')
    return dict(max_cell_change=maximum, rms_input_change=float(np.sqrt(np.mean(flat**2))),
                mean_absolute_change=float(np.abs(flat).mean()), max_sample_l2=l2)


def validity_summary(mse, clean_mse, leads, variable_index, threshold):
    """Descriptive lifetime at a PREDECLARED physical RMSE-increase threshold.

    This is neither statistical significance nor an atmospheric predictability limit.
    Last-above may include gaps; continuous-from-start is reported separately.
    """
    rmse = np.sqrt(mse[:, :, variable_index].mean(0))
    clean = np.sqrt(clean_mse[:, :, variable_index].mean(0))
    above = rmse-clean >= threshold
    where = np.flatnonzero(above)
    first_below = np.flatnonzero(~above)
    initial = int(first_below[0]) if len(first_below) else len(leads)
    return dict(threshold_rmse_increase=threshold,
                first_lead_above_hours=int(leads[where[0]]) if len(where) else None,
                last_lead_above_hours=int(leads[where[-1]]) if len(where) else None,
                continuous_from_start_hours=int(leads[initial-1]) if initial else 0,
                leads_above_threshold=int(above.sum()),
                still_above_at_last_lead=bool(above[-1]),
                max_rmse_increase=float((rmse-clean).max()),
                max_increase_at_hours=int(leads[np.argmax(rmse-clean)]))
