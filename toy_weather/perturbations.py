"""Attack ONE current weather map; history, clocks, other inputs and weights stay fixed."""
import torch


def project(delta, radius, norm, k=4, cell_cap=None):
    """[B,1,H,W] changes -> feasible changes independently for EACH example.

    L0: radius is the amplitude cap, k the changed-cell count. Other norms: radius
    is the norm budget. Optional cell_cap adds a separate amplitude restriction.
    L1 uses soft thresholding; L0 keeps the top-k absolute candidate changes.
    """
    if radius < 0 or norm not in ('l0', 'l1', 'l2', 'linf'):
        raise ValueError('Invalid norm/radius.')
    if cell_cap is not None and cell_cap < 0:
        raise ValueError('Cell cap must be nonnegative.')
    flat = delta.flatten(1)
    if norm == 'l0':
        if not 1 <= k <= flat.shape[1]:
            raise ValueError('k must be between 1 and the number of cells.')
        indices = flat.abs().topk(k, dim=1).indices
        mask = torch.zeros_like(flat).scatter_(1, indices, 1)
        result = (flat*mask).clamp(-radius, radius).reshape_as(delta)
    elif norm == 'linf':
        result = delta.clamp(-radius, radius)
    elif norm == 'l2':
        length = flat.norm(dim=1).clamp_min(1e-12)[:, None, None, None]
        result = delta*(radius/length).clamp(max=1)
    else:
        # L1 projection removes small entries until total absolute change fits.
        ordered = flat.abs().sort(dim=1, descending=True).values
        ranks = torch.arange(1, flat.shape[1]+1, device=delta.device, dtype=delta.dtype)
        thresholds = (ordered.cumsum(1)-radius)/ranks
        counts = (ordered > thresholds).sum(1).clamp_min(1)
        threshold = thresholds.gather(1, (counts-1)[:, None]).clamp_min(0)
        result = (flat.sign()*(flat.abs()-threshold).clamp_min(0)).reshape_as(delta)
    if cell_cap is not None:
        # Clipping can only decrease these norms/counts, so both limits still hold.
        result = result.clamp(-cell_cap, cell_cap)
    return result


def direction(gradient, norm):
    """[B,1,H,W] derivatives -> a direction suited to the chosen budget.

    L1 spends an infinitesimal step on the strongest cell. L0 uses gradient
    magnitude to rank candidate cells, followed by top-k projection. Neither is
    obtained by blindly applying FGSM's sign to every cell.
    """
    flat = gradient.flatten(1)
    if norm == 'linf':
        return gradient.sign()
    if norm == 'l1':
        index = flat.abs().argmax(1, keepdim=True)
        values = flat.gather(1, index).sign()
        return torch.zeros_like(flat).scatter_(1, index, values).reshape_as(gradient)
    if norm == 'l0':
        size = flat.abs().max(1).values.clamp_min(1e-12)
    else:
        size = flat.norm(dim=1).clamp_min(1e-12)
    return gradient/size[:, None, None, None]


def sample_change(template, radius, norm, generator=None, k=4, cell_cap=None):
    """Seeded random [B,1,H,W] changes; CPU draws work with CPU/CUDA/MPS models."""
    noise = torch.randn(template.shape, dtype=template.dtype, generator=generator).to(template.device)
    if norm == 'l0':
        # IID scores choose a uniformly random support; each chosen cell reaches cap.
        ranks = noise.flatten(1).abs().topk(k, dim=1).indices
        mask = torch.zeros_like(noise.flatten(1)).scatter_(1, ranks, 1).reshape_as(noise)
        delta = radius*noise.sign()*mask
    elif norm == 'linf':
        delta = radius*noise.sign()
    else:
        length = noise.flatten(1).abs().sum(1) if norm=='l1' else noise.flatten(1).norm(dim=1)
        delta = radius*noise/length.clamp_min(1e-12)[:, None, None, None]
    return project(delta, radius, norm, k, cell_cap)


def forecast_loss(prediction, target, objective_index=None):
    """[B,C,H,W] -> [B] normalized MSE, either one output or all outputs.

    An index chooses one forecast variable. None preserves the old five-output
    objective. The observation used as target is never changed by the attack.
    """
    error = prediction-target
    if objective_index is not None:
        error = error[:, objective_index:objective_index+1]
    return error.square().mean((1, 2, 3))


def change_input(model, x, y, channel, radius, method, norm='linf', steps=10,
                 generator=None, step_size=None, restarts=1, k=4, cell_cap=None,
                 momentum=1.0, objective_index=None, loss_fn=None):
    """x [B,input,H,W], y [B,C,H,W] -> changed x with identical shape.

    Only channel changes. Budgets/step_size/cell_cap are normalized units. Explicit
    step_size means that same update size on EVERY iteration. None preserves the
    historical PGD schedule (full one-step candidate, then 2*radius/steps).
    MI-FGSM defaults to radius/steps. top_k uses the same explicit/scheduled steps.
    Best candidates are retained PER example; more effort need not hurt the objective.
    """
    methods = ('random', 'one_step', 'pgd', 'mi_fgsm', 'top_k')
    if radius < 0 or min(steps, restarts) < 1 or norm not in ('l0','l1','l2','linf') or method not in methods:
        raise ValueError('Invalid attack settings.')
    if not 0 <= channel < x.shape[1] or not 0 <= momentum <= 1 or (step_size is not None and step_size <= 0):
        raise ValueError('Invalid channel, momentum or step size.')
    if (method=='mi_fgsm' and norm!='linf') or (method=='top_k' and norm!='l0'):
        raise ValueError('MI-FGSM here uses Linf; top_k uses L0.')
    if method in ('one_step','pgd') and norm=='l0':
        raise ValueError('Use top_k for an L0 gradient attack.')
    if norm=='l0' and not 1 <= k <= x.shape[-2]*x.shape[-1]:
        raise ValueError('Invalid changed-cell count.')
    # A custom function maps an input [B,input,H,W] to one loss per case [B].
    # This lets PGD differentiate through several forecasts without duplicating
    # the attack algorithm. The usual one-step objective remains the default.
    def input_loss(candidate):
        return loss_fn(candidate) if loss_fn is not None else forecast_loss(model(candidate), y, objective_index)

    original = x.detach()
    if radius == 0:
        return original.clone()
    selected = original[:, channel:channel+1]
    if method=='random':
        changed = original.clone()
        changed[:, channel:channel+1] += sample_change(selected, radius, norm, generator, k, cell_cap)
        return changed
    with torch.no_grad():
        best_loss = input_loss(original)
    best = original.clone()
    iterations = 1 if method=='one_step' else steps
    for restart in range(1 if method=='one_step' else restarts):
        # First attempt starts clean. Additional attempts start at random feasible inputs.
        candidate = original.clone()
        if restart:
            candidate[:, channel:channel+1] += sample_change(selected, radius, norm, generator, k, cell_cap)
            with torch.no_grad():
                loss = input_loss(candidate)
                better = loss > best_loss
                best = torch.where(better[:, None, None, None], candidate, best)
                best_loss = torch.maximum(best_loss, loss)
        remembered = torch.zeros_like(selected)
        for step in range(iterations):
            candidate = candidate.detach().requires_grad_(True)
            loss = input_loss(candidate)
            gradient = torch.autograd.grad(loss.sum(), candidate)[0][:, channel:channel+1]
            if method=='mi_fgsm':
                # L1 gradient normalization is NOT an L1 disturbance limit.
                size = gradient.flatten(1).abs().sum(1).clamp_min(1e-12)
                remembered = momentum*remembered + gradient/size[:, None, None, None]
                update = remembered.sign()
            else:
                if norm=='l1' and cell_cap is not None:
                    # Let other cells compete when the strongest cell is already capped.
                    current_delta = candidate[:, channel:channel+1].detach()-selected
                    blocked = (current_delta.abs() >= cell_cap-1e-7) & (current_delta*gradient > 0)
                    gradient = gradient.masked_fill(blocked, 0)
                update = direction(gradient, norm)
            if method=='one_step':
                alpha = radius                  # Always one full-budget gradient step.
            elif step_size is not None:
                alpha = step_size
            elif method=='mi_fgsm':
                alpha = radius/steps
            else:
                alpha = radius if step==0 else 2*radius/steps
            delta = candidate[:, channel:channel+1].detach()-selected + alpha*update
            candidate = original.clone()
            # Reproject ACCUMULATED change: k new cells per step is not an L0 budget.
            candidate[:, channel:channel+1] += project(delta, radius, norm, k, cell_cap)
            with torch.no_grad():
                loss = input_loss(candidate)
                better = loss > best_loss
                best = torch.where(better[:, None, None, None], candidate, best)
                best_loss = torch.maximum(best_loss, loss)
    return best.detach()
