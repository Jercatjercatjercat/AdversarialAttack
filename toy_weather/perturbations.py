"""Bounded changes to ONE variable in the CURRENT map; history/clocks stay fixed."""
import torch


def project(delta, radius, norm):
    """Changes [B,1,H,W] -> same shape inside each example's normalized budget."""
    if norm == 'linf':
        return delta.clamp(-radius, radius)
    length = delta.flatten(1).norm(dim=1).clamp_min(1e-12)[:, None, None, None]
    return delta * (radius/length).clamp(max=1)


def direction(gradient, norm):
    """[B,1,H,W] input derivatives -> sign or unit-L2 ascent direction."""
    if norm == 'linf':
        return gradient.sign()
    length = gradient.flatten(1).norm(dim=1).clamp_min(1e-12)[:, None, None, None]
    return gradient/length


def change_input(model, x, y, channel, radius, method, norm='linf', steps=10, generator=None):
    """x [B,input,H,W], y [B,C,H,W] -> changed x with the same shape.

    radius is in normalized units for ONE current-weather channel. All other
    channels stay exactly unchanged. Model parameters are frozen by the caller.
    The true future y stays fixed. No physical-consistency constraints are implied.
    """
    if radius < 0 or steps < 1 or norm not in ('linf', 'l2') or method not in ('random', 'one_step', 'pgd'):
        raise ValueError('Invalid attack settings.')
    original = x.detach()
    if radius == 0:
        return original.clone()
    selected = original[:, channel:channel+1]
    if method == 'random':
        noise = torch.randn(selected.shape, device=x.device, dtype=x.dtype, generator=generator)
        delta = radius*(noise.sign() if norm=='linf' else direction(noise, norm))
        changed = original.clone()
        changed[:, channel:channel+1] += delta
        return changed
    with torch.no_grad():
        best_loss = (model(original)-y).square().mean((1, 2, 3))
    best, candidate = original.clone(), original.clone()
    iterations = 1 if method=='one_step' else steps
    for step in range(iterations):
        candidate = candidate.detach().requires_grad_(True)
        losses = (model(candidate)-y).square().mean((1, 2, 3))
        gradient = torch.autograd.grad(losses.sum(), candidate)[0][:, channel:channel+1]
        # First include the full one-step candidate. Further steps refine it within
        # the same budget. Retain the highest-loss candidate for EACH example.
        step_size = radius if step==0 else 2*radius/steps
        delta = candidate[:, channel:channel+1].detach()-selected + step_size*direction(gradient, norm)
        candidate = original.clone()
        candidate[:, channel:channel+1] += project(delta, radius, norm)
        with torch.no_grad():
            losses = (model(candidate)-y).square().mean((1, 2, 3))
            better = losses > best_loss
            best = torch.where(better[:, None, None, None], candidate, best)
            best_loss = torch.maximum(best_loss, losses)
    return best.detach()
