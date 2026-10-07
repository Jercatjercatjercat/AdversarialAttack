"""Untargeted attacks: slightly change an input to increase its forecast MSE.

The label (true future) stays fixed. epsilon is a per-pixel maximum change, NOT
an average change. Call these on a frozen evaluation model, with gradients enabled.
"""
import torch
from torch.nn import functional as F


def fgsm(model, x, y, epsilon):
    """Normalized x,y [B,1,H,W] -> changed x of same shape; one uphill step.

    epsilon is in NORMALIZED units. The CLI converts Kelvin limits using train std.
    Generic ML attack: no assumption about which input variable is temperature.
    """
    if epsilon < 0:
        raise ValueError("epsilon must be nonnegative.")
    changed = x.detach().clone().requires_grad_(True)
    loss = F.mse_loss(model(changed), y)
    gradient = torch.autograd.grad(loss, changed)[0]
    # sign says increase or decrease each pixel; all changes obey the same limit.
    return (changed + epsilon * gradient.sign()).detach()


def pgd(model, x, y, epsilon, steps=10, step_size=None):
    """[B,1,H,W] -> same shape; repeated uphill steps inside x ± epsilon.

    Deterministic start at the clean input keeps the beginner example reproducible.
    Remember the strongest candidate per sample, including the clean input: a fixed
    step can overshoot, so the final iterate is not always the strongest attack.
    """
    if epsilon < 0 or steps < 1 or (step_size is not None and step_size <= 0):
        raise ValueError("Use epsilon >= 0, steps >= 1 and positive step size.")
    original = x.detach()
    changed = original.clone()
    step_size = step_size if step_size is not None else 2 * epsilon / steps
    best = original.clone()
    with torch.no_grad():
        best_loss = (model(best) - y).square().flatten(1).mean(1)
    for _ in range(steps):
        changed = changed.detach().requires_grad_(True)
        loss = F.mse_loss(model(changed), y)
        gradient = torch.autograd.grad(loss, changed)[0]
        changed = changed.detach() + step_size * gradient.sign()
        # Projection: clip the accumulated change, not the temperature itself.
        changed = original + (changed - original).clamp(-epsilon, epsilon)
        with torch.no_grad():
            candidate_loss = (model(changed) - y).square().flatten(1).mean(1)
            improved = candidate_loss > best_loss
            best[improved] = changed[improved]
            best_loss = torch.maximum(best_loss, candidate_loss)
    return best.detach()
