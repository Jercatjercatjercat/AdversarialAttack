"""Ways to change an input: random noise, a single gradient step, and PGD.

All functions take normalized tensors [B,1,H,W] and return the same shape.
The comparison script converts physical Kelvin budgets into normalized units.
Limits apply to EACH sample, never to the whole batch together.
"""
import math
import torch
from torch.nn import functional as F


def check_budget(epsilon, norm):
    """Check scalar settings; L2 means the flattened map norm, not a matrix operator norm."""
    if not math.isfinite(epsilon) or epsilon < 0 or norm not in ("linf", "l2"):
        raise ValueError("Use finite epsilon >= 0 and norm 'linf' or 'l2'.")


def direction(gradient, norm):
    """[B,1,H,W] gradients -> same shape with unit size in the chosen norm.

    L-infinity: increase/decrease each cell using its gradient sign.
    L2: divide each sample's gradient by its own length; strong cells get more change.
    A zero gradient gives zero change, without division by zero.
    """
    if norm == "linf":
        return gradient.sign()
    lengths = gradient.flatten(1).norm(p=2, dim=1).view(-1, 1, 1, 1)
    return gradient / lengths.clamp_min(1e-12)


def project(change, epsilon, norm):
    """[B,1,H,W] changes -> same shape inside each sample's allowed region."""
    if norm == "linf":
        return change.clamp(-epsilon, epsilon)
    lengths = change.flatten(1).norm(p=2, dim=1).view(-1, 1, 1, 1)
    scale = (epsilon / lengths.clamp_min(1e-12)).clamp(max=1)
    return change * scale


def random_noise(x, epsilon, norm="linf", generator=None):
    """[B,1,H,W] -> changed input, independent of the model and target.

    Use the FULL budget so the random baseline is not weakened by smaller changes.
    Linf: independent random +/- epsilon at every cell (RMS also equals epsilon).
    L2: random Gaussian direction scaled to length epsilon for each sample.
    Generate on CPU so the same seed works on CPU, CUDA and Mac devices.
    """
    check_budget(epsilon, norm)
    if norm == "linf":
        values = torch.randint(0, 2, x.shape, generator=generator, device="cpu")
        change = (2 * values - 1).to(device=x.device, dtype=x.dtype)
    else:
        values = torch.randn(x.shape, generator=generator, device="cpu", dtype=x.dtype)
        change = direction(values.to(x.device), "l2")
    return (x.detach() + epsilon * change).detach()


def fgsm(model, x, y, epsilon, norm="linf"):
    """[B,1,H,W] x,y -> one changed input per sample; keep target and model fixed.

    With Linf this is traditional sign-based FGSM. With L2 it is a one-step
    normalized-gradient attack; the comparison table labels it 'one_step_l2'.
    """
    check_budget(epsilon, norm)
    changed = x.detach().clone().requires_grad_(True)
    gradient = torch.autograd.grad(F.mse_loss(model(changed), y), changed)[0]
    return (changed + epsilon * direction(gradient, norm)).detach()


def pgd(model, x, y, epsilon, steps=10, step_size=None, norm="linf"):
    """[B,1,H,W] -> repeated uphill input changes, projected after every step.

    The original input anchors the budget. Remember the strongest candidate for
    EACH sample, including clean and single-step inputs: more steps can overshoot.
    This educational version starts clean and has no random restarts.
    """
    check_budget(epsilon, norm)
    if steps < 1 or (step_size is not None and (not math.isfinite(step_size) or step_size <= 0)):
        raise ValueError("Use steps >= 1 and a finite positive step size.")
    original = x.detach()
    changed = original.clone()
    step_size = 2 * epsilon / steps if step_size is None else step_size
    best = fgsm(model, x, y, epsilon, norm)
    with torch.no_grad():
        best_loss = (model(best) - y).square().flatten(1).mean(1)
        clean_loss = (model(original) - y).square().flatten(1).mean(1)
        use_clean = clean_loss > best_loss
        best[use_clean] = original[use_clean]
        best_loss = torch.maximum(best_loss, clean_loss)
    for _ in range(steps):
        changed = changed.detach().requires_grad_(True)
        gradient = torch.autograd.grad(F.mse_loss(model(changed), y), changed)[0]
        changed = changed.detach() + step_size * direction(gradient, norm)
        changed = original + project(changed - original, epsilon, norm)
        with torch.no_grad():
            candidate_loss = (model(changed) - y).square().flatten(1).mean(1)
            improved = candidate_loss > best_loss
            best[improved] = changed[improved]
            best_loss = torch.maximum(best_loss, candidate_loss)
    return best.detach()
