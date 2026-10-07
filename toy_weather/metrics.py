"""Errors over all samples and pixels, not an average of batch RMSE values."""
import torch


def mse(prediction, target):
    """Two [B,1,H,W] tensors -> scalar mean squared error (generic ML)."""
    return (prediction - target).square().mean()


def latitude_weighted_mse(prediction, target, latitudes):
    """[B,1,H,W], [H] latitudes in degrees -> area-weighted scalar MSE.

    Weather-specific: cells shrink toward the poles on a regular lat/lon grid.
    Normalize cosine weights so their mean is one. RMSE is sqrt of this result.
    """
    lat = torch.as_tensor(latitudes, device=prediction.device, dtype=prediction.dtype)
    if lat.numel() != prediction.shape[-2] or (lat.abs() > 90).any():
        raise ValueError("Supply one valid latitude per grid row.")
    weights = torch.cos(torch.deg2rad(lat)).clamp_min(0)
    if weights.mean() <= 1e-8:
        raise ValueError("Latitude weights have zero area.")
    weights = (weights / weights.mean()).view(1, 1, -1, 1)
    return ((prediction - target).square() * weights).mean()


def evaluate_loader(model, loader, device, std_k, latitudes=None):
    """Batches [B,1,H,W] -> metrics dict. Physical errors multiply by training std.

    Persistence predicts y=x. It is a useful simple weather baseline.
    """
    model.eval()
    squared, baseline, weighted, count = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            prediction = model(x)
            squared += (prediction - y).square().sum().item()
            baseline += (x - y).square().sum().item()
            count += y.numel()
            if latitudes is not None:
                weighted += latitude_weighted_mse(prediction, y, latitudes).item() * y.numel()
    result = {"mse_normalized": squared / count,
              "rmse_normalized": (squared / count)**0.5,
              "mse_k2": squared / count * std_k**2,
              "rmse_k": (squared / count)**0.5 * std_k,
              "persistence_rmse_k": (baseline / count)**0.5 * std_k}
    if latitudes is not None:
        result["latitude_weighted_rmse_k"] = (weighted / count)**0.5 * std_k
    return result
