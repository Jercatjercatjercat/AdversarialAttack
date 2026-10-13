"""Read prepared real-data pairs. This module never downloads weather data."""
import json
import numpy as np
import torch
from torch.utils.data import TensorDataset


def training_statistics(x):
    """Input: training inputs [N,1,H,W] in K. Output: two scalar statistics.

    Generic ML step: do not inspect validation or test values here.
    """
    mean, std = float(x.mean(dtype=np.float64)), float(x.std(dtype=np.float64))
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 1e-8:
        raise ValueError("Training temperatures need finite values and nonzero variation.")
    return {"mean_k": mean, "std_k": std}


def normalize(values, stats):
    """[N,1,H,W] Kelvin -> same shape, dimensionless values. Works on tensors too."""
    return (values - stats["mean_k"]) / stats["std_k"]


def denormalize(values, stats, celsius=False):
    """Any normalized array/tensor -> same shape in K, or Celsius if requested."""
    kelvin = values * stats["std_k"] + stats["mean_k"]
    return kelvin - 273.15 if celsius else kelvin


def tensor_splits(pairs, stats):
    """Physical pairs -> PyTorch datasets, each yielding two [1,H,W] tensors.

    The same training statistics transform both x and y in every split.
    """
    return {name: TensorDataset(
        torch.from_numpy(normalize(x, stats).astype("float32")),
        torch.from_numpy(normalize(y, stats).astype("float32")))
        for name, (x, y) in pairs.items()}


def load_prepared(path):
    """Load normalized pairs [N,1,H,W], statistics and geographic metadata."""
    with np.load(path, allow_pickle=False) as saved:
        stats = json.loads(str(saved["stats"]))
        metadata = json.loads(str(saved["metadata"]))
        metadata["latitudes"] = saved["latitudes"].tolist()
        datasets = {name: TensorDataset(torch.from_numpy(saved[name + "_x"].copy()),
                                        torch.from_numpy(saved[name + "_y"].copy()))
                    for name in ("train", "validation", "test")}
    return datasets, stats, metadata
