"""Shared, explicit saving/loading so training and evaluation use identical data."""
import json
from pathlib import Path
import numpy as np
import torch
from data.era5 import load_data
from models.residual_cnn import ResidualCNN


def configure(seed=42, device='cpu'):
    """Generic ML: reproducible starting weights/shuffle, small CPU thread count."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    return torch.device(device)


def save_json(path, value):
    """Save labelled values for interpretation, not arrays of weather maps."""
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def load_run(directory, data_path, device):
    """Trusted local checkpoint -> frozen model, split datasets and physical stats.

    Freezing parameters still allows gradients with respect to input weather.
    Explicit --data avoids stale machine-specific absolute paths after repo moves.
    """
    checkpoint = torch.load(Path(directory)/'best_model.pt', map_location=device, weights_only=True)
    settings = checkpoint['model_settings']
    datasets, stats, metadata, lat, lon = load_data(
        data_path, checkpoint['metadata']['variables'], settings['history'], settings['time_features'])
    if metadata['prepared_sha256'] != checkpoint['metadata']['prepared_sha256']:
        raise ValueError('Data contents differ from the dataset used to train this checkpoint.')
    if stats != checkpoint['stats']:
        raise ValueError('Normalization differs from the checkpoint.')
    model = ResidualCNN(**settings).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, datasets, stats, metadata, lat, lon


def print_scores(results):
    """Human-readable per-variable RMSE; physical units are never mixed together."""
    for name, score in results['variables'].items():
        gain = score['improvement_percent']
        gain_text = f'{gain:.1f}%' if gain is not None else 'undefined (zero baseline error)'
        print(f"{name:5s} RMSE={score['rmse']:.4f} {score['unit']:4s}  "
              f"no-change={score['persistence_rmse']:.4f}  improvement={gain_text}")
