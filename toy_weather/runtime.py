"""A few shared setup functions so every script uses the same data and checkpoint."""
import json
from pathlib import Path
import numpy as np
import torch
from config import OUTPUTS
from data.synthetic import synthetic_splits
from data.era5 import load_prepared, training_statistics, tensor_splits
from models.simple_cnn import SimpleCNN


def configure(seed, device="cpu"):
    """Set generic ML reproducibility defaults; return the requested torch device."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Tiny maps often run faster with a few CPU threads than with dozens.
    torch.set_num_threads(2)
    return torch.device(device)


def get_data(dataset, data_path=None, size=16, seed=42, stats=None):
    """Return datasets of [1,H,W] pairs, train stats, and provenance metadata."""
    if dataset == "synthetic":
        pairs = synthetic_splits(size, seed)
        stats = stats or training_statistics(pairs["train"][0])
        return tensor_splits(pairs, stats), stats, {"source": "synthetic", "size": size, "seed": seed}
    path = Path(data_path) if data_path else OUTPUTS / "era5" / "prepared.npz"
    if not path.exists():
        raise FileNotFoundError(f"Prepare a local file first: python -m data.preprocessing --input FILE.nc --output {path}")
    datasets, prepared_stats, metadata = load_prepared(path)
    if stats is not None and stats != prepared_stats:
        raise ValueError("Prepared data statistics differ from the checkpoint. Use its original prepared file.")
    metadata["prepared_path"] = str(path.resolve())
    return datasets, prepared_stats, metadata


def load_run(dataset, directory, data_path, device):
    """Trusted local checkpoint -> frozen eval model, test data and recorded metadata."""
    checkpoint = torch.load(Path(directory) / "best_model.pt", map_location=device, weights_only=True)
    if checkpoint["dataset"] != dataset:
        raise ValueError("Checkpoint dataset does not match --dataset.")
    model = SimpleCNN(checkpoint["padding_mode"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    # Freeze theta while preserving the ability to differentiate with respect to x.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    meta = checkpoint["metadata"]
    datasets, stats, metadata = get_data(dataset, data_path or meta.get("prepared_path"),
        meta.get("size", 16), meta.get("seed", 42), checkpoint["stats"])
    return model, datasets, stats, metadata


def save_json(path, value):
    """Write small readable metadata/results; no tensor operations."""
    Path(path).write_text(json.dumps(value, indent=2) + "\n")
