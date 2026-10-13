"""Small defaults. Paths are relative to this file, regardless of your terminal location."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
GRID_SIZE = 16
SEED = 42
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 1e-3


def run_directory(dataset, output_dir=None):
    """Return the folder for one dataset's model and plots (no tensor operations)."""
    return Path(output_dir) if output_dir else OUTPUTS / dataset
