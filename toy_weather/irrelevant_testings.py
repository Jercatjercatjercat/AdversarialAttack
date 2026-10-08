from pathlib import Path
import torch

# Start from this script's folder, regardless of where your terminal is open.
checkpoint_path = Path(__file__).resolve().parent / "outputs/synthetic/best_model.pt"

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True,
)

weights = checkpoint["model_state"]

# Show every parameter's name and shape.
for name, value in weights.items():
    print(name, tuple(value.shape))

# Inspect the first convolution's first learned 3×3 filter.
print(weights["conv1.weight"][0, 0])  # [out_channel, in_channel, kernel height, kernel width]
