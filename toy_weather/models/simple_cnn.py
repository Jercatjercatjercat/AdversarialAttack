"""Three learned spatial filters. No detaching or NumPy inside the forward pass."""
from torch import nn


class SimpleCNN(nn.Module):
    def __init__(self, padding_mode="replicate"):
        """Circular borders suit synthetic maps; replicated borders suit a local crop."""
        super().__init__()
        # All kernels look at 3x3 neighbourhoods. Padding 1 preserves H and W.
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1, padding_mode=padding_mode)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1, padding_mode=padding_mode)
        self.conv3 = nn.Conv2d(32, 1, 3, padding=1, padding_mode=padding_mode)
        self.relu = nn.ReLU()

    def forward(self, x):
        """[B,1,H,W] -> [B,1,H,W]. Generic CNN; channels inside are learned features."""
        features = self.relu(self.conv1(x))   # [B,32,H,W]: 32 views of temperature.
        features = self.relu(self.conv2(features))  # [B,32,H,W]: combine those views.
        return self.conv3(features)          # [B,1,H,W]: future normalized temperature.
        # No final ReLU: normalized temperature can be negative.
