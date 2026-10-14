"""A small CNN that learns corrections instead of rebuilding the entire weather."""
import torch
from torch import nn


class ResidualCNN(nn.Module):
    def __init__(self, variables=5, history=2, time_features=True, width=64,
                 depth=5, residual=True, feature_skip=True):
        """Defaults: [B,14,H,W] -> [B,5,H,W], with five 3x3 convolutions.

        Each hidden channel is a LEARNED feature, not another measured variable.
        Padding repeats the local crop's edge values and preserves spatial size.
        No dropout: a deterministic baseline is simpler for comparing input attacks.
        """
        super().__init__()
        if variables < 1 or history not in (1, 2) or width < 1 or depth < 3:
            raise ValueError('Positive variables/width, history 1/2, depth >= 3 required.')
        self.settings = dict(variables=variables, history=history, time_features=time_features,
                             width=width, depth=depth, residual=residual, feature_skip=feature_skip)
        self.variables, self.history, self.residual = variables, history, residual
        self.feature_skip = feature_skip
        incoming = variables*history + (4 if time_features else 0)
        self.input_conv = nn.Conv2d(incoming, width, 3, padding=1, padding_mode='replicate')
        self.hidden = nn.ModuleList([nn.Conv2d(width, width, 3, padding=1,
                                             padding_mode='replicate') for _ in range(depth-2)])
        self.output_conv = nn.Conv2d(width, variables, 3, padding=1, padding_mode='replicate')
        if residual:
            # Start at EXACT persistence: output corrections are initially zero.
            # The last layer learns first; earlier layers receive gradients after it moves.
            nn.init.zeros_(self.output_conv.weight)
            nn.init.zeros_(self.output_conv.bias)

    def current_state(self, x):
        """[B,L*C+clock,H,W] -> normalized current weather [B,C,H,W]."""
        begin = (self.history-1)*self.variables
        return x[:, begin:begin+self.variables]

    def forward(self, x):
        """Learn a correction [B,C,H,W], then add the current state in same units.

        Input features influence the forecast through BOTH the CNN and the direct
        current-state connection. No detach/NumPy: gradients can reach every input.
        """
        first = torch.relu(self.input_conv(x))       # [B,width,H,W].
        features = first
        for conv in self.hidden:
            features = torch.relu(conv(features))
        if self.feature_skip:
            features = features + first            # Internal feature skip connection.
        change = self.output_conv(features)          # [B,C,H,W], may be negative.
        return self.current_state(x)+change if self.residual else change
