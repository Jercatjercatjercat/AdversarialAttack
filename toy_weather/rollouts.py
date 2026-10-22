"""Six-hour forecasts fed back as inputs; future observations are targets only."""
import numpy as np
import torch
from torch.utils.data import Dataset
from data.era5 import clock_features


class RolloutDataset(Dataset):
    def __init__(self, base, steps):
        """Existing split -> cases with ALL future times inside that SAME split.

        base keeps physical weather [time,C,H,W] and training-only statistics.
        Missing six-hour times invalidate a case; neighbouring rows aren't enough.
        """
        if steps < 1:
            raise ValueError('Forecast steps must be positive.')
        self.base, self.steps = base, steps
        allowed = np.unique(base.triples)
        lookup = {int(base.times[i].astype('datetime64[ns]').astype('int64')): i for i in allowed}
        interval = int(np.timedelta64(6, 'h').astype('timedelta64[ns]').astype('int64'))
        indices, future = [], []
        for index, triple in enumerate(base.triples):
            now = int(base.times[triple[1]].astype('datetime64[ns]').astype('int64'))
            keys = [now + interval*j for j in range(1, steps+1)]
            if all(key in lookup for key in keys):
                indices.append(index)
                future.append([lookup[key] for key in keys])
        self.base_indices = np.asarray(indices, dtype=int)
        self.future = np.asarray(future, dtype=int).reshape(-1, steps)
        self.times = base.times[base.triples[self.base_indices, 1]]
        if not len(indices):
            raise ValueError('No complete rollout trajectories inside this split.')

    def __len__(self):
        return len(self.base_indices)

    def __getitem__(self, index):
        """Return x [L*C+clock,H,W], targets [K,C,H,W], clocks [K,4,H,W].

        Clocks describe CURRENT time at each forecast call, not its future target.
        They use calendar information only; future weather never enters an input.
        """
        x, _ = self.base[int(self.base_indices[index])]
        targets = (self.base.fields[self.future[index]]-self.base.mean)/self.base.std
        height, width = x.shape[-2:]
        if self.base.time_features:
            clocks = np.stack([clock_features(self.times[index]+np.timedelta64(6*j, 'h'), height, width)
                               for j in range(self.steps)])
        else:
            clocks = np.empty((self.steps, 0, height, width), dtype='float32')
        return x, torch.from_numpy(targets.astype('float32')), torch.from_numpy(clocks)


def rollout(model, x, clocks):
    """x [B,L*C+clock,H,W], clocks [B,K,clock,H,W] -> [B,K,C,H,W].

    Every predicted variable is fed back. Shift out the oldest state, append the
    prediction, and advance known calendar features. Keep the FULL gradient graph:
    a later forecast's error must differentiate back through all earlier calls.
    """
    predictions, current = [], x
    weather_channels = model.history*model.variables
    for step in range(clocks.shape[1]):
        prediction = model(current)
        predictions.append(prediction)
        if step+1 < clocks.shape[1]:
            history = current[:, :weather_channels]
            current = torch.cat([history[:, model.variables:], prediction, clocks[:, step+1]], dim=1)
    return torch.stack(predictions, dim=1)


def window_loss(model, x, targets, clocks, steps, variable_index=0):
    """One MSE per case [B], averaged over lead steps 1..steps and all cells.

    This is an all-steps window loss, not loss only at the final lead time. Model
    weights and true future stay fixed. Uses normalized forecast coordinates.
    """
    prediction = rollout(model, x, clocks[:, :steps])
    error = prediction[:, :, variable_index]-targets[:, :steps, variable_index]
    return error.square().mean((1, 2, 3))
