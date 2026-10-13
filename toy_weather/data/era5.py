"""Exact six-hour history/target pairing and training-only, per-variable scaling."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from config import VARIABLES


def pair_indices(times, start, end):
    """[T] times -> [N,3] indices of (t-6h,t,t+6h), all inside one split.

    Split STATES first, so no input/target can cross train/validation/test dates.
    Missing observations are skipped. Three nearby rows are not necessarily 6h apart.
    """
    times = np.asarray(times, dtype='datetime64[ns]')
    if np.isnat(times).any() or np.any(np.diff(times) <= np.timedelta64(0, 'ns')):
        raise ValueError('Times must be unique, finite and strictly increasing.')
    selected = np.flatnonzero((times >= np.datetime64(start)) & (times < np.datetime64(end)))
    lookup = {int(times[i].astype('int64')): i for i in selected}
    six = int(np.timedelta64(6, 'h').astype('timedelta64[ns]').astype('int64'))
    triples = []
    for i in selected:
        now = int(times[i].astype('int64'))
        if now - six in lookup and now + six in lookup:
            triples.append([lookup[now-six], i, lookup[now+six]])
    if not triples:
        raise ValueError(f'No complete six-hour triples inside {start}..{end}.')
    return np.asarray(triples, dtype='int64')


def training_statistics(fields, triples, history=2):
    """Physical [T,C,H,W] -> one mean/std per variable from UNIQUE training inputs.

    Count each historical/current state once; never include validation/test states.
    Future targets do not get independently fitted statistics. Std is population std.
    """
    if history not in (1, 2):
        raise ValueError('History must be 1 or 2.')
    indices = np.unique(triples[:, 2-history:2])
    inputs = fields[indices].astype('float64')
    mean = inputs.mean(axis=(0, 2, 3))
    std = inputs.std(axis=(0, 2, 3))
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 1e-8):
        raise ValueError('Each training variable needs finite, nonzero variation.')
    return {'mean': mean.tolist(), 'std': std.tolist(),
            'fitted_on': 'unique training input states only', 'input_state_count': len(indices)}


def clock_features(time, height, width):
    """One UTC datetime -> [4,H,W] sine/cosine features for hour and season.

    Time is known at prediction time, so this uses no future observed weather.
    Sine/cosine handles midnight/year boundaries smoothly; features are not targets.
    """
    time = np.datetime64(time, 'ns')
    day = time.astype('datetime64[D]')
    year = time.astype('datetime64[Y]')
    next_year = year + np.timedelta64(1, 'Y')
    hour_fraction = float((time-day) / np.timedelta64(1, 'D'))
    year_fraction = float((time-year.astype('datetime64[ns]')) /
                          (next_year.astype('datetime64[ns]')-year.astype('datetime64[ns]')))
    angles = 2*np.pi*np.asarray([hour_fraction, year_fraction])
    features = np.asarray([np.sin(angles[0]), np.cos(angles[0]),
                           np.sin(angles[1]), np.cos(angles[1])], dtype='float32')
    return np.broadcast_to(features[:, None, None], (4, height, width)).copy()


def denormalize(values, stats):
    """Normalized [...,C,H,W] numpy/tensor -> physical maps with the SAME shape.

    Temperature is Kelvin, wind m/s, pressure hPa, humidity g/kg. Subtract 273.15 only when displaying
    temperature/dew point in Celsius; never apply it to wind or pressure.
    """
    if isinstance(values, torch.Tensor):
        mean = values.new_tensor(stats['mean'])[:, None, None]
        std = values.new_tensor(stats['std'])[:, None, None]
    else:
        mean = np.asarray(stats['mean'])[:, None, None]
        std = np.asarray(stats['std'])[:, None, None]
    return values*std + mean


class WeatherDataset(Dataset):
    def __init__(self, fields, times, triples, stats, history=2, time_features=True):
        """Keep shared physical arrays; normalize only the requested [C,H,W] maps."""
        if history not in (1, 2):
            raise ValueError('History must be 1 or 2.')
        self.fields, self.times, self.triples = fields, times, triples
        self.mean = np.asarray(stats['mean'], dtype='float32')[:, None, None]
        self.std = np.asarray(stats['std'], dtype='float32')[:, None, None]
        self.history, self.time_features = history, time_features

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, index):
        """One example -> x [history*C+clock,H,W], y [C,H,W], both float32.

        Order: all variables at t-6h, all variables at t, then four clock features.
        The target is never supplied to the model. Current weather is the last C
        WEATHER channels, before the optional clock channels.
        """
        triple = self.triples[index]
        inputs = triple[2-self.history:2]
        maps = (self.fields[inputs]-self.mean)/self.std  # [L,C,H,W].
        x = maps.reshape(-1, *maps.shape[-2:])          # [L*C,H,W].
        if self.time_features:
            clock = clock_features(self.times[triple[1]], *x.shape[-2:])
            x = np.concatenate([x, clock], axis=0)
        y = (self.fields[triple[2]]-self.mean)/self.std
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())


def load_data(path, variables=VARIABLES, history=2, time_features=True):
    """Prepared NPZ -> split datasets, fitted stats, metadata and coordinates.

    Only named variables are selected. Shared arrays avoid duplicating overlapping
    histories. Train stats are reused for every time, target and evaluation split.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as saved:
        fields = saved['fields']
        times = saved['times'].astype('datetime64[ns]')
        metadata = json.loads(str(saved['metadata']))
        lat, lon = saved['latitude'].copy(), saved['longitude'].copy()
    selected = [metadata['variables'].index(v) for v in variables]
    fields = fields[:, selected]
    if fields.ndim != 4 or not np.isfinite(fields).all():
        raise ValueError('Expected finite [T,C,H,W] physical fields.')
    boundaries = metadata['boundaries']
    triples = {name: pair_indices(times, boundaries[i], boundaries[i+1])
               for i, name in enumerate(('train', 'validation', 'test'))}
    stats = training_statistics(fields, triples['train'], history)
    datasets = {name: WeatherDataset(fields, times, rows, stats, history, time_features)
                for name, rows in triples.items()}
    metadata.update(variables=list(variables), units=[metadata['units'][i] for i in selected],
                    counts={name: len(rows) for name, rows in triples.items()},
                    prepared_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    history=history, time_features=time_features)
    return datasets, stats, metadata, lat, lon
