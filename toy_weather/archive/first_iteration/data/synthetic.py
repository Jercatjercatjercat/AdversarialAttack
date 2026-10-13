"""Make independent pairs of smooth temperature maps, measured in Kelvin."""
import numpy as np
from scipy.ndimage import gaussian_filter


def generate_pairs(count, size=16, seed=42):
    """Return x,y [N,1,H,W] in Kelvin. This artificial weather rule is not ERA5.

    Each starting map is independent; its future map depends on that starting map.
    A periodic domain means the left/right and top/bottom edges join together.
    """
    rng = np.random.default_rng(seed)
    # Smooth only spatial axes, never the sample axis: samples stay independent.
    field = gaussian_filter(rng.normal(size=(count, size, size)),
                            sigma=(0, 1.4, 1.4), mode="wrap")
    rows, cols = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    for sample in range(count):
        for _ in range(2):
            cy, cx = rng.uniform(0, size, 2)
            # Shortest distances across joined edges make the blobs periodic too.
            dy = np.minimum(abs(rows - cy), size - abs(rows - cy))
            dx = np.minimum(abs(cols - cx), size - abs(cols - cx))
            width = rng.uniform(1.5, 3.5)
            field[sample] += rng.uniform(-1, 1) * np.exp(
                -(dx**2 + dy**2) / (2 * width**2))
    # Give each map a 4 K spatial standard deviation and a varying mean near 285 K.
    field -= field.mean(axis=(1, 2), keepdims=True)
    field /= field.std(axis=(1, 2), keepdims=True)
    x = 285 + rng.normal(0, 2, (count, 1, 1)) + 4 * field
    # Fixed one-cell eastward motion is learnable; blur reduces temperature contrast.
    shifted = np.roll(x, shift=1, axis=2)
    blurred = gaussian_filter(x, sigma=(0, 0.8, 0.8), mode="wrap")
    y = 0.85 * shifted + 0.15 * blurred + rng.normal(0, 0.1, x.shape)
    return x[:, None].astype("float32"), y[:, None].astype("float32")


def synthetic_splits(size=16, seed=42, counts=(4000, 500, 500)):
    """Return physical [N,1,H,W] arrays for three independent, reproducible splits."""
    return {name: generate_pairs(n, size, seed + i)
            for i, (name, n) in enumerate(zip(("train", "validation", "test"), counts))}
