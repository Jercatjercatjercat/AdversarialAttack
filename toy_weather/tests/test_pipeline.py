"""Checks for real failure modes: leakage, missing times, units and attack bounds."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
import torch
import xarray as xr
from data.preprocessing import chronological_pairs, prepare_fields, save_prepared
from data.era5 import load_prepared, denormalize
from data.synthetic import generate_pairs
from models.simple_cnn import SimpleCNN
from attacks import fgsm, pgd
from metrics import latitude_weighted_mse


def fixture_dataset():
    """Small hourly local-file fixture, explicitly fabricated rather than real ERA5."""
    times = np.arange(np.datetime64("2020-01-01"), np.datetime64("2020-01-31"), np.timedelta64(1, "h"))
    lat = np.array([60., 55., 50., 45.])  # Descending latitude is common in ERA5.
    lon = np.array([350., 355., 0., 5., 10.])
    values = 280 + np.arange(len(times))[:, None, None] * .01 + lat[None, :, None] * .1 + np.zeros((1, 1, len(lon)))
    return xr.Dataset({"t2m": (("valid_time", "latitude", "longitude"), values, {"units": "K"})},
                      coords={"valid_time": times, "latitude": lat, "longitude": lon})


def test_exact_lead_and_no_shared_states():
    ds = fixture_dataset()
    values, times, _, _ = prepare_fields(ds, time="valid_time")
    # Removing one timestamp must not make a 12-hour jump into a six-hour pair.
    values, times = np.delete(values, 5, axis=0), np.delete(times, 5)
    _, split_times = chronological_pairs(values, times)
    states = []
    for inputs, targets in split_times.values():
        assert np.all(targets - inputs == np.timedelta64(6, "h"))
        states.append(set(inputs.tolist()) | set(targets.tolist()))
    assert states[0].isdisjoint(states[1])
    assert states[0].isdisjoint(states[2])
    assert states[1].isdisjoint(states[2])


def test_training_statistics_ignore_future(tmp_path):
    ds = fixture_dataset()
    first = save_prepared(ds, tmp_path / "first.npz", time="valid_time")
    # Only change values AFTER the raw training timestamp boundary.
    ds["t2m"].values[int(.7 * len(ds.valid_time)):] += 100
    second = save_prepared(ds, tmp_path / "second.npz", time="valid_time")
    assert first == second
    datasets, stats, _ = load_prepared(tmp_path / "first.npz")
    x = datasets["train"].tensors[0]
    assert abs(x.mean().item()) < 1e-5
    assert abs(x.std(unbiased=False).item() - 1) < 1e-5
    restored = denormalize(x, stats)
    assert torch.isfinite(restored).all()


@pytest.mark.parametrize("method", [fgsm, pgd])
def test_attack_limit_gradient_and_frozen_weights(method):
    # Known differentiable forecast rule makes the expected uphill effect unambiguous.
    model = torch.nn.Conv2d(1, 1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1)
    model.requires_grad_(False)
    x, y = torch.ones(2, 1, 4, 4), torch.zeros(2, 1, 4, 4)
    changed = method(model, x, y, .1)
    assert (changed - x).abs().max() <= .100001
    assert (model(changed) - y).square().mean() > (model(x) - y).square().mean()
    assert torch.equal(method(model, x, y, 0), x)
    assert model.weight.grad is None
    assert not changed.requires_grad


def test_real_cnn_input_gradient():
    torch.manual_seed(42)
    model = SimpleCNN("circular").requires_grad_(False)
    x = torch.randn(1, 1, 16, 16, requires_grad=True)
    prediction = model(x)
    assert prediction.shape == x.shape
    prediction.square().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_weights_and_synthetic_reproducibility():
    x, y = generate_pairs(4)
    x_again, y_again = generate_pairs(4)
    assert np.array_equal(x, x_again) and np.array_equal(y, y_again)
    assert x.shape == y.shape == (4, 1, 16, 16)
    # All equal errors must yield the same MSE with or without latitude weights.
    prediction, target = torch.ones(2, 1, 3, 4), torch.zeros(2, 1, 3, 4)
    assert latitude_weighted_mse(prediction, target, [-60, 0, 60]).item() == pytest.approx(1)


def test_netcdf_and_zarr_agree(tmp_path):
    ds = fixture_dataset()
    nc, zarr = tmp_path / "fixture.nc", tmp_path / "fixture.zarr"
    ds.to_netcdf(nc)
    ds.to_zarr(zarr)
    with xr.open_dataset(nc) as source:
        stats_nc = save_prepared(source, tmp_path / "nc.npz", time="valid_time")
    with xr.open_zarr(zarr, chunks=None) as source:
        stats_zarr = save_prepared(source, tmp_path / "zarr.npz", time="valid_time")
    assert stats_nc == stats_zarr


def test_rejects_unknown_units_and_missing_values():
    ds = fixture_dataset()
    ds.t2m.attrs["units"] = "unknown"
    with pytest.raises(ValueError, match="units"):
        prepare_fields(ds, time="valid_time")
    ds.t2m.attrs["units"] = "K"
    ds.t2m.values[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Missing"):
        prepare_fields(ds, time="valid_time")
