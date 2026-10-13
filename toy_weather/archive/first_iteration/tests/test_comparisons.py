"""Check budget meaning, fair noise size, frozen weights and fixed evaluation dates."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset
from perturbations import fgsm, pgd, random_noise, project
from compare_attacks import compare
from data.preprocessing import chronological_pairs, save_prepared
from data.era5 import load_prepared, denormalize
from test_pipeline import fixture_dataset


@pytest.mark.parametrize("norm", ["linf", "l2"])
@pytest.mark.parametrize("method", ["random", "fgsm", "pgd"])
def test_per_sample_limits_and_zero_budget(norm, method):
    model = torch.nn.Conv2d(1, 1, 1, bias=False).requires_grad_(False)
    with torch.no_grad():
        model.weight.fill_(1)
    x = torch.ones(3, 1, 4, 4)
    y = torch.zeros_like(x)
    epsilon = .2
    def change(budget):
        if method == "random":
            return random_noise(x, budget, norm, torch.Generator().manual_seed(42))
        if method == "fgsm":
            return fgsm(model, x, y, budget, norm)
        return pgd(model, x, y, budget, norm=norm)
    changed = change(epsilon)
    delta = (changed - x).flatten(1)
    measured = delta.abs().max(1).values if norm == "linf" else delta.norm(dim=1)
    assert torch.all(measured <= epsilon + 1e-6)
    assert torch.equal(change(0), x)
    assert model.weight.grad is None
    assert not changed.requires_grad
    if method in ("random", "fgsm"):
        assert torch.allclose(measured, torch.full_like(measured, epsilon), atol=1e-6)
    if method != "random":
        assert (model(changed) - y).square().mean() > (model(x) - y).square().mean()


def test_l2_projection_does_not_share_batch_budget():
    change = torch.tensor([[[[.5, 0.]]], [[[100., 0.]]]])
    projected = project(change, 1, "l2")
    assert torch.equal(projected[0], change[0])
    assert projected[1].flatten().norm().item() == pytest.approx(1)


def test_random_reproducible_and_full_rms_budget():
    x = torch.zeros(2, 1, 16, 16)
    for norm, radius in (("linf", .5), ("l2", 8.)):
        a = random_noise(x, radius, norm, torch.Generator().manual_seed(7))
        b = random_noise(x, radius, norm, torch.Generator().manual_seed(7))
        assert torch.equal(a, b)
        assert torch.allclose(a.flatten(1).square().mean(1).sqrt(), torch.tensor([.5, .5]), atol=1e-6)


def test_zero_gradient_stays_finite():
    model = torch.nn.Conv2d(1, 1, 1, bias=False).requires_grad_(False)
    with torch.no_grad():
        model.weight.zero_()
    x = torch.ones(2, 1, 4, 4)
    for method in (fgsm, pgd):
        assert torch.equal(method(model, x, x, .5, norm="l2"), x)


def test_comparison_reports_physical_changes_and_same_clean_error():
    model = torch.nn.Conv2d(1, 1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1)
    data = TensorDataset(torch.ones(4, 1, 4, 4), torch.zeros(4, 1, 4, 4))
    rows = compare(model, data, {"std_k": 2.}, torch.device("cpu"), [.1],
                   ["linf", "l2"], ["random", "fgsm", "pgd"], random_repeats=2)
    assert len(rows) == 8
    assert all(r["clean_rmse_k"] == pytest.approx(2) for r in rows)
    assert all(r["rms_input_change_k"] == pytest.approx(.1, abs=1e-6) for r in rows)
    assert any(r["method"] == "one_step_l2" for r in rows)
    assert not model.weight.requires_grad
    for norm in ("linf", "l2"):
        subset = [r for r in rows if r["norm"] == norm]
        directed = [r["changed_rmse_k"] for r in subset if r["method"] != "random"]
        random = [r["changed_rmse_k"] for r in subset if r["method"] == "random"]
        assert min(directed) > max(random)


def test_calendar_splits_stay_fixed_when_training_window_changes(tmp_path):
    ds = fixture_dataset()
    options = dict(time="valid_time", validation_start="2020-01-15", test_start="2020-01-23")
    save_prepared(ds, tmp_path / "full.npz", **options)
    save_prepared(ds, tmp_path / "short.npz", train_end="2020-01-08", **options)
    full, full_stats, _ = load_prepared(tmp_path / "full.npz")
    short, short_stats, _ = load_prepared(tmp_path / "short.npz")
    assert len(full["train"]) > len(short["train"])
    assert full_stats != short_stats
    with np.load(tmp_path / "full.npz") as a, np.load(tmp_path / "short.npz") as b:
        for name in ("validation", "test"):
            assert np.array_equal(a[name + "_input_times"], b[name + "_input_times"])
            assert np.array_equal(a[name + "_target_times"], b[name + "_target_times"])
            # Normalized values differ, but their physical weather values match.
            assert torch.allclose(denormalize(full[name].tensors[0], full_stats),
                                  denormalize(short[name].tensors[0], short_stats), atol=1e-4)


def test_reject_incomplete_calendar_boundaries():
    times = np.arange(np.datetime64("2020-01-01"), np.datetime64("2020-01-10"), np.timedelta64(6, "h"))
    values = np.ones((len(times), 1, 2, 2))
    with pytest.raises(ValueError, match="both"):
        chronological_pairs(values, times, validation_start="2020-01-05")
