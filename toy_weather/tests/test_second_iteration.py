"""Fabricated fixtures test correctness; they are not scientific ERA5 results."""
import json
import numpy as np
import pytest
import torch
import xarray as xr
from torch.utils.data import DataLoader, TensorDataset
from config import SOURCE_NAMES, VARIABLES, UNITS
from data.prepare_era5 import prepare_fields, save_prepared
from data.era5 import pair_indices, training_statistics, load_data, denormalize, clock_features
from models.residual_cnn import ResidualCNN
from metrics import evaluate_loader
from perturbations import change_input


def fixture_fields():
    """Three four-state blocks, one per split, with deliberately different means."""
    times = np.concatenate([np.arange(np.datetime64(f'{year}-01-01T00'),
                           np.datetime64(f'{year}-01-02T00'), np.timedelta64(6, 'h'))
                            for year in (2020, 2021, 2022)])
    fields = np.arange(12*5*4*4, dtype='float32').reshape(12, 5, 4, 4)
    return fields, times


def prepared_file(tmp_path):
    fields, times = fixture_fields()
    path = tmp_path/'fields.npz'
    save_prepared(fields, times, np.linspace(45, 60, 4), np.linspace(-10, 10, 4),
                  path, 'fabricated test fixture, NOT ERA5',
                  ['2020-01-01', '2021-01-01', '2022-01-01', '2023-01-01'])
    return path


def test_exact_pairing_and_disjoint_states():
    _, times = fixture_fields()
    triples = [pair_indices(times, f'{y}-01-01', f'{y+1}-01-01') for y in (2020, 2021, 2022)]
    assert [len(v) for v in triples] == [2, 2, 2]
    for rows in triples:
        assert np.all(np.diff(times[rows], axis=1) == np.timedelta64(6, 'h'))
    for i in range(2):
        assert set(triples[i].ravel()).isdisjoint(set(triples[i+1].ravel()))
    # A missing intermediate state is never treated as a six-hour observation.
    with pytest.raises(ValueError, match='No complete'):
        pair_indices(times[[0, 2, 3]], '2020-01-01', '2021-01-01')


def test_normalization_training_inputs_only(tmp_path):
    path = prepared_file(tmp_path)
    datasets, stats, meta, _, _ = load_data(path)
    fields, times = fixture_fields()
    expected = fields[:3].astype('float64')  # Final training target is NOT an input.
    np.testing.assert_allclose(stats['mean'], expected.mean((0, 2, 3)))
    x, y = datasets['train'][0]
    assert x.shape == (14, 4, 4) and y.shape == (5, 4, 4)
    np.testing.assert_allclose(denormalize(x[:5], stats), fields[0], atol=1e-5)
    np.testing.assert_allclose(denormalize(x[5:10], stats), fields[1], atol=1e-5)
    np.testing.assert_allclose(denormalize(y, stats), fields[2], atol=1e-5)
    fields[4:] += 1e6  # Alter ONLY validation/test, then refit.
    rows = pair_indices(times, '2020-01-01', '2021-01-01')
    assert training_statistics(fields, rows) == stats
    assert meta['counts'] == {'train': 2, 'validation': 2, 'test': 2}


def test_variable_selection_and_single_history(tmp_path):
    datasets, stats, meta, _, _ = load_data(prepared_file(tmp_path), ['u10', 't2m'], history=1, time_features=False)
    assert datasets['train'][0][0].shape == (2, 4, 4)
    assert meta['variables'] == ['u10', 't2m']
    assert meta['units'] == ['m/s', 'K']
    assert stats['input_state_count'] == 2  # Only current states are used with history=1.


def test_preprocessing_coordinate_order_and_units():
    times = np.arange(np.datetime64('2020-01-01T00'), np.datetime64('2020-01-02T00'), np.timedelta64(1, 'h'))
    lat, lon = np.array([63., 60., 45., 42.]), np.array([0., 10., 12., 348., 350.])
    variables = {}
    for c, name in enumerate(SOURCE_NAMES):
        values = np.broadcast_to(np.arange(len(times))[:, None, None]+c+1., (len(times), 4, 5)).copy()
        attrs = {'units': ('K', 'm s**-1', 'm s**-1', 'Pa', 'kg kg**-1')[c]}
        dims = ('time', 'latitude', 'longitude')
        if name == 'specific_humidity':
            values, dims = values[:, None], ('time', 'level', 'latitude', 'longitude')
        variables[name] = xr.DataArray(values, dims=dims, attrs=attrs)
    ds = xr.Dataset(variables, coords=dict(time=times, latitude=lat, longitude=lon, level=[850]))
    values, selected, _, _ = prepare_fields(ds, 4)
    assert values.shape == (4, 5, 4, 4)
    np.testing.assert_allclose(values[:, 3, 0, 0], [0.04, .10, .16, .22])
    np.testing.assert_allclose(values[:, 4, 0, 0], [5000., 11000., 17000., 23000.])
    assert np.all(np.diff(selected) == np.timedelta64(6, 'h'))
    ds['2m_temperature'][0, 1, 0] = np.nan
    # This source NaN is inside the crop and must not be silently filled.
    with pytest.raises(ValueError, match='missing'):
        prepare_fields(ds, 4)
    kept, kept_times, _, _ = prepare_fields(ds, 4, missing='skip')
    assert kept.shape[0] == 3
    assert kept_times[0] == np.datetime64('2020-01-01T06')
    assert np.isfinite(kept).all()


def test_clock_features_not_weather_and_leap_year():
    features = clock_features(np.datetime64('2020-02-29T06'), 4, 4)
    assert features.shape == (4, 4, 4) and np.isfinite(features).all()
    assert features[0, 0, 0] == pytest.approx(1.)
    assert features[1, 0, 0] == pytest.approx(0., abs=1e-6)
    assert np.max(np.abs(features)) <= 1


def test_residual_initialization_and_input_gradient():
    torch.manual_seed(42)
    model = ResidualCNN(width=8)
    x, y = torch.randn(2, 14, 4, 4), torch.randn(2, 5, 4, 4)
    assert torch.equal(model(x), x[:, 5:10])  # Exact persistence at initialization.
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    for _ in range(2):
        optimizer.zero_grad()
        torch.nn.functional.mse_loss(model(x), y).backward()
        optimizer.step()
    weights = {name: p.detach().clone() for name, p in model.named_parameters()}
    for p in model.parameters():
        p.requires_grad_(False)
        p.grad = None
    x = x.clone().requires_grad_(True)
    torch.nn.functional.mse_loss(model(x), y).backward()
    assert x.grad.shape == (2, 14, 4, 4)
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
    assert x.grad[:, :5].abs().sum()>0  # Trained correction uses historical weather.
    assert all(torch.equal(weights[n], p) and p.grad is None for n, p in model.named_parameters())


def test_metrics_global_aggregation_uneven_batches():
    model = ResidualCNN(variables=1, history=1, time_features=False, width=4)
    x = torch.zeros(3, 1, 4, 4)
    y = torch.tensor([1., 2., 3.])[:, None, None, None].expand_as(x)
    result = evaluate_loader(model, DataLoader(TensorDataset(x, y), batch_size=2),
                             {'mean': [100.], 'std': [2.]}, ['t2m'], ['K'])
    expected = np.sqrt((1+4+9)/3)*2
    assert result['variables']['t2m']['rmse'] == pytest.approx(expected)
    assert result['variables']['t2m']['persistence_rmse'] == pytest.approx(expected)


@pytest.mark.parametrize('method', ['random', 'one_step', 'pgd'])
@pytest.mark.parametrize('norm', ['linf', 'l2'])
def test_attacks_preserve_history_clocks_and_bounds(method, norm):
    model = ResidualCNN(width=4)
    for p in model.parameters():
        p.requires_grad_(False)
    x, y = torch.randn(2, 14, 4, 4), torch.randn(2, 5, 4, 4)
    changed = change_input(model, x, y, channel=5, radius=.5, method=method, norm=norm, steps=3)
    assert torch.equal(changed[:, :5], x[:, :5])
    assert torch.equal(changed[:, 6:], x[:, 6:])
    delta = changed[:, 5:6]-x[:, 5:6]
    size = delta.flatten(1).abs().max(1).values if norm=='linf' else delta.flatten(1).norm(dim=1)
    assert torch.all(size <= .5+1e-6)
    assert torch.equal(change_input(model, x, y, 5, 0., method, norm), x)
    if method != 'random':
        assert torch.all((model(changed)-y).square().mean((1,2,3)) >= (model(x)-y).square().mean((1,2,3))-1e-6)


def test_local_netcdf_and_zarr_agree(tmp_path):
    """Both documented local file routes preserve the same weather values/units."""
    times = np.arange(np.datetime64('2020-01-01T00'), np.datetime64('2020-01-02T00'), np.timedelta64(6, 'h'))
    coords = dict(time=times, latitude=[42.,45.,60.,63.], longitude=[-12.,-10.,0.,10.,12.], level=[850])
    fields = {}
    for index, name in enumerate(SOURCE_NAMES):
        values = np.arange(4*4*5, dtype='float32').reshape(4,4,5)+index+1
        dims = ('time','latitude','longitude')
        if name == 'specific_humidity':
            values, dims = values[:,None]/100000., ('time','level','latitude','longitude')
        fields[name] = xr.DataArray(values, dims=dims, attrs={'units': ('K','m/s','m/s','Pa','kg/kg')[index]})
    ds = xr.Dataset(fields, coords=coords)
    nc, zarr = tmp_path/'weather.nc', tmp_path/'weather.zarr'
    ds.to_netcdf(nc)
    ds.to_zarr(zarr, consolidated=True)
    with xr.open_dataset(nc) as one, xr.open_zarr(zarr, consolidated=True, chunks=None) as two:
        a, b = prepare_fields(one,4), prepare_fields(two,4)
    for left, right in zip(a,b):
        np.testing.assert_array_equal(left,right)


def test_current_state_skip_is_separate_from_feature_skip():
    """Disabling correction prediction must not quietly change internal architecture."""
    residual = ResidualCNN(width=4, residual=True, feature_skip=True)
    direct = ResidualCNN(width=4, residual=False, feature_skip=True)
    with torch.no_grad():
        residual.output_conv.weight.normal_(0, .01)
    direct.load_state_dict(residual.state_dict())
    x = torch.randn(2,14,4,4)
    torch.testing.assert_close(residual(x)-direct(x), residual.current_state(x))
