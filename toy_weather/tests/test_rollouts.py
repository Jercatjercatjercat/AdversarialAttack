"""Analytic rollout gradients, correct clocks, split boundaries and window attacks."""
import numpy as np
import torch
from torch import nn
from data.era5 import WeatherDataset, pair_indices, clock_features
from rollouts import RolloutDataset, rollout, window_loss
from perturbations import change_input


class AnalyticModel(nn.Module):
    variables, history = 1, 2
    def forward(self, x):
        return 2*x[:,1:2]+.5*x[:,:1]


def test_predicted_history_and_full_trajectory_gradient():
    model = AnalyticModel()
    x = torch.tensor([[[[1.]],[[2.]]]],requires_grad=True)
    clocks = torch.empty(1,3,0,1,1)
    out = rollout(model,x,clocks)
    torch.testing.assert_close(out.flatten(),torch.tensor([4.5,10.,22.25]))
    # Last prediction = 2*(2*(2*current + .5*older)+.5*current)
    #                   + .5*(2*current + .5*older).
    gradient = torch.autograd.grad(out[:,-1].sum(),x)[0]
    torch.testing.assert_close(gradient.flatten(),torch.tensor([2.25,10.]))
    # Verify against an independent finite difference rather than only graph shape.
    epsilon=.001
    changed=x.detach().clone();changed[:,1]+=-epsilon
    lower=rollout(model,changed,clocks)[:,-1]
    changed[:,1]+=2*epsilon
    upper=rollout(model,changed,clocks)[:,-1]
    assert abs(float((upper-lower)/(2*epsilon))-10.)<.003


def test_window_attack_uses_custom_loss_and_only_initial_channel():
    model = AnalyticModel()
    x = torch.tensor([[[[1.]],[[2.]]]])
    clocks = torch.empty(1,3,0,1,1)
    targets = torch.zeros(1,3,1,1,1)
    def objective(candidate):
        return window_loss(model,candidate,targets,clocks,3)
    changed=change_input(model,x,targets,1,.2,'pgd',steps=4,step_size=.1,loss_fn=objective)
    assert torch.equal(changed[:,:1],x[:,:1])
    assert (changed-x).abs().max()<=.200001
    assert objective(changed)>objective(x)
    torch.testing.assert_close(changed[:,1],x[:,1]+.2)
    assert x.grad is None


def test_rollout_dataset_stays_in_split_and_advances_calendar():
    times=np.arange(np.datetime64('2021-12-31T00'),np.datetime64('2022-01-03T00'),np.timedelta64(6,'h')).astype('datetime64[ns]')
    fields=np.arange(len(times),dtype='float32')[:,None,None,None]
    stats=dict(mean=[0.],std=[1.])
    base=WeatherDataset(fields,times,pair_indices(times,'2021-01-01','2022-01-01'),stats)
    data=RolloutDataset(base,2)
    assert len(data)==1
    assert str(data.times[0]).startswith('2021-12-31T06')
    x,y,clocks=data[0]
    assert y[:,0,0,0].tolist()==[2.,3.]
    np.testing.assert_allclose(clocks[1],clock_features(np.datetime64('2021-12-31T12'),1,1))
    assert x.shape==(6,1,1) and clocks.shape==(2,4,1,1)


def test_rollout_skips_missing_future_and_duration_is_descriptive():
    from rollout_evaluation import validity_summary
    times=np.array(['2022-01-01T00','2022-01-01T06','2022-01-01T12','2022-01-02T00','2022-01-02T06','2022-01-02T12'],dtype='datetime64[ns]')
    fields=np.zeros((len(times),1,1,1),dtype='float32')
    base=WeatherDataset(fields,times,pair_indices(times,'2022-01-01','2023-01-01'),dict(mean=[0.],std=[1.]),time_features=False)
    data=RolloutDataset(base,1)
    assert len(data)==2
    import pytest
    with pytest.raises(ValueError,match='No complete'):
        RolloutDataset(base,2)
    clean=np.ones((2,4,1))
    attacked=np.broadcast_to(np.array([1.2,1.,1.3,1.4])[None,:,None]**2,(2,4,1))
    result=validity_summary(attacked,clean,np.array([6,12,18,24]),0,.1)
    assert result['continuous_from_start_hours']==6
    assert result['last_lead_above_hours']==24
    assert result['leads_above_threshold']==3


def test_different_windows_can_choose_opposite_initial_changes():
    model=AnalyticModel()
    x=torch.tensor([[[[1.]],[[2.]]]])
    clocks=torch.empty(1,2,0,1,1)
    # First clean prediction is4.5, slightly BELOW its reference5; the second
    # prediction is10, far ABOVE its reference0. Their objectives disagree.
    target=torch.tensor([[[[[5.]]],[[[0.]]]]])
    results=[]
    for k in (1,2):
        def objective(candidate):
            return window_loss(model,candidate,target,clocks,k)
        results.append(change_input(model,x,target,1,.1,'one_step',loss_fn=objective))
    assert results[0][0,1,0,0] < x[0,1,0,0]
    assert results[1][0,1,0,0] > x[0,1,0,0]
