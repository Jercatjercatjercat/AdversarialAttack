"""Constraints, iteration semantics and validation-to-test handoff, using fake maps."""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from config import ROOT
from models.residual_cnn import ResidualCNN
from perturbations import project,change_input,forecast_loss
from data.prepare_era5 import save_prepared
from data.era5 import load_data


def frozen_model():
    model=ResidualCNN(width=4)
    for p in model.parameters(): p.requires_grad_(False)
    return model


def test_l1_soft_projection_and_l0_count():
    delta=torch.tensor([[[[3.,-1.,.5,0.]]],[[[.1,-.2,.3,0.]]]])
    result=project(delta,2.,'l1')
    torch.testing.assert_close(result[0],torch.tensor([[[2.,0.,0.,0.]]]))
    torch.testing.assert_close(result[1],delta[1])
    assert torch.all(result.flatten(1).abs().sum(1)<=2.+1e-6)
    sparse=project(delta,.4,'l0',k=2)
    assert torch.all((sparse!=0).flatten(1).sum(1)<=2)
    assert sparse.abs().max()<=.4


@pytest.mark.parametrize('norm,method',[('l0','top_k'),('l1','pgd'),('l2','pgd'),('linf','pgd'),('linf','mi_fgsm')])
def test_iterative_bounds_and_unchanged_channels(norm,method):
    torch.manual_seed(10)
    model=frozen_model()
    x,y=torch.randn(2,14,4,4),torch.randn(2,5,4,4)
    before={n:p.clone() for n,p in model.named_parameters()}
    changed=change_input(model,x,y,5,.4,method,norm,steps=20,step_size=.1,
                         restarts=3,k=2,cell_cap=.25,objective_index=0,
                         generator=torch.Generator().manual_seed(5))
    assert torch.equal(x[:,:5],changed[:,:5]) and torch.equal(x[:,6:],changed[:,6:])
    delta=changed[:,5]-x[:,5]
    assert delta.abs().max()<=.25+1e-6
    if norm=='l0': assert torch.all((delta!=0).flatten(1).sum(1)<=2)
    if norm=='l1': assert torch.all(delta.flatten(1).abs().sum(1)<=.4+1e-6)
    if norm=='l2': assert torch.all(delta.flatten(1).norm(dim=1)<=.4+1e-6)
    assert torch.all(forecast_loss(model(changed),y,0)>=forecast_loss(model(x),y,0)-1e-6)
    assert all(torch.equal(before[n],p) and p.grad is None for n,p in model.named_parameters())


def test_fgsm_one_update_and_zero_momentum_equivalence():
    model=frozen_model()
    x,y=torch.randn(2,14,4,4),torch.randn(2,5,4,4)
    one=change_input(model,x,y,5,.3,'one_step',steps=1)
    many=change_input(model,x,y,5,.3,'one_step',steps=40,restarts=3,step_size=.001)
    assert torch.equal(one,many)
    pgd=change_input(model,x,y,5,.3,'pgd',steps=5,step_size=.06)
    momentum=change_input(model,x,y,5,.3,'mi_fgsm',steps=5,step_size=.06,momentum=0)
    torch.testing.assert_close(pgd,momentum)


def test_restarts_escape_zero_clean_gradient_and_zero_budget():
    model=frozen_model()
    x,y=torch.zeros(2,14,4,4),torch.zeros(2,5,4,4)
    clean_start=change_input(model,x,y,5,.3,'pgd',steps=3,restarts=1)
    restarted=change_input(model,x,y,5,.3,'pgd',steps=3,restarts=3,generator=torch.Generator().manual_seed(10))
    assert torch.equal(clean_start,x)
    assert forecast_loss(model(restarted),y).min()>0
    for norm,method in [('linf','mi_fgsm'),('l0','top_k'),('l1','pgd')]:
        result=change_input(model,x,y,5,.3,method,norm,steps=5,k=2)
        assert torch.isfinite(result).all()
        assert torch.equal(change_input(model,x,y,5,0.,method,norm,k=2),x)


def test_validation_selection_and_frozen_test_handoff(tmp_path):
    # Real-looking units but FABRICATED numbers; never scientific evidence.
    times=np.concatenate([np.arange(np.datetime64(f'{year}-01-01T00'),np.datetime64(f'{year}-01-02T00'),np.timedelta64(6,'h')) for year in (2020,2021,2022)])
    fields=np.arange(12*5*4*4,dtype='float32').reshape(12,5,4,4)*.001
    fields+=np.array([280.,1.,2.,1010.,4.],dtype='float32')[None,:,None,None]
    data=tmp_path/'fixture.npz'
    save_prepared(fields,times,np.linspace(45,60,4),np.linspace(-10,10,4),data,'FABRICATED test fixture',
                  ['2020-01-01','2021-01-01','2022-01-01','2023-01-01'])
    datasets,stats,metadata,_,_=load_data(data)
    model=frozen_model()
    run=tmp_path/'run';run.mkdir()
    torch.save(dict(model_state=model.state_dict(),model_settings=model.settings,stats=stats,metadata=metadata),run/'best_model.pt')
    tuning=tmp_path/'tuning'
    common=['--data',str(data),'--checkpoint-dir',str(run)]
    subprocess.run([sys.executable,str(ROOT/'tune_attacks.py'),*common,'--results-dir',str(tuning),
                    '--steps','1','2','--step-factors','1','--restarts','1','--norm','linf','l0',
                    '--k','2','--max-samples','2','--random-trials','1'],cwd=ROOT,check=True,capture_output=True,text=True)
    selected=json.loads((tuning/'selected_settings.json').read_text())
    assert selected['split']=='validation' and selected['validation_samples']==2
    scores=json.loads((tuning/'results.json').read_text())
    for c in selected['conditions']:
        if c['method'] in ('pgd','mi_fgsm','top_k'):
            same=[r for r in scores if all(r[field]==c[field] for field in ('method','norm','budget','k'))]
            chosen=[r for r in same if r['steps']==c['steps'] and r['step_factor']==c['step_factor']]
            baseline=same[0]['clean_objective_mse']
            threshold=baseline+.99*(max(r['objective_mse'] for r in same)-baseline)
            assert chosen[0]['objective_mse']>=threshold-1e-12
            assert chosen[0]['gradient_updates_per_case']==min(r['gradient_updates_per_case'] for r in same if r['objective_mse']>=threshold-1e-12)
    output=tmp_path/'test'
    subprocess.run([sys.executable,str(ROOT/'compare_attacks.py'),*common,'--settings',str(tuning/'selected_settings.json'),
                    '--results-dir',str(output)],cwd=ROOT,check=True,capture_output=True,text=True)
    config=json.loads((output/'comparison_config.json').read_text())
    assert config['split']=='test' and config['conditions']==selected['conditions']
    assert config['objective']=='t2m'
    # Reject settings selected on test, instead of silently blessing leakage.
    selected['split']='test';invalid=tmp_path/'invalid.json';invalid.write_text(json.dumps(selected))
    rejected=subprocess.run([sys.executable,str(ROOT/'compare_attacks.py'),*common,'--settings',str(invalid),
                    '--results-dir',str(tmp_path/'rejected')],cwd=ROOT,capture_output=True,text=True)
    assert rejected.returncode!=0 and 'validation' in rejected.stderr


def test_step_selection_uses_attack_gain_and_prefers_less_effort():
    from attack_evaluation import choose_condition
    rows=[dict(clean_objective_mse=1.,objective_mse=score,gradient_updates_per_case=steps,
               steps=steps,restarts=1,step_factor=1.) for score,steps in [(1.1,1),(1.995,10),(2.,40)]]
    assert choose_condition(rows,.99)['steps']==10
    assert choose_condition(rows,1.)['steps']==40
