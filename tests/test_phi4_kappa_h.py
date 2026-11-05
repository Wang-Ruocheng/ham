"""Tiny CPU tests only; no formal experiment or learned inference."""
import itertools
import json
import math
from pathlib import Path
import numpy as np
import pytest
from recurrent_rg.data import phi4 as base, phi4_hybrid as hybrid, phi4_hopping as old
from recurrent_rg.data import phi4_kappa_h as kh,phi4_kappa_h_pipeline as data
from experiments import phi4_kappa_h_cnn as spec

@pytest.mark.parametrize('dimension',[1,2,3])
@pytest.mark.parametrize('h',[-.07,0.,.07])
def test_action_mapping_force_and_symmetry(dimension,h):
    phi=np.random.default_rng(7).normal(scale=.4,size=(3,)*dimension)
    kappa,lam=.68,1.2
    mapping=kh.gradient_parameters(kappa,lam,dimension,h)
    val,grad=kh.action_and_gradient(phi,kappa,lam,h)
    value,force=base.action_and_gradient(np.sqrt(kappa)*phi,mapping['r'],mapping['u'],mapping['h'])
    assert val==pytest.approx(value+lam*phi.size,abs=1e-12)
    np.testing.assert_allclose(grad,np.sqrt(kappa)*force,atol=1e-13)
    assert kh.action_and_gradient(-phi,kappa,lam,-h)[0]==pytest.approx(val)
    eps=1e-6
    for index in np.ndindex(phi.shape):
        a,b=phi.copy(),phi.copy();a[index]+=eps;b[index]-=eps
        fd=(kh.action_and_gradient(a,kappa,lam,h)[0]-kh.action_and_gradient(b,kappa,lam,h)[0])/(2*eps)
        assert grad[index]==pytest.approx(fd,abs=2e-8)

class FixedRNG:
    def __init__(self,seed,values):self.seed=seed;self.values=iter(values)
    def integers(self,high):return self.seed
    def random(self):return next(self.values)

@pytest.mark.parametrize('h',[-.21,0.,.21])
def test_actual_corrected_cluster_kernel_detailed_balance(h):
    amplitude=np.array([.35,.8,1.1]);edges=((0,1),(1,2),(2,0))
    bond=[1-math.exp(-2*amplitude[a]*amplitude[b]) for a,b in edges]
    # Common random partition includes every possible cluster field-MH threshold.
    acceptance=[math.exp(min(0.,-2*h*sum(amplitude[i]*sign for i in range(3) if mask&(1<<i))))
                for mask in range(1,8) for sign in (-1,1)]
    cuts=sorted(set([0.,1.,*bond,*acceptance]));intervals=[((a+b)/2,b-a) for a,b in zip(cuts[:-1],cuts[1:])]
    # Enumerate zero-field growth draws exactly, then use two-point quadrature
    # around that proposal's own MH threshold for the final actual RNG draw.
    growthcuts=sorted([0.,*bond,1.]);growth=[((a+b)/2,b-a) for a,b in zip(growthcuts[:-1],growthcuts[1:])]
    transition=np.zeros((8,8));weight=[]
    for state in range(8):
        field=amplitude*np.array([1 if state&(1<<i) else -1 for i in range(3)])
        weight.append(math.exp(sum(field[a]*field[b] for a,b in edges)+h*field.sum()))
        for seed in range(3):
            for draws in itertools.product(growth,repeat=3):
                proposed,receipt=hybrid.cluster_step(field,FixedRNG(seed,[d[0] for d in draws]))
                threshold=math.exp(min(0.,h*np.sum(proposed-field)))
                last=[(threshold/2,threshold),((1+threshold)/2,1-threshold)]
                for x,p in last:
                    if p==0:continue
                    # Number of consumed growth draws comes from the real kernel.
                    vals=[d[0] for d in draws][:receipt['bond_tests']]+[x]
                    actual,meta=kh.field_cluster_step(field,h,FixedRNG(seed,vals))
                    target=sum(int(actual[i]>0)<<i for i in range(3))
                    transition[state,target]+=math.prod(d[1] for d in draws)*p/3
                    np.testing.assert_array_equal(np.abs(actual),amplitude)
    weight=np.array(weight);weight/=weight.sum()
    np.testing.assert_allclose(transition.sum(1),1,atol=3e-14)
    flux=weight[:,None]*transition
    np.testing.assert_allclose(flux,flux.T,atol=3e-14)
    np.testing.assert_allclose(weight@transition,weight,atol=3e-14)

def test_zero_field_replays_historical_and_nonzero_keeps_rejections():
    opts=dict(size=3,dimension=1,samples=8,burn_in=4,spacing=2,step_size=.04,leapfrog_steps=3,seed=971)
    a=old.sample_chain(.68,1.,**opts);b=kh.sample_chain(.68,1.,**opts)
    for key in historical_arrays():np.testing.assert_array_equal(a[key],b[key])
    c=kh.sample_chain(.68,1.,h=.02,**opts)
    assert np.isfinite(c['configurations']).all() and c['hmc_trajectories']==20
    assert not c['symmetry_augmentation']
    for field,raw in zip(c['configurations'][...,0],c['raw_observables']):
        assert raw[0]==pytest.approx(kh.action_and_gradient(field,.68,1.,.02)[0]/field.size)
        assert raw[1]==pytest.approx(field.mean())
    with pytest.raises(ValueError):old.sample_chain(.68,1.,h=.02,**opts)

def historical_arrays():return ('configurations','raw_observables','accepted','delta_h','cluster_sizes')

def test_full_plan_and_preset_no_label_loss():
    p=data.generation_plan(spec.physical_options())
    assert p['conditions']==45 and p['configurations']==23040
    assert p['total_hmc_trajectories']==368640 and p['total_force_site_evaluations']==11943936000
    assert len(p['train_cell_indices'])==34 and len(p['test_cell_indices'])==11
    config=spec.preset('a'*64)
    from experiments.phi4_ru_cnn_nonlinear import preset
    other=preset()
    for key in ('recipe','data_root','data_manifest_sha256','physical_options'):
        config.pop(key);other.pop(key)
    assert config==other
    assert config['loss']['closure_scale_weights'] is None

def test_tiny_generation_loader_and_tamper(tmp_path):
    options=spec.physical_options();options.update(sizes=[9],kappa_values=[.56,.80],h_values=[-.01,0,.01],
        samples=4,burn_in=1,spacing=1,leapfrog_steps=2)
    root=tmp_path/'new';m=data.generate(root,options)
    ds=data.load_dataset(root)
    assert ds.configurations.shape==(24,4,9,9,1)
    assert len(ds.test_indices)==8 and ds.labels.shape==(24,3)
    assert ds.labels[0].tolist()==[.56,-.01,1.]
    with pytest.raises(FileExistsError):data.generate(root,options)
    side=root/m['shards'][0]['sidecar'];side.write_text('{}')
    with pytest.raises(ValueError):data.load_dataset(root)

def test_standard_cli_and_strict_config(tmp_path,monkeypatch):
    from recurrent_rg.cli import main
    from experiments import phi4_ru_cnn_qcr as trainer
    path=tmp_path/'config.json';config=spec.preset('b'*64);path.write_text(json.dumps(config))
    assert trainer.validate_config(path)==config
    calls=[];monkeypatch.setattr(trainer,'run',lambda *a:calls.append(a))
    main(['train','--config',str(path),'--output',str(tmp_path/'run')]);assert len(calls)==1
    config['steps']=3000;path.write_text(json.dumps(config))
    with pytest.raises(ValueError):trainer.validate_config(path)

def test_runner_plan(tmp_path):
    from scripts import run_phi4_kappa_h_cnn as runner
    plan=runner.build_plan(tmp_path/'new',tmp_path)
    assert set(plan['pilot']['cells'])<=set(plan['physical']['train_cell_indices'])
    assert plan['pilot']['expected_chains']==18
    assert plan['training_template']['steps']==6000
    assert plan['budgets']['cpu_seconds_total']==14400

@pytest.mark.parametrize('passed',[True,False])
def test_runner_no_restart_and_gated_sequence(tmp_path,monkeypatch,passed):
    from scripts import run_phi4_kappa_h_cnn as runner
    root=tmp_path/'fresh';monkeypatch.setattr(spec,'ROOT',str(root))
    monkeypatch.setattr(spec,'SNAPSHOT',str(Path(runner.__file__).resolve().parents[1]))
    monkeypatch.setattr(runner,'check_host',lambda:None)
    calls=[]
    def child(root,job,name,command,budget,gpu=False):
        calls.append(name);job['stages'].append(dict(name=name,status='complete',returncode=0))
        if name=='sampler_preflight':runner.save(root/'sampler_preflight.json',dict(budget_passed=passed))
        elif name=='physical_sampling':
            (root/'data').mkdir();runner.save(root/'data/manifest.json',dict(options=spec.physical_options()))
            np.savez(root/'data/private.npz',configurations=np.zeros(1))
        else:
            assert gpu
            (root/'run').mkdir();runner.save(root/'run/summary.json',dict(actual_steps=6000))
            runner.save(root/'run/status.json',dict(status='complete',workflow_complete=True,returncode=0,
                actual_steps=6000,public_files={'summary.json':runner.sha(root/'run/summary.json')}))
    monkeypatch.setattr(runner,'run_child',child)
    monkeypatch.setattr(data,'sampling_validity',lambda *a:dict(status='complete',warning_count=3))
    assert runner.execute(root)==(0 if passed else 3)
    job=runner.read(root/'job.json')
    assert job['status']==('complete' if passed else 'budget_hold')
    if passed:
        assert calls==['sampler_preflight','physical_sampling','training_and_report']
        assert 'data/private.npz' not in job['public_files']
    else:
        assert calls==['sampler_preflight'] and not (root/'data').exists()
    with pytest.raises(FileExistsError):runner.execute(root)
