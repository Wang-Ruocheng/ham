import json
from itertools import product
import numpy as np
import pytest
import torch
from recurrent_rg.data.xy import sample_xy,observables
from recurrent_rg.xy_qcr import XYQCR,objectives,independent_step,circular_bin_probabilities,wrap,circular_features
from experiments.xy_qcr import preset,optimizers,validate_config


@pytest.fixture(autouse=True)
def one_thread():torch.set_num_threads(1)


def test_sampling_limits_reproducibility_and_continuity():
    x=sample_xy(0.,6,1500,91,burn=100,spacing=3)
    assert np.array_equal(x,sample_xy(0.,6,1500,91,burn=100,spacing=3))
    o=observables(x,0.)
    assert abs(o['energy'].mean())<.025
    assert abs(o['magnetization_squared'].mean()-1/36)<.004
    assert len(np.unique(x))>10000
    cold=sample_xy(8.,6,128,92,burn=1024,spacing=8)
    assert observables(cold,8.)['energy'].mean() < -1.9


@pytest.mark.parametrize('K',[.4,1.2])
def test_sampling_against_continuous_2x2_quadrature(K):
    # Fix one angle using exact global rotation invariance; periodic L2 has parallel bonds.
    v=np.array(list(product(np.arange(32)*(2*np.pi/32),repeat=3)))
    states=np.column_stack([np.zeros(len(v)),v]).reshape(-1,2,2)
    o=observables(states,K);logw=-4*K*o['energy'];w=np.exp(logw-logw.max());w/=w.sum()
    sampled=observables(sample_xy(K,2,24000,290+int(K*10),burn=1024,spacing=3),K)
    for n in ('energy','magnetization_squared'):
        assert abs(sampled[n].mean()-np.dot(w,o[n]))<.025


def test_observable_rotation_and_uniform_limits():
    theta=np.random.default_rng(33).uniform(-np.pi,np.pi,(8,9,9))
    a=observables(theta,1.);b=observables(theta+.7,1.)
    for k in a:np.testing.assert_allclose(a[k],b[k],atol=1e-12)
    z=observables(np.zeros((8,9,9)),1.)
    assert np.all(z['energy']==-2) and np.all(z['vortex_density']==0)
    assert np.all(z['helicity_raw']==1)


def test_periodic_model_and_invariant_Q_equivariant_C():
    model=XYQCR();x=torch.randn(2,4,18,18,1,dtype=torch.double)
    with torch.no_grad():
        torch.testing.assert_close(model.Q(x),model.Q(x+.7),atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(model.Q(x),model.Q(x+2*np.pi),atol=1e-12,rtol=1e-12)
        a=model.C(x,torch.Generator().manual_seed(9));b=model.C(x+.7,torch.Generator().manual_seed(9))
        torch.testing.assert_close(wrap(b-a-.7),torch.zeros_like(a),atol=1e-12,rtol=0)
        p=circular_features(x);torch.testing.assert_close(p.sum(-1),torch.ones(2,dtype=torch.double))
        torch.testing.assert_close(p,circular_features(x+.7),atol=1e-12,rtol=1e-12)


def test_wrapped_probability_normalization_periodicity_gradient():
    m=torch.linspace(-2*np.pi,2*np.pi,61,dtype=torch.double,requires_grad=True)
    p=circular_bin_probabilities(m)
    torch.testing.assert_close(p.sum(-1),torch.ones_like(m));assert (p>=0).all()
    torch.testing.assert_close(p,circular_bin_probabilities(m+2*np.pi),atol=1e-13,rtol=1e-10)
    g=torch.autograd.grad((p*torch.arange(16)).sum(),m)[0];assert torch.isfinite(g).all() and g.abs().max()>0


def test_independent_gradients_all_start_and_actual_updates():
    model=XYQCR();x=torch.randn(2,4,18,18,1,dtype=torch.double)
    q,c,r,m=objectives(model,x,torch.Generator().manual_seed(7),information_pairs=8)
    for loss,other in [(q,('C','R')),(c,('Q','R')),(r,('Q','C'))]:
        params=[p for n in other for p in getattr(model,n).parameters()]
        assert all(z is None for z in torch.autograd.grad(loss,params,allow_unused=True,retain_graph=True))
    assert float(r)==pytest.approx(5*m['closure_start_0_target_1']+2.5*(m['closure_start_0_target_2']+m['closure_start_1_target_2']))
    opts=optimizers(model)
    for _ in range(2):independent_step(model,x,opts,torch.Generator().manual_seed(9),information_pairs=8)
    assert all(int(v['step'])==2 for o in opts.values() for v in o.state.values())


def test_fresh_cli_recipe_and_strict_config(tmp_path,monkeypatch):
    from recurrent_rg.cli import main
    import experiments.xy_qcr as trial
    p=tmp_path/'config.json';p.write_text(json.dumps(preset()));assert validate_config(p)==preset()
    calls=[];monkeypatch.setattr(trial,'run',lambda *x:calls.append(x))
    main(['train','--config',str(p),'--output',str(tmp_path/'out')]);assert len(calls)==1
    with pytest.raises(SystemExit):main(['train','--config',str(p),'--output','unused','--allow-legacy-training'])


def test_array_report_and_BKT_claim_limits(tmp_path):
    from scripts.report_xy_qcr import render
    from experiments.phi4_hopping_qcr import closure_metrics
    T=np.array(preset()['temperatures']);temp=np.repeat(T,4);direct=np.column_stack([temp,temp*.95,temp*.9])[...,None]
    a=dict(direct_g=direct,direct_full_g=direct,one_step_g=direct[:,1:]+.01,rollout_g=direct,
        all_start_g=np.stack([direct[:,1],direct[:,2],direct[:,2]],1),all_start_pairs=np.array([[0,1],[0,2],[1,2]]),
        labels=np.column_stack([temp,1/temp]),cell_ids=np.repeat(np.arange(15),4),
        splits=np.repeat(np.where(np.arange(15)%3==1,'test','train'),4),
        flow_grid=np.linspace(.4,1.7,100)[:,None],flow_next=np.linspace(.4,1.7,100)[:,None]*.99)
    np.savez(tmp_path/'evaluation_arrays.npz',**a)
    rows=[dict(temperature=float(t),diagnostics=dict(helicity_modulus_over_J=.5,vortex_density=dict(mean=.1),correlation_mean=(np.arange(1,28)**-.3).tolist())) for t in temp]
    (tmp_path/'data').mkdir()
    values={'summary.json':dict(closure=closure_metrics(a),information_diagnostic={}),
        'roots.json':dict(roots=[]),'history.json':[dict(step=1250,q_total=.1,c_total=.1,r_total=.1)],
        'data/metadata.json':dict(chains=rows,warnings=[])}
    for name,v in values.items():(tmp_path/name).write_text(json.dumps(v))
    d=render(tmp_path)
    assert d['BKT_certified'] is False and d['critical_temperature_estimate'] is None and d['thermal_nu'] is None
    assert (tmp_path/'report/physical_diagnostics.png').is_file()
