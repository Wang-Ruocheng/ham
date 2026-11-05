import copy
import json
import numpy as np
import pytest
import torch
from scripts import run_phi4_hopping_cnn_R as run
from scripts import report_phi4_hopping_cnn_R as report


def test_fresh_R_reproducible_and_isolated():
    before=torch.random.get_rng_state().clone();a=run.make_r();b=run.make_r()
    assert torch.equal(before,torch.random.get_rng_state())
    assert run.common.state_sha(a.state_dict())==run.common.state_sha(b.state_dict())


def test_all_start_detachment_and_clock():
    q=torch.linspace(-.2,.5,160,dtype=torch.float64).reshape(40,4,1).requires_grad_();r=run.make_r();opt=torch.optim.Adam(r.parameters(),lr=.001)
    rng=torch.Generator().manual_seed(91)
    for _ in range(3):run.update(r,opt,q,rng)
    assert q.grad is None and all(p.grad is None for p in r.parameters())
    assert run.common.optimizer_counts(opt,r)==dict(min=3,max=3,parameters=6)
    weights=run.common.pair_weights('A_equal_target')
    assert weights.sum()==pytest.approx(10)
    for t in range(1,4):assert sum(w for (_,target),w in zip(run.PAIRS,weights) if target==t)==pytest.approx(10/3)


def test_signed_stability_uses_full_map_and_input_support():
    class Cubic(torch.nn.Module):
        def forward(self,g):return -g*(g-1)*(g+1)
    r=Cubic();g=np.zeros((60,4,1));g[:,:3,0]=np.linspace(-.8,.8,60)[:,None];g[:,3,0]=2
    roots=run.find_roots(r,np.linspace(-1.3,1.3,4001),g,np.ones(60,dtype=bool))['roots']
    assert [x['g'] for x in roots]==pytest.approx([-1,0,1],abs=1e-10)
    assert [x['full_R_derivative'] for x in roots]==pytest.approx([-1,2,-1],abs=1e-9)
    assert [x['inside_train_input_envelope'] for x in roots]==[False,True,False]
    assert roots[1]['numerical_stability']=='repeller'
    linear=torch.nn.Linear(1,1,bias=False).double()
    with torch.no_grad():linear.weight.fill_(-1.5)
    roots=run.find_roots(linear,np.linspace(-1,1,401),g,np.ones(60,dtype=bool))['roots']
    assert roots[0]['full_R_derivative']==pytest.approx(-.5) and roots[0]['numerical_stability']=='attractor'


def test_recurrence_gradients_not_truncated():
    r=torch.nn.Linear(1,1,bias=False).double()
    with torch.no_grad():r.weight.fill_(.5)
    g=torch.ones(1,4,1,dtype=torch.float64)
    run.common.predict_pairs(r,g)[:,3].sum().backward()
    assert r.weight.grad.item()==pytest.approx(3*1.5**2)


def test_posthoc_mapping_no_extrapolation_and_nonmonotone_rejected():
    k=np.repeat(np.linspace(.3,1,15),4);g=np.stack([k,k**2,-k,np.sin(k*10)],axis=1)[...,None]
    a=dict(direct_g=g,labels=np.stack([k,np.ones(60),np.zeros(60)],axis=1),cell_ids=np.repeat(range(15),4),splits=np.repeat(['train','train','test']*5,4))
    roots={'roots':[{'g':.5},{'g':2}]};rows=report.root_mappings(roots,a)
    assert rows[0]['posthoc_coordinate_mappings'][0]['posthoc_interpolated_kappa']==pytest.approx(.5)
    assert rows[0]['posthoc_coordinate_mappings'][3]['posthoc_interpolated_kappa'] is None
    assert all(x['posthoc_interpolated_kappa'] is None for x in rows[1]['posthoc_coordinate_mappings'])


def fixture(tmp_path):
    rng=np.random.default_rng(2);g=rng.normal(0,.2,(60,4,1));splits=np.repeat(['train','train','test']*5,4);tr=splits=='train'
    a=dict(direct_g=g,direct_full_g=g.copy(),labels=np.column_stack([np.repeat(np.linspace(.32,1.04,15),4),np.ones(60),np.zeros(60)]),
        splits=splits,cell_ids=np.repeat(range(15),4),all_start_pairs=np.array(run.PAIRS),train_target_std=g[tr].std(0)[:,0],train_target_mean=g[tr].mean(0)[:,0],flow_grid=np.linspace(-1,1,401)[:,None])
    metrics={}
    for n in ('initial','final'):
        pred=np.stack([g[:,t]+.02 for s,t in run.PAIRS],axis=1)
        a[n+'_all_start']=pred;a[n+'_one_step']=pred[:,[0,2,5]];a[n+'_flow_next']=a['flow_grid']*.9
        metrics[n]={s:run.metrics(g[splits==s],pred[splits==s],a['train_target_std'],a['train_target_mean']) for s in ('train','test')}
    np.savez(tmp_path/'evaluation_arrays.npz',**a)
    history=[dict(step=s,**metrics['final']['train']) for s in range(0,3001,100)]
    roots=dict(roots=[dict(g=0.,full_R_derivative=.9,residual_abs=0,numerical_stability='attractor',inside_train_input_envelope=True)])
    summary=dict(status='complete',actual_steps=3000,Q_updates=0,C_updates=0,labels_used_in_training=False,test_used_in_training=False,thermal_nu=None,physical_fixed_points_certified=False)
    for name,v in [('plan',{}),('history',history),('metrics',metrics),('roots',roots),('summary',summary)]: (tmp_path/(name+'.json')).write_text(json.dumps(v))


def test_report_render_preserves_inputs_no_overwrite(tmp_path):
    fixture(tmp_path);pins={n:report.common._sha(tmp_path/n) for n in report.INPUTS};out=report.render_report(tmp_path)
    assert len(out)==5 and all((tmp_path/p).stat().st_size>0 for p in out)
    assert pins=={n:report.common._sha(tmp_path/n) for n in report.INPUTS}
    with pytest.raises(FileExistsError):report.render_report(tmp_path)


@pytest.mark.parametrize('mutation',['metrics','half_steps','Q_changed','labels'])
def test_invalid_report_rejected(tmp_path,mutation):
    fixture(tmp_path);name='metrics' if mutation=='metrics' else 'summary';p=tmp_path/(name+'.json');v=json.loads(p.read_text())
    if mutation=='metrics':v['final']['test']['pairs'][0]['rmse']*=2
    elif mutation=='half_steps':v['actual_steps']=1500
    elif mutation=='Q_changed':v['Q_updates']=1
    else:v['labels_used_in_training']=True
    p.write_text(json.dumps(v))
    with pytest.raises(ValueError):report.load_run(tmp_path)
