import json
from dataclasses import replace
import hashlib
import numpy as np
import pytest
from recurrent_rg.data.phi4_ru_pipeline import Phi4RUOptions, generation_plan, generate_dataset
from recurrent_rg.data.phi4_hybrid_pipeline import parameter_cells, chain_seed, SCHEMA


def test_explicit_work_counts_and_historical_seed():
    o=Phi4RUOptions()
    message=[SCHEMA,o.seed,o.dimension,36,float(o.r_values[2]),float(o.u_values[3]),0.,1]
    assert chain_seed(o,36,13,1)==int.from_bytes(hashlib.sha256(json.dumps(message).encode()).digest()[:16],'little')
    points=tuple(parameter_cells(o))
    b=replace(o,parameter_cells=points)
    for k in ('total_hmc_trajectories','total_force_site_evaluations','configurations'):
        assert generation_plan(b)[k]==generation_plan(o)[k]
    assert chain_seed(b,36,13,1)==chain_seed(o,36,13,1)
    with pytest.raises(ValueError):generation_plan(replace(o,parameter_cells=(points[0],points[0])))
    with pytest.raises(ValueError):generation_plan(replace(o,parameter_cells=((-99.,.5),points[0])))


def test_nonrectangular_data_provenance_loader_and_report(tmp_path):
    import torch
    torch.set_num_threads(1)
    from experiments.phi4_qcr import load_dataset
    from scripts.report_phi4_ru_vae import scale_diagnostics
    pts=((-1.,.5),(-1.5,1.),(-.5,.5),(-1.,1.))
    o=Phi4RUOptions(r_values=(-1.5,-1.,-.5),u_values=(.5,1.),parameter_cells=pts,
        sizes=(36,),samples=4,burn_in=2,spacing=1,leapfrog_steps=2)
    manifest=generate_dataset(tmp_path/'data',o)
    assert len(manifest['shards'])==4 and manifest['total_hmc_trajectories']==96
    ds=load_dataset(tmp_path/'data')
    np.testing.assert_equal(ds.labels[::4,:2],np.array(pts))
    assert len(ds.train_indices)==12 and len(ds.test_indices)==4
    coords=[[-.25,.5],[-.25,1.],[.25,.5],[.25,1.]]
    design=dict(cells=[dict(parameters=list(p),design_coordinates=c) for p,c in zip(pts,coords)])
    rng=np.random.default_rng(91);targets=rng.random((2,16,3,5));targets/=targets.sum(-1,keepdims=True)
    arrays=dict(labels=ds.labels[:,:2],direct_g=rng.normal(size=(16,3,2)),splits=ds.splits)
    result=scale_diagnostics(arrays,dict(targets=targets,decoded=targets.copy()),design)
    assert result[0][0]['main_effect_parameter_names']==['r-r_0(u)','u']
    assert result[0][0]['reconstruction']['train']['sum_bin_mse']==0
    side=tmp_path/'data'/manifest['shards'][0]['sidecar']
    side.write_text(side.read_text()+' ')
    with pytest.raises(ValueError,match='SHA256'):load_dataset(tmp_path/'data')


def test_chain_bootstrap_detects_two_directions_and_collapse():
    from scripts.diagnose_phi4_ru_band import responses
    coords=np.array([[x,y] for x in [-1.,0.,1.] for y in [-1.,0.,1.]])
    rng=np.random.default_rng(1)
    g=coords[:,None,None,:]+rng.normal(scale=.001,size=(9,4,1,2))
    result=responses(g,coords,np.ones(9,dtype=bool),bootstrap=100)
    assert result['summary'][0]['weak_above_twice_noise95']==1
    g[:,:,:,1]=g[:,:,:,0]
    result=responses(g,coords,np.ones(9,dtype=bool),bootstrap=100)
    assert result['summary'][0]['weak_above_twice_noise95']==0
