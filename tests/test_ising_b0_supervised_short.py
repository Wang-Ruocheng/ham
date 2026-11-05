import copy
import json
from itertools import product
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from experiments import ising_b0_supervised_short as run
from experiments import ising_cnn_short_trial as binary
from experiments import ising_kb_cnn_qcr as kb
from experiments.phi4_hopping_qcr import state_fingerprint
from recurrent_rg.empirical_mi import binary_lattice_mi_statistics


def test_scalar_supervision_and_independent_gradients():
    torch.set_num_threads(1);cfg=run.build_config();model=binary.fresh(cfg,'cpu')
    x=torch.randint(2,(2,4,18,18,1)).double()*2-1;y=torch.tensor([[.3],[.6]],dtype=torch.float64)
    scaler=run.label_scaler(y);labels=run.standardized(y.requires_grad_(),scaler,'cpu')
    assert not labels.requires_grad;torch.testing.assert_close(labels.mean(0),torch.zeros(1,dtype=torch.float64))
    hist=binary_lattice_mi_statistics(x,block=3);hist.prepare_selection()
    patterns=torch.tensor(list(product((-1.,1.),repeat=9)),dtype=torch.float64)[None]
    other=copy.deepcopy(model)
    for m,w in [(model,10.),(other,0.)]:
        with binary.isolated_rng('cpu',7):v=run.step(m,x,labels,hist,torch.arange(2),patterns,cfg,kb.optimizers(m),weight=w)
        assert v['q_total']==pytest.approx(v['q_representation']+w*v['q_supervision'])
    assert state_fingerprint(model.theory_encoder)!=state_fingerprint(other.theory_encoder)
    for name in ['C','R']:assert state_fingerprint(binary.modules(model)[name])==state_fingerprint(binary.modules(other)[name])
    assert y.grad is None


def test_timing_only_endpoint_and_cli(tmp_path,monkeypatch):
    p='configs/ising2d_B0_cnn_g1_supervised_10min_seed91.json';assert run.validate_config(p)==run.preset()
    x=run.choose_steps(40,[50,20]+[.3]*8,.1,.04)
    assert x['hard_seconds']==600 and x['conservative_total_seconds']<540
    assert 200<=x['formal_steps']<=1500 and x['formal_steps']%50==0
    with pytest.raises(run.common.BudgetHold):run.choose_steps(530,[.3]*10,.1,.04)
    from recurrent_rg.cli import main
    seen=[];monkeypatch.setattr(run,'run',lambda *a:seen.append(a))
    main(['train','--config',p,'--output','unused']);assert len(seen)==1
    with pytest.raises(SystemExit):main(['train','--config',p,'--output','unused','--resume','x'])


def test_array_report(tmp_path):
    from scripts.report_ising_b0_supervised_short import render
    rng=np.random.default_rng(3);labels=np.repeat(np.linspace(.2,.68,64),3)[:,None]
    splits=np.repeat(np.array(['test' if i%3==1 else 'train' for i in range(64)]),3)
    g=rng.normal(size=(192,3,1));grid=np.linspace(-4,4,501)[:,None]
    a=dict(labels=labels,splits=splits,direct_g=g,direct_full_g=g,one_step_g=g[:,1:],rollout_g=g,
        all_start_g=g[:,[1,2,2]],all_start_pairs=np.array([[0,1],[0,2],[1,2]]),flow_grid=grid,flow_next=grid*.8)
    np.savez(tmp_path/'evaluation_arrays.npz',**a)
    m=dict(recipe=run.RECIPE,config={'training':{'epochs':200}},labels_used_in_training=True,test_used_in_training=False,label_scaler=run.label_scaler(labels[splits=='train']))
    s=dict(actual_steps=200,optimizer_steps={k:dict(min=200,max=200) for k in ['Q','C','R']},closure=binary.closure_metrics(a),supervision={})
    h=[dict(step=200,q_total=.1,q_representation=.05,q_supervision_weighted=.05,c_total=-.2,r_total=.3)]
    for n,v in [('manifest',m),('summary',s),('history',h),('roots',dict(roots=[]))]:(tmp_path/(n+'.json')).write_text(json.dumps(v))
    assert render(tmp_path)['status']=='complete'
    assert len(list((tmp_path/'report').glob('*.png')))==5
