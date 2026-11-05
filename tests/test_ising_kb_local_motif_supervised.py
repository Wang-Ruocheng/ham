"""Free decoder gain ablation: one intervention, pairing and model-free reporting."""
import json
from pathlib import Path
from itertools import product
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from experiments import ising_kb_local_motif_supervised as run
from experiments import ising_kb_supervised_short as sup
from experiments import ising_kb_cnn_qcr as kb
from experiments import ising_cnn_short_trial as binary
from experiments import phi4_hopping_qcr as common
from recurrent_rg.empirical_mi import binary_lattice_mi_statistics


@pytest.fixture(autouse=True)
def single_thread():torch.set_num_threads(1)


def test_strict_contract_cli_and_budget(tmp_path,monkeypatch):
    path='configs/ising2d_KB_local_motif_g2_supervised_matched1250_seed91.json'
    assert run.validate_config(path)==run.preset()
    from recurrent_rg.cli import main
    calls=[];monkeypatch.setattr(run,'run',lambda *a:calls.append(a))
    main(['train','--config',path,'--output',str(tmp_path/'out')]);assert len(calls)==1
    for extra in (['--resume','x'],['--initialize-from','x'],['--allow-legacy-training'],['--device','cuda:2']):
        with pytest.raises(SystemExit):main(['train','--config',path,'--output','x']+extra)
    for key,value in [('steps',1300),('supervision_weight',1.),('decoder_column_constraint','unit'),('supervision_weight',True),('seed',True)]:
        obj=run.preset();obj[key]=value;p=tmp_path/'bad.json';p.write_text(json.dumps(obj))
        with pytest.raises(ValueError):run.validate_config(p)
    assert run.fixed_budget(40,[10,1]+[.4]*8,.1,.02)['formal_steps']==1250
    with pytest.raises(common.BudgetHold):run.fixed_budget(1000,[1.]*10,.1,.02)


def test_only_Q_changes_and_C_stays_paired():
    from dataclasses import replace
    from experiments import ising_kb_free_decoder_matched as cnn
    cfg=run.matched_config();oldcfg=cnn.matched_config()
    assert cfg==replace(oldcfg,model=replace(oldcfg.model,q_encoder='local_motif'))
    local=binary.fresh(cfg,'cpu');old=binary.fresh(oldcfg,'cpu')
    paired,sha=run.paired_initialization(cfg,local)
    assert all(paired.values()) and sha==common.state_fingerprint(old)
    x=(torch.randint(2,(2,4,18,18,1))*2-1).double()
    labels=torch.tensor([[-1.,-1.],[1.,1.]],dtype=torch.float64,requires_grad=True)
    hist=binary_lattice_mi_statistics(x,block=3);hist.prepare_selection()
    patterns=torch.tensor(list(product((-1.,1.),repeat=9)),dtype=torch.float64)[None]
    opts=[kb.optimizers(m) for m in (local,old)]
    for i in range(3):
        for model,c,op in zip((local,old),(cfg,oldcfg),opts):
            with binary.isolated_rng('cpu',394+i):v=run.step(model,x,labels,hist,torch.arange(2),patterns,c,op)
            assert v['q_total']==pytest.approx(v['q_representation']+10*v['q_supervision'])
        assert common.state_fingerprint(local.rg_kernel)==common.state_fingerprint(old.rg_kernel)
    assert labels.grad is None
    for op in opts:assert common.optimizer_step_counts(op,3)['R']['max']==3


def test_supervision_is_microscopic_and_has_no_C_R_or_label_gradient():
    from experiments.rsmi_separate_qcr import microscopic_coupling_loss
    m=binary.fresh(run.matched_config(),'cpu')
    x=(torch.randint(2,(2,4,18,18,1))*2-1).double()
    y=torch.tensor([[-1.,-1.],[1.,1.]],dtype=torch.float64,requires_grad=True)
    loss=microscopic_coupling_loss(m.theory_encoder,x,y)
    assert float(loss.detach())==pytest.approx(float((m.theory_encoder(x)-y.detach()).square().sum(-1).mean().detach()),rel=1e-12)
    protected=list(m.rg_kernel.parameters())+list(m.theory_rg.parameters())+[y]
    assert all(v is None for v in torch.autograd.grad(loss,protected,allow_unused=True))


def test_model_free_weak_report(tmp_path,monkeypatch):
    from scripts import report_ising_kb_local_motif_supervised as report
    from recurrent_rg.phi4_ru_flow import bounded_flow
    labels=np.array([[.2+.03*k,-.01+.001*b] for k in range(17) for b in range(21) for _ in range(2)])
    splits=np.array(['test' if (k+abs(b-10))%3==1 else 'train' for k in range(17) for b in range(21) for _ in range(2)])
    g0=labels@np.array([[2.,3.],[1.,-4.]])+[.1,.2];g=np.stack([g0,g0*.9,g0*.8],1)
    a=dict(direct_g=g,direct_full_g=g,one_step_g=g[:,1:]+.1,rollout_g=g+.1,
        all_start_g=np.stack([g[:,1]+.1,g[:,2]+.2,g[:,2]+.1],1),all_start_pairs=np.array([[0,1],[0,2],[1,2]]),
        labels=labels,splits=splits,cell_ids=np.repeat(np.arange(357),2),train_direct=g[splits=='train'],test_direct=g[splits=='test'])
    roots,flow=bounded_flow(SimpleNamespace(advance=lambda x:1.1*x),a,'cpu')
    manifest=dict(recipe=run.RECIPE,labels_used_in_training=True,test_used_in_training=False,
        decoder_column_constraint='free_after_initialization',q_supervision=dict(weight=10.,enabled=True,scale=0,coarse_labels=False))
    summary=dict(recipe=run.RECIPE,status='complete',actual_steps=1250,
        closure=binary.closure_metrics(a),optimizer_steps={k:dict(min=1250,max=1250) for k in ('Q','C','R')})
    for name,value in [('manifest',manifest),('summary',summary),('roots',dict(roots=roots,flow_analysis=flow)),
        ('history',[dict(step=1250,decoder_column_norms=[.4,.7],q_total=.3,q_representation=.1,q_supervision_weighted=.2,c_total=.1,r_total=.1)]),
        ('train_probes',[dict(step=0,weighted_closure=.2),dict(step=1250,weighted_closure=.1)])]:
        common.write_json(tmp_path/(name+'.json'),value)
    np.savez(tmp_path/'evaluation_arrays.npz',**a)
    monkeypatch.setattr(report,'comparison',lambda *a:dict(synthetic_test=True))
    assert report.render(tmp_path)['actual_steps']==1250
    assert len(list((tmp_path/'report').glob('*.png')))==9


