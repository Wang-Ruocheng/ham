"""Label-free matched control contracts and small synthetic update checks."""
import copy
import json
from itertools import product
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from experiments import ising_kb_unsupervised_matched as run
from experiments import ising_kb_supervised_short as sup
from experiments import ising_kb_cnn_qcr as kb
from experiments import ising_cnn_short_trial as binary
from experiments import phi4_hopping_qcr as common
from recurrent_rg.empirical_mi import binary_lattice_mi_statistics


@pytest.fixture(autouse=True)
def single_thread():torch.set_num_threads(1)


def test_config_dispatch_and_fixed_budget(tmp_path,monkeypatch):
    path='configs/ising2d_KB_cnn_unsupervised_matched1250_seed91.json'
    assert run.validate_config(path)==run.preset()
    from recurrent_rg.cli import main
    calls=[];monkeypatch.setattr(run,'run',lambda *a:calls.append(a))
    main(['train','--config',path,'--output',str(tmp_path/'out')]);assert len(calls)==1
    for extra in (['--resume','x'],['--initialize-from','x'],['--allow-legacy-training'],['--device','cuda:2']):
        with pytest.raises(SystemExit):main(['train','--config',path,'--output','x']+extra)
    for key,value in [('steps',1300),('supervision_weight',1.),('seed',True)]:
        obj=run.preset();obj[key]=value;bad=tmp_path/'bad.json';bad.write_text(json.dumps(obj))
        with pytest.raises(ValueError):run.validate_config(bad)
    for speed in [.01,.3]:assert run.fixed_budget(50,[20,1]+[speed]*8,.1,.01)['formal_steps']==1250
    with pytest.raises(common.BudgetHold):run.fixed_budget(50,[1.]*10,.1,.01)
    assert run.matched_config().training.epochs==1250


def test_label_free_draw_probe_and_multistep_equivalence():
    x=torch.randint(2,(2,4,18,18,1),dtype=torch.int8)*2-1
    ds=SimpleNamespace(configurations=x,couplings=torch.tensor([[.3,-.01],[.6,.01]],dtype=torch.float64))
    scaler=sup.label_scaler(ds.couplings)
    rng1=torch.Generator().manual_seed(192);rng2=torch.Generator().manual_seed(192)
    a,z=sup.draw(ds,rng1,'cpu',scaler);b=kb.draw_batch(x,rng2,'cpu')
    torch.testing.assert_close(a,b);assert torch.equal(rng1.get_state(),rng2.get_state())
    cfg=run.matched_config();models=[binary.fresh(cfg,'cpu') for _ in range(3)]
    opts=[kb.optimizers(m) for m in models]
    hist=binary_lattice_mi_statistics(a,block=3);hist.prepare_selection()
    patterns=torch.tensor(list(product((-1.,1.),repeat=9)),dtype=torch.float64)[None]
    for step in range(3):
        for i,m in enumerate(models):
            with binary.isolated_rng('cpu',394+step):
                if i==0:binary.step(m,b,hist,torch.arange(2),patterns,cfg,opts[i])
                else:sup.step(m,a,z,hist,torch.arange(2),patterns,cfg,opts[i],weight=0. if i==1 else 10.)
    assert common.state_fingerprint(models[0])==common.state_fingerprint(models[1])
    assert common.state_fingerprint(models[0].rg_kernel)==common.state_fingerprint(models[2].rg_kernel)
    assert common.state_fingerprint(models[0].theory_encoder)!=common.state_fingerprint(models[2].theory_encoder)
    for op in opts:assert common.optimizer_step_counts(op,3)['R']['max']==3
    p=kb.train_probe(models[0],ds,'cpu')
    ds.couplings[:]=float('nan')
    assert p==kb.train_probe(models[0],ds,'cpu')


def test_posthoc_readout_and_model_free_report(tmp_path,monkeypatch):
    from scripts import report_ising_kb_unsupervised_matched as report
    from recurrent_rg.phi4_ru_flow import bounded_flow
    labels=np.array([[.2+.03*k,-.01+.001*b] for k in range(17) for b in range(21) for _ in range(2)])
    splits=np.array(['test' if (k+abs(b-10))%3==1 else 'train' for k in range(17) for b in range(21) for _ in range(2)])
    g0=labels@np.array([[2.,3.],[1.,-4.]])+[.1,.2];g=np.stack([g0,g0*.9,g0*.8],1)
    a=dict(direct_g=g,direct_full_g=g,one_step_g=g[:,1:]+.1,rollout_g=g+.1,
        all_start_g=np.stack([g[:,1]+.1,g[:,2]+.2,g[:,2]+.1],1),all_start_pairs=np.array([[0,1],[0,2],[1,2]]),
        labels=labels,splits=splits,cell_ids=np.repeat(np.arange(357),2),train_direct=g[splits=='train'],test_direct=g[splits=='test'])
    assert report.linear_readout(a)['test']['r2']==pytest.approx([1.,1.])
    roots,flow=bounded_flow(SimpleNamespace(advance=lambda x:1.1*x),a,'cpu')
    manifest=dict(recipe=run.RECIPE,labels_used_in_training=False,test_used_in_training=False,
        q_supervision=dict(weight=0.,enabled=False))
    summary=dict(recipe=run.RECIPE,status='complete',actual_steps=1250,
        closure=binary.closure_metrics(a),optimizer_steps={k:dict(min=1250,max=1250) for k in ('Q','C','R')})
    for name,value in [('manifest',manifest),('summary',summary),('roots',dict(roots=roots,flow_analysis=flow)),
        ('history',[dict(step=1250,q_total=.1,c_total=.1,r_total=.1)]),
        ('train_probes',[dict(step=0,weighted_closure=.2),dict(step=1250,weighted_closure=.1)])]:
        common.write_json(tmp_path/(name+'.json'),value)
    np.savez(tmp_path/'evaluation_arrays.npz',**a)
    monkeypatch.setattr(report,'comparison',lambda *a:dict(synthetic_test=True))
    assert report.render(tmp_path)['actual_steps']==1250
    assert len(list((tmp_path/'report').glob('*.png')))==7


def test_real_baseline_comparison_when_local_public_files_present():
    from scripts import report_ising_kb_unsupervised_matched as report
    root=Path('results/ising2d_KB_cnn_supervised_short_seed91_vm225_20261104_monitor/final')
    if not root.exists():pytest.skip('Local public baseline not installed')
    a=report.read_arrays(root/'evaluation_arrays.npz');s=json.loads((root/'summary.json').read_text())
    value=report.comparison(a,s,root)
    assert value['identical_C_target_features']
    assert value['supervised']==value['unsupervised']
    assert json.loads((root/'manifest.json').read_text())['config']==json.loads(json.dumps(run.matched_config().to_dict()))
