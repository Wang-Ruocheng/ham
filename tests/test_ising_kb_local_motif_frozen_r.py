import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from experiments import ising_kb_local_motif_frozen_r as run


def test_pinned_local_source_and_cli(tmp_path,monkeypatch):
    p='configs/ising2d_KB_local_motif_frozen_R_seed91.json'
    assert run.validate_config(p)==run.preset()
    assert 'local_motif_supervised_matched1250' in str(run.SOURCE)
    from recurrent_rg.cli import main
    called=[];monkeypatch.setattr(run,'run',lambda *a:called.append(a))
    main(['train','--config',p,'--output','unused']);assert len(called)==1
    for args in [['--resume','old'],['--allow-legacy-training'],['--device','cuda:3']]:
        with pytest.raises(SystemExit):main(['train','--config',p,'--output','unused']+args)
    bad=run.preset();bad['source']='wrong';(tmp_path/'bad.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError):run.validate_config(tmp_path/'bad.json')


def test_source_seeded_search_finds_repellers_and_central_saddle():
    torch.set_num_threads(1)
    model=SimpleNamespace(advance=lambda x:torch.stack([1.2*x[:,0],.8*x[:,1]+.2*x[:,1]**3],1))
    rng=np.random.default_rng(1);a=dict(flow_bounds=np.array([[-2.,2.],[-2.,2.]]),train_direct=rng.uniform(-1.8,1.8,(40,3,2)))
    roots=[];source=[dict(g=[.01,-1.03]),dict(g=[.02,1.02]),dict(g=[0.,.05])]
    receipt=run.supplement_root_search(model,a,roots,source,'cpu')
    assert len(roots)==3 and len(receipt['attempts'])==15
    roots=sorted(roots,key=lambda r:r['g'][1])
    assert [r['classification'] for r in roots]==['repelling','saddle','repelling']
    np.testing.assert_allclose([r['g'] for r in roots],[[0,-1],[0,0],[0,1]],atol=1e-7)
    # Repeated searches deduplicate; diagnostics do not create extra roots.
    run.supplement_root_search(model,a,roots,source,'cpu');assert len(roots)==3


def test_local_motif_model_freezes_exactly():
    from experiments.ising_kb_local_motif_supervised import matched_config
    from experiments.ising_cnn_short_trial import fresh
    from experiments.phi4_hopping_qcr import state_fingerprint
    torch.set_num_threads(1);m=fresh(matched_config(),'cpu');g=torch.randn(16,3,2,dtype=torch.float64)
    opt=torch.optim.Adam(m.theory_rg.parameters(),lr=.003);run.update(m,g,opt)
    old={n:state_fingerprint(mod) for n,mod in [('Q',m.theory_encoder),('C',m.rg_kernel)]}
    frozen=run.freeze_and_continue(m,dict(epoch=1,optimizers={'R':opt.state_dict()}))
    run.update(m,g,frozen)
    assert all(state_fingerprint(mod)==old[n] for n,mod in [('Q',m.theory_encoder),('C',m.rg_kernel)])
    assert all(p.grad is None for mod in (m.theory_encoder,m.rg_kernel) for p in mod.parameters())


def test_array_report_uses_same_coordinates(tmp_path):
    from scripts import report_ising_kb_local_motif_frozen_r as report
    from scripts.report_phi4_hopping_qcr import closure_metrics
    p=Path('results/ising2d_KB_local_motif_supervised_matched1250_seed91_vm225_20261104_monitor/final')
    if not p.exists():pytest.skip('Source public results not installed')
    a=dict(np.load(p/'evaluation_arrays.npz',allow_pickle=False));roots=json.loads((p/'roots.json').read_text())
    for k in ('one_step_g','rollout_g','all_start_g','flow_g','flow_R','flow_displacement'):a['source_'+k]=a[k].copy()
    np.savez(tmp_path/'evaluation_arrays.npz',**a)
    s=dict(additional_Q_updates=0,additional_C_updates=0,frozen_before_sha256={'Q':'q','C':'c'},frozen_after_sha256={'Q':'q','C':'c'},
        actual_extra_R_updates=2,before_closure=closure_metrics(a),after_closure=closure_metrics(a),source_roots=roots['roots'],reason='synthetic_test',root_search={})
    for name,obj in [('manifest',dict(labels_used_in_training=False,test_used_in_training=False)),('summary',s),('roots',roots),
        ('history',[dict(extra_R_updates=0,train={'total':1.},best_train_closure=1.),dict(extra_R_updates=2,train={'total':.8},best_train_closure=.8)])]:
        (tmp_path/(name+'.json')).write_text(json.dumps(obj))
    assert report.render(tmp_path)['Q_C_unchanged']
    assert len(list((tmp_path/'report').rglob('*.png')))==8
