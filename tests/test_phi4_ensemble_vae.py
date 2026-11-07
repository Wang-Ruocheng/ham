"""Continuous-VAE contracts and a synthetic complete pipeline, not physics."""
import copy
import json
import numpy as np
import pytest
import torch
from recurrent_rg.phi4_ensemble_vae import (Phi4EnsembleVAEQCR, calibrate,
    encode_levels, objectives, independent_step)
from recurrent_rg.phi4_aligned_qcr import AlignedPhi4QCR, aligned_objectives
from recurrent_rg.terminal_r_fit import fingerprint, fit_r, collect_train_targets
from recurrent_rg.config import LossConfig


@pytest.fixture(autouse=True)
def single_thread(): torch.set_num_threads(1)


def test_continuous_features_weak_KL_routes_and_unchanged_C():
    torch.manual_seed(91); old = AlignedPhi4QCR(1., theory_dim=2)
    m = Phi4EnsembleVAEQCR(1.)
    assert fingerprint(m.C) == fingerprint(old.C) and fingerprint(m.R) == fingerprint(old.R)
    x = torch.randn(2, 4, 9, 9, 1, dtype=torch.double)
    calibrate(m, x)
    assert m.Q.center.shape == (625,)
    p = m.Q.probabilities(x.requires_grad_())
    assert not p.requires_grad and torch.allclose(p.sum(-1), torch.ones(2, dtype=torch.double))
    m.Q.set_objective_step(300)
    q,c,r,t = objectives(m, x, torch.Generator().manual_seed(33), 8)
    assert t['vae_effective_KL_weight_level_0'] == .005
    assert float(q) == pytest.approx(np.mean([t[f'vae_cross_entropy_level_{k}']+.005*t[f'vae_KL_level_{k}'] for k in range(3)]))
    assert float(r) == pytest.approx(5*t['closure_start_0_target_1']+2.5*(t['closure_start_0_target_2']+t['closure_start_1_target_2']))
    for loss, names in [(q, ('C','R')), (c, ('Q','R')), (r, ('Q','C'))]:
        ps = [p for name in names for p in getattr(m, name).parameters()]
        assert all(g is None for g in torch.autograd.grad(loss, ps, allow_unused=True, retain_graph=True))
    old_c = aligned_objectives(old, x, torch.Generator().manual_seed(33), 8)[1]
    torch.testing.assert_close(c, old_c, atol=0, rtol=0)
    for a,b in zip(torch.autograd.grad(c, m.C.parameters()), torch.autograd.grad(old_c, old.C.parameters())):
        torch.testing.assert_close(a,b,atol=0,rtol=0)


def test_terminal_keeps_QC_and_Adam_and_half_encoding():
    m=Phi4EnsembleVAEQCR(1.); x=torch.randn(4,8,9,9,1,dtype=torch.double)
    calibrate(m,x)
    opts={n:torch.optim.Adam(getattr(m,n).parameters(),lr=.001) for n in ('Q','C','R')}
    independent_step(m,opts,x,1,information_pairs=8)
    cache,receipt=collect_train_targets(m,x,lambda v:encode_levels(m,m.coarse_grain(v)),samples=4,systems=2,repeats=8)
    assert cache.shape==(32,3,2) and receipt['independent_physical_chains']==4
    twin=Phi4EnsembleVAEQCR(1.);twin.load_state_dict(m.state_dict())
    torch.testing.assert_close(m.Q(x),twin.Q(x),atol=0,rtol=0)
    initial={n:fingerprint(getattr(m,n)) for n in ('Q','C')}
    result=fit_r(m,m.R,{'Q':m.Q,'C':m.C},opts['R'],cache,LossConfig(),max_steps=2,check_every=1)
    assert result['stop_reason']=='update_cap' and set(result['final_R_Adam_steps'])=={3}
    for n in ('Q','C'):
        assert fingerprint(getattr(m,n))==initial[n]
        assert all(int(opts[n].state[p]['step'])==1 for p in getattr(m,n).parameters())
        assert all(p.grad is None for p in getattr(m,n).parameters())


def test_complete_synthetic_pipeline_and_report(tmp_path,monkeypatch):
    from experiments import phi4_ru_vae_qcr as runner
    from experiments.phi4_qcr import FieldDataset
    from scripts.report_phi4_ru_vae import report
    from scripts.run_dual_ising_current import sha256
    config=json.load(open('configs/phi4_ru_weak_kl_vae_g2_seed91_20261105.json'))
    config['execution_check_only']=True
    config['training'].update(epochs=2,system_batch=2,sample_batch=4,information_pairs=8,
        terminal_r_max_steps=2,terminal_r_cache_repeats=1,log_every=1,checkpoint_every=2)
    data=tmp_path/'data';data.mkdir();(data/'manifest.json').write_text('{}')
    config['data']=dict(root=str(data),manifest_sha256=sha256(data/'manifest.json'))
    cells=np.array([[r,u,0] for r in (-2,-1) for u in (.5,1)])
    dataset=FieldDataset(torch.randn(16,4,36,36,1,dtype=torch.double),
        np.repeat(['train','test','train','test'],4),np.repeat(np.arange(4),4),
        np.repeat(cells,4,axis=0),config['data']['manifest_sha256'],[],0,'SYNTHETIC EXECUTION CHECK')
    monkeypatch.setattr(runner,'load_dataset',lambda _:dataset)
    c=tmp_path/'config.json';c.write_text(json.dumps(config));out=tmp_path/'run'
    from recurrent_rg.cli import main
    assert main(['train','--config',str(c),'--output',str(out),'--device','cpu'])==0
    s=json.loads((out/'summary.json').read_text())
    assert s['joint_Q_C_updates']==2 and s['total_R_updates']==4 and s['Q_C_unchanged']
    result=report(out)
    assert result['status']=='complete' and result['thermal_nu'] is None
    assert len(list((out/'report').glob('*.png')))==8
    with np.load(out/'reconstruction_arrays.npz') as arrays:
        for name in ('targets', 'decoded'):
            assert np.isfinite(arrays[name]).all() and (arrays[name] >= 0).all()
            np.testing.assert_allclose(arrays[name].sum(-1), 1., atol=1e-12, rtol=0)
    with pytest.raises(FileExistsError):runner.run(c,out)


def test_reject_silent_recipe_changes():
    from experiments.phi4_ru_vae_qcr import validate
    c=json.load(open('configs/phi4_ru_weak_kl_vae_g2_seed91_20261105.json'))
    validate(c)
    for key,value in [('epochs',2999),('terminal_r_fit',False),('learning_rate',.003)]:
        bad=copy.deepcopy(c);bad['training'][key]=value
        with pytest.raises(ValueError):validate(bad)


def test_uuid_prefix_normalized_without_accepting_other_device(monkeypatch):
    from types import SimpleNamespace
    from experiments.phi4_hopping_qcr import _device_receipt
    uuid='5bba87f1-7d32-4dde-fe28-caba4468ab42'
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda _:SimpleNamespace(uuid=uuid,name='stub',total_memory=24*1024**3))
    assert _device_receipt(torch.device('cuda:0'),'GPU-'+uuid)['uuid']==uuid
    with pytest.raises(RuntimeError):_device_receipt(torch.device('cuda:0'),'GPU-other')
