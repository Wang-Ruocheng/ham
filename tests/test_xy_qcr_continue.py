import copy
import json
import numpy as np
import pytest
import torch
from experiments import xy_qcr as original
from experiments import xy_qcr_continue as continuation
from experiments import phi4_hopping_qcr as base
from recurrent_rg.xy_qcr import XYQCR, independent_step


@pytest.fixture(autouse=True)
def one_thread():torch.set_num_threads(1)


def test_joint_restore_keeps_weights_moments_lr_and_cumulative_clock():
    m=XYQCR();opts=original.optimizers(m);x=torch.randn(2,4,18,18,1,dtype=torch.double)
    independent_step(m,x,opts,torch.Generator().manual_seed(8),information_pairs=8)
    for o in opts.values():
        for state in o.state.values():state['step'].fill_(1250)
    p=dict(recipe=original.RECIPE,step=1250,model=copy.deepcopy(m.state_dict()),
        optimizers={n:copy.deepcopy(o.state_dict()) for n,o in opts.items()},
        manifest=dict(expected_final_sha256=base.module_fingerprints(m)))
    restored,ro=continuation.restore(p,'cpu')
    assert base.module_fingerprints(restored)==base.module_fingerprints(m)
    for n,o in opts.items():
        for old,new in zip(o.state.values(),ro[n].state.values()):
            for key in ('step','exp_avg','exp_avg_sq'):torch.testing.assert_close(old[key],new[key],rtol=0,atol=0)
    independent_step(restored,x,ro,torch.Generator().manual_seed(9),information_pairs=8)
    base.optimizer_step_counts(ro,1251)
    assert all(base.module_fingerprints(restored)[n]!=base.module_fingerprints(m)[n] for n in opts)


def test_plateau_requires_two_metrics_and_confirmations():
    c=continuation.preset()
    rows=[dict(extra_steps=i*250,weighted_closure=1.,normalized_closure=.2) for i in range(11)]
    assert continuation.plateau_reached(rows,c)
    assert not continuation.plateau_reached(rows[:9],c)
    for key in c['stopping_metrics']:
        for factor in (.9,1.1):
            changing=copy.deepcopy(rows)
            for i,r in enumerate(changing):r[key]*=factor**i
            assert not continuation.plateau_reached(changing,c)
    noisy=copy.deepcopy(rows)
    for i,r in enumerate(noisy):r['normalized_closure']*=.5 if i%2 else 1.5
    assert not continuation.plateau_reached(noisy,c)


def test_normalized_metric_does_not_reward_coordinate_shrinkage():
    class Linear(torch.nn.Module):
        def forward(self,g):return .2*g
    g=torch.tensor([[[[1.],[2.],[4.]],[[2.],[4.],[8.]],[[3.],[6.],[12.]]]],dtype=torch.double)
    a=continuation.closure_statistics(Linear(),g);b=continuation.closure_statistics(Linear(),g*.1)
    assert b['weighted_closure']==pytest.approx(a['weighted_closure']*.01)
    assert b['normalized_closure']==pytest.approx(a['normalized_closure'])
    with pytest.raises(FloatingPointError):continuation.closure_statistics(Linear(),torch.zeros_like(g))


def test_fixed_probe_reproducible_and_model_unchanged():
    from types import SimpleNamespace
    m=XYQCR();data=SimpleNamespace(configurations=torch.randn(3,4,18,18,1),train_indices=np.array([0,2]))
    c={**continuation.preset(),'probe_samples':4};before=base.state_fingerprint(m)
    a,ca=continuation.fixed_panel(m,data,'cpu',c);b,cb=continuation.fixed_panel(m,data,'cpu',c)
    torch.testing.assert_close(a,b,rtol=0,atol=0);torch.testing.assert_close(ca,cb,rtol=0,atol=0)
    assert a.shape==(2,2,3,1) and before==base.state_fingerprint(m)


def test_continuation_cli_is_explicit_and_strict(tmp_path,monkeypatch):
    from recurrent_rg.cli import main
    p=tmp_path/'config.json';p.write_text(json.dumps(continuation.preset()))
    assert continuation.validate_config(p)==continuation.preset()
    calls=[];monkeypatch.setattr(continuation,'run',lambda *args:calls.append(args))
    main(['train','--config',str(p),'--output',str(tmp_path/'out')]);assert len(calls)==1
    with pytest.raises(SystemExit):main(['train','--config',str(p),'--output','unused','--allow-legacy-training'])
