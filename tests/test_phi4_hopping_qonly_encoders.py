"""Small synthetic CPU checks, not scientific evidence."""
import numpy as np
import pytest
import torch
from recurrent_rg.phi4_aligned_qcr import RawPatchTheoryEncoder
from scripts import run_phi4_hopping_qonly_encoders as run


def models():return run.make_models(RawPatchTheoryEncoder,.9,np.full((4,625),1/625))


def test_common_decoder_fresh_reproducible_rng_isolated():
    rng=torch.random.get_rng_state().clone();a=models();b=models()
    assert torch.equal(rng,torch.random.get_rng_state())
    assert len({run.base.state_sha(q.decoder.state_dict()) for q in a.values()})==1
    assert all(run.base.state_sha(a[n].state_dict())==run.base.state_sha(b[n].state_dict()) for n in run.ARMS)


@pytest.mark.parametrize('arm',run.ARMS)
def test_updates_gradients_scalar_and_counters(arm):
    q=models()[arm];opt=torch.optim.Adam(q.parameters(),lr=.001)
    levels=[torch.randn(2,4,8,8,1,dtype=torch.float64,requires_grad=True) for _ in range(4)]
    p=(torch.rand(2,4,4,625,dtype=torch.float64)/625).requires_grad_(True)
    for _ in range(2):run.update(q,opt,levels,p)
    assert all(x.grad is None for x in levels) and p.grad is None
    assert all(x.grad is None for x in q.parameters())
    counts=run.base.optimizer_counts(opt,q)
    assert counts['minimum']==counts['maximum']==2
    assert q.encode(levels[0],p[:,0]).shape==(2,1)
    assert q.decoder.bounds()['global_lipschitz_upper_bound']<=9+1e-10


def test_statistics_only_uses_its_subset():
    q=models()['B_statistics'];x=torch.randn(2,4,8,8,1,dtype=torch.float64);p=torch.rand(2,4,625,dtype=torch.float64)
    expected=q.encode(x[:,:2],p[:,:2]);p[:,2:]*=100
    torch.testing.assert_close(expected,q.encode(x[:,:2]*99,p[:,:2]),rtol=0,atol=0)
    torch.testing.assert_close(expected,q.encode(x[:,:2],p[:,:2].flip(1)),rtol=0,atol=0)


def test_hierarchical_symmetries_chunk_equivalence_and_amplitude():
    q=models()['C_hierarchical'];x=torch.randn(2,4,8,8,1,dtype=torch.float64)
    with torch.no_grad():
        y=q.encode(x,None)
        torch.testing.assert_close(y,q.encode(x.flip(1),None),rtol=1e-10,atol=1e-12)
        torch.testing.assert_close(y,q.encode(x.roll((2,3),(2,3)),None),rtol=1e-10,atol=1e-12)
        q.frame_chunk=3
        torch.testing.assert_close(y,q.encode(x,None),rtol=1e-10,atol=1e-12)
        assert not torch.allclose(y,q.encode(x*2,None),rtol=1e-5,atol=1e-10)


def test_objective_exact_weights_and_subset_boundaries():
    class Dummy:
        def __init__(self):self.seen=[]
        def encode(self,x,p):self.seen.append(p.clone());return p.mean((1,2))[:,None]
        def decode(self,g):return g.expand(-1,625)
    q=Dummy();x=[torch.zeros(2,4,8,8,1) for _ in range(4)];p=torch.rand(2,4,4,625)
    total,r,s=run.q_objective(q,x,p)
    torch.testing.assert_close(total,10*r.mean()+20*s[0]+5*s[1:].mean())
    for k in range(4):
        torch.testing.assert_close(q.seen[2*k],p[:,k,:2]);torch.testing.assert_close(q.seen[2*k+1],p[:,k,2:])
