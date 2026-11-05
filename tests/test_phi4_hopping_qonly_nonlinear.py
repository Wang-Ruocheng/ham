"""CPU synthetic verification of the explicit scalar nonlinear decoder arm."""
import copy

import numpy as np
import torch

from recurrent_rg.phi4_aligned_qcr import RawPatchTheoryEncoder
from scripts import run_phi4_hopping_qonly_nonlinear as run


def source():
    torch.manual_seed(7)
    return RawPatchTheoryEncoder(1.,hidden=8,theory_dim=1,local_features=3,patch_chunk_size=128)


def test_initial_decoder_and_encoder_functions_identical_and_source_protected():
    q=source();pin=run.base.state_sha(q.state_dict())
    models=run.make_models(q)
    fields=torch.randn(2,4,4,4,1,dtype=torch.float64)
    g=torch.linspace(-2,2,19,dtype=torch.float64).reshape(-1,1)
    assert torch.equal(models['A_affine'](fields),models['B_nonlinear'](fields))
    assert torch.equal(models['A_affine'].decode(g),models['B_nonlinear'].decode(g))
    assert run.base.state_sha(q.state_dict())==pin


def test_nonlinear_updates_are_finite_and_fields_targets_detached():
    models=run.make_models(source())
    fields=[torch.randn(2,4,4,4,1,dtype=torch.float64,requires_grad=True) for _ in range(4)]
    targets=(torch.rand(2,4,4,625,dtype=torch.float64)/625).requires_grad_(True)
    q=models['B_nonlinear'];opt=torch.optim.Adam(q.parameters(),lr=.001)
    for _ in range(3):run.base.update(q,opt,fields,targets,np.ones(4))
    assert all(x.grad is None for x in fields) and targets.grad is None
    assert all(p.grad is None for p in q.parameters())
    counts=run.base.optimizer_counts(opt,q)
    assert counts['minimum']==counts['maximum']==3
    assert q.decoder.residual[-1].weight.norm()>0
    assert q.decoder.bounds()['global_lipschitz_upper_bound']<=9+1e-10


def test_spectral_projection_enforces_global_gain_bound_and_jvp_matches_fd():
    q=run.NonlinearQ(source())
    with torch.no_grad():
        for layer in q.decoder.residual:
            if isinstance(layer,torch.nn.Linear):layer.weight.normal_(0,3)
    q.normalize_decoder_()
    g=np.linspace(-1,1,11).reshape(-1,1)
    decoded,jac,info=run.decoder_diagnostics(q,g)
    h=1e-5
    with torch.no_grad():
        t=torch.tensor(g,dtype=torch.float64)
        fd=((q.decode(t+h)-q.decode(t-h))/(2*h)).numpy()
    np.testing.assert_allclose(jac,fd,rtol=2e-6,atol=1e-8)
    assert info['global_lipschitz_upper_bound']<=9+1e-10
    with torch.no_grad():
        zero=torch.zeros(1,1,dtype=torch.float64)
        torch.testing.assert_close(q.decode(zero),q.decoder.affine(zero),rtol=0,atol=1e-14)


def test_initialization_is_reproducible_and_does_not_consume_cpu_rng():
    q=source();rng=torch.random.get_rng_state().clone()
    a=run.make_models(q);b=run.make_models(q)
    assert torch.equal(rng,torch.random.get_rng_state())
    assert run.base.state_sha(a['B_nonlinear'].state_dict())==run.base.state_sha(b['B_nonlinear'].state_dict())


def test_decoder_checkpoint_roundtrip():
    q=run.NonlinearQ(source());saved=copy.deepcopy(q.state_dict())
    restored=run.NonlinearQ(source());restored.load_state_dict(saved,strict=True)
    assert run.base.state_sha(q.state_dict())==run.base.state_sha(restored.state_dict())
