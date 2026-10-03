from dataclasses import replace
import numpy as np
import pytest
import torch
from recurrent_rg.config import ModelConfig
from recurrent_rg.checkpoints import build_model, qcr_state_dict
from recurrent_rg.evaluation import collect_trajectories
from recurrent_rg.training import evaluate
from recurrent_rg.data.statistics import two_spin_distribution


def model(lookup=True):
    return build_model(ModelConfig(hidden=6, kernel_sampling='bernoulli', binary_lookup=lookup), 2)


@pytest.mark.parametrize('lookup', [False, True])
def test_independent_draws_and_probability(lookup):
    torch.manual_seed(13)
    m = model(lookup)
    with torch.no_grad():
        m.rg_kernel.field_head.projection.weight.zero_()
        m.rg_kernel.field_head.projection.bias.fill_(.5*np.log(3.))
        m.eval()
        y = m.rg_kernel(torch.ones(1, 8000, 2, 1, dtype=torch.float64))
    assert torch.allclose(y.abs(), torch.ones_like(y), atol=1e-14, rtol=0)
    assert abs(float((y>0).double().mean())-.75) < .02


def test_lookup_forward_and_gradient_match_without_correlated_noise():
    torch.manual_seed(19)
    a, b = model(True), model(False)
    b.load_state_dict(a.state_dict())
    x = (2*torch.randint(2, (2,8,8,1))-1).double().requires_grad_(True)
    y = x.detach().clone().requires_grad_(True)
    torch.manual_seed(22); out_a = a.rg_kernel(x)
    torch.manual_seed(22); out_b = b.rg_kernel(y)
    torch.testing.assert_close(out_a, out_b)
    out_a.sum().backward(); out_b.sum().backward()
    torch.testing.assert_close(x.grad, y.grad)
    for p, q in zip(a.rg_kernel.parameters(), b.rg_kernel.parameters()):
        torch.testing.assert_close(p.grad, q.grad)


def test_evaluation_reproducible_and_preserves_training_rng():
    torch.manual_seed(5); m = model()
    x = (2*torch.randint(2, (4,8,16,1))-1).double()
    before = torch.get_rng_state().clone()
    a = collect_trajectories(m, x, samples=8)
    assert torch.equal(before, torch.get_rng_state())
    b = collect_trajectories(m, x, samples=8)
    np.testing.assert_array_equal(a['direct'], b['direct'])
    c = collect_trajectories(m, x, samples=8, sampling_seed=987)
    assert not np.array_equal(a['direct'][:,1:], c['direct'][:,1:])
    evaluate(m, x, two_spin_distribution, 8)
    assert torch.equal(before, torch.get_rng_state())


def test_sampling_does_not_change_initial_weights_or_parameter_count():
    c = ModelConfig(hidden=6, kernel_sampling='deterministic')
    torch.manual_seed(91); a = build_model(c,2)
    torch.manual_seed(91); b = build_model(replace(c,kernel_sampling='bernoulli'),2)
    for k,v in qcr_state_dict(a).items():
        assert torch.equal(v, qcr_state_dict(b)[k])
    with pytest.raises(ValueError):
        build_model(replace(c,fixed_kernel=True,kernel_sampling='bernoulli'),2)


def test_two_dimensional_joint_objective_has_finite_gradients():
    from recurrent_rg.losses import qcr_objective
    from recurrent_rg.data.statistics import plaquette_distribution
    torch.manual_seed(8)
    m = build_model(ModelConfig(dimension=2,theory_dim=2,hidden=6,
                               kernel_sampling='bernoulli'), 1)
    x = (2*torch.randint(2,(4,8,8,8,1))-1).double()
    loss, _ = qcr_objective(m,x,pattern_fn=plaquette_distribution,
                           kernel_noncollapse_mode='variance_floor')
    loss.backward()
    for module in [m.rg_kernel,m.theory_encoder,m.theory_rg]:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())


def test_temperature_does_not_change_hard_probability_draws():
    torch.manual_seed(9)
    m = model(); head = m.rg_kernel.field_head
    logits = torch.linspace(-2,2,1000,dtype=torch.float64)
    head.relaxation_temperature = .3
    torch.manual_seed(90); a = head.sample_logits(logits)
    head.relaxation_temperature = 2.
    torch.manual_seed(90); b = head.sample_logits(logits)
    assert torch.equal(a>0,b>0)
