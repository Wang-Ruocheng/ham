import copy
import torch

from experiments._utils import balanced_views
from experiments.joint_qcr_ising import distribution_geometry_loss, qcr_objective, two_spin_distribution
from recurrent_rg.model_nd import RecurrentRGNetwork1D


def test_variance_floor_allows_bias_but_penalizes_constant_output():
    from experiments.joint_qcr_ising import binary_kernel_variance_floor
    source = torch.tensor([-1., 1.] * 10, requires_grad=True)
    biased = torch.tensor([-1.] * 2 + [1.] * 18)
    assert binary_kernel_variance_floor([source, biased]) == 0
    collapsed = torch.ones(20, requires_grad=True)
    loss = binary_kernel_variance_floor([source, collapsed])
    assert loss > 0
    loss.backward()
    assert source.grad is None
    assert (collapsed.grad > 0).all()
    assert binary_kernel_variance_floor([torch.ones(20), collapsed]) == 0


def test_field_objective_has_finite_joint_gradients():
    from recurrent_rg.model_nd import RecurrentRGNetwork2D
    from experiments.joint_qcr_ising import plaquette_distribution
    torch.manual_seed(91)
    model = RecurrentRGNetwork2D(levels=3, hidden=8, theory_dim=2).double()
    model.theory_encoder.binary_lookup = model.rg_kernel.binary_lookup = True
    x = (2*(torch.rand(6, 12, 8, 8, 1) < .7)-1).double()
    loss, _ = qcr_objective(model, x, pattern_fn=plaquette_distribution,
                           objective_mode='data_geometry', rollout_weight=5.,
                           kernel_noncollapse_mode='variance_floor')
    loss.backward()
    for mod in (model.theory_encoder, model.rg_kernel, model.theory_rg):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in mod.parameters())


def test_identical_distributions_allow_constant_g():
    p = torch.tensor([[.25]*4]*3, dtype=torch.float64)
    g = torch.zeros(3, 1, dtype=torch.float64, requires_grad=True)
    loss, stats = distribution_geometry_loss(g, p, p)
    assert loss.item() == 0
    assert stats['resolved_fraction'].item() == 0
    loss.backward()
    assert torch.isfinite(g.grad).all()


def test_different_distributions_require_separation():
    p = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float64)
    g = torch.tensor([[0.], [1.]], dtype=torch.float64, requires_grad=True)
    loss, stats = distribution_geometry_loss(g, p, p)
    assert loss > 0 and stats['resolved_fraction'] == 1
    loss.backward()
    assert g.grad[0] > 0 and g.grad[1] < 0


def test_noise_interval_does_not_force_sampling_difference_apart():
    a = torch.tensor([[1., 0.], [.5, .5]], dtype=torch.float64)
    b = torch.tensor([[0., 1.], [.9, .1]], dtype=torch.float64)
    loss, stats = distribution_geometry_loss(torch.zeros(2, 1, dtype=torch.float64), a, b)
    assert loss == 0 and stats['resolved_fraction'] == 0


def test_target_is_detached_and_scale_is_not_batch_normalized():
    p = torch.tensor([[.7, .3], [.3, .7]], dtype=torch.float64, requires_grad=True)
    g = torch.tensor([[0.], [.1]], dtype=torch.float64, requires_grad=True)
    loss, stats = distribution_geometry_loss(g, p, p)
    loss.backward()
    assert p.grad is None
    _, smaller = distribution_geometry_loss(g, p.detach()*.1+.45, p.detach()*.1+.45)
    torch.testing.assert_close(smaller['target_distance'], .1*stats['target_distance'])


def test_all_scales_equal_and_binary_backend_gradients_match():
    torch.manual_seed(91)
    model = RecurrentRGNetwork1D(levels=4, hidden=8, theory_dim=1).double()
    lookup = copy.deepcopy(model)
    lookup.theory_encoder.binary_lookup = True
    lookup.rg_kernel.binary_lookup = True
    x = (2*torch.randint(0, 2, (6, 12, 32, 1))-1).double()
    x = balanced_views(x, 12)
    losses = []
    for m in [model, lookup]:
        loss, terms = qcr_objective(m, x, pattern_fn=two_spin_distribution, objective_mode='data_geometry', rollout_weight=5.)
        torch.testing.assert_close(terms['separation'], torch.stack([terms[f'data_geometry_level_{k}'] for k in range(4)]).mean())
        loss.backward()
        losses.append(loss)
    torch.testing.assert_close(losses[0], losses[1])
    for a, b in zip(model.parameters(), lookup.parameters()):
        if a.grad is not None:
            assert torch.isfinite(a.grad).all()
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-7, atol=1e-10)


def test_multistep_is_autonomous_and_differentiates_through_recursion():
    from experiments.joint_qcr_ising import multistep_closure_loss

    class LinearR:
        def __init__(self):
            self.a = torch.tensor(2., dtype=torch.float64, requires_grad=True)

        def advance_theory(self, g):
            return self.a * g

    model = LinearR()
    direct = torch.tensor([[[1.], [3.], [5.], [9.]]], dtype=torch.float64)
    loss, terms = multistep_closure_loss(model, direct)
    # Autonomous predictions are 2, 4, 8, not teacher-forced 2, 6, 10.
    torch.testing.assert_close(torch.stack(terms), torch.ones(3, dtype=torch.float64))
    loss.backward()
    torch.testing.assert_close(model.a.grad, torch.tensor(-34./3, dtype=torch.float64))


def test_rollout_weight_adds_exact_term():
    torch.manual_seed(2)
    model = RecurrentRGNetwork1D(levels=4, hidden=8, theory_dim=1).double()
    x = (2*torch.randint(0, 2, (4, 8, 32, 1))-1).double()
    base, _ = qcr_objective(model, x, pattern_fn=two_spin_distribution, objective_mode='data_geometry', rollout_weight=0.)
    total, terms = qcr_objective(model, x, pattern_fn=two_spin_distribution, objective_mode='data_geometry', rollout_weight=5.)
    torch.testing.assert_close(total-base, 5*terms['rollout'])
    torch.testing.assert_close(terms['rollout'], torch.stack([terms[f'rollout_level_{k}'] for k in range(1,4)]).mean())


def test_six_levels_cover_all_non_degenerate_scales():
    torch.manual_seed(91)
    model = RecurrentRGNetwork1D(levels=6, hidden=8, theory_dim=1).double()
    model.theory_encoder.binary_lookup = True
    model.rg_kernel.binary_lookup = True
    x = balanced_views((2*torch.randint(0,2,(4,8,128,1))-1).double(),8)
    assert [v.shape[2] for v in model.coarse_grain(x)] == [128,64,32,16,8,4]
    loss, terms = qcr_objective(model,x,pattern_fn=two_spin_distribution,
                                objective_mode='data_geometry',rollout_weight=5.)
    torch.testing.assert_close(terms['rollout'],torch.stack([terms[f'rollout_level_{k}'] for k in range(1,6)]).mean())
    torch.testing.assert_close(terms['separation'],torch.stack([terms[f'data_geometry_level_{k}'] for k in range(6)]).mean())
    loss.backward()
    for module in [model.theory_encoder,model.rg_kernel,model.theory_rg]:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())


def test_two_dimensional_geometry_and_rollout_gradients():
    from recurrent_rg.model_nd import RecurrentRGNetwork2D
    from experiments.joint_qcr_ising import plaquette_distribution
    torch.manual_seed(91)
    model=RecurrentRGNetwork2D(levels=2,hidden=8,theory_dim=1).double()
    fast=copy.deepcopy(model)
    fast.theory_encoder.binary_lookup=True
    fast.rg_kernel.binary_lookup=True
    x=balanced_views((2*torch.randint(0,2,(3,4,8,8,1))-1).double(),4)
    losses=[]
    for m in [model,fast]:
        loss,terms=qcr_objective(m,x,pattern_fn=plaquette_distribution,
                                objective_mode='data_geometry',rollout_weight=5.)
        loss.backward(); losses.append(loss)
        torch.testing.assert_close(terms['rollout'],terms['commutator'])
    torch.testing.assert_close(*losses)
    for a,b in zip(model.parameters(),fast.parameters()):
        if a.grad is not None:
            assert torch.isfinite(b.grad).all()
            torch.testing.assert_close(a.grad,b.grad,rtol=1e-7,atol=1e-10)
