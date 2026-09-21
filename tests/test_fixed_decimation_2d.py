import torch

from recurrent_rg.model_nd import RecurrentRGNetwork2D
from experiments.joint_qcr_ising2d_field import build_model


def test_fixed_2d_decimation_is_positive_subsampling():
    model = RecurrentRGNetwork2D(levels=3, hidden=8, theory_dim=2, learnable_kernel=False).double()
    x = (2*torch.randint(2, (3, 8, 16, 16, 1))-1).double()
    levels = model.coarse_grain(x)
    for k, level in enumerate(levels):
        torch.testing.assert_close(level, x[:, :, ::2**k, ::2**k])
    assert list(model.rg_kernel.parameters()) == []
    torch.testing.assert_close(model.rg_kernel(-x), -model.rg_kernel(x))


def test_fixed_control_preserves_qr_initialization_and_gradients():
    torch.manual_seed(91)
    learned = build_model(True)
    torch.manual_seed(91)
    fixed = build_model(False)
    for name in ('theory_encoder', 'theory_rg'):
        for a, b in zip(getattr(learned, name).parameters(), getattr(fixed, name).parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    from experiments.joint_qcr_ising import qcr_objective, plaquette_distribution
    x = (2*torch.randint(2, (3, 8, 16, 16, 1))-1).double()
    loss, _ = qcr_objective(fixed, x, pattern_fn=plaquette_distribution,
                           objective_mode='data_geometry', rollout_weight=5.,
                           kernel_noncollapse_mode='variance_floor')
    loss.backward()
    for module in (fixed.theory_encoder, fixed.theory_rg):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
