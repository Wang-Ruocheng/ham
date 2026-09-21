import torch

from experiments._utils import balanced_views
from experiments.joint_qcr_ising import binary_kernel_noncollapse
from archive.qcr_ablations.legacy_objective import theory_diameter
from recurrent_rg.ising import DTYPE
from recurrent_rg.model_nd import RecurrentRGNetwork1D


def test_theory_diameter_is_vector_translation_invariant():
    g = torch.tensor([[0., 0.], [3., 4.]], dtype=DTYPE, requires_grad=True)
    assert theory_diameter(g).item() == 5.
    assert theory_diameter(g + torch.tensor([100., -100.])).item() == 5.
    assert theory_diameter(torch.tensor([[100., -100.], [100., -100.]], dtype=DTYPE)).item() == 0.
    theory_diameter(g).backward()
    assert torch.isfinite(g.grad).all()


def test_scalar_diameter_matches_range():
    g = torch.tensor([[2.], [-3.], [1.]], dtype=DTYPE)
    assert theory_diameter(g) == g.max() - g.min()


def test_kernel_noncollapse_has_gradient_at_constant_binary_output() -> None:
    model = RecurrentRGNetwork1D(levels=3, hidden=8).to(dtype=DTYPE)
    with torch.no_grad():
        for parameter in model.rg_kernel.parameters():
            parameter.zero_()
        model.rg_kernel.field_head.projection.bias.fill_(1.0)

    microscopic = torch.where(
        torch.rand(4, 8, 16, 1) > 0.5,
        torch.tensor(1.0, dtype=DTYPE),
        torch.tensor(-1.0, dtype=DTYPE),
    )
    loss = binary_kernel_noncollapse(model.coarse_grain(microscopic))
    loss.backward()

    gradient_norm = sum(
        parameter.grad.square().sum()
        for parameter in model.rg_kernel.parameters()
        if parameter.grad is not None
    ).sqrt()
    assert loss.item() == 1.0
    assert gradient_norm.item() > 0.0


def test_kernel_noncollapse_vanishes_for_balanced_decimation() -> None:
    model = RecurrentRGNetwork1D(
        levels=3, hidden=8, learnable_kernel=False
    ).to(dtype=DTYPE)
    configurations = torch.where(
        torch.rand(3, 8, 16, 1) > 0.5,
        torch.tensor(1.0, dtype=DTYPE),
        torch.tensor(-1.0, dtype=DTYPE),
    )
    balanced = balanced_views(configurations, samples=8)
    loss = binary_kernel_noncollapse(model.coarse_grain(balanced))
    assert loss.item() == 0.0
