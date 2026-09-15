import numpy as np
import torch
import torch.nn.functional as F

from recurrent_rg.ising import DTYPE
from recurrent_rg.ising2d import bond_sum_2d, sample_wolff_ensembles
from recurrent_rg.model_nd import RecurrentRGNetwork2D
from recurrent_rg.unsupervised2d import (
    pseudolikelihood_2d,
    unsupervised_objective_2d,
)


def tiny_batch_2d() -> torch.Tensor:
    configurations, _ = sample_wolff_ensembles(
        np.asarray([0.3, 0.55]),
        replicas=1,
        samples=6,
        size=8,
        seed=9,
        thermalization_sweeps=4,
    )
    return configurations


def test_wolff_shapes_and_ordered_bond_sum() -> None:
    x = tiny_batch_2d()
    assert x.shape == (2, 6, 8, 8, 1)
    ordered = torch.ones(1, 8, 8, dtype=DTYPE)
    assert torch.equal(bond_sum_2d(ordered), torch.tensor([128.0], dtype=DTYPE))


def test_2d_shapes_without_hard_coded_internal_symmetry() -> None:
    x = tiny_batch_2d()
    model = RecurrentRGNetwork2D(levels=3).to(dtype=DTYPE)
    trajectory = model(x)
    assert [level.shape for level in trajectory.configurations] == [
        (2, 6, 8, 8, 1),
        (2, 6, 4, 4, 1),
        (2, 6, 2, 2, 1),
    ]
    parameter_names = {name for name, _ in model.rg_kernel.named_parameters()}
    assert "local_encoder.0.bias" in parameter_names
    assert "field_head.projection.bias" in parameter_names


def test_2d_local_pseudolikelihood_matches_full_action() -> None:
    x = tiny_batch_2d()
    theory = torch.tensor([[0.2], [0.7]], dtype=DTYPE)
    model = RecurrentRGNetwork2D(levels=3).to(dtype=DTYPE)
    fast = pseudolikelihood_2d(model, x, theory)
    flipped = x.clone()
    flipped[:, :, 0, 0, 0] *= -1
    brute = F.softplus(
        model.action_network(x, theory) - model.action_network(flipped, theory)
    ).mean()
    assert torch.allclose(fast, brute, atol=1e-12, rtol=1e-12)


def test_2d_unsupervised_objective_updates_all_modules() -> None:
    x = tiny_batch_2d()
    model = RecurrentRGNetwork2D(levels=3).to(dtype=DTYPE)
    loss, _ = unsupervised_objective_2d(model, x)
    loss.backward()
    modules = (model.rg_kernel, model.theory_encoder, model.theory_rg, model.action_network)
    assert all(any(parameter.grad is not None for parameter in module.parameters()) for module in modules)
