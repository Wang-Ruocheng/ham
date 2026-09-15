import torch

from recurrent_rg.adapters import (
    BinaryFieldHead,
    BinaryFlipConditional,
    CategoricalSiteConditional,
    CategoricalFieldHead,
    ContinuousFieldHead,
    DenoisingScoreConditional,
)
from recurrent_rg.geometry import cubic_geometry, square_geometry
from recurrent_rg.ising import DTYPE
from recurrent_rg.model_nd import RecurrentRGNetwork
from recurrent_rg.unsupervised_nd import unsupervised_objective


def test_cubic_geometry_extracts_patches_and_blocks() -> None:
    geometry = cubic_geometry(width=3, block_factor=2)
    x = torch.randn(2, 3, 4, 4, 4, 1, dtype=DTYPE)
    assert geometry.extract_patches(x).shape == (2, 3, 4, 4, 4, 27)
    assert geometry.extract_environment(x).shape == (2, 3, 4, 4, 4, 26)
    assert geometry.downsample(x).shape == (2, 3, 2, 2, 2, 1)


def test_three_dimensional_binary_model_and_vector_theory() -> None:
    hidden = 8
    model = RecurrentRGNetwork(
        cubic_geometry(),
        BinaryFieldHead(hidden),
        input_channels=1,
        levels=2,
        theory_dim=2,
        hidden=hidden,
        local_features=6,
    ).to(dtype=DTYPE)
    x = torch.where(
        torch.rand(3, 4, 4, 4, 4, 1) > 0.5,
        torch.tensor(1.0),
        torch.tensor(-1.0),
    ).to(DTYPE)
    trajectory = model(x)
    assert trajectory.configurations[1].shape == (3, 4, 2, 2, 2, 1)
    assert trajectory.direct_theory.shape == (3, 2, 2)
    assert trajectory.rollout_theory.shape == (3, 2, 2)
    information_field = model.rg_kernel.information_field(x)
    assert torch.allclose(
        information_field.abs(), torch.ones_like(information_field), atol=1e-12
    )


def test_field_heads_are_replaceable() -> None:
    geometry = square_geometry()
    categorical = RecurrentRGNetwork(
        geometry,
        CategoricalFieldHead(8, states=3),
        input_channels=3,
        levels=2,
        hidden=8,
    ).to(dtype=DTYPE)
    states = torch.randint(3, (2, 4, 4, 4))
    one_hot = torch.nn.functional.one_hot(states, 3).to(DTYPE)
    categorical_coarse = categorical.rg_kernel(one_hot)
    assert categorical_coarse.shape == (2, 4, 2, 2, 3)
    assert torch.allclose(categorical_coarse.sum(dim=-1), torch.ones_like(categorical_coarse[..., 0]))
    categorical_loss = CategoricalSiteConditional(3)(
        categorical.action_network, one_hot, torch.zeros(2, 1, dtype=DTYPE)
    )
    assert torch.isfinite(categorical_loss)

    continuous = RecurrentRGNetwork(
        geometry,
        ContinuousFieldHead(8, channels=2),
        input_channels=2,
        levels=2,
        hidden=8,
    ).to(dtype=DTYPE)
    continuous_coarse = continuous.rg_kernel(torch.randn(2, 4, 4, 4, 2, dtype=DTYPE))
    assert continuous_coarse.shape == (2, 4, 2, 2, 2)
    score_loss = DenoisingScoreConditional(0.2)(
        continuous.action_network,
        torch.randn(2, 4, 4, 4, 2, dtype=DTYPE),
        torch.zeros(2, 1, dtype=DTYPE),
    )
    assert torch.isfinite(score_loss)


def test_generic_objective_updates_all_core_modules() -> None:
    hidden = 8
    model = RecurrentRGNetwork(
        square_geometry(),
        BinaryFieldHead(hidden),
        input_channels=1,
        levels=2,
        theory_dim=2,
        hidden=hidden,
    ).to(dtype=DTYPE)
    x = torch.where(
        torch.rand(4, 4, 4, 4, 1) > 0.5,
        torch.tensor(1.0),
        torch.tensor(-1.0),
    ).to(DTYPE)
    loss, _ = unsupervised_objective(
        model, x, BinaryFlipConditional()
    )
    loss.backward()
    modules = (
        model.rg_kernel,
        model.theory_encoder,
        model.theory_rg,
        model.action_network,
    )
    assert all(any(parameter.grad is not None for parameter in module.parameters()) for module in modules)
