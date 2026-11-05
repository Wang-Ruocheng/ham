"""L108 size-control checks; synthetic fields only, no production sampling."""
import math

import pytest
import torch

from recurrent_rg.data.phi4 import Phi4Options, generation_plan
from recurrent_rg.phi4_qcr import Phi4QCR, continuous_objectives


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_large_size_and_authorized_integrator_do_not_reduce_physical_work():
    plan = generation_plan(Phi4Options(sizes=(108,), step_size=.04,
        leapfrog_steps=24, max_force_site_evaluations=27_000_000_000))
    assert plan["total_hmc_trajectories"] == 90_112
    assert plan["total_force_site_evaluations"] == 26_276_659_200
    assert plan["options"]["burn_in"] == 1024
    assert plan["options"]["samples"] == 128
    assert plan["options"]["chains"] == 4
    assert plan["options"]["spacing"] == 8
    assert plan["options"]["step_size"] * plan["options"]["leapfrog_steps"] == .96
    assert plan["labels_used_in_training"] is False


def test_large_geometry_preserves_independent_objectives_and_uniform_weights():
    torch.manual_seed(91)
    model = Phi4QCR(1., hidden=96, theory_dim=1)
    x = torch.randn(1, 4, 108, 108, 1, dtype=torch.float64)
    levels = model.coarse_grain(x, torch.Generator().manual_seed(5))
    assert [level.shape[2] for level in levels] == [108, 36, 12]
    losses = continuous_objectives(model, x, torch.Generator().manual_seed(6),
                                  information_pairs=128)
    metrics = losses[-1]
    assert all(math.isfinite(value) for value in metrics.values())
    assert metrics["r_loss"] == pytest.approx(
        5 * metrics["closure_one_step"] + 5 * metrics["closure_rollout"])
    for branch, loss in zip(("Q", "C", "R"), losses[:3]):
        for group in ("Q", "C", "R"):
            gradients = torch.autograd.grad(loss, tuple(getattr(model, group).parameters()),
                                            retain_graph=True, allow_unused=True)
            if group == branch:
                assert all(value is not None and torch.isfinite(value).all()
                           for value in gradients)
            else:
                assert all(value is None for value in gradients)
