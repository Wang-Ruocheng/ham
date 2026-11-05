"""Synthetic-only tests: no remote data, physical sampling, or CUDA work."""
import inspect

import numpy as np
import pytest
import torch

from scripts import run_phi4_hopping_qonly_balance as runner


class ToyQ(torch.nn.Module):
    def __init__(self, features=7):
        super().__init__()
        self.encoder = torch.nn.Linear(1, 1).double()
        self.decoder = torch.nn.Linear(1, features).double()

    def forward(self, fields):
        return self.encoder(fields.flatten(1).mean(dim=1, keepdim=True))

    def decode(self, g):
        return self.decoder(g)

    @torch.no_grad()
    def normalize_decoder_(self):
        self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-12))


def fixture(seed=13):
    generator = torch.Generator().manual_seed(seed)
    levels = [torch.randn((3, 8, 2, 2, 1), generator=generator, dtype=torch.float64,
        requires_grad=True) for _ in range(4)]
    target = torch.randn((3, 4, 8, 7), generator=generator, dtype=torch.float64,
        requires_grad=True)
    return levels, target


def test_scalar_balance_retains_all_components_and_initial_scale():
    rng = np.random.default_rng(7)
    target = rng.normal(size=(8, 4, 12, 7)) * np.array([.03, .1, .5, 1.])[None, :, None, None]
    decoded = target.mean(axis=2) + .2
    result = runner.fit_scale_weights(target, decoded)
    assert result["weights"].shape == (4,)
    assert np.isfinite(result["weights"]).all()
    assert (result["weights"] > 0).all()
    residual = result["source_reconstruction_by_scale"]
    np.testing.assert_allclose(np.mean(result["weights"] * residual), residual.mean())
    assert result["denominator"].max() / result["denominator"].min() <= 20. + 1e-12
    np.testing.assert_allclose(result["target_std"]**2 * 7, result["variance"])


def test_block_noise_is_exact_heuristic_formula():
    rng = np.random.default_rng(17)
    p = rng.normal(size=(9, 4, 16, 5))
    result = runner.fit_scale_weights(p, p.mean(axis=2) + 1.)
    chunks = np.stack(np.split(p, 4, axis=2), axis=2).mean(axis=3)
    expected = chunks.var(axis=2, ddof=1).sum(axis=-1).mean(axis=0) / 4
    np.testing.assert_allclose(result["noise"], expected)


def test_equal_objective_is_exact_original_q_formula():
    torch.manual_seed(3)
    q = ToyQ()
    levels, targets = fixture()
    total, reconstruction, subset = runner.q_objective(q, levels, targets, np.ones(4))
    expected = 10 * reconstruction.mean() + 20 * subset[0] + 5 * subset[1:].mean()
    torch.testing.assert_close(total, expected)
    total.backward()
    assert all(parameter.grad is not None for parameter in q.parameters())
    assert all(level.grad is None for level in levels)
    assert targets.grad is None


def test_relative_objective_changes_only_scalar_reconstruction_weights():
    q = ToyQ()
    levels, targets = fixture()
    a, ra, sa = runner.q_objective(q, levels, targets, np.ones(4))
    weights = np.array([2., 1., .5, .25])
    b, rb, sb = runner.q_objective(q, levels, targets, weights)
    torch.testing.assert_close(ra, rb)
    torch.testing.assert_close(sa, sb)
    torch.testing.assert_close(b-a, 10 * ((torch.tensor(weights)-1)*ra).mean())


def test_schedule_is_paired_distinct_and_has_no_label_argument():
    a, b = (torch.Generator().manual_seed(192) for _ in range(2))
    for _ in range(8):
        ia, sa = runner.batch_schedule(40, 128, generator=a)
        ib, sb = runner.batch_schedule(40, 128, generator=b)
        assert torch.equal(ia, ib) and torch.equal(sa, sb)
        assert len(torch.unique(ia)) == 8 and len(torch.unique(sa)) == 32
    assert set(inspect.signature(runner.q_objective).parameters) == {"q", "levels", "per_field_targets", "weights"}
    assert "label" not in inspect.getsource(runner.update)


def test_post_update_projection_finite_and_gradients_cleared():
    q = ToyQ()
    levels, targets = fixture()
    optimizer = torch.optim.Adam(q.parameters(), lr=.001)
    for _ in range(3):
        runner.update(q, optimizer, levels, targets, np.ones(4))
    counts = runner.optimizer_counts(optimizer, q)
    assert counts == dict(parameter_tensors=4, minimum=3, maximum=3)
    assert all(parameter.grad is None for parameter in q.parameters())
    torch.testing.assert_close(q.decoder.weight.norm(dim=0), torch.ones(1, dtype=torch.float64))


def test_per_field_soft_targets_average_to_direct_subset_features():
    from recurrent_rg.phi4_aligned_qcr import soft_plaquette_features
    generator = torch.Generator().manual_seed(5)
    fields = torch.randn((2, 8, 4, 4, 1), generator=generator, dtype=torch.float64)
    per = soft_plaquette_features(fields.reshape(16, 1, 4, 4, 1), .93, chunk_size=128).reshape(2, 8, 625)
    for selected in (torch.tensor([1, 3, 6, 7]), torch.tensor([0, 2, 4, 5])):
        expected = soft_plaquette_features(fields[:, selected], .93, chunk_size=128)
        torch.testing.assert_close(per[:, selected].mean(dim=1), expected, rtol=1e-12, atol=1e-12)


def test_metrics_use_original_train_std_and_mean_not_test_fit():
    target = np.arange(24, dtype=float).reshape(3, 4, 2)
    decoded = target + .5
    result = runner.calculate_metrics(target, decoded, np.array([1., 2., 3., 4.]), np.zeros((4, 2)))
    np.testing.assert_allclose(result["rmse"], .5)
    np.testing.assert_allclose(result["nrmse"], .5 / np.arange(1., 5.))


def test_resource_guard_boundary():
    okay = dict(cpus=24, load1=30, mem_available_bytes=36*1024**3, io_full_avg10=19.99)
    assert runner.resource_ok(okay)
    assert not runner.resource_ok(dict(okay, io_full_avg10=20.))
    assert not runner.resource_ok(dict(okay, load1=30.01))
    assert not runner.resource_ok(dict(okay, mem_available_bytes=36*1024**3-1))


def test_invalid_shapes_and_weights_rejected():
    q = ToyQ()
    levels, targets = fixture()
    with pytest.raises(ValueError):
        runner.q_objective(q, levels[:3], targets, np.ones(4))
    with pytest.raises(ValueError):
        runner.q_objective(q, levels, targets, [1., 1., -1., 1.])
    with pytest.raises(ValueError):
        runner.batch_schedule(3, 128, generator=torch.Generator())
    with pytest.raises(ValueError):
        runner.fit_scale_weights(np.zeros((8,4,13,7)),np.zeros((8,4,7)))


def test_dry_import_does_not_import_frozen_model_or_initialize_cuda():
    source = inspect.getsource(runner)
    assert "from recurrent_rg" not in source
    assert "def snapshot_imports()" in source
    assert not torch.cuda.is_initialized()
