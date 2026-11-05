"""Synthetic estimator/gradient/RNG checks; never read a physical cache."""
import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from recurrent_rg.checkpoints import build_model, isolated_rng
from recurrent_rg.config import ModelConfig
from recurrent_rg.encoding import half_subset_theory, split_theory_views
from scripts.ising_qcr_evaluation import (
    FULL_ENSEMBLE_ESTIMATOR, HALF_SUBSET_ESTIMATOR, collect_saved_encodings,
    collection_estimator, encoding_contract, require_matching_estimators,
    validate_encoding_arrays,
)


def binary(shape):
    generator = torch.Generator().manual_seed(sum(shape) + 17)
    return (2 * torch.randint(2, shape, generator=generator) - 1).double()


@pytest.mark.parametrize('shared', [False, True])
@pytest.mark.parametrize('detached', [False, True])
def test_shared_training_estimator_preserves_old_values_vjps_and_rng(shared, detached):
    torch.set_num_threads(1)
    torch.manual_seed(19)
    reference = build_model(ModelConfig(dimension=2, block_factor=3, hidden=5,
        theory_dim=2, binary_lookup=True, affine_reconstruction=True), 2).double()
    reference.detach_kernel_representation = detached
    reference.share_binary_q_tables = shared
    current = copy.deepcopy(reference)
    x = [binary((2, 4, side, side, 1)).requires_grad_() for side in (9, 3, 1)]
    y = [value.detach().clone().requires_grad_() for value in x]
    levels = [value.detach() for value in x] if detached else x
    first = torch.stack([reference.theory_encoder(level[:, :2]) for level in levels], 1)
    second = torch.stack([reference.theory_encoder(level[:, 2:]) for level in levels], 1)
    rng = torch.get_rng_state().clone()
    got_first, got_second = split_theory_views(current, y)
    assert torch.equal(rng, torch.get_rng_state())
    for expected, got in ((first, got_first), (second, got_second)):
        torch.testing.assert_close(expected, got, atol=1e-12, rtol=1e-12)
    (first.square().sum() + 2 * second.square().sum()).backward()
    (got_first.square().sum() + 2 * got_second.square().sum()).backward()
    for left, right in zip(reference.parameters(), current.parameters()):
        assert (left.grad is None) == (right.grad is None)
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, atol=1e-11, rtol=1e-10)
    for left, right in zip(x, y):
        assert (left.grad is None) == (right.grad is None) == detached
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, atol=1e-11, rtol=1e-10)
    assert all(parameter.grad is None for parameter in current.rg_kernel.parameters())


class MomentEncoder(torch.nn.Module):
    def __init__(self, nonlinear=True):
        super().__init__()
        self.nonlinear = nonlinear

    def forward(self, value):
        mean = value.mean(tuple(range(1, value.ndim))).unsqueeze(-1)
        return mean.square() if self.nonlinear else mean


def test_nonlinear_Q_has_real_finite_sample_gap_but_affine_Q_does_not():
    level = torch.tensor([[-1., -1., 1., 1.]], dtype=torch.double).reshape(1, 4, 1, 1, 1)
    for nonlinear in (True, False):
        model = SimpleNamespace(theory_encoder=MomentEncoder(nonlinear))
        half = half_subset_theory(model, [level])[:, 0]
        full = model.theory_encoder(level)
        torch.testing.assert_close(half - full, torch.ones_like(full) if nonlinear else torch.zeros_like(full))


class ToyModel:
    """Stochastic b3 trajectory with no learned parameters and cheap Q."""
    def __init__(self):
        self.theory_encoder = MomentEncoder()
        self.calls = 0

    def coarse_grain(self, batch):
        self.calls += 1
        levels = [batch]
        for _ in range(2):
            probabilities = .5 + .2 * levels[-1][:, :, ::3, ::3]
            levels.append((2 * torch.bernoulli(probabilities) - 1).double())
        return levels

    def advance_theory(self, value):
        return .3 + .7 * value + .1 * value.square()


def datasets():
    return [(split, SimpleNamespace(configurations=binary((3, 4, 108, 108, 1)),
                couplings=torch.zeros(3, 7), group_ids=torch.arange(3)))
            for split in ('train', 'test')]


def test_collection_reuses_C_trajectory_and_preserves_full_Q_values_and_RNG():
    data = datasets()
    torch.manual_seed(82)
    saved_rng = torch.get_rng_state().clone()
    model = ToyModel()
    with isolated_rng('cpu', 2026101723), torch.no_grad():
        arrays, protocol = collect_saved_encodings(model, data, 'cpu', draw_count=2, batch_size=2)
        after = torch.get_rng_state().clone()
    assert torch.equal(saved_rng, torch.get_rng_state())
    assert model.calls == 2 * 2 * 2  # splits * batches * draws, not once per Q estimator
    reference = ToyModel()
    with isolated_rng('cpu', 2026101723), torch.no_grad():
        for split, dataset in data:
            chunks = []
            for start in range(0, len(dataset.configurations), 2):
                draws = []
                for _ in range(2):
                    levels = reference.coarse_grain(dataset.configurations[start:start + 2])
                    draws.append(torch.stack([reference.theory_encoder(level) for level in levels], 1))
                chunks.append(torch.stack(draws, 1))
            np.testing.assert_allclose(arrays[f'full_ensemble_direct_draws_{split}'], torch.cat(chunks).numpy())
        assert torch.equal(after, torch.get_rng_state())
    metrics = dict(schema_version=2, samples=4, C_draws=2,
                   encoding_estimator=protocol['estimator'], encoding_protocol=protocol)
    assert validate_encoding_arrays(metrics, arrays) == HALF_SUBSET_ESTIMATOR
    assert any(np.any(arrays[f'Q_full_minus_half_draws_{split}'] != 0) for split in ('train', 'test'))
    for split in ('train', 'test'):
        direct, mapped, rollout = (arrays[f'{key}_draws_{split}'] for key in ('direct', 'R_g', 'rollout'))
        np.testing.assert_allclose(mapped, .3 + .7 * direct + .1 * direct ** 2)
        np.testing.assert_allclose(rollout[:, :, 0], direct[:, :, 0])
        np.testing.assert_allclose(rollout[:, :, 1:], .3 + .7 * rollout[:, :, :-1] + .1 * rollout[:, :, :-1] ** 2)


def modern_metrics(samples=4):
    protocol = encoding_contract(samples)
    return dict(schema_version=2, samples=samples, C_draws=2,
                encoding_estimator=protocol['estimator'], encoding_protocol=protocol)


def test_fixed_partition_uses_only_local_rng_and_is_reproducible():
    before = random.getstate()
    a, b = encoding_contract(64), encoding_contract(64)
    assert a == b and random.getstate() == before
    assert sorted(a['sample_partition']) == list(range(64))
    assert a['half_samples'] == 32
    assert 'not an average' in a['qualification']


@pytest.mark.parametrize('samples', [0, 1, 3, True, 4., -2])
def test_invalid_sample_counts_rejected(samples):
    with pytest.raises(ValueError): encoding_contract(samples)


@pytest.mark.parametrize('change', ['missing_estimator', 'unknown', 'missing_protocol', 'wrong_samples',
                                   'duplicate_partition', 'bool_partition', 'missing_partition'])
def test_modern_missing_or_invalid_contract_never_falls_back_to_legacy(change):
    metrics = modern_metrics()
    if change == 'missing_estimator': metrics.pop('encoding_estimator')
    elif change == 'unknown': metrics['encoding_estimator'] = 'unknown'
    elif change == 'missing_protocol': metrics.pop('encoding_protocol')
    elif change == 'wrong_samples': metrics['samples'] = 6
    elif change == 'duplicate_partition': metrics['encoding_protocol']['sample_partition'] = [0] * 4
    elif change == 'bool_partition': metrics['encoding_protocol']['sample_partition'][0] = True
    elif change == 'missing_partition': metrics['encoding_protocol'].pop('sample_partition')
    with pytest.raises(ValueError): collection_estimator(metrics)


def test_comparison_distinguishes_legacy_modern_sample_counts_and_partitions():
    assert require_matching_estimators([{}, {'samples': 64}]) == FULL_ENSEMBLE_ESTIMATOR
    assert require_matching_estimators([modern_metrics(), modern_metrics()]) == HALF_SUBSET_ESTIMATOR
    for rows in ([{}, modern_metrics()], [dict(samples=32), dict(samples=64)],
                 [modern_metrics(), modern_metrics(6)]):
        with pytest.raises(ValueError): require_matching_estimators(rows)
    other = modern_metrics()
    other['encoding_protocol']['sample_partition'] = list(reversed(other['encoding_protocol']['sample_partition']))
    with pytest.raises(ValueError): require_matching_estimators([modern_metrics(), other])


@pytest.mark.parametrize('shapes', [[], [(2, 3, 1)], [(2, 4, 1), (2, 6, 1)]])
def test_halves_reject_empty_odd_or_inconsistent_ensembles(shapes):
    with pytest.raises(ValueError):
        split_theory_views(SimpleNamespace(theory_encoder=MomentEncoder()), [torch.ones(shape) for shape in shapes])
