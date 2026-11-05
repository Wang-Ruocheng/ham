"""Synthetic CPU execution checks only; no historical data/model or sampling."""
from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from recurrent_rg.checkpoints import build_model, isolated_rng
from recurrent_rg.config import ModelConfig
from recurrent_rg.data.statistics import plaquette_distribution
from recurrent_rg.future_physics import binary2d_future_observables
from scripts.ising_large_size_probe_evaluation import (
    _row_seed, collect_chain, encoding_contract, exact_local_counts,
    finite_size_observables, module_shas, plaquette_counts, state_sha,
    theories_from_counts, projected_remaining_gpu_seconds, benchmark, build_parser,
)


@pytest.fixture
def model():
    with isolated_rng('cpu', 17):
        result = build_model(ModelConfig(dimension=2, width=3, block_factor=3,
            hidden=4, theory_dim=1, binary_lookup=True, affine_reconstruction=True,
            kernel_layout='nonoverlapping', kernel_sampling='bernoulli'), 2)
    return result.eval().requires_grad_(False)


def spins(samples=4, size=36):
    with isolated_rng('cpu', 29):
        return (2 * torch.randint(2, (samples, size, size)) - 1).to(torch.int8).numpy()


def test_streaming_q_exact_count_order_matches_dense_q_and_raw_decoder(model):
    data = spins()
    field = torch.from_numpy(data).double()[None, ..., None]
    counts = torch.stack([exact_local_counts(row[..., 0], model.geometry.patch_offsets())
                          for row in field[0]])
    protocol = encoding_contract(len(data))
    midpoint = len(data) // 2
    order = protocol['sample_partition']
    halves = torch.stack([counts[order[:midpoint]].sum(0), counts[order[midpoint:]].sum(0)])
    half, full = theories_from_counts(model.theory_encoder, halves[:, None])
    with torch.no_grad():
        expected = .5 * (model.theory_encoder(field[:, order[:midpoint]])
                        + model.theory_encoder(field[:, order[midpoint:]]))
        dense_full = model.theory_encoder(field)
    torch.testing.assert_close(half, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(full, dense_full, rtol=1e-12, atol=1e-12)
    assert counts.sum().item() == data.size
    torch.testing.assert_close(plaquette_counts(field[0, 0, ..., 0]).double() / 36 ** 2,
                              plaquette_distribution(field[:, :1])[0])


def test_nonlinear_head_not_averaged_per_sample(model):
    data = np.stack([np.full((12, 12), -1, np.int8), np.ones((12, 12), np.int8)] * 2)
    rows = torch.from_numpy(data).double()[None, ..., None]
    counts = torch.stack([exact_local_counts(field[..., 0], model.geometry.patch_offsets())
                          for field in rows[0]])
    halves = torch.stack([counts[:2].sum(0), counts[2:].sum(0)])
    _, full = theories_from_counts(model.theory_encoder, halves[:, None])
    with torch.no_grad():
        per_sample = torch.cat([model.theory_encoder(rows[:, i:i + 1]) for i in range(4)]).mean(0)
    assert (full[0] - per_sample).abs().max() > 1e-9


def _dense_sample_major(model, data, depth, seed):
    """Dense Q reference on the exact same sample-major C random stream."""
    samples = []
    with isolated_rng('cpu', seed), torch.no_grad():
        for array in data:
            levels = model.coarse_grain(torch.from_numpy(array).double()[None, None, ..., None],
                                        max_rg_steps=depth)
            samples.append(levels)
    return [torch.cat([sample[k] for sample in samples], 1) for k in range(depth + 1)]


def test_streaming_full_four_scales_one_c_path_rng_and_sha_unchanged(model):
    data, depth = spins(size=108), 3
    before, modules = state_sha(model), module_shas(model)
    rng = torch.get_rng_state().clone()
    calls = []
    hook = model.rg_kernel.register_forward_hook(lambda *args: calls.append(1))
    try:
        arrays, snapshots, protocol = collect_chain(model, data, 'cpu', row_index=2,
            draw_count=2, max_rg_steps=depth)
    finally:
        hook.remove()
    assert len(calls) == len(data) * depth * 2
    assert torch.equal(torch.get_rng_state(), rng)
    assert state_sha(model) == before and module_shas(model) == modules
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    assert model.levels == 3  # Trained depth2 was not changed by depth3 evaluation.
    assert arrays['direct_g'].shape == (2, 4, 1)
    assert arrays['p_observed'].shape == (2, 4, 16)
    assert [snapshots[k].shape for k in range(4)] == [(2, 108, 108), (2, 36, 36), (2, 12, 12), (2, 4, 4)]
    order, mid = protocol['sample_partition'], len(data) // 2
    for draw in range(2):
        levels = _dense_sample_major(model, data, depth, _row_seed(2, draw))
        with torch.no_grad():
            direct = torch.stack([.5 * (model.theory_encoder(level[:, order[:mid]])
                                + model.theory_encoder(level[:, order[mid:]])) for level in levels], 1)[0]
            full = torch.stack([model.theory_encoder(level) for level in levels], 1)[0]
            p = torch.stack([plaquette_distribution(level) for level in levels], 1)[0]
        np.testing.assert_allclose(arrays['direct_g'][draw], direct, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(arrays['full_direct_g'][draw], full, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(arrays['p_observed'][draw], p, rtol=1e-12, atol=1e-12)
        for scale in range(4):
            if draw == 0:
                np.testing.assert_array_equal(snapshots[scale], levels[scale][0, :2, ..., 0].sign().to(torch.int8))
    for prefix in ('', 'full_'):
        np.testing.assert_allclose(arrays[prefix + 'error_total'],
            arrays[prefix + 'error_R'] + arrays[prefix + 'error_repr'], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(arrays[prefix + 'R_rollout_g'][:, 0], arrays[prefix + 'direct_g'][:, 0])
    np.testing.assert_allclose(arrays['Q_full_minus_half'], arrays['full_direct_g'] - arrays['direct_g'])


def test_raw_decoder_outputs_not_clamped_or_renormalized(model):
    with torch.no_grad():
        model.theory_encoder.distribution_decoder.weight.zero_()
        model.theory_encoder.distribution_decoder.bias.fill_(-2.)
    arrays, _, _ = collect_chain(model, spins(size=12), 'cpu', draw_count=1, max_rg_steps=1)
    assert np.all(arrays['p_decoded_direct'] == -2.)
    assert np.all(arrays['p_decoded_rollout'].sum(-1) == -32.)
    np.testing.assert_allclose(arrays['p_observed'].sum(-1), 1.)


def test_rounded_distance_observables_match_old_adapter_where_defined():
    data = torch.from_numpy(spins(size=36)).double()
    values = [finite_size_observables(field)[0] for field in data]
    expected = binary2d_future_observables(data[None, ..., None])[0]
    torch.testing.assert_close(torch.stack(values).mean(0), expected, rtol=1e-12, atol=1e-12)
    assert finite_size_observables(torch.ones(40, 40, dtype=torch.float64))[1] == (7, 13)
    assert finite_size_observables(torch.ones(4, 4, dtype=torch.float64))[1] == (1, 1)


@pytest.mark.parametrize('samples,size,depth', [(3, 12, 1), (4, 10, 1), (4, 9, 1), (4, 36, -1)])
def test_invalid_counts_size_and_depth_rejected(model, samples, size, depth):
    with pytest.raises(ValueError):
        collect_chain(model, spins(samples, size), 'cpu', draw_count=1, max_rg_steps=depth)


def test_before_draw_guard_runs_before_any_sampling_and_fixed_seed_repeats(model):
    data = spins(size=12)
    calls = []
    def guard(row, draw):
        calls.append((row, draw))
        raise RuntimeError('hold')
    rng = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match='hold'):
        collect_chain(model, data, 'cpu', row_index=7, max_rg_steps=1, before_draw=guard)
    assert calls == [(7, 0)] and torch.equal(torch.get_rng_state(), rng)
    first, _, _ = collect_chain(model, data, 'cpu', draw_count=1, max_rg_steps=1)
    second, _, _ = collect_chain(model, data, 'cpu', draw_count=1, max_rg_steps=1)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


def test_projection_and_pinned_cli_contract():
    assert projected_remaining_gpu_seconds(3.) == 438.
    assert projected_remaining_gpu_seconds(3., remaining_units=0.) == 240.
    for value in (-1., float('nan'), float('inf')):
        with pytest.raises(ValueError):
            projected_remaining_gpu_seconds(value)
    args = build_parser().parse_args(['--root', '/new', 'benchmark', '--checkpoint', '/frozen/checkpoint.pt',
        '--snapshot-root', '/frozen/code', '--baseline-root', '/finite', '--device', 'cuda:2'])
    assert args.command == 'benchmark' and args.expected_q_sha.startswith('06e806')
    assert args.gpu_budget_seconds == 900.


def test_benchmark_only_timing_no_scientific_arrays_and_no_rerun(tmp_path, monkeypatch, model):
    import scripts.ising_large_size_probe_evaluation as probe
    monkeypatch.setattr(probe, '_load_pinned_model', lambda args: (None, model, torch.device('cpu')))
    monkeypatch.setattr(probe, '_large_condition', lambda *args: (np.stack([spins(size=108)] * 2),
        dict(input_sha256='synthetic_input_only')))
    args = SimpleNamespace(root=tmp_path, checkpoint=tmp_path / 'not_loaded.pt', snapshot_root=tmp_path,
        device='cpu', prior_gpu_seconds=0., gpu_budget_seconds=900.)
    before = state_sha(model)
    receipt = benchmark(args)
    assert receipt['status'] == 'complete'
    assert receipt['benchmark_only'] and not receipt['scientific_result']
    assert not receipt['physical_predictions_saved']
    assert receipt['physical_input_sha256'] == 'synthetic_input_only'
    assert receipt['projected_remaining_gpu_seconds'] > 240.
    assert receipt['projected_total_gpu_seconds'] > receipt['projected_remaining_gpu_seconds']
    assert state_sha(model) == before and receipt['all_parameters_frozen']
    assert sorted(path.name for path in tmp_path.iterdir()) == ['gpu_preflight.json']
    assert json.loads((tmp_path / 'gpu_preflight.json').read_text())['evaluated_shapes']['direct_g'] == [1, 4, 1]
    with pytest.raises(FileExistsError):
        benchmark(args)


def test_benchmark_budget_hold_not_a_training_failure(tmp_path, monkeypatch, model):
    import scripts.ising_large_size_probe_evaluation as probe
    monkeypatch.setattr(probe, '_load_pinned_model', lambda args: (None, model, torch.device('cpu')))
    monkeypatch.setattr(probe, '_large_condition', lambda *args: (np.stack([spins(size=108)] * 2),
        dict(input_sha256='synthetic_input_only')))
    args = SimpleNamespace(root=tmp_path, checkpoint=tmp_path / 'not_loaded.pt', snapshot_root=tmp_path,
        device='cpu', prior_gpu_seconds=899., gpu_budget_seconds=900.)
    receipt = benchmark(args)
    assert receipt['status'] == 'budget_hold' and receipt['requires_user_action']
    assert receipt['training_updates'] == 0 and receipt['optimizer_created'] is False


def test_nonbinary_inputs_never_accepted_as_histogram_data(model):
    data = spins(size=12).astype(float)
    data[0, 0, 0] = .75
    with pytest.raises(ValueError, match='hard finite'):
        collect_chain(model, data, 'cpu', draw_count=1, max_rg_steps=1)
