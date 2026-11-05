"""Frozen, small-sample b3 size/depth probe; never train or search for roots.

The production CLI is deliberately pinned to one checkpoint and a frozen code
snapshot. Q is evaluated from exact local-pattern counts, not from averages of
nonlinear per-configuration Q outputs. No old physical/MI cache is opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from scripts.ising_qcr_evaluation import encoding_contract


CHECKPOINT_SHA256 = '0687c8f4663810688833c54de1e2e6d2424fa9ba5a2c2dbc43042549d4303f4d'
MODULE_SHA256 = dict(Q='06e806078424ef4b44b337299d1abe50d2d54415026be1ebff20e66db852844d',
    C='575a473561cbeb92bbdd98de00cc4a43055ab451d65f4307b44088a71c0af42a',
    R='dc5d4c08ff5a0e0d3e94ba7fd175ecc6427bfcb20afc9c5d18dbbe0a5b6118fe')
SEED = 20261018
BASELINE_CONDITIONS = ((0, .42), (14, .44), (7, .46))
OBSERVABLE_NAMES = ('轴向近L/6关联', '轴向近L/3关联', '最低非零轴向结构因子',
                    '第二非零轴向结构因子', '最低对角结构因子')
RNG_LAYOUT = 'v1: independent row/draw seed; sample-major, then increasing k; one C trajectory reused for Q, motifs and diagnostics'


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def state_sha(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        if isinstance(value, torch.Tensor):
            digest.update(name.encode()); digest.update(str(value.dtype).encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def module_shas(model):
    return dict(Q=state_sha(model.theory_encoder), C=state_sha(model.rg_kernel), R=state_sha(model.theory_rg))


def save_json(path, value, *, new=False):
    path = Path(path)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if new:
        with path.open('x') as stream:
            stream.write(text)
    else:
        temporary = path.with_name(path.name + '.writing')
        with temporary.open('x') as stream:
            stream.write(text)
        os.replace(temporary, path)


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def peak_bytes(device):
    return torch.cuda.max_memory_allocated(device) if torch.device(device).type == 'cuda' else 0


def peak_reserved_bytes(device):
    return torch.cuda.max_memory_reserved(device) if torch.device(device).type == 'cuda' else 0


def projected_remaining_gpu_seconds(chain_draw_seconds, *, remaining_units=33.):
    """Wall time, conservatively charged as whole-card time, including reserve."""
    if (not np.isfinite(chain_draw_seconds) or chain_draw_seconds < 0
            or not np.isfinite(remaining_units) or remaining_units < 0):
        raise ValueError('Require finite nonnegative execution estimates')
    return 2 * (remaining_units * chain_draw_seconds + 120)


def validate_binary_square(spins):
    if spins.ndim != 2 or spins.shape[0] != spins.shape[1] or spins.shape[0] < 2:
        raise ValueError('Require one nonempty square binary configuration')
    if not bool(torch.isfinite(spins).all()) or not bool(((spins.abs() - 1).abs() < 1e-12).all()):
        raise ValueError('Require hard finite +/-1 spins')


def exact_local_counts(spins, offsets):
    """Integer counts without constructing a site-by-9-patch tensor."""
    code = torch.zeros_like(spins, dtype=torch.int64)
    for bit, (row, column) in enumerate(offsets):
        code.add_(torch.roll(spins, shifts=(-row, -column), dims=(0, 1)).gt(0).long() << bit)
    return torch.bincount(code.flatten(), minlength=2 ** len(offsets))


def plaquette_counts(spins):
    # Match recurrent_rg.data.statistics.plaquette_distribution, including order.
    bits = spins.gt(0).long()
    code = (8 * bits + 4 * torch.roll(bits, -1, 0) + 2 * torch.roll(bits, -1, 1)
            + torch.roll(bits, shifts=(-1, -1), dims=(0, 1)))
    return torch.bincount(code.flatten(), minlength=16)


def finite_size_observables(spins):
    """Five measured diagnostics for any L>=4; NOT an R physical readout.

    Distances use Python round(L/6), round(L/3), with a minimum of one.
    The actual integer distances are always saved (at L=4 they coincide).
    Structure factors are normalized by V^2 and averaged over axis/sign pairs.
    """
    size = spins.shape[0]
    distances = tuple(max(1, round(size / divisor)) for divisor in (6, 3))
    values = [((spins * torch.roll(spins, -distance, 0)).mean()
               + (spins * torch.roll(spins, -distance, 1)).mean()) / 2
              for distance in distances]
    q = 2 * torch.pi * torch.arange(size, device=spins.device, dtype=spins.dtype) / size
    row, column = spins.sum(1), spins.sum(0)
    for harmonic in (1, 2):
        powers = [((vector * torch.cos(harmonic * q)).sum().square()
                   + (vector * torch.sin(harmonic * q)).sum().square())
                  for vector in (row, column)]
        values.append(sum(powers) / (2 * size ** 4))
    powers = []
    for sign in (1, -1):
        phase = q[:, None] + sign * q[None, :]
        powers.append((spins * torch.cos(phase)).sum().square()
                      + (spins * torch.sin(phase)).sum().square())
    values.append(sum(powers) / (2 * size ** 4))
    return torch.stack(values), distances


def theories_from_counts(encoder, half_counts):
    """Pool exact frequencies before the nonlinear head, including full-Q."""
    from recurrent_rg.binary_pooling import binary_pattern_table
    reference = next(encoder.parameters())
    _, patterns = binary_pattern_table(encoder.geometry.patch_volume, reference)
    frequencies = half_counts.to(reference.dtype) / half_counts.sum(-1, keepdim=True)
    table = encoder.local_mlp(patterns)
    half_values = encoder.head(frequencies @ table)
    half_mean = half_values.mean(0)
    full = encoder.head(frequencies.mean(0) @ table)
    return half_mean, full


def _row_seed(row_index, draw_index, *, benchmark=False):
    return SEED + (1_000_000_000 if benchmark else 0) + row_index * 100_000 + draw_index * 100


@torch.no_grad()
def collect_chain(model, configurations, device, *, row_index=0, draw_count=5,
                  max_rg_steps=3, seed_override=None, before_draw=None, save_snapshots=True):
    """Exact streaming evaluation of one physical chain, isolated from RNG.

    Each sample runs C^0..C^depth in sequence. This declared new C RNG layout
    differs from dense batch-major historical collection; all estimators and
    diagnostics here share precisely the same sampled C trajectory.
    """
    from recurrent_rg.checkpoints import isolated_rng
    samples = len(configurations)
    protocol = encoding_contract(samples)
    if type(draw_count) is not int or draw_count < 1 or type(max_rg_steps) is not int or max_rg_steps < 0:
        raise ValueError('Require positive draw_count and nonnegative depth')
    if (model.geometry.dimensions != 2 or model.geometry.widths != (3, 3)
            or model.geometry.factors != (3, 3)
            or getattr(model.rg_kernel, 'kernel_layout', None) != 'nonoverlapping'):
        raise ValueError('This probe requires the frozen nonoverlapping b3 binary QCR model')
    if not isinstance(model.theory_encoder.distribution_decoder, torch.nn.Linear):
        raise ValueError('Require the existing raw affine motif decoder')
    size = int(configurations.shape[-1])
    if tuple(configurations.shape[1:]) != (size, size) or size % (3 ** max_rg_steps):
        raise ValueError('Every evaluated b3 level must have integral square size')
    if size // (3 ** max_rg_steps) < 4:
        raise ValueError('Last evaluation size must be at least four')
    levels = max_rg_steps + 1
    halves = {sample: half for half in (0, 1)
              for sample in protocol['sample_partition'][half * (samples // 2):(half + 1) * (samples // 2)]}
    result, snapshots = {}, {}
    for draw in range(draw_count):
        if before_draw is not None:
            before_draw(row_index, draw)
        counts = torch.zeros((levels, 2, 512), dtype=torch.int64, device=device)
        observed = torch.zeros((levels, 16), dtype=torch.int64, device=device)
        diagnostics = torch.zeros((levels, 5), dtype=torch.float64, device=device)
        seed = _row_seed(row_index, draw) if seed_override is None else seed_override + draw
        with isolated_rng(device, seed):
            for sample in range(samples):
                # Copy memmap slices before conversion, never retain a writable view.
                array = np.array(configurations[sample], copy=True)
                field = torch.as_tensor(array, device=device, dtype=torch.float64)
                validate_binary_square(field)
                for scale in range(levels):
                    counts[scale, halves[sample]].add_(exact_local_counts(field, model.geometry.patch_offsets()))
                    observed[scale].add_(plaquette_counts(field))
                    values, _ = finite_size_observables(field)
                    diagnostics[scale].add_(values)
                    if save_snapshots and draw == 0 and sample < 2:
                        snapshots.setdefault(scale, []).append(field.sign().to(torch.int8).cpu().numpy())
                    if scale < max_rg_steps:
                        field = model.rg_kernel(field[None, None, ..., None])[0, 0, ..., 0]
        direct, full = theories_from_counts(model.theory_encoder, counts.permute(1, 0, 2))
        p = observed.to(torch.float64) / observed.sum(-1, keepdim=True)
        values = dict(p_observed=p, long_range_observed=diagnostics / samples)
        for prefix, g in (('', direct), ('full_', full)):
            rollout = [g[0]]
            for _ in range(max_rg_steps):
                rollout.append(model.advance_theory(rollout[-1]))
            rollout = torch.stack(rollout)
            decoder = model.theory_encoder.distribution_decoder
            decoded_direct, decoded_rollout = decoder(g), decoder(rollout)
            values.update({prefix + 'direct_g': g, prefix + 'R_rollout_g': rollout,
                prefix + 'R_one_step_g': model.advance_theory(g),
                prefix + 'p_decoded_direct': decoded_direct,
                prefix + 'p_decoded_rollout': decoded_rollout,
                prefix + 'error_total': decoded_rollout - p,
                prefix + 'error_R': decoded_rollout - decoded_direct,
                prefix + 'error_repr': decoded_direct - p})
        values['Q_full_minus_half'] = full - direct
        for key, value in values.items():
            if not bool(torch.isfinite(value).all()):
                raise RuntimeError('Nonfinite probe statistic: ' + key)
            result.setdefault(key, []).append(value.cpu().numpy())
    return ({key: np.stack(value) for key, value in result.items()},
            {scale: np.stack(value) for scale, value in snapshots.items()}, protocol)


def _load_pinned_model(args):
    from recurrent_rg import checkpoints
    snapshot = Path(args.snapshot_root).resolve()
    if not Path(checkpoints.__file__).resolve().is_relative_to(snapshot):
        raise RuntimeError('Model implementation is not imported from the frozen training snapshot')
    checkpoint = Path(args.checkpoint).resolve()
    if file_sha(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError('Pinned checkpoint SHA mismatch')
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda') or (device.type == 'cuda' and device.index != 2):
        raise ValueError('The production GPU is pinned to cuda:2')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        # Bound this new process's allocator only; never alter other processes.
        # RTX3090: 25% is approximately 6 GiB, below the total 8 GiB process cap.
        torch.cuda.set_per_process_memory_fraction(.25, device)
        torch.cuda.reset_peak_memory_stats(device)
    loaded = checkpoints.load_run(checkpoint)
    if loaded.payload.get('epoch') != 6000:
        raise RuntimeError('Require the exact completed 6000-step source')
    model = loaded.model.to(device).eval().requires_grad_(False)
    if getattr(model.rg_kernel.field_head, 'sampling_mode', None) != 'bernoulli':
        raise ValueError('Frozen C must retain its trained Bernoulli sampling rule')
    expected = dict(Q=args.expected_q_sha, C=args.expected_c_sha, R=args.expected_r_sha)
    if module_shas(model) != expected:
        raise RuntimeError('Pinned Q/C/R SHA mismatch')
    return loaded, model, device


def _large_condition(root, index, coupling):
    directory = root / 'data' / f'c{index:03d}'
    sidecar = json.loads((directory / 'sidecar.json').read_text())
    path = directory / 'spins.npy'
    if (sidecar.get('size') != 1080 or sidecar.get('K0') != coupling
            or sidecar.get('status') != 'complete'):
        raise ValueError('New sampling condition metadata mismatch')
    if file_sha(path) != sidecar['spins_sha256']:
        raise RuntimeError('New physical data SHA mismatch')
    arrays = np.load(path, mmap_mode='r', allow_pickle=False)
    if arrays.shape != (2, 64, 1080, 1080) or arrays.dtype != np.int8:
        raise ValueError('Expected the authorized 2 chains x 64 configurations')
    return arrays, dict(sidecar_path=str(Path('data') / directory.name / 'sidecar.json'),
        sampling_quality=sidecar.get('quality'), input_sha256=sidecar['spins_sha256'])


def _baseline_condition(root, index, coupling):
    path = root / 'data' / 'conditions' / f'c{index:03d}' / 'data.pt'
    sidecar_path = path.with_name('sidecar.json')
    sidecar = json.loads(sidecar_path.read_text())
    condition = sidecar.get('condition', {})
    delta = sidecar.get('delta6')
    if (condition.get('K0') != coupling or condition.get('size') != 108
            or not isinstance(delta, list) or len(delta) != 6 or any(value != 0 for value in delta)
            or sidecar.get('status') != 'complete' or sidecar.get('smoke') is not False):
        raise ValueError('Read only the explicitly named zero-perturbation L108 baseline')
    if file_sha(path) != sidecar['data_sha256']:
        raise RuntimeError('Baseline physical data SHA mismatch')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    configurations = payload['configurations']
    if configurations.shape != (12, 128, 108, 108) or configurations.dtype != torch.int8:
        raise ValueError('Unexpected finite-perturbation baseline payload')
    # payload.block_factor describes the historical sampler, NOT this model C.
    arrays = configurations[:2, :64].numpy().copy()
    return arrays, dict(source_sidecar_sha256=file_sha(sidecar_path),
        source_condition=f'c{index:03d}', sampling_quality=sidecar.get('quality'),
        input_sha256=sidecar['data_sha256'], selection='replicas [0,1], first 64 only')


def _common_receipt(args, model, before, modules, started):
    return dict(schema_version=1, checkpoint_sha256=CHECKPOINT_SHA256,
        checkpoint_path=str(Path(args.checkpoint).resolve()), snapshot_root=str(Path(args.snapshot_root).resolve()),
        model_sha256_before=before, model_sha256_after=state_sha(model),
        module_sha256_before=modules, module_sha256_after=module_shas(model),
        training_updates=0, mutual_information_evaluations=0,
        all_parameters_frozen=all(not p.requires_grad for p in model.parameters()),
        all_grad_none=all(p.grad is None for p in model.parameters()),
        optimizer_created=False, training_depth_unchanged=model.levels - 1,
        evaluation_max_rg_steps=3, rng_layout=RNG_LAYOUT, seed=SEED,
        elapsed_seconds=time.perf_counter() - started, peak_gpu_allocated_bytes=peak_bytes(args.device),
        peak_gpu_reserved_bytes=peak_reserved_bytes(args.device),
        thermal_nu=None)


def benchmark(args):
    root = Path(args.root).resolve()
    out = root / 'gpu_preflight.json'
    if out.exists():
        raise FileExistsError('Refuse to rerun or overwrite a GPU preflight')
    started = time.perf_counter()
    _, model, device = _load_pinned_model(args)
    before, modules = state_sha(model), module_shas(model)
    configurations, source = _large_condition(root, 0, .42)
    synchronize(device); measured_start = time.perf_counter()
    values, _, _ = collect_chain(model, configurations[0], device,
        row_index=0, draw_count=1, seed_override=_row_seed(0, 0, benchmark=True), save_snapshots=False)
    synchronize(device); duration = time.perf_counter() - measured_start
    shapes = {key: list(value.shape) for key, value in values.items()}
    del values  # Execution check only: predictions are neither saved nor reused.
    receipt = _common_receipt(args, model, before, modules, started)
    actual = receipt['elapsed_seconds']
    remaining = projected_remaining_gpu_seconds(duration)
    projected = args.prior_gpu_seconds + actual + remaining
    passed = (projected <= args.gpu_budget_seconds
              and max(peak_bytes(device), peak_reserved_bytes(device)) <= 8 * 1024 ** 3
              and receipt['model_sha256_before'] == receipt['model_sha256_after']
              and receipt['module_sha256_before'] == receipt['module_sha256_after']
              and receipt['all_grad_none'] and receipt['all_parameters_frozen'])
    receipt.update(status='complete' if passed else 'budget_hold', workflow_complete=passed,
        benchmark_only=True, scientific_result=False, physical_predictions_saved=False,
        measured_chain_draw_seconds=duration, projected_total_gpu_seconds=projected,
        projected_remaining_gpu_seconds=remaining, physical_input_sha256=source['input_sha256'],
        gpu_budget_seconds=args.gpu_budget_seconds, prior_gpu_seconds=args.prior_gpu_seconds,
        projection='prior + benchmark actual + 2*(30 full large chain draws + 3 large-draw-equivalent small allowance + 120s load/write reserve)',
        evaluated_shapes=shapes, requires_user_action=not passed)
    save_json(out, receipt, new=True)
    return receipt


def collect(args):
    root = Path(args.root).resolve()
    gate = json.loads((root / 'gpu_preflight.json').read_text())
    if (gate.get('status') != 'complete' or not gate.get('benchmark_only')
            or gate.get('checkpoint_sha256') != CHECKPOINT_SHA256):
        raise ValueError('An accepted execution-only GPU preflight is required')
    out = root / 'collection'; out.mkdir(exist_ok=False)
    status_path = out / 'status.json'
    save_json(status_path, dict(status='running', workflow_complete=False), new=True)
    started = time.perf_counter()
    try:
        _, model, device = _load_pinned_model(args)
        before, modules = state_sha(model), module_shas(model)
        collected, snapshots, rows = {}, {}, []
        total_units, completed_units = 33., 0.
        prior = max(args.prior_gpu_seconds, gate['prior_gpu_seconds'] + gate['elapsed_seconds'])
        def budget_check(row, draw):
            synchronize(device)
            projected = prior + time.perf_counter() - started + projected_remaining_gpu_seconds(
                gate['measured_chain_draw_seconds'], remaining_units=max(0., total_units - completed_units))
            if projected > args.gpu_budget_seconds:
                raise RuntimeError('budget_hold: actual + conservative remaining GPU projection exceeds budget')
            if max(peak_bytes(device), peak_reserved_bytes(device)) > 8 * 1024 ** 3:
                raise RuntimeError('budget_hold: GPU peak exceeds authorized 8 GiB cap')
        for dataset_index, (size, baseline) in enumerate(((1080, False), (108, True))):
            for condition_index, (baseline_index, coupling) in enumerate(BASELINE_CONDITIONS):
                configurations, provenance = (_baseline_condition(Path(args.baseline_root).resolve(), baseline_index, coupling)
                    if baseline else _large_condition(root, condition_index, coupling))
                for replica in range(2):
                    row = len(rows)
                    def before_draw(index, draw):
                        nonlocal completed_units
                        # The previous draw completed before the next callback.
                        if draw:
                            completed_units += .1 if baseline else 1.
                        budget_check(index, draw)
                        save_json(status_path, dict(status='running', workflow_complete=False,
                            current=dict(row=index, size=size, K0=coupling, replica=replica, C_draw=draw)))
                    values, images, protocol = collect_chain(model, configurations[replica], device,
                        row_index=row, before_draw=before_draw)
                    completed_units += .1 if baseline else 1.
                    for key, value in values.items():
                        collected.setdefault(key, []).append(value)
                    for scale, image in images.items():
                        snapshots[f'row_{row:03d}_scale_{scale}'] = image
                    rows.append(dict(row=row, size=size, K0=coupling, replica=replica,
                        physical_chain_independent=True, C_draws_not_independent_physical_chains=True, **provenance))
                del configurations
        receipt = _common_receipt(args, model, before, modules, started)
        if (receipt['model_sha256_before'] != receipt['model_sha256_after']
                or receipt['module_sha256_before'] != receipt['module_sha256_after']
                or not receipt['all_grad_none'] or not receipt['all_parameters_frozen']):
            raise RuntimeError('Frozen model/gradient fingerprint changed')
        arrays = {key: np.stack(value) for key, value in collected.items()}
        arrays.update(row_size0=np.array([row['size'] for row in rows]),
            row_K0=np.array([row['K0'] for row in rows]), row_replica=np.array([row['replica'] for row in rows]),
            scales=np.array([[row['size'] // 3 ** k for k in range(4)] for row in rows]))
        arrays['distances'] = np.array([[[max(1, round(size / d)) for d in (6, 3)] for size in levels]
                                        for levels in arrays['scales']])
        for key, value in arrays.items():
            if not np.isfinite(value).all():
                raise RuntimeError('Nonfinite output: ' + key)
        np.savez_compressed(out / 'arrays.npz', **arrays)
        np.savez_compressed(out / 'snapshots.npz', **snapshots)
        receipt.update(status='complete', workflow_complete=True, scientific_result=True,
            samples=64, C_draws=5, encoding_estimator=protocol['estimator'], encoding_protocol=protocol,
            arrays_sha256=file_sha(out / 'arrays.npz'), snapshots_sha256=file_sha(out / 'snapshots.npz'),
            dataset_rows=rows, axis_order='physical chain row, C draw, scale k, coordinate or motif',
            decoder='Existing raw affine 16-motif decoder; no clamp, renormalization or softmax; save negative probabilities and mass error',
            error_decomposition='Vectors: total=decoded rollout-observed=R error+representation error; RMSE terms do not add',
            long_range_protocol=dict(version='rounded_distances_v1', names=OBSERVABLE_NAMES,
                rule='Python round(L/6), round(L/3), minimum1; actual distances saved',
                interpretation='Measurements on C trajectories only; NOT predictions by R', structure_factor='nonzero Fourier modes, V^2 normalization'),
            limitations=['Two physical chains per K and size only; exploratory, not equilibrium certification',
                '64 samples per chain; fixed single training seed and one fixed Q partition',
                'No retraining, root/exponent inference, MI or additional learned readout',
                'C-draw variation is conditional noise, not additional physical chains'],
            gpu_seconds_including_prior=prior + time.perf_counter() - started)
        if (receipt['gpu_seconds_including_prior'] > args.gpu_budget_seconds
                or max(peak_bytes(device), peak_reserved_bytes(device)) > 8 * 1024 ** 3):
            raise RuntimeError('budget_hold: budget exceeded at final write check')
        save_json(out / 'receipt.json', receipt, new=True)
        save_json(status_path, dict(status='complete', workflow_complete=True,
            returncode=0, error=None, requires_user_action=False, arrays_sha256=receipt['arrays_sha256']))
        return receipt
    except Exception as error:
        hold = str(error).startswith('budget_hold:')
        save_json(status_path, dict(status='budget_hold' if hold else 'failed', workflow_complete=False,
            error=dict(type=type(error).__name__, message=str(error)), requires_user_action=True,
            elapsed_seconds=time.perf_counter() - started))
        raise


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('benchmark', 'collect'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--snapshot-root', type=Path, required=True)
    parser.add_argument('--baseline-root', type=Path, required=True)
    parser.add_argument('--device', default='cuda:2')
    for name in ('q', 'c', 'r'):
        parser.add_argument(f'--expected-{name}-sha', default=MODULE_SHA256[name.upper()])
    parser.add_argument('--gpu-budget-seconds', type=float, default=900.)
    parser.add_argument('--prior-gpu-seconds', type=float, default=0.)
    return parser


def main():
    args = build_parser().parse_args()
    if (not np.isfinite(args.gpu_budget_seconds) or not 0 < args.gpu_budget_seconds <= 900
            or not np.isfinite(args.prior_gpu_seconds) or args.prior_gpu_seconds < 0):
        raise ValueError('GPU budget is capped at 900s; prior actual usage must be finite and nonnegative')
    receipt = benchmark(args) if args.command == 'benchmark' else collect(args)
    print(json.dumps(dict(status=receipt['status'], root=str(args.root), command=args.command), ensure_ascii=False))
    return 0 if receipt['status'] == 'complete' else 3


if __name__ == '__main__':
    raise SystemExit(main())
