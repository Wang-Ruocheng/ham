"""Compare old/new host gathers on the same optimized CUDA2 Q/C/R path.

This is a tiny synthetic execution regression, not a scientific experiment or
an estimate of the production dataset's RAM footprint. It never reads a
physical cache, MI cache, checkpoint, or experiment configuration.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

import torch

from experiments.rsmi_separate_qcr import independent_component_gradients
from recurrent_rg.checkpoints import build_model
from recurrent_rg.config import LossConfig, ModelConfig
from recurrent_rg.data.statistics import plaquette_distribution
from recurrent_rg.empirical_mi import binary_lattice_mi_statistics
from recurrent_rg.execution import configure_cuda_execution, select_training_batch
from recurrent_rg.losses import qcr_objective
from recurrent_rg.representation import normalize_affine_decoder_
from recurrent_rg.rsmi import kernel_probabilities


EXPECTED_UUID = 'GPU-713b07d0-118c-f939-c0a8-565390215e01'
DIMENSIONS = (1, 5, 10, 100)
ATOL, RTOL = 1e-10, 1e-9


def normalize_uuid(value):
    return str(value).lower().removeprefix('gpu-')


def gpu_snapshot(*, command=subprocess.check_output):
    """Read physical GPU2 only; compute-app listing is a read-only UUID filter."""
    raw = command(['nvidia-smi', '-i', '2', '--query-gpu=index,uuid,memory.free',
                   '--format=csv,noheader,nounits'], text=True, timeout=15)
    rows = [line.strip().split(',') for line in raw.splitlines() if line.strip()]
    if len(rows) != 1 or len(rows[0]) != 3:
        raise RuntimeError('Cannot establish the physical GPU2 identity')
    index, uuid, free = [part.strip() for part in rows[0]]
    apps = command(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                    '--format=csv,noheader,nounits'], text=True, timeout=15)
    pids = []
    for line in apps.splitlines():
        fields = [part.strip() for part in line.split(',')]
        if len(fields) == 2 and normalize_uuid(fields[0]) == normalize_uuid(uuid):
            pids.append(int(fields[1]))
    return dict(index=int(index), uuid=uuid, free_mib=float(free), compute_pids=pids,
                observed_monotonic=time.monotonic(),
                observed_utc=datetime.now(timezone.utc).isoformat())


def validate_snapshot(snapshot):
    if snapshot['index'] != 2 or normalize_uuid(snapshot['uuid']) != normalize_uuid(EXPECTED_UUID):
        raise RuntimeError('Physical GPU2 UUID mismatch; refusing CUDA execution')
    if snapshot['compute_pids'] or snapshot['free_mib'] < 12 * 1024:
        raise RuntimeError('GPU2 is not exclusively idle with at least 12 GiB free')


def acquire_gpu_gate(*, snapshot=gpu_snapshot, sleep=time.sleep):
    # Requiring the unmasked ordinal avoids an accidental CUDA_VISIBLE_DEVICES
    # remapping to another user's accelerator. No environment is modified.
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        raise RuntimeError('Unset CUDA_VISIBLE_DEVICES; verification requires physical cuda:2')
    first = snapshot()
    validate_snapshot(first)
    sleep(10.)
    second = snapshot()
    validate_snapshot(second)
    if second['observed_monotonic'] - first['observed_monotonic'] < 10.:
        raise RuntimeError('GPU idle observations must be at least 10 seconds apart')
    return [first, second]


def clone_cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: clone_cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [clone_cpu_tree(item) for item in value]
    return value


def compare_trees(left, right, *, exact=False):
    """Fail closed on structural, NaN, RNG, or strict float64 differences."""
    result = dict(tensor_count=0, maximum_absolute_difference=0.)

    def visit(a, b):
        if isinstance(a, torch.Tensor):
            if not isinstance(b, torch.Tensor) or a.dtype != b.dtype or a.shape != b.shape:
                raise AssertionError('Tensor shape or dtype differs')
            result['tensor_count'] += 1
            if a.is_floating_point() or a.is_complex():
                if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
                    raise AssertionError('Nonfinite values in execution comparison')
                difference = float((a-b).abs().max()) if a.numel() else 0.
                result['maximum_absolute_difference'] = max(
                    result['maximum_absolute_difference'], difference)
                if exact:
                    assert torch.equal(a, b), 'Exact tensor comparison failed'
                else:
                    torch.testing.assert_close(a, b, atol=ATOL, rtol=RTOL)
            else:
                assert torch.equal(a, b), 'Integer or RNG tensor differs'
        elif isinstance(a, dict):
            assert isinstance(b, dict) and a.keys() == b.keys()
            for key in a:
                visit(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            assert isinstance(b, (tuple, list)) and len(a) == len(b)
            for x, y in zip(a, b):
                visit(x, y)
        else:
            assert a == b, 'Non-tensor execution state differs'

    visit(left, right)
    return result


def synthetic_data():
    generator = torch.Generator().manual_seed(20261021)
    return (2*torch.randint(2, (4, 8, 16, 16, 1), generator=generator)-1).double()


def trajectory(configurations, histogram, dimension, device, *, old_host_selection, updates):
    """Only the host indexing expression differs between the paired runs."""
    device = torch.device(device)
    # Do not use torch.manual_seed, which can seed all visible CUDA devices.
    torch.random.default_generator.manual_seed(91)
    context = torch.cuda.device(device) if device.type == 'cuda' else nullcontext()
    with context:
        if device.type == 'cuda':
            torch.cuda.manual_seed(91)
        model = build_model(ModelConfig(dimension=2, theory_dim=dimension,
            hidden=96, block_factor=2, binary_lookup=True, kernel_sampling='bernoulli',
            affine_reconstruction=True), 2).to(device)
        model.detach_kernel_representation = True
        execution = configure_cuda_execution(model, device, reference=False, independent_c=True)
        if device.type == 'cuda' and not all(execution[field] for field in
                ('enabled', 'shared_binary_Q_tables', 'trusted_internal_binary_checks')):
            raise RuntimeError('Production CUDA binary fast paths were not enabled in both branches')
        q, c, r = [list(module.parameters()) for module in
                   (model.theory_encoder, model.rg_kernel, model.theory_rg)]
        optimizer = torch.optim.Adam(q+c+r, lr=.001)
        loss = LossConfig(closure_objective='one_step_and_rollout',
                          kernel_noncollapse_mode='variance_floor_ensemble',
                          kernel_variance_ratio=1., kernel_sign_weight=1.)
        batch_generator = torch.Generator().manual_seed(91)
        mi_generator = torch.Generator().manual_seed(182)
        codes = torch.arange(16)[:, None].bitwise_right_shift(torch.arange(3, -1, -1)) % 2
        patterns = (2*codes-1).double()[None].to(device)
        traces = []
        for step in range(1, updates+1):
            ids = torch.randperm(4, generator=batch_generator)
            ids = ids[torch.tensor([1, 0, 1])]
            indices = torch.randperm(8, generator=batch_generator)
            indices = indices[torch.tensor([2, 0, 2, 7])]
            selected_hist = torch.randperm(4, generator=mi_generator)[torch.tensor([2, 0, 2])]
            batch = select_training_batch(configurations, ids, indices, device,
                                          reference=old_host_selection)
            optimizer.zero_grad(set_to_none=True)
            base, terms = qcr_objective(model, batch, pattern_fn=plaquette_distribution,
                                       **loss.objective_kwargs())
            table = kernel_probabilities(model.rg_kernel, patterns)[0]
            info = histogram.information_for(table, selected_hist).mean()
            mi_objective = -2*info
            total = base+mi_objective
            closure = loss.weights.commutator*terms['commutator']+loss.rollout_weight*terms['rollout']
            c_objective = mi_objective+2*terms['kernel_noncollapse']+terms['kernel_sign']
            independent_component_gradients(total, closure, c_objective, q, c, r)
            gradients = [None if parameter.grad is None else parameter.grad for parameter in q+c+r]
            gradients = clone_cpu_tree(gradients)
            optimizer.step()
            normalize_affine_decoder_(model.theory_encoder)
            rng = dict(cpu=torch.get_rng_state(), batch=batch_generator.get_state(),
                       empirical_mi=mi_generator.get_state())
            if device.type == 'cuda':
                rng['cuda2'] = torch.cuda.get_rng_state(device)
            traces.append(clone_cpu_tree(dict(step=step, batch=batch,
                indices=dict(ensemble=ids, sample=indices, empirical_mi=selected_hist),
                losses=dict(total=total, closure=closure, c_objective=c_objective,
                            empirical_mi=info, **terms),
                gradients=gradients, parameters=model.state_dict(),
                adam=optimizer.state_dict(), rng=rng)))
        return dict(execution=execution, traces=traces)


def compare_dimension(configurations, histogram, dimension, device, *, updates):
    old = trajectory(configurations, histogram, dimension, device,
                     old_host_selection=True, updates=updates)
    new = trajectory(configurations, histogram, dimension, device,
                     old_host_selection=False, updates=updates)
    assert old['execution'] == new['execution']
    records = []
    for left, right in zip(old['traces'], new['traces']):
        row = dict(step=left['step'])
        for category in ('batch', 'indices', 'rng', 'losses', 'gradients', 'parameters', 'adam'):
            row[category] = compare_trees(left[category], right[category],
                                          exact=category in ('batch', 'indices', 'rng'))
        records.append(row)
    return dict(dimension=dimension, updates=updates, passed=True,
                shared_execution=old['execution'], comparisons=records)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:2', choices=['cuda:2'])
    parser.add_argument('--updates', type=int, default=2, choices=range(1, 5))
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite a verification receipt')
    checks = acquire_gpu_gate()
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 2:
        raise RuntimeError('Requested cuda:2 unavailable; no CPU fallback')
    properties = torch.cuda.get_device_properties(args.device)
    actual_uuid = getattr(properties, 'uuid', None)
    if actual_uuid is None or normalize_uuid(actual_uuid) != normalize_uuid(EXPECTED_UUID):
        raise RuntimeError('CUDA ordinal 2 UUID could not be verified; no fallback')
    torch.set_num_threads(1)
    configurations = synthetic_data()
    # Synthetic full-shell, buffer-one statistics only: no production MI I/O.
    histogram = binary_lattice_mi_statistics(configurations, flip=False, buffer=1, block=2)
    histogram.prepare_selection()
    histogram = histogram.to(args.device)
    torch.cuda.reset_peak_memory_stats(args.device)
    reports = []
    for dimension in DIMENSIONS:
        reports.append(compare_dimension(configurations, histogram, dimension,
                                         args.device, updates=args.updates))
        print(f'dimg={dimension}: exact batches/RNG; float64 loss/gradient/Adam/parameter equivalence passed', flush=True)
    torch.cuda.synchronize(args.device)
    result = dict(status='complete', passed=True, device=args.device,
        physical_uuid=EXPECTED_UUID, idle_gpu_checks=checks,
        synthetic_only=True, production_data_or_MI_opened=False,
        scientific_training=False, dimensions=reports,
        atol=ATOL, rtol=RTOL, floating_bitwise_equality_claimed=False,
        comparison='Only historical versus joint host gather differs; CUDA fast paths stay enabled in both.',
        gradient_routing='Q: representation only; C: -2 MI + 2 variance-floor ratio1 + 1 color NLL; R: uniform one-step5 + rollout5.',
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(args.device))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    return result


if __name__ == '__main__':
    main()
