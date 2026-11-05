"""Short, isolated default-QCR update cost audit; JSON only, no checkpoints.

This is not a training entry point. It runs two same-seed diagnostic trajectories
from fresh random initialization: a normal boundary-synchronized baseline, then
an exclusively timed twin. It never builds MI statistics, samples physical data,
or hashes the physical archive. Existing cache provenance is checked against
metadata and filesystem identities, not independently recertified.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import replace
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import torch

from experiments import rsmi_separate_qcr as driver
from experiments.potts_frozen_qc_continue import pattern_function
from recurrent_rg.checkpoints import build_model, isolated_rng
from recurrent_rg.config import RunConfig, ScaleConfig
from recurrent_rg.empirical_mi import binary_lattice_mi_statistics
from recurrent_rg.execution import configure_cuda_execution, select_training_batch
from recurrent_rg.learning_rate import LearningRatePolicy
from recurrent_rg import losses
from recurrent_rg.mi_cache import (
    _cache_identity, _statistics_from_payload, lattice_statistics_cache_key,
)
from recurrent_rg.representation import normalize_affine_decoder_
from recurrent_rg.rsmi import kernel_probabilities, region_offsets
from recurrent_rg.tensor_io import load_cpu_tensors


BUCKETS = (
    'batch_selection_and_transfer', 'C_multiscale_forward', 'Q_encoding_forward',
    'motif_statistics_forward', 'remaining_objective_forward',
    'R_forward_including_legacy_diagnostics', 'R_closure_assembly',
    'C_MI_forward', 'C_routed_backward', 'Q_routed_backward', 'R_routed_backward',
    'optimizer_and_decoder_normalization', 'overhead',
)


class ExclusiveClock:
    """Host critical-path timing with nested time subtracted exactly once.

    CUDA synchronization at every region boundary intentionally exposes costs
    and also perturbs overlap/launch overhead. The uninstrumented twin measures
    that perturbation; these are not CUDA-event kernel-only timings.
    """
    def __init__(self, synchronize=lambda: None, *, enabled=True, now=time.perf_counter):
        self.synchronize, self.enabled, self.now = synchronize, enabled, now
        self.stack = []
        self.values = defaultdict(float)

    @contextmanager
    def region(self, name):
        if not self.enabled:
            yield
            return
        self.synchronize()
        frame = [name, self.now(), 0.]
        self.stack.append(frame)
        try:
            yield
        finally:
            self.synchronize()
            elapsed = self.now() - frame[1]
            assert self.stack.pop() is frame
            exclusive = elapsed - frame[2]
            if exclusive < -1e-8:
                raise RuntimeError('Nested timing accounting became negative')
            self.values[name] += max(0., exclusive)
            if self.stack:
                self.stack[-1][2] += elapsed

    def reset(self):
        if self.stack:
            raise RuntimeError('Cannot reset a running timing region')
        self.values.clear()


def _wrapped(clock, name, function):
    def call(*args, **kwargs):
        with clock.region(name):
            return function(*args, **kwargs)
    return call


@contextmanager
def scoped_instrumentation(clock, model):
    """Patch only this diagnostic process and restore everything on exit."""
    if not clock.enabled:
        yield
        return
    c_ids = {id(p) for p in model.rg_kernel.parameters()}
    q_ids = {id(p) for p in model.theory_encoder.parameters()}
    r_ids = {id(p) for p in model.theory_rg.parameters()}
    original_grad = torch.autograd.grad
    gradient_depth = 0

    def gradient(outputs, inputs, *args, **kwargs):
        nonlocal gradient_depth
        # Custom BinaryMean/BinaryLocal VJPs re-enter autograd for recomputed
        # local tables/input derivatives. Their cost belongs to the enclosing
        # routed gradient, not a second independent backward region.
        if gradient_depth:
            return original_grad(outputs, inputs, *args, **kwargs)
        parameters = [inputs] if isinstance(inputs, torch.Tensor) else list(inputs)
        ids = {id(parameter) for parameter in parameters}
        names = {frozenset(c_ids): 'C_routed_backward',
                 frozenset(q_ids): 'Q_routed_backward',
                 frozenset(r_ids): 'R_routed_backward'}
        name = names.get(frozenset(ids))
        if name is None:
            raise ValueError('Diagnostic encountered an unsupported gradient route')
        with clock.region(name):
            gradient_depth += 1
            try:
                return original_grad(outputs, parameters, *args, **kwargs)
            finally:
                gradient_depth -= 1

    with patch.object(model, 'coarse_grain', _wrapped(
            clock, 'C_multiscale_forward', model.coarse_grain)), \
         patch.object(model, 'advance_theory', _wrapped(
            clock, 'R_forward_including_legacy_diagnostics', model.advance_theory)), \
         patch.object(losses, 'split_theory_views', _wrapped(
            clock, 'Q_encoding_forward', losses.split_theory_views)), \
         patch.object(losses, 'one_step_closure_loss', _wrapped(
            clock, 'R_closure_assembly', losses.one_step_closure_loss)), \
         patch.object(losses, 'multistep_closure_loss', _wrapped(
            clock, 'R_closure_assembly', losses.multistep_closure_loss)), \
         patch.object(losses, 'all_start_anchor_closure_loss', _wrapped(
            clock, 'R_closure_assembly', losses.all_start_anchor_closure_loss)), \
         patch.object(torch.autograd, 'grad', gradient):
        yield


def stat_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), device=stat.st_dev, inode=stat.st_ino,
                bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)


def _read_train(path):
    """Map existing TRAIN storage without full-data scanning or relabeling."""
    with Path(path).open('rb') as stream:
        if stream.read(4) != b'PK\x03\x04':
            raise ValueError('Diagnostic requires a modern ZIP tensor archive for private mmap')
    payload = load_cpu_tensors(path, mmap=True)
    if isinstance(payload, dict) and payload.get('schema_version') == 1:
        train = payload['train']['configurations']
    elif isinstance(payload, (tuple, list)) and len(payload) == 4:
        train = payload[0]
    else:
        raise ValueError('Unsupported physical cache payload')
    if not isinstance(train, torch.Tensor) or train.layout != torch.strided:
        raise ValueError('Require a strided TRAIN configuration tensor')
    return train


def _receipt_cache_path(receipt):
    record = receipt['splits']['train']['cache']
    return Path(record['path']), receipt['data_sha256']


def load_inputs(cfg, *, mi_receipt=None, mi_cache_path=None, data_sha256=None):
    if cfg.data.train_pool_replicas != 1 or cfg.data.test_pool_replicas != 1:
        raise ValueError('Cost audit requires native existing cache splits, not on-the-fly pooling')
    train = _read_train(cfg.data.cache)
    expected_rank = cfg.model.dimension + 3
    if train.ndim != expected_rank or train.shape[-1] != 1:
        raise ValueError('Physical cache dimension/channel mismatch')
    if len(train) < cfg.training.system_batch or train.shape[1] < cfg.training.sample_batch:
        raise ValueError('Not enough physical ensembles/configurations for the exact configured batch')
    ScaleConfig(tuple(train.shape[2:-1]), cfg.training.max_rg_steps,
                cfg.model.block_factor).shapes()
    receipt = None
    if mi_receipt is not None:
        receipt = json.loads(Path(mi_receipt).read_text())
        from_receipt, digest = _receipt_cache_path(receipt)
        if data_sha256 is not None and data_sha256 != digest:
            raise ValueError('Requested physical fingerprint disagrees with preprocessing receipt')
        data_sha256 = digest
        mi_cache_path = Path(mi_cache_path) if mi_cache_path else from_receipt
    if mi_cache_path is None or data_sha256 is None:
        raise ValueError('Supply --mi-receipt or both --mi-cache-path and --data-sha256')
    payload = load_cpu_tensors(mi_cache_path, mmap=True)
    _, environment = region_offsets(cfg.model.dimension, block=cfg.model.block_factor, buffer=1)
    key_data = lattice_statistics_cache_key(cfg, data_sha256, 'train', train, environment,
        binary_lattice_mi_statistics.__name__, dict(block=cfg.model.block_factor))
    identity, key = _cache_identity(key_data)
    if (set(payload) != {'identity', 'key', 'statistics'}
            or payload['identity'] != identity or payload['key'] != key):
        raise ValueError('Existing MI cache disagrees with physical fingerprint/shape/region conventions')
    if receipt is not None:
        record = receipt['splits']['train']
        if (record.get('data_shape') != list(train.shape)
                or record['cache'].get('key') != key):
            raise ValueError('Preprocessing receipt shape/key mismatch')
    histogram = _statistics_from_payload(payload['statistics'])
    if len(histogram) != len(train):
        raise ValueError('MI and physical TRAIN ensemble counts differ')
    return train, histogram, dict(data_sha256=data_sha256, MI_key=key,
        physical=stat_identity(cfg.data.cache), MI=stat_identity(mi_cache_path),
        receipt=(stat_identity(mi_receipt) if mi_receipt is not None else None),
        independent_data_rehash=False, validation='existing receipt/cache identity plus selected binary batches')


def validate_selected_batches(train, cfg, count):
    """Validate only the actual diagnostic panels; never scan the full archive."""
    generator = torch.Generator().manual_seed(cfg.training.seed)
    def validate(panel):
        if not bool(((panel == 1) | (panel == -1)).all()):
            raise ValueError('A selected physical diagnostic panel is not binary +/-1')
        return panel.numel()
    checked = validate(train[:min(32, len(train))])
    for _ in range(count):
        ids = torch.randperm(len(train), generator=generator)[:cfg.training.system_batch]
        indices = torch.randperm(train.shape[1], generator=generator)[:cfg.training.sample_batch]
        checked += validate(train[ids[:, None], indices[None, :]])
    return checked


def _tree_digest(value):
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode()); visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def _summary(values):
    ordered = sorted(values)
    return dict(mean=statistics.mean(values), median=statistics.median(values),
        minimum=min(values), maximum=max(values),
        p95=ordered[min(len(ordered)-1, math.ceil(.95*len(ordered))-1)])


def _make_model(cfg, train, device):
    torch.manual_seed(cfg.training.seed)
    model = build_model(cfg.model, cfg.training.max_rg_steps).to(device)
    model.detach_kernel_representation = True
    execution = configure_cuda_execution(model, device, independent_c=True)
    function = pattern_function(cfg)
    if cfg.model.affine_reconstruction:
        with isolated_rng(device, cfg.training.seed+9000), torch.no_grad():
            levels = model.coarse_grain(train[:min(32, len(train))].to(
                device=device, dtype=torch.float64))
            center = torch.stack([function(level).mean(0) for level in levels]).mean(0)
            model.theory_encoder.distribution_decoder.bias.copy_(center)
    return model, execution, function


def _run_phase(cfg, train, histogram, device, warmup, steps, *, instrumented):
    synchronize = ((lambda: torch.cuda.synchronize(device)) if device.type == 'cuda'
                   else lambda: None)
    model, execution, function = _make_model(cfg, train, device)
    q = list(model.theory_encoder.parameters())
    c = list(model.rg_kernel.parameters())
    r = list(model.theory_rg.parameters())
    optimizer = torch.optim.Adam(q+c+r, lr=cfg.training.learning_rate)
    policy = LearningRatePolicy(cfg.training)
    generator = torch.Generator().manual_seed(cfg.training.seed)
    mi_generator = torch.Generator().manual_seed(cfg.training.seed+2)
    selected_mi = bool(execution['enabled'])
    systems = min(16, len(train))
    visible = cfg.model.block_factor**cfg.model.dimension
    codes = (torch.arange(2**visible, device=device)[:, None]
             // (2**torch.arange(visible, device=device))) % 2
    patterns = (2*codes-1).double()[None]
    clock = ExclusiveClock(synchronize, enabled=instrumented)
    pattern = _wrapped(clock, 'motif_statistics_forward', function) if instrumented else function
    rows, total_times = [], []
    initial = driver.qcr_module_sha256(model)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    last_loss = None
    with scoped_instrumentation(clock, model):
        for update in range(1, warmup+steps+1):
            clock.reset()
            synchronize(); started = time.perf_counter()
            with clock.region('overhead'):
                if policy.kind != 'constant':
                    policy.apply_update(optimizer, update)
                model.train()
                with clock.region('batch_selection_and_transfer'):
                    ids = torch.randperm(len(train), generator=generator)[:cfg.training.system_batch]
                    indices = torch.randperm(train.shape[1], generator=generator)[:cfg.training.sample_batch]
                    selected = select_training_batch(train, ids, indices, device, reference=False)
                    half = cfg.training.sample_batch//2
                    a, b = selected[:, :half], selected[:, half:]
                    batch = torch.cat((a, -a, b, -b), 1) if cfg.data.spin_flip_augment else selected
                optimizer.zero_grad(set_to_none=True)
                with clock.region('remaining_objective_forward'):
                    base, terms = losses.qcr_objective(
                        model, batch, pattern_fn=pattern, **cfg.loss.objective_kwargs())
                with clock.region('C_MI_forward'):
                    selected_hist = torch.randperm(len(train), generator=mi_generator)[:systems]
                    table = kernel_probabilities(model.rg_kernel, patterns)[0]
                    information = (histogram.information_for(table, selected_hist).mean()
                        if selected_mi else histogram.information(table)[selected_hist].mean())
                mi_objective = -2.*information
                total = base+mi_objective
                closure = terms.get('closure_total', cfg.loss.weights.commutator*terms['commutator']
                                    + cfg.loss.rollout_weight*terms['rollout'])
                if not torch.isfinite(total):
                    raise FloatingPointError('Nonfinite diagnostic loss')
                driver.independent_component_gradients(total, closure,
                    mi_objective+2.*terms['kernel_noncollapse']+terms.get('kernel_sign', 0.),
                    q, c, r)
                with clock.region('optimizer_and_decoder_normalization'):
                    optimizer.step()
                    normalize_affine_decoder_(model.theory_encoder)
                last_loss = total.detach()
            synchronize(); elapsed = time.perf_counter()-started
            if update > warmup:
                total_times.append(elapsed)
                if instrumented:
                    row = {name: clock.values.get(name, 0.) for name in BUCKETS}
                    row['total_seconds'] = elapsed
                    row['unassigned_boundary_seconds'] = max(0., elapsed-sum(row[name] for name in BUCKETS))
                    rows.append(row)
    final = driver.qcr_module_sha256(model)
    result = dict(instrumented=instrumented, measured_updates=steps, warmup_updates=warmup,
        per_update_seconds=_summary(total_times), per_update_total_seconds=total_times,
        initial_module_sha256=initial, final_module_sha256=final,
        optimizer_sha256=_tree_digest(optimizer.state_dict()),
        RNG_sha256=_tree_digest(dict(cpu=torch.get_rng_state(),
            cuda=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
            batch=generator.get_state(), MI=mi_generator.get_state())),
        final_total_loss=float(last_loss), execution=execution,
        peak_allocated_bytes=(torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None))
    if instrumented:
        totals = {name: sum(row[name] for row in rows) for name in BUCKETS}
        measured_total = sum(total_times)
        result.update(buckets={name: dict(seconds=totals[name],
            mean_seconds=totals[name]/steps, fraction_of_instrumented_update=totals[name]/measured_total)
            for name in BUCKETS}, exclusive_rows=rows,
            accounted_fraction=sum(totals.values())/measured_total)
    return result, model, patterns, mi_generator


def _mi_only_probe(model, histogram, patterns, generator, count, device):
    """Marginal MI-only VJP diagnostic, separate from the additive update audit."""
    synchronize = ((lambda: torch.cuda.synchronize(device)) if device.type == 'cuda'
                   else lambda: None)
    copied_generator = torch.Generator().set_state(generator.get_state())
    forward, backward = [], []
    before = driver.qcr_module_sha256(model)
    gradients = []
    for _ in range(count):
        ids = torch.randperm(len(histogram), generator=copied_generator)[:min(16, len(histogram))]
        synchronize(); started = time.perf_counter()
        table = kernel_probabilities(model.rg_kernel, patterns)[0]
        information = (histogram.information_for(table, ids).mean() if device.type == 'cuda'
                       else histogram.information(table)[ids].mean())
        objective = -2.*information
        synchronize(); forward.append(time.perf_counter()-started)
        started = time.perf_counter()
        grad = torch.autograd.grad(objective, list(model.rg_kernel.parameters()), allow_unused=True)
        synchronize(); backward.append(time.perf_counter()-started)
        gradients.append(_tree_digest(grad))
    if driver.qcr_module_sha256(model) != before:
        raise RuntimeError('MI-only VJP probe changed parameters')
    return dict(probes=count, optimizer_updates=0, physical_or_C_sampling_draws=0,
        included_in_exclusive_update_sum=False, forward_seconds=_summary(forward),
        MI_only_C_VJP_seconds=_summary(backward), gradient_sha256=gradients,
        interpretation='Marginal MI-only graph at the same final kernel, not an additive decomposition of full C backward; no Q/C/R parameter updates.')


def profile(config_path, output, *, device='cpu', mi_receipt=None,
            mi_cache_path=None, data_sha256=None, warmup=5, steps=20, mi_only_steps=3):
    output = Path(output).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if warmup < 0 or steps < 1 or mi_only_steps < 0:
        raise ValueError('Need nonnegative warmup/probes and positive measured steps')
    device = torch.device(device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Explicit CPU or CUDA device required')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('Requested CUDA is unavailable')
        device = torch.device('cuda', 0 if device.index is None else device.index)
        torch.cuda.set_device(device)
    raw = json.loads(Path(config_path).read_text())
    cfg = RunConfig.load(config_path)
    cfg = driver.default_representation(cfg, raw)
    cfg = driver.default_closure(cfg, raw)
    cfg = replace(cfg, training=replace(cfg.training, gradient_mode='qc_r_separate'),
        loss=replace(cfg.loss, kernel_sign_weight=1., kernel_neutral_weight=0.,
            kernel_noncollapse_mode='variance_floor_ensemble', kernel_variance_ratio=1.,
            weights=replace(cfg.loss.weights, kernel_noncollapse=2.)))
    cfg.validate()
    arguments = driver.build_parser().parse_args(['--config', str(config_path), '--output', str(output)])
    driver.validate_recipe(arguments, cfg)
    if (cfg.model.dimension not in (2, 3) or cfg.model.categorical_states
            or cfg.model.reconstruction_hidden or not cfg.model.affine_reconstruction
            or cfg.loss.tangent_weight or cfg.loss.orientation_weight
            or cfg.loss.balanced_reconstruction_weight or cfg.loss.isotropy_weight
            or cfg.loss.kernel_neutral_weight):
        raise ValueError('Diagnostic currently supports standard binary 2D/3D affine-Q recipe only')
    torch.set_num_threads(cfg.training.threads)
    setup_started = time.perf_counter()
    train, histogram, inputs = load_inputs(cfg, mi_receipt=mi_receipt,
        mi_cache_path=mi_cache_path, data_sha256=data_sha256)
    checked = validate_selected_batches(train, cfg, warmup+steps)
    histogram = driver.place_mi_histogram_split(histogram, 'train', device,
                                               selected=device.type == 'cuda')
    setup_seconds = time.perf_counter()-setup_started
    baseline, baseline_model, _, _ = _run_phase(
        cfg, train, histogram, device, warmup, steps, instrumented=False)
    del baseline_model
    gc.collect()
    instrumented, model, patterns, generator = _run_phase(
        cfg, train, histogram, device, warmup, steps, instrumented=True)
    fields = ('initial_module_sha256', 'final_module_sha256', 'optimizer_sha256',
              'RNG_sha256', 'final_total_loss')
    equivalence = {field: baseline[field] == instrumented[field] for field in fields}
    if not all(equivalence.values()):
        raise RuntimeError('Timing instrumentation changed actual trajectory: '+str(equivalence))
    probe = (_mi_only_probe(model, histogram, patterns, generator, mi_only_steps, device)
             if mi_only_steps else None)
    for field in ('physical', 'MI', 'receipt'):
        if inputs[field] is not None and stat_identity(inputs[field]['path']) != inputs[field]:
            raise RuntimeError('An input identity changed during diagnostic: '+field)
    result = dict(status='complete', diagnostic_only=True, config=cfg.to_dict(),
        execution_device=str(device), physical_train_shape=list(train.shape),
        inputs=inputs, selected_binary_elements_validated=checked, setup_seconds=setup_seconds,
        labels_used_in_loss=False, physical_sampling=False, checkpoints_saved=False,
        baseline=baseline, instrumented=instrumented, exact_twin_equivalence=equivalence,
        instrumentation_inflation=(instrumented['per_update_seconds']['mean']
                                   / baseline['per_update_seconds']['mean']-1.),
        MI_only_probe=probe, limitations=[
            'Fresh random/default-calibrated model; diagnostic updates are discarded, not scientific training.',
            'CUDA boundary synchronization measures serialized host critical-path cost and perturbs overlap; use baseline for actual update speed.',
            'Exclusive fractions use instrumented time and cannot be assumed identical to uninstrumented GPU-kernel fractions.',
            'C_routed_backward includes -2 MI, variance-floor, color NLL and cross-scale STE input VJPs; it is not RSMI-only cost.',
            'Q_routed_backward is the actual grad(total-closure,Q) phase, including any cancellation-graph work, not only Q network kernels.',
            'R forward/assembly includes historical one-step and rollout diagnostics still computed by the all-start objective.',
            'No evaluation, reporting, checkpoint serialization, MI preprocessing or whole-run setup is included in per-update fractions.',
            'Existing physical fingerprint is trusted from the cache receipt; archive was privately mapped and selected batches validated, not rehashed.',
            'Physical data intentionally stays CPU-mapped; full archive GPU residency is disabled and this execution choice is explicit.',
        ])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--mi-receipt', type=Path)
    parser.add_argument('--mi-cache-path', type=Path)
    parser.add_argument('--data-sha256')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--mi-only-steps', type=int, default=3)
    args = parser.parse_args(argv)
    profile(args.config, args.output, device=args.device, mi_receipt=args.mi_receipt,
        mi_cache_path=args.mi_cache_path, data_sha256=args.data_sha256,
        warmup=args.warmup, steps=args.steps, mi_only_steps=args.mi_only_steps)


if __name__ == '__main__':
    main()
