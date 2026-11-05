"""Explicit label-free natural-coupling Q ablation, without changing core files.

The affine probability-MSE control uses the original model and objective.
Only the natural arm replaces the representation decoder and its reconstruction
term.  Process-local hooks are restored even on failure; do not use this module
concurrently with unrelated trainers in the same Python process.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from experiments import rsmi_separate_qcr as trainer
from recurrent_rg import checkpoints, losses, mi_cache, representation
from recurrent_rg.config import RunConfig, ScaleConfig
from recurrent_rg.coupling_decoder import (
    DirectionalCouplingDecoder, OPERATOR_DEFINITIONS, OPERATOR_NAMES,
    normalize_coupling_decoder_,
)


DATA_SHA256 = 'f0e319a771d220e948c70e97fda8bbd373d46fdbb280211f0cbd29ce9d36de54'
MI_KEYS = {
    '53b4cb45bda6c9271a4d18348e9f20cf75bbcf9f63647e76e5ca401dc19f5bea',
    '85992c4ab2ce55ab36a0c45842d3a6e9c695bdd418c4cc3688117d6a60ebd8b6',
}
OVERRIDE_KIND = 'directional_even_gibbs_crossfit_pseudolikelihood_v1'
MODE = 'directional_pseudolikelihood'
_CORE_BUILD_MODEL = checkpoints.build_model
_HOOK_ACTIVE = False


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_digest(state, *, exclude_decoder=False):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if exclude_decoder and name.startswith('distribution_decoder.'):
            continue
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _exclusive_json(path, value):
    path = Path(path)
    with path.open('x') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def representation_contract():
    return dict(kind=OVERRIDE_KIND, reconstruction_mode=MODE,
        objective='Shared finite-basis Gibbs natural parameters; symmetric held-half single-site pseudolikelihood',
        hamiltonian='H_g(s) = -sum_j (a_j + W_j g) S_j(s)',
        operator_names=list(OPERATOR_NAMES), operator_definitions=list(OPERATOR_DEFINITIONS),
        effective_core_affine_reconstruction_flag=False,
        architecture='Original random Q/C/R built first; original affine decoder replaced, not retained',
        initialization='Random W under isolated CPU RNG; natural-parameter bias explicitly zero; no data calibration, PCA or pretraining',
        gauge='Decoder W columns normalized after updates; zero columns remain zero; no full-rank/orthogonality constraint',
        reconstruction_units='Mean conditional negative log probability in nats per site/configuration/ensemble, then mean over scales; not probability sum-bin MSE',
        reconstruction_weight=10., weight=10., subset_effective_weights=[20., 5.],
        decoder_targets='Opposite half of the same independent physical ensemble at the same scale; observed spins detached',
        binary_targets='Only +/-1 roundoff <=1e-12 from straight-through arithmetic is snapped to its sign; finite off-alphabet values are rejected',
        gradient='Q and its decoder only; C receives empirical MI/variance/color only; R receives original closure only',
        labels_used_in_training=False, Kc_or_exponent_target=False,
        limitations=['Explicit seven-operator finite-range/even Hamiltonian prior; not exact RG closure.',
            'A low-dimensional affine natural-parameter subspace need not contain the full renormalized family.',
            'Matching conditional likelihood or geometry does not certify a thermal eigenmode.'])


def attach_coupling_decoder(model, config):
    """Replace the old decoder, preserving every original Q/C/R tensor and RNG."""
    if (config.dimension != 2 or config.categorical_states or config.width != 3
            or config.reconstruction_hidden or config.theory_dim not in (1, 2, 3)):
        raise ValueError('Natural representation needs binary 2D width-three dimg1/2/3')
    reference = next(model.theory_encoder.parameters())
    with torch.random.fork_rng(devices=[]):
        decoder = DirectionalCouplingDecoder(config.theory_dim, device=reference.device, dtype=reference.dtype)
        with torch.no_grad():
            decoder.linear.bias.zero_()
    model.theory_encoder.distribution_decoder = decoder
    model.theory_encoder.distribution_reconstruction_mode = MODE
    model.theory_encoder.representation_override_kind = OVERRIDE_KIND
    normalize_coupling_decoder_(decoder)
    return model


def build_coupling_model(config, max_rg_steps):
    # The old affine auxiliary is itself RNG-isolated. Constructing it first
    # explicitly retains the default Q/C/R initialization and then removes it.
    original_config = replace(config, affine_reconstruction=True, reconstruction_hidden=0)
    return attach_coupling_decoder(_CORE_BUILD_MODEL(original_config, max_rg_steps), config)


def natural_data_geometry_objective(model, levels, first, second, pattern_fn, weights,
        geometry_scale, noise_multiplier, rollout_weight=5., kernel_noncollapse_mode='balance',
        kernel_variance_ratio=.25, coordinate_reduction='sum', geometry_options=None,
        balanced_reconstruction_weight=0.):
    """Original subset/C/R algebra, with only reconstruction replaced by cross-fit PL."""
    del pattern_fn, geometry_scale, noise_multiplier, geometry_options
    if balanced_reconstruction_weight:
        raise ValueError('Natural-coupling ablation does not combine balanced probability reconstruction')
    decoder = getattr(model.theory_encoder, 'distribution_decoder', None)
    if (not isinstance(decoder, DirectionalCouplingDecoder)
            or getattr(model.theory_encoder, 'distribution_reconstruction_mode', None) != MODE):
        raise ValueError('Require the explicitly attached directional Gibbs decoder')
    if len(levels) < 2 or first.shape != second.shape or first.shape[1] != len(levels):
        raise ValueError('Need paired half-subset Q values at each saved scale')
    if not getattr(model, 'detach_kernel_representation', False):
        raise ValueError('Q inputs must detach C; this ablation requires independent C gradients')
    if kernel_noncollapse_mode not in ('balance', 'variance_floor', 'variance_floor_ensemble'):
        raise ValueError('Unknown original binary variance convention')
    kernel = (losses.binary_kernel_variance_floor(levels, kernel_variance_ratio,
        per_ensemble=kernel_noncollapse_mode == 'variance_floor_ensemble')
        if kernel_noncollapse_mode in ('variance_floor', 'variance_floor_ensemble')
        else losses.binary_kernel_noncollapse(levels))
    direct = .5 * (first + second)
    reconstruction, extra = [], {}
    for scale, level in enumerate(levels):
        if level.shape[1] < 2 or level.shape[1] % 2:
            raise ValueError('Natural decoder requires equal nonempty configuration halves')
        midpoint = level.shape[1] // 2
        term = decoder.crossfit_loss(first[:, scale], level[:, midpoint:].detach(),
                                     second[:, scale], level[:, :midpoint].detach())
        reconstruction.append(term)
        extra[f'data_geometry_level_{scale}'] = term
        extra[f'reconstruction_nll_level_{scale}'] = term
        extra[f'pairwise_level_{scale}'] = losses.mean_pairwise_square(direct[:, scale])
    # Precisely the existing algebra, including its internal factor ten on
    # micro subset consistency. With the fixed recipe the effective factors
    # are 20/5, and the R factors are 5/5.
    anchor = 10 * losses.latent_squared_error(first[:, 0], second[:, 0], coordinate_reduction)
    consistency = losses.latent_squared_error(first[:, 1:], second[:, 1:], coordinate_reduction)
    separation = torch.stack(reconstruction).mean()
    predicted = model.advance_theory(direct[:, :-1])
    commutator = losses.latent_squared_error(predicted, direct[:, 1:], coordinate_reduction)
    rollout, rollout_levels = losses.multistep_closure_loss(model, direct, coordinate_reduction)
    total = (weights.anchor*anchor + weights.coarse_consistency*consistency
             + weights.kernel_noncollapse*kernel + weights.separation*separation
             + weights.commutator*commutator + rollout_weight*rollout)
    extra['rollout'] = rollout
    for scale, term in enumerate(rollout_levels, start=1):
        extra[f'rollout_level_{scale}'] = term
    return total, dict(anchor=anchor, coarse_consistency=consistency,
        kernel_noncollapse=kernel, separation=separation, commutator=commutator, **extra)


@contextmanager
def coupling_representation_hooks():
    """Standalone-process natural arm only; the core source files stay untouched."""
    global _HOOK_ACTIVE
    if _HOOK_ACTIVE:
        raise RuntimeError('Natural representation hooks cannot be nested or used concurrently')
    _HOOK_ACTIVE = True
    original_default = trainer.default_representation
    original_contract = trainer.add_supervision_contract
    original_checkpoint = trainer.checkpoint_payload
    original_normalize = representation.normalize_affine_decoder_

    def effective_configuration(config, raw, *, continuing=False):
        if continuing:
            raise ValueError('Natural representation is a fresh architecture ablation, not a resume')
        parsed = original_default(config, raw, continuing=False)
        # This disables ONLY the old affine decoder's 16-bin bias calibration
        # and old probability-MSE metadata branch. The named override records
        # the actual custom decoder; no hidden fallback to a geometry objective.
        return replace(parsed, model=replace(parsed.model, affine_reconstruction=False, reconstruction_hidden=0))

    def contract_override(contract, supervision_weight):
        if supervision_weight:
            raise ValueError('Coupling/K labels cannot supervise this unsupervised representation ablation')
        original_contract(contract, supervision_weight)
        contract.update(training_driver='ising_coupling_representation',
            underlying_training_driver='rsmi_separate_qcr', representation_override=representation_contract(),
            labels_used_in_training=False, K_labels_used_in_loss=False, q_supervision=None,
            q_reconstruction=representation_contract(),
            gradient_QC='Q: cross-fit Gibbs conditional likelihood + original subset losses only; C: original empirical MI/variance/color only',
            note='Explicit unsupervised representation-objective/decoder ablation; all C/R losses and independent gradients unchanged. No K/Kc/exponent target.')

    def normalize(encoder):
        if getattr(encoder, 'distribution_reconstruction_mode', None) == MODE:
            return normalize_coupling_decoder_(encoder.distribution_decoder)
        return original_normalize(encoder)

    def checkpoint(*args, **kwargs):
        payload = original_checkpoint(*args, **kwargs)
        metadata = payload.get('metadata', {})
        if metadata.get('representation_override', {}).get('kind') != OVERRIDE_KIND:
            raise ValueError('Natural checkpoint lacks its explicit architecture/objective contract')
        if not any(key.startswith('theory_encoder.distribution_decoder.linear.') for key in payload['model']):
            raise ValueError('Natural decoder parameters missing from Q checkpoint')
        return payload

    try:
        with ExitStack() as stack:
            for owner, name, value in (
                (checkpoints, 'build_model', build_coupling_model),
                (trainer, 'build_model', build_coupling_model),
                (trainer, 'default_representation', effective_configuration),
                (trainer, 'add_supervision_contract', contract_override),
                (trainer, 'checkpoint_payload', checkpoint),
                (losses, 'data_geometry_objective', natural_data_geometry_objective),
                (representation, 'normalize_affine_decoder_', normalize),
            ):
                stack.enter_context(patch.object(owner, name, value))
            # The current generic driver imports the normalizer inside its
            # update loop. Also expose the same hook if an existing caller
            # retained the driver-level reference, without persisting it.
            stack.enter_context(patch.object(trainer, 'normalize_affine_decoder_', normalize, create=True))
            yield
    finally:
        _HOOK_ACTIVE = False


def load_coupling_run(path, checkpoint=None):
    """Strict custom reconstruction; generic core loader cannot infer this override."""
    path = Path(path)
    if path.is_file() and checkpoint is not None:
        raise ValueError('Choose a checkpoint file OR directory and selector')
    selector = checkpoint if checkpoint is not None else 'checkpoint_last.pt'
    filename = path if path.is_file() else path / (
        f'training_step_{selector}.pt' if isinstance(selector, int) else str(selector))
    payload = torch.load(filename, map_location='cpu', weights_only=True)
    metadata = payload.get('metadata', {})
    if (metadata.get('representation_override', {}).get('kind') != OVERRIDE_KIND
            or metadata.get('labels_used_in_training') is not False or metadata.get('q_supervision')):
        raise ValueError('Need a declared fully unsupervised natural-representation numbered checkpoint')
    # Loading only needs the architecture factory, not any objective or trainer
    # hook. The original loader retains its strict model-state validation.
    with patch.object(checkpoints, 'build_model', build_coupling_model):
        loaded = checkpoints.load_run(path, selector if path.is_dir() else None)
    if loaded.model.theory_encoder.distribution_reconstruction_mode != MODE:
        raise AssertionError('Custom loader failed to restore the natural representation')
    return loaded


def _stat(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Require an existing regular data/receipt file')
    value = path.stat()
    return dict(size=value.st_size, mtime_ns=value.st_mtime_ns, inode=value.st_ino, device=value.st_dev)


@contextmanager
def cached_data_hooks(config, expected_data_sha256):
    """Reuse audited L108 cache and MI only; never rehash/audit the 9GB tensor data."""
    data = Path(config.data.cache).resolve()
    sidecar = data.with_suffix('.json')
    before_data, before_sidecar = _stat(data), _stat(sidecar)
    sidecar_sha = _sha256(sidecar)
    receipt = json.loads(sidecar.read_text())
    if (expected_data_sha256 != DATA_SHA256 or receipt.get('data_sha256') != expected_data_sha256
            or config.data.train_pool_replicas != 1 or config.data.test_pool_replicas != 1):
        raise ValueError('Need the original audited L108 fingerprint and unpooled split')
    original_fingerprint = trainer.file_fingerprint
    original_cache = mi_cache.load_or_build_mi_statistics

    def fingerprint(path):
        if Path(path).resolve() == data:
            if _stat(data) != before_data or _stat(sidecar) != before_sidecar:
                raise ValueError('Existing physical cache/sidecar stat identity changed')
            return expected_data_sha256
        return original_fingerprint(path)

    def prepared(cfg):
        if Path(cfg.data.cache).resolve() != data:
            raise ValueError('Trainer attempted a different physical cache')
        cfg.validate()
        values = torch.load(data, map_location='cpu', weights_only=True, mmap=True)
        if isinstance(values, dict) and values.get('schema_version') == 1:
            configurations = (values['train']['configurations'], values['test']['configurations'])
        elif isinstance(values, (tuple, list)) and len(values) == 4:
            # Original audited legacy layout: (TRAIN X, TRAIN labels, TEST X,
            # TEST labels). Only the physical configuration slots are read.
            configurations = (values[0], values[2])
        else:
            raise ValueError('Require original schema-one split dictionary or four-slot physical cache')
        result = []
        for observed, (split, count) in zip(configurations, (('train', 516), ('test', 252))):
            # Deliberately never access couplings/group_ids: the audited split
            # is reused, not recreated or checked by labels in a training run.
            if observed.shape != (count, 128, 108, 108, 1) or observed.dtype != torch.float64:
                raise ValueError('Original unpooled L108 configuration metadata differs')
            shapes = ScaleConfig((108, 108), cfg.training.max_rg_steps, cfg.model.block_factor).shapes()
            if min(shapes[-1]) < 5 or cfg.training.sample_batch > 128 or cfg.evaluation.samples > 128:
                raise ValueError('The declared decoder range/sample budget is incompatible with this scale chain')
            result.append(SimpleNamespace(configurations=observed, couplings=None, group_ids=None))
        return tuple(result)

    def cache_only(directory, key_data, build):
        del build
        def forbidden():
            raise RuntimeError('Predeclared empirical MI cache miss; rebuilding is forbidden')
        value, record = original_cache(directory, key_data, forbidden)
        if record.get('hit') is not True or record.get('key') not in MI_KEYS or record.get('cache_version') != 1:
            raise RuntimeError('Need a hit on the predeclared original block-two TRAIN/TEST MI statistics')
        return value, record

    with ExitStack() as stack:
        stack.enter_context(patch.object(trainer, 'file_fingerprint', fingerprint))
        stack.enter_context(patch.object(trainer, 'prepared_data', prepared))
        stack.enter_context(patch.object(mi_cache, 'load_or_build_mi_statistics', cache_only))
        yield dict(data_sha256=expected_data_sha256, data_stat=before_data, sidecar_sha256=sidecar_sha,
                   data_rehashed=False, data_full_audit_performed=False, coupling_labels_accessed=False)
    if _stat(data) != before_data or _stat(sidecar) != before_sidecar or _sha256(sidecar) != sidecar_sha:
        raise ValueError('Historical physical cache or its sidecar changed during training')


def validate_fresh_config(config):
    config.validate()
    m, t, loss = config.model, config.training, config.loss
    if (config.schema_version != 5 or m.dimension != 2 or m.categorical_states or m.width != 3
            or m.block_factor != 2 or m.theory_dim not in (1, 2, 3)
            or m.fixed_kernel or m.kernel_initialization != 'random' or m.kernel_sampling != 'bernoulli'
            or m.kernel_layout != 'nonoverlapping' or m.reconstruction_hidden or m.theory_head != 'nonlinear'
            or t.max_rg_steps != 2 or config.evaluation.max_rg_steps != 2
            or tuple(t.train_modules) != ('Q', 'C', 'R') or t.seed != 91
            or loss.balanced_reconstruction_weight or loss.isotropy_weight or loss.tangent_weight
            or loss.orientation_weight or loss.closure_metric != 'euclidean'
            or loss.coordinate_reduction != 'sum'
            or (loss.weights.anchor, loss.weights.coarse_consistency, loss.weights.separation,
                loss.weights.commutator, loss.rollout_weight) != (2., 5., 10., 5., 5.)):
        raise ValueError('Require the explicit fresh unsupervised b2 width3 depth-two seed91 independent QCR recipe')


def run_training(config, output, *, device, mi_cache_dir, initialization_receipt,
                 arm='natural', expected_data_sha256=DATA_SHA256):
    """Call the original trainer with only declared, bounded process-local hooks."""
    if arm not in ('control', 'natural'):
        raise ValueError('Choose the untouched probability-MSE control or natural representation')
    config_path, output = Path(config), Path(output)
    cfg = RunConfig.load(config_path)
    validate_fresh_config(cfg)
    if cfg.model.affine_reconstruction != (arm == 'control'):
        raise ValueError('Explicit control config needs affine_reconstruction=true; natural config needs false')
    if output.exists() or output.is_symlink() or Path(initialization_receipt).exists():
        raise FileExistsError('Refuse overwrite, resume, or duplicate initialization receipts')
    arguments = ['--config', str(config_path), '--output', str(output), '--device', str(device),
                 '--mi-cache-dir', str(mi_cache_dir), '--skip-final-report']
    with ExitStack() as stack:
        data_record = stack.enter_context(cached_data_hooks(cfg, expected_data_sha256))
        if arm == 'natural':
            stack.enter_context(coupling_representation_hooks())
        factory = trainer.build_model
        def record_initialization(*args, **kwargs):
            model = factory(*args, **kwargs)
            _exclusive_json(initialization_receipt, dict(status='complete', seed=cfg.training.seed,
                fresh=True, source_checkpoint=None, arm=arm, labels_used_in_training=False,
                C_tensor_sha256=_tensor_digest(model.rg_kernel.state_dict()),
                Q_backbone_tensor_sha256=_tensor_digest(model.theory_encoder.state_dict(), exclude_decoder=True),
                Q_tensor_sha256=_tensor_digest(model.theory_encoder.state_dict()),
                R_tensor_sha256=_tensor_digest(model.theory_rg.state_dict()),
                full_model_tensor_sha256=_tensor_digest(model.state_dict()), data_reuse=data_record,
                representation_override=representation_contract() if arm == 'natural' else None))
            return model
        stack.enter_context(patch.object(trainer, 'build_model', record_initialization))
        return trainer.main(arguments)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('control', 'natural'), default='natural')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--mi-cache-dir', type=Path, required=True)
    parser.add_argument('--initialization-receipt', type=Path, required=True)
    parser.add_argument('--data-sha256', default=DATA_SHA256)
    args = parser.parse_args(argv)
    return run_training(args.config, args.output, device=args.device, mi_cache_dir=args.mi_cache_dir,
        initialization_receipt=args.initialization_receipt, arm=args.arm,
        expected_data_sha256=args.data_sha256)


if __name__ == '__main__':
    main()
