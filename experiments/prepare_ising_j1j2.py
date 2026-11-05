"""Label-free J1--J2 Ising adapters for the unmodified default QCR trainer.

Physical parameters belong only to a separate JSON evaluation index.  The
training cache contains int8 configurations, no coupling tensor, and opaque
integer source identities.  This module neither samples spins nor trains a
model; in particular preparation is not C selection or model pretraining.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re

from recurrent_rg.config import (
    DataConfig, EvaluationConfig, LossConfig, ModelConfig, QCRWeights,
    RunConfig, TrainingConfig,
)


TRAIN_RHOS = (0., .2, .35, .7, 1.)
HELDOUT_RHOS = (.3, .8)
SEEDS = (91, 92, 93)
UPDATES = 12000
PROFILE_UPDATES = 20


def make_config(cache, *, theory_dim=5, seed=91, updates=UPDATES, threads=2,
                sample_batch=128, system_batch=16, max_rg_steps=2,
                evaluation_systems=24):
    """Explicit schema-5 recipe; only dg=1/2/5 and seeds differ across arms.

    Small execution tests may reduce samples/updates/geometry through their
    dedicated cache; the objectives, random initialization, and gradients are
    not changed.  No diagnostic run is a production checkpoint source.
    """
    if theory_dim not in (1, 2, 5):
        raise ValueError('This registered campaign uses theory_dim 1, 2, or 5')
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('A nonnegative integer training seed is required')
    cfg = RunConfig(
        schema_version=5,
        data=DataConfig(cache=str(cache), physical_model='ising2d_j1j2',
            coupling_names=(), spin_flip_augment=False, spin_flip_symmetric=True,
            train_pool_replicas=1, test_pool_replicas=1),
        model=ModelConfig(dimension=2, theory_dim=theory_dim, hidden=96, width=3,
            block_factor=3, kernel_layout='nonoverlapping', fixed_kernel=False,
            binary_lookup=True, kernel_initialization='random',
            kernel_sampling='bernoulli', kernel_temperature=1.,
            affine_reconstruction=True, reconstruction_hidden=0,
            theory_head='nonlinear', local_features=12),
        loss=LossConfig(closure_objective='all_start_anchor',
            closure_scale_weights=None, rollout_weight=5.,
            kernel_sign_weight=1., kernel_noncollapse_mode='variance_floor_ensemble',
            kernel_variance_ratio=1.,
            weights=QCRWeights(anchor=2., coarse_consistency=5.,
                kernel_noncollapse=2., separation=10., commutator=5.)),
        training=TrainingConfig(max_rg_steps=max_rg_steps, epochs=updates,
            system_batch=system_batch, sample_batch=sample_batch,
            learning_rate=.001, seed=seed, threads=threads,
            train_modules=('Q', 'C', 'R'), checkpoint_every=min(1000, updates),
            log_every=min(200, updates), gradient_mode='qc_r_separate',
            lr_schedule='multistep', lr_milestones=(6000, 9000), lr_gamma=.3),
        evaluation=EvaluationConfig(max_rg_steps=max_rg_steps,
            samples=sample_batch, systems=evaluation_systems),
    )
    cfg.validate()
    from experiments.rsmi_separate_qcr import build_parser, validate_recipe
    args = build_parser().parse_args(['--config', 'unused.json', '--output', 'unused'])
    validate_recipe(args, cfg)
    # Round-trip deliberately serializes an explicit closure mode, rather than
    # relying on fresh-driver policy or changing historical loading semantics.
    return cfg.to_dict()


def campaign_configs(cache, *, seeds=SEEDS, theory_dims=(5,), **kwargs):
    """Build only the requested arms. dg=1/2 are not launched implicitly."""
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Require a nonempty set of unique training seeds')
    if not theory_dims or len(set(theory_dims)) != len(theory_dims):
        raise ValueError('Require nonempty unique latent dimensions')
    return {f'dg{dg}_seed{seed}': make_config(cache, theory_dim=dg, seed=seed, **kwargs)
            for dg in theory_dims for seed in seeds}


def training_command(python, config, output, *, device='cpu', mi_cache_dir=None,
                     benchmark_warmup=None):
    """Fresh, fixed-length training with no source, resume, or legacy flags.

    The generic report assumes ordinary K-labelled Ising data.  A campaign
    report must instead join the external physical index after training.
    """
    if not isinstance(device, str) or not re.fullmatch(r'cpu|cuda:[0-9]+', device):
        raise ValueError('Choose cpu or an explicit physical cuda:<index>')
    if mi_cache_dir is not None and device == 'cpu':
        raise ValueError('CPU execution does not accept --mi-cache-dir')
    command = [str(python), '-m', 'experiments.rsmi_separate_qcr',
               '--config', str(config), '--output', str(output), '--device', device,
               '--skip-final-report']
    if device.startswith('cuda:'):
        command.append('--mi-test-cpu')
    if mi_cache_dir is not None:
        command += ['--mi-cache-dir', str(mi_cache_dir)]
    if benchmark_warmup is not None:
        if (isinstance(benchmark_warmup, bool) or not isinstance(benchmark_warmup, int)
                or benchmark_warmup < 0):
            raise ValueError('benchmark_warmup must be a nonnegative integer')
        command += ['--benchmark-warmup', str(benchmark_warmup)]
    return command


def mi_preprocessing_command(python, config, output, *, cache_dir=None):
    """CPU data-only full-shell MI preprocessing using the existing key API."""
    command = [str(python), '-m', 'scripts.prepare_qcr_mi_cache',
               '--config', str(config), '--output', str(output)]
    if cache_dir is not None:
        command += ['--cache-dir', str(cache_dir)]
    return command


def protocol_contract():
    """Public declaration, not a statement that sampling or training passed."""
    return dict(
        training_driver='experiments.rsmi_separate_qcr', labels_used_in_training=False,
        source_checkpoint=None, fresh_QCR=True, fresh_Adam=True,
        C_initialization='random; trainable probabilistic nonoverlapping b3',
        C_objective='-2 empirical full-shell MI +2 ensemble variance floor +1 color NLL',
        Q_objective='shared affine motif reconstruction10 + subset consistency20/5',
        R_objective='10/S sum_t (1/t) sum_{s<t} ||R^(t-s)(sg g_s)-sg g_t||^2',
        gradient_routing='Q representation only; C own objective only; R closure only',
        train_rhos=list(TRAIN_RHOS), heldout_rhos=list(HELDOUT_RHOS),
        train_groups=list(range(8)), same_rho_test_groups=list(range(8, 12)),
        training_seeds=list(SEEDS), main_latent_dimension=5,
        future_dimension_ablation=dict(dimensions=[1, 2], automatic_execution=False),
        updates=UPDATES, optimizer='fresh Adam .001; milestones6000/9000 gamma.3',
        physical_labels='separate evaluation metadata only; no physical labels in cache',
        groups='opaque integer [rho_index, temperature_index, independent_repeat]',
        independent_unit='one entire parallel-tempering repeat across all temperatures',
        default_final_report=False, report_requires='dedicated label-free external-index evaluation',
        automatic_retry=False, resume=False, thermal_nu=None,
    )


def validate_label_free_cache(config_path, metadata_path=None):
    """Check a newly assembled cache's label and whole-repeat split boundary.

    Hashing and spin validation deliberately read the new cache.  This helper
    must never be pointed at historical physical data or a checkpoint.  It
    does not calculate MI, run a model, or certify physical equilibration.
    """
    import torch
    from recurrent_rg.runner import file_fingerprint, prepared_data
    cfg = RunConfig.load(config_path)
    if (cfg.data.physical_model != 'ising2d_j1j2' or cfg.data.coupling_names
            or cfg.data.train_pool_replicas != 1 or cfg.data.test_pool_replicas != 1):
        raise ValueError('Expected unpooled J1--J2 cache without coupling names')
    train, test = prepared_data(cfg)
    digest = file_fingerprint(cfg.data.cache)
    external = None if metadata_path is None else json.loads(Path(metadata_path).read_text())
    if external is not None and (external.get('schema_version') != 1
            or external.get('data_sha256') != digest):
        raise ValueError('External physical metadata fingerprint does not match cache')
    seen = {}
    splits = {}
    for split, ds in (('train', train), ('test', test)):
        x, groups = ds.configurations, ds.group_ids
        if ds.couplings is not None:
            raise ValueError('Physical coupling/temperature tensors must not enter training cache')
        if x.dtype != torch.int8:
            raise ValueError('New J1--J2 cache must retain int8 binary configurations')
        if (groups is None or groups.dtype != torch.int64 or groups.shape != (len(x), 3)
                or bool((groups < 0).any()) or bool((groups[:, 2] >= 12).any())):
            raise ValueError('Expected nonnegative int64 opaque [rho,Tindex,repeat] identities')
        identities = [tuple(row) for row in groups.tolist()]
        if len(set(identities)) != len(identities):
            raise ValueError('Duplicate physical ensemble identity in cache split')
        rows = None if external is None else external.get(split)
        if rows is not None and (not isinstance(rows, list) or len(rows) != len(x)):
            raise ValueError('External physical rows do not match cache split')
        if external is not None and rows is None:
            raise ValueError('External metadata must index both train and test')
        for index, identity in enumerate(identities):
            rho_index, temperature_index, group = identity
            # A PT repeat is a physical-source unit across temperature slots.
            repeat_identity = (rho_index, group)
            if repeat_identity in seen and seen[repeat_identity] != split:
                raise ValueError('Entire PT repeats must remain in the same split')
            seen[repeat_identity] = split
            if rows is None:
                continue
            row = rows[index]
            if (row.get('rho_index'), row.get('temperature_index'), row.get('group')) != identity:
                raise ValueError('External physical row order does not match opaque identities')
            rho = row.get('rho')
            if split == 'train':
                valid = rho in TRAIN_RHOS and group < 8 and row.get('partition') == 'train'
            else:
                valid = ((rho in TRAIN_RHOS and group >= 8
                          and row.get('partition') == 'same_rho_new_group')
                         or (rho in HELDOUT_RHOS and row.get('partition') == 'heldout_rho'))
            if not valid:
                raise ValueError('External physical partition violates the registered split')
        splits[split] = dict(shape=list(x.shape), dtype=str(x.dtype), ensembles=len(x),
                             physical_labels_in_cache=False, opaque_group_columns=3)
    if file_fingerprint(cfg.data.cache) != digest:
        raise ValueError('Training data changed during label-boundary validation')
    return dict(status='complete', data_sha256=digest, splits=splits,
                labels_used_in_training=False, model_initialized=False,
                optimizer_updates=0, equilibrium_certified=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=91)
    parser.add_argument('--theory-dim', type=int, choices=(1, 2, 5), default=5)
    parser.add_argument('--updates', type=int, default=UPDATES)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args(argv)
    config = make_config(args.cache, seed=args.seed, theory_dim=args.theory_dim,
                         updates=args.updates, threads=args.threads)
    # Exclusive creation preserves every previous configuration and result.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(config, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    return deepcopy(config)


if __name__ == '__main__':
    main()
