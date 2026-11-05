"""Launch a fresh default QCR experiment followed by checkpoint-only reports."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from recurrent_rg.config import RunConfig
from recurrent_rg.runner import file_fingerprint


def phase_report_module(cfg):
    """Route diagnostics by explicit physical metadata, not directory names."""
    if cfg.model.dimension != 2 or cfg.model.theory_dim != 2:
        raise ValueError('This pipeline requires a two-dimensional vector experiment')
    names = tuple(cfg.data.coupling_names)
    if names == ('K', 'B') and not cfg.model.categorical_states:
        return 'scripts.report_ising2d_KB_vector'
    if names in (('K1', 'K2'), ('K1',), ('K2',)) and cfg.model.categorical_states == 4:
        return 'scripts.report_dual_ising_vector'
    raise ValueError('Unsupported physical metadata for this report pipeline')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--log-prefix', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, help='Existing run for loss-report comparison only')
    generation = parser.add_mutually_exclusive_group()
    generation.add_argument('--generate-grid', action='store_true',
                        help='Generate a fresh independent dual-Ising grid before training')
    generation.add_argument('--generate-field-grid', action='store_true',
                            help='Generate a fresh single-layer Ising K/B grid')
    parser.add_argument('--data-length', type=int, default=36)
    parser.add_argument('--data-replicas', type=int, default=2)
    parser.add_argument('--grid-points', type=int, default=14)
    parser.add_argument('--grid-max', type=float, default=1.3)
    parser.add_argument('--b-max', type=float, default=.01)
    args = parser.parse_args()
    cfg = RunConfig.load(args.config)
    report_module = phase_report_module(cfg)
    if args.output.exists():
        raise FileExistsError(args.output)
    cache = Path(cfg.data.cache)
    generate = args.generate_grid or args.generate_field_grid
    if generate:
        if args.generate_grid and (tuple(cfg.data.coupling_names) != ('K1', 'K2') or cfg.model.categorical_states != 4):
            raise ValueError('Grid generation requires the independent four-state dual-Ising configuration')
        if args.generate_field_grid and (tuple(cfg.data.coupling_names) != ('K', 'B') or cfg.model.categorical_states):
            raise ValueError('Field generation requires a binary single-layer K/B configuration')
        if args.generate_field_grid and cfg.training.sample_batch != 64:
            raise ValueError('The existing field generator records 64 configurations per ensemble')
        if cache.name != 'ensembles.pt' or cache.parent.exists():
            raise FileExistsError(f'Refusing to replace an existing cache directory: {cache.parent}')
        if args.data_length % cfg.model.block_factor ** cfg.training.max_rg_steps:
            raise ValueError('Data length must support every requested nonoverlapping RG scale')
    prefix = args.log_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    status_file = prefix.with_name(prefix.name + '_pipeline.json')
    if status_file.exists():
        raise FileExistsError(status_file)
    status = dict(pipeline_pid=os.getpid(), started_at=time.time(), phase='starting',
                  target_epoch=cfg.training.epochs, output=str(args.output.resolve()),
                  config=str(args.config.resolve()),
                  data_sha256=None if generate else file_fingerprint(cache),
                  training='Fresh Q/C/R; independent gradients; locked C/R objectives; no wall limit',
                  additional_Q_balanced_weight=cfg.loss.balanced_reconstruction_weight,
                  comparison_baseline=str(args.baseline.resolve()) if args.baseline else None)
    commands = []
    if args.generate_grid:
        command = [sys.executable, '-m', 'scripts.generate_dual_ising2d',
                   '--output', str(cache.parent), '--grid', '--length', str(args.data_length),
                   '--samples', str(cfg.training.sample_batch), '--replicas', str(args.data_replicas),
                   '--grid-points', str(args.grid_points), '--grid-max', str(args.grid_max),
                   '--seed', str(cfg.training.seed), '--skip-probes']
        commands.append(('data_generation', command))
        status['data_generation'] = dict(length=args.data_length, samples=cfg.training.sample_batch,
                                        replicas=args.data_replicas, points_per_axis=args.grid_points,
                                        maximum=args.grid_max, seed=cfg.training.seed,
                                        skip_extra_probes=True)
    if args.generate_field_grid:
        command = [sys.executable, '-m', 'scripts.generate_symmetric_ising2d',
                   '--output', str(cache.parent), '--grid', 'positive-uniform-B',
                   '--sites', str(args.data_length), '--b-max', str(args.b_max),
                   '--workers', str(cfg.training.threads), '--seed', str(cfg.training.seed)]
        commands.append(('data_generation', command))
        status['data_generation'] = dict(length=args.data_length, samples=64, replicas=2,
                                        K_range=[.2, .68], K_step=.03,
                                        B_range=[-args.b_max, args.b_max], B_points=21,
                                        equilibration_sweeps=512, sweeps_between=16,
                                        seed=cfg.training.seed, spin_flip_augment=False)
    commands += [
        ('training', [sys.executable, '-m', 'experiments.rsmi_separate_qcr',
                      '--config', str(args.config), '--output', str(args.output)]),
        ('phase_report', [sys.executable, '-m', report_module, '--run', str(args.output)]),
        ('loss_report', [sys.executable, '-m', 'scripts.report_mi_training', '--run', str(args.output)]),
    ]
    if args.baseline:
        commands[-1][1].extend(['--baseline',str(args.baseline)])
    try:
        for name, command in commands:
            if name == 'training':
                status['data_sha256'] = file_fingerprint(cache)
            log = prefix.with_name(prefix.name + '_' + name + '.log')
            with log.open('x') as stream:
                child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                status.update(phase=name, child_pid=child.pid, stage_log=str(log.resolve()), stage_started_at=time.time())
                status_file.write_text(json.dumps(status, indent=2) + '\n')
                print(f'{name}: pid={child.pid}, log={log}', flush=True)
                returncode = child.wait()
            if returncode:
                status.update(phase='failed', failed_stage=name, returncode=returncode, ended_at=time.time())
                status_file.write_text(json.dumps(status, indent=2) + '\n')
                return returncode
        summary = json.loads((args.output / 'summary.json').read_text())
        if summary['actual_epochs'] != cfg.training.epochs:
            raise RuntimeError('Training did not reach the requested endpoint')
        status.update(phase='completed', actual_epochs=summary['actual_epochs'], ended_at=time.time())
        status_file.write_text(json.dumps(status, indent=2) + '\n')
        return 0
    except Exception as error:
        status.update(phase='failed', error=repr(error), ended_at=time.time())
        status_file.write_text(json.dumps(status, indent=2) + '\n')
        raise


if __name__ == '__main__':
    sys.exit(main())
