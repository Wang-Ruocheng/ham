"""Reproducible signed-KB b=3 experiment: sample, train QC, freeze QC, fit R, plot.

Run from the repository root. Existing run directories are deliberately rejected.
Each subprocess streams into pipeline.log and failures stop all later stages.
"""
import json
import argparse
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
QC = 'results/ising2d_symmetricKB_b3_QC6000_seed91'
RUN = 'results/ising2d_symmetricKB_b3_QC6000_R6000_seed91'
CACHE = 'data_cache/ising2d_symmetricKB_L36_n64_seed91'
CONTROL = 'results/ising2d_symmetricKB_b3_staged_pipeline_seed91'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset', choices=['signed-b3', 'nonnegative-b2', 'nonnegative-b2-Cweight200', 'original-b2-Cweight200'], default='signed-b3')
    args = parser.parse_args()
    qc, run, cache, control_name = QC, RUN, CACHE, CONTROL
    protocol = 'QC6000 then frozen-QC R6000, b=3, L36->12->4, seed91'
    if args.preset.startswith('nonnegative-b2'):
        qc = 'results/ising2d_nonnegativeKB_QC6000_seed91'
        run = 'results/ising2d_nonnegativeKB_QC6000_R6000_seed91'
        cache = 'data_cache/ising2d_nonnegativeKB_L16_n64_seed91'
        control_name = 'results/ising2d_nonnegativeKB_staged_pipeline_seed91'
        protocol = 'K>=0 subset, QC6000 then frozen-QC R6000, b=2, L16->8->4, seed91'
    reuse_cache = args.preset == 'nonnegative-b2-Cweight200'
    if reuse_cache:
        qc = 'results/ising2d_nonnegativeKB_Cweight200_QC6000_seed91'
        run = 'results/ising2d_nonnegativeKB_Cweight200_QC6000_R6000_seed91'
        control_name = 'results/ising2d_nonnegativeKB_Cweight200_staged_pipeline_seed91'
        protocol += ', QC kernel_noncollapse weight 200 (baseline 2), identical cached data'
    if args.preset == 'original-b2-Cweight200':
        reuse_cache = True
        qc = 'results/ising2d_originalKB_Cweight200_QC6000_seed91'
        run = 'results/ising2d_originalKB_Cweight200_QC6000_R6000_seed91'
        cache = 'results/fast_qcr_ising2d_focusedB_g2_L16_n64_1000_seed91'
        control_name = 'results/ising2d_originalKB_Cweight200_staged_pipeline_seed91'
        protocol = 'Original K=.20:.03:.68 cache, QC6000 then frozen-QC R6000, b=2, L16->8->4, seed91, C weight200'
    os.chdir(ROOT)
    control = Path(control_name)
    control.mkdir(exist_ok=False)
    status = dict(pid=os.getpid(), started=datetime.now(timezone.utc).isoformat(),
                  qc_run=qc, r_run=run, cache=cache, protocol=protocol)

    def save(**updates):
        status.update(updates)
        (control/'status.json').write_text(json.dumps(status, indent=2)+'\n')

    commands = [
        ('data', ['scripts.generate_symmetric_ising2d', '--output', CACHE, '--sites', '36', '--workers', '2', '--seed', '91']),
        ('QC', ['recurrent_rg', 'train', '--config', 'configs/ablations/ising2d_symmetricKB_b3_QC_stage1.json', '--output', QC]),
        ('R', ['recurrent_rg', 'train', '--config', 'configs/ablations/ising2d_symmetricKB_b3_R_stage2.json', '--initialize-from', QC+'/qcr_model.pt', '--output', RUN]),
        ('report', ['scripts.report_joint_isotropy', '--run', RUN]),
        ('flow', ['scripts.plot_flow_interpolated_KB', '--run', RUN, '--artifacts', RUN+'/full_report', '--critical-reference', '--signed-reference']),
        ('stage_audit', ['scripts.report_staged_qcr', '--qc', QC, '--run', RUN, '--joint', 'results/ising2d_symmetricKB_g2_decimationInit_12000_seed91']),
    ]
    if args.preset.startswith('nonnegative-b2'):
        commands = [
            ('data', ['scripts.prepare_nonnegative_ising2d', '--source', 'data_cache/ising2d_symmetricKB_L16_n64_seed91/ensembles.pt', '--output', cache]),
            ('QC', ['recurrent_rg', 'train', '--config', 'configs/ablations/ising2d_nonnegativeKB_QC_stage1.json', '--output', qc]),
            ('R', ['recurrent_rg', 'train', '--config', 'configs/ablations/ising2d_nonnegativeKB_R_stage2.json', '--initialize-from', qc+'/qcr_model.pt', '--output', run]),
            ('report', ['scripts.report_joint_isotropy', '--run', run]),
            ('flow', ['scripts.plot_flow_interpolated_KB', '--run', run, '--artifacts', run+'/full_report', '--critical-reference']),
            ('stage_audit', ['scripts.report_staged_qcr', '--qc', qc, '--run', run, '--joint', 'results/ising2d_symmetricKB_g2_decimationInit_12000_seed91']),
        ]
    if reuse_cache:
        commands = commands[1:]
        commands[0] = ('QC', ['recurrent_rg', 'train', '--config', 'configs/ablations/ising2d_nonnegativeKB_Cweight200_QC_stage1.json', '--output', qc])
    if args.preset == 'original-b2-Cweight200':
        commands = [
            ('QC', ['recurrent_rg', 'train', '--config', 'configs/ablations/ising2d_originalKB_Cweight200_QC_stage1.json', '--output', qc]),
            ('R', ['recurrent_rg', 'train', '--config', 'configs/ablations/ising2d_originalKB_Cweight200_R_stage2.json', '--initialize-from', qc+'/qcr_model.pt', '--output', run]),
            ('report', ['scripts.report_joint_isotropy', '--run', run]),
            ('flow', ['scripts.plot_flow_interpolated_KB', '--run', run, '--artifacts', run+'/full_report', '--critical-reference']),
            ('stage_audit', ['scripts.report_staged_qcr', '--qc', qc, '--run', run, '--joint', 'results/ising2d_g2_relative_local_decimationInit_12000_seed91']),
        ]
    try:
        if Path(qc).exists() or Path(run).exists() or (Path(cache).exists() and not reuse_cache):
            raise FileExistsError('Refusing to overwrite existing cache or experiment')
        if reuse_cache and not (Path(cache)/'ensembles.pt').is_file():
            raise FileNotFoundError('The matched nonnegative-K cache is required')
        with (control/'pipeline.log').open('w', buffering=1) as log:
            for stage, args in commands:
                save(state='running', stage=stage, command=args, updated=datetime.now(timezone.utc).isoformat())
                print(f'\nStage: {stage}', file=log, flush=True)
                subprocess.run([sys.executable, '-u', '-m', *args], stdout=log,
                               stderr=subprocess.STDOUT, check=True)
        save(state='complete', completed=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        save(state='failed', error=repr(exc), updated=datetime.now(timezone.utc).isoformat())
        raise


if __name__ == '__main__':
    main()
