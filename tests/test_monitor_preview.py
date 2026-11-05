import json
import os
import sys
from types import SimpleNamespace

import pytest

from scripts import preview_latest_checkpoint, report_r_plateau_loss
from scripts.preview_latest_checkpoint import latest_saved_checkpoint, preview_stages


def checkpoint(tmp_path, epoch, mtime):
    path = tmp_path / f'training_step_{epoch}.pt'
    path.write_bytes(b'checkpoint')
    os.utime(path, (mtime, mtime))
    return path


def test_latest_checkpoint_uses_epoch_not_lexical_order(tmp_path):
    checkpoint(tmp_path, 200, 1.)
    latest = checkpoint(tmp_path, 1000, 1.)
    assert latest_saved_checkpoint(tmp_path, now=100.) == (1000, latest)


def test_just_written_checkpoint_is_not_read(tmp_path):
    settled = checkpoint(tmp_path, 400, 1.)
    checkpoint(tmp_path, 600, 99.)
    assert latest_saved_checkpoint(tmp_path, now=100.) == (400, settled)


def test_empty_or_absent_checkpoints_are_not_used(tmp_path):
    (tmp_path / 'training_step_200.pt').touch()
    (tmp_path / 'checkpoint_last.pt').write_bytes(b'not-numbered')
    with pytest.raises(FileNotFoundError):
        latest_saved_checkpoint(tmp_path, now=100.)


def test_scalar_preview_retains_existing_reports(tmp_path):
    stages = preview_stages(tmp_path, 200, tmp_path / 'preview', 1)
    assert [stage for stage, _ in stages] == ['scalar', 'phase', 'loss']
    assert 'scripts.report_scalar_joint' in stages[0][1]
    assert '--sampling-repeats' in stages[0][1]
    assert 'scripts.report_mi_training' in stages[-1][1]


def test_positive_single_ising_preview_uses_one_transition_annotation(tmp_path):
    stages=preview_stages(tmp_path,200,tmp_path/'preview',1,coupling_names=['K'],
        physical_model='Zero-field square-lattice classical Ising, positive K in [0.2,0.68]')
    assert '--positive-ising' in stages[1][1]
    assert stages[2][0] == 'exponents'
    assert 'scripts.report_ising_scalar_exponents' in stages[2][1]
    assert stages[2][1][-4:] == ['--checkpoint', '200', '--output', str(tmp_path/'preview')]
    historical=preview_stages(tmp_path,200,tmp_path/'old',1,coupling_names=['K'],
        physical_model='Signed K square Ising')
    assert '--positive-ising' not in historical[1][1]
    assert all(name != 'exponents' for name, _ in historical)


def test_vector_preview_uses_vector_report_without_mi(tmp_path):
    stages = preview_stages(tmp_path, 200, tmp_path / 'preview', 2)
    assert [stage for stage, _ in stages] == ['phase', 'loss']
    assert 'scripts.report_dual_ising_vector' in stages[0][1]
    assert all('experiments.rsmi_separate_qcr' not in cmd for _, cmd in stages)
    assert all(cmd[-2:] == ['--threads', '1'] for _, cmd in stages)


@pytest.mark.parametrize('theory_dim', [1, 2])
def test_r_plateau_preview_uses_saved_closure_report(theory_dim, tmp_path):
    stages = preview_stages(tmp_path, 200, tmp_path / 'preview', theory_dim,
                            training_driver='frozen_qc_r_plateau')
    assert stages[-1][0] == 'loss'
    assert 'scripts.report_r_plateau_loss' in stages[-1][1]
    assert all('scripts.report_mi_training' not in command for _, command in stages)
    assert stages[-1][1][-2:] == ['--threads', '1']


@pytest.mark.parametrize('training_driver', [None, 'rsmi_separate_qcr', 'frozen_qc_r_continue'])
def test_other_preview_drivers_keep_existing_loss_report(training_driver, tmp_path):
    stages = preview_stages(tmp_path, 200, tmp_path / 'preview', 2,
                            training_driver=training_driver)
    assert 'scripts.report_mi_training' in stages[-1][1]


def test_preview_main_reads_training_driver_from_manifest(tmp_path, monkeypatch):
    checkpoint(tmp_path, 200, 1.)
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(
        config=dict(model=dict(theory_dim=2), data=dict(coupling_names=['K', 'B'])),
        training_driver='frozen_qc_r_plateau',
    )))
    commands = []
    monkeypatch.setattr(sys, 'argv', ['preview', '--run', str(tmp_path)])
    monkeypatch.setattr(preview_latest_checkpoint.os, 'nice', lambda _: None)
    monkeypatch.setattr(preview_latest_checkpoint.subprocess, 'run',
                        lambda command, **kwargs: commands.append(command))
    assert preview_latest_checkpoint.main() == 0
    assert 'scripts.report_r_plateau_loss' in commands[-1]
    assert all('scripts.report_mi_training' not in command for command in commands)


@pytest.mark.parametrize('selection', [200, None])
def test_r_plateau_loss_uses_only_selected_checkpoint_history(selection, tmp_path, monkeypatch):
    run = tmp_path / 'run'
    run.mkdir()
    newer_history = run / 'history_progress.json'
    newer_history.write_text(json.dumps([dict(epoch=999, extra_R_updates=999)]))
    before = newer_history.read_bytes()
    history = [dict(epoch=200, extra_R_updates=0,
                    train=dict(one_step=.1, multistep=.2, total=1.5),
                    test=dict(one_step=.2, multistep=.3, total=2.5))]
    selected_path = run / (f'training_step_{selection}.pt' if selection is not None
                           else 'checkpoint_last.pt')
    calls = []

    def load_saved(source, requested_checkpoint):
        calls.append((source, requested_checkpoint))
        return SimpleNamespace(checkpoint=selected_path, payload=dict(epoch=200, history=history))

    plotted = []

    def plot_saved(rows, out):
        plotted.append(rows)
        (out / 'loss.png').write_bytes(b'test image')

    monkeypatch.setattr(report_r_plateau_loss, 'load_run', load_saved)
    monkeypatch.setattr(report_r_plateau_loss, 'plot_loss', plot_saved)
    monkeypatch.setattr(report_r_plateau_loss.torch, 'set_num_threads', lambda _: None)
    out = tmp_path / 'preview'
    argv = ['--run', str(run), '--output', str(out), '--threads', '1']
    if selection is not None:
        argv += ['--checkpoint', str(selection)]
    records = report_r_plateau_loss.main(argv)

    assert calls == [(run, selection if selection is not None else 'checkpoint_last.pt')]
    assert plotted == [history]
    assert records['history'] == history
    assert json.loads((out / 'loss_metrics.json').read_text())['history'] == history
    assert (out / 'loss.png').is_file()
    assert newer_history.read_bytes() == before
