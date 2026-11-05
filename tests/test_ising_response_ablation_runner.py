import argparse
import hashlib
import json

import pytest

from scripts.run_ising_response_loss_ablation import build_plan, execute


def arguments(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    checkpoint = source / 'training_step_9000.pt'
    checkpoint.write_bytes(b'source saved weights')
    (source / 'manifest.json').write_text('{}')
    report = tmp_path / 'reports' / 'epoch_9000' / 'full_report'
    report.mkdir(parents=True)
    (report / 'encodings.npz').write_bytes(b'saved public encoding')
    (report.parent / 'checkpoint_summary.json').write_text(json.dumps(dict(
        status='complete', actual_epochs=9000,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())))
    return argparse.Namespace(source_run=source, source_report=report,
                              checkpoint=9000, output=tmp_path / 'new',
                              updates=6000, alpha=1., learning_rate=.001)


def test_plan_has_fixed_matched_budget_and_no_gpu_or_theory_target(tmp_path):
    args = arguments(tmp_path)
    plan = build_plan(args)
    train, report = plan['commands']
    assert plan['source_epoch'] == 9000 and plan['extra_R_updates'] == 6000
    assert train[train.index('--updates') + 1] == '6000'
    assert train[train.index('--threads') + 1] == '1'
    assert '--device' not in train and '--nu' not in train
    assert report[-1] == str(args.output / 'report')
    assert not args.output.exists()


def test_checkpoint_receipt_mismatch_refused(tmp_path):
    args = arguments(tmp_path)
    (args.source_run / 'training_step_9000.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='回执'):
        build_plan(args)


def test_existing_output_refused(tmp_path):
    args = arguments(tmp_path)
    args.output.mkdir()
    with pytest.raises(FileExistsError):
        build_plan(args)


@pytest.mark.parametrize('field,value', [('updates', 0), ('alpha', float('nan')),
                                         ('learning_rate', float('inf'))])
def test_invalid_budget_or_hyperparameters(tmp_path, field, value):
    args = arguments(tmp_path)
    setattr(args, field, value)
    with pytest.raises(ValueError):
        build_plan(args)


def test_failed_training_does_not_start_report_or_retry(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    calls = []

    class Child:
        pid = 77

        def wait(self):
            return 9

    def launch(command, **kwargs):
        calls.append(command)
        return Child()

    monkeypatch.setattr('scripts.run_ising_response_loss_ablation.subprocess.Popen', launch)
    assert execute(args) == 1
    saved = json.loads((args.output / 'job.json').read_text())
    assert len(calls) == 1
    assert saved['status'] == 'failed' and saved['requires_user_action']
    assert saved['stages'][0]['returncode'] == 9
    assert not saved['workflow_complete']
    with pytest.raises(FileExistsError):
        execute(args)
