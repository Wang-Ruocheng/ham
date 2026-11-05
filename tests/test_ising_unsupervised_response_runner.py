import argparse
import hashlib
import json

import pytest

from scripts.run_ising_unsupervised_response_ablation import build_plan


def setup_args(tmp_path):
    source = tmp_path / 'source' / 'run'
    report = source / 'full_report'
    report.mkdir(parents=True)
    (source / 'training_step_12000.pt').write_bytes(b'unsupervised numbered checkpoint')
    (source / 'checkpoint_last.pt').write_bytes(b'last checkpoint')
    manifest = source / 'manifest.json'
    manifest.write_text(json.dumps(dict(labels_used_in_training=False, q_supervision=None)))
    encoded = report / 'encodings.npz'
    encoded.write_bytes(b'public saved trajectory')
    (source.parent / 'job.json').write_text(json.dumps(dict(status='complete', workflow_complete=True,
        training=dict(actual_epochs=12000, manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest()),
        reports=dict(files={'full_report/encodings.npz': hashlib.sha256(encoded.read_bytes()).hexdigest()}))))
    return argparse.Namespace(source_run=source, source_report=report, checkpoint=12000,
                              output=tmp_path / 'new', updates=6000, alpha=1., learning_rate=.001)


def test_supervised_source_rejected(tmp_path):
    args = setup_args(tmp_path)
    path = args.source_run / 'manifest.json'
    path.write_text(json.dumps(dict(labels_used_in_training=True)))
    with pytest.raises(ValueError, match='K监督'):
        build_plan(args)


def test_changed_encoding_rejected(tmp_path):
    args = setup_args(tmp_path)
    (args.source_report / 'encodings.npz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='回执'):
        build_plan(args)


def test_existing_output_rejected(tmp_path):
    args = setup_args(tmp_path)
    args.output.mkdir()
    with pytest.raises(FileExistsError):
        build_plan(args)


def test_valid_plan_has_no_label_target_or_gpu(tmp_path):
    args = setup_args(tmp_path)
    plan = build_plan(args)
    assert plan['source_epoch'] == 12000
    assert plan['K_label_access_in_training'] is False
    assert plan['Kc_or_theory_target'] is False
    assert plan['extra_R_updates'] == 6000
    command = plan['commands'][0]
    assert 'experiments.ising_unsupervised_response_ablation' in command
    assert '--device' not in command and '--Kc' not in command and '--nu' not in command
    assert not args.output.exists()
