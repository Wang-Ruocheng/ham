"""Synthetic plumbing checks only; these are not sampling or training results."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import report_ising_j1j2_unthermalized as report


def test_diagnostic_partitions_never_fabricate_heldout_rho():
    metadata = dict(train=[dict(rho_index=0, partition='train')],
                    test=[dict(rho_index=0, partition='same_rho_new_group')])
    partitions = report.diagnostic_partitions(metadata)
    assert set(partitions) == {'train', 'same_rho_new_group'}
    assert 'heldout_rho' not in partitions
    metadata['test'][0]['partition'] = 'heldout_rho'
    with pytest.raises(ValueError, match='伪造'):
        report.diagnostic_partitions(metadata)
    metadata['test'][0] = dict(rho_index=1, partition='same_rho_new_group')
    with pytest.raises(ValueError, match='全部 rho'):
        report.diagnostic_partitions(metadata)


def test_warning_receipt_is_required_bound_and_not_modified(tmp_path):
    source = tmp_path / 'quality_warning.json'
    source.write_text(json.dumps(dict(status='quality_warning', passed=False)))
    original = source.read_bytes()
    value, reference = report.warning_reference(source)
    assert value['passed'] is False
    assert reference['source_sha256'] == report.base.sha256(source)
    assert reference['blocking_for_this_diagnostic'] is False
    assert source.read_bytes() == original
    alias = tmp_path / 'alias.json'
    alias.symlink_to(source)
    with pytest.raises(ValueError, match='非符号链接'):
        report.warning_reference(alias)
    source.write_text('{}')
    with pytest.raises(ValueError, match='非空'):
        report.warning_reference(source)


def test_all_new_figures_get_visible_banner_without_changing_plot_pixels(tmp_path):
    from PIL import Image
    from scripts.report_ising_zh import select_chinese_font

    pixels = np.zeros((31, 900, 3), dtype=np.uint8)
    pixels[..., 0] = 12
    pixels[..., 1] = 73
    pixels[..., 2] = 111
    for name in report.REQUIRED_FIGURES:
        Image.fromarray(pixels).save(tmp_path / name)
    records = report.add_warning_banners(tmp_path, font=select_chinese_font())
    assert set(records) == set(report.REQUIRED_FIGURES)
    for name, record in records.items():
        with Image.open(tmp_path / name) as saved:
            actual = np.asarray(saved)
        header = record['warning_header_height']
        assert header > 30
        np.testing.assert_array_equal(actual[header:], pixels)
        assert np.any(np.all(actual[:header] == [160, 0, 0], axis=-1))
        assert record['warning_lines'] == list(report.WARNING_LINES)
    (tmp_path / report.REQUIRED_FIGURES[0]).unlink()
    with pytest.raises(RuntimeError, match='图不完整'):
        report.add_warning_banners(tmp_path, font=select_chinese_font())


def synthetic_run(tmp_path):
    """Create an untrained toy endpoint fixture; it is not a scientific run."""
    from recurrent_rg.checkpoints import build_model, isolated_rng
    from recurrent_rg.config import (DataConfig, EvaluationConfig, ModelConfig,
                                     RunConfig, TrainingConfig)

    torch.set_num_threads(1)
    def row(temperature_index, group, partition):
        return dict(rho_index=0, temperature_index=temperature_index, group=group,
                    rho=0., temperature=1. + temperature_index,
                    condition_id=f'c{temperature_index}', partition=partition)
    metadata = dict(schema_version=1,
        train=[row(0, 0, 'train'), row(1, 0, 'train')],
        test=[row(0, 8, 'same_rho_new_group'), row(1, 8, 'same_rho_new_group')])
    generator = torch.Generator().manual_seed(17)
    data = dict(schema_version=1)
    for split in ('train', 'test'):
        data[split] = dict(configurations=(2 * torch.randint(2, (2, 128, 12, 12, 1),
            generator=generator) - 1).to(torch.int8), couplings=None,
            group_ids=torch.tensor([[r['rho_index'], r['temperature_index'], r['group']]
                                    for r in metadata[split]]))
    cache = tmp_path / 'data.pt'
    torch.save(data, cache)
    fingerprint = report.base.sha256(cache)
    metadata['data_sha256'] = fingerprint
    metadata_path = tmp_path / 'metadata.json'
    metadata_path.write_text(json.dumps(metadata))
    quality = tmp_path / 'quality_warning.json'
    quality.write_text(json.dumps(dict(diagnostic_only=True,
        equilibrium_not_validated=True, failures=['synthetic quality warning'])))
    cfg = RunConfig(data=DataConfig(cache=str(cache), physical_model='ising2d'),
        model=ModelConfig(dimension=2, theory_dim=2, hidden=8, local_features=4),
        training=TrainingConfig(max_rg_steps=1, epochs=12000, gradient_mode='qc_r_separate'),
        evaluation=EvaluationConfig(max_rg_steps=1, samples=128, systems=2))
    cfg.validate()
    with isolated_rng('cpu', 21):
        model = build_model(cfg.model, 1)
    run = tmp_path / 'synthetic_fixture_run'
    run.mkdir()
    contract = dict(config=cfg.to_dict(), labels_used_in_training=False, data_sha256=fingerprint)
    (run / 'manifest.json').write_text(json.dumps(contract))
    (run / 'summary.json').write_text(json.dumps(dict(actual_epochs=12000)))
    history = [dict(epoch=1, train_commutator=.02, test_commutator=.03),
               dict(epoch=12000, train_commutator=.001, test_commutator=.002)]
    torch.save(dict(model=model.state_dict(), epoch=12000, metadata=contract, history=history),
               run / 'training_step_12000.pt')
    return SimpleNamespace(run=run, data_metadata=metadata_path, quality_warning=quality,
        output=tmp_path / 'report', max_seconds=60., draws=5, font=None), history


def test_synthetic_frozen_report_all_warnings_and_formal_hash_validator(tmp_path):
    from scripts.run_ising_j1j2 import validate_report

    args, history = synthetic_run(tmp_path)
    inputs_before = {path: report.base.sha256(path) for path in
        (args.run / 'training_step_12000.pt', args.data_metadata, args.quality_warning)}
    result = report.run_report(args)
    for key in ('diagnostic_only', 'equilibrium_not_validated'):
        assert result[key] is True
    for key in ('phase_transition_certified', 'physical_sampling_certified', 'root_search_performed'):
        assert result[key] is False
    assert result['thermal_nu'] is None
    assert result['checkpoint_step'] == result['model_training_updates'] == 12000
    assert result['training_updates'] == result['MI_evaluations'] == 0
    assert set(result['closure']) == {'train', 'same_rho_new_group'}
    assert 'heldout_rho' in result['omitted_partitions']
    assert set(result['warning_banners']) == set(report.REQUIRED_FIGURES)
    assert json.loads((args.output / 'loss_history.json').read_text()) == history
    for filename in ('report.json', 'status.json', 'sampling_quality_warning.json'):
        saved = json.loads((args.output / filename).read_text())
        assert saved['diagnostic_only'] is True
        assert saved['equilibrium_not_validated'] is True
    status = json.loads((args.output / 'status.json').read_text())
    assert status['report_complete'] is True
    for filename, fingerprint in status['sha256'].items():
        assert report.base.sha256(args.output / filename) == fingerprint
    assert validate_report(args.output)['public_files_verified'] == 9
    with np.load(args.output / 'evaluation_arrays.npz', allow_pickle=False) as saved:
        assert saved['diagnostic_only'].item() is True
        assert saved['equilibrium_not_validated'].item() is True
        assert saved['direct_test'].shape == (2, 5, 2, 2)
    assert inputs_before == {path: report.base.sha256(path) for path in inputs_before}
    with pytest.raises(FileExistsError):
        report.run_report(args)


def test_missing_warning_fails_with_diagnostic_status_instead_of_fallback(tmp_path):
    args = SimpleNamespace(output=tmp_path / 'report', max_seconds=1., draws=5,
                           quality_warning=tmp_path / 'missing.json')
    with pytest.raises(ValueError, match='采样质量警告'):
        report.run_report(args)
    status = json.loads((args.output / 'status.json').read_text())
    assert status['status'] == 'failed'
    assert status['report_complete'] is False
    assert status['diagnostic_only'] is status['equilibrium_not_validated'] is True
    assert status['thermal_nu'] is None


def test_diagnostic_cli_does_not_offer_training_checkpoint_or_gate_override():
    import inspect

    source = inspect.getsource(report.main)
    assert '--quality-warning' in source
    assert '--checkpoint' not in source
    assert '--allow-legacy-training' not in source
    assert '--ignore-quality' not in source
