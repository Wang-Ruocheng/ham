"""Exact continuation regression: no production data, SSH, or GPU execution."""
from contextlib import nullcontext
import json
from types import SimpleNamespace

import pytest

from scripts import verify_highdim_resume as verify


def test_actual_trainer_checkpoint_resume_preserves_complete_cpu_trajectory(tmp_path):
    result = verify.verify_resume(tmp_path/'synthetic_fixture', 'cpu')
    assert result['status'] == 'complete' and result['passed'] is True
    assert result['synthetic_only'] is True
    assert result['production_data_or_MI_opened'] is False
    assert result['scientific_training'] is False
    assert result['dimension'] == 1 and result['hidden'] == 96
    assert result['geometry'] == [16, 8, 4]
    assert result['resumed_segment_updates'] == 2
    assert result['trace_comparisons']['batches']['maximum_absolute_difference'] == 0.
    assert result['trace_comparisons']['evaluation_panels']['maximum_absolute_difference'] == 0.
    assert result['trace_comparisons']['objectives_and_gradients']['maximum_absolute_difference'] == 0.
    for row in result['checkpoints']:
        assert row['comparisons']['model']['maximum_absolute_difference'] == 0.
        assert row['comparisons']['optimizer']['maximum_absolute_difference'] == 0.
        assert row['comparisons']['batch_generator_state']['tensor_count'] == 1
        assert row['comparisons']['rsmi_generator_state']['tensor_count'] == 1
        assert row['comparisons']['torch_rng_state']['tensor_count'] == 1
        assert row['comparisons']['history']['maximum_absolute_difference'] == 0.


def test_cached_resume_forwards_checkpoint_without_fresh_constructor_receipt(tmp_path, monkeypatch):
    from experiments import rsmi_separate_qcr as trainer
    from recurrent_rg import mi_cache
    from scripts import run_ising2d_dimg_ablation as pipeline
    from scripts import report_ising_zh

    original_model = trainer.build_model
    original_cache = mi_cache.load_or_build_mi_statistics
    checkpoint = tmp_path/'training_step_5000.pt'
    initial_receipt = tmp_path/'must_not_claim_fresh.json'
    calls = []

    def actual_trainer(argv):
        calls.append(argv)
        assert argv[argv.index('--resume-checkpoint')+1] == str(checkpoint)
        assert trainer.build_model is original_model
        assert '--source' not in argv and '--allow-nondefault-recipe' not in argv

    monkeypatch.setattr(trainer, 'main', actual_trainer)
    monkeypatch.setattr(report_ising_zh, 'chinese_figures', nullcontext)
    pipeline.cached_training(SimpleNamespace(config=tmp_path/'config.json',
        output=tmp_path/'run', device='cuda:2', mi_cache_dir=tmp_path/'mi',
        initialization_receipt=initial_receipt, benchmark_warmup=0,
        resume_checkpoint=checkpoint))
    assert len(calls) == 1 and not initial_receipt.exists()
    assert trainer.build_model is original_model
    assert mi_cache.load_or_build_mi_statistics is original_cache


def test_cached_resume_receipt_uses_loaded_manifest_not_random_constructor(tmp_path, monkeypatch):
    from experiments import rsmi_separate_qcr as trainer
    from scripts import run_ising2d_dimg_ablation as pipeline
    from scripts import report_ising_zh
    checkpoint = tmp_path/'training_step_5000.pt'
    checkpoint.write_bytes(b'authorized source fixture, never torch-loaded by wrapper')
    run = tmp_path/'run'
    receipt = tmp_path/'resume.json'
    expected_modules = {'Q': 'saved-Q', 'C': 'saved-C', 'R': 'saved-R'}
    provenance = dict(source_epoch=5000, source_initial_metadata_preserved=['training_start_module_sha256'])
    def actual_trainer(argv):
        run.mkdir()
        (run/'manifest.json').write_text(json.dumps(dict(resumed_from=str(checkpoint),
            resume_start_module_sha256=expected_modules, resume_start_provenance=provenance,
            labels_used_in_training=False)))
    monkeypatch.setattr(trainer, 'main', actual_trainer)
    monkeypatch.setattr(report_ising_zh, 'chinese_figures', nullcontext)
    pipeline.cached_training(SimpleNamespace(config=tmp_path/'config.json', output=run,
        device='cuda:2', mi_cache_dir=tmp_path/'mi', resume_checkpoint=checkpoint,
        resume_receipt=receipt, initialization_receipt=tmp_path/'initialization.json'))
    saved = json.loads(receipt.read_text())
    assert saved['fresh'] is False and saved['source_epoch'] == 5000
    assert saved['source_checkpoint_sha256'] == pipeline.fingerprint(checkpoint)
    assert saved['resume_start_module_sha256'] == expected_modules
    assert not (tmp_path/'initialization.json').exists()


@pytest.mark.parametrize('no_source', [False, True])
def test_resume_receipt_invalid_target_fails_before_training(tmp_path, monkeypatch, no_source):
    from experiments import rsmi_separate_qcr as trainer
    from scripts import run_ising2d_dimg_ablation as pipeline
    receipt = tmp_path/'resume.json'
    if not no_source:
        receipt.write_text('keep')
    monkeypatch.setattr(trainer, 'main', lambda argv: pytest.fail('Must fail before updates'))
    args = SimpleNamespace(config=tmp_path/'config.json', output=tmp_path/'run',
        device='cuda:2', mi_cache_dir=tmp_path/'mi', initialization_receipt=tmp_path/'init.json',
        resume_checkpoint=None if no_source else tmp_path/'source.pt', resume_receipt=receipt)
    with pytest.raises(ValueError if no_source else FileExistsError):
        pipeline.cached_training(args)


def test_capture_restores_trainer_hooks_on_error(monkeypatch):
    from experiments import rsmi_separate_qcr as trainer
    from recurrent_rg import execution
    before = (execution.select_training_batch, trainer.independent_component_gradients,
              trainer.evaluate)
    with pytest.raises(RuntimeError):
        with verify.capture_training():
            raise RuntimeError('synthetic failure')
    assert (execution.select_training_batch, trainer.independent_component_gradients,
            trainer.evaluate) == before


def test_history_float_tolerance_never_relaxes_epochs_structure_or_finiteness():
    tree = verify.history_comparison_tree
    verify.compare_trees(tree([dict(epoch=3, loss=1.)]),
                         tree([dict(epoch=3, loss=1.+1e-12)]))
    for changed in ([dict(epoch=4, loss=1.)], [dict(epoch=3, other=1.)],
                    [dict(epoch=3, loss=float('nan'))], [dict(epoch=3, loss=float('inf'))],
                    [dict(epoch=3, loss=1.+1e-4)]):
        with pytest.raises(AssertionError):
            verify.compare_trees(tree([dict(epoch=3, loss=1.)]), tree(changed))


def test_tiny_config_exercises_both_lr_boundaries_with_original_objectives(tmp_path):
    cfg = verify.synthetic_config(tmp_path/'synthetic.pt', 4)
    assert cfg.schema_version == 5
    assert cfg.model.theory_dim == 1 and cfg.model.hidden == 96
    assert cfg.model.block_factor == 2 and cfg.model.affine_reconstruction
    assert cfg.loss.closure_scale_weights is None
    assert cfg.loss.weights.commutator == cfg.loss.rollout_weight == 5.
    assert cfg.loss.weights.anchor == 2. and cfg.loss.weights.coarse_consistency == 5.
    assert cfg.loss.weights.separation == 10.
    assert cfg.loss.kernel_variance_ratio == 1.
    assert cfg.training.lr_milestones == (2, 3) and cfg.training.lr_gamma == .3


def test_existing_receipt_is_preserved_before_any_gpu_gate(tmp_path, monkeypatch):
    path = tmp_path/'receipt.json'
    path.write_text('existing')
    monkeypatch.setattr(verify, 'acquire_gpu_gate', lambda: pytest.fail('No GPU when output exists'))
    with pytest.raises(FileExistsError):
        verify.main(['--output', str(path)])
    assert path.read_text() == 'existing'


@pytest.mark.parametrize('device', ['cpu', 'cuda:0', 'cuda:1', 'cuda:3'])
def test_cli_never_falls_back_or_selects_other_gpu(tmp_path, monkeypatch, device):
    monkeypatch.setattr(verify, 'acquire_gpu_gate', lambda: pytest.fail('No invalid-device GPU gate'))
    with pytest.raises(SystemExit):
        verify.main(['--device', device, '--output', str(tmp_path/'receipt.json')])
