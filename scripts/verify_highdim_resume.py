"""End-to-end exact-resume regression on a tiny synthetic Q/C/R fixture.

This exercises the real trainer, saved checkpoints, Adam, both CPU generators,
CPU/selected-CUDA RNG, fixed evaluation panels, and cumulative LR changes.  It
never reads a production dataset, MI cache, checkpoint, or configuration.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

import torch

from experiments import rsmi_separate_qcr as trainer
from recurrent_rg import execution
from recurrent_rg.config import (DataConfig, EvaluationConfig, LossConfig,
    ModelConfig, RunConfig, TrainingConfig)
from scripts.verify_low_ram_cuda import (EXPECTED_UUID, acquire_gpu_gate,
    ATOL, RTOL, clone_cpu_tree, compare_trees, normalize_uuid, synthetic_data)


TOTAL_UPDATES, SPLIT_UPDATE = 4, 2


def history_comparison_tree(value):
    """Apply float64 tolerances only to history floats; epochs stay exact."""
    if isinstance(value, float):
        return torch.tensor(value, dtype=torch.float64)
    if isinstance(value, dict):
        return {key: history_comparison_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [history_comparison_tree(item) for item in value]
    return value


def synthetic_config(cache, epochs):
    """Historical b2/h96/dimg1, unchanged default objectives, tiny geometry."""
    return RunConfig(data=DataConfig(cache=str(cache), physical_model='synthetic_resume_fixture'),
        model=ModelConfig(dimension=2, theory_dim=1, hidden=96, block_factor=2,
                          affine_reconstruction=True),
        loss=LossConfig(closure_objective='one_step_and_rollout',
                        kernel_noncollapse_mode='variance_floor_ensemble',
                        kernel_variance_ratio=1., kernel_sign_weight=1.),
        training=TrainingConfig(max_rg_steps=2, epochs=epochs, system_batch=3,
            sample_batch=8, threads=1, log_every=1, checkpoint_every=1,
            gradient_mode='qc_r_separate', lr_schedule='multistep',
            lr_milestones=(2, 3), lr_gamma=.3),
        evaluation=EvaluationConfig(max_rg_steps=2, samples=8, systems=3))


@contextmanager
def capture_training():
    """Read-only captures; no additional draws or forward passes."""
    original_batch = execution.select_training_batch
    original_gradients = trainer.independent_component_gradients
    original_evaluate = trainer.evaluate
    captures = dict(batches=[], objectives_and_gradients=[], evaluation_panels=[])

    def selected_batch(x, ids, indices, device, **kwargs):
        result = original_batch(x, ids, indices, device, **kwargs)
        captures['batches'].append(clone_cpu_tree(dict(ids=ids, indices=indices, batch=result)))
        return result

    def gradients(total, closure, c_objective, q, c, r, critic=()):
        # Materialize an iterator once, preserving the exact parameter order.
        q, c, r, critic = list(q), list(c), list(r), list(critic)
        result = original_gradients(total, closure, c_objective, q, c, r, critic)
        captures['objectives_and_gradients'].append(clone_cpu_tree(dict(
            total=total, closure=closure, c_objective=c_objective,
            gradients=[parameter.grad for parameter in q+c+r+critic])))
        return result

    def evaluate(model, x, *args, **kwargs):
        captures['evaluation_panels'].append(clone_cpu_tree(x))
        return original_evaluate(model, x, *args, **kwargs)

    execution.select_training_batch = selected_batch
    trainer.independent_component_gradients = gradients
    trainer.evaluate = evaluate
    try:
        yield captures
    finally:
        execution.select_training_batch = original_batch
        trainer.independent_component_gradients = original_gradients
        trainer.evaluate = original_evaluate


def run_segment(config_path, output, cache_dir, device, *, resume_checkpoint=None):
    argv = ['--config', str(config_path), '--output', str(output), '--device', str(device),
            '--benchmark-warmup', '0', '--skip-final-report']
    if torch.device(device).type == 'cuda':
        argv += ['--mi-cache-dir', str(cache_dir)]
    if resume_checkpoint is not None:
        argv += ['--resume-checkpoint', str(resume_checkpoint)]
    with capture_training() as captures:
        trainer.main(argv)
    return captures


def verify_resume(workspace, device='cpu'):
    """Run 4 uninterrupted updates versus 2 + checkpoint reload + 2."""
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=False)
    device = torch.device(device)
    configurations = synthetic_data()
    cache = workspace/'synthetic.pt'
    torch.save(dict(schema_version=1,
        train=dict(configurations=configurations, group_ids=torch.arange(4)),
        test=dict(configurations=-configurations.roll(1, 1), group_ids=torch.arange(4, 8))), cache)
    configs = {}
    for name, epochs in [('full', TOTAL_UPDATES), ('split', SPLIT_UPDATE), ('resumed', TOTAL_UPDATES)]:
        path = workspace/f'{name}.json'
        path.write_text(json.dumps(synthetic_config(cache, epochs).to_dict(), indent=2)+'\n')
        configs[name] = path
    cache_dir = workspace/'synthetic_mi'
    full = run_segment(configs['full'], workspace/'full', cache_dir, device)
    split = run_segment(configs['split'], workspace/'split', cache_dir, device)
    source = workspace/'split'/f'training_step_{SPLIT_UPDATE}.pt'
    resumed = run_segment(configs['resumed'], workspace/'resumed', cache_dir, device,
                          resume_checkpoint=source)
    comparisons = {}
    for category in full:
        comparisons[category] = compare_trees(full[category], split[category]+resumed[category],
                                              exact=category != 'objectives_and_gradients')
    checkpoints = []
    for step in range(1, TOTAL_UPDATES+1):
        branch = 'split' if step <= SPLIT_UPDATE else 'resumed'
        left = torch.load(str(workspace/'full'/f'training_step_{step}.pt'),
                          map_location='cpu', weights_only=True)
        right = torch.load(str(workspace/branch/f'training_step_{step}.pt'),
                           map_location='cpu', weights_only=True)
        row = dict(step=step, comparisons={})
        fields = ['model', 'optimizer', 'history', 'torch_rng_state',
                  'batch_generator_state', 'rsmi_generator_state', 'lr_scheduler_state']
        if device.type == 'cuda':
            fields += ['cuda_rng_state', 'cuda_device_index']
        for field in fields:
            a, b = left[field], right[field]
            if field == 'history':
                a, b = history_comparison_tree(a), history_comparison_tree(b)
            row['comparisons'][field] = compare_trees(a, b,
                exact=field not in ('model', 'optimizer', 'history'))
        assert left['epoch'] == right['epoch'] == step
        checkpoints.append(row)
    summary = json.loads((workspace/'resumed'/'summary.json').read_text())
    manifest = json.loads((workspace/'resumed'/'manifest.json').read_text())
    assert summary['actual_epochs'] == summary['requested_epochs'] == TOTAL_UPDATES
    assert summary['timing']['updates'] == TOTAL_UPDATES-SPLIT_UPDATE
    assert [row['epoch'] for row in summary['history']] == [1, 2, 3, 4]
    assert manifest['resumed_from'] == str(source)
    assert manifest['labels_used_in_training'] is False
    assert manifest['module_initialization_provenance']['schema_version'] == 1
    source_metadata = torch.load(str(source), map_location='cpu', weights_only=True)['metadata']
    for key in trainer._INITIAL_MODULE_METADATA_FIELDS:
        assert manifest.get(key) == source_metadata.get(key)
    assert manifest['resume_start_provenance']['source_epoch'] == SPLIT_UPDATE
    assert manifest['execution_optimizations']['low_ram_execution']['enabled'] is True
    if device.type == 'cuda':
        assert manifest['execution_optimizations']['selected_empirical_mi'] is True
        assert manifest['execution_optimizations']['shared_binary_Q_tables'] is True
        assert all(record['hit'] for record in manifest['execution_optimizations']['mi_statistics_cache'].values())
    for state in right['optimizer']['state'].values():
        assert int(state['step']) == TOTAL_UPDATES
    return dict(status='complete', passed=True, dimension=1, device=str(device),
        total_updates=TOTAL_UPDATES, split_update=SPLIT_UPDATE,
        resumed_segment_updates=TOTAL_UPDATES-SPLIT_UPDATE,
        synthetic_only=True, production_data_or_MI_opened=False, scientific_training=False,
        geometry=[16, 8, 4], block_factor=2, hidden=96, precision='torch.float64',
        objectives='Unmodified unsupervised default: Q representation; C -2MI+2variance+color; R uniform one-step5+rollout5.',
        comparison='Actual trainer with checkpoint save/reload; no constructor-only or optimizer-reset approximation.',
        floating_tolerances=dict(atol=ATOL, rtol=RTOL),
        cross_hardware_or_version_bitwise_equality_claimed=False,
        trace_comparisons=comparisons, checkpoints=checkpoints,
        lr_test_milestones=[2, 3], lr_test_gamma=.3,
        preserved_initial_provenance=manifest['module_initialization_provenance'],
        resume_start_provenance=manifest['resume_start_provenance'],
        execution=manifest['execution_optimizations'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:2', choices=['cuda:2'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite a resume verification receipt')
    checks = acquire_gpu_gate()
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 2:
        raise RuntimeError('Requested cuda:2 unavailable; no CPU fallback')
    actual_uuid = getattr(torch.cuda.get_device_properties(args.device), 'uuid', None)
    if actual_uuid is None or normalize_uuid(actual_uuid) != normalize_uuid(EXPECTED_UUID):
        raise RuntimeError('CUDA ordinal 2 UUID mismatch; no fallback')
    torch.cuda.reset_peak_memory_stats(args.device)
    with tempfile.TemporaryDirectory(prefix='highdim-resume-synthetic-') as temporary:
        result = verify_resume(Path(temporary)/'fixture', args.device)
    torch.cuda.synchronize(args.device)
    result.update(physical_uuid=EXPECTED_UUID, idle_gpu_checks=checks,
        completed_at=datetime.now(timezone.utc).isoformat(),
        temporary_fixture_removed=True,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(args.device))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print('Synthetic exact resume: actual trainer objective/Adam/RNG/history/LR checks passed', flush=True)
    return result


if __name__ == '__main__':
    main()
