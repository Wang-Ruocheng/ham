"""Read-only frozen-R evaluation for a qualified PC12 diagnostic slice.

Run outside the frozen training source tree. No Q/C evaluation, physical data
loading, optimizer, GPU, sampling, root search or checkpoint export is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def evaluate_displacement(model, states, *, chunk=2048):
    """Use the full residual map R(g)=g+theory_rg(g), not R(g)-2g."""
    import torch
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or not np.isfinite(states).all():
        raise ValueError('Require finite full-coordinate states')
    parts = []
    with torch.inference_mode():
        for start in range(0, len(states), chunk):
            tensor = torch.from_numpy(np.ascontiguousarray(states[start:start + chunk]))
            parts.append((model.advance_theory(tensor) - tensor).numpy())
    result = np.concatenate(parts)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite frozen-map displacement')
    return result


def plane_states(xy, center, basis):
    xy, center, basis = map(np.asarray, (xy, center, basis))
    if (xy.shape[-1] != 2 or center.shape != (5,) or basis.shape != (2, 5)
            or not np.allclose(basis @ basis.T, np.eye(2), atol=1e-12, rtol=0)):
        raise ValueError('Require the original orthonormal TRAIN PCA basis')
    return center + xy @ basis


def support_mask(train_inputs, queries, *, radius=.35):
    """A geometric plot heuristic, not certification of 5D/physical support."""
    from scipy.spatial import Delaunay, cKDTree
    points, queries = np.asarray(train_inputs), np.asarray(queries)
    scale = points.std(0, ddof=0)
    if points.shape[1] != 2 or np.any(scale <= 0):
        raise ValueError('TRAIN inputs must span both projected coordinates')
    normalized = points / scale
    query = queries.reshape(-1, 2) / scale
    distances = cKDTree(normalized).query(query)[0]
    inside = Delaunay(normalized).find_simplex(query) >= 0
    return (inside & (distances <= radius)).reshape(queries.shape[:-1]), scale


def rms_norm(value):
    return float(np.sqrt(np.mean(np.sum(np.asarray(value) ** 2, axis=-1))))


def validate_persisted_parameters(model, payload):
    """Check saved Q/C/R bits, excluding the unused, unsaved action network."""
    import torch
    actual = {key: value for key, value in model.state_dict().items()
              if not key.startswith('action_network.')}
    expected = payload['model']
    if set(actual) != set(expected) or any(
            actual[key].dtype != expected[key].dtype
            or not torch.equal(actual[key].cpu(), expected[key].cpu()) for key in actual):
        raise ValueError('Loaded Q/C/R parameters differ from persisted checkpoint tensors')
    return True


def export(args):
    code = Path(args.code_root).resolve()
    sys.path.insert(0, str(code))
    import torch
    from recurrent_rg.checkpoints import load_run
    from scripts.ising_highdim_fast_evaluation import _model_digest
    from scripts.report_ising_j1j2 import validate_completed_run
    from scripts.run_ising_j1j2_unthermalized import verify

    torch.set_num_threads(1)
    run, output = Path(args.run).resolve(), Path(args.output).resolve()
    plan, job = verify(run.parent.parent)
    if (job['status'] != 'complete' or not job['workflow_complete']
            or Path(plan['code_root']).resolve() != code
            or run != Path(plan['output']) / 'dg5_seed91/run'
            or not plan['diagnostic']['diagnostic_only']):
        raise ValueError('Require the completed, frozen diagnostic run')
    if output.exists() or output.is_symlink() or output.is_relative_to(Path(plan['output'])):
        raise FileExistsError('New analysis output must be outside the immutable run')
    report_path = Path(args.report).resolve()
    evaluation = Path(args.evaluation).resolve()
    metadata = Path(args.metadata).resolve()
    expected_report = run.parent / 'report'
    if (report_path != expected_report / 'report.json'
            or evaluation != expected_report / 'evaluation_arrays.npz'
            or metadata != Path(plan['output']) / 'data/metadata.json'):
        raise ValueError('Inputs must belong to the declared frozen report')
    checkpoint = run / 'training_step_12000.pt'
    inputs = [report_path, evaluation, metadata, checkpoint, run / 'manifest.json']
    before = {str(path): sha(path) for path in inputs}
    report = json.loads(report_path.read_text())
    status = json.loads((expected_report / 'status.json').read_text())
    if (report.get('diagnostic_only') is not True
            or report.get('equilibrium_not_validated') is not True
            or before[str(checkpoint)] != report['checkpoint_sha256']
            or before[str(evaluation)] != status['sha256']['evaluation_arrays.npz']):
        raise ValueError('Checkpoint or public evaluation fingerprint mismatch')
    with np.load(evaluation, allow_pickle=False) as archive:
        z = {name: archive[name] for name in archive.files}
    center, basis = z['pca_mean'], z['pca_components'][:2]
    states = {split: z[f'direct_{split}'].mean(1) for split in ('train', 'test')}
    projected = {split: (value - center) @ basis.T for split, value in states.items()}
    train_inputs = projected['train'][:, :2].reshape(-1, 2)
    span = np.maximum(np.ptp(train_inputs, axis=0), 1e-8)
    lower, upper = train_inputs.min(0) - .18 * span, train_inputs.max(0) + .18 * span
    x, y = [np.linspace(lower[d], upper[d], n) for d, n in enumerate((241, 181))]
    grid = np.stack(np.meshgrid(x, y), -1)
    support, support_scale = support_mask(train_inputs, grid)
    loaded = load_run(run, 12000)
    manifest = json.loads((run / 'manifest.json').read_text())
    summary = json.loads((run / 'summary.json').read_text())
    validate_completed_run(loaded, manifest, summary)
    model = loaded.model.cpu().double().eval().requires_grad_(False)
    model_sha = _model_digest(model)
    validate_persisted_parameters(model, loaded.payload)
    delta = evaluate_displacement(model, plane_states(grid.reshape(-1, 2), center, basis))
    delta = delta.reshape(*grid.shape[:-1], 5)
    projected_delta = delta @ basis.T
    normal = delta - projected_delta @ basis
    observed = z['R_of_mean_g_test'] - states['test']
    actual = evaluate_displacement(model, states['test'].reshape(-1, 5)).reshape(96, 3, 5)
    saved_error = float(np.max(np.abs(actual - observed)))
    if saved_error > 2e-12:
        raise ValueError('Frozen R no longer reproduces saved observed displacements')
    sliced_test = plane_states(projected['test'].reshape(-1, 2), center, basis)
    slice_delta = evaluate_displacement(model, sliced_test).reshape(96, 3, 5)
    discrepancy = (slice_delta[:, :2] - observed[:, :2]) @ basis.T
    denominator = rms_norm(observed[:, :2] @ basis.T)
    source_metadata = json.loads(metadata.read_text())
    if (_model_digest(model) != model_sha
            or any(v.grad is not None or v.requires_grad for v in model.parameters())
            or before != {str(path): sha(path) for path in inputs}):
        raise RuntimeError('Frozen model or source inputs changed')
    verify(run.parent.parent)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / 'field.npz', grid_x=x, grid_y=y,
        projected_delta=projected_delta, full_delta=delta, support_mask=support,
        projected_train=projected['train'], projected_test=projected['test'],
        observed_projected_delta_test=observed @ basis.T,
        rho_test=np.asarray([row['rho'] for row in source_metadata['test']]),
        pca_ratio=z['pca_explained_variance_ratio'], pca_mean=center, pca_basis=basis,
        normal_displacement_norm=np.linalg.norm(normal, axis=-1),
        slice_discrepancy_test=discrepancy,
        diagnostic_only=np.asarray(True), equilibrium_not_validated=np.asarray(True))
    full_rms, normal_rms = rms_norm(delta[support]), rms_norm(normal[support])
    receipt = dict(status='complete', diagnostic_only=True, equilibrium_not_validated=True,
        device='cpu', cpu_threads=1, checkpoint_step=12000, training_updates=0,
        Q_evaluations=0, C_evaluations=0, sampling_performed=False, root_search_performed=False,
        source_sha256=before, model_sha256_before=model_sha, model_sha256_after=model_sha,
        persisted_QCR_parameters_bitwise_match=True,
        cross_load_digest_caveat='未使用的 action_network 不在 checkpoint 中，重载会随机初始化；不比较含该模块的跨加载全模型摘要。已逐张量核对全部已保存 Q/C/R，并复核原报告 R(g)-g。',
        exporter_sha256=sha(Path(__file__)), field_sha256=sha(output / 'field.npz'),
        slice_description='TRAIN PCA 前两轴平面；PC3–PC5 固定为 TRAIN 均值。不是已认证的二维自治 RG。',
        support_description='TRAIN k=0/1 投影凸包内，按各轴 TRAIN std 标准化后距最近输入点<=0.35；仅几何示意，不认证高维或物理支持。',
        support_axis_std=support_scale.tolist(), support_grid_fraction=float(support.mean()),
        explained_variance_ratio=z['pca_explained_variance_ratio'].tolist(),
        saved_observed_displacement_max_abs_error=saved_error,
        projected_slice_discrepancy_RMS=rms_norm(discrepancy),
        observed_projected_displacement_RMS=denominator,
        slice_discrepancy_to_observed_displacement_RMS_ratio=rms_norm(discrepancy) / denominator,
        supported_full_displacement_RMS=full_rms, supported_normal_displacement_RMS=normal_rms,
        supported_normal_to_full_displacement_RMS_ratio=normal_rms / full_rms,
        qualification='流线只显示离散位移 R(g)-g 的方向，不是离散 R 的真实迭代轨迹；未充分热化，不能认证平衡相图或不动点。')
    (output / 'export.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('code-root', 'run', 'evaluation', 'metadata', 'report', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    return export(parser.parse_args())


if __name__ == '__main__':
    main()
