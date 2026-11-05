import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.rsmi_separate_qcr import build_parser, validate_recipe
from recurrent_rg.config import RunConfig
from scripts.dual_ising_grid_diagnostics import critical_curves, grid_encodings, write_grid_report


def test_grid_config_preserves_scan_model_losses_and_training():
    root = Path(__file__).resolve().parents[1]
    previous = RunConfig.load(root / 'configs/dual_ising2d_K1halfK2_g2_affineQ_joint_1200_seed91.json')
    grid = RunConfig.load(root / 'configs/dual_ising2d_uniformK1K2_g2_affineQ_joint_1200_seed91.json')
    assert grid.model == previous.model
    assert grid.loss == previous.loss
    assert grid.training == previous.training
    assert grid.evaluation == previous.evaluation
    assert grid.schema_version == 5
    assert tuple(grid.data.coupling_names) == ('K1', 'K2')
    args = build_parser().parse_args(['--config', 'grid.json', '--output', 'run'])
    validate_recipe(args, grid)
    assert not hasattr(args, 'r_local_difference_test')
    with pytest.raises(SystemExit):
        build_parser().parse_args(['--config', 'grid.json', '--output', 'run', '--r-local-difference-test'])


def test_grid_mean_and_critical_interpolation_preserve_both_axes():
    axes = [np.array([0., .5, 1.3]), np.array([0., .6, 1.3])]
    labels = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 2)
    direct = np.stack([labels, 2 * labels, 3 * labels], axis=1)
    # Replicas and input order must not change Cartesian axes or interpolation.
    labels = np.repeat(labels, 2, axis=0)[::-1]
    direct = np.repeat(direct, 2, axis=0)[::-1]
    actual_axes, values = grid_encodings(labels, direct)
    kc, params, curves, centers = critical_curves(actual_axes, values)
    np.testing.assert_allclose(centers, np.arange(1, 4)[:, None] * np.array([[kc, kc]]))
    for k in range(3):
        for j in range(2):
            np.testing.assert_allclose(curves[k][j], (k + 1) * params[j])
    np.testing.assert_allclose(params[0][-1], [kc, 1.3])
    np.testing.assert_allclose(params[1][-1], [1.3, kc])


def test_incomplete_grid_rejected_instead_of_silently_flattening_labels():
    with pytest.raises(ValueError, match='Incomplete Cartesian'):
        grid_encodings(np.array([[0., 0.], [1., 1.]]), np.zeros((2, 3, 2)))
    with pytest.raises(ValueError, match='two columns'):
        grid_encodings(np.array([0., 1.]), np.zeros((2, 3, 2)))


def test_grid_report_smoke_writes_actual_figures_and_metrics(tmp_path):
    import torch
    axis = np.array([0., .5, 1.3])
    labels = np.stack(np.meshgrid(axis, axis, indexing='ij'), -1).reshape(-1, 2)
    direct = np.stack([labels, .9 * labels, .81 * labels], axis=1)
    trajectories = dict(direct=direct, rollout=direct.copy(), one_step=direct[:, 1:].copy())
    dataset = SimpleNamespace(couplings=torch.from_numpy(labels))
    model = SimpleNamespace(advance_theory=lambda g: .9 * g)
    run = SimpleNamespace(model=model, model_config=SimpleNamespace(block_factor=3),
                          checkpoint=tmp_path / 'checkpoint_last.pt', payload={'epoch': 1200})
    roots = dict(search_box=[[-.2, -.2], [1.5, 1.5]], roots=[])
    report = write_grid_report(run, trajectories, trajectories, dataset, dataset, tmp_path,
                              roots, {'L': 36}, 'test-fingerprint')
    assert report['lattice_sizes'] == [36, 12, 4]
    assert report['heldout_closure']['one_step']['nrmse'] == [0., 0.]
    assert json.loads((tmp_path / 'metrics.json').read_text())['grid_mode'] == 'independent'
    for name in ('R_with_theory_boundaries.png', 'theory_boundaries_by_scale.png',
                 'coupling_by_scale.png', 'latent_phases.png', 'grid_encodings.npz'):
        assert (tmp_path / name).stat().st_size > 0
