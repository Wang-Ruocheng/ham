from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.rsmi_separate_qcr import build_parser, validate_recipe
from experiments.train_qcr_report import phase_report_module
from recurrent_rg.config import RunConfig
from scripts.preview_latest_checkpoint import preview_stages
from scripts.report_ising2d_KB_vector import field_reference, write_field_report


def test_KB_config_changes_only_data_and_required_spin_adapter():
    root = Path(__file__).resolve().parents[1]
    before = RunConfig.load(root / 'configs/dual_ising2d_uniformK1K2_g2_affineQ_joint_1200_seed91.json')
    now = RunConfig.load(root / 'configs/ising2d_KB_g2_affineQ_joint_1200_seed91.json')
    assert now.loss == before.loss
    assert now.training == before.training
    assert now.evaluation == before.evaluation
    for name in now.model.__dataclass_fields__:
        if name not in ('categorical_states', 'kernel_sampling'):
            assert getattr(now.model, name) == getattr(before.model, name)
    assert now.model.kernel_sampling == 'bernoulli'
    assert now.model.categorical_states == 0
    args = build_parser().parse_args(['--config', 'kb.json', '--output', 'run'])
    validate_recipe(args, now)
    assert phase_report_module(now) == 'scripts.report_ising2d_KB_vector'
    assert phase_report_module(before) == 'scripts.report_dual_ising_vector'


def test_KB_preview_does_not_change_existing_dual_preview():
    assert 'scripts.report_ising2d_KB_vector' in preview_stages('run', 200, 'out', 2, ['K', 'B'])[0][1]
    assert 'scripts.report_dual_ising_vector' in preview_stages('run', 200, 'out', 2, ['K1', 'K2'])[0][1]


def test_KB_reference_keeps_axis_order_and_zero_field():
    axes = [np.array([.2, .5, .68]), np.array([-.01, 0, .01])]
    labels = np.stack(np.meshgrid(*axes, indexing='ij'), -1)
    values = np.stack([labels, .9 * labels, .81 * labels], axis=2)
    kc, parameters, curves, centers = field_reference(axes, values)
    np.testing.assert_allclose(centers, np.array([[kc, 0], [.9 * kc, 0], [.81 * kc, 0]]))
    np.testing.assert_allclose(curves[1], .9 * parameters)
    assert np.all(parameters[:, 1] == 0)
    with pytest.raises(ValueError, match='bracket'):
        field_reference([axes[0], np.array([.01, .02])], values[:, :2])


def test_KB_report_writes_maps_and_train_normalized_closure(tmp_path):
    import torch
    axes = [np.array([.2, .5, .68]), np.array([-.01, 0, .01])]
    labels = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 2)
    direct = np.stack([labels, .9 * labels, .81 * labels], axis=1)
    tr = dict(direct=direct, rollout=direct.copy(), one_step=direct[:, 1:].copy())
    ds = SimpleNamespace(couplings=torch.from_numpy(labels))
    run = SimpleNamespace(model=SimpleNamespace(advance_theory=lambda g: .9 * g),
                          model_config=SimpleNamespace(block_factor=3), checkpoint=tmp_path / 'model.pt',
                          payload={'epoch': 1200})
    roots = dict(search_box=[[0, -.02], [.8, .02]], roots=[])
    report = write_field_report(run, tr, tr, ds, ds, tmp_path, roots, {'sites': 36}, 'fingerprint')
    assert report['lattice_sizes'] == [36, 12, 4]
    assert report['heldout_closure']['one_step']['nrmse'] == [0., 0.]
    assert report['coupling_names'] == ['K', 'B']
    for name in ['R_with_KB_reference.png', 'zero_field_by_scale.png', 'coupling_by_scale.png',
                 'grid_encodings.npz', 'metrics.json']:
        assert (tmp_path / name).stat().st_size > 0
