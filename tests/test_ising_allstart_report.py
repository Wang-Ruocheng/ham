import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.report_ising_allstart_closure import (
    ARMS, C_SEEDS, comparable_configs, fit_train_pca, module_comparison,
    plot_results, project,
)
from scripts.report_ising_b0_multiseed import grouped_path_metrics, path_errors
from scripts.report_ising_highdim_fast import Deadline


def test_pair_config_diff_is_only_closure_objective():
    control = dict(loss=dict(closure_objective='one_step_and_rollout', rollout_weight=5),
                   training=dict(seed=91, max_rg_steps=3))
    anchor = copy.deepcopy(control)
    anchor['loss']['closure_objective'] = 'all_start_anchor'
    assert comparable_configs(control, anchor) == ('one_step_and_rollout', 'all_start_anchor')
    assert control['loss']['closure_objective'] == 'one_step_and_rollout'
    anchor['training']['seed'] = 92
    with pytest.raises(ValueError, match='只改变'):
        comparable_configs(control, anchor)


def test_train_only_pca_and_deterministic_sign():
    rng = np.random.default_rng(8)
    direct = rng.normal(size=(10, 4, 5))
    direct[..., 0] *= 10
    pca = fit_train_pca(direct)
    np.testing.assert_allclose(pca['variance_ratio'].sum(), 1)
    assert pca['variance_ratio'][0] > .8
    for component in pca['components']:
        assert component[np.argmax(np.abs(component))] >= 0
    np.testing.assert_allclose(project(direct, pca).reshape(-1, 5).mean(axis=0), 0, atol=1e-14)
    with pytest.raises(ValueError):
        fit_train_pca(direct[:, :3])


def test_six_path_errors_and_C_repeats_not_independent_chains():
    model = SimpleNamespace(advance_theory=lambda value: value + 1.)
    direct = np.broadcast_to(np.arange(4)[None, :, None], (3, 4, 5)).copy().astype(float)
    direct += np.arange(3)[:, None, None]
    errors = path_errors(model, dict(direct=direct))
    assert errors['paths'] == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    np.testing.assert_array_equal(errors['R'], 0)
    metrics = grouped_path_metrics([errors] * len(C_SEEDS), direct)
    assert metrics['physical_chain_count'] == metrics['independent_sample_count'] == 3
    assert metrics['C_draw_count'] == 5
    assert all(row['R']['rmse'] == 0 for row in metrics['paths'])


def test_module_identity_detects_QC_difference():
    first = torch.nn.Linear(5, 5).double()
    second = copy.deepcopy(first)
    assert module_comparison(first, second)['exact_equal']
    with torch.no_grad():
        second.bias[0] += .25
    result = module_comparison(first, second)
    assert not result['exact_equal']
    assert result['max_abs_difference'] == pytest.approx(.25)


def test_all_new_plot_text_is_chinese_and_arrows_are_actual(tmp_path):
    from scripts.report_ising_zh import select_chinese_font
    try:
        select_chinese_font()
    except RuntimeError:
        pytest.skip('测试环境没有现成中文字体；正式报告仍会拒绝缺字或未翻译文字')
    rng = np.random.default_rng(4)
    direct = rng.normal(size=(8, 4, 5))
    model = SimpleNamespace(advance_theory=lambda value: value + .1)
    errors = path_errors(model, dict(direct=direct))
    metric = grouped_path_metrics([errors] * 5, direct)
    result = dict(arms={arm: dict(closure=dict(test=metric)) for arm in ARMS})
    pcas = {arm: fit_train_pca(direct) for arm in ARMS}
    collected = {arm: dict(test=[dict(direct=direct)] * 5,
        test_R_of_g=[direct[:, :-1] + .1] * 5) for arm in ARMS}
    couplings = dict(test=np.linspace(.2, .68, 8))
    history = [dict(epoch=step, train_commutator=.1, test_commutator=.2,
                    train_rollout=.3, test_rollout=.4, train_separation=.5,
                    test_separation=.6, train_kernel_noncollapse=.7,
                    test_kernel_noncollapse=.8, train_rsmi_bits=.9, test_rsmi_bits=1.)
               for step in (200, 12000)]
    plot_results(tmp_path, result, collected, pcas, couplings,
                 {arm: history for arm in ARMS}, Deadline(120))
    assert {path.name for path in tmp_path.iterdir()} == {
        'principal_components.png', 'R_flow_pc1.png', 'coupling_to_g.png',
        'loss_curves.png', 'closure_comparison.png'}
