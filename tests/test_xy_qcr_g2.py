import json
import numpy as np
import pytest
import torch

from experiments import xy_qcr as original
from experiments import xy_qcr_g2 as trial
from experiments.phi4_hopping_qcr import closure_metrics
from recurrent_rg.xy_qcr import XYQCR, objectives, independent_step
from recurrent_rg.phi4_ru_flow import full_map_and_jacobian


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def test_only_dimension_changes_training_settings_and_cli(tmp_path, monkeypatch):
    c = trial.preset(); old = original.preset()
    assert {k for k in old if c[k] != old[k]} == {'recipe', 'theory_dim'}
    assert c['steps'] == 1250 and c['supervision_weight'] == 0
    p = tmp_path/'config.json'; p.write_text(json.dumps(c))
    assert trial.validate_config(p) == c
    from recurrent_rg.cli import main
    calls = []; monkeypatch.setattr(trial, 'run', lambda *args: calls.append(args))
    main(['train', '--config', str(p), '--output', str(tmp_path/'out')])
    assert len(calls) == 1
    with pytest.raises(SystemExit):
        main(['train', '--config', str(p), '--output', 'unused', '--allow-legacy-training'])


def test_same_C_trajectory_and_independent_dim2_gradients():
    models = [XYQCR(seed=91, dim=d) for d in (1, 2)]
    opts = [original.optimizers(m) for m in models]
    x = torch.randn(2, 4, 18, 18, 1, dtype=torch.double, generator=torch.Generator().manual_seed(18))
    q, c, r, metrics = objectives(models[1], x, torch.Generator().manual_seed(7), information_pairs=8)
    for loss, others in [(q, ('C', 'R')), (c, ('Q', 'R')), (r, ('Q', 'C'))]:
        params = [p for n in others for p in getattr(models[1], n).parameters()]
        assert all(v is None for v in torch.autograd.grad(loss, params, allow_unused=True, retain_graph=True))
    assert float(r.detach()) == pytest.approx(5*metrics['closure_start_0_target_1'] +
        2.5*(metrics['closure_start_0_target_2']+metrics['closure_start_1_target_2']))
    for _ in range(3):
        for model, opt in zip(models, opts):
            independent_step(model, x, opt, torch.Generator().manual_seed(21), information_pairs=8)
        for p, q in zip(models[0].C.parameters(), models[1].C.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
    assert all(int(v['step']) == 3 for o in opts[1].values() for v in o.state.values())
    with torch.no_grad():
        torch.testing.assert_close(models[1].Q(x), models[1].Q(x+.7), rtol=1e-11, atol=1e-12)


def test_full_DR_finite_difference_and_identity_guard():
    model = XYQCR(seed=91, dim=2)
    x = np.array([[.2, -.3], [-.1, .4]])
    _, dr = full_map_and_jacobian(model, x, 'cpu')
    eps = 1e-5
    for j in range(2):
        v = np.eye(2)[j]*eps
        with torch.no_grad():
            fd = (model.advance(torch.tensor(x+v))-model.advance(torch.tensor(x-v))).numpy()/(2*eps)
        np.testing.assert_allclose(fd, dr[:, :, j], rtol=1e-7, atol=1e-9)
    for p in model.R.parameters():
        with torch.no_grad(): p.zero_()
    a = mock_arrays()
    result = trial.flow_diagnostics(model, a, 'cpu')
    assert result['roots'] == [] and result['flow_analysis']['near_identity_on_tested_grid']
    np.testing.assert_allclose(a['flow_DR'], np.broadcast_to(np.eye(2), (1089, 2, 2)))
    assert all(p.grad is None for p in model.parameters())


def mock_arrays():
    temp = np.repeat(original.preset()['temperatures'], 4)
    rng = np.random.default_rng(93)
    g = np.stack([np.column_stack([temp*.8**k, (temp-1)**2+.07*k]) for k in range(3)], 1)
    g += rng.normal(0, .002, g.shape)
    target = rng.random((60, 3, 8)); target /= target.sum(-1, keepdims=True)
    return dict(direct_g=g, direct_full_g=g, one_step_g=g[:, 1:]+.01, rollout_g=g,
        all_start_g=np.stack([g[:, 1], g[:, 2], g[:, 2]], 1),
        all_start_pairs=np.array([[0, 1], [0, 2], [1, 2]]),
        labels=np.column_stack([temp, 1/temp]), cell_ids=np.repeat(np.arange(15), 4),
        splits=np.repeat(np.where(np.arange(15)%3 == 1, 'test', 'train'), 4),
        target_features=target, reconstructed_features=target.copy())


def test_report_geometry_and_collapse(tmp_path):
    from scripts.report_xy_qcr_g2 import render, geometry_statistics, correlation
    a = mock_arrays(); model = XYQCR(seed=91, dim=2)
    roots = trial.flow_diagnostics(model, a, 'cpu')
    np.savez(tmp_path/'evaluation_arrays.npz', **a)
    (tmp_path/'data').mkdir()
    values = {'summary.json': dict(actual_steps=1250, closure=closure_metrics(a), information_diagnostic={}, C_final_exactly_matches_scalar=True),
        'roots.json': roots, 'history.json': [dict(step=1250, q_total=.1, c_total=.1, r_total=.1)],
        'data/metadata.json': dict(warnings=[]), 'reference_scalar.json': dict(comparison_step=1250)}
    for name, value in values.items(): (tmp_path/name).write_text(json.dumps(value))
    report = render(tmp_path)
    assert not report['BKT_certified'] and not report['physical_dimension_certified']
    assert len(list((tmp_path/'report').glob('*.png'))) >= 7
    assert report['representation']['test']['feature_rmse_by_scale'] == [0., 0., 0.]
    assert report['geometry']['condition_mean_pca'][0]['explained_variance_ratio'][1] > 0
    a['direct_g'][:] = 0
    assert geometry_statistics(a)['pooled_input_explained_variance_ratio'] == [0., 0.]
    assert correlation(np.arange(4), np.zeros(4)) is None
