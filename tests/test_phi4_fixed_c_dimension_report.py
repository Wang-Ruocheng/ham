import json

import numpy as np
import pytest
import torch

from experiments.phi4_fixed_c_dimension import endpoint_metrics, predict_r
from experiments.phi4_scalar_r_validation import split_conditions
from recurrent_rg.phi4_qcr import ContinuousTheoryEncoder
from recurrent_rg.phi4_r_calibration import FrozenResidualMap
from scripts import report_phi4_fixed_c_dimension as report


def state(model):
    return {name: value.detach().numpy().copy() for name, value in model.state_dict().items()}


@pytest.mark.parametrize("dim", [1, 2])
def test_numpy_normalized_residual_map_matches_torch_and_full_DR(dim):
    torch.manual_seed(135)
    original = FrozenResidualMap(dim, 5, center=np.arange(dim)*.2, scale=np.arange(dim)*.2+.3)
    values = state(original)
    frozen = report.FrozenMap(values, report.state_fingerprint(values), dim=dim, hidden=5)
    points = np.random.default_rng(57).normal(size=(11, dim))
    actual, full_dr = frozen.forward_and_derivative(points)
    x = torch.tensor(points, dtype=torch.float64, requires_grad=True)
    y = original.advance(x)
    expected_dr = torch.stack([torch.autograd.grad(y[:, index].sum(), x, retain_graph=True)[0]
                               for index in range(dim)], dim=1)
    np.testing.assert_allclose(actual, y.detach().numpy(), rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(full_dr, expected_dr.detach().numpy(), rtol=1e-12, atol=1e-13)
    direct = np.random.default_rng(74).normal(size=(4, 6, 3, dim))
    predicted, expected = frozen.predict(direct), predict_r(original, torch.tensor(direct))
    for kind in predicted:
        np.testing.assert_allclose(predicted[kind], expected[kind], rtol=1e-12, atol=1e-13)
    frozen.assert_unchanged()
    assert all(parameter.grad is None for parameter in original.parameters())


def test_whole_Q_fingerprint_and_decoder_only_reconstruction():
    torch.manual_seed(23)
    original = ContinuousTheoryEncoder(5, 2)
    values = state(original)
    frozen = report.FrozenDecoder(values, report.state_fingerprint(values), dim=2, hidden=5)
    direct = np.random.default_rng(71).normal(size=(4, 8, 3, 2))
    np.testing.assert_allclose(frozen.decode(direct), original.decode(torch.tensor(direct)).detach().numpy(),
                               rtol=1e-13, atol=1e-14)
    frozen.assert_unchanged()
    mutated = {name: value.copy() for name, value in values.items()}
    mutated["net.0.bias"][0] += .01  # even an unused encoder parameter is pinned
    with pytest.raises(ValueError, match="指纹"):
        report.FrozenDecoder(mutated, report.state_fingerprint(values), dim=2, hidden=5)


def test_common_feature_normalizer_is_target_centered_and_dimension_independent():
    target = np.zeros((4, 2, 3, 105))
    target[:, 0] = np.array([1., 2., 3.])[None, :, None]
    target[:, 1] = np.array([-1., -2., -3.])[None, :, None]
    scales, mean = report.feature_normalizers(target)
    np.testing.assert_allclose(scales, [1., 2., 3.])
    assert not np.any(mean)
    alternating = np.broadcast_to(np.array([1., -1., 1., -1.])[:, None, None], (4, 2, 105))
    metrics = report.residual_metrics(alternating, 2.)
    assert metrics["raw_repeat_rmse"] == 1.
    assert metrics["raw_repeat_nrmse"] == .5
    assert metrics["chain_mean_rmse"] == 0.
    assert metrics["chain_mean_nrmse"] == 0.
    with pytest.raises(ValueError, match="为正"):
        report.feature_normalizers(np.ones((4, 2, 3, 105)))


def test_plot_bounds_ignore_coarsest_target_and_TEST():
    direct = np.array([[[[-.5, -.2], [.5, .3], [-900., 1000.]]]])
    bounds, observed = report.input_bounds(direct)
    np.testing.assert_allclose(observed, [[-.5, .5], [-.2, .3]])
    np.testing.assert_allclose(bounds, [[-.6, .6], [-.25, .35]])


def test_common_feature_metrics_match_independent_training_arithmetic():
    torch.manual_seed(91)
    q = ContinuousTheoryEncoder(5, 2)
    values = state(q)
    decoder = report.FrozenDecoder(values, report.state_fingerprint(values), dim=2, hidden=5)
    rng = np.random.default_rng(8)
    direct = rng.normal(0., .1, (4, 6, 3, 2))
    target = rng.dirichlet(np.ones(105), (4, 6, 3))
    r = FrozenResidualMap(2, 5)
    prediction = predict_r(r, torch.tensor(direct))
    normalizers = np.array([.01, .02, .03])
    actual = report.common_feature_metrics(decoder, direct, prediction, target, normalizers)
    expected, _ = endpoint_metrics(q, torch.tensor(direct), prediction, target, np.ones((3, 2)))
    for a, b in zip(actual, expected):
        for kind in ("prediction", "target_Q_reconstruction", "identity_decoded", "identity_feature"):
            for field in ("raw_repeat_rmse", "chain_mean_rmse", "repeat_rmse", "coordinate_rmse"):
                np.testing.assert_allclose(a[kind][field], b[kind][field], rtol=1e-12, atol=1e-14)


def test_geometry_reports_rank_deficiency_without_forcing_rank():
    q = ContinuousTheoryEncoder(5, 2)
    with torch.no_grad():
        q.decoder.weight[:, 1].copy_(q.decoder.weight[:, 0])
    values = state(q)
    decoder = report.FrozenDecoder(values, report.state_fingerprint(values), dim=2, hidden=5)
    one = np.arange(24, dtype=np.float64).reshape(4, 2, 3, 1)
    direct = np.concatenate((one, 2*one), axis=-1)
    geometry = report.geometry_diagnostics(decoder, direct)
    assert geometry["normalized_decoder_singular_values"][1] < 1e-14
    assert geometry["TRAIN_input_variance_explained_fractions"][0] == pytest.approx(1.)
    assert geometry["TRAIN_input_variance_participation_ratio"] == pytest.approx(1.)
    assert not geometry["full_rank_forced"]


def _fixture(tmp_path):
    training, features, output = [tmp_path/name for name in ("training", "features", "report")]
    training.mkdir()
    features.mkdir()
    ids, test_ids = np.repeat(np.arange(8), 4), np.repeat(np.arange(8, 13), 4)
    split = {key: value.tolist() if isinstance(value, np.ndarray) else value
             for key, value in split_conditions(ids).items()}
    rng = np.random.default_rng(639)
    arrays, feature_arrays = {}, {}
    for name, values in (("train", ids), ("test", test_ids)):
        arrays[f"{name}_cell_ids"] = values
        arrays[f"{name}_labels"] = np.column_stack([-.5+values*.01, np.ones(len(values)), np.zeros(len(values))])
        arrays[f"{name}_feature_target"] = rng.dirichlet(np.ones(105), (4, len(values), 3))
        feature_arrays[f"{name}_cell_ids"] = values
        feature_arrays[f"{name}_labels"] = arrays[f"{name}_labels"]
        feature_arrays[f"eval_{name}_feature_target"] = arrays[f"{name}_feature_target"]
    feature_manifest = dict(lattice_sizes=[36, 12, 4], feature_dim=105)
    report.write_json(features/"manifest.json", feature_manifest)
    np.savez(features/"features.npz", **feature_arrays)
    report.write_json(features/"receipt.json", dict(source_unchanged=True, features_sha256=report.sha256(features/"features.npz")))
    q_params, r_params, q_fingerprints, r_fingerprints = {}, {}, {}, {}
    metrics, q_records, r_records = {}, {}, {}
    for dim in report.DIMS:
        torch.manual_seed(91)
        q = ContinuousTheoryEncoder(96, dim).eval()
        with torch.no_grad():
            q.decoder.bias.fill_(1/105)
        q_values = state(q)
        q_fingerprints[str(dim)] = report.state_fingerprint(q_values)
        q_params.update({f"dg{dim}.{key}": value for key, value in q_values.items()})
        q_records[str(dim)] = dict(actual_steps=3000, history=[dict(step=step, loss=.1/(step+1),
            terms=dict(reconstruction=.001/(step+1), subset_micro=.002/(step+1), subset_coarse=.003/(step+1)))
            for step in (0, 100, 3000)])
        for name, count in (("train", 32), ("test", 20)):
            direct = rng.normal(0., .02, (4, count, 3, dim))
            arrays[f"dg{dim}_{name}_direct"] = direct
            arrays[f"dg{dim}_{name}_Q_reconstruction"] = q.decode(torch.tensor(direct)).detach().numpy()
        for seed in report.SEEDS:
            original = FrozenResidualMap(dim, 96)
            with torch.no_grad():
                for parameter in original.parameters():
                    parameter.zero_()
            r_values = state(original)
            base = f"dg{dim}_seed{seed}"
            r_records[base] = dict(history={stage: [dict(step=step, fit_loss=.01/(step+1),
                validation=dict(chain_mean_loss=.02/(step+1))) for step in (0, 100)]
                for stage in ("adam", "lbfgs")})
            for endpoint in report.ENDPOINTS:
                key = f"{base}_{endpoint}"
                r_fingerprints[key] = report.state_fingerprint(r_values)
                r_params.update({f"{key}.{name}": value for name, value in r_values.items()})
                for name in ("train", "test"):
                    prediction = predict_r(original, torch.tensor(arrays[f"dg{dim}_{name}_direct"]))
                    arrays.update({f"eval_{key}_{name}_{kind}": value for kind, value in prediction.items()})
                metrics[key] = {}
                for name in ("train", "fit", "validation", "test"):
                    source = "test" if name == "test" else "train"
                    mask = np.array(split[f"{name}_mask"]) if name in ("fit", "validation") else slice(None)
                    direct = arrays[f"dg{dim}_{source}_direct"][:, mask]
                    prediction = {kind: arrays[f"eval_{key}_{source}_{kind}"][:, mask]
                                  for kind in ("one_step", "rollout")}
                    metrics[key][name] = endpoint_metrics(q, torch.tensor(direct), prediction,
                        arrays[f"{source}_feature_target"][:, mask], np.ones((3, dim)))[0]
    manifest = dict(theory_dims=[1, 2], q_steps=3000, q_hidden=96, r_hidden=96,
        feature_manifest=feature_manifest, explicit_fixed_C_dimension_ablation=True,
        exact_online_QCR_reproduction=False, labels_loaded_before_all_selections=False,
        TEST_loaded_before_all_selections=False, split=split, options=dict(q_seed=91), source_C_sha256="c"*64)
    summary = dict(Q_records=q_records, R_records=r_records, final_Q_sha256=q_fingerprints,
        final_R_sha256=r_fingerprints, feature_metrics=metrics,
        all_Q_fits_and_R_selections_before_TEST_and_labels=True, Q_frozen_during_R=True,
        C_updates=0, common_feature_targets_identical=True)
    report.write_json(training/"manifest.json", manifest)
    report.write_json(training/"summary.json", summary)
    report.write_json(training/"history.json", dict(Q=q_records, R=r_records))
    np.savez(training/"evaluation_arrays.npz", **arrays)
    np.savez(training/"Q_parameters.npz", **q_params)
    np.savez(training/"R_parameters.npz", **r_params)
    for root in (training, features):
        report.write_json(root/"status.json", dict(status="complete", workflow_complete=True, returncode=0, error=None))
        report.write_json(root/"public_files.json", {path.name: report.sha256(path) for path in root.iterdir() if path.is_file()})
    return training, features, output


def test_complete_fixed_C_dimension_report_metrics_and_chinese_plots(tmp_path):
    training, features, output = _fixture(tmp_path)
    result = report.run(training, features, output)
    assert result["status"] == "complete" and result["common_feature_dim"] == 105
    assert result["Q_steps"] == 3000 and result["R_Adam_steps"] == 6000
    assert len(result["feature_metrics"]) == 12
    assert result["feature_metric_recomputation_max_abs_difference"] < 1e-12
    assert max(result["cached_prediction_max_abs_difference"].values()) < 1e-12
    assert result["all_frozen_parameter_fingerprints_unchanged"]
    assert result["normalization_identical_across_dimensions"]
    assert not result["normalization_uses_TEST"]
    assert not result["plot_bounds_use_TEST_or_labels"]
    assert not result["Q_encoder_invoked"] and not result["C_invoked"]
    assert not result["checkpoint_loaded"] and not result["root_search_performed"]
    assert result["critical_r"] is None and result["thermal_nu"] is None
    assert result["dimension_comparison"]["selected_validation_R_seed_pairs"] == 3
    assert result["dimension_comparison"]["Q_seed_replication_gate"]["engineering_gate_only"]
    assert len(result["geometry_diagnostics"]["2"]["TRAIN_input_covariance_eigenvalues"]) == 2
    assert all((output/name).stat().st_size > 1000 for name in ("feature_comparison.png", "feature_nrmse.png",
        "R_vs_g.png", "R_displacement.png", "coupling_by_scale.png", "Q_loss.png", "R_loss.png"))
    inventory = json.loads((output/"public_files.json").read_text())
    assert inventory["status.json"] == report.sha256(output/"status.json")
    with pytest.raises(FileExistsError):
        report.run(training, features, output)


def test_report_rejects_TEST_selection_contract(tmp_path):
    training, features, output = _fixture(tmp_path)
    summary = json.loads((training/"summary.json").read_text())
    summary["all_Q_fits_and_R_selections_before_TEST_and_labels"] = False
    report.write_json(training/"summary.json", summary)
    inventory = json.loads((training/"public_files.json").read_text())
    inventory["summary.json"] = report.sha256(training/"summary.json")
    report.write_json(training/"public_files.json", inventory)
    with pytest.raises(ValueError, match="选定"):
        report.run(training, features, output)
