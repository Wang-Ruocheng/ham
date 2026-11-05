import json

import numpy as np
import pytest
import torch

from experiments.phi4_scalar_r_ablation import ScalarClosureMaps, condition_folds, predict
from scripts import report_phi4_scalar_r_ablation as report


def public_state(model):
    return {name: value.detach().numpy().copy() for name, value in model.state_dict().items()}


@pytest.mark.parametrize("kind,heads", [("shared", 1), ("separate", 2)])
def test_public_numpy_heads_match_torch_both_transitions_and_composition(kind, heads):
    torch.manual_seed(51)
    original = ScalarClosureMaps(5, kind, center=[.03], scale=[.08]).eval()
    state = public_state(original)
    frozen = report.FrozenClosureMaps(state, report.state_fingerprint(state), heads=heads, hidden=5)
    direct = np.random.default_rng(62).normal(.1, .2, (4, 7, 3, 1))
    expected, actual = predict(original, direct), frozen.predict(direct)
    for name in ("one_step", "rollout"):
        np.testing.assert_allclose(actual[name], expected[name], rtol=1e-12, atol=1e-13)
    # In the separate arm the second transition head must also be used after
    # the first prediction; swapping to head zero changes this composition.
    if heads == 2:
        first = frozen.heads[0].forward_and_derivative(direct[..., 0, :].reshape(-1, 1))[0]
        wrong = frozen.heads[0].forward_and_derivative(first)[0].reshape(4, 7, 1)
        assert not np.allclose(actual["rollout"][..., 2, :], wrong)
    frozen.assert_unchanged()
    assert all(parameter.grad is None for parameter in original.parameters())
    assert all(not value.flags.writeable for value in frozen.state.values())


def test_head_inventory_and_overall_fingerprint_are_both_enforced():
    original = ScalarClosureMaps(3, "separate")
    state = public_state(original)
    digest = report.state_fingerprint(state)
    with pytest.raises(ValueError, match="指纹"):
        report.FrozenClosureMaps(state, "a"*64, heads=2, hidden=3)
    with pytest.raises(ValueError, match="数目"):
        report.FrozenClosureMaps(state, digest, heads=1, hidden=3)
    with pytest.raises(ValueError, match="宽度"):
        report.FrozenClosureMaps(state, digest, heads=2, hidden=5)


def test_cached_prediction_audit_includes_rollout_and_second_head():
    original = ScalarClosureMaps(3, "separate")
    state = public_state(original)
    frozen = report.FrozenClosureMaps(state, report.state_fingerprint(state), heads=2, hidden=3)
    key, arrays = "fold0_B_separate96_seed91_selected", {}
    for split in ("train", "test"):
        direct = np.random.default_rng(len(split)).normal(size=(4, 8, 3, 1))
        arrays[f"{split}_direct"] = direct
        arrays.update({f"eval_{key}_{split}_{kind}": value for kind, value in frozen.predict(direct).items()})
    assert report.check_predictions(frozen, arrays, key) == 0.
    arrays[f"eval_{key}_test_rollout"][0, 0, 2, 0] += .001
    with pytest.raises(ValueError, match="复合"):
        report.check_predictions(frozen, arrays, key)


def test_root_diagnostic_never_searches_separate_transition_heads(monkeypatch):
    models = {}
    for key, kind in (("shared_selected", "shared"), ("separate_selected", "separate")):
        original = ScalarClosureMaps(2, kind)
        state = public_state(original)
        models[key] = report.FrozenClosureMaps(state, report.state_fingerprint(state),
            heads=1 if kind == "shared" else 2, hidden=2)
    calls = []
    monkeypatch.setattr(report, "find_roots", lambda model, *args, **kwargs: (calls.append(model) or [], {}))
    shape, roots, searches = report.diagnose_maps(models, [-1., 1.], lambda: None)
    assert calls == [models["shared_selected"].heads[0]]
    assert "separate_selected" not in roots
    assert not searches["separate_selected"]["performed"]
    assert not shape["separate_selected"]["autonomous_map"]
    assert shape["shared_selected"]["heads"][0]["full_DR_includes_identity"]


def test_repeat_aware_metric_scores_average_residual_before_squaring():
    direct = np.zeros((4, 2, 3, 1))
    prediction = dict(one_step=np.zeros((4, 2, 2, 1)), rollout=np.zeros((4, 2, 3, 1)))
    for kind in prediction:
        prediction[kind][..., 0] = np.array([1., -1., 1., -1.])[:, None, None]
    rows, closure = report.metric_rows(direct, prediction, [1., 2., 4.])
    assert all(row["rmse"] == 1. for row in rows)
    assert all(row["chain_mean_rmse"] == 0. for row in rows)
    assert [row["nrmse"] for row in rows] == [.5, .25, .25]
    assert closure == 0.


def _complete_fixture(tmp_path, monkeypatch):
    # All four folds and all 96 endpoints are present.  Only the synthetic
    # network width is reduced to make report tests independent of CPU speed.
    arms = {name: dict(spec, hidden=2) for name, spec in report.ARMS.items()}
    monkeypatch.setattr(report, "ARMS", arms)
    cache, training, output = [tmp_path/name for name in ("cache", "training", "report")]
    cache.mkdir()
    training.mkdir()
    ids, test_ids = np.repeat(np.arange(8), 4), np.repeat(np.arange(8, 13), 4)
    folds = condition_folds(ids)
    splits = [{name: value.tolist() if isinstance(value, np.ndarray) else value
               for name, value in row.items()} for row in folds]
    rng = np.random.default_rng(8)
    arrays = {}
    for split, values in (("train", ids), ("test", test_ids)):
        arrays[f"{split}_cell_ids"] = values
        arrays[f"{split}_labels"] = np.column_stack([-.5+.01*values, np.ones(len(values)), np.zeros(len(values))])
        arrays[f"{split}_direct"] = rng.uniform(-.5, .5, (4, len(values), 3, 1))
    cache_manifest = dict(train_input_bounds=[-.5, .5], flow_padding_fraction=.1, target_stds=[.1, .2, .3])
    np.savez(cache/"cache.npz", eval_train_direct=arrays["train_direct"], eval_test_direct=arrays["test_direct"])
    report.write_json(cache/"manifest.json", cache_manifest)
    parameters, fingerprints, metrics, records = {}, {}, {}, {}
    for fold in report.FOLDS:
        for arm, spec in arms.items():
            for seed in report.SEEDS:
                original = ScalarClosureMaps(spec["hidden"], spec["kind"])
                # Exact identity permits a finite-search continuum audit and
                # prevents synthetic accidental steep roots in this smoke.
                with torch.no_grad():
                    for parameter in original.parameters():
                        parameter.zero_()
                state = public_state(original)
                frozen = report.FrozenClosureMaps(state, report.state_fingerprint(state),
                    heads=spec["heads"], hidden=spec["hidden"])
                base = f"fold{fold}_{arm}_seed{seed}"
                records[base] = dict(fold=fold, arm=arm, seed=seed, history={
                    stage: [dict(step=0, fit_loss=.2, validation=dict(chain_mean_loss=.3)),
                            dict(step=1, fit_loss=.1, validation=dict(chain_mean_loss=.2))]
                    for stage in ("adam", "lbfgs")})
                for endpoint in ("selected", "adam6000"):
                    key = f"{base}_{endpoint}"
                    fingerprints[key] = report.state_fingerprint(state)
                    parameters.update({f"{key}.{name}": value for name, value in state.items()})
                    metrics[key] = {}
                    for split in ("train", "test"):
                        predicted = frozen.predict(arrays[f"{split}_direct"])
                        arrays.update({f"eval_{key}_{split}_{kind}": value for kind, value in predicted.items()})
                    for split in ("train", "fit", "validation", "test"):
                        source = "test" if split == "test" else "train"
                        mask = np.array(splits[fold][f"{split}_mask"]) if split in ("fit", "validation") else slice(None)
                        direct = arrays[f"{source}_direct"][:, mask]
                        predicted = {kind: arrays[f"eval_{key}_{source}_{kind}"][:, mask] for kind in ("one_step", "rollout")}
                        metrics[key][split] = report.metric_rows(direct, predicted, cache_manifest["target_stds"])[0]
    report.write_json(training/"manifest.json", dict(arms=arms, splits=splits,
        cache_manifest=cache_manifest, target_stds=cache_manifest["target_stds"]))
    report.write_json(training/"summary.json", dict(final_R_sha256=fingerprints,
        all_runs_selected_before_TEST_and_labels_loaded=True, latent_metrics=metrics))
    report.write_json(training/"history.json", records)
    np.savez(training/"evaluation_arrays.npz", **arrays)
    np.savez(training/"R_parameters.npz", **parameters)
    for directory in (cache, training):
        report.write_json(directory/"status.json", dict(status="complete", workflow_complete=True, returncode=0, error=None))
        report.write_json(directory/"public_files.json", {path.name: report.sha256(path)
            for path in directory.iterdir() if path.is_file()})
    return cache, training, output


def test_completed_four_arm_report_public_arrays_only_and_chinese_plots(tmp_path, monkeypatch):
    cache, training, output = _complete_fixture(tmp_path, monkeypatch)
    result = report.run(training, cache, output)
    assert result["status"] == "complete"
    assert result["original_TRAIN_input_interval"] == [-.5, .5]
    assert result["search_interval"] == [-.6, .6]
    assert len(result["roughness"]) == 96
    assert len(result["numerical_fixed_points"]) == 36
    assert all("B_separate96" not in key for key in result["numerical_fixed_points"])
    assert not result["Q_or_C_loaded"] and not result["checkpoint_loaded"]
    assert result["metric_recomputation_max_abs_difference"] == 0.
    assert max(result["cached_prediction_max_abs_difference"].values()) == 0.
    assert result["frozen_parameters_unchanged"]
    assert result["paired_comparisons"]["validation_consistency"][0]["fold_seed_pairs"] == 12
    assert result["critical_r"] is None and result["thermal_nu"] is None
    assert all((output/name).stat().st_size > 1000 for name in ("R_vs_g.png", "R_displacement.png",
        "transition_pairs.png", "closure_comparison.png", "loss.png", "coupling_by_scale.png"))
    inventory = json.loads((output/"public_files.json").read_text())
    assert inventory["status.json"] == report.sha256(output/"status.json")
    with pytest.raises(FileExistsError):
        report.run(training, cache, output)


def test_report_rejects_preselection_test_leak_contract(tmp_path, monkeypatch):
    cache, training, output = _complete_fixture(tmp_path, monkeypatch)
    summary = json.loads((training/"summary.json").read_text())
    summary["all_runs_selected_before_TEST_and_labels_loaded"] = False
    report.write_json(training/"summary.json", summary)
    inventory = json.loads((training/"public_files.json").read_text())
    inventory["summary.json"] = report.sha256(training/"summary.json")
    report.write_json(training/"public_files.json", inventory)
    with pytest.raises(ValueError, match="选定"):
        report.run(training, cache, output)
