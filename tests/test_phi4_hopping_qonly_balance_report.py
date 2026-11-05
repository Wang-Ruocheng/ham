"""Synthetic saved-array checks only, not physical or training evidence."""
import ast
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import report_phi4_hopping_qonly_balance as report


def fixture_run(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    rng = np.random.default_rng(123)
    splits = np.repeat(["train", "train", "test"] * 5, 4)
    labels = np.column_stack((np.repeat(np.linspace(.32, 1.04, 15), 4), np.ones(60), np.zeros(60)))
    target = rng.normal(size=(60, 4, 625)) * np.arange(1, 5)[None, :, None] * .001
    train = splits == "train"
    arrays = {"target_features": target, "labels": labels, "splits": splits, "cell_ids": np.repeat(np.arange(15), 4),
              "train_target_mean": target[train].mean(axis=0),
              "train_target_std": np.sqrt(target[train].var(axis=0, ddof=0).mean(axis=1)),
              "metric_denominators": np.asarray([1., 2., 3., 4.]), "relative_weights": np.asarray([4., 2., 1., .5]),
              "source_flow_grid": np.linspace(-.1, .1, 101)[:, None]}
    arrays["source_flow_next"] = .9 * arrays["source_flow_grid"] + .01
    direction = rng.normal(size=625)
    direction /= np.linalg.norm(direction)
    bias = rng.normal(size=625) * .001
    for index, model in enumerate(report.MODELS):
        g = rng.normal(size=(60, 4, 1)) * .03 + index * .01
        jitter = rng.normal(size=g.shape) * .001
        arrays.update({f"{model}_direct_g": g, f"{model}_direct_full_g": g + .0003,
                       f"{model}_g_first": g + jitter, f"{model}_g_second": g - jitter,
                       f"{model}_decoded": bias + g * direction})
    np.savez(root / "evaluation_arrays.npz", **arrays)
    history = [{"step": step, "arms": {arm: {"q_total": .02 / (step + 1),
                "reconstruction_raw": .001 / (step + 1), "reconstruction_relative": .002 / (step + 1),
                "reconstruction_by_scale": [.001 * (k + 1) / (step + 1) for k in range(4)]}
                for arm in report.ARMS}} for step in (0, 1, 2)]
    for name, value in (("plan", {}), ("history", history), ("metrics", {}),
                        ("summary", {"status": "complete", "thermal_nu": None, "critical_kappa": None})):
        (root / f"{name}.json").write_text(json.dumps(value))
    return root, arrays


def test_module_has_no_network_or_training_dependency():
    tree = ast.parse(Path(report.__file__).read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(item.name for item in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module)
    assert not any(name and name.split(".")[0] in {"torch", "recurrent_rg", "experiments"} for name in imports)


def test_common_reconstruction_normalization_and_frozen_mean_baseline(tmp_path):
    root, arrays = fixture_run(tmp_path)
    loaded = report.load_public_run(root)[4]
    result = report.comparison_metrics(loaded)
    for model in report.MODELS:
        for split in ("train", "test"):
            mask = arrays["splits"] == split
            rmse = np.sqrt(np.square(arrays[f"{model}_decoded"][mask] - arrays["target_features"][mask]).mean(axis=(0, 2)))
            rows = result["models"][model][split]["by_scale"]
            assert [row["rmse"] for row in rows] == pytest.approx(rmse)
            assert [row["nrmse"] for row in rows] == pytest.approx(rmse / arrays["train_target_std"])
            assert [row["half_mean_vs_full_rmse"] for row in rows] == pytest.approx([.0003] * 4)
            assert result["models"][model][split]["reconstruction_raw"] == pytest.approx(625 * np.mean(rmse ** 2))
        geometry = result["models"][model]["decoder_geometry"]
        assert geometry["direction_norm"] == pytest.approx(1.)
        assert geometry["affine_fit_max_abs_residual"] < 1e-15
        assert len(geometry["abs_cos_per_scale_train_pc1"]) == 4
    assert [row["nrmse"] for row in result["constant_train_mean"]["train"]] == pytest.approx([1.] * 4)


def test_spearman_ties_constant_and_sign():
    assert report._spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.)
    assert report._spearman([1, 2, 3, 4], [1, 1, 2, 2]) == pytest.approx(.8944271909999159)
    assert report._spearman([1, 2, 3], [2, 2, 2]) is None


def test_collapsed_q_and_zero_target_variance_are_reported_not_fabricated(tmp_path):
    root, arrays = fixture_run(tmp_path)
    for suffix in ("direct_g", "direct_full_g", "g_first", "g_second"):
        arrays[f"B_relative_{suffix}"][:] = .2
    arrays["B_relative_decoded"][:] = .01
    arrays["target_features"][:, 0] = .002
    arrays["train_target_mean"] = arrays["target_features"][arrays["splits"] == "train"].mean(axis=0)
    arrays["train_target_std"] = np.sqrt(arrays["target_features"][arrays["splits"] == "train"].var(axis=0, ddof=0).mean(axis=1))
    # Exactly zero by construction (floating summation of .002 otherwise has tiny roundoff).
    arrays["target_features"][:, 0] = 0.
    arrays["train_target_mean"][0] = 0.
    arrays["train_target_std"][0] = 0.
    np.savez(root / "evaluation_arrays.npz", **arrays)
    result = report.comparison_metrics(report.load_public_run(root)[4])
    geometry = result["models"]["B_relative"]["decoder_geometry"]
    assert geometry["direction_identifiable_from_saved_g"] is False
    assert geometry["direction_norm"] is None
    assert geometry["abs_cos_global_train_pc1"] is None
    assert result["models"]["B_relative"]["train"]["by_scale"][0]["nrmse"] is None
    assert result["models"]["B_relative"]["train"]["by_scale"][1]["posthoc_kappa_spearman"] is None
    assert result["normalization_warning"]


@pytest.mark.parametrize("mutation", ["nan", "split_cell", "overlap", "std_scale", "mean_test", "weight_zero", "flow_order", "half_mismatch", "chain_count"])
def test_mixed_or_corrupt_arrays_rejected(tmp_path, mutation):
    root, arrays = fixture_run(tmp_path)
    if mutation == "nan":
        arrays["B_relative_decoded"][0, 0, 0] = np.nan
    elif mutation == "split_cell":
        arrays["splits"][0] = "test"
    elif mutation == "overlap":
        arrays["labels"][4:8] = arrays["labels"][:4]
    elif mutation == "std_scale":
        arrays["train_target_std"] *= 2
    elif mutation == "mean_test":
        arrays["train_target_mean"] = arrays["target_features"].mean(axis=0)
    elif mutation == "weight_zero":
        arrays["relative_weights"][1] = 0
    elif mutation == "flow_order":
        arrays["source_flow_grid"] = arrays["source_flow_grid"][::-1]
    elif mutation == "half_mismatch":
        arrays["A_absolute_g_first"][0, 0, 0] += 1
    elif mutation == "chain_count":
        arrays["cell_ids"][0] = 1
    np.savez(root / "evaluation_arrays.npz", **arrays)
    with pytest.raises(ValueError):
        report.load_public_run(root)


@pytest.mark.parametrize("mutation", ["running", "exponent", "root", "missing_arm", "history_order", "history_missing_scale"])
def test_incomplete_or_claimed_physics_rejected(tmp_path, mutation):
    root, _ = fixture_run(tmp_path)
    summary = json.loads((root / "summary.json").read_text())
    history = json.loads((root / "history.json").read_text())
    if mutation == "running":
        summary["status"] = "running"
    elif mutation == "exponent":
        summary["thermal_nu"] = 1.
    elif mutation == "root":
        summary["critical_kappa"] = .68
    elif mutation == "missing_arm":
        del history[1]["arms"]["B_relative"]
    elif mutation == "history_order":
        history = history[::-1]
    elif mutation == "history_missing_scale":
        history[1]["arms"]["A_absolute"]["reconstruction_by_scale"] = [1, 2]
    (root / "summary.json").write_text(json.dumps(summary))
    (root / "history.json").write_text(json.dumps(history))
    with pytest.raises(ValueError):
        report.load_public_run(root)


def test_render_preserves_inputs_and_refuses_overwrite(tmp_path):
    root, _ = fixture_run(tmp_path)
    pins = {name: report._sha(root / name) for name in report.INPUT_FILES}
    outputs = report.render_report(root)
    assert outputs == sorted(outputs)
    assert len(outputs) == 5
    assert all((root / name).stat().st_size > 1000 for name in outputs)
    result = json.loads((root / "report/comparison.json").read_text())
    assert result["source_sha256"] == pins
    assert not result["old_R_valid_for_new_Q"]
    assert not result["fixed_point_search_performed"]
    assert result["critical_kappa"] is result["thermal_nu"] is None
    assert result["C_updates"] == result["R_updates"] == 0
    assert result["additional_Q_updates_per_arm"] == 2
    assert {name: report._sha(root / name) for name in report.INPUT_FILES} == pins
    with pytest.raises(FileExistsError):
        report.render_report(root)
