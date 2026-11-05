"""Synthetic saved-array checks only; these are not physical/training evidence."""
import ast
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import report_phi4_hopping_b3_macro_R_ablation as report


def fixture_run(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    rng = np.random.default_rng(123)
    direct = rng.normal(size=(60, 3, 1)) * .03
    pairs = report.EXPECTED_PAIRS.copy()
    splits = np.repeat(["train", "train", "test"] * 5, 4)
    labels = np.column_stack((np.repeat(np.linspace(.32, 1.04, 15), 4), np.ones(60), np.zeros(60)))
    arrays = {"direct_g": direct, "labels": labels, "splits": splits,
              "cell_ids": np.repeat(np.arange(15), 4), "all_start_pairs": pairs,
              "flow_grid": np.linspace(-.1, .1, 101)[:, None],
              "train_target_mean": direct[splits == "train"].mean(axis=0).ravel(),
              "train_target_std": direct[splits == "train"].std(axis=0, ddof=0).ravel()}
    for index, model in enumerate(report.MODELS):
        all_start = np.stack([direct[:, t] + .003 * (index + 1) for _, t in pairs], axis=1)
        one_step = np.stack([all_start[:, np.flatnonzero((pairs[:, 0] == s) & (pairs[:, 1] == s + 1))[0]] for s in range(2)], axis=1)
        rollout = np.stack([direct[:, 0]] + [all_start[:, np.flatnonzero((pairs[:, 0] == 0) & (pairs[:, 1] == t))[0]] for t in range(1, 3)], axis=1)
        arrays.update({f"{model}_all_start": all_start, f"{model}_one_step": one_step,
                       f"{model}_rollout": rollout, f"{model}_flow_next": arrays["flow_grid"] * .9 + index * .001})
    np.savez(root / "evaluation_arrays.npz", **arrays)
    history = [{"step": step, "arms": {arm: {"objective": .01 / (step + 1),
                "equal_target_objective": .012 / (step + 1), "macro_source_objective": .015 / (step + 1)}
                for arm in report.ARMS}} for step in (0, 1, 2)]
    for name, value in (("plan", {}), ("history", history), ("metrics", {}),
                        ("summary", {"status": "complete", "thermal_nu": None, "critical_kappa": None})):
        (root / f"{name}.json").write_text(json.dumps(value))
    return root, arrays


def test_module_has_no_model_or_training_dependency():
    source = Path(report.__file__).read_text()
    tree = ast.parse(source)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(item.name for item in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module)
    assert not any(name and name.split(".")[0] in {"torch", "recurrent_rg", "experiments"} for name in imports)


def test_metrics_use_saved_target_std_and_same_pair_identity(tmp_path):
    root, arrays = fixture_run(tmp_path)
    _, _, _, _, loaded, _ = report.load_public_run(root)
    result = report.comparison_metrics(loaded)
    for index, model in enumerate(report.MODELS):
        for split in ("train", "test"):
            values = result["models"][model][split]
            expected = .003 * (index + 1)
            assert values["equal_target_objective"] == pytest.approx(10 * expected ** 2)
            assert values["macro_source_objective"] == pytest.approx(10 * expected ** 2)
            for row in values["pairs"]:
                assert row["rmse"] == pytest.approx(expected)
                assert row["nrmse"] == pytest.approx(expected / arrays["train_target_std"][row["target"]])
                if split == "train":
                    assert row["constant_train_mean_nrmse"] == pytest.approx(1.)
    a = result["models"]["source"]["test"]["pairs"]
    b = result["models"]["B_macro_source"]["test"]["pairs"]
    assert [row["identity_rmse"] for row in a] == [row["identity_rmse"] for row in b]


def test_two_target_std_vector_is_supported_without_renormalizing(tmp_path):
    root, arrays = fixture_run(tmp_path)
    arrays["train_target_std"] = np.asarray([.1, .2])
    np.savez(root / "evaluation_arrays.npz", **arrays)
    loaded = report.load_public_run(root)[4]
    assert report.comparison_metrics(loaded)["train_target_std_by_target"] == [.1, .2]


@pytest.mark.parametrize("mutation", ["nan", "pair_order", "cross_cache", "split_cell", "label_overlap", "std_zero", "std_shape", "flow_order", "chain_count"])
def test_corrupt_or_mixed_arrays_fail_closed(tmp_path, mutation):
    root, arrays = fixture_run(tmp_path)
    if mutation == "nan":
        arrays["direct_g"][0, 0, 0] = np.nan
    elif mutation == "pair_order":
        arrays["all_start_pairs"] = arrays["all_start_pairs"][::-1]
    elif mutation == "cross_cache":
        arrays["source_one_step"][0, 0, 0] += 1
    elif mutation == "split_cell":
        arrays["splits"][0] = "test"
    elif mutation == "label_overlap":
        arrays["labels"][8:12] = arrays["labels"][:4]
    elif mutation == "std_zero":
        arrays["train_target_std"][2] = 0
    elif mutation == "std_shape":
        arrays["train_target_std"] = np.ones(4)
    elif mutation == "flow_order":
        arrays["flow_grid"] = arrays["flow_grid"][::-1]
    elif mutation == "chain_count":
        arrays["cell_ids"][0] = 1
    np.savez(root / "evaluation_arrays.npz", **arrays)
    with pytest.raises(ValueError):
        report.load_public_run(root)


@pytest.mark.parametrize("mutation", ["running", "exponent", "root", "missing_arm", "history_order"])
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
        del history[1]["arms"]["B_macro_source"]
    elif mutation == "history_order":
        history = history[::-1]
    (root / "summary.json").write_text(json.dumps(summary))
    (root / "history.json").write_text(json.dumps(history))
    with pytest.raises(ValueError):
        report.load_public_run(root)


def test_render_does_not_mutate_inputs_and_refuses_overwrite(tmp_path):
    root, _ = fixture_run(tmp_path)
    pins = {name: report._sha(root / name) for name in report.INPUT_FILES}
    outputs = report.render_report(root)
    assert outputs == sorted(outputs)
    assert len(outputs) == 5
    assert all((root / item).stat().st_size > 1000 for item in outputs)
    result = json.loads((root / "report/comparison.json").read_text())
    assert result["source_sha256"] == pins
    assert not result["fixed_point_search_performed"]
    assert result["thermal_nu"] is result["critical_kappa"] is None
    assert sum(result["per_pair_weight_equal_target"]) == pytest.approx(1.)
    assert sum(result["per_pair_weight_macro_source"]) == pytest.approx(1.)
    assert result["per_pair_weight_macro_source"][-1] == pytest.approx(2 / 3)
    assert {name: report._sha(root / name) for name in report.INPUT_FILES} == pins
    with pytest.raises(FileExistsError):
        report.render_report(root)


def test_distinct_pair_errors_reproduce_b3_weight_coefficients(tmp_path):
    root, arrays = fixture_run(tmp_path)
    errors = np.asarray([.01, .02, .03])
    for model in report.MODELS:
        arrays[f"{model}_all_start"] = np.stack(
            [arrays["direct_g"][:, target] + error for (_, target), error in zip(report.EXPECTED_PAIRS, errors)], axis=1)
        arrays[f"{model}_one_step"] = arrays[f"{model}_all_start"][:, [0, 2]]
        arrays[f"{model}_rollout"] = np.concatenate(
            [arrays["direct_g"][:, :1], arrays[f"{model}_all_start"][:, :2]], axis=1)
    np.savez(root / "evaluation_arrays.npz", **arrays)
    loaded = report.load_public_run(root)[4]
    result = report.comparison_metrics(loaded)["models"]["source"]["test"]
    assert result["equal_target_objective"] == pytest.approx(np.dot([5, 2.5, 2.5], errors ** 2))
    assert result["macro_source_objective"] == pytest.approx(np.dot([5 / 3, 5 / 3, 20 / 3], errors ** 2))


def test_geometry_and_warning_count_are_historical_b3():
    text = "\n".join(report.LIMITATIONS)
    assert "12 项采样质量警告" in text
    assert "全部三个预测对" in text
    assert "1:2 加权" in text
    assert report.EXPECTED_PAIRS.tolist() == [[0, 1], [0, 2], [1, 2]]


def test_render_draws_full_grid_displacements(tmp_path, monkeypatch):
    root, arrays = fixture_run(tmp_path)
    loaded = report.load_public_run(root)[4]
    plt = report._plot_setup()
    original_quiver = plt.Axes.quiver
    seen = []
    def capture(self, starts, rows, deltas, zero, **kwargs):
        seen.append((np.asarray(starts), np.asarray(deltas)))
        return original_quiver(self, starts, rows, deltas, zero, **kwargs)
    monkeypatch.setattr(plt.Axes, "quiver", capture)
    report._flow(loaded, tmp_path / "flow.png", plt)
    assert len(seen) == 3
    for model, (starts, deltas) in zip(report.MODELS, seen):
        assert starts.min() == arrays["flow_grid"].min()
        assert starts.max() == arrays["flow_grid"].max()
        indices = [np.argmin(abs(arrays["flow_grid"][:, 0] - item)) for item in starts]
        assert np.allclose(deltas, arrays[f"{model}_flow_next"][indices, 0] - starts)

