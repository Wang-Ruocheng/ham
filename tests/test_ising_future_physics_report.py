"""Public-artifact-only report contract; synthetic arrays, no model loading."""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import report_ising_future_physics as report


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture_run(root, steps=6):
    data_sha = "a" * 64
    pairs = dict(schema_version=1, data_sha256=data_sha)
    write_json(root / "data/pairs.json", pairs)
    pair_sha = report.public_sha(root / "data/pairs.json")
    norm = dict(schema_version=1, data_sha256=data_sha, pairs_sha256=pair_sha,
                fitting_split="train", target_mean=np.zeros((3, 5)).tolist(),
                target_std=np.ones((3, 5)).tolist())
    write_json(root / "data/normalizer.json", norm)
    norm_sha = report.public_sha(root / "data/normalizer.json")
    write_json(root / "dataset_receipt.json", {"warnings": ["原始生产链自相关警告"], "samples_are_iid": False})
    cfg = dict(schema_version=5, model=dict(block_factor=3, dimension=2, theory_dim=1,
               kernel_initialization="random", kernel_sampling="bernoulli", kernel_layout="nonoverlapping",
               fixed_kernel=False, affine_reconstruction=True), training=dict(max_rg_steps=2, epochs=steps),
               loss=dict(weights=dict(anchor=2., coarse_consistency=5., separation=10., commutator=5.), rollout_weight=5.))
    for arm in report.ARMS:
        contract = dict(mode="paired_control" if arm == "control" else "future_physics",
            weight=0. if arm == "control" else 1., labels_used=False, paths=[list(p) for p in report.PATHS],
            observables=list(report.OBSERVABLES), data_sha256=data_sha, pairs_sha256=pair_sha,
            normalizer_sha256=None if arm == "control" else norm_sha,
            target_mean=None if arm == "control" else norm["target_mean"],
            target_std=None if arm == "control" else norm["target_std"])
        manifest = dict(config=cfg, labels_used_in_training=False, data_sha256=data_sha, q_future_physics=contract)
        history = []
        for epoch in (1, steps):
            row = dict(epoch=epoch)
            for split in ("train", "test"):
                for key in ("anchor", "coarse_consistency", "separation", "commutator", "rollout", "kernel_noncollapse", "kernel_sign"):
                    row[split + "_" + key] = .1 / epoch
                row[split + "_rsmi"] = -.2
                if arm == "future":
                    row.update({split + "_future_physics": .04,
                                split + "_future_physics_nrmse": .2,
                                split + "_future_physics_raw_mse": .0004,
                                split + "_future_physics_raw_rmse": .02})
            history.append(row)
        run = root / ("run_" + arm)
        write_json(run / "manifest.json", manifest)
        write_json(run / "summary.json", dict(config=cfg, actual_epochs=steps, requested_epochs=steps, history=history))
        write_json(run / "report/metrics.json", dict(actual_steps=steps, scales=[108, 36, 12]))
        arrays = {}
        for split in ("train", "test"):
            direct = np.arange(12, dtype=float).reshape(4, 3, 1) / 10
            arrays.update({"direct_" + split: direct,
                "R_g_" + split: direct ** 2 + .2,
                "R_of_mean_g_" + split: direct ** 2,
                "rollout_" + split: np.stack((direct[:, 0], direct[:, 0] ** 2, direct[:, 0] ** 4), axis=1),
                "observables_" + split: np.ones((4, 3, 5)) * .1,
                "couplings_" + split: np.column_stack((np.ones(4) * (.42 if split == "train" else .44), np.zeros((4, 6)))),
                "group_ids_" + split: np.repeat([0, 1], 2)})
        np.savez_compressed(run / "report/encodings.npz", **arrays)
    return root


def test_public_report_complete_and_honest_missing_control_loss(tmp_path):
    fixture_run(tmp_path)
    result = report.report_run(tmp_path, expected_steps=6, make_plots=False)
    assert result["status"] == "complete"
    assert result["control_future_loss"] is None
    assert result["thermal_nu"] is None and result["thermal_mode_certified"] is False
    assert not result["root_search_performed"] and not result["objectives_compared_as_total"]
    assert result["dataset_receipt"]["warnings"] == ["原始生产链自相关警告"]
    assert set(result["input_sha256"]) >= {"data/normalizer.json", "run_future/report/encodings.npz"}
    assert all(Path(name).suffix in (".json", ".npz") for name in result["input_sha256"])


def test_early_actual_step_never_claims_fixed_endpoint(tmp_path):
    fixture_run(tmp_path)
    path = tmp_path / "run_future/summary.json"
    summary = json.loads(path.read_text()); summary["actual_epochs"] = 3
    summary["history"][-1]["epoch"] = 3; summary["stopped_by_training_limit"] = True
    write_json(path, summary)
    write_json(tmp_path / "run_future/report/metrics.json", dict(actual_steps=3, scales=[108, 36, 12]))
    result, _ = report.build_report(tmp_path, expected_steps=6)
    assert result["status"] == "incomplete" and not result["formal_comparison_available"]
    assert result["arms"]["future"]["actual_steps"] == 3


def test_wrong_collection_epoch_or_changed_normalizer_is_rejected(tmp_path):
    fixture_run(tmp_path)
    write_json(tmp_path / "run_future/report/metrics.json", dict(actual_steps=5))
    with pytest.raises(ValueError, match="实际训练步数"):
        report.build_report(tmp_path, expected_steps=6)
    write_json(tmp_path / "run_future/report/metrics.json", dict(actual_steps=6))
    path = tmp_path / "data/normalizer.json"
    norm = json.loads(path.read_text()); norm["target_std"][0][0] = 2.
    write_json(path, norm)
    with pytest.raises(ValueError, match="归一化回执"):
        report.build_report(tmp_path, expected_steps=6)


def test_closure_normalization_uses_matching_original_train_scale():
    train = np.array([[[0.], [0.], [0.]], [[2.], [4.], [8.]]])
    test = np.zeros_like(train)
    arrays = dict(direct_train=train, direct_test=test, R_g_test=np.ones_like(test), rollout_test=np.ones_like(test))
    values = report.closure_summary(arrays, "test")
    assert [row["train_target_std"] for row in values] == [2., 4., 4.]
    assert [row["nrmse"] for row in values] == [.5, .25, .25]


def test_true_draw_flow_and_mean_flow_are_distinguished():
    g = np.array([-1., 0., 1., 1., 2., 3.]).reshape(1, 2, 3, 1)
    arrays = dict(direct_draws_test=g, R_g_draws_test=g*g, direct_test=g.mean(1), R_g_test=(g*g).mean(1))
    points, mapped, levels, label = report.flow_points(arrays)
    np.testing.assert_array_equal(points, g.reshape(-1))
    np.testing.assert_array_equal(mapped-points, (g*g-g).reshape(-1))
    np.testing.assert_array_equal(levels, [0, 1, 2, 0, 1, 2])
    assert "逐 C" in label
    arrays.pop("direct_draws_test"); arrays.pop("R_g_draws_test")
    points, mapped, _, label = report.flow_points(arrays)
    assert "不是 R(平均 g)" in label
    assert mapped[0] == 1 and points[0] == 0


def test_missing_r_arrays_never_approximated_from_Q(tmp_path):
    fixture_run(tmp_path)
    path = tmp_path / "run_future/report/encodings.npz"
    with np.load(path) as z: arrays = {key: z[key] for key in z.files if key != "R_g_test"}
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="不能用 Q"):
        report.build_report(tmp_path, expected_steps=6)


def test_no_model_or_physical_cache_readers_and_no_overwrite(tmp_path):
    source = Path(report.__file__).read_text()
    assert "import torch" not in source and "load_run(" not in source
    assert "solve_root" not in source and "brentq" not in source
    with pytest.raises(ValueError, match="模型与物理缓存"):
        report.public_sha(tmp_path / "must_not_read.pt")
    fixture_run(tmp_path)
    report.report_run(tmp_path, expected_steps=6, make_plots=False)
    with pytest.raises(FileExistsError, match="不覆盖"):
        report.report_run(tmp_path, expected_steps=6, make_plots=False)


def test_complete_chinese_plot_render(tmp_path):
    try:
        font = report.select_chinese_font()
    except RuntimeError:
        pytest.skip("No installed Chinese font; never download one in tests")
    fixture_run(tmp_path)
    result = report.report_run(tmp_path, expected_steps=6, font=font)
    assert result["status"] == "complete"
    for name in report.FIGURES:
        assert (tmp_path / "report" / name).stat().st_size > 1000
