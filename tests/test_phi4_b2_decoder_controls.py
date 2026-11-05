"""Synthetic decoder controls: whole-condition validation and immutable public I/O."""
from __future__ import annotations

import numpy as np
import pytest

from scripts import diagnose_phi4_b2_decoders as controls
from test_phi4_b2_closure_report import _fixture, _refresh


def _quadratic_arrays():
    rng = np.random.default_rng(45)
    basis = np.linalg.qr(rng.normal(size=(105, 2)))[0]
    arrays = {}
    for split, values, start in (("train", np.linspace(-1, 1, 8), 0),
                                 ("test", np.array([-.78, -.15, .21, .88]), 20)):
        ids = np.repeat(np.arange(start, start + len(values)), 4)
        g = np.repeat(values, 4)[:, None, None] + np.arange(4)[None, :, None] * .03
        y = .03 + g * basis[:, 0] + np.square(g) * basis[:, 1]
        arrays.update({f"{split}_direct": g, f"{split}_feature_target": y,
                       f"{split}_feature_reconstruction": .03 + g * basis[:, 0],
                       f"{split}_cell_ids": ids})
    return arrays


def test_quadratic_selected_on_train_and_generalizes_without_labels():
    arrays = _quadratic_arrays()
    result, predictions = controls.analyze_arrays(arrays)
    assert result["polynomial_selection"]["selected_degree"] in (2, 3)
    assert result["polynomial_selection"]["test_data_used"] is False
    assert result["polynomial_selection"]["coupling_values_used"] is False
    assert result["physical_dimension_inferred"] is False
    for row in result["rows"]:
        if row["method"] in ("fixed_g_polynomial", "pooled_pca_rank2"):
            assert row["absolute_rmse"] < 1e-12
    assert predictions["fixed_g_polynomial_test_prediction"].shape == (16, 4, 105)
    assert len(result["rows"]) == 7 * 2 * 4


def test_test_values_cannot_choose_degree_or_change_fits():
    arrays = _quadratic_arrays()
    before, _ = controls.analyze_arrays(arrays)
    arrays["test_feature_target"] = np.full_like(arrays["test_feature_target"], 99)
    arrays["test_direct"] *= 5
    after, _ = controls.analyze_arrays(arrays)
    assert before["polynomial_selection"] == after["polynomial_selection"]
    assert before["fits"] == after["fits"]


def test_polynomial_fit_is_fold_local_and_groups_whole_cells():
    arrays = _quadratic_arrays()
    result = controls.choose_polynomial_degree(arrays["train_direct"], arrays["train_feature_target"], arrays["train_cell_ids"])
    for candidate in result["candidates"]:
        for fold in candidate["folds"]:
            keep = arrays["train_cell_ids"] != fold["heldout_train_cell"]
            assert fold["training_observations"] == int(keep.sum()) * 4
            assert fold["g_center"] == pytest.approx(arrays["train_direct"][keep].mean())
            assert fold["g_std"] == pytest.approx(arrays["train_direct"][keep].std())


def test_unstable_fixed_degrees_rejected_without_numerical_repair():
    g = np.repeat([-1., 1.], 8).reshape(4, 4, 1)
    target = np.zeros((4, 4, 105))
    result = controls.choose_polynomial_degree(g, target, np.arange(4))
    assert result["selected_degree"] == 1
    assert result["candidates"][0]["accepted"] is True
    assert result["candidates"][1]["accepted"] is False
    assert result["candidates"][2]["accepted"] is False
    assert controls._fit_polynomial(np.ones(8), np.ones((8, 105)), 1)["accepted"] is False


def test_metrics_reuse_fixed_train_same_scale_std():
    arrays = _quadratic_arrays()
    result, _ = controls.analyze_arrays(arrays)
    for row in result["rows"]:
        reference = arrays["train_feature_target"][:, row["scale"]]
        denominator = np.sqrt(np.var(reference, axis=0).mean())
        assert row["train_target_population_std"] == pytest.approx(denominator)
        assert row["fixed_train_nrmse"] == pytest.approx(row["absolute_rmse"] / denominator)


def test_public_run_preserves_all_inputs_and_refuses_overwrite(tmp_path):
    root = _fixture(tmp_path / "training")
    (root / "checkpoint.pt").write_bytes(b"must not be read")
    (root / "data.npz").write_bytes(b"must not be read")
    before = {path.name: controls.sha256(path) for path in root.iterdir()}
    receipt, result = controls.run_decoder_controls(root, tmp_path / "control")
    assert before == {path.name: controls.sha256(path) for path in root.iterdir()}
    assert receipt["source_public_sha256_unchanged"] is True
    assert receipt["labels_used_for_fit"] is receipt["test_data_used_for_fit_or_selection"] is False
    assert receipt["checkpoints_loaded"] is receipt["raw_physical_data_loaded"] is False
    assert receipt["QCR_training_performed"] is receipt["GPU_used"] is False
    assert all(pool["num_threads"] == 1 for pool in receipt["threadpools"])
    assert result["critical_r"] is result["thermal_nu"] is None
    for name, digest in receipt["outputs_sha256"].items():
        assert controls.sha256(tmp_path / "control" / name) == digest
    with pytest.raises(FileExistsError):
        controls.run_decoder_controls(root, tmp_path / "control")


def test_tampered_public_arrays_rejected(tmp_path):
    root = _fixture(tmp_path / "training")
    with np.load(root / "evaluation_arrays.npz") as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["legacy_train_direct"] += .1
    np.savez_compressed(root / "evaluation_arrays.npz", **arrays)
    with pytest.raises(ValueError, match="SHA256"):
        controls.run_decoder_controls(root, tmp_path / "control")
    assert not (tmp_path / "control").exists()


def test_unpaired_public_statistics_rejected_even_with_updated_sha(tmp_path):
    root = _fixture(tmp_path / "training")
    with np.load(root / "evaluation_arrays.npz") as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["all_start_test_feature_target"] += .1
    np.savez_compressed(root / "evaluation_arrays.npz", **arrays)
    _refresh(root)
    with pytest.raises(ValueError, match="Paired"):
        controls.run_decoder_controls(root, tmp_path / "control")


def test_diagnostic_output_disjoint_and_budget_limited(tmp_path):
    root = _fixture(tmp_path / "training")
    with pytest.raises(ValueError, match="disjoint"):
        controls.run_decoder_controls(root, root / "control")
    with pytest.raises(ValueError, match="CPU budget"):
        controls.run_decoder_controls(root, tmp_path / "control", cpu_budget_seconds=121)


def test_fixed_condition_cap_rejects_without_changing_degree(monkeypatch):
    monkeypatch.setattr(np.linalg, "cond", lambda _: 1e8 + 1)
    fit = controls._fit_polynomial(np.arange(8), np.zeros((8, 105)), 2)
    assert fit["degree"] == 2
    assert fit["accepted"] is False
    assert fit["design_condition"] > 1e8
    assert "coefficients" not in fit


def test_output_figure_uses_chinese_explanations(tmp_path, monkeypatch):
    result, _ = controls.analyze_arrays(_quadratic_arrays())
    plt = controls._plotting(None)
    texts, close = [], plt.close
    def capture(fig=None):
        if hasattr(fig, "findobj"):
            from matplotlib.text import Text
            texts.extend(text.get_text() for text in fig.findobj(Text) if text.get_text())
        close(fig)
    monkeypatch.setattr(plt, "close", capture)
    controls.plot_errors(result, tmp_path)
    assert any("几何下界" in text for text in texts)
    assert any("留出条件" in text for text in texts)
    assert any("固定训练标准差" in text for text in texts)
    assert any("不是新 Q 或物理维度认证" in text for text in texts)


def test_sources_rechecked_after_numerical_analysis(tmp_path, monkeypatch):
    root = _fixture(tmp_path / "training")
    original = controls.analyze_arrays
    def alter_after_analysis(*args, **kwargs):
        result = original(*args, **kwargs)
        (root / "summary.json").write_text("{}", encoding="utf-8")
        return result
    monkeypatch.setattr(controls, "analyze_arrays", alter_after_analysis)
    with pytest.raises(ValueError, match="sources changed"):
        controls.run_decoder_controls(root, tmp_path / "control")
    assert not (tmp_path / "control").exists()


@pytest.mark.parametrize("prefix,suffix", [("openblas_", "64_"),
    ("openblas_", "_64_"), ("scipy_openblas_", "64_"), ("scipy_openblas64_", "_")])
def test_missing_optional_threadpoolctl_uses_verified_bundled_symbols(monkeypatch, prefix, suffix):
    state = {"threads": 3}
    def getter():
        return state["threads"]
    def setter(value):
        state["threads"] = value
    library = type("BundledOpenBLAS", (), {})()
    setattr(library, prefix + "get_num_threads" + suffix, getter)
    setattr(library, prefix + "set_num_threads" + suffix, setter)
    monkeypatch.setattr(controls, "threadpool_limits", None)
    monkeypatch.setattr(controls, "threadpool_info", None)
    monkeypatch.setattr(controls.ctypes, "CDLL", lambda _: library)
    with controls._single_thread_blas() as pools:
        assert state["threads"] == 1
        assert all(pool["num_threads"] == 1 for pool in pools)
    assert state["threads"] == 3


def test_module_import_does_not_require_threadpoolctl(monkeypatch):
    import builtins
    import runpy
    original = builtins.__import__
    def without_threadpoolctl(name, *args, **kwargs):
        if name == "threadpoolctl":
            raise ImportError("simulated remote optional dependency absent")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", without_threadpoolctl)
    namespace = runpy.run_path(controls.__file__, run_name="test_missing_optional_dependency")
    assert namespace["threadpool_limits"] is namespace["threadpool_info"] is None
