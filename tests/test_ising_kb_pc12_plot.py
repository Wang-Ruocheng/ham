"""Offline PC1/PC2 contracts using only tiny synthetic public JSON/NPZ files."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import plot_ising_kb_pc12 as plot


def _json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _refresh_receipt(source):
    files = {str(path.relative_to(source)): _sha(path)
             for path in source.glob("dim*/report/*") if path.is_file()}
    _json(source / "download_receipt.json", dict(status="verified", all_sha256_verified=True, files=files))


def _write_synthetic_report(source, dimension, *, test_shift=0., last_train_shift=0.):
    """A nonlinear R yields a nonzero, independently checkable Jensen gap."""
    rng = np.random.default_rng(300 + dimension)
    K, B = np.linspace(.2, .68, 4), np.array([-.01, 0., .01])
    labels = np.repeat(np.stack(np.meshgrid(K, B, indexing="ij"), -1).reshape(-1, 2), 2, axis=0)
    cells = np.repeat(np.arange(len(K) * len(B)), 2)
    is_train = cells % 3 != 1
    direct = rng.normal(scale=.035, size=(len(labels), 3, dimension))
    direct[..., 0] += labels[:, None, 0]
    direct[..., 1] += 8 * labels[:, None, 1]
    direct += np.arange(dimension)[None, None] * .3
    direct[~is_train] += test_shift
    direct[is_train, -1] += last_train_shift
    train, test = direct[is_train], direct[~is_train]
    inputs = train[:, :-1].reshape(-1, dimension)
    center = inputs.mean(0)
    _, singular, components = np.linalg.svd(inputs - center, full_matrices=False)
    for component in components:
        if component[np.argmax(abs(component))] < 0:
            component *= -1
    fractions = singular ** 2 / np.sum(singular ** 2)

    def mapping(value):
        return .8 * value + .15 * value ** 2 + .02

    shape = (len(K), len(B), 3, dimension)
    means = np.empty(shape)
    mean_R = np.empty(shape)
    for i, k in enumerate(K):
        for j, b in enumerate(B):
            selected = (labels[:, 0] == k) & (labels[:, 1] == b)
            means[i, j] = direct[selected].mean(0)
            mean_R[i, j] = mapping(direct[selected]).mean(0)
    R_of_mean = mapping(means)
    displacement = R_of_mean - means
    grid = dict(K=K, B=B, mean_by_scale=means, pca_center=center,
        pca_components=components, R_of_mean=R_of_mean, mean_R=mean_R,
        displacement=displacement, jensen_difference=mean_R - R_of_mean,
        projected_mean_by_scale=(means - center) @ components.T,
        projected_true_displacement=displacement @ components.T)
    encoded = dict(train_direct=train, test_direct=test,
        train_labels=labels[is_train], test_labels=labels[~is_train],
        train_one_step=mapping(train[:, :-1]), test_one_step=mapping(test[:, :-1]))
    metrics = dict(theory_dim=dimension, checkpoint_epoch=12000, labels_used_in_training=False,
        coupling_names=["K", "B"], lattice_sizes=[36, 12, 4],
        pca=dict(center=center.tolist(), components=components.tolist(), singular_values=singular.tolist(),
                 explained_variance_fraction=fractions.tolist(),
                 fit_source="TRAIN direct[:, :-1] only; no labels or TEST"))
    roots = dict(roots=[dict(root_id=0, g=center.tolist(), residual_norm=1e-12,
        geometric_support_candidate=False, physical_support_certified=False)], accepted_unique_root_count=1)
    report = source / f"dim{dimension}" / "report"
    report.mkdir(parents=True)
    _json(report / "metrics.json", metrics)
    _json(report / "fixed_points.json", roots)
    np.savez_compressed(report / "encodings.npz", **encoded)
    np.savez_compressed(report / "grid_encodings.npz", **grid)
    _refresh_receipt(source)
    return report


def _mutate_npz(path, function):
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in source.files}
    function(arrays)
    np.savez_compressed(path, **arrays)


def test_points_are_centered_but_true_vectors_are_not():
    center = np.array([10., -4., 3.])
    components = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., -1.]])
    points = np.array([[11., -2., 8.], [9., -7., 2.]])
    displacement = np.array([[1., 2., 5.], [-1., -3., -1.]])
    point2, vector2 = plot.project_pc12(points, displacement, center, components)
    np.testing.assert_array_equal(point2, (points - center) @ components[:2].T)
    np.testing.assert_array_equal(vector2, displacement @ components[:2].T)
    np.testing.assert_array_equal(point2 + vector2, (points + displacement - center) @ components[:2].T)
    # Retain the saved PCA axis signs, even when PC1 is a negative axis.
    np.testing.assert_array_equal(vector2[0], [-2., 1.])
    assert not np.array_equal(vector2, (displacement - center) @ components[:2].T)


@pytest.mark.parametrize("invalid", ["nonfinite", "nonorthogonal", "shape"])
def test_projection_rejects_invalid_geometry(invalid):
    points, vector, center, components = np.ones((2, 3)), np.ones((2, 3)), np.zeros(3), np.eye(3)
    if invalid == "nonfinite":
        vector[0, 0] = np.nan
    elif invalid == "nonorthogonal":
        components[1] = components[0]
    else:
        vector = vector[:1]
    with pytest.raises(ValueError):
        plot.project_pc12(points, vector, center, components)


def test_projection_energy_is_separate_from_explained_static_variance():
    drift = np.broadcast_to([.3, .4, 1.2], (2, 3, 3)).copy()
    result = plot.drift_diagnostics(drift, np.eye(3))
    assert result["full_displacement_rms_norm"] == pytest.approx(1.3)
    assert result["pc12_displacement_rms_norm"] == pytest.approx(.5)
    assert result["outside_pc12_displacement_rms_norm"] == pytest.approx(1.2)
    assert result["outside_pc12_energy_fraction"] == pytest.approx(1.44 / 1.69)
    assert result["outside_pc12_rms_fraction"] == pytest.approx(1.2 / 1.3)
    assert result["zero_flow"] is False
    zero = plot.drift_diagnostics(np.zeros((2, 3)), np.eye(3))
    assert zero["zero_flow"] is True
    assert zero["outside_pc12_energy_fraction"] is None
    assert zero["outside_pc12_rms_fraction"] is None


@pytest.mark.parametrize("dimension", [5, 10])
def test_verified_public_reports_load_without_models_and_retain_jensen_gap(tmp_path, dimension):
    report_path = _write_synthetic_report(tmp_path, dimension)
    before = {str(path): _sha(path) for path in report_path.iterdir()}
    report = plot.load_report(tmp_path, dimension)
    assert report.dimension == dimension
    assert set(report.source_hashes) == {f"dim{dimension}/report/{name}" for name in plot.SOURCE_NAMES}
    assert {str(path): _sha(path) for path in report_path.iterdir()} == before
    result = plot.summarize(report)
    assert result["certified_physical_roots"] == 0
    assert result["accepted_numerical_roots"] == 1
    assert result["saved_roots"][0]["interpretation"] == "仅已存数值根；不标注为相变点"
    assert all(row["jensen_difference_rms_norm"] > 0 for row in result["displacement_by_input_scale"])
    assert any(row["R_of_mean"] != row["mean_R"] for row in result["displacement_by_input_scale"])


def test_chart_uses_only_train_input_scales_not_test_or_final_scale(tmp_path):
    for name, test_shift, last_train_shift in (("base", 0., 0.), ("changed", 20., -15.)):
        source = tmp_path / name
        source.mkdir()
        _write_synthetic_report(source, 5, test_shift=test_shift, last_train_shift=last_train_shift)
    original = plot.load_report(tmp_path / "base", 5)
    changed = plot.load_report(tmp_path / "changed", 5)
    np.testing.assert_array_equal(original.grid["pca_center"], changed.grid["pca_center"])
    np.testing.assert_array_equal(original.grid["pca_components"], changed.grid["pca_components"])
    assert not np.array_equal(original.grid["mean_by_scale"], changed.grid["mean_by_scale"])


def test_loader_rejects_test_contaminated_pca_even_when_json_and_npz_agree(tmp_path):
    report = _write_synthetic_report(tmp_path, 5)
    metrics = json.loads((report / "metrics.json").read_text())
    metrics["pca"]["center"] = (np.asarray(metrics["pca"]["center"]) + 5).tolist()
    _json(report / "metrics.json", metrics)
    _mutate_npz(report / "grid_encodings.npz", lambda arrays: arrays.update(pca_center=arrays["pca_center"] + 5))
    _refresh_receipt(tmp_path)
    with pytest.raises(ValueError, match="TRAIN-input-only"):
        plot.load_report(tmp_path, 5)


def test_loader_rejects_unverified_receipt_or_source_sha_change(tmp_path):
    report = _write_synthetic_report(tmp_path, 5)
    receipt = json.loads((tmp_path / "download_receipt.json").read_text())
    receipt["all_sha256_verified"] = False
    _json(tmp_path / "download_receipt.json", receipt)
    with pytest.raises(ValueError, match="verified"):
        plot.load_report(tmp_path, 5)
    _refresh_receipt(tmp_path)
    (report / "fixed_points.json").write_text("{}")
    with pytest.raises(ValueError, match="fingerprint"):
        plot.load_report(tmp_path, 5)


def test_loader_checks_saved_mean_R_against_ensemble_one_step_not_only_jensen_identity(tmp_path):
    report = _write_synthetic_report(tmp_path, 5)
    def damage(arrays):
        arrays["mean_R"][..., :2, :] += .4
        arrays["jensen_difference"] = arrays["mean_R"] - arrays["R_of_mean"]
    _mutate_npz(report / "grid_encodings.npz", damage)
    _refresh_receipt(tmp_path)
    with pytest.raises(ValueError, match="(?i)(mean.*R|ensemble|one.step)"):
        plot.load_report(tmp_path, 5)


def test_loader_rejects_parameter_cell_overlap(tmp_path):
    report = _write_synthetic_report(tmp_path, 5)
    def overlap(arrays):
        arrays["test_labels"][0] = arrays["train_labels"][0]
    _mutate_npz(report / "encodings.npz", overlap)
    _refresh_receipt(tmp_path)
    with pytest.raises(ValueError, match="overlap"):
        plot.load_report(tmp_path, 5)


def test_new_plots_do_not_overwrite_existing_directory_or_enter_frozen_reports(tmp_path):
    source, output = tmp_path / "source", tmp_path / "old-output"
    source.mkdir()
    output.mkdir()
    marker = output / "preserve.txt"
    marker.write_text("历史报告不可覆盖")
    with pytest.raises(FileExistsError):
        plot.write_plots(source, output)
    assert marker.read_text() == "历史报告不可覆盖"
    for dimension in (5, 10):
        _write_synthetic_report(source, dimension)
    with pytest.raises(ValueError, match="(?i)(frozen|source|report|inside)"):
        plot.write_plots(source, source / "dim5" / "report" / "new-plots")
    assert not (source / "dim5" / "report" / "new-plots").exists()


def test_tiny_synthetic_plot_smoke_keeps_sources_and_saved_semantics(tmp_path):
    source, output = tmp_path / "source", tmp_path / "pc12"
    source.mkdir()
    for dimension in (5, 10):
        _write_synthetic_report(source, dimension)
    before = {str(path.relative_to(source)): _sha(path)
              for path in source.rglob("*") if path.is_file()}
    result = plot.write_plots(source, output)
    assert result["status"] == "complete"
    assert result["additional_model_inference"] is False
    assert result["physical_sampling"] is False
    assert result["training_updates"] == 0
    assert result["new_root_search"] is False
    assert result["interpolation_or_off_data_grid"] is False
    expected = ("PC12_phase_portrait.png", "PC12_coupling_map.png", "PC12_scale_trajectories.png",
                "PC12_projection_diagnostics.png", "PC12_arrays.npz", "PC12_report.json")
    assert {path.name for path in output.iterdir()} == set(expected)
    for name in expected:
        assert (output / name).stat().st_size > 0
    for name, digest in result["output_sha256"].items():
        assert _sha(output / name) == digest
    with np.load(output / "PC12_arrays.npz", allow_pickle=False) as saved:
        for dimension in (5, 10):
            report = plot.load_report(source, dimension)
            grid = report.grid
            prefix = f"dim{dimension}_"
            basis = grid["pca_components"][:2]
            np.testing.assert_allclose(saved[prefix + "pc12_R_of_mean_displacement"], grid["displacement"] @ basis.T)
            np.testing.assert_allclose(saved[prefix + "pc12_mean_R_displacement"],
                (grid["mean_R"] - grid["mean_by_scale"]) @ basis.T)
            assert not np.allclose(saved[prefix + "pc12_R_of_mean_displacement"],
                                   saved[prefix + "pc12_mean_R_displacement"])
    assert {str(path.relative_to(source)): _sha(path)
            for path in source.rglob("*") if path.is_file()} == before
