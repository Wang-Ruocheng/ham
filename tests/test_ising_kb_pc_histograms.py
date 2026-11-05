"""Fixed-PCA distributions from public arrays only; no model or new sampling."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import plot_ising_kb_pc_histograms as plot
from test_ising_kb_pc12_plot import _sha, _write_synthetic_report


def test_principal_scores_keep_saved_center_axes_and_all_scales_without_whitening():
    direct = np.arange(30, dtype=float).reshape(5, 3, 2) * .2
    center = np.array([4., -3.])
    components = np.array([[0., -1.], [1., 0.]])
    report = SimpleNamespace(
        encoded=dict(train_direct=direct[:3], test_direct=direct[3:]),
        grid=dict(pca_center=center, pca_components=components))
    before = direct.copy()
    scores = plot.principal_scores(report)
    assert scores.shape == (5, 3, 2)
    np.testing.assert_array_equal(scores, (direct - center) @ components.T)
    np.testing.assert_array_equal(scores[:, :, 0], -direct[:, :, 1] - 3.)
    np.testing.assert_array_equal(direct, before)
    # Terminal k=2 remains a direct observation; it is neither dropped nor refitted.
    np.testing.assert_array_equal(scores[:, 2], (direct[:, 2] - center) @ components.T)


def test_summary_uses_shared_equal_width_bins_and_independently_normalized_densities():
    scores = np.arange(72, dtype=float).reshape(8, 3, 3) / 11.
    scores[:, 1, 0] -= 4.
    scores[:, 2, 2] *= .01
    original = scores.copy()
    fractions = np.array([.7, .25, .05])
    summary = plot.distribution_summary(scores, fractions, bins=7)
    assert summary["records_per_scale"] == 8
    assert summary["dimensions"] == 3
    assert summary["scales"] == [0, 1, 2]
    assert len(summary["coordinates"]) == 3
    for pc, coordinate in enumerate(summary["coordinates"]):
        assert coordinate["principal_component"] == pc + 1
        assert coordinate["training_input_variance_fraction"] == fractions[pc]
        edges = np.asarray(coordinate["bin_edges"])
        assert len(edges) == 8
        np.testing.assert_allclose(np.diff(edges), np.diff(edges)[0])
        assert edges[0] <= scores[:, :, pc].min()
        assert edges[-1] >= scores[:, :, pc].max()
        for scale, row in enumerate(coordinate["by_scale"]):
            sample = scores[:, scale, pc]
            assert row["scale"] == scale
            assert row["n"] == len(sample)
            assert row["mean"] == pytest.approx(sample.mean())
            assert row["std"] == pytest.approx(sample.std(ddof=0))
            assert row["minimum"] == sample.min()
            assert row["maximum"] == sample.max()
            expected, _ = np.histogram(sample, edges)
            np.testing.assert_array_equal(row["histogram_counts"], expected)
            assert sum(row["histogram_counts"]) == len(sample)
            assert np.dot(row["histogram_density"], np.diff(edges)) == pytest.approx(1.)
    np.testing.assert_array_equal(scores, original)
    json.dumps(summary, allow_nan=False)


def test_constant_component_has_finite_bins_and_unit_density_area():
    result = plot.distribution_summary(np.full((4, 3, 1), 2.5), [1.], bins=4)
    coordinate, = result["coordinates"]
    widths = np.diff(coordinate["bin_edges"])
    assert np.all(widths > 0)
    for row in coordinate["by_scale"]:
        assert row["std"] == 0.
        assert sum(row["histogram_counts"]) == 4
        assert np.isfinite(row["histogram_density"]).all()
        assert np.dot(row["histogram_density"], widths) == pytest.approx(1.)


@pytest.mark.parametrize("kind", ["rank", "empty", "scales", "dimension", "nan", "inf",
    "fraction_shape", "fraction_nan", "fraction_negative", "fraction_sum", "bins_small",
    "bins_float", "bins_bool"])
def test_invalid_distribution_inputs_are_rejected(kind):
    scores, fractions, bins = np.ones((4, 3, 2)), np.array([.7, .3]), 4
    if kind == "rank":
        scores = scores[:, 0]
    elif kind == "empty":
        scores = scores[:0]
    elif kind == "scales":
        scores = scores[:, :2]
    elif kind == "dimension":
        scores, fractions = scores[:, :, :0], np.array([])
    elif kind in ("nan", "inf"):
        scores[0, 0, 0] = np.nan if kind == "nan" else np.inf
    elif kind == "fraction_shape":
        fractions = fractions[:1]
    elif kind == "fraction_nan":
        fractions[0] = np.nan
    elif kind == "fraction_negative":
        fractions = np.array([1.2, -.2])
    elif kind == "fraction_sum":
        fractions *= .5
    elif kind == "bins_small":
        bins = 1
    elif kind == "bins_float":
        bins = 4.
    else:
        bins = True
    with pytest.raises(ValueError):
        plot.distribution_summary(scores, fractions, bins)


def test_outputs_cannot_overwrite_or_enter_frozen_source(tmp_path):
    source, output = tmp_path / "source", tmp_path / "existing"
    source.mkdir()
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("保留历史")
    with pytest.raises(FileExistsError):
        plot.write_plots(source, output)
    assert marker.read_text() == "保留历史"
    with pytest.raises(ValueError, match="(?i)(source|frozen|inside)"):
        plot.write_plots(source, source / "new")
    assert not (source / "new").exists()


def test_tiny_plot_smoke_preserves_sources_and_exports_all_saved_pc_coordinates(tmp_path):
    source, output = tmp_path / "source", tmp_path / "histograms"
    source.mkdir()
    for dimension in (5, 10):
        _write_synthetic_report(source, dimension)
    before = {str(path.relative_to(source)): _sha(path)
              for path in source.rglob("*") if path.is_file()}
    result = plot.write_plots(source, output, bins=8)
    assert result["status"] == "complete"
    assert result["model_inference"] is False
    assert result["physical_sampling"] is False
    assert result["training_updates"] == 0
    assert result["pca_refit"] is False
    assert {path.name for path in output.iterdir()} == {
        "PC_histograms.png", "PC_variance_spectrum.png", "PC_histogram_arrays.npz", "PC_histogram_report.json"}
    for name, digest in result["output_sha256"].items():
        assert _sha(output / name) == digest
        assert (output / name).stat().st_size > 0
    with np.load(output / "PC_histogram_arrays.npz", allow_pickle=False) as arrays:
        for dimension, summary in zip((5, 10), result["models"]):
            report = plot.load_report(source, dimension)
            expected = plot.principal_scores(report)
            prefix = f"dim{dimension}_"
            np.testing.assert_array_equal(arrays[prefix + "principal_scores"], expected)
            np.testing.assert_array_equal(arrays[prefix + "pca_center"], report.grid["pca_center"])
            np.testing.assert_array_equal(arrays[prefix + "pca_components"], report.grid["pca_components"])
            expected_split = np.arange(len(expected)) < len(report.encoded["train_direct"])
            np.testing.assert_array_equal(arrays[prefix + "split_is_train"], expected_split)
            assert summary["dimension"] == dimension
            assert len(summary["coordinates"]) == dimension
    assert {str(path.relative_to(source)): _sha(path)
            for path in source.rglob("*") if path.is_file()} == before


def test_sha_is_checked_again_after_plotting(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "new"
    source.mkdir()
    for dimension in (5, 10):
        _write_synthetic_report(source, dimension)
    def mutate_source(reports, destination):
        path = source / "dim5" / "report" / "fixed_points.json"
        path.write_text(path.read_text() + "\n")
    monkeypatch.setattr(plot, "_plot_histograms", lambda *args: None)
    monkeypatch.setattr(plot, "_plot_variance", mutate_source)
    with pytest.raises(ValueError, match="changed during plotting"):
        plot.write_plots(source, output)
    assert not (output / "PC_histogram_report.json").exists()
