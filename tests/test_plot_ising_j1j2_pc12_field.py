"""Tiny synthetic arrays only: no SSH, model, checkpoint, or inference."""
from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

from scripts import plot_ising_j1j2_pc12_field as plot


def _example(tmp_path):
    x, y = np.linspace(-1, 1, 39), np.linspace(-.3, .3, 31)
    xx, yy = np.meshgrid(x, y)
    delta = np.stack((.08 * xx - .15 * yy, -.035 * yy), axis=-1)
    rng = np.random.default_rng(17)
    values = dict(grid_x=x, grid_y=y, projected_delta=delta,
                  full_delta=np.concatenate((delta, np.zeros((*xx.shape, 3))), axis=-1),
                  support_mask=(xx / .85) ** 2 + (yy / .25) ** 2 < 1,
                  projected_test=rng.normal(size=(16, 3, 2)) * [.3, .08],
                  projected_train=rng.normal(size=(16, 3, 2)) * [.3, .08],
                  rho_test=np.repeat([0., .35, .7, 1.], 4),
                  observed_projected_delta_test=rng.normal(scale=.01, size=(16, 3, 2)),
                  pca_ratio=np.array([.9336, .0657, .0004, .0002, .0001]))
    field, metadata = tmp_path / "field.npz", tmp_path / "export.json"
    np.savez_compressed(field, **values)
    details = dict(status="complete", diagnostic_only=True, equilibrium_not_validated=True,
                   field_npz_sha256=plot.sha256(field),
                   support_description="训练集输入尺度的投影凸包与近邻距离交集，仅几何掩码。",
                   slice_description="其余三个主成分固定训练均值。",
                   source_hashes={"checkpoint": "a" * 64})
    metadata.write_text(json.dumps(details, ensure_ascii=False), encoding="utf-8")
    return field, metadata, values, details


def test_single_field_render_is_chinese_readonly_and_exclusive(tmp_path):
    field, metadata, original, _ = _example(tmp_path)
    before = plot.sha256(field), plot.sha256(metadata)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        receipt = plot.write_plot(field, metadata, tmp_path / "new_plot")
    assert not any("Glyph" in str(item.message) for item in recorded)
    assert (plot.sha256(field), plot.sha256(metadata)) == before
    assert set(path.name for path in (tmp_path / "new_plot").iterdir()) == {"flow2d.png", "plot.json"}
    assert (tmp_path / "new_plot" / "flow2d.png").stat().st_size > 20000
    assert receipt["panels"] == 1
    assert receipt["plotted_observed_input_scales"] == [0, 1]
    assert receipt["final_scale_as_R_input"] is False
    assert receipt["model_inference_performed"] is False
    assert receipt["root_search_performed"] is False
    assert receipt["streamlines_are_discrete_rg_trajectories"] is False
    assert receipt["source_sha256"] == dict(zip(("field_npz", "metadata_json"), before))
    assert receipt["pc12_variance_fraction"] == pytest.approx(.9993)
    assert "未热化诊断：不是已认证的平衡态相图" in receipt["rendered_text"]
    assert receipt["output_sha256"]["flow2d.png"] == plot.sha256(tmp_path / "new_plot" / "flow2d.png")
    with np.load(field, allow_pickle=False) as loaded:
        for name, array in original.items():
            np.testing.assert_array_equal(loaded[name], array)
    with pytest.raises(FileExistsError):
        plot.write_plot(field, metadata, tmp_path / "new_plot")


@pytest.mark.parametrize("damage", ["sha", "nonfinite", "shape", "mask", "metadata", "spacing"])
def test_rejects_inconsistent_export_before_output(tmp_path, damage):
    field, metadata, values, details = _example(tmp_path)
    if damage == "nonfinite":
        values["projected_delta"][0, 0, 0] = np.nan
    elif damage == "shape":
        values["full_delta"] = values["full_delta"][..., :4]
    elif damage == "mask":
        values["support_mask"] = values["support_mask"].astype(int)
    elif damage == "spacing":
        values["grid_x"][1] += .001
    elif damage == "metadata":
        details["equilibrium_not_validated"] = False
    elif damage == "sha":
        details["field_npz_sha256"] = "0" * 64
    np.savez_compressed(field, **values)
    if damage != "sha":
        details["field_npz_sha256"] = plot.sha256(field)
    metadata.write_text(json.dumps(details, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError):
        plot.write_plot(field, metadata, tmp_path / "must_not_exist")
    assert not (tmp_path / "must_not_exist").exists()


@pytest.mark.parametrize("mask_value", [False, True])
def test_degenerate_support_masks_still_render_without_changing_vectors(tmp_path, mask_value):
    field, metadata, values, details = _example(tmp_path)
    values["support_mask"][:] = mask_value
    values["projected_delta"][:] = 0.
    values["full_delta"][:] = 0.
    np.savez_compressed(field, **values)
    details["field_npz_sha256"] = plot.sha256(field)
    metadata.write_text(json.dumps(details, ensure_ascii=False), encoding="utf-8")
    receipt = plot.write_plot(field, metadata, tmp_path / "zero")
    assert receipt["projected_support_grid_points"] == int(mask_value) * values["support_mask"].size
    assert receipt["color_range"][0] == 0.
