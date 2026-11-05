import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import compare_ising3d_no_rsmi as comparison
from scripts import report_ising3d as reporting


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def arrays():
    direct = np.array([[[-.3], [-.2], [-.1]], [[.3], [.2], [.1]]])
    result = dict(R_grid=np.linspace(-.3, .3, 11)[:, None])
    result["R_mapped"] = 2 * result["R_grid"]
    result["R_displacement"] = result["R_mapped"] - result["R_grid"]
    result["sample_partition"] = np.array([1, 0])
    for split in ("train", "test"):
        result[f"direct_{split}"] = direct.copy()
        result[f"R_g_{split}"] = 2 * direct
        result[f"rollout_{split}"] = direct.copy()
        result[f"K_{split}"] = np.array([.1, .3])
        result[f"condition_index_{split}"] = np.array([0, 1])
        result[f"motif_target_{split}"] = np.full((2, 3, 256), 1 / 256)
        result[f"motif_reconstruction_{split}"] = result[f"motif_target_{split}"].copy()
    return result


@pytest.mark.parametrize("weight,skip", [(0., True), (2., False)])
def test_policy_binds_three_saved_contracts(weight, skip):
    record = dict(rsmi_weight=weight)
    result = reporting.mi_diagnostic_policy(record, record, record, skip_mi_diagnostic=skip)
    assert result["computed"] is (not skip)
    assert result["enabled"] is (weight > 0)
    if skip:
        assert "不是实测为零" in result["reason"]


@pytest.mark.parametrize("weights,skip", [([0, 0, 0], False), ([2, 2, 2], True),
    ([0, 2, 0], True), ([None, 0, 0], True), ([float("nan"), 0, 0], True), ([True, 1, 1], False)])
def test_policy_rejects_ambiguous_or_wrong_objective(weights, skip):
    with pytest.raises(ValueError):
        reporting.mi_diagnostic_policy(*(dict(rsmi_weight=x) for x in weights), skip_mi_diagnostic=skip)


@pytest.mark.parametrize("weight,skip", [(0., True), (2., False)])
def test_report_skip_never_calls_mi_and_baseline_still_does(tmp_path, monkeypatch, weight, skip):
    cfg = reporting.RunConfig.load(Path(__file__).resolve().parents[1] / "configs/ising3d_b2_g1_seed91.json")
    write_json(tmp_path / "config.json", cfg.to_dict())
    record = dict(rsmi_weight=weight)
    write_json(tmp_path / "run/manifest.json", record)
    write_json(tmp_path / "run/summary.json", dict(contract=record, history=[]))
    write_json(tmp_path / "data/evaluation_metadata.json", dict(cache_sha256="data"))

    class Model(torch.nn.Module):
        def advance_theory(self, x):
            return 2 * x

    model = Model()
    loaded = SimpleNamespace(model=model, payload=dict(metadata=dict(data_sha256="data", **record), epoch=6000), config=cfg.to_dict())
    monkeypatch.setattr(reporting, "load_run", lambda *args, **kwargs: loaded)
    monkeypatch.setattr(reporting, "prepared_data", lambda *args: (SimpleNamespace(configurations=torch.empty((2, 64, 16, 16, 16, 1))), None))
    monkeypatch.setattr(reporting, "file_fingerprint", lambda *args: "data")
    monkeypatch.setattr(reporting, "_model_state_sha256", lambda *args: "model")
    monkeypatch.setattr(reporting, "collect", lambda *args: (arrays(), {"version": 1}))
    called = []
    monkeypatch.setattr(reporting, "mi_null_diagnostic", lambda *args: called.append("MI") or {"diagnostic": 1.})
    monkeypatch.setattr(reporting, "figures", lambda *args, **kwargs: called.append(kwargs["mi_enabled"]) or "font")
    result = reporting.report(tmp_path, skip_mi_diagnostic=skip)
    assert ("MI" in called) is (not skip)
    assert called[-1] is (weight > 0)
    assert (result["MI_shuffle_diagnostic"] is None) is skip
    assert result["model_state_unchanged"] is True
    assert result["thermal_nu"] is None


def test_disabled_mi_loss_panel_is_not_a_zero_curve(tmp_path, monkeypatch):
    import matplotlib.axes
    text = []
    original = matplotlib.axes.Axes.text

    def capture(self, x, y, message, *args, **kwargs):
        text.append(message)
        return original(self, x, y, message, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "text", capture)
    history = [dict(epoch=1, **{f"{split}_{metric}": None if metric == "rsmi" else .1
        for split in ("train", "test") for metric in
        ("separation", "rsmi", "kernel_noncollapse", "kernel_sign", "commutator", "rollout")})]
    reporting.figures(arrays(), history, tmp_path, mi_enabled=False)
    assert any("未计算 MI（不是实测零）" in value for value in text)
    assert (tmp_path / "loss.png").exists()


def make_public(root, *, weight):
    cfg = json.loads((Path(__file__).resolve().parents[1] / "configs/ising3d_b2_g1_seed91.json").read_text())
    manifest = dict(config=cfg, rsmi_weight=weight, data_sha256="same-data")
    for key in ("random_construction_module_sha256", "pre_affine_initialization_module_sha256", "training_start_module_sha256"):
        manifest[key] = dict(Q="Q", C="C", R="R")
    archive = arrays()
    (root / "report").mkdir(parents=True)
    np.savez_compressed(root / "report/encodings.npz", **archive)
    report = dict(status="complete", workflow_complete=True, model_state_unchanged=True, model_gradients_none=True,
        model_sha256=f"model-{weight}", actual_training_updates=6000, C_draws=5, samples=64,
        encoding_protocol={"partition": [1, 0]}, scale_shapes=[[16] * 3, [8] * 3, [4] * 3],
        closure=reporting.closure_metrics(archive),
        fixed_point_candidates={"candidates": [dict(g=0., candidate_nu=.95), dict(g=.2, candidate_nu=.64)], "thermal_nu": None},
        MI_shuffle_diagnostic=None if weight == 0 else {"real": [1.]},
        MI_diagnostic_policy=dict(computed=weight != 0),
        public_arrays_sha256=comparison.sha256(root / "report/encodings.npz"))
    write_json(root / "report/report.json", report)
    write_json(root / "run/manifest.json", manifest)
    write_json(root / "run/summary.json", dict(contract={"rsmi_weight": weight}, history=[]))
    return manifest, report


def test_compare_reads_only_public_artifacts_and_keeps_all_candidates(tmp_path):
    old, new, output = (tmp_path / name for name in ("old", "new", "comparison"))
    make_public(old, weight=2.)
    make_public(new, weight=0.)
    result = comparison.compare(old, new, output)
    assert result["model_or_data_loaded"] is False
    assert result["all_initial_module_SHA_match"] is True
    assert result["strict_cross_run_closure_ranking_valid"] is False
    assert result["selected_best_root"] is None
    assert len(result["runs"]["candidate"]["fixed_point_candidates"]["candidates"]) == 2
    assert "不能作为" in result["coordinate_warning"]
    assert result["public_arrays_sha256"] == comparison.sha256(output / "comparison_arrays.npz")
    with pytest.raises(FileExistsError):
        comparison.compare(old, new, output)


@pytest.mark.parametrize("failure", ["data", "mi", "array_sha", "objective", "initial"])
def test_compare_rejects_nonpaired_or_invalid_reports(tmp_path, failure):
    old, new = (tmp_path / name for name in ("old", "new"))
    make_public(old, weight=2.)
    manifest, report = make_public(new, weight=0.)
    if failure == "data":
        manifest["data_sha256"] = "different-data"
    elif failure == "mi":
        report["MI_diagnostic_policy"]["computed"] = True
    elif failure == "array_sha":
        report["public_arrays_sha256"] = "wrong"
    elif failure == "objective":
        manifest["config"]["training"]["seed"] += 1
    else:
        manifest["training_start_module_sha256"]["C"] = "different-C"
    write_json(new / "run/manifest.json", manifest)
    write_json(new / "report/report.json", report)
    with pytest.raises(ValueError):
        comparison.compare(old, new, tmp_path / "comparison")
