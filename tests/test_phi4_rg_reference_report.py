"""The reference report reads public saved diagnostics, never a model or fields."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import report_phi4_rg_reference as report


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _refresh_data(data):
    manifest = json.loads((data / "manifest.json").read_text())
    _write(data / "status.json", dict(schema=report.DATA_SCHEMA, sample_semantics=report.SAMPLE_SEMANTICS,
        status="complete", workflow_complete=True, returncode=0, error=None,
        requires_user_action=False, completed_conditions=13,
        manifest_sha256=report.generic.sha256(data / "manifest.json")))
    return manifest


def _refresh_run(run):
    _write(run / "status.json", dict(status="complete", recipe=report.generic.RECIPE,
        workflow_complete=True, returncode=0, error=None, requires_user_action=False,
        actual_steps=3000, latest_checkpoint_step=3000,
        summary_sha256=report.generic.sha256(run / "summary.json"),
        evaluation_arrays_sha256=report.generic.sha256(run / "evaluation_arrays.npz")))


def _fixture(root):
    data, run = root / "data", root / "run"
    data.mkdir()
    run.mkdir()
    options = dict(r_values=report.R_VALUES.tolist(), u_values=[1.], h=0., sizes=[36],
        dimension=2, chains=4, samples=128, test_every=3, test_offset=0, cluster_probability=.5)
    pins = {"sampler.py": "a" * 64}
    _write(data / "plan.json", dict(options=options, action=report.ACTION, boundary="periodic", source_pins=pins))
    shards = []
    for cell, r in enumerate(options["r_values"]):
        split = "test" if cell % 3 == 0 else "train"
        chains = []
        for chain in range(4):
            means = dict(action_density=.5 + .01 * cell, magnetization=(-1.) ** chain * .02,
                absolute_magnetization=.5 - .015 * cell, magnetization_squared=.4 - .01 * cell,
                magnetization_fourth=.25 - .005 * cell, mean_field_squared=.8 - .02 * cell,
                structure_factor_zero=36 ** 2 * (.4 - .01 * cell), structure_factor_minimum_momentum=2.)
            chains.append(dict(chain_index=chain, initial="hot" if chain < 2 else "cold_positive",
                production_acceptance=.8, saved_magnetization_sign_changes=10,
                warnings=["Saved ESS below 32"] if chain == 0 else [],
                diagnostics={name: dict(samples=128, mean=value, std=.1, ess=16. if chain == 0 else 64.,
                    lag1=.3, iat_saved_intervals=4.) for name, value in means.items()}))
        summary = dict(binder_1_minus_m4_over_3m2sq=1. - means["magnetization_fourth"] / (3 * means["magnetization_squared"] ** 2),
            susceptibility_connected=36 ** 2 * means["magnetization_squared"],
            structure_factor_zero=means["structure_factor_zero"], structure_factor_minimum_momentum=2.,
            hot_cold_comparison=dict(action_density=dict(descriptive_three_se_warning=True,
                hot_minus_cold=.1, between_chain_standard_error=.01)))
        relative = f"conditions/L36_c{cell:04d}/sidecar.json"
        _write(data / relative, dict(schema=report.DATA_SCHEMA, sample_semantics=report.SAMPLE_SEMANTICS,
            status="complete", cell_index=cell, size=36, dimension=2, r=r, u=1., h=0., split=split,
            independent_chains=4, samples_per_chain=128, cluster_probability=.5,
            source_pins=pins, symmetry_augmentation=False, artificial_global_sign_flips=0,
            production_adaptation=False, labels_used_in_training=False,
            data_sha256="unopened-field-archive", chains=chains, summary=summary))
        (data / relative).with_name("data.npz").write_bytes(b"must not be opened")
        shards.append(dict(cell_index=cell, size=36, split=split, sidecar=relative,
            sidecar_sha256=report.generic.sha256(data / relative), sha256="unopened-field-archive",
            path=str(Path(relative).with_name("data.npz"))))
    _write(data / "quality_summary.json", dict(schema=report.DATA_SCHEMA, status="complete",
        sample_semantics=report.SAMPLE_SEMANTICS, warning_count=26, conditions=[dict(warnings=["retained"])],
        equilibrium_certified=False))
    manifest = dict(schema=report.DATA_SCHEMA, status="complete", sample_semantics=report.SAMPLE_SEMANTICS,
        options=options, shards=shards, plan_sha256=report.generic.sha256(data / "plan.json"),
        source_pins=pins, source_pins_match=True, cluster_probability=.5,
        symmetry_augmentation=False, artificial_global_sign_flips=0, production_adaptation=False,
        quality_summary="quality_summary.json", quality_summary_sha256=report.generic.sha256(data / "quality_summary.json"),
        labels_used_in_training=False, warning_count=26)
    _write(data / "manifest.json", manifest)
    _refresh_data(data)
    arrays = {}
    for split in ("train", "test"):
        selected = [row for row in shards if row["split"] == split]
        r = np.repeat([options["r_values"][row["cell_index"]] for row in selected], 4)
        direct = r[:, None, None] + .1 * np.arange(3)[None, :, None]
        direct += np.tile([-.01, 0., .01, .02], len(selected))[:, None, None]
        arrays.update({f"{split}_direct": direct, f"{split}_one_step": direct[:, 1:] + .01,
            f"{split}_rollout": direct.copy(),
            f"{split}_labels": np.column_stack((r, np.ones(len(r)), np.zeros(len(r)))),
            f"{split}_cell_ids": np.repeat([row["cell_index"] for row in selected], 4)})
        arrays[f"{split}_rollout"][:, 1:] += .02
    grid = np.linspace(-2., 1., 50)
    arrays.update(flow_g=grid, flow_R=.7 * grid, flow_displacement=-.3 * grid)
    np.savez(run / "evaluation_arrays.npz", **arrays)
    data_sha = report.generic.sha256(data / "manifest.json")
    _write(run / "manifest.json", dict(schema_version=5, recipe=report.generic.RECIPE,
        options=dict(seed=91, steps=3000), data_manifest_sha256=data_sha, labels_used_in_training=False,
        lattice_sizes=[36, 12, 4], theory_dim=1, block_size=3, stride=3,
        closure_objective="all_start_anchor", closure_scale_weights=None, q_initialization="train_mean_bias"))
    _write(run / "summary.json", dict(status="complete", recipe=report.generic.RECIPE,
        actual_steps=3000, latest_checkpoint_step=3000, data_manifest_sha256=data_sha,
        lattice_sizes=[36, 12, 4], theory_dim=1, thermal_nu=None,
        optimizer_steps={name: dict(min=3000, max=3000, parameters=2) for name in ("Q", "C", "R")},
        numerical_fixed_points=[]))
    _write(run / "history_progress.json", dict(latest_step=3000, latest_checkpoint_step=3000,
        history=[dict(step=step, reconstruction=.1 / step, subset_micro=.2 / step,
            subset_coarse=.1 / step, information_proxy=.3, variance_floor=.1, uniform_amplitude=.1,
            closure_one_step=.1 / step, closure_rollout=.2 / step) for step in (1, 3000)]))
    _refresh_run(run)
    (run / "training_step_3000.pt").write_bytes(b"not a model")
    return data, run


def _change_sidecar(data, callback):
    manifest = json.loads((data / "manifest.json").read_text())
    shard = manifest["shards"][0]
    path = data / shard["sidecar"]
    sidecar = json.loads(path.read_text())
    callback(sidecar)
    _write(path, sidecar)
    shard["sidecar_sha256"] = report.generic.sha256(path)
    _write(data / "manifest.json", manifest)
    _refresh_data(data)


def test_same_action_reference_heldout_and_physics_from_sidecars(tmp_path):
    data, run = _fixture(tmp_path)
    _, manifest, _, sidecars, fingerprints, quality = report.load_public_data(data)
    public = report.generic.load_public_run(run)
    report.check_data_run_consistency(manifest, sidecars, public, fingerprints)
    summary = report.physics_summary(manifest, sidecars, public, quality)
    assert len(summary["rows"]) == 13
    assert summary["rows"][6]["split"] == "test"
    assert summary["rows"][6]["critical_reference_condition"] is True
    assert summary["reference"]["r_critical"] == pytest.approx(-1.2724682)
    assert summary["reference"]["r_uncertainty"] == pytest.approx(4.8e-6)
    assert summary["reference"]["RG_bare_mass_prediction"] is False
    assert summary["reference"]["finite_size_reference"] is False
    assert summary["sampling_diagnostics"]["preserved_warning_count"] == 26
    assert summary["sampling_diagnostics"]["warning_count_matches_manifest"] is True
    assert summary["rows"][0]["observable_means"]["magnetization"] == pytest.approx(0.)
    assert summary["rows"][0]["susceptibility_connected"] == pytest.approx(36 ** 2 * .4)
    assert summary["rows"][0]["susceptibility_absolute_subtraction_per_site"] == pytest.approx(36 ** 2 * (.4 - .5 ** 2))
    assert summary["susceptibility_definitions"]["absolute_subtraction_is_h0_linear_response"] is False
    assert summary["susceptibility_definitions"]["pseudocritical_fit_performed"] is False
    assert summary["rows"][0]["minimum_saved_ess"] == 16.
    assert summary["critical_r"] is summary["thermal_nu"] is None
    assert not summary["equilibrium_certified"]


def test_report_unchanged_public_safe_chinese_and_without_root_inference(tmp_path, monkeypatch):
    data, run = _fixture(tmp_path)
    before = {str(path): path.read_bytes() for root in (data, run) for path in root.rglob("*") if path.is_file()}
    original_load = np.load
    def guarded_load(path, *args, **kwargs):
        assert Path(path) == run / "evaluation_arrays.npz"
        assert kwargs["allow_pickle"] is False
        return original_load(path, *args, **kwargs)
    monkeypatch.setattr(np, "load", guarded_load)
    def forbidden(*args, **kwargs):
        pytest.fail("No root interpolation or generic root-writing report is permitted")
    monkeypatch.setattr(report.generic, "interpolated_flow_zeros", forbidden)
    monkeypatch.setattr(report.generic, "write_report", forbidden)
    import matplotlib.figure
    saved_text, original_save = [], matplotlib.figure.Figure.savefig
    def capture_save(figure, *args, **kwargs):
        from matplotlib.text import Text
        saved_text.extend(artist.get_text() for artist in figure.findobj(match=Text))
        return original_save(figure, *args, **kwargs)
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", capture_save)
    output = tmp_path / "report"
    provenance, summary = report.write_report(data, run, output)
    assert before == {str(path): path.read_bytes() for root in (data, run) for path in root.rglob("*") if path.is_file()}
    assert provenance["return_files"] == report.return_files(output) == provenance["public_files"]
    actual = {str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()}
    assert actual == set(provenance["return_files"])
    assert all(not Path(name).is_absolute() and Path(name).suffix in (".json", ".npz", ".png") for name in actual)
    assert all(report.generic.sha256(output / name) == digest for name, digest in provenance["outputs_sha256"].items())
    for name in ("model_loaded", "physical_data_loaded", "GPU_used", "inference_performed",
                 "root_search_performed", "saved_grid_zero_interpolation_performed"):
        assert provenance[name] is False
    assert "文献无限体积蒙特卡罗参考" in saved_text
    assert not any("Monte Carlo" in text for text in saved_text)
    with original_load(output / "physics_arrays.npz", allow_pickle=False) as arrays:
        assert arrays["g_mean_by_scale"].shape == (13, 3)
        assert arrays["critical_reference_condition"].sum() == 1
        assert arrays["split"][6] == "test"
        assert arrays["susceptibility_absolute_subtraction_per_site"][0] == pytest.approx(36 ** 2 * (.4 - .5 ** 2))
        assert arrays["mean_field_squared"][0] == pytest.approx(.8)
    assert summary["sampling_sidecars"][0]["chains"][0]["warnings"] == ["Saved ESS below 32"]
    with pytest.raises(FileExistsError, match="fresh"):
        report.write_report(data, run, output)


@pytest.mark.parametrize("field,value,message", [
    ("r_values", list(np.linspace(-1., 1., 13)), "reference-centered"),
    ("u_values", [.2], "u=1"), ("sizes", [108], "L=36"),
    ("test_offset", 1, "must be held out"),
])
def test_wrong_benchmark_or_center_not_held_out_rejected(tmp_path, field, value, message):
    data, _ = _fixture(tmp_path)
    manifest = json.loads((data / "manifest.json").read_text())
    plan = json.loads((data / "plan.json").read_text())
    manifest["options"][field] = value
    plan["options"][field] = value
    _write(data / "plan.json", plan)
    manifest["plan_sha256"] = report.generic.sha256(data / "plan.json")
    _write(data / "manifest.json", manifest)
    _refresh_data(data)
    with pytest.raises(ValueError, match=message):
        report.load_public_data(data)


def test_tampered_sidecar_and_escaping_json_rejected(tmp_path):
    data, _ = _fixture(tmp_path)
    manifest = json.loads((data / "manifest.json").read_text())
    path = data / manifest["shards"][0]["sidecar"]
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="Sidecar SHA256"):
        report.load_public_data(data)
    manifest["shards"][0]["sidecar"] = "../outside.json"
    _write(data / "manifest.json", manifest)
    _refresh_data(data)
    with pytest.raises(ValueError, match="in-root public JSON"):
        report.load_public_data(data)


@pytest.mark.parametrize("mutation,message", [
    (lambda row: row.__setitem__("r", 1.), "physical identity"),
    (lambda row: row["chains"][0]["diagnostics"].pop("mean_field_squared"), "inventory"),
    (lambda row: row["chains"][0]["diagnostics"]["action_density"].__setitem__("mean", float("nan")), "finite"),
    (lambda row: row["chains"][0].__setitem__("chain_index", 3), "chain inventory"),
])
def test_diagnostic_and_condition_integrity_rejected(tmp_path, mutation, message):
    data, _ = _fixture(tmp_path)
    _change_sidecar(data, mutation)
    with pytest.raises(ValueError, match=message):
        report.load_public_data(data)


def test_unavailable_ess_retained_as_null_not_fabricated(tmp_path):
    data, run = _fixture(tmp_path)
    _change_sidecar(data, lambda row: row["chains"][0]["diagnostics"]["action_density"].__setitem__("ess", None))
    _, manifest, _, sidecars, _, quality = report.load_public_data(data)
    summary = report.physics_summary(manifest, sidecars, report.generic.load_public_run(run), quality)
    assert summary["rows"][0]["unavailable_ess_count"] == 1
    assert summary["sampling_sidecars"][0]["chains"][0]["diagnostics"]["action_density"]["ess"] is None


def test_training_recipe_and_data_link_must_match(tmp_path):
    data, run = _fixture(tmp_path)
    _, manifest, _, sidecars, fingerprints, _ = report.load_public_data(data)
    public = report.generic.load_public_run(run)
    public.manifest["closure_objective"] = "one_step_and_rollout"
    with pytest.raises(ValueError, match="all-start"):
        report.check_data_run_consistency(manifest, sidecars, public, fingerprints)
    public.manifest["closure_objective"] = "all_start_anchor"
    public.summary["data_manifest_sha256"] = "wrong"
    with pytest.raises(ValueError, match="Run/data"):
        report.check_data_run_consistency(manifest, sidecars, public, fingerprints)
