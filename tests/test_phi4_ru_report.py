"""Synthetic saved-array fixtures; no sampling, training, model or inference."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import report_phi4_ru_qcr as report


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _refresh_data(data):
    _write(data / "status.json", dict(schema=report.DATA_SCHEMA, sample_semantics=report.SAMPLE_SEMANTICS,
        status="complete", workflow_complete=True, returncode=0, error=None, requires_user_action=False,
        completed_conditions=45, manifest_sha256=report.generic.sha256(data / "manifest.json")))


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
    pins = {"sampler.py": "a" * 64}
    options = dict(r_values=report.R_VALUES.tolist(), u_values=report.U_VALUES.tolist(), h=0.,
        dimension=2, sizes=[36], chains=4, samples=128, test_every=4, test_offset=1, cluster_probability=.5)
    _write(data / "plan.json", dict(options=options, action=report.ACTION, boundary="periodic", source_pins=pins))
    shards, condition_labels = [], []
    for cell in range(45):
        r, u = float(report.R_VALUES[cell // 5]), float(report.U_VALUES[cell % 5])
        condition_labels.append([r, u, 0.])
        split = "test" if cell % 4 == 1 else "train"
        means = dict(action_density=.5, magnetization=0., absolute_magnetization=.4,
            magnetization_squared=.2, magnetization_fourth=.06, mean_field_squared=.8,
            structure_factor_zero=36 ** 2 * .2, structure_factor_minimum_momentum=2.)
        chains = [dict(chain_index=index, initial="hot" if index < 2 else "cold_positive",
            production_acceptance=.9, saved_magnetization_sign_changes=12,
            warnings=["Saved ESS below 32"] if index == 0 else [],
            diagnostics={name: dict(samples=128, mean=value, std=.1, ess=24. if index == 0 else 64.,
                lag1=.2, iat_saved_intervals=4.) for name, value in means.items()}) for index in range(4)]
        relative = f"conditions/L36_c{cell:04d}/sidecar.json"
        _write(data / relative, dict(schema=report.DATA_SCHEMA, sample_semantics=report.SAMPLE_SEMANTICS,
            status="complete", cell_index=cell, size=36, dimension=2, r=r, u=u, h=0., split=split,
            independent_chains=4, samples_per_chain=128, cluster_probability=.5, source_pins=pins,
            symmetry_augmentation=False, artificial_global_sign_flips=0, production_adaptation=False,
            labels_used_in_training=False, data_sha256="unopened-field-archive", chains=chains,
            summary=dict(binder_1_minus_m4_over_3m2sq=.5, susceptibility_connected=259.2,
                hot_cold_comparison={})))
        (data / relative).with_name("data.npz").write_bytes(b"not a field archive; must never open")
        shards.append(dict(cell_index=cell, size=36, split=split, sidecar=relative,
            sidecar_sha256=report.generic.sha256(data / relative), sha256="unopened-field-archive",
            path=str(Path(relative).with_name("data.npz"))))
    _write(data / "quality_summary.json", dict(schema=report.DATA_SCHEMA, status="complete",
        sample_semantics=report.SAMPLE_SEMANTICS, equilibrium_certified=False, warning_count=45))
    _write(data / "manifest.json", dict(schema=report.DATA_SCHEMA, sample_semantics=report.SAMPLE_SEMANTICS,
        status="complete", options=options, shards=shards, source_pins=pins, source_pins_match=True,
        cluster_probability=.5, symmetry_augmentation=False, artificial_global_sign_flips=0,
        production_adaptation=False, labels_used_in_training=False, warning_count=45,
        plan_sha256=report.generic.sha256(data / "plan.json"), quality_summary="quality_summary.json",
        quality_summary_sha256=report.generic.sha256(data / "quality_summary.json")))
    wrapper_pins = dict(pins, **{"hybrid.py": "b" * 64, "base.py": "c" * 64, "ru.py": "d" * 64})
    _write(data / "wrapper_receipt.json", dict(schema_version=5, status="complete", workflow_complete=True,
        returncode=0, historical_kernel_unchanged=True, historical_work_cap_unchanged=True,
        source_pins_before=wrapper_pins, source_pins_after=wrapper_pins, labels_used_in_training=False))
    _refresh_data(data)
    arrays = {}
    for split in ("train", "test"):
        cells = np.array([cell for cell in range(45) if (cell % 4 == 1) == (split == "test")])
        ids = np.repeat(cells, 4)
        labels = np.asarray(condition_labels)[ids]
        direct = np.repeat(labels[:, None, :2], 3, axis=1)
        direct += np.array([0., .03, .07])[None, :, None]
        direct += np.tile([-.006, -.002, .002, .006], len(cells))[:, None, None]
        one = direct[:, 1:] + np.array([.01, -.01])
        rollout = direct.copy()
        rollout[:, 1:] += .02
        feature = np.repeat(direct[:, :, :1], 105, axis=-1) * .01
        arrays.update({f"{split}_direct": direct, f"{split}_direct_full": direct + .001,
            f"{split}_one_step": one, f"{split}_rollout": rollout,
            f"{split}_labels": labels, f"{split}_cell_ids": ids,
            f"{split}_feature_target": feature, f"{split}_feature_reconstruction": feature + .001,
            f"{split}_one_step_feature_prediction": feature[:, 1:] + .002,
            f"{split}_rollout_feature_prediction": feature + .003})
    ids = np.concatenate((arrays["train_cell_ids"], arrays["test_cell_ids"]))
    direct = np.concatenate((arrays["train_direct"], arrays["test_direct"]))
    one = np.concatenate((arrays["train_one_step"], arrays["test_one_step"]))
    means = np.stack([direct[ids == cell].mean(axis=0) for cell in range(45)])
    mean_R = np.stack([one[ids == cell].mean(axis=0) for cell in range(45)])
    R_of_mean = mean_R + .004
    arrays.update(grid_cell_ids=np.arange(45), grid_labels=np.asarray(condition_labels), grid_mean_direct=means,
        grid_R_of_mean=R_of_mean, grid_mean_R=mean_R, grid_jensen_difference=mean_R - R_of_mean,
        grid_displacement=R_of_mean - means[:, :2])
    inputs = arrays["train_direct"][:, :2].reshape(-1, 2)
    bound = np.stack([inputs.min(axis=0) - .01, inputs.max(axis=0) + .01], axis=-1)
    x, y = np.meshgrid(np.linspace(*bound[0], 33), np.linspace(*bound[1], 33))
    grid = np.column_stack((x.ravel(), y.ravel()))
    _, singular, components = np.linalg.svd(inputs - inputs.mean(axis=0), full_matrices=False)
    variance = singular ** 2
    jacobian = np.diag([1.2, .8])
    mapped = grid @ jacobian.T
    arrays.update(flow_g=grid, flow_R=mapped, flow_displacement=mapped - grid,
        flow_DR=np.broadcast_to(jacobian, (1089, 2, 2)), flow_eigenvalues_real=np.tile([1.2, .8], (1089, 1)),
        flow_eigenvalues_imag=np.zeros((1089, 2)), flow_in_train_hull=np.all((grid >= inputs.min(axis=0)) & (grid <= inputs.max(axis=0)), axis=-1),
        flow_bounds=bound, flow_train_mean=inputs.mean(axis=0), flow_pca_components=components,
        flow_pca_explained_variance_ratio=variance / variance.sum())
    np.savez(run / "evaluation_arrays.npz", **arrays)
    data_sha = report.generic.sha256(data / "manifest.json")
    _write(run / "manifest.json", dict(schema_version=5, recipe=report.generic.RECIPE,
        options=dict(seed=91, steps=3000), data_manifest_sha256=data_sha, labels_used_in_training=False,
        lattice_sizes=[36, 12, 4], theory_dim=2, hidden=96, dtype="float64", block_size=3, stride=3,
        closure_objective="all_start_anchor", closure_scale_weights=None, q_initialization="train_mean_bias",
        scale=1.3, learning_rate=.001, learning_rate_policy="constant", model_input_members=["configurations"]))
    _write(run / "summary.json", dict(status="complete", recipe=report.generic.RECIPE,
        actual_steps=3000, latest_checkpoint_step=3000, data_manifest_sha256=data_sha,
        lattice_sizes=[36, 12, 4], theory_dim=2, scale=1.3, thermal_nu=None,
        frozen_evaluation_verified=True, labels_used_in_training=False,
        optimizer_steps={name: dict(min=3000, max=3000, parameters=2) for name in ("Q", "C", "R")},
        numerical_fixed_points=[dict(g=[0., 0.], full_DR=jacobian.tolist(),
            eigenvalues=[dict(real=1.2, imag=0., modulus=1.2), dict(real=.8, imag=0., modulus=.8)],
            classification="saddle", within_train_hull=False, local_rank=2,
            nearest_train_distance=.7, residual=0., thermal_nu=None)], flow_analysis=dict(root_search_performed=True)))
    _write(run / "history_progress.json", dict(latest_step=3000, latest_checkpoint_step=3000,
        history=[dict(step=step, reconstruction=.1 / step, subset_micro=.2 / step, subset_coarse=.1 / step,
            information_proxy=.3, variance_floor=.1, uniform_amplitude=.1,
            closure_one_step=.1 / step, closure_rollout=.2 / step) for step in (1, 3000)]))
    _refresh_run(run)
    (run / "training_step_3000.pt").write_bytes(b"not a model; must never open")
    return data, run


def _change_json(path, callback):
    value = json.loads(path.read_text())
    callback(value)
    _write(path, value)


def test_exact_grid_endpoint_metrics_and_train_only_pca(tmp_path):
    data, run = _fixture(tmp_path)
    _, manifest, _, sidecars, fingerprints, quality = report.load_public_data(data)
    public = report.load_public_run(run)
    report.check_data_run_consistency(manifest, sidecars, public, fingerprints)
    assert len(sidecars) == 45
    assert sum(row["split"] == "test" for row in sidecars) == 11
    rows, projections = report.pca_diagnostics(public.arrays)
    expected = public.arrays["train_direct"][:, :2].reshape(-1, 2).mean(axis=0)
    np.testing.assert_allclose(rows[-1]["mean"], expected)
    assert projections["pca_all_inputs_test_projection"].shape == (88, 2)
    assert rows[-1]["relative_1percent_rank"] == 2
    summary = report.physics_summary(manifest, sidecars, quality)
    assert summary["sampling_diagnostics"]["preserved_warning_count"] == 45
    assert summary["absolute_subtraction_is_h0_linear_response"] is False
    assert summary["rows"][0]["susceptibility_absolute_subtraction_per_site"] == pytest.approx(36 ** 2 * (.2 - .4 ** 2))


def test_report_chinese_safe_saved_roots_only_and_read_only(tmp_path, monkeypatch):
    data, run = _fixture(tmp_path)
    before = {str(path): path.read_bytes() for source in (data, run) for path in source.rglob("*") if path.is_file()}
    original = np.load
    def guarded(path, *args, **kwargs):
        assert Path(path) == run / "evaluation_arrays.npz"
        assert kwargs["allow_pickle"] is False
        return original(path, *args, **kwargs)
    monkeypatch.setattr(np, "load", guarded)
    def forbidden(*args, **kwargs):
        pytest.fail("No interpolation or generic root-producing report permitted")
    monkeypatch.setattr(report.generic, "interpolated_flow_zeros", forbidden)
    monkeypatch.setattr(report.generic, "load_public_run", forbidden)
    import matplotlib.figure
    original_save, saved_text = matplotlib.figure.Figure.savefig, []
    def capture(figure, *args, **kwargs):
        from matplotlib.text import Text
        saved_text.extend(artist.get_text() for artist in figure.findobj(match=Text))
        return original_save(figure, *args, **kwargs)
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", capture)
    output = tmp_path / "report"
    provenance, metrics = report.write_report(data, run, output)
    assert before == {str(path): path.read_bytes() for source in (data, run) for path in source.rglob("*") if path.is_file()}
    assert set(report.return_files(output)) == set(report.PUBLIC_FILES)
    assert all(report.generic.sha256(output / name) == digest for name, digest in provenance["outputs_sha256"].items())
    assert metrics["saved_numerical_roots"][0]["g"] == [0., 0.]
    assert metrics["saved_numerical_roots"][0]["within_train_hull"] is False
    assert metrics["thermal_nu"] is None
    for key in ("model_loaded", "physical_data_loaded", "GPU_used", "inference_performed", "root_search_performed", "saved_grid_zero_interpolation_performed"):
        assert provenance[key] is False
    assert "文献无限体积蒙特卡罗参考" in saved_text
    assert any("真实逐链输入" in value for value in saved_text)
    assert not any("Monte Carlo" in value for value in saved_text)
    with pytest.raises(FileExistsError, match="fresh"):
        report.write_report(data, run, output)


@pytest.mark.parametrize("mutation", [
    lambda m: m.update(theory_dim=1), lambda m: m.update(block_size=2),
    lambda m: m.update(dtype="float32"), lambda m: m.update(hidden=12),
    lambda m: m.update(labels_used_in_training=True), lambda m: m.update(closure_scale_weights=[1.8, .2]),
    lambda m: m.update(learning_rate=.002), lambda m: m.update(model_input_members=["configurations", "u"]),
])
def test_wrong_recipe_and_supervision_rejected(tmp_path, mutation):
    _, run = _fixture(tmp_path)
    _change_json(run / "manifest.json", mutation)
    with pytest.raises(ValueError):
        report.load_public_run(run)


def test_tamper_missing_and_incomplete_rejected(tmp_path):
    data, run = _fixture(tmp_path)
    with (data / "conditions/L36_c0000/sidecar.json").open("a") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="SHA256"):
        report.load_public_data(data)
    with (run / "evaluation_arrays.npz").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="SHA256"):
        report.load_public_run(run)
    (run / "history_progress.json").unlink()
    with pytest.raises(FileNotFoundError):
        report.load_public_run(run)


@pytest.mark.parametrize("relative", ["checkpoint.pt", "../../outside.json", "/tmp/outside.json"])
def test_manifest_cannot_request_pt_or_outside_data(tmp_path, relative):
    data, _ = _fixture(tmp_path)
    _change_json(data / "manifest.json", lambda m: m["shards"][0].update(sidecar=relative))
    _refresh_data(data)
    with pytest.raises(ValueError, match="JSON"):
        report.load_public_data(data)


def test_output_overlap_and_changed_run_data_rejected(tmp_path):
    data, run = _fixture(tmp_path)
    for output in (tmp_path, run, data, data / "report", run / "report"):
        with pytest.raises(ValueError, match="overlap"):
            report.write_report(data, run, output)
    _change_json(run / "manifest.json", lambda m: m.update(data_manifest_sha256="wrong"))
    with pytest.raises(ValueError, match="manifest SHA256"):
        report.write_report(data, run, tmp_path / "report")


def test_wrong_full_jacobian_spectrum_and_flow_displacement_rejected(tmp_path):
    _, run = _fixture(tmp_path)
    _change_json(run / "summary.json", lambda m: m["numerical_fixed_points"][0].update(full_DR=[[.2, 0.], [0., -.2]]))
    _refresh_run(run)
    with pytest.raises(ValueError, match="full DR"):
        report.load_public_run(run)


@pytest.mark.parametrize("mutation,message", [
    (lambda a: a["flow_displacement"].__iadd__(.1), "R\\(g\\)-g"),
    (lambda a: a["flow_pca_explained_variance_ratio"].__setitem__(0, .99), "PCA"),
    (lambda a: (a["flow_g"].__setitem__((0, 0), -100.),
                a["flow_R"].__setitem__((0, 0), -100. + a["flow_displacement"][0, 0])), "33x33"),
    (lambda a: a["flow_eigenvalues_real"].__iadd__(1.), "full DR"),
    (lambda a: a.update(configurations=np.zeros((1, 1, 1, 1))), "public evaluation contract"),
    (lambda a: a.pop("test_feature_target"), "Both splits"),
])
def test_saved_array_semantics_and_payload_allowlist(tmp_path, mutation, message):
    _, run = _fixture(tmp_path)
    with np.load(run / "evaluation_arrays.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    mutation(arrays)
    np.savez(run / "evaluation_arrays.npz", **arrays)
    _refresh_run(run)
    with pytest.raises(ValueError, match=message):
        report.load_public_run(run)


@pytest.mark.parametrize("jacobian,eigenvalues", [
    ([[-1.2, 0.], [0., .8]], [(-1.2, 0.), (.8, 0.)]),
    ([[.8, -.4], [.4, .8]], [(.8, .4), (.8, -.4)]),
])
def test_full_negative_and_complex_saved_spectra_preserved(tmp_path, jacobian, eigenvalues):
    _, run = _fixture(tmp_path)
    def change(summary):
        root = summary["numerical_fixed_points"][0]
        root.update(full_DR=jacobian, eigenvalues=[dict(real=real, imag=imag, modulus=float(np.hypot(real, imag)))
                                                for real, imag in eigenvalues])
    _change_json(run / "summary.json", change)
    _refresh_run(run)
    public = report.load_public_run(run)
    assert [(row["real"], row["imag"]) for row in public.summary["numerical_fixed_points"][0]["eigenvalues"]] == eigenvalues


def test_wrapper_freeze_receipt_required(tmp_path):
    data, _ = _fixture(tmp_path)
    _change_json(data / "wrapper_receipt.json", lambda m: m["source_pins_after"].update(**{"ru.py": "changed"}))
    with pytest.raises(ValueError, match="wrapper frozen source"):
        report.load_public_data(data)


def test_no_label_fit_or_test_pca_refit(tmp_path):
    _, run = _fixture(tmp_path)
    public = report.load_public_run(run)
    rows, projections = report.pca_diagnostics(public.arrays)
    public.arrays["test_direct"] += 1000.
    changed, new_projections = report.pca_diagnostics(public.arrays)
    assert rows == changed
    np.testing.assert_array_equal(projections["pca_all_inputs_train_projection"], new_projections["pca_all_inputs_train_projection"])
    assert not np.array_equal(projections["pca_all_inputs_test_projection"], new_projections["pca_all_inputs_test_projection"])
