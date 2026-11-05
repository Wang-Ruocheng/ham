"""Tiny public-array fixtures; no real training/model/configuration loading."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import report_phi4_qcr_mean_init_comparison as report
from scripts.report_phi4_qcr import RECIPE


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _refresh(root):
    status = json.loads((root / "status.json").read_text())
    status.update(summary_sha256=report.sha256(root / "summary.json"),
                  evaluation_arrays_sha256=report.sha256(root / "evaluation_arrays.npz"))
    _write(root / "status.json", status)


def _fixture(root, *, new=False):
    root.mkdir()
    rng, arrays = np.random.default_rng(24), {}
    for split, r, cells in (("train", [-.5, -.5, .3, .3], ["a", "a", "b", "b"]),
                            ("test", [-.1, -.1], ["c", "c"])):
        direct = rng.normal(size=(len(r), 3, 1))
        if new:
            direct = direct * 2 + 7
        target = rng.uniform(.001, .03, size=(len(r), 3, 105))
        arrays.update({f"{split}_direct": direct, f"{split}_direct_full": direct + .001,
            f"{split}_one_step": direct[:, 1:] + (.1 if new else .2),
            f"{split}_rollout": np.concatenate((direct[:, :1], direct[:, 1:] + (.15 if new else .3)), axis=1),
            f"{split}_labels": np.column_stack((r, np.full(len(r), .2), np.zeros(len(r)))),
            f"{split}_cell_ids": np.array(cells),
            f"{split}_feature_target": target,
            f"{split}_feature_reconstruction": target + (.001 if new else .003),
            f"{split}_one_step_feature_prediction": target[:, 1:] + (.002 if new else .004),
            f"{split}_rollout_feature_prediction": target + (.003 if new else .005)})
    g = np.linspace(-1, 1, 11) + (7 if new else 0)
    drift = -.1 * (g - (7 if new else 0))
    arrays.update(flow_g=g, flow_R=g + drift, flow_displacement=drift)
    np.savez_compressed(root / "evaluation_arrays.npz", **arrays)
    manifest = dict(schema_version=5, recipe=RECIPE,
        options=dict(steps=3000, system_batch=8, sample_batch=64, seed=91, threads=1,
                     log_every=100, checkpoint_every=1000, information_pairs=128),
        data_manifest_sha256="d" * 64, data_shards=[dict(path="a/data.npz", sha256="a" * 64)],
        theory_dim=1, hidden=96, lattice_sizes=[36, 12, 4], block_size=3, stride=3, dtype="float64",
        scale=.8, scale_definition="TRAIN RMS", learning_rate=.001, learning_rate_policy="constant",
        optimizer="three independent fresh Adam", closure_objective="all_start_anchor", closure_target_scales=2,
        closure_implementation="same", closure_scale_weights=None, closure_branch_normalization="uniform",
        gradient_routing="independent", labels_used_in_training=False, model_input_members=["configurations"],
        evaluation_label_columns=["r", "u", "h"], evaluation_seed=20261022,
        evaluation_encoding="mean-half", evaluation_sample_count=128, training_sample_count=64,
        training_encoding="mean-half", data_warning_count=3, sampler_schema="hybrid",
        sample_semantics="independent_correlated_hybrid_markov_chains",
        initial_state_sha256="initial", source_sha256={"/a/recurrent_rg/phi4_qcr.py": "model-hash",
                                                    "/a/experiments/phi4_qcr.py": "new-trainer" if new else "old-trainer"})
    summary = dict(status="complete", recipe=RECIPE, actual_steps=3000, latest_checkpoint_step=3000,
        theory_dim=1, lattice_sizes=[36, 12, 4], initial_state_sha256="initial", final_state_sha256="final",
        optimizer_steps={name: dict(min=3000, max=3000, parameters=3) for name in ("Q", "C", "R")},
        numerical_fixed_points=[dict(g=7 if new else 0, R_derivative=.9, classification="attracting",
                                    thermal_nu=None, physical_critical_point_certified=False)], thermal_nu=None)
    if new:
        before, after = dict(Q="Q0", C="C0", R="R0"), dict(Q="Q1", C="C0", R="R0")
        protected = dict(Q_encoder="encoder", Q_decoder_weight="direction")
        receipt = dict(mode="train_mean_bias", constructor_state_sha256="initial", constructor_module_sha256=before,
            training_start_state_sha256="post-mean", training_start_module_sha256=after,
            constructor_protected_sha256=protected, training_start_protected_sha256=protected,
            labels_used=False, test_data_used=False, pca_initialization=False, warm_start=False, kernel_selection=False,
            optimizer_updates=dict(Q=0, C=0, R=0), global_cpu_rng_unchanged=True, global_cuda_rng_unchanged=True,
            all_parameter_gradients_none=True, C_trainable=True, training_batch_rng_created_after_initialization=True,
            training_field_rng_created_after_initialization=True, generator_seed=9091, scales=[0, 1, 2],
            scale_weighting="uniform", physical_ensemble_weighting="uniform", physical_ensemble_count=4,
            fields_per_ensemble=128, total_fields=512, mean=[.01] * 105, bias_after_sha256="mean", mean_sha256="mean")
        common = dict(q_initialization="train_mean_bias", q_initialization_receipt=receipt,
            random_construction_state_sha256="initial", random_construction_module_sha256=before,
            training_start_state_sha256="post-mean", training_start_module_sha256=after, initial_state_sha256="post-mean")
        manifest.update(common)
        summary.update(common)
    history = dict(latest_step=3000, latest_checkpoint_step=3000,
        history=[dict(step=step, q_loss=.1 / step, c_loss=-.1, r_loss=.01 / step,
                      reconstruction=.01 / step, subset_micro=.001, subset_coarse=.0001,
                      information_proxy=.1, variance_floor=.01, uniform_amplitude=.1,
                      closure_one_step=.01 / step, closure_rollout=.02 / step) for step in (1, 3000)])
    _write(root / "manifest.json", manifest)
    _write(root / "summary.json", summary)
    _write(root / "history_progress.json", history)
    _write(root / "status.json", dict(status="complete", workflow_complete=True, returncode=0, error=None,
        requires_user_action=False, recipe=RECIPE, actual_steps=3000, latest_checkpoint_step=3000))
    _refresh(root)
    return arrays


def _pair(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    _fixture(old)
    _fixture(new, new=True)
    return old, new


def _mutate_json(root, file, mutate):
    path = root / file
    value = json.loads(path.read_text())
    mutate(value)
    _write(path, value)
    _refresh(root)


def test_full_public_report_source_unchanged_no_checkpoint_reads(tmp_path):
    old, new = _pair(tmp_path)
    for root in (old, new):
        (root / "training_step_3000.pt").write_bytes(b"deliberately not a checkpoint")
    before = {str(path): report.sha256(path) for root in (old, new) for path in root.iterdir()}
    quality = tmp_path / "quality.json"
    _write(quality, dict(warning_count=3, conditions=[dict(cell_index=0, hot_cold_warning_observables=["energy"],
        chain_warnings=[dict(chain_index=0, warnings=["x: ESS below 32", "y: ESS below 32"]),
                        dict(chain_index=1, warnings=[])])]))
    provenance, comparison = report.write_report(old, new, tmp_path / "comparison", sampling_quality=quality)
    assert {str(path): report.sha256(path) for root in (old, new) for path in root.iterdir()} == before
    assert provenance["model_loaded"] is provenance["physical_data_loaded"] is provenance["GPU_used"] is False
    assert provenance["root_search_performed"] is provenance["MI_reestimated"] is False
    assert comparison["critical_r"] is comparison["thermal_nu"] is None
    assert comparison["latent_coordinates_identified_across_runs"] is False
    assert comparison["sampling_warnings"]["unique_low_ess_chains"] == 1
    assert comparison["sampling_warnings"]["unique_hot_cold_warning_conditions"] == 1
    assert comparison["native_target_pairing"]["all_saved_native_targets_equal"] is True
    for row in comparison["native_target_pairing"]["reconstruction_pairing_on_matching_targets"]["test"]:
        assert row["mean_new_minus_old_mse"] == pytest.approx(.001**2 - .003**2)
    expected = {"Q_reconstruction_comparison.png", "R_vs_g_comparison.png", "coupling_by_scale_comparison.png",
                "loss_curves_comparison.png", "comparison.json", "comparison_arrays.npz", "report.json", "status.json"}
    assert set(provenance["public_files"]) == expected
    for name, digest in provenance["outputs_sha256"].items():
        assert report.sha256(tmp_path / "comparison" / name) == digest
    with pytest.raises(FileExistsError):
        report.write_report(old, new, tmp_path / "comparison")


def test_latent_normalization_uses_each_runs_own_train_std(tmp_path):
    old, new = _pair(tmp_path)
    a, b = report.load_public_run(old), report.load_public_run(new)
    report.validate_pair(a, b)
    ma, mb = report._metrics(a), report._metrics(b)
    ra, rb = ma["latent_closure"]["rows"][0], mb["latent_closure"]["rows"][0]
    assert rb["model"]["train_target_population_std"] == pytest.approx(2 * ra["model"]["train_target_population_std"])
    assert rb["model"]["fixed_train_nrmse"] == pytest.approx(ra["model"]["fixed_train_nrmse"] / 4)


@pytest.mark.parametrize("key,value", [("dtype", "float32"), ("data_manifest_sha256", "different"),
    ("closure_scale_weights", [1.8, .2]), ("random_construction_state_sha256", "different"),
    ("gradient_routing", "joint"), ("theory_dim", 2)])
def test_pair_mismatch_rejected(tmp_path, key, value):
    old, new = _pair(tmp_path)
    _mutate_json(new, "manifest.json", lambda value_: value_.__setitem__(key, value))
    with pytest.raises(ValueError):
        report.validate_pair(report.load_public_run(old), report.load_public_run(new))


@pytest.mark.parametrize("key,value", [("labels_used", True), ("test_data_used", True),
    ("pca_initialization", True), ("global_cpu_rng_unchanged", False),
    ("physical_ensemble_count", 3), ("generator_seed", 91),
    ("optimizer_updates", {"Q": 1, "C": 0, "R": 0})])
def test_bad_initialization_receipts_rejected(tmp_path, key, value):
    old, new = _pair(tmp_path)
    for file in ("manifest.json", "summary.json"):
        _mutate_json(new, file, lambda obj: obj["q_initialization_receipt"].__setitem__(key, value))
    with pytest.raises(ValueError):
        report.validate_pair(report.load_public_run(old), report.load_public_run(new))


def test_changed_micro_targets_rejected_coarse_change_recorded(tmp_path):
    old, new = _pair(tmp_path)
    a, b = report.load_public_run(old), report.load_public_run(new)
    b.arrays["test_feature_target"][:, 1] += .002
    paired = report._paired_native(a, b)
    assert not paired["all_saved_native_targets_equal"]
    assert [row["scale"] for row in paired["reconstruction_pairing_on_matching_targets"]["test"]] == [0, 2]
    b.arrays["train_feature_target"][:, 0] += .01
    with pytest.raises(ValueError, match="Microscopic"):
        report._paired_native(a, b)


def test_only_formally_saved_roots_forwarded_no_interpolation(tmp_path):
    old, new = _pair(tmp_path)
    for root in (old, new):
        _mutate_json(root, "summary.json", lambda obj: obj.__setitem__("numerical_fixed_points", []))
    # Saved flow crosses zero, but report must not invent roots or new interpolation.
    assert report._metrics(report.load_public_run(new))["saved_numerical_roots"] == []


def test_incomplete_adam_and_wrong_architecture_pin_rejected(tmp_path):
    old, new = _pair(tmp_path)
    _mutate_json(new, "summary.json", lambda obj: obj["optimizer_steps"]["R"].__setitem__("max", 2999))
    with pytest.raises(ValueError, match="Adam"):
        report.load_public_run(new)
    _mutate_json(new, "summary.json", lambda obj: obj["optimizer_steps"]["R"].__setitem__("max", 3000))
    _mutate_json(new, "manifest.json", lambda obj: obj["source_sha256"].__setitem__("/a/recurrent_rg/phi4_qcr.py", "changed"))
    with pytest.raises(ValueError, match="model source"):
        report.validate_pair(report.load_public_run(old), report.load_public_run(new))


def test_sampling_quality_gate_issues_deduplicated(tmp_path):
    quality = tmp_path / "quality.json"
    _write(quality, dict(issues=[dict(cell_index=2, chain_index=1, kind="existing_ess_below32_or_unavailable", observable=name)
                                for name in ("m2", "m4")] + [dict(cell_index=1, kind="existing_hot_cold_three_se_warning")],
                        conditions=[dict(chains=[{}, {}, {}, {}])], source_warning_count=3))
    result, source = report.sampling_warnings(quality, [{"data_warning_count": 3}] * 2)
    assert result["unique_low_ess_chains"] == result["unique_hot_cold_warning_conditions"] == 1
    assert result["total_physical_chains"] == 4
    assert source[1] == report.sha256(quality)
