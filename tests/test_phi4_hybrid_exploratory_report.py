"""Tiny public-only exploratory receipt tests, with source held gate retained."""
from __future__ import annotations

import json

import numpy as np
import pytest

from scripts import report_phi4_hybrid_exploratory as report


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _fixture(tmp_path):
    source = tmp_path / "held"
    data = source / "data"
    child = source / "exploratory_training_v1"
    child.mkdir(parents=True)
    pins = {name: ("a" if index == 0 else "b") * 64 for index, name in enumerate(report.MODEL_SOURCES)}
    _write(source / "plan.json", dict(source_sha256=pins))
    _write(source / "job.json", dict(status="sampling_quality_hold", workflow_complete=False,
        stages=[dict(name="physical_sampling", status="complete", returncode=0)]))
    manifest = dict(schema=report.DATA_SCHEMA, status="complete",
        options=dict(chains=4, samples=128), shards=[dict(cell_index=index) for index in range(12)])
    _write(data / "manifest.json", manifest)
    digest = report.sha256(data / "manifest.json")
    conditions, issues = [], []
    for cell in range(12):
        chains = []
        for chain in range(4):
            global_index = cell * 4 + chain
            ess = 9.25 if global_index == 0 else 20. if global_index < 7 else 64.
            chains.append(dict(chain_index=chain, diagnostics=dict(action_density=dict(ess=ess)),
                               warnings=["保存的低 ESS 证据"] if ess < 32 else []))
            if ess < 32:
                issues.append(dict(cell_index=cell, chain_index=chain,
                                   kind="existing_ess_below32_or_unavailable", ess=ess))
        flag = cell < 3
        if flag:
            issues.append(dict(cell_index=cell, kind="existing_hot_cold_three_se_warning"))
        conditions.append(dict(cell_index=cell, r=-.5 + .8 * cell / 11, chains=chains,
            summary=dict(hot_cold_comparison=dict(action_density=dict(descriptive_three_se_warning=flag,
                hot_minus_cold=.2, between_chain_standard_error=.01)))))
    quality = dict(status="sampling_quality_hold", issues=issues, conditions=conditions,
        data_manifest_sha256=digest, training_started=False, GPU_profile_started=False,
        equilibrium_certified=False, qualification="完整保留原门控诊断")
    _write(source / "sampling_quality.json", quality)
    authorization = dict(user_authorization="允许保留质量警告继续探索性训练", source_root=str(source),
        source_data_manifest_sha256=digest, source_job_sha256=report.sha256(source / "job.json"),
        source_sampling_quality_sha256=report.sha256(source / "sampling_quality.json"),
        source_quality_issue_count=len(issues), source_quality=quality, source_quality_passed=False)
    _write(child / "authorization.json", authorization)
    plan = dict(scientific_qualification=report.QUALIFICATION, source_root=str(source), data_root=str(data),
        source_data_manifest_sha256=digest, source_quality_status="sampling_quality_hold",
        source_validation=dict(metadata_sha256={"sampling_quality.json": report.sha256(source / "sampling_quality.json"),
            "data/manifest.json": digest}, shards_sha256={"declared-only": "c" * 64}),
        authorization_file="authorization.json", authorization_sha256=report.sha256(child / "authorization.json"),
        source_sha256=pins)
    _write(child / "plan.json", plan)
    _write(child / "job.json", dict(status="running", workflow_complete=False,
        current="exploratory_qualification_report", scientific_qualification=report.QUALIFICATION,
        source_root=str(source), source_quality_passed=False,
        stages=[dict(name=name, status="complete", returncode=0) for name in report.REQUIRED_STAGES]
            + [dict(name="exploratory_qualification_report", status="running")]))
    manifest_run = dict(recipe=report.TRAINING_RECIPE, options=dict(seed=91, steps=3000),
        initialization="random; no warm start or kernel selection", data_manifest_sha256=digest,
        labels_used_in_training=False, source_sha256={"/frozen/code/" + name: value for name, value in pins.items()})
    clocks = {name: dict(min=3000, max=3000, parameters=3) for name in ("Q", "C", "R")}
    summary = dict(recipe=report.TRAINING_RECIPE, actual_steps=3000, latest_checkpoint_step=3000,
        frozen_evaluation_verified=True, data_manifest_sha256=digest, labels_used_in_training=False,
        optimizer_steps=clocks)
    _write(child / "run" / "manifest.json", manifest_run)
    _write(child / "run" / "summary.json", summary)
    (child / "run" / "evaluation_arrays.npz").write_bytes(b"public-array-hash-only-no-load")
    _write(child / "run" / "status.json", dict(status="complete", workflow_complete=True,
        returncode=0, error=None, requires_user_action=False, actual_steps=3000, latest_checkpoint_step=3000,
        summary_sha256=report.sha256(child / "run" / "summary.json"),
        evaluation_arrays_sha256=report.sha256(child / "run" / "evaluation_arrays.npz")))
    run_hashes = {name: report.sha256(child / "run" / name)
                  for name in ("manifest.json", "summary.json", "status.json", "evaluation_arrays.npz")}
    for folder, payload_name, payload in (("report", "metrics.json", dict(actual_steps=3000, thermal_nu=None)),
            ("calibration_report", "calibration.json", dict(actual_steps=3000, thermal_nu=None,
                optimizer_steps=clocks, data_schema=report.DATA_SCHEMA))):
        directory = child / folder
        _write(directory / payload_name, payload)
        (directory / "existing.png").write_bytes(b"public-figure-hash-only-no-load")
        _write(directory / "status.json", dict(status="complete", workflow_complete=True,
            error=None, requires_user_action=False))
        _write(directory / "report.json", dict(status="complete", public_sources_unchanged=True,
            model_loaded=False, GPU_used=False, inference_performed=False, root_search_performed=False,
            sources_sha256_after=run_hashes if folder == "report" else dict(run=run_hashes, data={"manifest.json": digest}),
            outputs_sha256={name: report.sha256(directory / name) for name in (payload_name, "existing.png")}))
    (data / "configurations.npz").write_bytes(b"forbidden physical configs")
    (child / "run" / "training_step_3000.pt").write_bytes(b"forbidden checkpoint")
    return source, child


def _refresh_authorization_plan(child):
    value = json.loads((child / "plan.json").read_text())
    value["authorization_sha256"] = report.sha256(child / "authorization.json")
    _write(child / "plan.json", value)


def test_statistics_derived_not_hardcoded_actual_experiment_numbers(tmp_path):
    _, child = _fixture(tmp_path)
    evidence = report.load_evidence(child)
    stats = evidence["quality_statistics"]
    assert stats["gate_record_count"] == 10  # deliberately not the real diagnostic's 72
    assert stats["independent_chain_count"] == 48
    assert stats["chains_with_any_observable_ess_below32"] == 7
    assert stats["minimum_saved_single_chain_ess"] == 9.25
    assert stats["hot_cold_warning_condition_count"] == 3
    assert not stats["ess_or_quality_reestimated"]


def test_parent_running_final_stage_allowed_and_source_held_unchanged(tmp_path, monkeypatch):
    source, child = _fixture(tmp_path)
    evidence = report.load_evidence(child)
    before = evidence["fingerprints"]
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: pytest.fail("No arrays may be loaded or recomputed"))
    receipt = report.write_report(child)
    assert {path: report.sha256(path) for path in before} == before
    assert json.loads((source / "job.json").read_text())["status"] == "sampling_quality_hold"
    assert receipt["scientific_label"] == report.SCIENTIFIC_LABEL
    assert not receipt["source_quality_passed"]
    assert receipt["source_quality"] == receipt["user_authorization"]["source_quality"]
    assert receipt["quality_statistics"]["gate_record_count"] == 10
    assert receipt["actual_training_steps"] == 3000 and receipt["seed"] == 91
    assert receipt["source_saved_configuration_count"] == 6144
    assert receipt["thermal_nu"] is receipt["critical_r"] is None
    assert not any(receipt[key] for key in ("model_loaded", "sampling_executed", "training_executed",
        "evaluation_recomputed", "equilibrium_certified", "physical_payloads_rehashed"))
    assert report.sha256(child / "qualification_report" / "warning.png") == receipt["warning_png_sha256"]
    assert set(receipt["public_files"]) == {"exploratory_receipt.json", "warning.png", "status.json"}
    with pytest.raises(FileExistsError):
        report.write_report(child)


def test_authorization_file_hash_and_waiver_cannot_be_changed(tmp_path):
    _, child = _fixture(tmp_path)
    value = json.loads((child / "authorization.json").read_text())
    value["user_authorization"] = "没有授权"
    _write(child / "authorization.json", value)
    with pytest.raises(ValueError, match="Authorization SHA256"):
        report.load_evidence(child)
    _refresh_authorization_plan(child)
    with pytest.raises(ValueError, match="Explicit human authorization"):
        report.load_evidence(child)


def test_failed_quality_cannot_be_relabeled_passed(tmp_path):
    _, child = _fixture(tmp_path)
    value = json.loads((child / "job.json").read_text())
    value["source_quality_passed"] = True
    _write(child / "job.json", value)
    with pytest.raises(ValueError, match="retain the failed"):
        report.load_evidence(child)


def test_original_quality_sha_and_full_warning_evidence_immutable(tmp_path):
    source, child = _fixture(tmp_path)
    value = json.loads((source / "sampling_quality.json").read_text())
    value["issues"].pop()
    _write(source / "sampling_quality.json", value)
    with pytest.raises(ValueError, match="held source quality evidence"):
        report.load_evidence(child)


@pytest.mark.parametrize("stage", report.REQUIRED_STAGES)
def test_all_prior_stages_must_finish_with_zero_returncode(tmp_path, stage):
    _, child = _fixture(tmp_path)
    value = json.loads((child / "job.json").read_text())
    next(row for row in value["stages"] if row["name"] == stage)["returncode"] = 1
    _write(child / "job.json", value)
    with pytest.raises(ValueError, match="preceding stage"):
        report.load_evidence(child)


def test_source_6144_data_and_child_double_link_required(tmp_path):
    _, child = _fixture(tmp_path)
    value = json.loads((child / "run" / "manifest.json").read_text())
    value["data_manifest_sha256"] = "wrong"
    _write(child / "run" / "manifest.json", value)
    with pytest.raises(ValueError, match="run/data identity"):
        report.load_evidence(child)


def test_all_adam_parameters_reach3000_not_just_log_step(tmp_path):
    _, child = _fixture(tmp_path)
    value = json.loads((child / "run" / "summary.json").read_text())
    value["optimizer_steps"]["C"]["min"] = 2999
    _write(child / "run" / "summary.json", value)
    with pytest.raises(ValueError, match="Adam parameter clocks"):
        report.load_evidence(child)


def test_model_and_trainer_fingerprints_must_match_held_source(tmp_path):
    _, child = _fixture(tmp_path)
    value = json.loads((child / "plan.json").read_text())
    value["source_sha256"][report.MODEL_SOURCES[0]] = "changed"
    _write(child / "plan.json", value)
    with pytest.raises(ValueError, match="model/loss/trainer"):
        report.load_evidence(child)


@pytest.mark.parametrize("folder", ["report", "calibration_report"])
def test_completed_public_report_artifacts_sha_enforced(tmp_path, folder):
    _, child = _fixture(tmp_path)
    path = child / folder / "existing.png"
    path.write_bytes(b"modified public figure")
    with pytest.raises(ValueError, match="output SHA256"):
        report.load_evidence(child)


def test_qualification_cannot_overwrite_source_or_existing_output(tmp_path):
    source, child = _fixture(tmp_path)
    with pytest.raises(ValueError, match="contain or replace"):
        report.write_report(child, source)


def test_cli_only_dispatches_read_only_qualification(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(report, "write_report", lambda root, output, font=None:
        calls.append((root, output)) or {"status": "complete"})
    assert report.main(["--root", str(tmp_path), "--output", str(tmp_path / "qualification")]) == 0
    assert calls == [(tmp_path, tmp_path / "qualification")]

