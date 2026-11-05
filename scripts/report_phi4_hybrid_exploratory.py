"""Read-only qualification receipt for user-authorized exploratory QCR training.

python -m scripts.report_phi4_hybrid_exploratory --root CHILD --output NEW_DIR

Read completed public child reports and source quality JSON, retain all warning
evidence, and never sample, load a model, recompute evaluation, or alter the held
source experiment. A running parent job is expected at this final child stage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.report_phi4_qcr import _plotting, sha256


QUALIFICATION = "exploratory_all_sampling_warnings_retained"
SCIENTIFIC_LABEL = "保留未通过采样质量门控的警告之探索性训练"
DATA_SCHEMA = "euclidean_phi4_hmc_embedded_wolff_v1"
TRAINING_RECIPE = "phi4_continuous_gaussian_soft_statistics_v1"
REQUIRED_STAGES = ("training_preflight", "training", "report", "phi4_hybrid_report")
MODEL_SOURCES = ("recurrent_rg/phi4_qcr.py", "experiments/phi4_qcr.py")
LIMITATIONS = [
    SCIENTIFIC_LABEL + "；用户允许继续不等于原采样质量门控通过。",
    "完整保留原门控记录、独立链诊断、热冷初态差异及源元数据 SHA；不删链、重采、换种子或修改采样参数。",
    "门控记录可能对同一证据重复计数，记录条数不是独立失败链数或独立统计证据数。",
    "ESS 与热冷比较仅为有限链估计；频繁符号反转、训练完成或 loss 降低不认证平衡。",
    "沿用连续场概率高斯 C、软局部统计与平滑信息代理，不是二值精确经验直方图 MI；本次不修复既有代理偏差。",
    "新训练从随机初始化开始，seed91，Q/C/R 各 fresh Adam 固定3000步；未加载源实验 checkpoint。",
    "相同模型/loss 的声明由冻结源与训练清单的共同源码 SHA 关联，不在本报告重新加载模型或重跑评价。",
    "R(g)-g、耦合映射和损失图仅在这些采样限制下解释；不认证临界位置、热模或临界指数。",
    "本报告不采样、不训练、不推理、不重算评价或物理统计；thermal_nu 始终为 null。",
]


def _public_path(root, relative):
    rel = Path(relative)
    path = (root / rel).resolve()
    if rel.is_absolute() or not path.is_relative_to(root) or path.suffix not in (".json", ".png", ".npz"):
        raise ValueError("Only explicitly referenced in-root public JSON/PNG/NPZ may be read")
    return path


def _read_json(path, fingerprints):
    path = Path(path).resolve()
    digest = sha256(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or sha256(path) != digest:
        raise ValueError("Public JSON changed during reading or is not an object")
    fingerprints[str(path)] = digest
    return value


def _record_hash(path, fingerprints):
    path = Path(path).resolve()
    value = sha256(path)
    fingerprints[str(path)] = value
    return value


def _complete(status, label):
    if (status.get("status") != "complete" or status.get("workflow_complete") is not True
            or status.get("error") is not None or status.get("requires_user_action", False)
            or status.get("returncode", 0) != 0):
        raise ValueError(f"{label} execution is not successfully complete")


def _optimizer_clock(value):
    if set(value) != {"Q", "C", "R"} or any(row.get("min") != 3000 or row.get("max") != 3000
            or row.get("parameters", 0) <= 0 for row in value.values()):
        raise ValueError("All Q/C/R Adam parameter clocks must reach exactly 3000")


def quality_statistics(quality):
    """Only summarize saved diagnostics, never estimate new ESS or Rhat."""
    low, unavailable, all_ess, identities, hot_cold = set(), set(), [], set(), []
    for condition in quality.get("conditions", []):
        cell = condition["cell_index"]
        comparisons = condition.get("summary", {}).get("hot_cold_comparison", {})
        flagged = {name: value for name, value in comparisons.items() if value.get("descriptive_three_se_warning")}
        if flagged:
            hot_cold.append(dict(cell_index=cell, r=condition.get("r"), comparisons=flagged))
        for chain in condition.get("chains", []):
            identity = (cell, chain["chain_index"])
            if identity in identities:
                raise ValueError("Duplicate chain identities in the preserved source quality evidence")
            identities.add(identity)
            for diagnostic in chain.get("diagnostics", {}).values():
                value = diagnostic.get("ess")
                if isinstance(value, (int, float)) and np.isfinite(value):
                    all_ess.append(float(value))
                    if value < 32:
                        low.add(identity)
                else:
                    unavailable.add(identity)
    return dict(gate_record_count=len(quality.get("issues", [])), parameter_condition_count=len(quality.get("conditions", [])),
        independent_chain_count=len(identities), ess_warning_threshold=32,
        chains_with_any_observable_ess_below32=len(low),
        chains_with_any_unavailable_ess=len(unavailable),
        minimum_saved_single_chain_ess=min(all_ess) if all_ess else None,
        low_ess_chain_identities=[list(value) for value in sorted(low)],
        unavailable_ess_chain_identities=[list(value) for value in sorted(unavailable)],
        hot_cold_warning_condition_count=len(hot_cold), hot_cold_warning_conditions=hot_cold,
        derived_from_saved_metadata_only=True, ess_or_quality_reestimated=False)


def _validate_report(folder, fingerprints, *, calibration=False):
    provenance = _read_json(folder / "report.json", fingerprints)
    status = _read_json(folder / "status.json", fingerprints)
    _complete(status, str(folder))
    if provenance.get("status") != "complete" or provenance.get("public_sources_unchanged") is not True:
        raise ValueError("Completed child report has inconsistent provenance")
    for key in ("model_loaded", "GPU_used", "inference_performed", "root_search_performed"):
        if provenance.get(key) is not False:
            raise ValueError("Expected completed model-free, root-search-free child report")
    for name, expected in provenance.get("outputs_sha256", {}).items():
        path = _public_path(folder, name)
        if _record_hash(path, fingerprints) != expected:
            raise ValueError("Child report output SHA256 disagrees with its public manifest")
    payload_name = "calibration.json" if calibration else "metrics.json"
    payload = _read_json(folder / payload_name, fingerprints)
    if payload.get("thermal_nu") is not None or payload.get("physical_critical_point_certified", False):
        raise ValueError("Exploratory results cannot certify a thermal exponent or physical critical point")
    return provenance, payload


def load_evidence(root):
    root = Path(root).resolve()
    fingerprints = {}
    plan = _read_json(root / "plan.json", fingerprints)
    job = _read_json(root / "job.json", fingerprints)
    if (plan.get("scientific_qualification") != QUALIFICATION or job.get("scientific_qualification") != QUALIFICATION
            or job.get("source_quality_passed") is not False or plan.get("source_quality_status") != "sampling_quality_hold"):
        raise ValueError("Continuation must explicitly retain the failed source quality gate")
    if not ((job.get("status") == "running" and job.get("current") == "exploratory_qualification_report")
            or (job.get("status") == "complete" and job.get("workflow_complete") is True)):
        raise ValueError("Qualification must follow the completed child training/report stages")
    for name in REQUIRED_STAGES:
        rows = [row for row in job.get("stages", []) if row.get("name") == name]
        if len(rows) != 1 or rows[0].get("status") != "complete" or rows[0].get("returncode") != 0:
            raise ValueError(f"Required preceding stage {name} is not complete with rc0")
    authorization_path = _public_path(root, plan.get("authorization_file", "authorization.json"))
    authorization = _read_json(authorization_path, fingerprints)
    if fingerprints[str(authorization_path)] != plan.get("authorization_sha256"):
        raise ValueError("Authorization SHA256 disagrees with the frozen child plan")
    if (authorization.get("user_authorization") != "允许保留质量警告继续探索性训练"
            or authorization.get("source_quality_passed") is not False):
        raise ValueError("Explicit human authorization and failed-quality qualification are required")
    source_root, data_root = Path(plan["source_root"]).resolve(), Path(plan["data_root"]).resolve()
    if (source_root == root or source_root.is_relative_to(root) or root == data_root or root.is_relative_to(data_root)
            or Path(authorization.get("source_root", "")).resolve() != source_root
            or Path(job.get("source_root", "")).resolve() != source_root):
        raise ValueError("The child must not replace the held root or write inside its physical data directory")
    source_job = _read_json(source_root / "job.json", fingerprints)
    source_plan = _read_json(source_root / "plan.json", fingerprints)
    quality = _read_json(source_root / "sampling_quality.json", fingerprints)
    manifest = _read_json(data_root / "manifest.json", fingerprints)
    quality_sha, source_job_sha = fingerprints[str(source_root / "sampling_quality.json")], fingerprints[str(source_root / "job.json")]
    data_sha = fingerprints[str(data_root / "manifest.json")]
    if (source_job.get("status") != "sampling_quality_hold" or source_job.get("workflow_complete") is not False
            or authorization.get("source_job_sha256") != source_job_sha
            or authorization.get("source_sampling_quality_sha256") != quality_sha
            or authorization.get("source_quality") != quality
            or quality.get("status") != "sampling_quality_hold" or quality.get("equilibrium_certified") is not False
            or quality.get("training_started") is not False or quality.get("GPU_profile_started") is not False):
        raise ValueError("Original held source quality evidence was altered, misattached, or relabeled")
    metadata_hashes = plan.get("source_validation", {}).get("metadata_sha256", {})
    if (metadata_hashes.get("sampling_quality.json") != quality_sha
            or metadata_hashes.get("data/manifest.json") != data_sha):
        raise ValueError("Source metadata SHA256 ledger disagrees with the preserved original evidence")
    if (plan.get("source_data_manifest_sha256") != data_sha
            or authorization.get("source_data_manifest_sha256") != data_sha
            or quality.get("data_manifest_sha256") != data_sha
            or manifest.get("schema") != DATA_SCHEMA or manifest.get("status") != "complete"):
        raise ValueError("Source data manifest SHA256 linkage or hybrid identity disagrees")
    options = manifest.get("options", {})
    count = len(manifest.get("shards", [])) * options.get("chains", 0) * options.get("samples", 0)
    if count != 6144 or len(manifest.get("shards", [])) != 12 or options.get("chains") != 4 or options.get("samples") != 128:
        raise ValueError("Expected the original 6144 saved configurations across twelve conditions and 48 chains")
    stats = quality_statistics(quality)
    if (stats["gate_record_count"] == 0 or stats["gate_record_count"] != authorization.get("source_quality_issue_count")
            or stats["independent_chain_count"] != 48 or stats["parameter_condition_count"] != 12):
        raise ValueError("Preserved source quality records/chain inventory are incomplete")
    run_manifest = _read_json(root / "run" / "manifest.json", fingerprints)
    run_summary = _read_json(root / "run" / "summary.json", fingerprints)
    run_status = _read_json(root / "run" / "status.json", fingerprints)
    _complete(run_status, "QCR training")
    if (run_manifest.get("recipe") != TRAINING_RECIPE or run_summary.get("recipe") != TRAINING_RECIPE
            or run_manifest.get("options", {}).get("seed") != 91 or run_manifest.get("options", {}).get("steps") != 3000
            or run_summary.get("actual_steps") != 3000 or run_status.get("actual_steps") != 3000
            or run_summary.get("latest_checkpoint_step") != 3000 or run_status.get("latest_checkpoint_step") != 3000
            or run_summary.get("frozen_evaluation_verified") is not True
            or run_manifest.get("initialization") != "random; no warm start or kernel selection"
            or run_manifest.get("data_manifest_sha256") != data_sha or run_summary.get("data_manifest_sha256") != data_sha
            or run_manifest.get("labels_used_in_training") is not False or run_summary.get("labels_used_in_training") is not False):
        raise ValueError("Fresh seed91/3000-step unsupervised run/data identity disagrees")
    _optimizer_clock(run_summary.get("optimizer_steps", {}))
    if run_status.get("summary_sha256") != fingerprints[str(root / "run" / "summary.json")]:
        raise ValueError("Training status summary SHA256 disagrees")
    arrays_sha = _record_hash(root / "run" / "evaluation_arrays.npz", fingerprints)
    if run_status.get("evaluation_arrays_sha256") != arrays_sha:
        raise ValueError("Training status public evaluation-array SHA256 disagrees")
    training_pins = {}
    for name in MODEL_SOURCES:
        original_sha, child_sha = source_plan.get("source_sha256", {}).get(name), plan.get("source_sha256", {}).get(name)
        matches = [digest for path, digest in run_manifest.get("source_sha256", {}).items() if str(path).endswith(name)]
        if not original_sha or original_sha != child_sha or matches != [child_sha]:
            raise ValueError("Frozen model/loss/trainer source fingerprints differ from the held experiment")
        training_pins[name] = child_sha
    standard_provenance, standard = _validate_report(root / "report", fingerprints)
    calibration_provenance, calibration = _validate_report(root / "calibration_report", fingerprints, calibration=True)
    if (standard.get("actual_steps") != 3000 or calibration.get("actual_steps") != 3000
            or calibration.get("optimizer_steps") != run_summary["optimizer_steps"]
            or calibration.get("data_schema") != DATA_SCHEMA):
        raise ValueError("Completed child reports disagree with actual training/Adam endpoints")
    for provenance, nested in ((standard_provenance, False), (calibration_provenance, True)):
        hashes = provenance.get("sources_sha256_after", {})
        hashes = hashes.get("run", {}) if nested else hashes
        for name in ("manifest.json", "summary.json", "status.json", "evaluation_arrays.npz"):
            if hashes.get(name) != fingerprints[str(root / "run" / name)]:
                raise ValueError("Completed reports refer to a different public training endpoint")
    if calibration_provenance.get("sources_sha256_after", {}).get("data", {}).get("manifest.json") != data_sha:
        raise ValueError("Hybrid calibration report is not attached to the original data manifest")
    if fingerprints != {path: sha256(path) for path in fingerprints}:
        raise ValueError("Public exploratory evidence changed during reading")
    return dict(root=root, plan=plan, job=job, authorization=authorization, source_root=source_root,
        data_root=data_root, source_job=source_job, source_quality=quality, data_manifest=manifest,
        quality_statistics=stats, fingerprints=fingerprints, model_training_source_pins=training_pins,
        run_summary=run_summary, standard_metrics=standard, calibration=calibration)


def _warning_plot(evidence, output, *, font=None):
    plt = _plotting(font)
    stats = evidence["quality_statistics"]
    minimum = stats["minimum_saved_single_chain_ess"]
    fig = plt.figure(figsize=(13.5, 7.4), facecolor="#fffaf0")
    fig.text(.045, .91, "探索性训练结果：采样质量警告未消除", fontsize=23, weight="bold", color="#8e2633")
    fig.text(.045, .84, SCIENTIFIC_LABEL, fontsize=16, color="#8e2633")
    lines = [
        "原质量门控状态：未通过；原实验及其门控记录保持不变。",
        f"完整保留 {stats['gate_record_count']} 条门控记录；记录可重复，不等于独立失败链数。",
        f"{stats['chains_with_any_observable_ess_below32']}/{stats['independent_chain_count']} 条链至少一项保存观测的 ESS < 32；最小 ESS = {minimum:.4f}。" if minimum is not None else "单链 ESS 不可用，不能推断平衡。",
        f"{stats['hot_cold_warning_condition_count']} 个参数条件存在热冷初态均值的描述性警告。",
        "用户明确允许保留以上警告继续；这不是质量门控通过或平衡认证。",
        "新模型从随机初始化训练：seed91，Q／C／R 的 Adam 参数均完成 3000 次更新。",
        "模型、loss 与源数据保持关联；没有再次采样、删除链或选择结果更好的种子。",
        "数值流场、重构／闭合误差与 loss 曲线必须在上述采样限制下解释。",
        "沿用连续场软统计／平滑信息代理，不是二值精确直方图 MI。",
        "未认证临界位置、热方向或临界指数；thermal_nu = null。",
    ]
    for index, text in enumerate(lines):
        fig.text(.05, .74 - index * .062, text, fontsize=13, color="#292f36")
    fig.text(.05, .045, "执行完成 ≠ 采样已平衡 ≠ 物理 RG 已被认证", fontsize=15, color="#8e2633", weight="bold")
    fig.savefig(output / "warning.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_report(root, output=None, *, font=None):
    evidence = load_evidence(root)
    output = (evidence["root"] / "qualification_report" if output is None else Path(output)).resolve()
    for source in (evidence["root"], evidence["source_root"], evidence["data_root"]):
        if output == source or source.is_relative_to(output):
            raise ValueError("Qualification output must not contain or replace source evidence")
    if output.exists():
        raise FileExistsError(f"Qualification output already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    _warning_plot(evidence, output, font=font)
    after = {path: sha256(path) for path in evidence["fingerprints"]}
    if after != evidence["fingerprints"]:
        raise ValueError("Public source evidence changed during exploratory qualification")
    receipt = dict(status="complete", scientific_qualification=QUALIFICATION, scientific_label=SCIENTIFIC_LABEL,
        source_quality_passed=False, equilibrium_certified=False, thermal_nu=None,
        critical_r=None, thermal_mode_certified=False, physical_critical_point_certified=False,
        root_search_performed=False, sampling_executed=False, model_loaded=False,
        training_executed=False, evaluation_recomputed=False, physical_payloads_rehashed=False,
        actual_training_steps=3000, optimizer_steps=evidence["run_summary"]["optimizer_steps"],
        seed=91, training_recipe=TRAINING_RECIPE, source_saved_configuration_count=6144,
        source_root=str(evidence["source_root"]), data_root=str(evidence["data_root"]),
        source_data_manifest_sha256=evidence["plan"]["source_data_manifest_sha256"],
        source_sampling_quality_sha256=evidence["authorization"]["source_sampling_quality_sha256"],
        source_quality=evidence["source_quality"], quality_statistics=evidence["quality_statistics"],
        source_validation=evidence["plan"]["source_validation"],
        user_authorization=evidence["authorization"],
        unchanged_model_training_source_pins=evidence["model_training_source_pins"],
        parent_job_snapshot=evidence["job"],
        parent_job_snapshot_note="This stage-time job snapshot is expected to change when its parent marks the final stage complete; its captured bytes are represented by this embedded object and the recorded SHA.",
        sources_sha256_before=evidence["fingerprints"], sources_sha256_after=after,
        source_artifacts_unchanged_during_qualification=True,
        standard_report_metrics=evidence["standard_metrics"], calibration_report=evidence["calibration"],
        warning_png_sha256=sha256(output / "warning.png"),
        public_files=["exploratory_receipt.json", "warning.png", "status.json"], limitations=LIMITATIONS)
    (output / "exploratory_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2,
        allow_nan=False) + "\n", encoding="utf-8")
    (output / "status.json").write_text(json.dumps(dict(status="complete", workflow_complete=True,
        error=None, requires_user_action=False, scientific_qualification=QUALIFICATION,
        source_quality_passed=False, equilibrium_certified=False, thermal_nu=None),
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--font", type=Path)
    args = parser.parse_args(argv)
    receipt = write_report(args.root, args.output, font=args.font)
    print(json.dumps(dict(status=receipt["status"], scientific_qualification=QUALIFICATION,
        source_quality_passed=False, output=str((args.output or args.root / "qualification_report").resolve())), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
