"""Read-only public-artifact comparison of paired 3D Ising RSMI/no-MI runs.

No model, checkpoint, physical data, or MI cache is opened. Learned g coordinates
and coarse-graining kernels differ, so native closure errors are not a strict
common-coordinate ranking. All saved fixed-point candidates are retained.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np


COORDINATE_WARNING = ("两次运行分别学习 Q、C、R，g 坐标与粗粒化目标均可能变化。"
    "各自原始 RMSE 及按各自 TRAIN 目标标准差归一化的 NRMSE 仅作并列描述，"
    "不能作为同一坐标或同一物理粗粒化目标下的严格胜负。"
    "微观局部概率重建误差处于共同可观测量空间，但单一指标也不能认证 RG 或热指数。")


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _json(path):
    return json.loads(path.read_text())


def load_public(root):
    root = Path(root).resolve()
    report = _json(root / "report/report.json")
    manifest = _json(root / "run/manifest.json")
    summary = _json(root / "run/summary.json")
    if report.get("status") != "complete" or report.get("workflow_complete") is not True:
        raise ValueError("Both source reports must be complete")
    if not report.get("model_state_unchanged") or not report.get("model_gradients_none"):
        raise ValueError("Both source reports must preserve the frozen model and gradients")
    arrays_path = root / "report/encodings.npz"
    if sha256(arrays_path) != report.get("public_arrays_sha256"):
        raise ValueError("Public report arrays fingerprint mismatch")
    weight = manifest.get("rsmi_weight")
    if weight != summary.get("contract", {}).get("rsmi_weight"):
        raise ValueError("Manifest and summary objective contracts disagree")
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    return dict(root=root, report=report, manifest=manifest, summary=summary, arrays=arrays,
        fingerprints={name: sha256(root / name) for name in
            ("report/report.json", "report/encodings.npz", "run/manifest.json", "run/summary.json")})


def _config_for_comparison(manifest):
    cfg = copy.deepcopy(manifest["config"])
    cfg["data"].pop("cache", None)
    # Removed legacy knobs were inactive in the baseline, not an objective change.
    loss = cfg["loss"]
    if loss.get("kernel_distinct_alpha", 0.) == 0.:
        loss.pop("kernel_distinct_alpha", None)
        loss.pop("kernel_distinct_margin", None)
    return cfg


def _observable_reconstruction(arrays):
    result = {}
    for split in ("train", "test"):
        difference = arrays[f"motif_reconstruction_{split}"] - arrays[f"motif_target_{split}"]
        result[split] = np.sqrt(np.mean(difference ** 2, axis=(0, 2))).tolist()
    return dict(RMSE_by_scale=result,
        qualification="局部 2×2×2 概率向量的逐分量 RMSE；先平均 C。仅 k=0 的物理目标跨运行相同，k>0 的目标随 C 变化。")


def compare(baseline, candidate, output):
    baseline, candidate, output = Path(baseline).resolve(), Path(candidate).resolve(), Path(output).resolve()
    if baseline == candidate:
        raise ValueError("Baseline and candidate must be distinct runs")
    if output.exists():
        raise FileExistsError(output)
    old, new = load_public(baseline), load_public(candidate)
    if old["manifest"].get("rsmi_weight", 0.) <= 0 or new["manifest"].get("rsmi_weight") != 0.:
        raise ValueError("Expected positive-RSMI baseline and explicit zero-RSMI candidate")
    policy = new["report"].get("MI_diagnostic_policy", {})
    if new["report"].get("MI_shuffle_diagnostic") is not None or policy.get("computed") is not False:
        raise ValueError("No-MI candidate must explicitly record its MI diagnostic as uncomputed")
    if not old["manifest"].get("data_sha256") or old["manifest"]["data_sha256"] != new["manifest"].get("data_sha256"):
        raise ValueError("Paired comparison requires identical label-free data fingerprints")
    for field in ("C_draws", "samples", "encoding_protocol", "scale_shapes", "actual_training_updates"):
        if old["report"].get(field) != new["report"].get(field):
            raise ValueError(f"Paired evaluation protocol differs: {field}")
    if _config_for_comparison(old["manifest"]) != _config_for_comparison(new["manifest"]):
        raise ValueError("Paired experiment config differs beyond inactive legacy distinct fields or cache path")
    for key in ("sample_partition", "K_train", "K_test", "condition_index_train", "condition_index_test"):
        if not np.array_equal(old["arrays"][key], new["arrays"][key]):
            raise ValueError(f"Paired physical evaluation rows differ: {key}")
    initial_keys = ("random_construction_module_sha256", "pre_affine_initialization_module_sha256", "training_start_module_sha256")
    initial = {key: bool(old["manifest"].get(key)) and old["manifest"].get(key) == new["manifest"].get(key)
        for key in initial_keys}
    if not all(initial.values()):
        raise ValueError("Paired experiment must match Q/C/R initialization SHA at all three saved stages")
    runs, arrays = {}, {}
    for label, loaded in (("baseline", old), ("candidate", new)):
        report = loaded["report"]
        runs[label] = dict(root=str(loaded["root"]), rsmi_weight=loaded["manifest"]["rsmi_weight"],
            actual_training_updates=report["actual_training_updates"], closure=report["closure"],
            fixed_point_candidates=report["fixed_point_candidates"], thermal_nu=None,
            motif_probability_reconstruction=_observable_reconstruction(loaded["arrays"]),
            model_sha256=report["model_sha256"], source_fingerprints=loaded["fingerprints"],
            MI_shuffle_diagnostic=report.get("MI_shuffle_diagnostic"),
            MI_diagnostic_policy=report.get("MI_diagnostic_policy"), qualification=report.get("qualification"))
        for key in ("R_grid", "R_mapped", "R_displacement", "K_train", "K_test", "direct_train", "direct_test"):
            arrays[f"{label}_{key}"] = loaded["arrays"][key]
    result = dict(status="complete", workflow_complete=True, schema_version=1,
        comparison="三维铁磁 Ising：原 RSMI 与仅保色＋方差防塌缩 C 目标的配对消融",
        labels_used_in_training=False, data_sha256=old["manifest"]["data_sha256"],
        initial_module_SHA_matches=initial, all_initial_module_SHA_match=all(initial.values()),
        coordinate_warning=COORDINATE_WARNING, strict_cross_run_closure_ranking_valid=False,
        selected_best_root=None, thermal_nu=None,
        root_policy="保留双方正式报告全部候选；不新搜根，不按接近理论 Kc 或理论斜率挑选。候选 nu 非认证热指数。",
        runs=runs, opened_artifact_types=["JSON", "NPZ"], model_or_data_loaded=False)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "comparison_arrays.npz", **arrays)
    result["public_arrays_sha256"] = sha256(output / "comparison_arrays.npz")
    (output / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.baseline, args.candidate, args.output)
    print(json.dumps(dict(status=result["status"], all_initial_module_SHA_match=result["all_initial_module_SHA_match"],
        strict_cross_run_closure_ranking_valid=False), ensure_ascii=False))


if __name__ == "__main__":
    main()
