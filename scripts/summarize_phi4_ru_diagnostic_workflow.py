"""Verify and assemble the independent phi4 diagnostic stages, JSON only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.report_phi4_qcr import sha256


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def run(root):
    root = Path(root).resolve()
    if (root / "summary.json").exists() or (root / "status.json").exists():
        raise FileExistsError("Existing workflow summary is never overwritten")
    reports, inventory = {}, {}
    for stage in ("dynamics", "physics", "geometry", "r_optimizer", "r_refinement", "r_convergence"):
        folder = root / stage
        summary = json.loads((folder / "summary.json").read_text())
        if (summary.get("status") != "complete" or not summary.get("workflow_complete")
                or summary.get("returncode") != 0):
            raise ValueError(f"Incomplete stage: {stage}")
        receipt_name = "public_files.json" if (folder / "public_files.json").exists() else "receipt.json"
        receipt = json.loads((folder / receipt_name).read_text())
        if receipt_name == "public_files.json":
            for name, digest in receipt.items():
                file = (folder / name).resolve()
                if folder not in file.parents or sha256(file) != digest:
                    raise ValueError(f"Output fingerprint mismatch: {file}")
        else:
            # The physical stage records output hashes in its separate receipt.
            hashes = receipt.get("outputs_sha256", receipt.get("output_sha256", {}))
            for name, digest in hashes.items():
                file = Path(name)
                file = file if file.is_absolute() else folder / file
                if sha256(file) != digest:
                    raise ValueError(f"Output fingerprint mismatch: {file}")
        for file in folder.iterdir():
            if file.is_file() and file.suffix in (".json", ".npz", ".png"):
                inventory[str(file.relative_to(root))] = sha256(file)
        reports[stage] = summary
    physics, dynamics, geometry = [reports[x] for x in ("physics", "dynamics", "geometry")]
    output = dict(status="complete", workflow_complete=True, returncode=0, error=None,
        plan_sha256=sha256(root / "plan.json"),
        completed_stages=list(reports),
        physical_crossover_intervals=[{k: r[k] for k in (
            "u", "sampled_r_interval", "latent_distance_ratio_12_over_36_and_4_over_12")}
            for r in physics["crossover_intervals"]],
        empirical_flow=dict(resolved_displacements=dynamics["resolved_empirical_displacements_by_scale"],
            neighborhood_summary=dynamics["neighborhood_summary"],
            selected_original_endpoint_metrics=dynamics["model_metrics"][dynamics["selected_model"]]),
        geometry=dict(nearest_stencil=geometry["analyses"]["stride1"]["counts"],
                      stencil_sensitivity=geometry["stencil_sensitivity"]),
        r_optimizer=reports["r_optimizer"], r_refinement=reports["r_refinement"],
        r_convergence=reports["r_convergence"],
        Q_updates=0, C_updates=0, new_physical_sampling=False, GPU_used=False,
        scientific_conclusions=[
            "已有物理数据跨过五个u切片上的明显有限尺寸变化区；Q/C保留相区差异，区间两端随粗粒化进一步分离。",
            "原归一化R的第一步方向总体一致，末尺度有局部方向失配；编码重复均值后偏差仍然明显。",
            "邻域和局部几何检查尚不足以证明缺少状态变量、真实内禀维度或必须增加dg。",
            "R优化器对照只改变数值求解，原始g的all-start目标、冻结Q/C和样本保持一致。",
            "单L36的粗粒化轨迹及稀疏过渡数据尚不认证物理临界线、插值鞍点或临界指数。"],
        physical_critical_point_certified=False, critical_r=None, thermal_nu=None,
        new_root_search=False, automatic_next_experiment=False,
        labels_used_for_training=False,
        limitations=["现有TEST已经多次查看，只作探索性评价。", "R的三个优化种子共用同一Q/C，不代表三个完整QCR训练种子。",
                     "每条件只有四条物理链，55个已有采样警告保留；子集/C重复不增加独立物理链数。",
                     "L-BFGS与Adam的更新次数和函数求值次数不同，不能据此声称严格等计算量优越性。"])
    write_json(root / "summary.json", output)
    write_json(root / "status.json", dict(status="complete", workflow_complete=True,
               returncode=0, error=None, stages={s: "complete" for s in reports},
               summary_sha256=sha256(root / "summary.json")))
    for name in ("summary.json", "status.json", "plan.json"):
        inventory[name] = sha256(root / name)
    inventory["refinement_plan.json"] = sha256(root / "refinement_plan.json")
    inventory["convergence_plan.json"] = sha256(root / "convergence_plan.json")
    write_json(root / "public_files.json", inventory)
    print(json.dumps(dict(output=str(root), completed_stages=list(reports)), ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    run(parser.parse_args().root)
