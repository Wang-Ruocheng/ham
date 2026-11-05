"""单因素、从零初始化的完整 φ⁴ QCR 对照；只复用既有物理数据。"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts import run_phi4_q_diagnostic as guard
from scripts.run_gaussian_scalar_qcr import source_hashes, sha, save, directory_bytes


SOURCE = guard.SOURCE
OLD_RUN = SOURCE / "exploratory_training_v1/run"
Q_DIAGNOSTIC = SOURCE.parent / "phi4_Q_frozenC_diagnostic_seed91_vm225_20261027"
INITIAL_SHA = "69b1bd877234db2a849a0312ae32b7c3dec744a0556f4a0b0d82c68f372e0664"
FINAL_C_SHA = "c71b1df615990eb737384cbbbce1d1b6902a588fd774d3d036232bcf0aebcfb8"
DATA_SHA = "588921b87e3f10be9a8c2ce552d70926758b042481a3800a4efc0fa37430e361"
MODEL_SHA = "016dd47a77ee7aa149a1ed65a33e98e35447aeaed904e994030d4eac1c55ed1d"
CPU_SECONDS, GPU_SECONDS = 1800, 1800
read, now, cpu_seconds = guard.read, guard.now, guard.cpu_seconds


def immutable_inputs():
    files = guard.source_files()
    files.update({str(path): sha(path) for path in sorted(SOURCE.rglob("*.json")) if path.is_file()})
    files[str(Q_DIAGNOSTIC / "training/summary.json")] = sha(Q_DIAGNOSTIC / "training/summary.json")
    return files


def commands(root):
    return [
        ("mean_bias_full_QCR", [sys.executable, "-u", "-m", "experiments.phi4_qcr",
            "--data", str(SOURCE / "data"), "--output", str(root / "run"),
            "--device", "cuda:1", "--steps", "3000", "--system-batch", "8",
            "--sample-batch", "64", "--seed", "91", "--threads", "1",
            "--log-every", "100", "--checkpoint-every", "1000", "--information-pairs", "128",
            "--max-training-seconds", "1500", "--max-cuda-bytes", str(8*1024**3),
            "--expected-gpu-uuid", guard.GPU_UUID, "--q-initialization", "train_mean_bias",
            "--expected-random-construction-sha256", INITIAL_SHA], True),
        ("standard_report", [sys.executable, "-u", "-m", "scripts.report_phi4_qcr",
            "--run", str(root / "run"), "--output", str(root / "report")], False),
        ("hybrid_calibration_report", [sys.executable, "-u", "-m", "scripts.report_phi4_hybrid_qcr",
            "--data", str(SOURCE / "data"), "--run", str(root / "run"),
            "--output", str(root / "calibration_report")], False),
        ("paired_initialization_comparison", [sys.executable, "-u", "-m",
            "scripts.report_phi4_qcr_mean_init_comparison", "--old-run", str(OLD_RUN),
            "--new-run", str(root / "run"), "--sampling-quality", str(SOURCE / "sampling_quality.json"),
            "--output", str(root / "comparison")], False)]


def prepare(root):
    guard.validate_host()
    root, code = Path(root).resolve(), Path(__file__).resolve().parents[1]
    if root.exists() or root.is_symlink():
        raise FileExistsError("No overwrite, restart or resume is permitted")
    if root == SOURCE or SOURCE in root.parents or root == code or root in code.parents:
        raise ValueError("New independent output must not replace or sit inside an immutable source")
    original = immutable_inputs()
    old_manifest, old_summary = read(OLD_RUN / "manifest.json"), read(OLD_RUN / "summary.json")
    if (read(SOURCE / "exploratory_training_v1/job.json")["status"] != "complete"
            or old_summary.get("actual_steps") != 3000
            or old_manifest["initial_state_sha256"] != INITIAL_SHA
            or old_manifest["data_manifest_sha256"] != DATA_SHA
            or sha(SOURCE / "data/manifest.json") != DATA_SHA
            or original[str(OLD_RUN / "training_step_3000.pt")] != guard.CHECKPOINT_SHA):
        raise RuntimeError("Completed historical endpoint/data provenance mismatch")
    if read(SOURCE / "sampling_validity.json").get("severe") != []:
        raise RuntimeError("Severe physical-data invalidity cannot be waived")
    diagnostic = read(Q_DIAGNOSTIC / "training/summary.json")
    if (diagnostic["frozen_source_before"]["C"] != FINAL_C_SHA
            or diagnostic["source_pins"]["checkpoint_sha256"] != guard.CHECKPOINT_SHA):
        raise RuntimeError("Reference final-C identity differs from the audited source")
    if sha(code / "recurrent_rg/phi4_qcr.py") != MODEL_SHA:
        raise RuntimeError("Model/loss source changed; this is an initialization-only comparison")
    estimate = 1.5*old_summary["timing"]["wall_seconds"]+300
    if estimate > GPU_SECONDS:
        raise RuntimeError("Conservative GPU projection exceeds the new run budget")
    plan = dict(schema_version=5, recipe="phi4_mean_bias_full_QCR_paired_seed91_v1",
        output=str(root), code_root=str(code), host=guard.HOST, device="cuda:1", gpu_uuid=guard.GPU_UUID,
        prepared_at=now(), source_sha256=source_hashes(code), immutable_input_sha256=original,
        baseline_run=str(OLD_RUN), data_root=str(SOURCE / "data"), data_manifest_sha256=DATA_SHA,
        expected_random_construction_state_sha256=INITIAL_SHA, expected_final_C_sha256=FINAL_C_SHA,
        model_loss_source_sha256=MODEL_SHA,
        authorization="用户要求检查修正 Q 初始化后的完整 QCR 训练；同数据单 seed 单因素从头对照",
        single_change="Q.decoder.bias 初始化为初始随机 C 的 TRAIN 各尺度统计均值；不改方向、模型或 loss",
        training=dict(steps=3000, seed=91, lattice_sizes=[36,12,4], theory_dim=1, dtype="float64",
            system_batch=8, sample_batch=64, information_pairs=128, learning_rate=.001,
            optimizers="three independent fresh Adam", warm_start=False, labels_used=False,
            closure_objective="all_start_anchor", closure_scale_weights=None,
            initialization_seed=9091, batch_seed=192, field_seed=293,
            C_trainable=True, C_selection=False, Q_or_C_pretraining=False, PCA_initialization=False),
        resource_policy=dict(cpu_threads=1,nice=10,cpu_seconds=CPU_SECONDS,gpu_stage_seconds=GPU_SECONDS,
            own_cuda_peak_bytes=8*1024**3,directory_bytes=1024**3,minimum_host_available_gib=36,
            gpu_free_mib=12288,gpu_compute_processes=0,two_gpu_checks_seconds_apart=10,
            load_per_cpu_max=1.25,io_full_avg10_percent_max=20),
        budget_projection=dict(baseline_wall_seconds=old_summary["timing"]["wall_seconds"],
            conservative_multiplier=1.5,extra_reserve_seconds=300,
            projected_GPU_stage_seconds=estimate,not_a_wall_time_guarantee=True),
        qualification="exploratory_all_sampling_warnings_retained",
        original_warning_authorization="允许保留质量警告继续探索性训练",
        scientific_limits=["同一已完成物理数据，无补样、换 seed 或删链；保留 11/48 链低 ESS 与热冷警告",
            "不认证平衡；单 seed、dimg1、有限尺寸与连续信息代理的限制仍在",
            "新旧 g 坐标不同，原始 latent closure/loss 不直接跨 run 排优",
            "C 独立梯度与随机流保持配对；需核对最终 C 与同尺度统计目标",
            "数值根不认证物理相变；critical_r、thermal_nu 保持 null"],
        commands=[dict(stage=n,command=c,gpu=g) for n,c,g in commands(root)])
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    save(root / "plan.json",plan,exclusive=True)
    save(root / "job.json",dict(status="prepared",workflow_complete=False,current=None,
        stages=[],error=None,requires_user_action=False),exclusive=True)
    return plan


def unchanged(plan):
    if source_hashes(plan["code_root"]) != plan["source_sha256"]:
        raise RuntimeError("Frozen new source inventory changed")
    if immutable_inputs() != plan["immutable_input_sha256"]:
        raise RuntimeError("Immutable original experiment/diagnostic metadata changed")


def execute(root):
    guard.validate_host()
    root=Path(root).resolve()
    plan,job=read(root/"plan.json"),read(root/"job.json")
    if job["status"] != "prepared" or job["stages"]:
        raise RuntimeError("No duplicate launch, restart or resume is permitted")
    with (root/"execution.lock").open("x") as stream:
        stream.write(str(os.getpid())+"\n")
    started,cpu_start=time.perf_counter(),cpu_seconds()
    job.update(status="running",parent_pid=os.getpid(),started_at=now())
    save(root/"launch.json",dict(host=guard.HOST,parent_pid=os.getpid(),at=now(),
        plan_sha256=sha(root/"plan.json"),code_root=plan["code_root"],
        actual_nice=os.getpriority(os.PRIO_PROCESS,0),cpu_threads=1,
        resource_policy=plan["resource_policy"],duplicate_launch_permitted=False),exclusive=True)
    try:
        unchanged(plan)
        for name,cmd,gpu in commands(root):
            guard.admission(root,job,name,gpu)
            unchanged(plan)
            row=dict(name=name,status="running",returncode=None,started_at=now(),command=cmd,gpu=gpu)
            job["stages"].append(row)
            job.update(status="running",current=name)
            save(root/"job.json",job)
            tick=time.perf_counter()
            with (root/"logs"/(name+".log")).open("x") as stream:
                child=subprocess.Popen(cmd,cwd=plan["code_root"],stdout=stream,stderr=subprocess.STDOUT)
                row["pid"]=child.pid
                save(root/"job.json",job)
                rc=child.wait()
            row.update(returncode=rc,status="complete" if rc==0 else "failed",
                       finished_at=now(),wall_seconds=time.perf_counter()-tick)
            job.update(cpu_seconds=cpu_seconds()-cpu_start,directory_bytes=directory_bytes(root))
            if gpu:
                job["gpu_stage_wall_seconds"]=row["wall_seconds"]
            save(root/"job.json",job)
            unchanged(plan)
            if rc:
                status=read(root/"run/status.json") if (root/"run/status.json").exists() else {}
                if status.get("status")=="budget_hold":
                    row["status"]="budget_hold"
                    job.update(status="budget_hold",requires_user_action=True,error=status.get("error"),returncode=rc)
                    save(root/"job.json",job)
                    return 3
                raise RuntimeError(f"{name} nonzero returncode {rc}; no restart or repair")
            if job["cpu_seconds"]>CPU_SECONDS or job.get("gpu_stage_wall_seconds",0)>GPU_SECONDS or job["directory_bytes"]>1024**3:
                job.update(status="budget_hold",requires_user_action=True,error="完整对照达到资源预算；不缩步数或重跑",returncode=3)
                save(root/"job.json",job)
                return 3
        manifest,summary=read(root/"run/manifest.json"),read(root/"run/summary.json")
        if manifest.get("random_construction_state_sha256") != INITIAL_SHA:
            raise RuntimeError("Random source construction did not match the historical run")
        paired_C=summary.get("final_module_sha256",{}).get("C")==FINAL_C_SHA
        public={}
        for folder in ("run","report","calibration_report","comparison"):
            for path in sorted((root/folder).rglob("*")):
                if path.is_file() and path.suffix in (".json",".npz",".png"):
                    public[str(path.relative_to(root))]=dict(sha256=sha(path),bytes=path.stat().st_size)
        job.update(status="complete" if paired_C else "comparison_hold",workflow_complete=paired_C,
            current=None,finished_at=now(),returncode=0 if paired_C else 4,
            cpu_seconds=cpu_seconds()-cpu_start,wall_seconds=time.perf_counter()-started,
            public_files=public,source_fingerprints_unchanged=True,paired_final_C_identical=paired_C,
            actual_QCR_steps=summary["actual_steps"],
            error=None if paired_C else "最终 C 指纹未保持配对；不自动重跑，需要审查随机流/数值差异",
            requires_user_action=not paired_C)
        save(root/"job.json",job)
        return 0 if paired_C else 4
    except Exception as exc:
        job.update(status="failed",workflow_complete=False,error=str(exc),requires_user_action=True,
                   finished_at=now(),returncode=1,cpu_seconds=cpu_seconds()-cpu_start)
        save(root/"job.json",job)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    parser.add_argument("--prepare",action="store_true")
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    if args.prepare==args.execute:
        parser.error("Choose exactly one of --prepare or --execute")
    if args.prepare:
        prepare(args.output)
        return 0
    return execute(args.output)


if __name__=="__main__":
    raise SystemExit(main())
