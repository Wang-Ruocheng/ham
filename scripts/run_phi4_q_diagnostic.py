"""隔离的固定 C／Q-only 诊断；不采样、不运行 MI、不重训 R。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time

from scripts.run_gaussian_scalar_qcr import source_hashes, sha, save, directory_bytes


HOST = "thinklab-105-225"
GPU_UUID = "GPU-18fac97c-11ee-4624-d298-9074e8d7028c"
SOURCE = Path("/mnt/nas-new/home/yange/ham/results/phi4_u02_hybrid_QCR_b3_g1_L36_seed91_vm225_20261027")
CHECKPOINT_SHA = "5df6f4d55d97792be13ecf1811eb44b829ba283210cd46148c8d478e46bc2a7e"


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def cpu_seconds():
    own, children = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime


def source_files():
    run = SOURCE / "exploratory_training_v1/run"
    paths = [SOURCE / "job.json", SOURCE / "data/manifest.json",
             SOURCE / "exploratory_training_v1/job.json"]
    paths += [run / name for name in ("manifest.json", "summary.json", "status.json",
                                      "evaluation_arrays.npz", "training_step_3000.pt")]
    return {str(path): sha(path) for path in paths}


def validate_host():
    if socket.gethostname() != HOST or "CUDA_VISIBLE_DEVICES" in os.environ:
        raise RuntimeError("Only the pinned physical GPU1 on thinklab-105-225 is authorized")


def commands(root, code):
    source = str(SOURCE / "exploratory_training_v1/run")
    return [
        ("frozen_C_statistics", [sys.executable, "-u", "-m", "scripts.collect_phi4_frozen_q",
            "--source-run", source, "--output", str(root / "cache"), "--device", "cuda:1",
            "--cpu-budget-seconds", "180", "--gpu-budget-seconds", "120"], True),
        ("five_Q_only_arms", [sys.executable, "-u", "-m", "experiments.phi4_q_only_diagnostic",
            "--source-run", source, "--cache", str(root / "cache"),
            "--output", str(root / "training"), "--steps", "3000",
            "--cpu-budget-seconds", "600"], False),
        ("Q_only_report", [sys.executable, "-u", "-m", "scripts.report_phi4_q_diagnostic",
            "--training", str(root / "training"), "--output", str(root / "report")], False)]


def prepare(root):
    validate_host()
    root, code = Path(root).resolve(), Path(__file__).resolve().parents[1]
    if root.exists() or root.is_symlink():
        raise FileExistsError("Never overwrite or resume a diagnostic output")
    original = source_files()
    if original[str(SOURCE / "exploratory_training_v1/run/training_step_3000.pt")] != CHECKPOINT_SHA:
        raise RuntimeError("Source checkpoint SHA does not match the authorized endpoint")
    if read(SOURCE / "exploratory_training_v1/job.json").get("status") != "complete":
        raise RuntimeError("Source exploratory workflow must already be complete")
    plan = dict(schema_version=5, recipe="phi4_frozen_C_Q_only_diagnostic_v1", host=HOST,
        code_root=str(code), output=str(root), created_at=now(), source_sha256=source_hashes(code),
        immutable_source_files_sha256=original, device="cuda:1", gpu_uuid=GPU_UUID,
        authorization="用户要求单独检查连续 phi4 的 Q 能否解决；仅隔离 Q-only 诊断",
        steps_per_arm=3000, optimizer="fresh Adam .001", seed=91,
        arms={"A": "源 Q warm-start；新 Adam；原完整 Q 目标",
              "B": "原非线性 Q 随机初始化；原完整 Q 目标",
              "C": "与 B 相同，仅 decoder bias 初始化为 TRAIN 特征均值",
              "D": "与 C 相同 decoder，encoder 换为线性；完整 Q 目标",
              "E": "与 C 相同初始化，仅去 subset 优化项；仍评估完整 Q 目标"},
        full_Q_objective="10 reconstruction +20 microscopic subset +5 coarse subset",
        paired_batch=dict(system_batch=8, sample_batch=64, halves=[32,32],
                          train_C_draw_seeds=[202610270910,202610270911], evaluation_C_seed=20261022),
        policy=dict(cpu_threads=1, nice=10, cpu_seconds=1000, gpu_stage_seconds=120,
            gpu_peak_bytes=8*1024**3, directory_bytes=1024**3, host_min_available_gib=36,
            load_per_cpu_max=1.25, io_full_avg10_max=20, gpu_free_mib=12288,
            gpu_compute_processes=0, two_gpu_checks_seconds_apart=10),
        forbidden=["物理补样", "MI 计算", "R 训练或新 R(g) 推断", "根搜索",
                   "更改默认配方", "修改源实验或数据", "TEST 选模", "PCA 初始化"],
        limits=["源数据保留 11/48 链低 ESS 警告，未认证平衡",
                "固定最终 C 的两个训练噪声缓存不同于原动态 C 联合训练",
                "Q-only 重构改善不证明 RG 不动点或临界指数修复",
                "单 seed、有限尺寸与统计基；PCA 只为 TRAIN-only 解析诊断参考"])
    plan["commands"] = [dict(stage=name, command=cmd, gpu=gpu)
                         for name, cmd, gpu in commands(root, code)]
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    save(root / "plan.json", plan, exclusive=True)
    save(root / "job.json", dict(status="prepared", workflow_complete=False, stages=[],
         current=None, error=None, requires_user_action=False), exclusive=True)
    return plan


def unchanged(plan):
    if source_hashes(plan["code_root"]) != plan["source_sha256"]:
        raise RuntimeError("Frozen diagnostic source inventory changed")
    if source_files() != plan["immutable_source_files_sha256"]:
        raise RuntimeError("Immutable source experiment fingerprint changed")


def snapshot():
    import re
    available = int(re.search(r"MemAvailable:\s+(\d+)", Path("/proc/meminfo").read_text())[1])*1024
    io = float(re.search(r"full avg10=([\d.]+)", Path("/proc/pressure/io").read_text())[1])
    values = subprocess.check_output(["nvidia-smi", "-i", "1", "--query-gpu=uuid,memory.free",
                                      "--format=csv,noheader,nounits"], text=True).strip().split(",")
    uuid, free = values[0].strip(), int(values[1])
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                        "--format=csv,noheader,nounits"], text=True).splitlines()
    pids = [int(line.split(",")[1]) for line in processes if uuid in line]
    cpus, load = os.cpu_count(), os.getloadavg()[0]
    host = available >= 36*1024**3 and load <= 1.25*cpus and io < 20
    return dict(at=now(), host_ready=host, gpu_ready=host and free>=12288 and not pids,
                gpu_uuid=uuid, free_mib=free, compute_pids=pids, cpus=cpus, load1=load,
                mem_available_bytes=available, io_full_avg10_percent=io)


def admission(root, job, name, gpu):
    count = 0
    while True:
        row = snapshot()
        if row["gpu_uuid"] != GPU_UUID:
            raise RuntimeError("Physical GPU UUID differs from the frozen plan")
        count = count+1 if row["gpu_ready" if gpu else "host_ready"] else 0
        job.update(status="waiting_resources", current=name)
        job.setdefault("resource_checks", []).append(row)
        job["resource_checks"] = job["resource_checks"][-60:]
        save(root/"job.json",job)
        if count >= (2 if gpu else 1):
            return
        time.sleep(10 if count else 30)


def execute(root):
    validate_host()
    root = Path(root).resolve()
    plan, job = read(root/"plan.json"), read(root/"job.json")
    if job.get("status") != "prepared" or job.get("stages"):
        raise RuntimeError("No restart, resume or duplicate launch is permitted")
    with (root/"execution.lock").open("x") as stream:
        stream.write(str(os.getpid())+"\n")
    started, cpu_start = time.perf_counter(), cpu_seconds()
    job.update(parent_pid=os.getpid(), started_at=now(), status="running")
    save(root/"launch.json",dict(host=socket.gethostname(),pid=os.getpid(),at=now(),
         code_root=plan["code_root"],plan_sha256=sha(root/"plan.json"),
         actual_nice=os.getpriority(os.PRIO_PROCESS,0),cpu_threads=1,
         invocation=sys.argv,policy=plan["policy"],duplicate_launch_permitted=False),exclusive=True)
    try:
        unchanged(plan)
        for name, cmd, gpu in commands(root, Path(plan["code_root"])):
            admission(root,job,name,gpu)
            unchanged(plan)
            row=dict(name=name,status="running",started_at=now(),command=cmd,returncode=None)
            job["stages"].append(row)
            job.update(status="running",current=name)
            save(root/"job.json",job)
            stage_start=time.perf_counter()
            with (root/"logs"/(name+".log")).open("x") as stream:
                child=subprocess.Popen(cmd,cwd=plan["code_root"],stdout=stream,stderr=subprocess.STDOUT)
                row["pid"]=child.pid
                save(root/"job.json",job)
                rc=child.wait()
            row.update(returncode=rc,status="complete" if rc==0 else "failed",
                       finished_at=now(),wall_seconds=time.perf_counter()-stage_start)
            job["cpu_seconds"]=cpu_seconds()-cpu_start
            job["directory_bytes"]=directory_bytes(root)
            save(root/"job.json",job)
            unchanged(plan)
            if rc:
                stage_root=root/({"frozen_C_statistics":"cache","five_Q_only_arms":"training"}.get(name,"report"))
                status=read(stage_root/"status.json") if (stage_root/"status.json").exists() else {}
                if status.get("status") == "budget_hold":
                    row["status"]="budget_hold"
                    job.update(status="budget_hold",requires_user_action=True,returncode=rc,
                               error=status.get("error", "诊断预算门控；不缩短或重跑"))
                    save(root/"job.json",job)
                    return 3
                raise RuntimeError(f"{name}: rc={rc}, status={status}")
            if (gpu and row["wall_seconds"]>120) or job["cpu_seconds"]>1000 or job["directory_bytes"]>1024**3:
                job.update(status="budget_hold",requires_user_action=True,
                           error="诊断预算达到上限；不缩短训练或重跑")
                save(root/"job.json",job)
                return 3
        public={}
        for folder in ("cache","training","report"):
            for path in sorted((root/folder).rglob("*")):
                if path.is_file() and path.suffix in (".json",".npz",".png"):
                    public[str(path.relative_to(root))]=dict(sha256=sha(path),bytes=path.stat().st_size)
        job.update(status="complete",workflow_complete=True,current=None,finished_at=now(),
                   returncode=0,wall_seconds=time.perf_counter()-started,
                   cpu_seconds=cpu_seconds()-cpu_start,public_files=public,
                   source_fingerprints_unchanged=True,error=None,requires_user_action=False)
        save(root/"job.json",job)
        return 0
    except Exception as exc:
        job.update(status="failed",error=str(exc),requires_user_action=True,finished_at=now(),
                   returncode=1,cpu_seconds=cpu_seconds()-cpu_start)
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
