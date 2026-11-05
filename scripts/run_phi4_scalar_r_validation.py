"""Exclusive, CPU-only fixed-u1 frozen-QC scalar-R validation workflow."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import socket
import subprocess
import sys
import time


HOST = "thinklab-105-225"
SOURCE = "/mnt/nas-new/home/yange/ham/results/phi4_u1_rg_reference_b3_g1_seed91_vm225_20261028"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = Path(str(path) + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def now():
    return datetime.now(timezone.utc).isoformat()


def host_resources():
    available = int(re.search(r"MemAvailable:\s+(\d+)", Path("/proc/meminfo").read_text())[1])*1024
    io = float(re.search(r"full avg10=([\d.]+)", Path("/proc/pressure/io").read_text())[1])
    return dict(at=now(), mem_available_bytes=available, load1=os.getloadavg()[0],
                cpus=os.cpu_count(), io_full_avg10=io)


def run(output):
    if socket.gethostname() != HOST:
        raise ValueError("This workflow is pinned to the existing225 host")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU-only workflow requires CUDA_VISIBLE_DEVICES empty")
    root, code = Path(output).resolve(), Path(__file__).resolve().parents[1]
    if root == code or root in code.parents:
        raise ValueError("Output must not replace the deployment")
    source = Path(SOURCE).resolve()
    if root == source or source in root.parents or root in source.parents:
        raise ValueError("Output must be separate from the immutable historical source")
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    source_pins = {str(p):sha(p) for folder in ("experiments", "recurrent_rg", "scripts")
                   for p in (code / folder).rglob("*.py")}
    commands = [
        ("frozen_collection", [sys.executable,"-u","-m","experiments.phi4_scalar_r_cache",
            "--source",SOURCE,"--output",str(root / "cache"),"--cpu-budget-seconds","1200"]),
        ("scalar_R_validation", [sys.executable,"-u","-m","experiments.phi4_scalar_r_validation",
            "--cache",str(root / "cache"),"--output",str(root / "training"),"--cpu-budget-seconds","1200"]),
        ("report", [sys.executable,"-u","-m","scripts.report_phi4_scalar_r_validation",
            "--validation",str(root / "training"),"--cache",str(root / "cache"),"--output",str(root / "report")])]
    plan = dict(schema_version=5,created_at=now(),source=SOURCE,output=str(root),code_root=str(code),
        source_sha256=source_pins, stages=[dict(name=k,command=v) for k,v in commands],
        scientific_design=dict(source_QCR_steps=3000,Q_updates=0,C_updates=0,d_g=1,
            fresh_R_seeds=[91,92,93],Adam_steps_per_seed=6000,Adam_lr=.001,
            LBFGS_max_iterations=1000,LBFGS_block=100,patience_blocks=3,relative_improvement=.001,
            condition_split_seed=20261030,R_fit_conditions=6,R_validation_conditions=2,
            original_TEST_conditions=5,QC_has_seen_R_validation_conditions=True,
            fit_encoding_repeats=4,evaluation_encoding_repeats=4,fields_per_encoding=64,
            labels_training=False,new_physical_samples=0,closure_objective="all_start_anchor",
            closure_weights=[5.,2.5,2.5],select_seed_by_TEST=False,thermal_nu=None),
        resources=dict(device="cpu",gpu_seconds=0,cpu_threads=1,nice=10,
            cpu_budget_seconds=3000,output_budget_bytes=256*1024**2,
            min_MemAvailable_GiB=36,max_load_per_cpu=1.25,max_IO_full_avg10=20),
        qualification="冻结QC后的R条件级验证；旧TEST仅事后评价；旧采样警告保留，不认证平衡")
    save(root / "plan.json", plan)
    job = dict(status="prepared",workflow_complete=False,current=None,stages=[],error=None,
        returncode=None,requires_user_action=False,parent_pid=os.getpid(),started_at=now(),
        plan_sha256=sha(root / "plan.json"),cpu_seconds=0.,gpu_seconds=0.)
    save(root / "job.json", job)
    started = time.process_time()

    def cpu():
        children=resource.getrusage(resource.RUSAGE_CHILDREN)
        return time.process_time()-started + children.ru_utime+children.ru_stime

    try:
        if os.getpriority(os.PRIO_PROCESS,0) != 10:
            raise ValueError("Launch exactly once through nice10; do not double-nice children")
        for name, command in commands:
            job.update(current=name,status="waiting_resources")
            while True:
                row=host_resources()
                job.update(resources=row,cpu_seconds=cpu())
                save(root / "job.json",job)
                if (row["mem_available_bytes"]>=36*1024**3 and row["load1"]<=1.25*row["cpus"]
                        and row["io_full_avg10"]<20):
                    break
                time.sleep(30)
            if any(sha(path)!=expected for path,expected in source_pins.items()):
                raise RuntimeError("Frozen deployment source changed")
            if cpu()>2700:
                job.update(status="budget_hold",requires_user_action=True,
                           returncode=3,error="Insufficient fixed CPU budget before next stage")
                save(root / "job.json",job)
                return 3
            stage=dict(name=name,status="running",command=command,started_at=now(),returncode=None)
            job["stages"].append(stage)
            with (root / "logs" / f"{name}.log").open("xb") as log:
                process=subprocess.Popen(command,cwd=code,stdout=log,stderr=subprocess.STDOUT)
                stage["pid"]=process.pid
                job.update(status="running")
                save(root / "job.json",job)
                rc=process.wait()
            stage.update(status="complete" if rc==0 else "failed",returncode=rc,ended_at=now())
            job.update(cpu_seconds=cpu())
            save(root / "job.json",job)
            if rc:
                detail=root / {"frozen_collection":"cache","scalar_R_validation":"training","report":"report"}[name] / "status.json"
                child=json.loads(detail.read_text()) if detail.exists() else {}
                job.update(status=child.get("status","failed"),returncode=rc,
                    error=child.get("error") or f"{name} exited{rc}",requires_user_action=True)
                save(root / "job.json",job)
                return rc
        if cpu()>3000 or sum(p.stat().st_size for p in root.rglob("*") if p.is_file())>256*1024**2:
            raise RuntimeError("Completed workflow exceeded declared total resource budget")
        job.update(status="complete",workflow_complete=True,returncode=0,current=None,
            ended_at=now(),cpu_seconds=cpu())
        save(root / "job.json",job)
        public={}
        for folder in ("cache","training","report"):
            inventory=json.loads((root / folder / "public_files.json").read_text())
            for name,expected in inventory.items():
                path=root / folder / name
                if path.suffix not in (".json",".npz",".png") or sha(path)!=expected:
                    raise RuntimeError("Unexpected or changed public artifact")
                public[f"{folder}/{name}"]=expected
        public.update({name:sha(root / name) for name in ("job.json","plan.json")})
        save(root / "public_files.json",public)
        print(json.dumps(job,ensure_ascii=False),flush=True)
        return 0
    except Exception as exc:
        job.update(status="failed",workflow_complete=False,returncode=1,error=str(exc),
                   requires_user_action=True,cpu_seconds=cpu(),ended_at=now())
        save(root / "job.json",job)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    raise SystemExit(run(args.output))


if __name__ == "__main__":
    main()
