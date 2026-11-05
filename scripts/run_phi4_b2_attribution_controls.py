"""冻结 b2 源实验的 C／解码器／R 归因诊断，禁止覆盖、自动重试或采样。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import resource
import socket
import subprocess
import sys
import time

from scripts.run_gaussian_scalar_qcr import save, sha, source_hashes, directory_bytes


HOST = "thinklab-105-225"
GPU_INDEX = 2
GPU_UUID = "GPU-713b07d0-118c-f939-c0a8-565390215e01"
SOURCE = Path("/mnt/nas-new/home/yange/ham/results/phi4_b2_closure_comparison_L32_seed91_vm225_20261027")
SOURCE_PLAN_SHA = "994abc7ae83b62bb661de99111ec0fb2712124e06d3d73a9f167543fc55f3c08"
MODULE = "scripts.run_phi4_b2_attribution_controls"
RECIPE = "phi4_b2_frozen_attribution_parent_v1"
CPU_SECONDS, GPU_SECONDS, DIRECTORY_BYTES = 900., 240., 1024 ** 3


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def cpu_seconds():
    own, child = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + child.ru_utime + child.ru_stime


def validate_host():
    if socket.gethostname() != HOST or "CUDA_VISIBLE_DEVICES" in os.environ:
        raise RuntimeError("Host must be vm225 with unmasked physical GPU numbering")


def source_inputs(source=SOURCE):
    source = Path(source).resolve()
    if source != SOURCE:
        raise ValueError("Only the explicitly authorized completed b2 source is allowed")
    job = read(source / "job.json")
    if (job.get("status") != "complete" or job.get("workflow_complete") is not True
            or job.get("error") is not None or len(job.get("stages", [])) != 5
            or any(s.get("returncode") != 0 for s in job["stages"])):
        raise RuntimeError("Source workflow is not successfully complete")
    if sha(source / "plan.json") != SOURCE_PLAN_SHA:
        raise RuntimeError("Source identity/plan SHA mismatch")
    paths = [source / "job.json"]
    for name, digest in job["public_files"].items():
        relative = Path(name)
        path = source / relative
        if (relative.is_absolute() or ".." in relative.parts or path.is_symlink()
                or not path.resolve().is_relative_to(source)
                or sha(path) != digest):
            raise RuntimeError("Source public inventory changed or unsafe")
        paths.append(path)
    summary, manifest = read(source / "training/summary.json"), read(source / "training/manifest.json")
    if (summary.get("actual_steps_by_arm") != {"legacy": 3000, "all_start": 3000}
            or summary.get("C_Q_pairing_verified") is not True
            or manifest.get("recipe") != "phi4_b2_closure_objective_comparison_v1"
            or manifest.get("lattice_sizes") != [32, 16, 8, 4]
            or manifest.get("labels_used_in_training") is not False):
        raise RuntimeError("Source representation/pairing/step contract mismatch")
    for arm in ("legacy", "all_start"):
        paths.append(source / "training" / arm / "training_step_3000.pt")
    return {str(path): sha(path) for path in paths}


def commands(root, pins=None):
    root = Path(root)
    result = [
        ("extract_R_state", [sys.executable, "-u", "-m", MODULE, "--extract-r-state", "--output", str(root)], False),
        ("decoder_controls", [sys.executable, "-u", "-m", "scripts.diagnose_phi4_b2_decoders",
            "--training", str(SOURCE / "training"), "--output", str(root / "decoders")], False),
        ("frozen_R_controls", [sys.executable, "-u", "-m", "experiments.phi4_b2_frozen_r_diagnostic",
            "--training", str(SOURCE / "training"), "--r-state", str(root / "R_only_state.pt"),
            "--output", str(root / "frozen_R")], False),
        ("frozen_C_diagnostic", [sys.executable, "-u", "-m", "experiments.phi4_b2_c_diagnostic",
            "--source", str(SOURCE), "--output", str(root / "frozen_C"), "--device", "cuda:2",
            "--expected-gpu-uuid", GPU_UUID], True),
    ]
    checkpoint = str(SOURCE / "training/all_start/training_step_3000.pt")
    if pins and checkpoint in pins:
        result[-1][1].extend(["--expected-checkpoint-sha256",pins[checkpoint]])
    return result


def prepare(root):
    validate_host()
    root, code = Path(root).resolve(), Path(__file__).resolve().parents[1]
    if root.exists() or root.is_symlink():
        raise FileExistsError("No overwrite or resume")
    for protected in (SOURCE, code):
        if root == protected or root in protected.parents or protected in root.parents:
            raise ValueError("Output must be independent of protected source/code directories")
    pins = source_inputs()
    plan = dict(schema_version=5, recipe=RECIPE, host=HOST, source=str(SOURCE),
        output=str(root), code_root=str(code), prepared_at=now(), source_sha256=source_hashes(code),
        immutable_inputs=pins, device="cuda:2", gpu_uuid=GPU_UUID,
        authorization="用户明确要求对照检查数据、C、Q表示约束与R拟合的归因；不是新的完整QCR训练",
        controls=dict(decoders="同一冻结特征；只在TRAIN拟合，非线性阶数仅TRAIN整条件交叉验证",
            R="固定原始Q/C缓存编码；两源R继续及随机R各3000新Adam步，不更新Q/C",
            C="冻结C；原路径重放及仅均值奇对称化反事实，共用Gaussian噪声，不选择或训练新核",
            per_scale_mapping="TRAIN拟合各转移PCHIP，非统一RG流，不进行物理认证"),
        resource_policy=dict(cpu_threads=1, nice=10, cpu_seconds=CPU_SECONDS, gpu_stage_seconds=GPU_SECONDS,
            cuda_peak_bytes=8*1024**3, directory_bytes=DIRECTORY_BYTES, minimum_host_available_gib=36,
            gpu_free_mib=12288, no_compute_processes=True, gpu_checks_interval_seconds=10,
            max_load_per_cpu=1.25, io_full_avg10_percent_max=20),
        commands=[dict(name=n, command=c, gpu=g) for n,c,g in commands(root,pins)],
        forbidden=["新物理采样", "Q/C参数更新", "MI计算", "修改源文件或默认配方", "TEST选模", "自动修复/重跑/迁移", "新根搜索或热指数认证"],
        limits=["原始35条采样警告保留，不认证平衡", "固定缓存R诊断不是随机物理批次的原QCR严格续训",
            "PCA/多项式重构只是特征几何对照，不代表物理有效维度或新Q学习成功",
            "对称化C反事实并不自动保持物理分布或正确临界性质", "critical_r=null; thermal_nu=null"])
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    save(root / "plan.json", plan, exclusive=True)
    save(root / "job.json", dict(status="prepared", workflow_complete=False, current=None,
        stages=[], error=None, requires_user_action=False, returncode=None), exclusive=True)
    return dict(status="prepared", source_files=len(plan["source_sha256"]), output=str(root))


def unchanged(plan):
    if source_hashes(plan["code_root"]) != plan["source_sha256"]:
        raise RuntimeError("Frozen diagnostic code changed")
    if source_inputs() != plan["immutable_inputs"]:
        raise RuntimeError("Completed scientific source changed")


def extract_r_state(root):
    from experiments.phi4_qcr import state_fingerprint
    from recurrent_rg.tensor_io import load_cpu_tensors
    import torch
    root = Path(root).resolve()
    plan = read(root / "plan.json")
    unchanged(plan)
    if (root / "R_only_state.pt").exists():
        raise FileExistsError("R-only extraction must not be overwritten")
    summary = read(SOURCE / "training/summary.json")
    arms = {}
    for arm, mode in (("legacy", "one_step_and_rollout"), ("all_start", "all_start_anchor")):
        path = SOURCE / "training" / arm / "training_step_3000.pt"
        payload = load_cpu_tensors(path)
        if payload["step"] != 3000 or payload["closure_objective"] != mode:
            raise RuntimeError("Checkpoint endpoint/mode mismatch")
        state = {key[2:]: value.detach().cpu().clone() for key,value in payload["model"].items() if key.startswith("R.")}
        model = torch.nn.Sequential(torch.nn.Linear(1,96),torch.nn.Tanh(),
            torch.nn.Linear(96,96),torch.nn.Tanh(),torch.nn.Linear(96,1)).double()
        model.load_state_dict(state,strict=True)
        if not state or state_fingerprint(model) != summary["arms"][arm]["final_module_sha256"]["R"]:
            raise RuntimeError("Extracted R tensor fingerprint differs from completed source")
        arms[arm] = dict(state_dict=state, source_checkpoint_sha256=sha(path),
            source_checkpoint_path=str(path), step=3000, closure_objective=mode)
    payload = dict(recipe="phi4_b2_R_only_extraction_v1", arms=arms,
        source_manifest_sha256=sha(SOURCE / "training/manifest.json"),
        source_summary_sha256=sha(SOURCE / "training/summary.json"))
    torch.save(payload, root / "R_only_state.pt")
    unchanged(plan)
    receipt = dict(status="complete", workflow_complete=True, returncode=0,
        Q_C_loaded_for_training=False, Q_C_updates=0, parameters_extracted="R only", source_steps=3000,
        R_only_state_sha256=sha(root / "R_only_state.pt"),
        source_checkpoint_sha256={k:v["source_checkpoint_sha256"] for k,v in arms.items()})
    save(root / "extraction.json", receipt, exclusive=True)
    return receipt


def snapshot(gpu=False):
    available = int(re.search(r"MemAvailable:\s+(\d+)", Path("/proc/meminfo").read_text())[1])*1024
    io = float(re.search(r"full avg10=([\d.]+)", Path("/proc/pressure/io").read_text())[1])
    load, cpus = os.getloadavg()[0], os.cpu_count()
    ready = available >= 36*1024**3 and io < 20 and load <= 1.25*cpus
    row = dict(at=now(), mem_available_bytes=available, io_full_avg10_percent=io, load1=load, cpus=cpus, ready=ready)
    if gpu:
        query = subprocess.check_output(["nvidia-smi", "-i", str(GPU_INDEX),
            "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"], text=True).strip().split(",")
        uuid, free = query[0].strip(), int(query[1])
        if uuid != GPU_UUID:
            raise RuntimeError("Physical GPU UUID mismatch")
        lines = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits"], text=True).splitlines()
        pids = [int(s.split(",")[1]) for s in lines if GPU_UUID in s]
        row.update(gpu_uuid=uuid, gpu_free_mib=free, compute_pids=pids, ready=ready and free>=12288 and not pids)
    return row


def admission(root, job, name, gpu):
    count = 0
    while True:
        row = snapshot(gpu)
        job.update(status="waiting_resources", current=name)
        job.setdefault("resource_checks", []).append(row)
        job["resource_checks"] = job["resource_checks"][-60:]
        save(root / "job.json", job)
        count = count+1 if row["ready"] else 0
        if count >= (2 if gpu else 1):
            return
        time.sleep(10 if count else 30)


def execute(root):
    validate_host()
    root = Path(root).resolve()
    plan, job = read(root / "plan.json"), read(root / "job.json")
    if job["status"] != "prepared" or job["stages"]:
        raise RuntimeError("Already executed; no automatic restart")
    unchanged(plan)
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        os.nice(10-os.getpriority(os.PRIO_PROCESS, 0))
    start, cpu_start, gpu_seconds = time.monotonic(), cpu_seconds(), 0.
    job.update(parent_pid=os.getpid(), actual_nice=os.getpriority(os.PRIO_PROCESS,0), started_at=now())
    execution = dict(status="running", parent_pid=os.getpid(), plan_sha256=sha(root/"plan.json"), started_at=now())
    save(root / "execution.json", execution, exclusive=True)
    try:
        for name, command, gpu in commands(root,plan["immutable_inputs"]):
            unchanged(plan)
            if cpu_seconds()-cpu_start >= CPU_SECONDS or directory_bytes(root)>DIRECTORY_BYTES:
                raise TimeoutError("Fixed CPU/directory budget reached")
            admission(root, job, name, gpu)
            env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                NUMBA_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", CUDA_DEVICE_ORDER="PCI_BUS_ID",
                MPLCONFIGDIR=str(root/"matplotlib_cache"), PYTHONUNBUFFERED="1")
            if not gpu:
                env["CUDA_VISIBLE_DEVICES"]=""
            stage_start = time.monotonic()
            with (root/"logs"/(name+".log")).open("x") as log:
                child = subprocess.Popen(command, cwd=plan["code_root"], env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                stage = dict(name=name, command=command, pid=child.pid, gpu=gpu, status="running", started_at=now())
                job["stages"].append(stage);job.update(status="running", current=name)
                save(root/"job.json", job)
                code = child.wait()
            elapsed=time.monotonic()-stage_start
            if gpu:gpu_seconds+=elapsed
            stage.update(returncode=code,status="complete" if code==0 else "failed",ended_at=now(),wall_seconds=elapsed)
            save(root/"job.json",job)
            if code:
                if code==3:raise TimeoutError(name+" budget hold; no retry")
                raise RuntimeError(name+" returned "+str(code)+"; no retry")
            artifact = root/"extraction.json" if name=="extract_R_state" else root/{
                "decoder_controls":"decoders", "frozen_R_controls":"frozen_R", "frozen_C_diagnostic":"frozen_C"}[name]/"status.json"
            receipt = read(artifact)
            if (receipt.get("status")!="complete" or receipt.get("workflow_complete") is not True
                    or receipt.get("returncode")!=0):
                raise RuntimeError(name+" has incomplete or inconsistent completion receipt")
            unchanged(plan)
            if cpu_seconds()-cpu_start>CPU_SECONDS or gpu_seconds>GPU_SECONDS or directory_bytes(root)>DIRECTORY_BYTES:
                raise TimeoutError("Actual fixed resource budget exceeded; artifacts preserved")
        job.update(status="complete", workflow_complete=True, current=None, returncode=0, ended_at=now())
    except Exception as error:
        held=isinstance(error,TimeoutError)
        job.update(status="budget_hold" if held else "failed", workflow_complete=False,
            returncode=3 if held else 1,error=f"{type(error).__name__}: {error}",requires_user_action=True,ended_at=now())
    finally:
        job.update(cpu_seconds=cpu_seconds()-cpu_start,gpu_stage_seconds=gpu_seconds,
            wall_seconds=time.monotonic()-start,directory_bytes=directory_bytes(root))
        execution.update({key:job.get(key) for key in ("status","workflow_complete","returncode","ended_at",
            "cpu_seconds","gpu_stage_seconds","wall_seconds","directory_bytes","actual_nice","error","requires_user_action")})
        save(root/"execution.json",execution)
        if job["status"]=="complete":
            job["public_files"]={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob("*"))
                if p.is_file() and not p.is_symlink() and p.suffix in (".json",".npz",".png") and p.name!="job.json"
                and "matplotlib_cache" not in p.relative_to(root).parts}
        save(root/"job.json",job)
    return job


def launch(root):
    validate_host()
    root=Path(root).resolve();plan,job=read(root/"plan.json"),read(root/"job.json")
    if job["status"]!="prepared" or job["stages"] or (root/"execution.json").exists():
        raise RuntimeError("No duplicate launch or resume")
    unchanged(plan)
    with (root/"launch.json").open("x") as receipt,(root/"logs/parent.log").open("x") as log:
        child=subprocess.Popen([sys.executable,"-u","-m",MODULE,"--execute","--output",str(root)],
            cwd=plan["code_root"],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True,env=dict(os.environ,PYTHONUNBUFFERED="1"))
        value=dict(status="launched",parent_pid=child.pid,at=now(),plan_sha256=sha(root/"plan.json"),output=str(root))
        receipt.write(json.dumps(value,ensure_ascii=False,indent=2)+"\n")
    return value


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);m=p.add_mutually_exclusive_group(required=True)
    for mode in ("prepare","launch","execute","extract-r-state"):m.add_argument("--"+mode,action="store_true")
    p.add_argument("--output",type=Path,required=True);args=p.parse_args(argv)
    value=prepare(args.output) if args.prepare else launch(args.output) if args.launch else execute(args.output) if args.execute else extract_r_state(args.output)
    print(json.dumps(value,ensure_ascii=False,allow_nan=False),flush=True)
    return 0 if value["status"] in ("prepared","launched","complete") else 3


if __name__=="__main__":
    raise SystemExit(main())
