"""Fresh, bounded TRAIN-closure early-stop comparison; dry unless --execute.

Reuse only the completed physical data.  No sampler, benchmark, historical
checkpoint load, resume, automatic repair, process kill or GPU migration occurs.
The old 3000-update workflow and frozen code remain immutable evidence.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time

from scripts import run_phi4_hopping_qcr as base


MODULE = "scripts.run_phi4_hopping_early_stopping"
POLICY = "train_closure_100_200_patience20_v1"
GPU_INDEX = 3
GPU_UUID = "GPU-a167decd-ce49-6c8f-6bcf-136fde596683"
CPU_SECONDS, GPU_SECONDS = 86400., 57600.
DIRECTORY_BYTES, WORKFLOW_SECONDS = 2 * 1024**3, 5400.
CORE_SOURCES = (
    "recurrent_rg/phi4_aligned_qcr.py", "recurrent_rg/phi4_qcr.py",
    "recurrent_rg/losses.py", "recurrent_rg/data/phi4_hopping_loader.py",
)
EVIDENCE_FILES = (
    "job.json", "plan.json", "training_runtime_config.json",
    "data/status.json", "data/manifest.json", "data/plan.json",
    "training/status.json", "training/summary.json", "training/manifest.json",
)
PUBLIC_SUFFIXES = {".json", ".png", ".npz"}


def _read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _code_root():
    return Path(__file__).resolve().parents[1]


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid finite nonnegative {name}")
    return float(value)


def _digest(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(x not in "0123456789abcdef" for x in value):
        raise ValueError(f"Invalid SHA256 for {name}")
    return value


def _complete(value, label):
    if (value.get("status") != "complete" or value.get("workflow_complete") is not True
            or type(value.get("returncode")) is not int or value["returncode"] != 0
            or value.get("error") is not None or value.get("requires_user_action") is not False):
        raise RuntimeError(f"{label} is not successfully complete")


def _clocks(value, expected, label):
    if (not isinstance(value, dict) or set(value) != {"Q", "C", "R"}
            or any(not isinstance(row, dict) or row.get("min") != expected
                or row.get("max") != expected or type(row.get("parameters")) is not int
                or row["parameters"] < 1 for row in value.values())):
        raise RuntimeError(f"{label} Q/C/R Adam counters differ")


def _inside(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise RuntimeError("Public path must be nonempty and relative")
    path = (root / relative).resolve()
    if root not in path.parents or path.suffix.lower() not in PUBLIC_SUFFIXES or not path.is_file():
        raise RuntimeError("Public path escapes root or is not JSON/PNG/NPZ")
    return path


def directory_bytes(root):
    """Metadata-only accounting; never read physical arrays or checkpoint bytes."""
    return sum(path.stat().st_size for path in Path(root).rglob("*") if path.is_file())


def budget_ok(cpu, gpu, size, *, remaining=0.):
    values = (cpu, gpu, size, remaining)
    return (all(not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(x) and x >= 0
                for x in values)
            and cpu + remaining <= CPU_SECONDS and gpu + remaining <= GPU_SECONDS
            and size <= DIRECTORY_BYTES)


def source_evidence(source_root, *, code=None):
    """Metadata and pinned source only: no physical-data/model deserialization."""
    source, code = Path(source_root).resolve(), Path(code or _code_root()).resolve()
    rows = {name: _read(source / name) for name in EVIDENCE_FILES}
    job, status, summary, manifest = (rows[name] for name in (
        "job.json", "training/status.json", "training/summary.json", "training/manifest.json"))
    _complete(job, "Old workflow")
    _complete(status, "Old training")
    _complete(rows["data/status.json"], "Physical dataset")
    if (not job.get("stages") or any(row.get("returncode") != 0 or row.get("status") != "complete"
            for row in job["stages"]) or job.get("formal_optimizer_updates") != 3000
            or summary.get("status") != "complete" or summary.get("actual_steps") != 3000
            or status.get("actual_steps") != 3000 or status.get("latest_checkpoint_step") != 3000
            or summary.get("labels_used_in_training") is not False
            or manifest.get("labels_used_in_training") is not False
            or manifest.get("closure_objective") != "all_start_anchor"
            or manifest.get("closure_scale_weights") is not None
            or manifest.get("lattice_sizes") != [128, 64, 32, 16]
            or summary.get("lattice_sizes") != [128, 64, 32, 16]
            or manifest.get("theory_dim") != 1 or summary.get("theory_dim") != 1
            or manifest.get("recipe") != "phi4_hopping_raw_patch_joint625_v1"
            or summary.get("recipe") != manifest.get("recipe")
            or summary.get("thermal_nu") is not None):
        raise RuntimeError("Old completed 3000-update b2/dg1 source contract differs")
    _clocks(summary.get("optimizer_steps"), 3000, "Old")
    pins = {str(source / name): base.sha(source / name) for name in EVIDENCE_FILES}
    if (job.get("plan_sha256") != pins[str(source / "plan.json")]
            or status.get("summary_sha256") != pins[str(source / "training/summary.json")]):
        raise RuntimeError("Old job/plan or training/summary binding differs")
    data_sha = pins[str(source / "data/manifest.json")]
    if (rows["data/status.json"].get("manifest_sha256") != data_sha
            or rows["data/manifest.json"].get("plan_sha256") != pins[str(source / "data/plan.json")]
            or manifest.get("data_manifest_sha256") != data_sha
            or summary.get("data_manifest_sha256") != data_sha
            or job.get("data_manifest_sha256") != data_sha):
        raise RuntimeError("Old physical data/manifest/plan bindings differ")
    initial = _digest(summary.get("q_initialization_receipt", {}).get("training_start_state_sha256"), "initial state")
    if manifest.get("q_initialization_receipt", {}).get("training_start_state_sha256") != initial:
        raise RuntimeError("Old initialization receipts disagree")
    old_sources = manifest.get("source_sha256")
    if not isinstance(old_sources, dict) or not old_sources:
        raise RuntimeError("Old pinned training sources are absent")
    for path, digest in old_sources.items():
        if base.sha(path) != _digest(digest, path):
            raise RuntimeError("Immutable old source fingerprint differs")
        pins[path] = digest
    for relative in CORE_SOURCES:
        matches = [digest for path, digest in old_sources.items() if path.endswith("/" + relative)]
        if len(matches) != 1 or base.sha(code / relative) != matches[0]:
            raise RuntimeError(f"Core model/data/loss differs from old run: {relative}")
    options = manifest.get("options", {})
    fixed = dict(steps=3000, system_batch=32, sample_batch=64, seed=91, threads=1,
                 information_pairs=128, block_size=2, depth=3, max_cuda_bytes=base.PEAK_BYTES)
    if any(options.get(key) != value for key, value in fixed.items()):
        raise RuntimeError("Old model/batch/seed/gradient recipe differs")
    runtime = rows["training_runtime_config.json"]
    from experiments.phi4_hopping_qcr import LOSS_POLICY, model_policy
    if (runtime.get("loss") != LOSS_POLICY or runtime.get("model") != model_policy(2, 3)
            or any(runtime.get("training", {}).get(key) != value for key, value in fixed.items()
                   if key not in {"block_size", "depth"})
            or Path(runtime.get("data", "")).resolve() != source / "data"):
        raise RuntimeError("Old runtime configuration differs from its training manifest")
    prior_cpu = _number(job.get("cpu_seconds"), "prior CPU seconds")
    prior_gpu = _number(job.get("gpu_seconds"), "prior GPU seconds")
    recorded_bytes = _number(job.get("directory_bytes"), "prior directory bytes")
    recorded_new = _number(job.get("new_directory_bytes"), "prior new-root bytes")
    prior_bytes = recorded_bytes + max(0., directory_bytes(source) - recorded_new)
    train_seconds = _number(summary.get("timing", {}).get("training_seconds"), "old training seconds")
    if train_seconds <= 0:
        raise RuntimeError("Old actual training timing is absent")
    projection = train_seconds / 3000. * 200. * 1.5 + 120.
    return dict(source_root=str(source), data_root=str(source / "data"),
        data_manifest_sha256=data_sha, initial_state_sha256=initial,
        prior_cpu_seconds=prior_cpu, prior_gpu_seconds=prior_gpu, prior_directory_bytes=int(prior_bytes),
        old_training_seconds=train_seconds, projection_seconds=projection,
        estimate_not_guarantee=True, evidence_sha256=pins,
        data_warning_count=manifest.get("data_warning_count"), sampling_reused=True,
        physical_data_deserialized=False, old_model_loaded=False)


def _configuration(source, output, evidence):
    configuration = copy.deepcopy(_read(source / "training_runtime_config.json"))
    configuration.update(data=evidence["data_root"], output=str(output / "training"))
    configuration["training"].update(steps=200, log_every=10, checkpoint_every=100,
        max_training_seconds=WORKFLOW_SECONDS, early_stopping=POLICY,
        device="cuda:0", expected_gpu_uuid=GPU_UUID,
        expected_initial_state_sha256=evidence["initial_state_sha256"])
    return configuration


def plan(source_root, output):
    from experiments.phi4_hopping_qcr import TrainingOptions, training_plan
    source, root, code = Path(source_root).resolve(), Path(output).resolve(), _code_root()
    if (root.exists() or root == source or source in root.parents or root in source.parents
            or root == code or code in root.parents or root in code.parents):
        raise ValueError("Require a fresh output separate from immutable source and code")
    evidence = source_evidence(source, code=code)
    configuration = _configuration(source, root, evidence)
    training_values = dict(configuration["training"], data=configuration["data"],
                           output=configuration["output"], block_size=2, depth=3)
    training = training_plan(TrainingOptions(**training_values))
    return dict(schema_version=5, created_at=base.now(), execution_requested=False,
        host=base.HOST, code_root=str(code), source_root=str(source), output=str(root),
        GPU=dict(index=GPU_INDEX, uuid=GPU_UUID), prior_evidence=evidence,
        training_config=configuration, training=training,
        command=[sys.executable, "-u", "-m", "recurrent_rg", "train", "--config",
            str(root / "training_config.json"), "--output", str(root / "training"), "--device", "cuda:0"],
        stages=["training"], resources=dict(cpu_seconds=CPU_SECONDS, gpu_card_seconds=GPU_SECONDS,
            directory_bytes=DIRECTORY_BYTES, workflow_seconds=WORKFLOW_SECONDS,
            directory_scope="old cumulative job baseline plus new root; shared data not duplicated",
            cpu_threads=1, nice=10, max_cuda_bytes=base.PEAK_BYTES, min_memory_bytes=36*1024**3,
            max_load_per_cpu=1.25, max_io_full_avg10=20., idle_check_spacing_seconds=10.,
            min_gpu_free_bytes=12*1024**3, require_no_compute_processes=True),
        training_budget_admission=dict(projection_seconds=evidence["projection_seconds"],
            projection_basis="old actual seconds / 3000 * 200 * 1.5 + 120 seconds",
            workflow_cap_seconds=WORKFLOW_SECONDS, boundary_reserve_seconds=60.,
            projection_is_estimate_not_guarantee=True),
        no_sampling=True, no_benchmark=True, no_resume=True, no_retry=True,
        no_auto_repair=True, no_migration=True, old_workflow_preserved=True,
        qualification="固定TRAIN探针早停；Q/C目标随训练变化，不证明过拟合或物理临界点；仅已检查点最低，不保证整个窗口全局最低")


def verify_pins(pins):
    if any(base.sha(path) != digest for path, digest in pins.items()):
        raise RuntimeError("Pinned source/config/evidence changed; no automatic repair")


def validate_training(root, evidence):
    folder = Path(root).resolve() / "training"
    status, summary, manifest, early = [_read(folder / name) for name in
        ("status.json", "summary.json", "manifest.json", "early_stopping.json")]
    _complete(status, "New training")
    actual, selected = summary.get("actual_steps"), summary.get("evaluated_checkpoint_step")
    if (type(actual) is not int or not 100 <= actual <= 200 or actual % 10
            or type(selected) is not int or not 100 <= selected <= actual or selected % 10
            or status.get("actual_steps") != actual or summary.get("status") != "complete"
            or status.get("latest_checkpoint_step") != actual
            or summary.get("latest_checkpoint_step") != actual
            or manifest.get("options", {}).get("early_stopping") != POLICY
            or manifest.get("options", {}).get("steps") != 200
            or manifest.get("options", {}).get("expected_initial_state_sha256") != evidence["initial_state_sha256"]
            or manifest.get("closure_objective") != "all_start_anchor"
            or manifest.get("closure_scale_weights") is not None
            or manifest.get("lattice_sizes") != [128, 64, 32, 16]
            or summary.get("labels_used_in_training") is not False
            or manifest.get("labels_used_in_training") is not False
            or early != summary.get("early_stopping")
            or early.get("actual_steps") != actual or early.get("best_step") != selected
            or early.get("evaluated_checkpoint_step") != selected
            or early.get("labels_used") is not False or early.get("test_data_used") is not False
            or summary.get("thermal_nu") is not None
            or summary.get("frozen_evaluation_verified") is not True
            or _number(summary.get("peak_cuda_bytes"), "new own CUDA peak") > base.PEAK_BYTES):
        raise RuntimeError("New early-stop endpoint/label/frozen-evaluation contract differs")
    _digest(early.get("selected_checkpoint_sha256"), "selected checkpoint")
    selected_path = folder / f"training_step_{selected}.pt"
    if base.sha(selected_path) != early["selected_checkpoint_sha256"]:
        raise RuntimeError("Selected checkpoint bytes differ from the trainer receipt")
    _clocks(summary.get("optimizer_steps"), actual, "Actual")
    _clocks(summary.get("selected_optimizer_steps"), selected, "Selected")
    for value in (summary, manifest):
        if (value.get("data_manifest_sha256") != evidence["data_manifest_sha256"]
                or value.get("q_initialization_receipt", {}).get("training_start_state_sha256") != evidence["initial_state_sha256"]):
            raise RuntimeError("New data or fresh initialization binding differs")
    if (status.get("summary_sha256") != base.sha(folder / "summary.json")
            or status.get("evaluation_arrays_sha256") != base.sha(folder / "evaluation_arrays.npz")):
        raise RuntimeError("New summary/array binding differs")
    public = {}
    for row in status.get("public_files", []):
        path = _inside(folder, row.get("path"))
        if base.sha(path) != row.get("sha256"):
            raise RuntimeError("New public-file SHA mismatch")
        public[str(path.relative_to(folder))] = row["sha256"]
    required = {"manifest.json", "summary.json", "history.json", "plan.json", "early_stopping.json",
        "evaluation_arrays.npz", "report/report.json", "report/R_vs_g.png",
        "report/coupling_by_scale.png", "report/loss_curves.png", "report/all_start_closure.png"}
    if not required.issubset(public):
        raise RuntimeError("New formal report/public artifact set is incomplete")
    report = _read(folder / "report/report.json")
    if (report.get("status") != "complete" or report.get("actual_steps") != actual
            or report.get("evaluated_checkpoint_step") != selected
            or report.get("early_stopping") != early
            or any(report.get("source_sha256", {}).get(name) != public[name]
                for name in ("manifest.json", "summary.json", "history.json", "evaluation_arrays.npz"))):
        raise RuntimeError("New report endpoint or input fingerprints differ")
    return dict(actual_steps=actual, evaluated_checkpoint_step=selected,
                public_files=public, selected_checkpoint_sha256=early["selected_checkpoint_sha256"])


def _public_files(root, *, completed=False):
    public = {name: base.sha(root / name) for name in
              ("job.json", "plan.json", "training_config.json", "prior_evidence.json")}
    status_path = root / "training/status.json"
    if status_path.is_file():
        public["training/status.json"] = base.sha(status_path)
    if completed:
        for row in _read(status_path)["public_files"]:
            path = _inside(root / "training", row["path"])
            if base.sha(path) != row["sha256"]:
                raise RuntimeError("Completed public output changed")
            public[f"training/{row['path']}"] = row["sha256"]
    base.save(root / "public_files.json", public)


def run(source_root, output):
    definition = plan(source_root, output)
    if socket.gethostname() != base.HOST:
        raise RuntimeError("Wrong host; no GPU migration")
    if os.getpriority(os.PRIO_PROCESS, 0) != 10:
        raise RuntimeError("Launch parent with exactly nice 10; no online renice")
    from experiments.phi4_hopping_qcr import source_pins
    evidence, root, code = definition["prior_evidence"], Path(output).resolve(), _code_root()
    pins = dict(evidence["evidence_sha256"])
    pins.update(source_pins())
    for path in (Path(__file__).resolve(), Path(base.__file__).resolve(), code / "recurrent_rg/cli.py"):
        pins[str(path)] = base.sha(path)
    verify_pins(pins)
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    for name, value in (("training_config.json", definition["training_config"]),
                        ("prior_evidence.json", evidence)):
        base.save(root / name, value)
        pins[str(root / name)] = base.sha(root / name)
    base.save(root / "plan.json", dict(definition, execution_requested=True, source_sha256=pins))
    pins[str(root / "plan.json")] = base.sha(root / "plan.json")
    started_cpu, child_start = time.process_time(), resource.getrusage(resource.RUSAGE_CHILDREN)
    job = dict(status="prepared", workflow_complete=False, parent_pid=os.getpid(), current=None,
        returncode=None, error=None, requires_user_action=False, started_at=base.now(), stages=[],
        plan_sha256=base.sha(root / "plan.json"), formal_training_started=False,
        formal_optimizer_updates=0, additional_sampling=0, additional_benchmark_updates=0,
        prior_cpu_seconds=evidence["prior_cpu_seconds"], prior_gpu_seconds=evidence["prior_gpu_seconds"],
        old_workflow_preserved=True)
    active, active_stage, gpu_seconds = None, None, evidence["prior_gpu_seconds"]

    def checkpoint():
        child = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu = evidence["prior_cpu_seconds"] + time.process_time() - started_cpu
        cpu += child.ru_utime - child_start.ru_utime + child.ru_stime - child_start.ru_stime
        if active is not None and active.poll() is None:
            try:
                fields = Path(f"/proc/{active.pid}/stat").read_text().split(")", 1)[1].split()
                cpu += (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
            except (FileNotFoundError, ProcessLookupError):
                pass
        job.update(cpu_seconds=cpu, gpu_seconds=gpu_seconds + (
            time.monotonic() - active_stage["monotonic_start"] if active is not None and active.poll() is None else 0.),
            new_directory_bytes=directory_bytes(root))
        job["directory_bytes"] = evidence["prior_directory_bytes"] + job["new_directory_bytes"]
        base.save(root / "job.json", job)

    def admitted(remaining=0.):
        return budget_ok(job["cpu_seconds"], job["gpu_seconds"], job["directory_bytes"], remaining=remaining)

    def hold(message):
        job.update(status="budget_hold", workflow_complete=False, current=None, returncode=3,
            error=message, requires_user_action=True, ended_at=base.now())
        checkpoint()
        _public_files(root)
        return 3

    def child_status():
        try:
            value = _read(root / "training/status.json")
        except (FileNotFoundError, json.JSONDecodeError):
            return
        active_stage["child_status"] = value
        job["formal_optimizer_updates"] = value.get("actual_steps", 0)

    checkpoint()
    try:
        from scripts.report_phi4_hopping_qcr import select_chinese_font
        job["chinese_font"] = str(select_chinese_font(os.environ.get("QCR_ZH_FONT")))
        if evidence["projection_seconds"] > WORKFLOW_SECONDS:
            return hold("Old actual timing conservative projection exceeds the new workflow cap")
        job.update(status="waiting_resources", current="training")
        while True:
            verify_pins(pins)
            first = dict(host=base.host_resources(), gpu=base.gpu_resources(GPU_INDEX, GPU_UUID))
            job["resources"] = first
            checkpoint()
            if not admitted(WORKFLOW_SECONDS + 60.):
                return hold("Cumulative CPU/GPU/directory budget cannot admit bounded early-stop workflow")
            if not base.ready(first, True):
                time.sleep(30)
                continue
            check_started = time.monotonic()
            time.sleep(10)
            second = dict(host=base.host_resources(), gpu=base.gpu_resources(GPU_INDEX, GPU_UUID))
            spacing = time.monotonic() - check_started
            gpu_seconds += spacing
            job["gpu_idle_checks"] = dict(first=first, second=second, spacing_seconds=spacing)
            checkpoint()
            if not admitted(WORKFLOW_SECONDS + 60.):
                return hold("Resource checks consumed remaining admission allowance")
            if spacing >= 10. and base.ready(second, True):
                break
        verify_pins(pins)
        active_stage = dict(name="training", status="running", returncode=None,
            started_at=base.now(), monotonic_start=time.monotonic(), command=definition["command"])
        job["stages"].append(active_stage)
        job.update(status="running", current="training", formal_training_started=True)
        with (root / "logs/training.log").open("xb") as log:
            active = subprocess.Popen(definition["command"], cwd=code,
                env=base.child_environment(GPU_UUID), stdout=log, stderr=subprocess.STDOUT)
            active_stage["pid"] = active.pid
            checkpoint()
            while active.poll() is None:
                verify_pins(pins)
                child_status()
                checkpoint()
                if not admitted():
                    job.update(requires_user_action=True, budget_exceeded_with_child_running=True,
                        budget_note="Child retained under own cap; no kill, restart or next stage")
                time.sleep(10)
            rc = active.returncode
        child_status()
        elapsed = time.monotonic() - active_stage["monotonic_start"]
        gpu_seconds += elapsed
        state = "complete" if rc == 0 else "budget_hold" if rc == 3 else "failed"
        active_stage.update(status=state, returncode=rc, ended_at=base.now(), wall_seconds=elapsed)
        active = None
        checkpoint()
        verify_pins(pins)
        if rc:
            job.update(status=state, workflow_complete=False, current=None, returncode=rc,
                error=f"training exited {rc}; all completed outputs preserved", requires_user_action=True,
                ended_at=base.now())
            checkpoint()
            _public_files(root)
            return rc
        result = validate_training(root, evidence)
        job.update(formal_optimizer_updates=result["actual_steps"],
            evaluated_checkpoint_step=result["evaluated_checkpoint_step"],
            selected_checkpoint_sha256=result["selected_checkpoint_sha256"])
        if not admitted():
            return hold("Completed early-stop workflow exceeds cumulative budget")
        job.update(status="complete", workflow_complete=True, current=None, returncode=0,
                   error=None, requires_user_action=False, ended_at=base.now())
        checkpoint()
        _public_files(root, completed=True)
        return 0
    except BaseException as exc:
        if active is not None and active.poll() is None:
            job.update(status="waiting_running_child_after_parent_error", requires_user_action=True,
                       error=f"{type(exc).__name__}: {exc}; existing child retained")
            checkpoint()
            active.wait()  # Never terminate, restart or migrate this child.
        if active is not None and active_stage is not None:
            elapsed = time.monotonic() - active_stage["monotonic_start"]
            gpu_seconds += elapsed
            child_status()
            active_stage.update(status="child_exited_after_parent_error", returncode=active.returncode,
                                ended_at=base.now(), wall_seconds=elapsed)
            active = None
        job.update(status="failed", workflow_complete=False, current=None, returncode=1,
            requires_user_action=True, error=f"{type(exc).__name__}: {exc}", ended_at=base.now())
        checkpoint()
        _public_files(root)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.execute:
        return run(args.source_root, args.output)
    print(json.dumps(plan(args.source_root, args.output), indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
