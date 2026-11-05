"""User-authorized I/O-only scheduling override for an idle frozen KB queue.

The frozen source/plan/results stay unchanged. This independent coordinator
records a handoff and runs only the unexecuted suffix, without checkpoint
resumption, repeated smokes, or scientific configuration changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

PHASES = ("cuda_verification", "smoke_dim5", "smoke_dim10", "train_dim5",
          "report_dim5", "train_dim10", "report_dim10")
AUTHORIZATION = "225的I/O压力是什么意思，22.5只比20高一点点吧，直接用呗"
EXPECTED_PARENT_PID = 6412
EXPECTED_PARENT_START_TICKS = "1168097819"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def load_runtime(root):
    plan = read(Path(root) / "plan.json")
    sys.path.insert(0, plan["code_root"])
    from scripts import run_ising_kb_highdim as base
    from scripts import run_ising_highdim_low_ram as guard
    if Path(base.__file__).resolve() != Path(plan["code_root"]) / "scripts/run_ising_kb_highdim.py":
        raise RuntimeError("Runtime is not the exact frozen experiment source")
    if socket.gethostname() != base.HOST:
        raise RuntimeError("Wrong host; no migration")
    base.native_cuda_environment()
    base.unchanged(plan)
    return base, guard, plan


def remaining_phases(job):
    stages = job.get("stages", [])
    names = [s.get("name") for s in stages]
    if (job.get("status") != "waiting_resources" or job.get("workflow_complete")
            or job.get("error") or job.get("requires_user_action")
            or not names or names != list(PHASES[:len(names)]) or len(names) >= len(PHASES)
            or any(s.get("status") != "complete" or s.get("returncode") != 0 for s in stages)
            or job.get("current") != PHASES[len(names)]
            or set(job.get("receipts", {})) != set(names)):
        raise RuntimeError("Require an idle, successful prefix and no active or failed stage")
    return list(PHASES[len(names):])


def admission_ready(row, *, gpu):
    """Only I/O ceases to be an admission veto; all other gates unchanged."""
    host = (row["mem_available_gib"] >= 18 and row["load1"] <= 1.25 * row["cpu_count"]
            and row["disk_free_gib"] >= 20)
    return host and (not gpu or (not row["compute_processes"] and row["gpu_free_mib"] >= 16 * 1024))


def verify_pending(root, plan, names):
    for name in names:
        kind, arm_name = name.split("_", 1)
        arm = next(a for a in plan["arms"] if a["name"] == arm_name)
        target = (root / "smoke" / arm_name / "run" if kind == "smoke" else
                  Path(arm["run"]) if kind == "train" else Path(arm["report"]))
        for path in (target, root / "logs" / f"{name}.log", root / "profiles" / f"{name}.json"):
            if path.exists() or path.is_symlink():
                raise RuntimeError(f"Unexecuted stage already has artifacts: {path}")


def stop_idle_parent(root, job, guard, pending_check=None):
    pid = job["parent_pid"]
    before = guard.process_identity(pid)
    expected_cmd = ["-m", "scripts.run_ising_kb_highdim", "--execute", "--output", str(root)]
    cmd = (Path("/proc") / str(pid) / "cmdline").read_bytes().decode().split("\0")
    if (pid != EXPECTED_PARENT_PID or before is None
            or before["starttime_ticks"] != EXPECTED_PARENT_START_TICKS
            or before["uid"] != os.getuid() or before["session"] != pid
            or before["process_group"] != pid or before["state"] == "Z"
            or cmd[2:7] != expected_cmd
            or len(guard.child_tree(pid, before["starttime_ticks"])) != 1):
        raise RuntimeError("Not the exact own idle parent; no signal sent")
    stopped = False
    try:
        # Stop scheduling first, then recheck state/job/children to close the
        # admission-to-spawn race. An active child makes us release the parent.
        os.kill(pid, signal.SIGSTOP)
        stopped = True
        for _ in range(100):
            current = guard.process_identity(pid)
            if current and current["state"] in ("T", "t"):
                break
            time.sleep(.01)
        if (current is None or current["starttime_ticks"] != before["starttime_ticks"]
                or current["state"] not in ("T", "t")
                or read(root / "job.json") != job
                or len(guard.child_tree(pid, before["starttime_ticks"])) != 1):
            raise RuntimeError("Parent advanced or child appeared; no handoff")
        if pending_check is not None:
            pending_check()
        os.kill(pid, signal.SIGTERM)
        os.kill(pid, signal.SIGCONT)
        stopped = False
        for _ in range(100):
            current = guard.process_identity(pid)
            if current is None or current["state"] == "Z":
                return dict(**before, termination_confirmed=True, own_waiting_parent_only=True)
            time.sleep(.05)
        raise RuntimeError("Old idle parent termination not confirmed; no replacement launch")
    finally:
        if stopped:
            current = guard.process_identity(pid)
            if current and current["starttime_ticks"] == before["starttime_ticks"]:
                os.kill(pid, signal.SIGCONT)


def verify_completed(root, job, plan, base, guard):
    for stage in job["stages"]:
        name = stage["name"]
        profile = guard.profile_receipt(read(stage["profile"]))
        if profile.get("child_pid") != stage["pid"]:
            raise RuntimeError("Completed profile belongs to a different child")
        if name == "cuda_verification":
            receipt = read(root / "cuda_verification.json")
        elif name.startswith(("smoke_", "train_")):
            arm = next(a for a in plan["arms"] if a["name"] == name.split("_", 1)[1])
            receipt = base.validate_training(root, plan, arm, smoke=name.startswith("smoke_"))
        else:
            receipt = job["receipts"][name]
            if any(sha(Path(receipt["path"]) / n) != v for n, v in receipt["files"].items()):
                raise RuntimeError("Completed report fingerprint changed")
        if receipt != job["receipts"][name]:
            raise RuntimeError("Completed stage receipt changed")


def prepare(root, output):
    base, guard, plan = load_runtime(root)
    if output.parent != root or output.name != "execution_io_relaxed_v1":
        raise ValueError("Execution override must be the exact experiment-local subdirectory")
    if output.exists() or output.is_symlink() or (root / "execution_override.json").exists():
        raise FileExistsError("No overwrite/retry of execution override")
    job = read(root / "job.json")
    remaining = remaining_phases(job)
    if len(job["stages"]) < 2:
        raise RuntimeError("The two observed preflights must already be complete")
    verify_completed(root, job, plan, base, guard)
    verify_pending(root, plan, remaining)
    # Create the exclusive handoff location before touching the waiting parent.
    output.mkdir()
    base.save(output / "original_job.json", job, exclusive=True)
    identity = stop_idle_parent(root, job, guard, lambda: verify_pending(root, plan, remaining))
    policy = dict(status="prepared", root=str(root), output=str(output), at=base.now(),
                  original_plan_sha256=sha(root / "plan.json"), original_job_sha256=sha(root / "job.json"),
                  wrapper_path=str(Path(__file__).resolve()), wrapper_sha256=sha(__file__),
                  authorization=AUTHORIZATION, previous_parent=identity, remaining_phases=remaining,
                  retained_completed_stages=[s["name"] for s in job["stages"]],
                  io_full_avg10_admission="record_only_not_a_veto", prior_io_limit=20,
                  all_other_resource_guards_unchanged=True, scientific_configuration_unchanged=True,
                  checkpoint_resume=False, completed_stages_repeated=False, automatic_retry=False)
    base.save(output / "policy.json", policy, exclusive=True)
    base.save(output / "policy_fingerprint.json", dict(sha256=sha(output / "policy.json")), exclusive=True)
    base.save(output / "job.json", dict(status="prepared", current=None, stages=[], workflow_complete=False,
              completed_arms=job.get("completed_arms", []), receipts=dict(job["receipts"]),
              error=None, requires_user_action=False, parent_pid=None), exclusive=True)
    base.save(root / "execution_override.json", dict(policy=str(output / "policy.json"),
              authoritative_job=str(output / "job.json"), original_parent_terminated=True,
              original_job_preserved=True, io_only_resource_change=True), exclusive=True)
    return policy


def unchanged(root, output, base, guard, plan):
    policy = read(output / "policy.json")
    original_job = read(output / "original_job.json")
    if (sha(output / "policy.json") != read(output / "policy_fingerprint.json")["sha256"]
            or policy["remaining_phases"] != remaining_phases(original_job)
            or policy["retained_completed_stages"] != [s["name"] for s in original_job["stages"]]
            or original_job != read(root / "job.json")
            or sha(policy["wrapper_path"]) != policy["wrapper_sha256"]
            or sha(root / "plan.json") != policy["original_plan_sha256"]
            or sha(root / "job.json") != policy["original_job_sha256"]):
        raise RuntimeError("Frozen handoff/source/plan changed")
    old = policy["previous_parent"]
    current = guard.process_identity(old["pid"])
    if current and current["starttime_ticks"] == old["starttime_ticks"] and current["state"] != "Z":
        raise RuntimeError("Old scheduling parent still alive")
    base.unchanged(plan)
    return policy


def admission(output, job, name, *, gpu, base, guard):
    ready = 0
    while True:
        row = guard.snapshot(18)
        row.update(requires_gpu=gpu, io_admission_veto=False)
        row["ready_under_authorized_policy"] = admission_ready(row, gpu=gpu)
        ready = ready + 1 if row["ready_under_authorized_policy"] else 0
        job.update(status="waiting_resources", current=name)
        job.setdefault("resource_checks", []).append(row)
        job["resource_checks"] = job["resource_checks"][-120:]
        base.save(output / "job.json", job)
        if ready >= (2 if gpu else 1):
            return
        time.sleep(10 if ready else 60)


def execute(root, output):
    base, guard, plan = load_runtime(root)
    policy = unchanged(root, output, base, guard, plan)
    job = read(output / "job.json")
    if job["status"] != "prepared" or job["stages"]:
        raise RuntimeError("No repeated execution")
    os.nice(10)
    job.update(parent_pid=os.getpid(), actual_nice=os.getpriority(os.PRIO_PROCESS, 0), started_at=base.now())
    base.save(output / "job.json", job)
    try:
        for name in policy["remaining_phases"]:
            if name == "cuda_verification":
                raise RuntimeError("This override requires the CUDA identity preflight already complete")
            kind, arm_name = name.split("_", 1)
            arm = next(a for a in plan["arms"] if a["name"] == arm_name)
            gpu = kind != "report"
            unchanged(root, output, base, guard, plan)
            admission(output, job, name, gpu=gpu, base=base, guard=guard)
            unchanged(root, output, base, guard, plan)
            if gpu:
                run = root / "smoke" / arm_name / "run" if kind == "smoke" else Path(arm["run"])
                command = [sys.executable, "-u", "-m", "experiments.rsmi_separate_qcr", "--config",
                           arm["smoke_config_path"] if kind == "smoke" else arm["config_path"],
                           "--output", str(run), "--device", base.GPU, "--mi-cache-dir", str(root / "mi_cache"),
                           "--skip-final-report", "--benchmark-warmup", "0"]
            else:
                command = ["/usr/bin/timeout", "--signal=TERM", "--kill-after=5s", "900s", sys.executable,
                           "-u", "-m", "scripts.report_ising_kb_highdim", "--run", arm["run"], "--output", arm["report"],
                           "--checkpoint", "12000", "--threads", "1", "--sampling-repeats", "5", "--max-seconds", "840",
                           "--progress-path", str(output / f"{name}.progress.json")]
            env = base.native_cuda_environment()
            if not gpu:
                env["CUDA_VISIBLE_DEVICES"] = ""
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
                env[key] = "2" if gpu else "1"
            env.update(PYTHONUNBUFFERED="1", MPLCONFIGDIR=str(root / "matplotlib_cache"))
            start = time.monotonic()
            with (output / f"{name}.log").open("x") as log:
                child = subprocess.Popen(command, cwd=plan["code_root"], env=env, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                stage = dict(name=name, pid=child.pid, command=command, status="running", started_at=base.now())
                job["stages"].append(stage)
                job.update(status="running", current=name)
                base.save(output / "job.json", job)
                try:
                    guard.monitor(child, gpu=gpu, profile_path=str(output / f"{name}.profile.json"))
                finally:
                    stage.update(returncode=child.poll(), elapsed_seconds=time.monotonic() - start,
                                 status="complete" if child.poll() == 0 else "failed", ended_at=base.now())
                    base.save(output / "job.json", job)
            if child.returncode:
                progress = output / f"{name}.progress.json"
                hold = not gpu and (base.report_timeout(child.returncode, stage["elapsed_seconds"])
                         or (progress.exists() and read(progress).get("status") == "analysis_budget_hold"))
                if hold:
                    job.update(status="analysis_budget_hold", requires_user_action=True, error=f"{name}: analysis budget")
                    return
                raise RuntimeError(f"{name} failed with {child.returncode}; no retry")
            unchanged(root, output, base, guard, plan)
            if gpu:
                receipt = base.validate_training(root, plan, arm, smoke=kind == "smoke")
            else:
                marker = read(Path(arm["report"]) / "status.json")
                files = marker.get("public_files", [])
                if (marker.get("status") != "complete" or not marker.get("report_complete")
                        or marker.get("checkpoint_epoch") != 12000 or marker.get("theory_dim") != arm["theory_dim"]
                        or not files or any(Path(n).name != n or Path(n).suffix not in (".json", ".png", ".npz") for n in files)):
                    raise RuntimeError("Incomplete report or invalid whitelist")
                receipt = dict(status="complete", path=arm["report"], files={n:sha(Path(arm["report"]) / n) for n in files})
                job["completed_arms"].append(arm_name)
            job["receipts"][name] = receipt
            base.save(output / "job.json", job)
        job.update(status="complete", workflow_complete=True, current=None, ended_at=base.now())
    except BaseException as error:
        job.update(status="failed", workflow_complete=False, requires_user_action=True,
                   error=f"{type(error).__name__}: {error}", ended_at=base.now())
        raise
    finally:
        base.save(output / "job.json", job)


def launch(root, output):
    base, guard, plan = load_runtime(root)
    unchanged(root, output, base, guard, plan)
    if read(output / "job.json")["status"] != "prepared":
        raise RuntimeError("Not prepared; no restart")
    with (output / "launch.json").open("x") as receipt, (output / "parent.log").open("x") as log:
        parent = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--execute", "--root", str(root),
                                   "--output", str(output)], cwd=plan["code_root"], stdin=subprocess.DEVNULL,
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                  env=dict(base.native_cuda_environment(), PYTHONUNBUFFERED="1"))
        value = dict(status="launched", parent_pid=parent.pid, at=base.now(), policy_sha256=sha(output / "policy.json"))
        receipt.write(json.dumps(value, ensure_ascii=False, indent=2)+"\n")
    return value


def request(root, output):
    """Defer an I/O-only handoff until the existing child has finished."""
    base, guard, plan = load_runtime(root)
    if output.parent != root or output.name != "execution_io_relaxed_v1" or output.exists():
        raise ValueError("Require a new exact experiment-local override directory")
    path = root / "io_override_request.json"
    value = dict(status="requested", at=base.now(), authorization=AUTHORIZATION,
                 wrapper_path=str(Path(__file__).resolve()), wrapper_sha256=sha(__file__),
                 previous_parent_pid=EXPECTED_PARENT_PID, previous_parent_start_ticks=EXPECTED_PARENT_START_TICKS,
                 policy="I/O logged only; all other resource/scientific settings unchanged",
                 handoff_only_while_idle=True, output=str(output))
    base.save(path, value, exclusive=True)
    with (root / "io_override_request.log").open("x") as log:
        child = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--await-handoff",
                                  "--root", str(root), "--output", str(output)], cwd=plan["code_root"],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, env=base.native_cuda_environment())
    return dict(status="waiting_for_idle_boundary", coordinator_pid=child.pid, request=str(path))


def await_handoff(root, output):
    base, guard, plan = load_runtime(root)
    path = root / "io_override_request.json"
    requested = read(path)
    if requested["wrapper_sha256"] != sha(__file__):
        raise RuntimeError("Requested execution wrapper fingerprint changed")
    requested.update(status="waiting_for_idle_boundary", coordinator_pid=os.getpid())
    base.save(path, requested)
    try:
        while True:
            old = guard.process_identity(EXPECTED_PARENT_PID)
            job = read(root / "job.json")
            if job.get("workflow_complete") and job.get("status") == "complete":
                requested.update(status="not_needed_workflow_already_complete")
                return
            if (old is None or old["starttime_ticks"] != EXPECTED_PARENT_START_TICKS
                    or old["state"] == "Z" or job.get("error") or job.get("requires_user_action")):
                raise RuntimeError("Original workflow needs attention; no restart or handoff")
            if job.get("status") == "waiting_resources":
                if requested["wrapper_sha256"] != sha(__file__):
                    raise RuntimeError("Requested execution wrapper fingerprint changed before handoff")
                try:
                    prepare(root, output)
                except RuntimeError:
                    # The parent may have started the next child before we
                    # took the idle snapshot. Never stop that active child.
                    if not output.exists() and read(root / "job.json").get("status") == "running":
                        time.sleep(5)
                        continue
                    raise
                launched = launch(root, output)
                requested.update(status="handoff_complete", execution_launch=launched,
                                 authoritative_job=str(output / "job.json"))
                return
            time.sleep(5)
    except BaseException as error:
        requested.update(status="requires_user_action", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        base.save(path, requested)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    for mode in ("prepare", "launch", "execute", "request", "await-handoff"):
        group.add_argument(f"--{mode}", action="store_true")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if args.request:
        print(json.dumps(request(root, output), ensure_ascii=False))
    elif args.await_handoff:
        await_handoff(root, output)
    elif args.prepare:
        print(json.dumps(prepare(root, output), ensure_ascii=False))
    elif args.launch:
        print(json.dumps(launch(root, output), ensure_ascii=False))
    else:
        execute(root, output)


if __name__ == "__main__":
    main()
