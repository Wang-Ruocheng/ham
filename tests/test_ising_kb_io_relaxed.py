"""Pure tests of the user-authorized I/O-only scheduling override.

No SSH, signals, CUDA initialization, tensor loads, or child launches occur.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import continue_ising_kb_io_relaxed as override


def row(**changes):
    result = dict(mem_available_gib=18.0, load1=30.0, cpu_count=24,
                  disk_free_gib=20.0, compute_processes=[], gpu_free_mib=16 * 1024,
                  io_full_avg10=22.5, host_ready=False, gpu_ready=False)
    result.update(changes)
    return result


@pytest.fixture(autouse=True)
def forbid_external_mutation(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Unit tests must not signal processes or launch a subprocess")
    monkeypatch.setattr(override.os, "kill", forbidden)
    monkeypatch.setattr(override.subprocess, "Popen", forbidden)


@pytest.mark.parametrize("gpu", [False, True])
@pytest.mark.parametrize("pressure", [0, 19.99, 20, 22.5, 100])
def test_only_io_pressure_is_not_a_veto(gpu, pressure):
    sample = row(io_full_avg10=pressure)
    before = deepcopy(sample)
    assert override.admission_ready(sample, gpu=gpu)
    assert sample == before


@pytest.mark.parametrize("gpu", [False, True])
@pytest.mark.parametrize("changes", [
    dict(mem_available_gib=17.999),
    dict(load1=30.001),
    dict(disk_free_gib=19.999),
])
def test_all_other_host_guards_remain(gpu, changes):
    assert not override.admission_ready(row(**changes), gpu=gpu)


@pytest.mark.parametrize("changes", [
    dict(compute_processes=[dict(pid=9001)]),
    dict(gpu_free_mib=16 * 1024 - 1),
])
def test_gpu_guard_still_vetoes_gpu_not_cpu(changes):
    sample = row(**changes)
    assert not override.admission_ready(sample, gpu=True)
    assert override.admission_ready(sample, gpu=False)


def idle_job():
    names = list(override.PHASES[:2])
    return dict(status="waiting_resources", workflow_complete=False, current="smoke_dim10",
                error=None, requires_user_action=False, parent_pid=6412,
                stages=[dict(name=n, status="complete", returncode=0) for n in names],
                receipts={n: dict(status="complete") for n in names}, completed_arms=[])


def test_only_unexecuted_suffix_is_selected():
    job = idle_job()
    before = deepcopy(job)
    assert override.remaining_phases(job) == ["smoke_dim10", "train_dim5", "report_dim5",
                                            "train_dim10", "report_dim10"]
    assert job == before


@pytest.mark.parametrize("change", [
    dict(status="running"), dict(workflow_complete=True), dict(error="something failed"),
    dict(requires_user_action=True), dict(current="train_dim5"), dict(receipts={}),
])
def test_nonidle_or_unverified_state_fails_closed(change):
    job = idle_job()
    job.update(change)
    with pytest.raises(RuntimeError):
        override.remaining_phases(job)


@pytest.mark.parametrize("field,value", [("status", "running"), ("status", "failed"),
                                        ("returncode", -15), ("returncode", None),
                                        ("name", "smoke_dim10")])
def test_completed_prefix_requires_successful_ordered_stages(field, value):
    job = idle_job()
    job["stages"][-1][field] = value
    with pytest.raises(RuntimeError):
        override.remaining_phases(job)


def test_guard_monitor_is_not_replaced_in_source():
    """The wrapper must retain exactly the already-approved live hard guard."""
    from pathlib import Path
    source = Path(override.__file__).read_text()
    assert "guard.monitor(child, gpu=gpu," in source
    assert "guard.guard_violation =" not in source
    assert "guard.monitor =" not in source


def test_unrecorded_pending_artifact_cannot_be_reexecuted(tmp_path):
    plan = dict(arms=[dict(name="dim5", run=str(tmp_path / "dim5/run"), report=str(tmp_path / "dim5/report")),
                      dict(name="dim10", run=str(tmp_path / "dim10/run"), report=str(tmp_path / "dim10/report"))])
    names = list(override.PHASES[2:])
    override.verify_pending(tmp_path, plan, names)
    (tmp_path / "smoke/dim10/run").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="already has artifacts"):
        override.verify_pending(tmp_path, plan, names)


@pytest.mark.parametrize("changed", ["pid", "starttime_ticks"])
def test_stale_parent_identity_never_receives_signal(monkeypatch, tmp_path, changed):
    identity = dict(pid=override.EXPECTED_PARENT_PID, starttime_ticks=override.EXPECTED_PARENT_START_TICKS,
                    uid=override.os.getuid(), session=override.EXPECTED_PARENT_PID,
                    process_group=override.EXPECTED_PARENT_PID, state="S")
    job = idle_job()
    if changed == "pid":
        job["parent_pid"] += 1
    else:
        identity["starttime_ticks"] = "1111111111"
    command = ["python", "-u", "-m", "scripts.run_ising_kb_highdim", "--execute", "--output", str(tmp_path), ""]
    class ProcPath:
        def __truediv__(self, child):
            return self
        def read_bytes(self):
            return "\0".join(command).encode()
    monkeypatch.setattr(override, "Path", lambda p: ProcPath() if p == "/proc" else Path(p))
    guard = SimpleNamespace(process_identity=lambda pid: identity, child_tree=lambda pid, start: [identity])
    with pytest.raises(RuntimeError, match="no signal sent"):
        override.stop_idle_parent(tmp_path, job, guard)


def test_idle_handoff_signals_only_after_stop_and_recheck(monkeypatch, tmp_path):
    identity = dict(pid=override.EXPECTED_PARENT_PID, starttime_ticks=override.EXPECTED_PARENT_START_TICKS,
                    uid=override.os.getuid(), session=override.EXPECTED_PARENT_PID,
                    process_group=override.EXPECTED_PARENT_PID, state="S")
    job = idle_job()
    signals = []
    state = dict(stopped=False, terminated=False, pending_checked=False)
    command = ["python", "-u", "-m", "scripts.run_ising_kb_highdim", "--execute", "--output", str(tmp_path), ""]
    class ProcPath:
        def __truediv__(self, child):
            return self
        def read_bytes(self):
            return "\0".join(command).encode()
    def process_identity(pid):
        if state["terminated"]:
            return None
        return dict(identity, state="T" if state["stopped"] else "S")
    def signal_only_in_memory(pid, sig):
        assert pid == override.EXPECTED_PARENT_PID
        signals.append(sig)
        if sig == override.signal.SIGSTOP:
            state["stopped"] = True
        elif sig == override.signal.SIGTERM:
            assert state["stopped"] and state["pending_checked"]
        elif sig == override.signal.SIGCONT:
            state["terminated"] = True
    def pending_check():
        assert state["stopped"]
        state["pending_checked"] = True
    monkeypatch.setattr(override, "Path", lambda p: ProcPath() if p == "/proc" else Path(p))
    monkeypatch.setattr(override, "read", lambda p: deepcopy(job))
    monkeypatch.setattr(override.os, "kill", signal_only_in_memory)
    guard = SimpleNamespace(process_identity=process_identity, child_tree=lambda pid, start: [identity])
    result = override.stop_idle_parent(tmp_path, job, guard, pending_check)
    assert result["termination_confirmed"] and result["own_waiting_parent_only"]
    assert signals == [override.signal.SIGSTOP, override.signal.SIGTERM, override.signal.SIGCONT]


def simulated_await(monkeypatch, tmp_path, job):
    """All process identity, sleep, I/O and launches here are in-memory stubs."""
    root, output = tmp_path, tmp_path / "execution_io_relaxed_v1"
    state = dict(job=deepcopy(job), saved=[], sleeps=[], prepared=[], launched=[])
    identity = dict(starttime_ticks=override.EXPECTED_PARENT_START_TICKS, state="S")
    requested = dict(wrapper_sha256="fixed-wrapper-sha")
    def read_only_state(path):
        return deepcopy(state["job"] if Path(path).name == "job.json" else requested)
    base = SimpleNamespace(save=lambda path, value, **kw: state["saved"].append(deepcopy(value)))
    guard = SimpleNamespace(process_identity=lambda pid: deepcopy(identity))
    monkeypatch.setattr(override, "load_runtime", lambda root: (base, guard, {}))
    monkeypatch.setattr(override, "read", read_only_state)
    monkeypatch.setattr(override, "sha", lambda path: "fixed-wrapper-sha")
    monkeypatch.setattr(override.os, "nice", lambda amount: 10)
    monkeypatch.setattr(override.time, "sleep", lambda seconds: state["sleeps"].append(seconds))
    monkeypatch.setattr(override, "prepare", lambda r, o: state["prepared"].append((r, o)))
    def launch_only_in_memory(r, o):
        state["launched"].append((r, o))
        return dict(status="launched", parent_pid=123456)
    monkeypatch.setattr(override, "launch", launch_only_in_memory)
    return root, output, state, guard


def test_await_completed_workflow_never_hands_off_even_if_parent_gone(monkeypatch, tmp_path):
    root, output, state, guard = simulated_await(monkeypatch, tmp_path,
                                                dict(status="complete", workflow_complete=True))
    guard.process_identity = lambda pid: None
    override.await_handoff(root, output)
    assert state["saved"][-1]["status"] == "not_needed_workflow_already_complete"
    assert not state["prepared"] and not state["launched"] and not state["sleeps"]


def test_await_failure_never_restarts(monkeypatch, tmp_path):
    root, output, state, guard = simulated_await(monkeypatch, tmp_path,
        dict(status="failed", workflow_complete=False, error="hard guard", requires_user_action=True))
    with pytest.raises(RuntimeError, match="no restart or handoff"):
        override.await_handoff(root, output)
    assert state["saved"][-1]["status"] == "requires_user_action"
    assert not state["prepared"] and not state["launched"]


def test_await_lost_parent_identity_never_restarts(monkeypatch, tmp_path):
    root, output, state, guard = simulated_await(monkeypatch, tmp_path, dict(status="running"))
    guard.process_identity = lambda pid: dict(starttime_ticks="different-process", state="S")
    with pytest.raises(RuntimeError, match="no restart or handoff"):
        override.await_handoff(root, output)
    assert state["saved"][-1]["status"] == "requires_user_action"
    assert not state["prepared"] and not state["launched"]


def test_await_running_child_is_left_alone_until_idle_boundary(monkeypatch, tmp_path):
    root, output, state, guard = simulated_await(monkeypatch, tmp_path, dict(status="running"))
    def simulated_five_seconds(seconds):
        assert seconds == 5
        assert not state["prepared"] and not state["launched"]
        state["sleeps"].append(seconds)
        state["job"] = idle_job()
    monkeypatch.setattr(override.time, "sleep", simulated_five_seconds)
    override.await_handoff(root, output)
    assert state["sleeps"] == [5]
    assert state["prepared"] == [(root, output)] and state["launched"] == [(root, output)]
    assert state["saved"][-1]["status"] == "handoff_complete"


def test_await_parent_wins_spawn_race_without_override_artifacts(monkeypatch, tmp_path):
    root, output, state, guard = simulated_await(monkeypatch, tmp_path, idle_job())
    def parent_wins_race(r, o):
        state["prepared"].append((r, o))
        state["job"] = dict(status="running")
        raise RuntimeError("Require an idle prefix")
    def simulated_normal_completion(seconds):
        assert seconds == 5
        state["sleeps"].append(seconds)
        state["job"] = dict(status="complete", workflow_complete=True)
    monkeypatch.setattr(override, "prepare", parent_wins_race)
    monkeypatch.setattr(override.time, "sleep", simulated_normal_completion)
    override.await_handoff(root, output)
    assert state["saved"][-1]["status"] == "not_needed_workflow_already_complete"
    assert state["sleeps"] == [5] and not state["launched"]

