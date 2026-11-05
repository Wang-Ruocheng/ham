import json
from pathlib import Path
import signal

import pytest

from scripts import continue_ising3d_gpu as h


def waiting(root, pid=541):
    return dict(status="waiting_resources", current="gpu_preflight", parent_pid=pid,
        error=None, requires_user_action=False, device="cuda:2",
        stages=[dict(name=name, status="complete", returncode=0) for name in h.CPU_STAGES])


def test_only_never_started_complete_cpu_waiter_is_eligible(tmp_path):
    job = waiting(tmp_path)
    h.validate_waiting(tmp_path, job)
    for name in ("gpu_preflight", "run", "report"):
        (tmp_path / name).mkdir()
        with pytest.raises(ValueError, match="outputs"):
            h.validate_waiting(tmp_path, job)
        (tmp_path / name).rmdir()
    job["current"] = "train"
    with pytest.raises(ValueError, match="waiter"):
        h.validate_waiting(tmp_path, job)
    job = waiting(tmp_path)
    job["stages"][0]["returncode"] = 1
    with pytest.raises(ValueError, match="three completed"):
        h.validate_waiting(tmp_path, job)


def fake_process(monkeypatch, root, *, child_race=False):
    state = dict(status="S", pending=False, alive=True)
    signals = []
    def identity(pid):
        if not state["alive"]:
            return None
        return dict(pid=pid, state=state["status"], start_ticks="1234", argv=["python", "-m", "scripts.run_ising3d",
            "--root", str(root), "--device", "cuda:2"])
    def kill(pid, sig):
        assert pid == 541
        signals.append(sig)
        if sig == signal.SIGSTOP:
            state["status"] = "T"
        elif sig == signal.SIGTERM:
            state["pending"] = True
        elif sig == signal.SIGCONT:
            state["status"] = "S"
            if state["pending"]:
                state["alive"] = False
    monkeypatch.setattr(h, "process_identity", identity)
    # These tests never open/signal real PIDs, even on Linux test hosts.
    monkeypatch.setattr(h.os, "pidfd_open", None, raising=False)
    monkeypatch.setattr(h, "children", lambda pid: ["999"] if child_race and state["status"] == "T" else [])
    monkeypatch.setattr(h.os, "kill", kill)
    monkeypatch.setattr(h.socket, "gethostname", lambda: "thinklab-105-225")
    monkeypatch.setattr(h, "verify_science", lambda root, expected=None: {"config.json": "unchanged"})
    return signals, state


def test_prepare_stops_only_quiescent_waiter_and_preserves_original_job(tmp_path, monkeypatch):
    original = waiting(tmp_path)
    (tmp_path / "job.json").write_text(json.dumps(original))
    signals, state = fake_process(monkeypatch, tmp_path)
    record = h.prepare(tmp_path)
    assert signals == [signal.SIGSTOP, signal.SIGTERM, signal.SIGCONT]
    assert not state["alive"]
    assert record["status"] == "handoff_ready"
    assert record["physical_sampling_updates"] == record["MI_rebuilds"] == 0
    assert h.read(tmp_path / h.HANDOFF / "original_job.json") == original
    job = h.read(tmp_path / "job.json")
    assert job["stages"] == original["stages"]
    assert job["parent_pid"] is None
    assert job["device"] == "cuda:0"
    with pytest.raises(FileExistsError):
        h.prepare(tmp_path)


def test_child_spawn_race_restores_parent_without_term(tmp_path, monkeypatch):
    original = waiting(tmp_path)
    (tmp_path / "job.json").write_text(json.dumps(original))
    signals, state = fake_process(monkeypatch, tmp_path, child_race=True)
    with pytest.raises(RuntimeError, match="spawned a child|Parent changed"):
        h.prepare(tmp_path)
    assert signals == [signal.SIGSTOP, signal.SIGCONT]
    assert state["alive"]
    assert h.read(tmp_path / "job.json") == original
    assert not (tmp_path / h.HANDOFF).exists()


def test_reused_pid_is_never_treated_as_old_scheduler(monkeypatch):
    record = {"old_parent": {"pid": 541, "start_ticks": "1234"}}
    monkeypatch.setattr(h, "process_identity", lambda pid: dict(pid=pid, start_ticks="9876", state="S"))
    assert h.running_old(record) is False


def test_original_frozen_stage_builder_accepts_gpu0_without_recipe_changes():
    from scripts.run_ising3d import stages, build_plan
    config = json.loads((Path(__file__).resolve().parents[1] / "configs/ising3d_b2_g1_seed91.json").read_text())
    commands = [s for s in stages(Path("/exact-original-root"), build_plan(config), "cuda:0") if s[2]]
    assert [c[0] for c in commands] == ["gpu_preflight", "train", "report"]
    for _, command, _ in commands:
        assert command[command.index("--device") + 1] == "cuda:0"
        assert "--resume-checkpoint" not in command
        assert "--allow-nondefault-recipe" not in command
    assert "--epochs" not in commands[1][1]
    assert commands[0][1][commands[0][1].index("--epochs") + 1] == "5"
