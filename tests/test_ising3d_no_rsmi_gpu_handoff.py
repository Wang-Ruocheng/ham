"""No real process signals, GPU use, or physical training in these tests."""
import json
import signal

import pytest

from scripts import continue_ising3d_no_rsmi_gpu as h


def waiting(pid=541):
    return dict(status="waiting_resources", current="gpu_preflight", parent_pid=pid,
        error=None, requires_user_action=False, device="cuda:2", stages=[],
        actual_training_updates=0, GPU_stage_seconds=0.)


def test_only_never_started_waiter_can_move(tmp_path):
    job = waiting()
    h.validate_waiting(tmp_path, job)
    for name in ("gpu_preflight", "run", "report", "comparison"):
        (tmp_path / name).mkdir()
        with pytest.raises(ValueError, match="output exists"):
            h.validate_waiting(tmp_path, job)
        (tmp_path / name).rmdir()
    for changes in ({"actual_training_updates": 1}, {"stages": [{"name": "gpu_preflight"}]},
                    {"GPU_stage_seconds": 1.}, {"current": "train"}, {"status": "running"}):
        with pytest.raises(ValueError, match="waiter"):
            h.validate_waiting(tmp_path, job | changes)


def fake_process(monkeypatch, root, *, child_race=False):
    state = dict(status="S", pending=False, alive=True)
    signals = []
    def identity(pid):
        if not state["alive"]:
            return None
        return dict(pid=pid, state=state["status"], start_ticks="1234",
            argv=["python", "-m", "scripts.run_ising3d_no_rsmi", "--root", str(root), "--run"])
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
    monkeypatch.setattr(h.os, "pidfd_open", None, raising=False)
    monkeypatch.setattr(h, "children", lambda pid: ["999"] if child_race and state["status"] == "T" else [])
    monkeypatch.setattr(h.os, "kill", kill)
    monkeypatch.setattr(h.socket, "gethostname", lambda: "thinklab-105-225")
    monkeypatch.setattr(h, "verify_science", lambda root, expected=None: {"config.json": "unchanged"})
    return signals, state


def test_prepare_stops_only_waiter_and_archives_original_state(tmp_path, monkeypatch):
    original = waiting()
    (tmp_path / "job.json").write_text(json.dumps(original))
    signals, state = fake_process(monkeypatch, tmp_path)
    record = h.prepare(tmp_path)
    assert signals == [signal.SIGSTOP, signal.SIGTERM, signal.SIGCONT]
    assert not state["alive"]
    assert record["status"] == "handoff_ready"
    assert record["physical_sampling_updates"] == record["MI_computations"] == 0
    assert record["trained_updates_before_handoff"] == 0
    assert h.read(tmp_path / h.HANDOFF / "original_job.json") == original
    job = h.read(tmp_path / "job.json")
    assert job["stages"] == [] and job["actual_training_updates"] == 0
    assert job["parent_pid"] is None and job["device"] == "auto_idle"
    assert job["resource_handoff"]["scientific_plan_unmodified"]
    with pytest.raises(FileExistsError):
        h.prepare(tmp_path)


def test_child_spawn_race_restores_parent_without_term(tmp_path, monkeypatch):
    original = waiting()
    (tmp_path / "job.json").write_text(json.dumps(original))
    signals, state = fake_process(monkeypatch, tmp_path, child_race=True)
    with pytest.raises(RuntimeError, match="without children"):
        h.prepare(tmp_path)
    assert signals == [signal.SIGSTOP, signal.SIGCONT]
    assert state["alive"] and h.read(tmp_path / "job.json") == original
    assert not (tmp_path / h.HANDOFF).exists()


def test_wrong_process_is_never_signalled(tmp_path, monkeypatch):
    (tmp_path / "job.json").write_text(json.dumps(waiting()))
    signals, _ = fake_process(monkeypatch, tmp_path)
    monkeypatch.setattr(h, "process_identity", lambda pid: dict(pid=pid, argv=["python", "another_job.py"]))
    with pytest.raises(RuntimeError, match="identity"):
        h.prepare(tmp_path)
    assert signals == []


def test_commands_change_device_only_and_keep_zero_mi(tmp_path):
    from scripts import run_ising3d_no_rsmi as original
    (tmp_path / "plan.json").write_text(json.dumps(dict(baseline="/same/baseline")))
    before = original.commands(tmp_path)
    after = h.stage_commands(original, tmp_path)
    for (old_name, old_command, old_gpu), (name, command, gpu) in zip(before, after):
        assert (old_name, old_gpu) == (name, gpu)
        expected = list(old_command)
        if gpu:
            expected[expected.index("--device") + 1] = "cuda:3"
        assert command == expected
    assert "--rsmi-weight" in after[1][1]
    assert after[1][1][after[1][1].index("--rsmi-weight") + 1] == "0"
    assert "--resume-checkpoint" not in after[1][1]
    assert "--skip-mi-diagnostic" in after[2][1]


def test_verify_science_never_changes_frozen_files(tmp_path):
    code = tmp_path / "code"
    code.mkdir()
    frozen = code / "model.py"
    frozen.write_text("# frozen scientific implementation\n")
    cache = tmp_path / "external-cache.pt"
    cache.write_bytes(b"not a torch model; fixture hash only")
    data_sha = h.sha(cache)
    (tmp_path / "data").mkdir()
    (tmp_path / "data/evaluation_metadata.json").write_text(json.dumps(dict(cache_sha256=data_sha)))
    (tmp_path / "config.json").write_text(json.dumps(dict(training=dict(epochs=6000), data=dict(cache=str(cache)))))
    baseline = tmp_path / "baseline"
    (baseline / "run").mkdir(parents=True)
    (baseline / "report").mkdir()
    (baseline / "run/manifest.json").write_text("{}")
    (baseline / "report/report.json").write_text("{}")
    (tmp_path / "plan.json").write_text(json.dumps(dict(source_fingerprints={"model.py": h.sha(frozen)},
        fixed_training_updates=6000, data_cache=str(cache), data_sha256=data_sha,
        MI_computed=False, MI_cache_read=False, command_changes=dict(rsmi_weight=[2, 0]),
        baseline=str(baseline), baseline_manifest_sha256=h.sha(baseline / "run/manifest.json"),
        baseline_report_sha256=h.sha(baseline / "report/report.json"))))
    before = {name: (tmp_path / name).read_bytes() for name in ("config.json", "plan.json", "code/model.py")}
    expected = h.verify_science(tmp_path)
    assert h.verify_science(tmp_path, expected) == expected
    assert all((tmp_path / name).read_bytes() == value for name, value in before.items())
    frozen.write_text("# changed\n")
    with pytest.raises(RuntimeError, match="source mismatch"):
        h.verify_science(tmp_path, expected)


def test_reused_pid_is_not_the_original_waiter(monkeypatch):
    record = dict(old_parent=dict(pid=541, start_ticks="1234"))
    monkeypatch.setattr(h, "process_identity", lambda pid: dict(pid=pid, start_ticks="9999", state="S"))
    assert not h.running_old(record)


def test_choose_same_idle_card_twice_not_a_transient_free_card(tmp_path, monkeypatch):
    (tmp_path / h.HANDOFF).mkdir()
    job = dict(resource_handoff={})
    frames = []
    for tick, ready in ((0., {3}), (10., {0}), (20., {0})):
        frames.append([dict(gpu_index=index, gpu_uuid=uuid, free_mib=24000,
            ready=index in ready, checked_at=tick) for index, uuid in h.GPU_UUIDS.items()])
    monkeypatch.setattr(h, "all_snapshots", lambda: frames.pop(0))
    sleeps = []
    monkeypatch.setattr(h.time, "sleep", lambda duration: sleeps.append(duration))
    selected = h.select_idle_gpu(tmp_path, job)
    assert selected["device"] == "cuda:0" and selected["first_idle_check_at"] == 10.
    assert selected["checked_at"] == 20. and sleeps == [10, 10]
    assert job["resource_handoff"]["to_device"] == "cuda:0"
    assert h.read(tmp_path / h.HANDOFF / "selection.json") == selected


def test_train_receipt_uses_actual_run_output_directory(tmp_path):
    from types import SimpleNamespace
    calls = []
    original = SimpleNamespace(validate_training_receipt=lambda *args: calls.append(args))
    h.validate_stage_receipt(original, tmp_path, "gpu_preflight")
    h.validate_stage_receipt(original, tmp_path, "train")
    assert calls == [(tmp_path, "gpu_preflight", 5), (tmp_path, "run", 6000)]
