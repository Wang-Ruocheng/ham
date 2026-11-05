from pathlib import Path
import json

import pytest

from scripts import run_phi4_q_diagnostic as runner


def test_commands_are_fixed_and_q_only(tmp_path):
    commands = runner.commands(tmp_path, Path("/frozen/code"))
    assert [row[0] for row in commands] == ["frozen_C_statistics", "five_Q_only_arms", "Q_only_report"]
    assert [row[2] for row in commands] == [True, False, False]
    flattened = " ".join(token for _, command, _ in commands for token in command)
    assert "--steps 3000" in flattened
    assert "--gpu-budget-seconds 120" in flattened
    assert "--cpu-budget-seconds 600" in flattened
    assert "cuda:1" in flattened
    for forbidden in ("experiments.phi4_qcr", "prepare_phi4", "--resume", "allow-nondefault", "allow-legacy"):
        assert forbidden not in flattened


def test_prepare_refuses_existing_output(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "validate_host", lambda: None)
    with pytest.raises(FileExistsError):
        runner.prepare(tmp_path)


def test_prepare_pins_original_endpoint_and_five_arms(tmp_path, monkeypatch):
    target = tmp_path / "new"
    monkeypatch.setattr(runner, "validate_host", lambda: None)
    original = {str(runner.SOURCE / "exploratory_training_v1/run/training_step_3000.pt"): runner.CHECKPOINT_SHA}
    monkeypatch.setattr(runner, "source_files", lambda: original)
    monkeypatch.setattr(runner, "read", lambda path: {"status": "complete"})
    monkeypatch.setattr(runner, "source_hashes", lambda path: {"pinned.py": "abc"})
    plan = runner.prepare(target)
    assert set(plan["arms"]) == set("ABCDE")
    assert plan["steps_per_arm"] == 3000
    assert plan["immutable_source_files_sha256"] == original
    assert plan["policy"]["nice"] == 10
    job = json.loads((target/"job.json").read_text())
    assert job["status"] == "prepared" and not job["workflow_complete"]


def test_prepare_rejects_wrong_source_sha_before_creation(tmp_path, monkeypatch):
    target = tmp_path / "new"
    monkeypatch.setattr(runner, "validate_host", lambda: None)
    monkeypatch.setattr(runner, "source_files", lambda: {
        str(runner.SOURCE / "exploratory_training_v1/run/training_step_3000.pt"): "wrong"})
    with pytest.raises(RuntimeError, match="checkpoint SHA"):
        runner.prepare(target)
    assert not target.exists()


def test_execute_refuses_relaunch(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "validate_host", lambda: None)
    monkeypatch.setattr(runner, "read", lambda path: {"status": "running"})
    with pytest.raises(RuntimeError, match="restart"):
        runner.execute(tmp_path)
    assert not (tmp_path/"execution.lock").exists()


def test_unchanged_rejects_source_mutation(monkeypatch):
    monkeypatch.setattr(runner, "source_hashes", lambda path: {"pinned.py": "wrong"})
    with pytest.raises(RuntimeError, match="inventory"):
        runner.unchanged({"code_root": "x", "source_sha256": {"pinned.py": "abc"}})


def test_host_rejects_cuda_remapping(monkeypatch):
    monkeypatch.setattr(runner.socket, "gethostname", lambda: runner.HOST)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with pytest.raises(RuntimeError, match="pinned"):
        runner.validate_host()
