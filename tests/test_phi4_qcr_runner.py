"""No remote launch, production sampling or GPU allocation in these checks."""
import json
from pathlib import Path

import pytest

from scripts import run_phi4_qcr as runner


def test_fixed_endpoint_and_independent_preflight_command(tmp_path):
    formal = runner.training_command(tmp_path, profile=False)
    pilot = runner.training_command(tmp_path, profile=True)
    def option(command, name):
        return command[command.index(name) + 1]
    assert option(formal, "--steps") == "3000"
    assert option(pilot, "--steps") == "20"
    assert option(formal, "--seed") == "91"
    assert option(pilot, "--seed") == "191"
    assert option(formal, "--device") == "cuda:2"
    assert option(formal, "--output") != option(pilot, "--output")
    assert "--resume" not in formal and "--allow-legacy-training" not in formal


def test_preparation_refuses_unknown_host_or_existing_output(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.socket, "gethostname", lambda: "unapproved-host")
    with pytest.raises(ValueError, match="thinklab"):
        runner.prepare(tmp_path / "new")
    assert not (tmp_path / "new").exists()
    monkeypatch.setattr(runner.socket, "gethostname", lambda: runner.HOST)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(FileExistsError):
        runner.prepare(tmp_path)


def test_frozen_sources_checked_without_mutation(tmp_path):
    path = tmp_path / "code.py"
    path.write_text("# initial\n")
    plan = dict(code_root=str(tmp_path), source_sha256={"code.py": runner.sha(path)})
    runner.unchanged(plan)
    path.write_text("# changed\n")
    with pytest.raises(RuntimeError, match="Frozen source"):
        runner.unchanged(plan)


def test_exclusive_receipt_is_not_overwritten(tmp_path):
    path = tmp_path / "receipt.json"
    runner.save(path, {"value": 1}, exclusive=True)
    with pytest.raises(FileExistsError):
        runner.save(path, {"value": 2}, exclusive=True)
    assert json.loads(path.read_text()) == {"value": 1}


def test_gpu_needs_two_ready_samples_not_one(tmp_path, monkeypatch):
    values = iter([dict(gpu_uuid=runner.GPU_UUID, gpu_ready=False, host_ready=True),
                   dict(gpu_uuid=runner.GPU_UUID, gpu_ready=True, host_ready=True),
                   dict(gpu_uuid=runner.GPU_UUID, gpu_ready=True, host_ready=True)])
    monkeypatch.setattr(runner, "resource_snapshot", lambda: next(values))
    sleeps = []
    monkeypatch.setattr(runner.time, "sleep", sleeps.append)
    job = {}
    runner.admission(tmp_path, job, "training", gpu=True)
    assert sleeps == [30, 10]
    assert len(job["resource_checks"]) == 3


def test_unknown_gpu_is_not_silently_mapped_to_another_device(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "resource_snapshot", lambda: dict(gpu_uuid="GPU-other"))
    with pytest.raises(RuntimeError, match="UUID"):
        runner.admission(tmp_path, {}, "training", gpu=True)
