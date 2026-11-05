"""Orchestration tests with mocked child execution; no GPU or physical sampling."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_phi4_hopping_qcr as runner


CONFIG = Path(__file__).resolve().parents[1] / "configs/phi4_hopping_b3_g1_aligned_seed91.json"
UUID = "GPU-713b07d0-118c-f939-c0a8-565390215e01"


def resources():
    return dict(host=dict(cpus=24, load1=7., mem_available_bytes=120*1024**3,
        io_full_avg10=.4), gpu=dict(index=2, uuid=UUID, free_bytes=24*1024**3, compute_pids=[]))


def test_dry_plan_is_full_fresh_experiment(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner, "run", lambda *_: pytest.fail("Dry plan executed"))
    output = tmp_path / "new"
    assert runner.main(["--output", str(output), "--config", str(CONFIG)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["execution_requested"] is False and not output.exists()
    assert plan["stages"] == ["gpu_preflight", "sampling", "training"]
    assert plan["physical"]["options"]["kappa_values"] == pytest.approx([.50+.03*i for i in range(13)])
    assert plan["physical"]["options"]["chains"] == 4
    assert plan["physical"]["options"]["samples"] == 128
    assert plan["training"]["options"]["system_batch"] == 32
    assert plan["training"]["options"]["sample_batch"] == 64
    assert plan["training"]["options"]["steps"] == 3000
    assert plan["benchmark"]["physical_result"] is False
    assert plan["resources"]["max_training_seconds"] == 1800.
    command = plan["commands"]["training"]
    assert command[2:5] == ["-m", "recurrent_rg", "train"]
    assert "--allow-legacy-training" not in command and "--resume" not in command


@pytest.mark.parametrize("changed", ["steps", "system_batch", "sample_batch", "seed", "threads", "information_pairs", "max_training_seconds", "max_cuda_bytes"])
def test_full_design_not_silently_changed(tmp_path, changed):
    value = json.loads(CONFIG.read_text())
    value["training"][changed] += 1
    source = tmp_path / "config.json"
    source.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        runner.plan(tmp_path / "out", source, 2, UUID)
    assert not (tmp_path / "out").exists()


def test_output_is_exclusive(tmp_path):
    with pytest.raises(ValueError, match="fresh output"):
        runner.plan(tmp_path, CONFIG, 2, UUID)


@pytest.mark.parametrize("host_field,value,gpu,expected", [
    ("load1", 30., True, True), ("load1", 30.1, True, False),
    ("mem_available_bytes", 36*1024**3, True, True),
    ("mem_available_bytes", 36*1024**3-1, True, False),
    ("io_full_avg10", 20., True, False), ("io_full_avg10", 19.9, True, True),
])
def test_host_gates(host_field, value, gpu, expected):
    row = resources()
    row["host"][host_field] = value
    assert runner.ready(row, gpu) is expected


def test_gpu_gate_requires_empty_and_memory():
    row = resources()
    row["gpu"]["compute_pids"] = [123]
    assert not runner.ready(row, True) and runner.ready(row, False)
    row["gpu"].update(compute_pids=[], free_bytes=12*1024**3-1)
    assert not runner.ready(row, True)


def test_environment_masks_only_requested_gpu():
    env = runner.child_environment(UUID)
    assert env["CUDA_VISIBLE_DEVICES"] == UUID
    assert all(env[name] == "1" for name in runner.THREAD_ENV)
    assert runner.child_environment()["CUDA_VISIBLE_DEVICES"] == ""


def test_gpu_inventory_uuid_fail_closed(monkeypatch):
    def output(command, **_):
        return "2, GPU-other, 24000\n"
    monkeypatch.setattr(runner.subprocess, "check_output", output)
    with pytest.raises(RuntimeError, match="UUID mismatch"):
        runner.gpu_resources(2, UUID)


def fake_training(root, data_sha="a"*64):
    from experiments.phi4_hopping_qcr import RECIPE
    folder = root / "training"
    folder.mkdir()
    (folder / "evaluation_arrays.npz").write_bytes(b"mock-public-array")
    summary = dict(recipe=RECIPE, actual_steps=3000, theory_dim=1, thermal_nu=None,
        data_manifest_sha256=data_sha, peak_cuda_bytes=100,
        optimizer_steps={name: dict(min=3000, max=3000, parameters=1) for name in ("Q", "C", "R")})
    manifest = dict(recipe=RECIPE, data_manifest_sha256=data_sha,
        closure_objective="all_start_anchor", closure_scale_weights=None)
    runner.save(folder / "summary.json", summary)
    runner.save(folder / "manifest.json", manifest)
    public = [dict(path=name, sha256=runner.sha(folder / name))
              for name in ("manifest.json", "summary.json", "evaluation_arrays.npz")]
    runner.save(folder / "status.json", dict(status="complete", workflow_complete=True,
        returncode=0, error=None, requires_user_action=False, actual_steps=3000,
        latest_checkpoint_step=3000, summary_sha256=runner.sha(folder / "summary.json"),
        evaluation_arrays_sha256=runner.sha(folder / "evaluation_arrays.npz"), public_files=public))


@pytest.fixture
def fake_runtime(tmp_path, monkeypatch):
    import recurrent_rg.data.phi4_hopping_loader as loader
    root = tmp_path / "result"
    calls, modes = [], {}
    monkeypatch.setattr(runner.socket, "gethostname", lambda: runner.HOST)
    monkeypatch.setattr(runner.os, "getpriority", lambda *_: 10)
    monkeypatch.setattr(runner, "host_resources", lambda: resources()["host"])
    monkeypatch.setattr(runner, "gpu_resources", lambda *_: resources()["gpu"])
    ticks = iter(range(0, 100000, 10))
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    monkeypatch.setattr(loader, "load_dataset", lambda _: SimpleNamespace(
        manifest_sha256="a"*64, warning_count=17))

    class Process:
        def __init__(self, command, **kwargs):
            self.pid, self.returncode = 1234, 0
            if "--benchmark-output" in command:
                name = "gpu_preflight"
                runner.save(root / "gpu_preflight.json", dict(status="complete",
                    conservative_full_training_projection_seconds=modes.get("projection", 100.)))
            elif "scripts.prepare_phi4_hopping" in command:
                name = "sampling"
                (root / "data").mkdir()
                runner.save(root / "data/manifest.json", dict(public_files=["manifest.json", "quality_summary.json", "status.json"]))
                runner.save(root / "data/status.json", dict(status="complete"))
                runner.save(root / "data/quality_summary.json", dict(conditions=[
                    dict(production_acceptance=[modes.get("acceptance", .8)]*4) for _ in range(13)]))
            else:
                name = "training"
                fake_training(root)
            calls.append((name, command, kwargs))
            self.returncode = modes.get(name, 0)
        def poll(self):
            return self.returncode
        def wait(self):
            return self.returncode

    monkeypatch.setattr(runner.subprocess, "Popen", Process)
    return root, calls, modes


def test_parent_sequential_success_freezes_public_inventory(fake_runtime):
    root, calls, modes = fake_runtime
    assert runner.run(root, CONFIG, 2, UUID) == 0
    assert [name for name, *_ in calls] == ["gpu_preflight", "sampling", "training"]
    assert calls[0][2]["env"]["CUDA_VISIBLE_DEVICES"] == UUID
    assert calls[1][2]["env"]["CUDA_VISIBLE_DEVICES"] == ""
    job = json.loads((root / "job.json").read_text())
    assert job["status"] == "complete" and job["workflow_complete"]
    assert job["data_warning_count"] == 17
    assert all(stage["returncode"] == 0 for stage in job["stages"])
    assert job["gpu_idle_checks"]["spacing_seconds"] >= 10
    inventory = json.loads((root / "public_files.json").read_text())
    assert not any(Path(name).suffix == ".pt" or name.endswith("data.npz") for name in inventory)
    assert all(runner.sha(root / name) == expected for name, expected in inventory.items())
    actual = json.loads((root / "training_config.json").read_text())
    assert actual["training"]["expected_gpu_uuid"] == UUID and actual["training"]["device"] == "cuda:0"


@pytest.mark.parametrize("failed,rc,expected", [("gpu_preflight", 3, ["gpu_preflight"]),
    ("sampling", 1, ["gpu_preflight", "sampling"]),
    ("training", 3, ["gpu_preflight", "sampling", "training"])])
def test_child_failure_never_restarts_or_advances(fake_runtime, failed, rc, expected):
    root, calls, modes = fake_runtime
    modes[failed] = rc
    assert runner.run(root, CONFIG, 2, UUID) == rc
    assert [name for name, *_ in calls] == expected
    job = json.loads((root / "job.json").read_text())
    assert job["status"] == ("budget_hold" if rc == 3 else "failed")
    assert job["requires_user_action"] and not job["workflow_complete"]
    assert not (root / "public_files.json").exists()


@pytest.mark.parametrize("acceptance", [0., float("nan")])
def test_zero_nonfinite_acceptance_preserves_data_no_training(fake_runtime, acceptance):
    root, calls, modes = fake_runtime
    # JSON writer rejects NaN before the acceptance gate, also safely stopping.
    modes["acceptance"] = acceptance
    with pytest.raises((RuntimeError, ValueError)):
        runner.run(root, CONFIG, 2, UUID)
    expected = ["gpu_preflight", "sampling"] if acceptance == 0 else ["gpu_preflight"]
    assert [name for name, *_ in calls] == expected
    assert (root / "data").exists() and not (root / "training").exists()


def test_completed_contract_requires_all_3000_adam_steps(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    fake_training(root)
    runner.validate_training(root, "a"*64)
    summary = json.loads((root / "training/summary.json").read_text())
    summary["optimizer_steps"]["C"]["min"] = 2999
    runner.save(root / "training/summary.json", summary)
    with pytest.raises(RuntimeError, match="3000-step"):
        runner.validate_training(root, "a"*64)
