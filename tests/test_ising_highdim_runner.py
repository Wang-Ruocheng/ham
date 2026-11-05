"""High-dimensional Ising launch contracts; no GPU, data load or training here."""
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from recurrent_rg.config import RunConfig
from recurrent_rg.learning_rate import LearningRatePolicy
from scripts import run_ising2d_dimg_ablation as prior
from scripts import run_ising_highdim as pipeline
from scripts.run_scalar_lr_sweep import make_configs


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def normalized(value):
    return json.loads(json.dumps(value))


@pytest.fixture
def source(tmp_path, monkeypatch):
    """Only tiny mock bytes; a physical/cache tensor is never constructed."""
    root = tmp_path / "historical"
    run = root / "run"
    cache = tmp_path / "physical.pt"
    cache.write_bytes(b"not a tensor: never load or fingerprint these bytes")
    write_json(cache.with_suffix(".json"), {"data_sha256": prior.DATA_SHA})
    monkeypatch.setattr(prior, "SIDECAR_SHA", prior.fingerprint(cache.with_suffix(".json")))
    cfg = make_configs(cache)["multistep"]
    cfg["model"]["block_factor"] = 2
    records = {}
    for split, key in prior.MI_KEYS.items():
        target = root / "mi_cache" / f"mi-v1-{key}.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"not a tensor: immutable existing MI")
        records[split] = dict(hit=True, key=key, path=str(target), cache_version=1)
    write_json(run / "manifest.json", dict(config=cfg, data_sha256=prior.DATA_SHA,
        labels_used_in_training=False, source=None,
        execution_optimizations=dict(mi_statistics_cache=records)))
    write_json(run / "summary.json", dict(config=cfg, actual_epochs=12000,
        requested_epochs=12000, stopped_by_training_limit=False, stopped_by_wall_limit=False))
    write_json(root / "job.json", dict(status="complete", workflow_complete=True,
        training=dict(actual_epochs=12000, manifest_sha256=prior.fingerprint(run / "manifest.json"))))
    write_json(root / "mi_preprocessing.json", dict(status="complete", data_sha256=prior.DATA_SHA,
        model_initialized=False, optimizer_updates=0,
        splits={split: {"cache": record} for split, record in records.items()}))
    return run, cfg


def test_plan_json_stat_only_exact_dimensions_and_only_output_changes(source, tmp_path, monkeypatch):
    run, cfg = source
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    old_fingerprint = prior.fingerprint

    def small_receipts_only(path):
        assert Path(path).suffix != ".pt", "Planning must not rehash data, MI or checkpoints"
        return old_fingerprint(path)

    monkeypatch.setattr(prior, "fingerprint", small_receipts_only)
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("Plan cannot load tensors"))
    output = tmp_path / "fresh"
    plan = pipeline.build_plan(run, output)
    base = normalized(RunConfig.from_dict(cfg).to_dict())
    assert [arm["theory_dim"] for arm in plan["arms"]] == [1, 5, 10, 100]
    assert plan["protocol"]["labels_used_in_training"] is False
    assert plan["protocol"]["automatic_retry"] is plan["protocol"]["resume"] is False
    assert plan["protocol"]["extra_response_or_future_loss"] is False
    assert plan["protocol"]["hidden"] == 96
    assert plan["protocol"]["local_Q_features"] == 12
    assert plan["protocol"]["affine_decoder_bins"] == 16
    for arm in plan["arms"]:
        config = normalized(arm["config"])
        expected = deepcopy(base)
        expected["model"]["theory_dim"] = arm["theory_dim"]
        assert config == expected
        assert config["model"]["block_factor"] == 2
        assert config["model"]["hidden"] == 96
        assert config["loss"]["closure_scale_weights"] is None
        assert config["loss"]["coordinate_reduction"] == "sum"
        assert config["training"]["train_modules"] == ["Q", "C", "R"]
        assert config["training"]["gradient_mode"] == "qc_r_separate"
        assert config["training"]["epochs"] == 12000
        assert arm["changes"] == ([] if arm["theory_dim"] == 1 else ["model.theory_dim"])
        for command in (arm["training"], arm["smoke"]):
            assert command[command.index("--device") + 1] == "cuda:2"
            assert "--cached-training" in command
        assert arm["training"][arm["training"].index("--output") + 1] != arm["smoke"][arm["smoke"].index("--output") + 1]
        commands = " ".join(" ".join(arm[key]) for key in ("training", "smoke", "report"))
        for forbidden in ("--resume-checkpoint", "--source ", "--allow-nondefault-recipe",
                          "--allow-legacy-training", "--q-coupling-weight", "--q-future-weight",
                          "prepare_qcr_mi", "--max-wall-seconds", "--max-training-seconds"):
            assert forbidden not in commands
    assert not output.exists()
    assert {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("dimensions", [(1, 5, 10), (1, 5, 10, 99), (100, 10, 5, 1),
                                        (1, 5, 10, 100, 100), (True, 5, 10, 100)])
def test_plan_rejects_unauthorized_dimension_sets(source, tmp_path, dimensions):
    with pytest.raises(ValueError):
        pipeline.build_plan(source[0], tmp_path / "fresh", dimensions)


def test_plan_rejects_historical_nested_or_existing_output(source, tmp_path):
    run, _ = source
    for target in (run, run / "new", run.parent):
        with pytest.raises((ValueError, FileExistsError)):
            pipeline.build_plan(run, target)
    target = tmp_path / "existing"
    target.mkdir()
    with pytest.raises(FileExistsError):
        pipeline.build_plan(run, target)


@pytest.fixture
def resource_state(monkeypatch):
    state = dict(host=pipeline.HOST, uuid=pipeline.GPU_UUID, free_mib=12288,
                 active=False, available=36.0, cpus=24, load=30.0, io=19.99, disk=20.0)
    monkeypatch.setattr(pipeline.socket, "gethostname", lambda: state["host"])
    monkeypatch.setattr(pipeline.os, "cpu_count", lambda: state["cpus"])
    monkeypatch.setattr(pipeline.os, "getloadavg", lambda: (state["load"], 0, 0))
    monkeypatch.setattr(pipeline.shutil, "disk_usage",
                        lambda path: SimpleNamespace(free=int(state["disk"] * 1024**3)))
    original_read = Path.read_text

    def fake_read(path, *args, **kwargs):
        if str(path) == "/proc/meminfo":
            return f"MemAvailable: {int(state['available'] * 1024**2)} kB\n"
        if str(path) == "/proc/pressure/io":
            return f"some avg10=99.0 avg60=0.0 total=0\nfull avg10={state['io']} avg60=0.0 total=0\n"
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fake_read)

    def fake_query(command, **kwargs):
        assert command[0] == "nvidia-smi"
        if "--query-gpu=index,uuid,memory.free" in command:
            return f"0, GPU-other, 1\n2, {state['uuid']}, {state['free_mib']}\n"
        assert "--query-compute-apps=gpu_uuid,pid,used_memory" in command
        return "GPU-other, 99, 23000\n" + (f"{pipeline.GPU_UUID}, 123, 1\n" if state["active"] else "")

    monkeypatch.setattr(pipeline.subprocess, "check_output", fake_query)
    return state


def test_resource_gate_checks_exact_idle_physical_gpu_and_host(resource_state):
    row = pipeline.snapshot()
    assert row["host_ready"] is row["gpu_ready"] is True
    assert row["gpu"] == 2 and row["gpu_uuid"] == pipeline.GPU_UUID
    assert row["compute_processes"] == []
    assert row["mem_available_gib"] == 36 and row["load1"] == 30


@pytest.mark.parametrize("field,value", [("active", True), ("free_mib", 12287),
    ("available", 35.999), ("load", 30.001), ("io", 20.0), ("disk", 19.999)])
def test_resource_gate_waits_for_each_required_resource(resource_state, field, value):
    resource_state[field] = value
    row = pipeline.snapshot()
    assert row["gpu_ready"] is False
    assert row["host_ready"] is (field in ("active", "free_mib"))


@pytest.mark.parametrize("field,value", [("host", "wrong-host"), ("uuid", "GPU-unexpected")])
def test_resource_gate_never_migrates_or_accepts_wrong_identity(resource_state, field, value):
    resource_state[field] = value
    with pytest.raises(RuntimeError):
        pipeline.snapshot()


def test_gpu_admission_requires_two_consecutive_checks_ten_seconds_apart(tmp_path, monkeypatch):
    observations = iter([True, False, True, True])
    seen, sleeps = [], []

    def snapshot():
        ready = next(observations)
        seen.append(ready)
        return dict(host_ready=True, gpu_ready=ready)

    monkeypatch.setattr(pipeline, "snapshot", snapshot)
    monkeypatch.setattr(pipeline.time, "sleep", sleeps.append)
    job = {}
    pipeline.admit(tmp_path, job, gpu=True)
    assert seen == [True, False, True, True]
    assert sleeps == [10, 60, 10]
    assert len(job["resource_checks"]) == 4
    assert all(row["requires_gpu"] for row in job["resource_checks"])


def test_cpu_report_admission_does_not_require_other_users_gpu_to_be_idle(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "snapshot", lambda: dict(host_ready=True, gpu_ready=False))
    monkeypatch.setattr(pipeline.time, "sleep", lambda seconds: pytest.fail("CPU report can proceed"))
    job = {}
    pipeline.admit(tmp_path, job, gpu=False)
    assert job["resource_checks"] == [dict(host_ready=True, gpu_ready=False, requires_gpu=False)]


def training_receipts(plan, arm, smoke):
    cfg = normalized(arm["config"])
    if smoke:
        cfg["training"].update(epochs=2, checkpoint_every=1, log_every=1)
    config_path = arm["smoke_config_path"] if smoke else arm["config_path"]
    write_json(config_path, cfg)
    run = Path(plan["output"]) / "smoke" / arm["name"] / "run" if smoke else Path(arm["root"]) / "run"
    records = {split: dict(hit=True, key=key, cache_version=1,
        path=str(Path(plan["mi_cache_dir"]) / f"mi-v1-{key}.pt")) for split, key in plan["mi_cache_keys"].items()}
    manifest = dict(config=cfg, config_source=config_path, data_sha256=plan["data_sha256"],
        training_driver="rsmi_separate_qcr", precision="torch.float64", estimator="histogram",
        environment_mode="full_shell", buffer=1, independent_c=True, labels_used_in_training=False,
        rsmi_weight=2., execution_device="cuda:2", source=None,
        execution_optimizations=dict(mi_statistics_cache=records))
    steps = 2 if smoke else pipeline.UPDATES
    policy = LearningRatePolicy(RunConfig.from_dict(cfg).training)
    summary = dict(config=cfg, actual_epochs=steps, requested_epochs=steps,
        final_report_skipped=True, lr_scheduler_state=policy.checkpoint_state(steps),
        history=[dict(epoch=n, learning_rate=policy.rate_for_update(n))
                 for n in ([1, 2] if smoke else [1, *range(200, steps + 1, 200)])])
    write_json(run / "manifest.json", manifest)
    write_json(run / "summary.json", summary)
    (run / f"training_step_{steps}.pt").write_bytes(b"not loaded: numbered checkpoint existence only")
    initialization = dict(fresh=True, source_checkpoint=None,
        C_tensor_sha256="fresh-C", Q_tensor_sha256="fresh-Q", R_tensor_sha256="fresh-R",
        full_model_tensor_sha256="fresh-all")
    write_json(run.parent / "initialization.json", initialization)
    if not smoke:
        write_json(Path(plan["output"]) / "smoke" / arm["name"] / "initialization.json", initialization)
    return run, manifest, summary


@pytest.mark.parametrize("smoke", [False, True])
@pytest.mark.parametrize("dim", [1, 5, 10, 100])
def test_training_receipt_requires_fresh_exact_endpoint_cache_hits(source, tmp_path, smoke, dim):
    plan = pipeline.build_plan(source[0], tmp_path / "fresh")
    arm = next(a for a in plan["arms"] if a["theory_dim"] == dim)
    training_receipts(plan, arm, smoke)
    row = pipeline.validate_training(plan, arm, smoke=smoke)
    assert row["actual_updates"] == (2 if smoke else 12000)
    assert row["smoke_only"] is smoke
    assert row["labels_used_in_training"] is False


@pytest.mark.parametrize("mutation", ["cache", "label", "source", "resume", "frozen", "steps",
                                      "wall_stop", "plateau", "lr", "checkpoint", "initialization",
                                      "history_endpoint", "history_lr", "config_source", "smoke_initialization"])
def test_training_receipt_rejects_protocol_or_completion_changes(source, tmp_path, mutation):
    plan = pipeline.build_plan(source[0], tmp_path / "fresh")
    arm = plan["arms"][-1]
    run, manifest, summary = training_receipts(plan, arm, False)
    if mutation == "cache": manifest["execution_optimizations"]["mi_statistics_cache"]["train"]["hit"] = False
    if mutation == "label": manifest["labels_used_in_training"] = True
    if mutation == "source": manifest["source"] = "historical.pt"
    if mutation == "resume": manifest["resumed_from"] = "historical.pt"
    if mutation == "frozen": manifest["frozen_C"] = True
    if mutation == "steps": summary["actual_epochs"] = 11999
    if mutation == "wall_stop": summary["stopped_by_wall_limit"] = True
    if mutation == "plateau": summary["stopped_by_train_plateau"] = True
    if mutation == "lr": summary["lr_scheduler_state"] = {}
    if mutation == "checkpoint": (run / "training_step_12000.pt").unlink()
    if mutation == "initialization": write_json(run.parent / "initialization.json", dict(fresh=False))
    if mutation == "history_endpoint": summary["history"][-1]["epoch"] = 11999
    if mutation == "history_lr": summary["history"][-1]["learning_rate"] = .001
    if mutation == "config_source": manifest["config_source"] = "other-config.json"
    if mutation == "smoke_initialization":
        initialization = pipeline.read(Path(plan["output"]) / "smoke" / arm["name"] / "initialization.json")
        initialization["Q_tensor_sha256"] = "wrong-smoke-construction"
        write_json(Path(plan["output"]) / "smoke" / arm["name"] / "initialization.json", initialization)
    write_json(run / "manifest.json", manifest)
    write_json(run / "summary.json", summary)
    with pytest.raises(RuntimeError):
        pipeline.validate_training(plan, arm, smoke=False)


def test_frozen_sources_code_and_configs_checked_without_loading_tensors(source, tmp_path):
    plan = pipeline.build_plan(source[0], tmp_path / "fresh")
    project = tmp_path / "frozen_code"
    project.mkdir()
    science = project / "science.py"
    science.write_text("original science\n")
    config = tmp_path / "frozen.json"
    write_json(config, dict(a=1))
    hashes, configs = {"science.py": pipeline.sha(science)}, {str(config): pipeline.sha(config)}
    pipeline.ensure_unchanged(plan, project, hashes, configs)
    science.write_text("changed science\n")
    with pytest.raises(RuntimeError, match="code changed"):
        pipeline.ensure_unchanged(plan, project, hashes, configs)
    science.write_text("original science\n")
    write_json(config, dict(a=2))
    with pytest.raises(RuntimeError, match="configuration"):
        pipeline.ensure_unchanged(plan, project, hashes, configs)


def mock_execution(plan, tmp_path, monkeypatch, fail_at=None):
    project = tmp_path / "project"
    script = project / "scripts" / "placeholder.py"
    script.parent.mkdir(parents=True)
    script.write_text("# tiny frozen stand-in; never executed\n")
    monkeypatch.setattr(pipeline, "__file__", str(project / "scripts" / "run_ising_highdim.py"))
    monkeypatch.setattr(pipeline, "source_hashes", lambda root: {"scripts/placeholder.py": pipeline.sha(script)})
    monkeypatch.setattr(pipeline.socket, "gethostname", lambda: pipeline.HOST)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    calls = []

    def fake_stage(root, job, name, command, **kwargs):
        calls.append(name)
        assert command and kwargs["check"] is not None
        if name.startswith("train_"):
            assert job.get("all_smoke_checks_passed") is True
            assert calls[:4] == [f"smoke_dim{dim}" for dim in pipeline.DIMENSIONS]
        row = dict(name=name, status="failed" if name == fail_at else "complete",
                   returncode=7 if name == fail_at else 0)
        job["stages"].append(row)
        if name == fail_at:
            raise RuntimeError(f"{name} returncode=7; no retry")
        if name.startswith("report_") or name == "comparison":
            folder = root / "comparison" if name == "comparison" else root / name.removeprefix("report_") / "report"
            write_json(folder / "report.json", {"status": "complete", "report_complete": True})

    monkeypatch.setattr(pipeline, "stage", fake_stage)
    return calls


def test_all_four_smokes_pass_before_any_formal_training_and_reports_are_serial(source, tmp_path, monkeypatch):
    plan = pipeline.build_plan(source[0], tmp_path / "fresh")
    calls = mock_execution(plan, tmp_path, monkeypatch)
    job = pipeline.execute(plan)
    expected = [f"smoke_dim{dim}" for dim in pipeline.DIMENSIONS]
    expected += [name for dim in pipeline.DIMENSIONS for name in (f"train_dim{dim}", f"report_dim{dim}")]
    assert calls == [*expected, "comparison"]
    assert job["status"] == "complete" and job["workflow_complete"] is True
    assert job["all_smoke_checks_passed"] is True
    assert job["completed_arms"] == [f"dim{dim}" for dim in pipeline.DIMENSIONS]
    assert job["physical_sampling_updates"] == 0
    for arm in plan["arms"]:
        config, smoke = pipeline.read(arm["config_path"]), pipeline.read(arm["smoke_config_path"])
        expected_smoke = deepcopy(config)
        expected_smoke["training"].update(epochs=2, checkpoint_every=1, log_every=1)
        assert smoke == expected_smoke
    assert all(Path(name).suffix != ".pt" for name in job["public_files"])
    with pytest.raises(FileExistsError):
        pipeline.execute(plan)
    assert calls == [*expected, "comparison"]


@pytest.mark.parametrize("failure", ["smoke_dim10", "train_dim5", "report_dim10", "comparison"])
def test_stage_failure_stops_later_stages_and_never_retries(source, tmp_path, monkeypatch, failure):
    plan = pipeline.build_plan(source[0], tmp_path / "fresh")
    calls = mock_execution(plan, tmp_path, monkeypatch, fail_at=failure)
    with pytest.raises(RuntimeError, match=failure):
        pipeline.execute(plan)
    job = pipeline.read(Path(plan["output"]) / "job.json")
    assert job["status"] == "failed" and job["workflow_complete"] is False
    assert job["requires_user_action"] is True
    assert calls[-1] == failure and calls.count(failure) == 1
    if failure.startswith("smoke_"):
        assert "all_smoke_checks_passed" not in job
        assert not any(name.startswith("train_") for name in calls)


def test_nonzero_child_has_no_retry_or_postcheck(tmp_path, monkeypatch):
    root = tmp_path / "run"
    (root / "logs").mkdir(parents=True)
    (root / "code").mkdir()
    starts, checks = [], []
    monkeypatch.setattr(pipeline, "ensure_unchanged", lambda *args: checks.append("frozen"))
    monkeypatch.setattr(pipeline, "admit", lambda *args, **kwargs: None)

    def popen(command, **kwargs):
        starts.append((command, kwargs))
        assert kwargs["cwd"] == root / "code"
        assert kwargs["env"]["PYTHONPATH"] == str(root / "code")
        return SimpleNamespace(pid=123, wait=lambda: 7)

    monkeypatch.setattr(pipeline.subprocess, "Popen", popen)
    job = dict(stages=[], GPU_stage_seconds=0)
    with pytest.raises(RuntimeError, match="no retry"):
        pipeline.stage(root, job, "smoke_dim100", ["never-executed"], gpu=True,
            plan={}, fingerprints={}, configs={}, check=lambda: pytest.fail("Cannot validate failed child"))
    assert len(starts) == 1
    assert checks == ["frozen", "frozen"]
    assert job["stages"][0]["returncode"] == 7
    assert job["stages"][0]["status"] == "failed"


def test_cache_only_entrypoint_cannot_fall_back_to_rebuilding_mi(tmp_path, monkeypatch):
    from experiments import rsmi_separate_qcr as trainer
    from recurrent_rg import mi_cache
    from scripts import report_ising_zh

    monkeypatch.setattr(report_ising_zh, "chinese_figures", nullcontext)

    def fake_cache(directory, key_data, build):
        build()
        pytest.fail("Miss must stop instead of computing MI")

    monkeypatch.setattr(mi_cache, "load_or_build_mi_statistics", fake_cache)
    original_model = trainer.build_model
    monkeypatch.setattr(trainer, "main", lambda args: mi_cache.load_or_build_mi_statistics(
        "existing", {"key": "missing"}, lambda: pytest.fail("MI must never be rebuilt")))
    with pytest.raises(RuntimeError, match="禁止重算"):
        pipeline.main(["--cached-training", "--output", str(tmp_path / "run"),
            "--config", str(tmp_path / "config.json"), "--device", "cuda:2",
            "--mi-cache-dir", str(tmp_path / "cache"),
            "--initialization-receipt", str(tmp_path / "initialization.json")])
    assert mi_cache.load_or_build_mi_statistics is fake_cache
    assert trainer.build_model is original_model


def test_cache_only_entrypoint_rejects_other_gpu(tmp_path):
    with pytest.raises(SystemExit):
        pipeline.main(["--cached-training", "--output", str(tmp_path / "run"),
            "--config", str(tmp_path / "config.json"), "--device", "cuda:0",
            "--mi-cache-dir", str(tmp_path / "cache"),
            "--initialization-receipt", str(tmp_path / "initialization.json")])


def complete_report_fixture(path, dim=100):
    files = ("metrics.json", "fixed_points.json", "critical_exponents.json", "encodings.npz",
        "trajectories.npz", "fixed_point_arrays.npz", "R_vs_g.png", "R_displacement.png",
        "coupling_by_scale.png", "coupling_all_coordinates.png", "loss_comparison.png",
        "representation_diagnostics.png", "fixed_point_spectrum.png", "status.json")
    path.mkdir()
    for name in files:
        (path / name).write_bytes(b"small mock public artifact")
    write_json(path / "report.json", dict(status="complete", report_complete=True,
        theory_dim=dim, checkpoint_epoch=12000))
    metrics = dict(theory_dim=dim, checkpoint_epoch=12000, labels_used_in_training=False,
        thermal_nu=None, freeze_verification=dict(model_state_unchanged=True,
            model_gradients_none=True, checkpoint_unchanged=True))
    write_json(path / "metrics.json", metrics)
    return files, metrics


def test_report_receipt_requires_all_public_results_and_freeze_verification(tmp_path):
    path = tmp_path / "report"
    files, _ = complete_report_fixture(path)
    receipt = pipeline.report_receipt(path, 100)
    assert set(receipt["files"]) == {*files, "report.json"}
    assert all(Path(name).suffix in (".json", ".npz", ".png") for name in receipt["files"])


@pytest.mark.parametrize("mutation", ["missing_flow", "symlink_flow", "wrong_dim", "wrong_endpoint",
    "supervised", "nu", "changed_model", "gradients", "changed_checkpoint"])
def test_report_receipt_rejects_incomplete_or_mutated_scientific_report(tmp_path, mutation):
    path = tmp_path / "report"
    _, metrics = complete_report_fixture(path)
    if mutation in ("missing_flow", "symlink_flow"):
        (path / "R_vs_g.png").unlink()
        if mutation == "symlink_flow":
            (path / "R_vs_g.png").symlink_to(path / "R_displacement.png")
    if mutation == "wrong_dim": metrics["theory_dim"] = 10
    if mutation == "wrong_endpoint": metrics["checkpoint_epoch"] = 11999
    if mutation == "supervised": metrics["labels_used_in_training"] = True
    if mutation == "nu": metrics["thermal_nu"] = 1.0
    if mutation == "changed_model": metrics["freeze_verification"]["model_state_unchanged"] = False
    if mutation == "gradients": metrics["freeze_verification"]["model_gradients_none"] = False
    if mutation == "changed_checkpoint": metrics["freeze_verification"]["checkpoint_unchanged"] = False
    write_json(path / "metrics.json", metrics)
    with pytest.raises(RuntimeError):
        pipeline.report_receipt(path, 100)
