"""Pure metadata/mock subprocess contracts; no physical sampling, CUDA or PT load."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_phi4_hopping_early_stopping as driver


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def complete(**extra):
    return dict(status="complete", workflow_complete=True, returncode=0,
                error=None, requires_user_action=False, **extra)


def clocks(step):
    return {name: dict(min=step, max=step, parameters=6) for name in ("Q", "C", "R")}


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    from experiments.phi4_hopping_qcr import LOSS_POLICY, model_policy
    source, code, old_code = [tmp_path / name for name in ("old", "code", "old_code")]
    old_pins = {}
    for relative in driver.CORE_SOURCES:
        for folder in (old_code, code):
            path = folder / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# frozen identical model/loss/data loader\n")
        old_pins[str(old_code / relative)] = driver.base.sha(old_code / relative)
    for relative in ("experiments/phi4_hopping_qcr.py", "scripts/report_phi4_hopping_qcr.py"):
        path = old_code / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# frozen old trainer/report\n")
        old_pins[str(path)] = driver.base.sha(path)
    (code / "recurrent_rg/cli.py").write_text("# copied CLI\n")
    data_plan = dict(schema_version=5, labels_used_in_training=False)
    write(source / "data/plan.json", data_plan)
    write(source / "data/manifest.json", dict(plan_sha256=driver.base.sha(source / "data/plan.json")))
    data_sha = driver.base.sha(source / "data/manifest.json")
    write(source / "data/status.json", complete(manifest_sha256=data_sha))
    init = "a" * 64
    options = dict(steps=3000, system_batch=32, sample_batch=64, seed=91, threads=1,
        log_every=100, checkpoint_every=1000, information_pairs=128,
        max_cuda_bytes=driver.base.PEAK_BYTES, block_size=2, depth=3,
        device="cuda:0", max_training_seconds=57000., expected_gpu_uuid=driver.GPU_UUID)
    runtime = dict(schema_version=5, recipe="phi4_hopping_raw_patch_joint625_v1",
        data=str(source / "data"), output=str(source / "training"),
        model=model_policy(2, 3), loss=copy.deepcopy(LOSS_POLICY),
        training={key: value for key, value in options.items() if key not in {"block_size", "depth"}})
    write(source / "training_runtime_config.json", runtime)
    manifest = dict(recipe=runtime["recipe"], source_sha256=old_pins,
        labels_used_in_training=False, closure_objective="all_start_anchor", closure_scale_weights=None,
        theory_dim=1, lattice_sizes=[128, 64, 32, 16], options=options,
        data_manifest_sha256=data_sha, data_warning_count=75,
        q_initialization_receipt=dict(training_start_state_sha256=init))
    write(source / "training/manifest.json", manifest)
    summary = dict(status="complete", recipe=runtime["recipe"], actual_steps=3000,
        optimizer_steps=clocks(3000), labels_used_in_training=False, thermal_nu=None,
        theory_dim=1, lattice_sizes=[128, 64, 32, 16], data_manifest_sha256=data_sha,
        q_initialization_receipt=dict(training_start_state_sha256=init), timing=dict(training_seconds=36944.))
    write(source / "training/summary.json", summary)
    write(source / "training/status.json", complete(actual_steps=3000, latest_checkpoint_step=3000,
        summary_sha256=driver.base.sha(source / "training/summary.json")))
    write(source / "plan.json", dict(code_root=str(old_code)))
    write(source / "job.json", complete(formal_optimizer_updates=3000,
        stages=[dict(name="training", status="complete", returncode=0)],
        data_manifest_sha256=data_sha, plan_sha256=driver.base.sha(source / "plan.json"),
        cpu_seconds=50749., gpu_seconds=37139., directory_bytes=1021564111,
        new_directory_bytes=1020802866))
    monkeypatch.setattr(driver, "_code_root", lambda: code)
    return SimpleNamespace(source=source, code=code, old_code=old_code,
                           output=tmp_path / "fresh", initial=init, data_sha=data_sha)


def test_dry_plan_changes_only_termination_logging_paths_and_initial_binding(evidence, monkeypatch):
    monkeypatch.setattr(driver.subprocess, "Popen", lambda *_a, **_k: pytest.fail("dry spawned child"))
    value = driver.plan(evidence.source, evidence.output)
    assert not evidence.output.exists()
    assert value["stages"] == ["training"]
    assert value["no_sampling"] and value["no_benchmark"] and value["no_resume"]
    training = value["training_config"]["training"]
    assert training["steps"] == 200 and training["seed"] == 91
    assert training["system_batch"] == 32 and training["sample_batch"] == 64
    assert training["log_every"] == 10 and training["checkpoint_every"] == 100
    assert training["early_stopping"] == driver.POLICY
    assert training["expected_initial_state_sha256"] == evidence.initial
    assert value["training_config"]["data"] == str(evidence.source / "data")
    assert value["prior_evidence"]["projection_seconds"] == pytest.approx(3814.4)
    assert value["resources"]["workflow_seconds"] == 5400.


def test_existing_or_nested_output_fails_closed(evidence):
    evidence.output.mkdir()
    with pytest.raises(ValueError, match="fresh output"):
        driver.plan(evidence.source, evidence.output)
    with pytest.raises(ValueError, match="fresh output"):
        driver.plan(evidence.source, evidence.source / "new")


@pytest.mark.parametrize("relative,field,value", [
    ("job.json", "status", "failed"),
    ("job.json", "formal_optimizer_updates", 2999),
    ("training/status.json", "actual_steps", 2999),
    ("training/manifest.json", "labels_used_in_training", True),
    ("training/summary.json", "thermal_nu", 1.),
])
def test_source_requires_complete_label_free_endpoint(evidence, relative, field, value):
    path = evidence.source / relative
    data = json.loads(path.read_text()); data[field] = value; write(path, data)
    with pytest.raises(RuntimeError):
        driver.source_evidence(evidence.source, code=evidence.code)


def test_core_source_change_or_old_frozen_source_change_rejected(evidence):
    path = evidence.code / driver.CORE_SOURCES[0]
    path.write_text("# silently changed model\n")
    with pytest.raises(RuntimeError, match="Core model"):
        driver.source_evidence(evidence.source, code=evidence.code)
    path.write_bytes((evidence.old_code / driver.CORE_SOURCES[0]).read_bytes())
    (evidence.old_code / driver.CORE_SOURCES[1]).write_text("# changed old evidence\n")
    with pytest.raises(RuntimeError, match="Immutable old source"):
        driver.source_evidence(evidence.source, code=evidence.code)


def test_data_metadata_binding_is_checked_without_opening_array(evidence):
    (evidence.source / "data/do_not_read.npz").write_bytes(b"invalid array deliberately")
    value = driver.source_evidence(evidence.source, code=evidence.code)
    assert value["physical_data_deserialized"] is False and value["old_model_loaded"] is False
    write(evidence.source / "data/plan.json", dict(changed=True))
    with pytest.raises(RuntimeError, match="physical data"):
        driver.source_evidence(evidence.source, code=evidence.code)


def test_cumulative_budget_includes_prior_cpu_gpu_and_projected_work():
    assert driver.budget_ok(50749., 37139., 1021564111, remaining=5460.)
    assert not driver.budget_ok(81000., 37139., 1021564111, remaining=5460.)
    assert not driver.budget_ok(50749., 53000., 1021564111, remaining=5460.)
    assert not driver.budget_ok(1., 1., driver.DIRECTORY_BYTES + 1)
    assert not driver.budget_ok(float("nan"), 1., 1.)


def mock_run_environment(monkeypatch):
    monkeypatch.setattr(driver.socket, "gethostname", lambda: driver.base.HOST)
    monkeypatch.setattr(driver.os, "getpriority", lambda *_: 10)
    monkeypatch.setattr(driver.base, "host_resources", lambda: dict(cpus=24, load1=1.,
        mem_available_bytes=100 * 1024**3, io_full_avg10=0.))
    reads = []
    def gpu(index, uuid):
        reads.append((index, uuid))
        return dict(index=index, uuid=uuid, compute_pids=[], free_bytes=24 * 1024**3)
    monkeypatch.setattr(driver.base, "gpu_resources", gpu)
    clock = [0.]
    monkeypatch.setattr(driver.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(driver.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    from scripts import report_phi4_hopping_qcr
    monkeypatch.setattr(report_phi4_hopping_qcr, "select_chinese_font", lambda _: Path("/mock/font"))
    return reads, clock


def test_parent_runs_only_one_training_child_on_twice_idle_pinned_gpu(evidence, monkeypatch):
    reads, clock = mock_run_environment(monkeypatch)
    before = {str(path): driver.base.sha(path) for path in evidence.source.rglob("*") if path.is_file()}
    calls = []
    def popen(command, **kwargs):
        calls.append((command, kwargs))
        write(evidence.output / "training/status.json", complete(actual_steps=120, public_files=[]))
        return SimpleNamespace(pid=999999999, returncode=0, poll=lambda: 0)
    monkeypatch.setattr(driver.subprocess, "Popen", popen)
    monkeypatch.setattr(driver, "validate_training", lambda *_: dict(actual_steps=120,
        evaluated_checkpoint_step=100, selected_checkpoint_sha256="b" * 64))
    assert driver.run(evidence.source, evidence.output) == 0
    assert len(calls) == 1 and len(reads) == 2 and clock[0] >= 10.
    assert all(row == (3, driver.GPU_UUID) for row in reads)
    assert calls[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == driver.GPU_UUID
    assert calls[0][0][3:5] == ["recurrent_rg", "train"]
    job = json.loads((evidence.output / "job.json").read_text())
    assert job["formal_optimizer_updates"] == 120 and job["evaluated_checkpoint_step"] == 100
    assert job["gpu_seconds"] >= 37149. and job["additional_sampling"] == 0
    assert before == {str(path): driver.base.sha(path) for path in evidence.source.rglob("*") if path.is_file()}


def test_cumulative_budget_hold_never_spawns_or_resamples(evidence, monkeypatch):
    mock_run_environment(monkeypatch)
    path = evidence.source / "job.json"
    value = json.loads(path.read_text()); value["gpu_seconds"] = 56000.; write(path, value)
    monkeypatch.setattr(driver.subprocess, "Popen", lambda *_a, **_k: pytest.fail("budget hold spawned"))
    assert driver.run(evidence.source, evidence.output) == 3
    value = json.loads((evidence.output / "job.json").read_text())
    assert value["status"] == "budget_hold" and value["formal_training_started"] is False


def final_run(evidence):
    folder = evidence.output / "training"
    folder.mkdir(parents=True)
    actual, selected = 120, 100
    checkpoint = folder / "training_step_100.pt"
    checkpoint.write_bytes(b"mock checkpoint, never loaded")
    early = dict(actual_steps=actual, evaluated_checkpoint_step=selected, best_step=selected,
        selected_checkpoint_sha256=driver.base.sha(checkpoint), test_data_used=False, labels_used=False)
    manifest = dict(options=dict(early_stopping=driver.POLICY, steps=200,
        expected_initial_state_sha256=evidence.initial), closure_objective="all_start_anchor",
        closure_scale_weights=None, lattice_sizes=[128, 64, 32, 16], labels_used_in_training=False,
        data_manifest_sha256=evidence.data_sha,
        q_initialization_receipt=dict(training_start_state_sha256=evidence.initial))
    summary = dict(status="complete", actual_steps=actual, evaluated_checkpoint_step=selected,
        latest_checkpoint_step=actual, labels_used_in_training=False, early_stopping=early,
        thermal_nu=None, frozen_evaluation_verified=True, peak_cuda_bytes=1000,
        optimizer_steps=clocks(actual), selected_optimizer_steps=clocks(selected),
        data_manifest_sha256=evidence.data_sha,
        q_initialization_receipt=dict(training_start_state_sha256=evidence.initial))
    for name, value in (("manifest.json", manifest), ("summary.json", summary),
                        ("history.json", {}), ("plan.json", {}), ("early_stopping.json", early)):
        write(folder / name, value)
    (folder / "evaluation_arrays.npz").write_bytes(b"mock public array, no inference")
    report = dict(status="complete", actual_steps=actual, evaluated_checkpoint_step=selected,
        early_stopping=early, source_sha256={name: driver.base.sha(folder / name)
            for name in ("manifest.json", "summary.json", "history.json", "evaluation_arrays.npz")})
    write(folder / "report/report.json", report)
    for name in ("R_vs_g.png", "coupling_by_scale.png", "loss_curves.png", "all_start_closure.png"):
        (folder / "report" / name).write_bytes(b"mock public figure")
    public = [dict(path=str(path.relative_to(folder)), sha256=driver.base.sha(path))
        for path in folder.rglob("*") if path.is_file() and path.suffix in driver.PUBLIC_SUFFIXES]
    write(folder / "status.json", complete(actual_steps=actual, latest_checkpoint_step=actual,
        summary_sha256=driver.base.sha(folder / "summary.json"),
        evaluation_arrays_sha256=driver.base.sha(folder / "evaluation_arrays.npz"), public_files=public))
    return folder


def test_completion_distinguishes_actual_selected_and_checks_public_sha(evidence):
    folder = final_run(evidence)
    result = driver.validate_training(evidence.output, dict(initial_state_sha256=evidence.initial,
        data_manifest_sha256=evidence.data_sha))
    assert result["actual_steps"] == 120 and result["evaluated_checkpoint_step"] == 100
    assert all(Path(name).suffix in driver.PUBLIC_SUFFIXES for name in result["public_files"])
    (folder / "report/R_vs_g.png").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="public-file SHA"):
        driver.validate_training(evidence.output, dict(initial_state_sha256=evidence.initial,
            data_manifest_sha256=evidence.data_sha))


def test_public_paths_reject_pt_and_traversal(tmp_path):
    (tmp_path / "x.pt").write_bytes(b"private")
    with pytest.raises(RuntimeError, match="not JSON/PNG/NPZ"):
        driver._inside(tmp_path, "x.pt")
    with pytest.raises(RuntimeError):
        driver._inside(tmp_path, "../outside.json")
