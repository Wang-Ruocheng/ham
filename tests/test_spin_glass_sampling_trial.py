"""Sampling-only pilot preflight must be fixed, bounded, and non-destructive."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.run_spin_glass_sampling_trial as trial
from scripts.run_spin_glass_sampling_trial import MAX_OUTPUT_BYTES, REFERENCE, build_plan


def test_formal_plan_uses_same_bonds_and_matched_fresh_controls():
    root = Path(__file__).resolve().parents[1]
    plan = build_plan(root / REFERENCE)
    assert plan["training_updates"] == plan["model_inference_calls"] == 0
    assert plan["total_single_spin_visits"] == 20_384_317_440
    assert plan["replicas"] == 4
    assert len(plan["cases"]) == 12
    assert len({row["sampling_seed"] for row in plan["cases"]}) == 12
    assert {row["disorder_id"] for row in plan["cases"]} == {0, 1, 7}
    assert all(len(row["expected_bond_sha256"]) == 64 for row in plan["cases"])
    assert plan["projected_raw_array_bytes"] < MAX_OUTPUT_BYTES
    for disorder in plan["disorder_ids"]:
        cases = [row for row in plan["cases"] if row["disorder_id"] == disorder]
        assert {(row["algorithm"], row["initial"]) for row in cases} == {
            (algorithm, initial) for algorithm in ("pt", "houdayer_pt") for initial in ("hot", "cold")}
        assert len({row["expected_bond_sha256"] for row in cases}) == 1


def test_smoke_is_explicitly_not_scientific_and_does_not_write(tmp_path):
    root = Path(__file__).resolve().parents[1]
    plan = build_plan(root / REFERENCE, smoke=True)
    assert plan["execution_smoke"] and not plan["scientific_result"]
    assert plan["shape"] == [6, 6]
    assert all(row["expected_bond_sha256"] is None for row in plan["cases"])
    assert not list(tmp_path.iterdir())


def test_unfinished_historical_reference_rejected(tmp_path):
    reference = tmp_path / "manifest.json"
    reference.write_text('{"status": "running"}')
    with pytest.raises(ValueError, match="completed historical"):
        build_plan(reference)


@pytest.fixture
def mocked_trial(tmp_path, monkeypatch):
    """Exercise orchestration without launching sampling, reports, or training."""
    repository = Path(__file__).resolve().parents[1]
    plan = build_plan(repository / REFERENCE, smoke=True)
    plan["cases"] = plan["cases"][:1]
    reference = tmp_path / "historical_manifest.json"
    historical_bytes = (repository / REFERENCE).read_bytes()
    reference.write_bytes(historical_bytes)
    root = tmp_path / "new_trial"
    state = dict(cpu=0., returncode=0, report_returncode=0,
                 after_child_cpu=0., after_report_cpu=0.,
                 peak_rss_native_units=1024, launches=[], reports=[])
    for variable in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(variable, "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(trial, "build_plan", lambda *args, **kwargs: copy.deepcopy(plan))
    monkeypatch.setattr(trial, "cpu_seconds", lambda: state["cpu"])
    monkeypatch.setattr(trial, "resource_snapshot", lambda: {"ready": True})

    def freeze(_source, target):
        target.mkdir()
        pinned = target / "scripts" / "run_spin_glass_sampling_trial.py"
        pinned.parent.mkdir()
        pinned.write_bytes(Path(trial.__file__).read_bytes())
        fingerprints = {str(pinned.relative_to(target)): trial.sha(pinned)}
        trial.write_json(target / "fingerprints.json", fingerprints)
        return fingerprints

    monkeypatch.setattr(trial, "freeze_code", freeze)

    class Child:
        pid = 12345

        def __init__(self, command, **kwargs):
            state["launches"].append((command, kwargs))

        def wait(self):
            state["cpu"] = max(state["cpu"], state["after_child_cpu"])
            if not state["returncode"]:
                key = plan["cases"][0]["key"]
                trial.write_json(root / "cases" / f"{key}.json", dict(
                    key=key, status="complete", returncode=0,
                    sampling_cpu_seconds=0., jit_execution_check_cpu_seconds=0.,
                    peak_rss_native_units=state["peak_rss_native_units"], peak_rss_bytes=1024 * 1024,
                    peak_rss_units="KiB on Linux; bytes on macOS"))
            return state["returncode"]

    def report(command, **kwargs):
        state["reports"].append((command, kwargs))
        state["cpu"] = max(state["cpu"], state["after_report_cpu"])
        return state["report_returncode"]

    monkeypatch.setattr(trial.subprocess, "Popen", Child)
    monkeypatch.setattr(trial.subprocess, "call", report)
    return SimpleNamespace(root=root, reference=reference, plan=plan, state=state,
                           historical_bytes=historical_bytes)


def test_existing_output_root_is_rejected_without_modification(mocked_trial):
    fixture = mocked_trial
    fixture.root.mkdir()
    sentinel = fixture.root / "existing_result.json"
    sentinel.write_bytes(b"historical result must survive\n")
    with pytest.raises(FileExistsError):
        trial.run_trial(fixture.root, fixture.reference, smoke=True)
    assert list(fixture.root.iterdir()) == [sentinel]
    assert sentinel.read_bytes() == b"historical result must survive\n"
    assert fixture.reference.read_bytes() == fixture.historical_bytes
    assert not fixture.state["launches"]


@pytest.mark.parametrize("extension", ["json", "npz"])
def test_existing_case_receipt_or_samples_are_not_overwritten(mocked_trial, extension):
    fixture = mocked_trial
    fixture.root.mkdir()
    (fixture.root / "cases").mkdir()
    trial.write_json(fixture.root / "plan.json", fixture.plan)
    key = fixture.plan["cases"][0]["key"]
    existing = fixture.root / "cases" / f"{key}.{extension}"
    existing.write_bytes(b"partial or complete samples are preserved\n")
    with pytest.raises(FileExistsError, match="no resume or overwrite"):
        trial.run_case(fixture.root, key)
    assert existing.read_bytes() == b"partial or complete samples are preserved\n"
    assert not fixture.state["launches"]


def test_initial_cpu_projection_rejects_before_creating_output(mocked_trial):
    fixture = mocked_trial
    fixture.plan["budgets"]["initial_projected_cpu_seconds"] = trial.MAX_CPU_SECONDS + 1
    with pytest.raises(RuntimeError, match="CPU projection exceeds budget"):
        trial.run_trial(fixture.root, fixture.reference, smoke=True)
    assert not fixture.root.exists()
    assert not fixture.state["launches"]


def test_thread_environment_mismatch_rejects_before_creating_output(mocked_trial, monkeypatch):
    fixture = mocked_trial
    monkeypatch.setenv("NUMBA_NUM_THREADS", "2")
    with pytest.raises(RuntimeError, match="single CPU thread"):
        trial.run_trial(fixture.root, fixture.reference, smoke=True)
    assert not fixture.root.exists()
    assert not fixture.state["launches"]


@pytest.mark.parametrize("operation", ["change", "delete"])
def test_frozen_source_changes_or_missing_files_are_rejected(tmp_path, operation):
    pinned = tmp_path / "pinned.py"
    pinned.write_text("# original\n")
    fingerprints = {pinned.name: trial.sha(pinned)}
    if operation == "change":
        pinned.write_text("# modified\n")
    else:
        pinned.unlink()
    with pytest.raises((RuntimeError, FileNotFoundError)):
        trial.verify_code(tmp_path, fingerprints)


def test_source_fingerprint_failure_never_launches_a_child(mocked_trial, monkeypatch):
    def changed_source(*_args):
        raise RuntimeError("Frozen source fingerprint mismatch")

    monkeypatch.setattr(trial, "verify_code", changed_source)
    fixture = mocked_trial
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        trial.run_trial(fixture.root, fixture.reference, smoke=True)
    job = json.loads((fixture.root / "job.json").read_text())
    assert job["status"] == "failed"
    assert not job["workflow_complete"]
    assert job["requires_user_action"]
    assert not fixture.state["launches"]
    assert not fixture.state["reports"]


def test_resource_hold_does_not_launch_sampling_or_report(mocked_trial, monkeypatch):
    monkeypatch.setattr(trial, "resource_snapshot", lambda: {"ready": False})
    fixture = mocked_trial
    job = trial.run_trial(fixture.root, fixture.reference, smoke=True)
    assert job["status"] == "resource_hold"
    assert not job["workflow_complete"]
    assert job["requires_user_action"]
    assert job["training_updates"] == job["model_inference_calls"] == 0
    assert not job["stages"]
    assert not fixture.state["launches"]
    assert not fixture.state["reports"]
    assert fixture.reference.read_bytes() == fixture.historical_bytes


def test_output_budget_hold_does_not_launch_sampling(mocked_trial, monkeypatch):
    monkeypatch.setattr(trial, "directory_bytes", lambda _root: trial.MAX_OUTPUT_BYTES + 1)
    fixture = mocked_trial
    job = trial.run_trial(fixture.root, fixture.reference, smoke=True)
    assert job["status"] == "budget_hold"
    assert not job["workflow_complete"]
    assert not fixture.state["launches"]
    assert not fixture.state["reports"]


def test_nonzero_sampling_child_never_completes_or_starts_report(mocked_trial):
    fixture = mocked_trial
    fixture.state["returncode"] = 7
    with pytest.raises(RuntimeError, match="returned 7"):
        trial.run_trial(fixture.root, fixture.reference, smoke=True)
    job = json.loads((fixture.root / "job.json").read_text())
    assert job["status"] == "failed"
    assert not job["workflow_complete"]
    assert job["stages"][0]["returncode"] == 7
    assert job["stages"][0]["status"] == "failed"
    assert not fixture.state["reports"]
    assert fixture.reference.read_bytes() == fixture.historical_bytes


def test_nonzero_report_never_marks_workflow_complete(mocked_trial):
    fixture = mocked_trial
    fixture.state["report_returncode"] = 9
    with pytest.raises(RuntimeError, match="Report returned 9"):
        trial.run_trial(fixture.root, fixture.reference, smoke=True)
    job = json.loads((fixture.root / "job.json").read_text())
    assert job["status"] == "failed"
    assert not job["workflow_complete"]
    assert job["stages"][-1]["returncode"] == 9
    assert job["stages"][-1]["status"] == "failed"
    assert len(fixture.state["launches"]) == len(fixture.state["reports"]) == 1


def test_success_only_launches_snapshot_sampling_and_readonly_report(mocked_trial):
    fixture = mocked_trial
    job = trial.run_trial(fixture.root, fixture.reference, smoke=True)
    assert job["status"] == "complete" and job["workflow_complete"]
    assert job["training_updates"] == job["model_inference_calls"] == 0
    assert not job["equilibrium_certified"]
    assert job["launch_source_sha256"] == job["frozen_runner_sha256"]
    command, arguments = fixture.state["launches"][0]
    assert command[1:3] == ["-m", "scripts.run_spin_glass_sampling_trial"]
    assert "--case-key" in command
    assert arguments["cwd"] == fixture.root / "code"
    report_command, report_arguments = fixture.state["reports"][0]
    assert report_command[1:3] == ["-m", "scripts.report_spin_glass_sampling_trial"]
    assert report_arguments["cwd"] == fixture.root / "code"
    assert all("train" not in argument for argument in command[1:])
    assert fixture.reference.read_bytes() == fixture.historical_bytes


def test_case_cpu_accounting_includes_whole_child_not_receipt_sampling_only(mocked_trial):
    fixture = mocked_trial
    fixture.state["after_child_cpu"] = 123.
    job = trial.run_trial(fixture.root, fixture.reference, smoke=True)
    assert job["status"] == "complete"
    assert job["stages"][0]["case_cpu_seconds"] == 123.
    assert job["cpu_core_seconds"] == 123.


def test_measured_case_memory_overshoot_stops_before_report(mocked_trial):
    fixture = mocked_trial
    # More than the byte cap even when native ru_maxrss units are bytes.
    fixture.state["peak_rss_native_units"] = fixture.plan["budgets"]["memory_bytes"] + 1
    job = trial.run_trial(fixture.root, fixture.reference, smoke=True)
    assert job["status"] == "budget_hold"
    assert not job["workflow_complete"]
    assert job["requires_user_action"]
    assert not fixture.state["reports"]


def test_postreport_cpu_overshoot_cannot_claim_completion(mocked_trial):
    fixture = mocked_trial
    fixture.state["after_report_cpu"] = trial.MAX_CPU_SECONDS + 1
    job = trial.run_trial(fixture.root, fixture.reference, smoke=True)
    assert job["status"] == "budget_hold"
    assert not job["workflow_complete"]
    assert job["requires_user_action"]
    assert job["cpu_core_seconds"] > trial.MAX_CPU_SECONDS


def test_postreport_output_overshoot_cannot_claim_completion(mocked_trial, monkeypatch):
    fixture = mocked_trial
    monkeypatch.setattr(trial, "directory_bytes", lambda _root:
        trial.MAX_OUTPUT_BYTES + 1 if fixture.state["reports"] else 0)
    job = trial.run_trial(fixture.root, fixture.reference, smoke=True)
    assert job["status"] == "budget_hold"
    assert not job["workflow_complete"]
    assert job["requires_user_action"]
    assert job["directory_bytes"] > trial.MAX_OUTPUT_BYTES
