"""Budget-only continuation contracts; no CUDA, model updates or physical chains."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import continue_phi4_hopping_budget as continuation
from recurrent_rg.data.phi4_hopping_pipeline import Phi4HoppingOptions, generation_plan


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/phi4_hopping_b3_g1_aligned_seed91.json"


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False))


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    old_code, code, prior, queue = [tmp_path / name for name in ("old_code", "code", "prior", "queue")]
    for folder in (old_code, code, prior, queue):
        folder.mkdir()
    pins = {}
    for relative in ("scripts/run_phi4_hopping_qcr.py", "recurrent_rg/phi4_aligned_qcr.py"):
        for folder in (old_code, code):
            (folder / relative).parent.mkdir(exist_ok=True)
            (folder / relative).write_text("# frozen computational source\n")
        pins[str(old_code / relative)] = continuation.base.sha(old_code / relative)
    (code / "scripts/continue_phi4_hopping_budget.py").write_text("# new orchestration only\n")
    (old_code / "scripts/._run_phi4_hopping_qcr.py").write_bytes(b"nonexecutable metadata")
    pins[str(old_code / "scripts/._run_phi4_hopping_qcr.py")] = continuation.base.sha(old_code / "scripts/._run_phi4_hopping_qcr.py")
    old_config = old_code / "config.json"
    old_config.write_bytes(CONFIG.read_bytes())
    pins[str(old_config)] = continuation.base.sha(old_config)
    (prior / "training_config.json").write_bytes(CONFIG.read_bytes())
    pins[str(prior / "training_config.json")] = continuation.base.sha(prior / "training_config.json")
    old_plan = dict(code_root=str(old_code), output=str(prior), host=continuation.base.HOST,
        GPU=dict(index=3, uuid=continuation.GPU_UUID), config_source=str(old_config),
        config_source_sha256=continuation.base.sha(old_config), source_sha256=pins,
        physical=generation_plan(Phi4HoppingOptions()))
    write(prior / "plan.json", old_plan)
    write(prior / "gpu_preflight.json", dict(status="budget_hold", returncode=3,
        workflow_complete=True, error=None, synthetic=True, physical_result=False,
        checkpoint_saved=False, formal_model_reused=False, physical_data_read=False,
        actual_updates=7, training_batch=[32,64,36,36,1], training_cap_seconds=1800.,
        peak_cuda_bytes=1522532352, device=dict(CUDA_VISIBLE_DEVICES=continuation.GPU_UUID),
        optimizer_steps={name:dict(min=7,max=7,parameters=1) for name in ("Q","C","R")},
        conservative_full_training_projection_seconds=12669.778172127903))
    monkeypatch.setattr(continuation, "PREFLIGHT_SHA256", continuation.base.sha(prior / "gpu_preflight.json"))
    job = dict(status="budget_hold", workflow_complete=False, parent_pid=None, returncode=3,
        current=None, requires_user_action=True, cpu_seconds=25.896719011,
        gpu_seconds=50.38484049588442, plan_sha256=continuation.base.sha(prior / "plan.json"),
        stages=[dict(name="gpu_preflight", status="budget_hold", returncode=3, pid=None)])
    write(prior / "job.json", job)
    write(queue / "plan.json", {"queue": True})
    write(queue / "job.json", dict(status="cancelled_before_compute", stages=[], current=None,
        gpu_seconds=0., cpu_seconds=.185175389, error=None,
        plan_sha256=continuation.base.sha(queue / "plan.json")))
    new_config = tmp_path / "budget4h.json"
    value = json.loads(CONFIG.read_text())
    value["training"]["max_training_seconds"] = 14400
    write(new_config, value)
    monkeypatch.setattr(continuation, "_code_root", lambda: code)
    return SimpleNamespace(prior=prior, queue=queue, config=new_config, code=code,
                           old_code=old_code, output=tmp_path / "fresh")


def prerequisites(row):
    return continuation.prerequisites(row.prior, row.config, 3, continuation.GPU_UUID, row.queue)


def test_bind_budget_change_and_charge_all_prior_use(evidence):
    row = prerequisites(evidence)
    assert row["prior_cpu_seconds"] == pytest.approx(26.0818944)
    assert row["prior_gpu_seconds"] == pytest.approx(50.38484049588442)
    assert row["projection_seconds"] == pytest.approx(12669.778172127903)
    assert row["original_receipt_status"] == "budget_hold" and row["original_receipt_returncode"] == 3
    assert row["new_budget_gate"] == "pass" and row["additional_benchmark_updates"] == 0
    assert row["checkpoint_loaded"] is False and row["formal_model_reused"] is False
    assert row["executable_source_files"] == 2 and row["excluded_appledouble_metadata_files"] == 1


def test_dry_plan_runs_no_compute_and_creates_nothing(evidence, monkeypatch, capsys):
    monkeypatch.setattr(continuation, "run", lambda *_: pytest.fail("Dry plan executed"))
    assert continuation.main(["--prior-root", str(evidence.prior), "--prior-queue-root", str(evidence.queue),
        "--config", str(evidence.config), "--output", str(evidence.output)]) == 0
    row = json.loads(capsys.readouterr().out)
    assert not evidence.output.exists() and row["execution_requested"] is False
    assert row["stages"] == ["sampling", "training"]
    assert set(row["commands"]) == {"sampling", "training"}
    assert row["resources"]["cpu_seconds"] == 21600
    assert row["resources"]["gpu_card_seconds"] == 14400
    assert row["training"]["options"]["steps"] == 3000
    assert row["training"]["options"]["max_training_seconds"] == 14400
    assert row["training"]["options"]["system_batch"] == 32
    assert row["training"]["options"]["sample_batch"] == 64
    assert row["physical"]["options"]["samples"] == 128
    assert row["physical"]["options"]["chains"] == 4
    assert row["commands"]["training"][2:5] == ["-m", "recurrent_rg", "train"]
    assert "--resume" not in str(row["commands"]) and "--allow-legacy-training" not in str(row["commands"])


@pytest.mark.parametrize("section,key,value", [
    ("training", "steps", 3001), ("training", "system_batch", 8),
    ("training", "seed", 92), ("training", "max_training_seconds", 1800),
    ("model", "theory_dim", 2), ("loss", "commutator", 6),
    ("loss", "closure_scale_weights", [1.8,.2]),
])
def test_reject_any_computational_config_change(evidence, section, key, value):
    config = json.loads(evidence.config.read_text())
    config[section][key] = value
    write(evidence.config, config)
    with pytest.raises(ValueError, match="Only the authorized"):
        prerequisites(evidence)


@pytest.mark.parametrize("field,value", [
    ("status", "failed"), ("returncode", 1), ("workflow_complete", True),
    ("current", "sampling"), ("requires_user_action", False),
])
def test_reject_nonterminal_or_wrong_prior_job(evidence, field, value):
    job = json.loads((evidence.prior / "job.json").read_text())
    job[field] = value
    write(evidence.prior / "job.json", job)
    with pytest.raises(ValueError, match="exclusive completed"):
        prerequisites(evidence)


@pytest.mark.parametrize("folder", ["data", "training"])
def test_refuse_if_production_started_even_empty_directory(evidence, folder):
    (evidence.prior / folder).mkdir()
    with pytest.raises(ValueError, match="exclusive completed"):
        prerequisites(evidence)


def test_receipt_is_bound_by_exact_sha_and_not_falsified(evidence):
    receipt = json.loads((evidence.prior / "gpu_preflight.json").read_text())
    receipt["status"], receipt["returncode"] = "complete", 0
    write(evidence.prior / "gpu_preflight.json", receipt)
    with pytest.raises(ValueError, match="Synthetic receipt"):
        prerequisites(evidence)


def test_prior_plan_is_sha_bound(evidence):
    plan = json.loads((evidence.prior / "plan.json").read_text())
    plan["physical"]["options"]["samples"] = 64
    write(evidence.prior / "plan.json", plan)
    with pytest.raises(ValueError, match="exclusive completed"):
        prerequisites(evidence)


@pytest.mark.parametrize("location", ["new", "old"])
def test_both_old_and_new_computational_sources_are_frozen(evidence, location):
    folder = evidence.code if location == "new" else evidence.old_code
    (folder / "recurrent_rg/phi4_aligned_qcr.py").write_text("# altered\n")
    with pytest.raises(ValueError, match="source"):
        prerequisites(evidence)


def test_added_executable_source_is_not_silently_accepted(evidence):
    (evidence.code / "scripts/extra.py").write_text("# unreviewed\n")
    with pytest.raises(ValueError, match="inventory"):
        prerequisites(evidence)


@pytest.mark.parametrize("field,value", [("stages", [{"name":"sampling"}]), ("gpu_seconds", 1),
    ("status", "running"), ("cpu_seconds", float("inf"))])
def test_queue_must_have_zero_compute_and_finite_charge(evidence, field, value):
    path = evidence.queue / "job.json"
    row = json.loads(path.read_text())
    row[field] = value
    write(path, row)
    with pytest.raises(ValueError):
        prerequisites(evidence)


def test_exclusive_output(evidence):
    evidence.output.mkdir()
    with pytest.raises(ValueError, match="fresh output"):
        continuation.plan(evidence.prior, evidence.output, evidence.config)


def test_wrong_gpu_is_not_migrated(evidence):
    with pytest.raises(ValueError, match="exclusive completed"):
        continuation.prerequisites(evidence.prior, evidence.config, 2, "GPU-other")


def test_host_nice_guard_precedes_all_output(evidence, monkeypatch):
    monkeypatch.setattr(continuation.socket, "gethostname", lambda: "wrong-host")
    with pytest.raises(ValueError, match="pinned 225"):
        continuation.run(evidence.prior, evidence.output, evidence.config)
    assert not evidence.output.exists()


def test_live_prior_process_blocks_start(evidence, monkeypatch):
    monkeypatch.setattr(continuation.socket, "gethostname", lambda: continuation.base.HOST)
    monkeypatch.setattr(continuation.os, "getpriority", lambda *_: 10)
    path = evidence.prior / "job.json"
    row = json.loads(path.read_text())
    row["parent_pid"] = 1
    write(path, row)
    if not Path("/proc/1").exists():
        pytest.skip("Linux procfs test")
    with pytest.raises(ValueError, match="PID is still present"):
        continuation.run(evidence.prior, evidence.output, evidence.config)
    assert not evidence.output.exists()


def test_projected_budget_refusal_precedes_compute(evidence, monkeypatch):
    monkeypatch.setattr(continuation, "GPU_SECONDS", 100)
    # Compare config uses cap100 too, then the cumulative gate rejects it.
    config = json.loads(evidence.config.read_text())
    config["training"]["max_training_seconds"] = 100
    write(evidence.config, config)
    with pytest.raises(ValueError, match="newly authorized budget"):
        prerequisites(evidence)


def test_mock_failure_retains_outputs_and_never_retries(evidence, monkeypatch):
    monkeypatch.setattr(continuation.socket, "gethostname", lambda: continuation.base.HOST)
    monkeypatch.setattr(continuation.os, "getpriority", lambda *_: 10)
    monkeypatch.setattr(continuation.base, "host_resources", lambda: dict(cpus=24, load1=1,
        mem_available_bytes=100*1024**3, io_full_avg10=0))
    monkeypatch.setattr(continuation.time, "sleep", lambda _: None)
    from scripts import report_phi4_hopping_qcr
    monkeypatch.setattr(report_phi4_hopping_qcr, "select_chinese_font", lambda _: Path("/mock/font"))
    calls = []
    class Child:
        pid = 99999999
        returncode = 7
        def poll(self):
            return self.returncode
    def popen(command, **kwargs):
        calls.append(command)
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        return Child()
    monkeypatch.setattr(continuation.subprocess, "Popen", popen)
    old_before = {str(p): p.read_bytes() for p in evidence.prior.iterdir()}
    assert continuation.run(evidence.prior, evidence.output, evidence.config, prior_queue_root=evidence.queue) == 7
    assert len(calls) == 1 and "scripts.prepare_phi4_hopping" in calls[0]
    job = json.loads((evidence.output / "job.json").read_text())
    assert job["status"] == "failed" and job["returncode"] == 7
    assert job["prior_cpu_seconds"] == pytest.approx(26.0818944)
    assert job["gpu_seconds"] == pytest.approx(50.38484049588442)
    assert job["additional_benchmark_updates"] == 0
    assert continuation.base.sha(evidence.output / "reused_preflight.json") == continuation.PREFLIGHT_SHA256
    assert json.loads((evidence.output / "reused_preflight.json").read_text())["returncode"] == 3
    assert all(Path(p).read_bytes() == value for p,value in old_before.items())
    assert not (evidence.output / "training").exists()


def test_mock_success_two_idle_checks_and_cumulative_accounting(evidence, monkeypatch):
    monkeypatch.setattr(continuation.socket, "gethostname", lambda: continuation.base.HOST)
    monkeypatch.setattr(continuation.os, "getpriority", lambda *_: 10)
    monkeypatch.setattr(continuation.base, "host_resources", lambda: dict(cpus=24, load1=1,
        mem_available_bytes=100*1024**3, io_full_avg10=0))
    gpu_reads = []
    def gpu(index, uuid):
        gpu_reads.append((index, uuid))
        return dict(index=index, uuid=uuid, free_bytes=24*1024**3, compute_pids=[])
    monkeypatch.setattr(continuation.base, "gpu_resources", gpu)
    clock = [0.]
    monkeypatch.setattr(continuation.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(continuation.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0]+seconds))
    from scripts import report_phi4_hopping_qcr
    monkeypatch.setattr(report_phi4_hopping_qcr, "select_chinese_font", lambda _: Path("/mock/font"))
    monkeypatch.setattr(continuation, "_validate_sampling", lambda root: dict(
        data_manifest_sha256="a"*64, data_warning_count=3))
    trained, public = [], []
    monkeypatch.setattr(continuation.base, "validate_training", lambda *args: trained.append(args))
    monkeypatch.setattr(continuation, "_public_files", lambda root: public.append(root))
    calls = []
    class Child:
        pid = 99999999
        returncode = 0
        def poll(self):
            return self.returncode
    def popen(command, **kwargs):
        calls.append((command, kwargs["env"]["CUDA_VISIBLE_DEVICES"]))
        return Child()
    monkeypatch.setattr(continuation.subprocess, "Popen", popen)
    assert continuation.run(evidence.prior, evidence.output, evidence.config, prior_queue_root=evidence.queue) == 0
    job = json.loads((evidence.output / "job.json").read_text())
    assert job["status"] == "complete" and job["workflow_complete"] is True and job["returncode"] == 0
    assert [row["name"] for row in job["stages"]] == ["sampling", "training"]
    assert all(row["returncode"] == 0 for row in job["stages"])
    assert len(calls) == 2 and calls[0][1] == "" and calls[1][1] == continuation.GPU_UUID
    assert len(gpu_reads) == 2 and job["gpu_idle_checks"]["spacing_seconds"] == 10
    assert job["gpu_seconds"] == pytest.approx(60.38484049588442)
    assert job["data_warning_count"] == 3 and trained == [(evidence.output,"a"*64)]
    assert public == [evidence.output] and job["additional_benchmark_updates"] == 0


@pytest.mark.parametrize("acceptances", [[0.]*52, [1.]*51, [float("nan")]+[1.]*51])
def test_zero_or_invalid_acceptance_does_not_admit_training(tmp_path, monkeypatch, acceptances):
    from recurrent_rg.data import phi4_hopping_loader
    monkeypatch.setattr(phi4_hopping_loader, "load_dataset", lambda _: SimpleNamespace(
        manifest_sha256="a"*64, warning_count=2))
    (tmp_path / "data").mkdir()
    write(tmp_path / "data/quality_summary.json", dict(conditions=[dict(production_acceptance=acceptances)]))
    with pytest.raises(RuntimeError, match="production acceptance"):
        continuation._validate_sampling(tmp_path)


def test_nonzero_acceptance_retains_quality_warnings(tmp_path, monkeypatch):
    from recurrent_rg.data import phi4_hopping_loader
    monkeypatch.setattr(phi4_hopping_loader, "load_dataset", lambda _: SimpleNamespace(
        manifest_sha256="a"*64, warning_count=8))
    (tmp_path / "data").mkdir()
    write(tmp_path / "data/quality_summary.json", dict(conditions=[dict(production_acceptance=[.1]*52)]))
    assert continuation._validate_sampling(tmp_path) == dict(data_manifest_sha256="a"*64, data_warning_count=8)
