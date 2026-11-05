"""CPU-only parent policies; no torch, source checkpoint or remote host needed."""
import copy
import json
from pathlib import Path

import pytest

from scripts import run_phi4_scalar_closure_ablation as runner


def input_fixture():
    pins = {str(root / name): f"{root.name}-{name}"
            for root in (runner.SOURCE, runner.OLD_SOURCE)
            for name in ("manifest.json", "summary.json", "status.json",
                         "evaluation_arrays.npz", "training_step_3000.pt")}
    pins.update({str(runner.SOURCE / "training_step_3000.pt"): runner.SOURCE_CHECKPOINT_SHA,
                 str(runner.OLD_SOURCE / "training_step_3000.pt"): runner.OLD_CHECKPOINT_SHA,
                 str(runner.CACHE / "arrays.npz"): runner.CACHE_ARRAYS_SHA,
                 str(runner.CACHE / "receipt.json"): "receipt-sha"})
    values = {
        str(runner.SOURCE.parent / "job.json"): dict(status="complete", workflow_complete=True,
            returncode=0, stages=[dict(returncode=0)]),
        str(runner.SOURCE / "manifest.json"): dict(theory_dim=1, lattice_sizes=[36, 12, 4],
            data_manifest_sha256=runner.DATA_SHA, scale=.82),
        str(runner.SOURCE / "summary.json"): dict(status="complete", actual_steps=3000,
            final_state_sha256=runner.SOURCE_STATE_SHA, final_module_sha256={"C": runner.FINAL_C_SHA}),
        str(runner.SOURCE / "status.json"): dict(status="complete", workflow_complete=True,
            returncode=0, summary_sha256=pins[str(runner.SOURCE / "summary.json")],
            evaluation_arrays_sha256=pins[str(runner.SOURCE / "evaluation_arrays.npz")]),
        str(runner.CACHE / "status.json"): dict(status="complete", workflow_complete=True,
            returncode=0, arrays_sha256=runner.CACHE_ARRAYS_SHA, receipt_sha256="receipt-sha"),
        str(runner.CACHE / "receipt.json"): dict(status="complete", source_run=str(runner.OLD_SOURCE),
            source_checkpoint=str(runner.OLD_SOURCE / "training_step_3000.pt"),
            source_checkpoint_sha256=runner.OLD_CHECKPOINT_SHA,
            source_manifest_sha256=pins[str(runner.OLD_SOURCE / "manifest.json")],
            source_state_sha256="old-state", source_state_sha256_after="old-state",
            source_data_manifest_sha256=runner.DATA_SHA, arrays_sha256=runner.CACHE_ARRAYS_SHA, scale=.82)}
    return pins, values


def test_commands_are_paired_cpu_only_without_collection(tmp_path):
    commands = runner.commands(tmp_path)
    assert [row[0] for row in commands] == ["paired_training", "report"]
    assert not any(row[2] for row in commands)
    flat = " ".join(token for _, command, _ in commands for token in command)
    assert "--steps 3000" in flat and "--cpu-budget-seconds 800" in flat
    assert str(runner.SOURCE) in flat and str(runner.CACHE) in flat
    for forbidden in ("cuda", "nvidia", "collect_phi4", "prepare_phi4", "--resume", "allow-legacy",
                      "allow-nondefault", "experiments.phi4_qcr "):
        assert forbidden not in flat


def test_cpu_environment_hides_all_gpus_and_sets_single_thread(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setenv("OMP_NUM_THREADS", "24")
    env = runner.cpu_environment()
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        assert env[name] == "1"
    assert runner.os.environ["CUDA_VISIBLE_DEVICES"] == "2"


def test_host_accepts_empty_cuda_mask_but_requires_pinned_host(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(runner.socket, "gethostname", lambda: runner.HOST)
    runner.validate_host()
    monkeypatch.setattr(runner.socket, "gethostname", lambda: "different-host")
    with pytest.raises(RuntimeError, match="pinned"):
        runner.validate_host()


def test_validate_inputs_pins_latest_and_historical_cache_sources(monkeypatch):
    pins, values = input_fixture()
    monkeypatch.setattr(runner, "read", lambda path: values[str(path)])
    result = runner.validate_inputs(pins)
    assert result["latest_source_checkpoint_sha256"] == runner.SOURCE_CHECKPOINT_SHA
    assert result["historical_cache_source_checkpoint_sha256"] == runner.OLD_CHECKPOINT_SHA
    assert result["cache_C_identity_requires_tensor_check"] is True


@pytest.mark.parametrize("root,expected", [("SOURCE", "Latest source checkpoint"),
                                          ("OLD_SOURCE", "historical source endpoint")])
def test_validate_inputs_rejects_any_endpoint_change(monkeypatch, root, expected):
    pins, values = input_fixture()
    pins[str(getattr(runner, root) / "training_step_3000.pt")] = "wrong"
    monkeypatch.setattr(runner, "read", lambda path: values[str(path)])
    with pytest.raises(RuntimeError, match=expected):
        runner.validate_inputs(pins)


@pytest.mark.parametrize("field,value", [("source_run", "/unauthorized/source"),
                                        ("source_checkpoint", "/unauthorized/model.pt"),
                                        ("arrays_sha256", "wrong"),
                                        ("source_state_sha256_after", "modified"),
                                        ("scale", .99)])
def test_validate_inputs_rejects_receipt_drift(monkeypatch, field, value):
    pins, values = input_fixture()
    values[str(runner.CACHE / "receipt.json")][field] = value
    monkeypatch.setattr(runner, "read", lambda path: values[str(path)])
    with pytest.raises(RuntimeError, match="cache receipt"):
        runner.validate_inputs(pins)


def test_prepare_refuses_existing_output_before_source_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "validate_host", lambda: None)
    monkeypatch.setattr(runner, "immutable_inputs", lambda: pytest.fail("Existing output must be rejected first"))
    with pytest.raises(FileExistsError, match="overwrite"):
        runner.prepare(tmp_path)


def test_prepare_records_source_steps_separately_and_same_guard(tmp_path, monkeypatch):
    target = tmp_path / "new"
    pins, values = input_fixture()
    monkeypatch.setattr(runner, "validate_host", lambda: None)
    monkeypatch.setattr(runner, "immutable_inputs", lambda: pins)
    monkeypatch.setattr(runner, "read", lambda path: values[str(path)])
    monkeypatch.setattr(runner, "source_hashes", lambda path: {"pinned.py": "sha"})
    plan = runner.prepare(target)
    assert set(plan["arms"]) == {"A", "B"}
    assert plan["training"]["additional_steps_per_arm"] == 3000
    assert plan["training"]["source_actual_steps"] == 3000
    assert plan["training"]["C_updates"] == 0
    assert plan["training"]["guard_floor_ratio"] == .5
    assert plan["training"]["affine_decoder_unchanged"] is True
    assert plan["resource_policy"]["cpu_seconds"] == 900
    assert plan["resource_policy"]["GPU_queries"] == 0
    assert plan["immutable_input_sha256"] == pins
    job = json.loads((target / "job.json").read_text())
    assert job["status"] == "prepared" and job["workflow_complete"] is False


def test_prepare_bad_pin_does_not_create_output(tmp_path, monkeypatch):
    target = tmp_path / "new"
    pins, _ = input_fixture()
    pins[str(runner.SOURCE / "training_step_3000.pt")] = "bad"
    monkeypatch.setattr(runner, "validate_host", lambda: None)
    monkeypatch.setattr(runner, "immutable_inputs", lambda: pins)
    with pytest.raises(RuntimeError, match="checkpoint SHA"):
        runner.prepare(target)
    assert not target.exists()


def test_immutable_inventory_rejects_code_and_input_mutation(monkeypatch):
    plan = dict(code_root="code", source_sha256={"a.py": "same"}, immutable_input_sha256={"input": "same"})
    monkeypatch.setattr(runner, "source_hashes", lambda path: {"a.py": "different"})
    with pytest.raises(RuntimeError, match="source inventory"):
        runner.unchanged(plan)
    monkeypatch.setattr(runner, "source_hashes", lambda path: plan["source_sha256"])
    monkeypatch.setattr(runner, "immutable_inputs", lambda: {"input": "different"})
    with pytest.raises(RuntimeError, match="fingerprints"):
        runner.unchanged(plan)


def test_host_snapshot_does_not_query_any_gpu(monkeypatch):
    def proc_read(path, *args, **kwargs):
        return {"/proc/meminfo": "MemAvailable: 67108864 kB\n",
                "/proc/pressure/io": "some avg10=0.0\nfull avg10=0.25 avg60=0.0\n"}[str(path)]
    monkeypatch.setattr(Path, "read_text", proc_read)
    monkeypatch.setattr(runner.os, "cpu_count", lambda: 24)
    monkeypatch.setattr(runner.os, "getloadavg", lambda: (10., 10., 10.))
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: pytest.fail("No GPU query is authorized"))
    row = runner.snapshot()
    assert row["host_ready"] and row["mem_available_bytes"] == 64 * 1024**3


def test_execute_refuses_relaunch_before_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "validate_host", lambda: None)
    monkeypatch.setattr(runner, "read", lambda path: {"status": "running", "stages": []})
    with pytest.raises(RuntimeError, match="restart"):
        runner.execute(tmp_path)
    assert not (tmp_path / "execution.lock").exists()


def test_public_files_never_include_checkpoint_or_symlink(tmp_path):
    folder = tmp_path / "training"
    folder.mkdir()
    for name in ("status.json", "plot.png", "arrays.npz", "training_step_3000.pt", "raw.npy"):
        (folder / name).write_bytes(b"test")
    (folder / "linked.json").symlink_to(folder / "status.json")
    assert set(runner.public_files(tmp_path)) == {"training/status.json", "training/plot.png", "training/arrays.npz"}


def completed_training():
    modules = dict(Q="q", R="r", C=runner.FINAL_C_SHA)
    arm = dict(actual_steps=3000, source_actual_steps=3000, C_updates=0,
        initial_module_sha256=modules, final_module_sha256=modules,
        optimizer_steps={name: dict(min=3000, max=3000, parameters=4) for name in ("Q", "R")})
    return dict(status="complete", actual_steps=3000, actual_steps_by_arm={"A": 3000, "B": 3000},
        frozen_source_verified=True, source_state_sha256_before=runner.SOURCE_STATE_SHA,
        source_state_sha256_after=runner.SOURCE_STATE_SHA, source_gradients_none=True, C_updates=0,
        physical_sampling_calls=0, MI_calls=0, GPU_used=False, labels_used_in_training=False,
        default_recipe_modified=False, thermal_nu=None, critical_r=None, equilibrium_certified=False,
        arms={"source": dict(final_module_sha256=modules), "A": copy.deepcopy(arm), "B": copy.deepcopy(arm)})


def test_validate_training_requires_full_adam_counts_and_fixed_C(tmp_path, monkeypatch):
    summary = completed_training()
    status = dict(status="complete", workflow_complete=True, returncode=0,
                  summary_sha256="sha", evaluation_arrays_sha256="sha")
    monkeypatch.setattr(runner, "read", lambda path: summary if path.name == "summary.json" else status)
    monkeypatch.setattr(runner, "sha", lambda path: "sha")
    assert runner.validate_training(tmp_path) is summary
    summary["arms"]["B"]["optimizer_steps"]["Q"]["min"] = 2999
    with pytest.raises(RuntimeError, match="Arm B"):
        runner.validate_training(tmp_path)


def test_validate_training_rejects_unfrozen_source(tmp_path, monkeypatch):
    summary = completed_training()
    summary["source_state_sha256_after"] = "changed"
    status = dict(status="complete", workflow_complete=True, returncode=0)
    monkeypatch.setattr(runner, "read", lambda path: summary if path.name == "summary.json" else status)
    with pytest.raises(RuntimeError, match="immutability"):
        runner.validate_training(tmp_path)


@pytest.mark.parametrize("status_name,returncode", [("failed", 17), ("budget_hold", 3)])
def test_execute_preserves_nonzero_stage_without_retry(tmp_path, monkeypatch, status_name, returncode):
    (tmp_path / "logs").mkdir()
    plan = dict(output=str(tmp_path), code_root="/frozen/code", recipe=runner.RECIPE,
        resource_policy=dict(cpu_seconds=900),
        commands=[dict(stage=name, command=cmd, gpu=gpu) for name, cmd, gpu in runner.commands(tmp_path)])
    runner.save(tmp_path / "plan.json", plan)
    runner.save(tmp_path / "job.json", dict(status="prepared", workflow_complete=False, stages=[]))
    monkeypatch.setattr(runner, "validate_host", lambda: None)
    monkeypatch.setattr(runner, "unchanged", lambda _: None)
    monkeypatch.setattr(runner, "admission", lambda *args: None)
    monkeypatch.setattr(runner.os, "getpriority", lambda *args: 10)
    calls = []

    class Child:
        pid = 12345

        def __init__(self, command, **kwargs):
            calls.append((command, kwargs))
            output = tmp_path / "training"
            output.mkdir()
            runner.save(output / "status.json", dict(status=status_name, returncode=returncode,
                workflow_complete=False, error="intentional gate", requires_user_action=True))

        def wait(self):
            return returncode

    monkeypatch.setattr(runner.subprocess, "Popen", Child)
    if status_name == "failed":
        with pytest.raises(RuntimeError, match="nonzero returncode=17"):
            runner.execute(tmp_path)
    else:
        assert runner.execute(tmp_path) == 3
    job = runner.read(tmp_path / "job.json")
    assert job["status"] == status_name and job["workflow_complete"] is False
    assert job["requires_user_action"] is True
    assert len(job["stages"]) == 1 and job["stages"][0]["returncode"] == returncode
    assert len(calls) == 1 and calls[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert (tmp_path / "execution.lock").exists()
    with pytest.raises(RuntimeError, match="restart"):
        runner.execute(tmp_path)
