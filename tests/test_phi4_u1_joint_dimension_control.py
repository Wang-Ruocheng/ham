import json

import pytest

from scripts import run_phi4_u1_joint_dimension_control as joint


def make_prior(root, q_seed, ratio=.6):
    (root / "training").mkdir(parents=True, exist_ok=True)
    (root / "report").mkdir(exist_ok=True)
    summary = dict(status="complete", workflow_complete=True, returncode=0, feature_metrics={})
    for r_seed in (91, 92, 93):
        for dim in (1, 2):
            rmse = 1. if dim == 1 else ratio**.5
            summary["feature_metrics"][f"dg{dim}_seed{r_seed}_selected"] = dict(validation=[
                dict(path=path, prediction=dict(chain_mean_rmse=rmse))
                for path in ("0_to_1", "1_to_2", "0_to_2")])
    files = {
        "job.json": dict(status="complete", workflow_complete=True, returncode=0, error=None,
                         requires_user_action=False, stages=[dict(returncode=0)]),
        "plan.json": dict(q_seed=q_seed), "training/summary.json": summary,
        "report/analysis.json": dict(status="complete"),
    }
    for name, value in files.items():
        (root / name).write_text(json.dumps(value))
    (root / "public_files.json").write_text(json.dumps({name: joint.sha(root / name) for name in files}))


def resources():
    return dict(mem_available_bytes=40*1024**3, load1=20., cpus=24, io_full_avg10=1.)


def GPUs():
    return {assignment["index"]: dict(index=assignment["index"], uuid=assignment["uuid"],
            free_bytes=20*1024**3, compute_pids=[]) for assignment in joint.GPU_ASSIGNMENTS.values()}


def test_three_completed_Q_seed_gates_use_paired_feature_MSE(tmp_path):
    roots = [tmp_path / f"seed{seed}" for seed in (91, 92, 93)]
    for seed, root, ratio in zip((91, 92, 93), roots, (.61, .608, .651)):
        make_prior(root, seed, ratio)
    evidence, pins = joint.prerequisite_gate(roots)
    assert len(pins) == 15
    assert all(row["passed"] and row["improved_pairs"] == 3 for row in evidence.values())
    assert evidence["91"]["median_ratio"] == pytest.approx(.61)
    assert evidence["93"]["median_ratio"] == pytest.approx(.651)
    make_prior(roots[1], 92, .9)
    with pytest.raises(ValueError, match="gate"):
        joint.prerequisite_gate(roots)


def test_prerequisite_tamper_and_incomplete_stage_rejected(tmp_path):
    roots = [tmp_path / f"seed{seed}" for seed in (91, 92, 93)]
    for seed, root in zip((91, 92, 93), roots):
        make_prior(root, seed)
    (roots[0] / "report/analysis.json").write_text("{}")
    with pytest.raises(ValueError, match="SHA"):
        joint.prerequisite_gate(roots)
    make_prior(roots[0], 91)
    job = json.loads((roots[1] / "job.json").read_text())
    job["stages"][0]["returncode"] = 1
    (roots[1] / "job.json").write_text(json.dumps(job))
    inventory = json.loads((roots[1] / "public_files.json").read_text())
    inventory["job.json"] = joint.sha(roots[1] / "job.json")
    (roots[1] / "public_files.json").write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match="complete"):
        joint.prerequisite_gate(roots)


def test_idle_gpu_gate_checks_both_UUIDs_free_memory_and_no_other_process():
    rows = GPUs()
    assert joint.gpu_ready(rows)
    rows[2]["compute_pids"] = [123]
    assert not joint.gpu_ready(rows)
    rows = GPUs()
    rows[3]["free_bytes"] = 12*1024**3-1
    assert not joint.gpu_ready(rows)
    rows = GPUs()
    rows[3]["uuid"] = "GPU-wrong"
    with pytest.raises(RuntimeError, match="UUID"):
        joint.gpu_ready(rows)


def test_resource_checks_and_at_least_ten_second_spacing():
    first = dict(host=resources(), gpus=GPUs())
    second = dict(host=resources(), gpus=GPUs())
    assert joint.idle_pair_receipt(first, second, 10.)
    assert not joint.idle_pair_receipt(first, second, 9.999)
    for name, value in (("mem_available_bytes", 35*1024**3), ("load1", 30.1), ("io_full_avg10", 20.)):
        row = resources()
        row[name] = value
        assert not joint.host_ready(row)
    assert joint.host_ready(dict(resources(), mem_available_bytes=36*1024**3, load1=30., io_full_avg10=19.999))


def test_existing_child_commands_pin_recipe_limits_and_physical_devices(tmp_path):
    for dim in (1, 2):
        command = joint.training_command(tmp_path, dim)
        assert command[command.index("-m")+1] == ("experiments.phi4_qcr" if dim == 1 else "experiments.phi4_ru_qcr")
        expected = {"--data": str(joint.SOURCE / "data"), "--output": str(tmp_path / f"dg{dim}"),
            "--steps": "3000", "--seed": "91", "--system-batch": "8", "--sample-batch": "64",
            "--threads": "1", "--information-pairs": "128", "--device": "cuda:0",
            "--max-training-seconds": "1200.0", "--max-cuda-bytes": str(8*1024**3),
            "--q-initialization": "train_mean_bias", "--expected-gpu-uuid": joint.GPU_ASSIGNMENTS[dim]["uuid"]}
        assert all(command[command.index(flag)+1] == value for flag, value in expected.items())
        assert "--profile-only" not in command and "nice" not in command
        env = joint.child_environment(dim)
        assert env["CUDA_VISIBLE_DEVICES"] == joint.GPU_ASSIGNMENTS[dim]["uuid"]
        assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        assert all(env[key] == "1" for key in joint.THREAD_ENV)
    with pytest.raises(ValueError):
        joint.training_command(tmp_path, 3)


def test_public_child_whitelist_excludes_all_checkpoints_and_raw_fields():
    assert set(joint.PUBLIC_CHILDREN) == {"manifest.json", "summary.json", "status.json", "history_progress.json", "evaluation_arrays.npz"}
    assert not any(name.endswith(".pt") or "checkpoint" in name or "configurations" in name for name in joint.PUBLIC_CHILDREN)


def test_snapshot_parser_maps_physical_indices_and_compute_UUID(monkeypatch):
    calls = []
    def output(command, **kwargs):
        calls.append(command)
        if "--query-gpu=index,uuid,memory.free" in command:
            return "2, GPU-713b07d0-118c-f939-c0a8-565390215e01, 20000\n3, GPU-a167decd-ce49-6c8f-6bcf-136fde596683, 21000\n"
        return "GPU-713b07d0-118c-f939-c0a8-565390215e01, 100\n"
    monkeypatch.setattr(joint.subprocess, "check_output", output)
    rows = joint.gpu_resources()
    assert rows[2]["compute_pids"] == [100] and rows[3]["compute_pids"] == []
    assert rows[3]["free_bytes"] == 21000*1024**2
    assert len(calls) == 2


def test_status_poll_partial_empty_missing_then_valid_is_unknown_not_failure(tmp_path):
    path = tmp_path / "status.json"
    stage = dict(name="dg1", status="running", returncode=None,
                 child_status=dict(status="training", actual_steps=100))
    for text in (None, "", '{"status":"training", "actual_steps":'):
        if text is not None:
            path.write_text(text)
        assert not joint.poll_child_status(stage, path)
        assert stage["child_status_read_pending"]
        assert stage["child_status"] == dict(status="training", actual_steps=100)
        assert stage["status"] == "running" and stage["returncode"] is None
    path.write_text(json.dumps(dict(status="training", actual_steps=200)))
    assert joint.poll_child_status(stage, path)
    assert stage["child_status"] == dict(status="training", actual_steps=200)
    assert stage["child_status_read_pending"] is False
    assert stage["status"] == "running" and stage["returncode"] is None
