"""Only synthetic receipts/process mocks; no SSH, models, physical data or GPU."""
from copy import deepcopy
from pathlib import Path

import pytest

from scripts import run_ising_baseline_comparison as runner


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "BASE", tmp_path)
    reference = tmp_path / "results/reference"
    cache = tmp_path / "data/cache.pt"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"synthetic physical payload, never read or hashed")
    checkpoint = reference / "seed91/run/training_step_12000.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"synthetic checkpoint, never deserialized")
    monkeypatch.setattr(runner, "DATA", cache)
    monkeypatch.setattr(runner, "REFERENCE", reference)
    monkeypatch.setattr(runner, "CHECKPOINT_SHA", runner.sha(checkpoint))
    config = dict(schema_version=5, data=dict(cache=str(cache), physical_model="ising2d"),
        model=dict(block_factor=3, theory_dim=5), training=dict(seed=91, epochs=12000))
    values = {
        reference / "configs/seed91.json": config,
        cache.with_suffix(".json"): dict(data_sha256=runner.DATA_SHA, B=0., coupling_names=["K"]),
        reference / "seed91/report/report.json": dict(status="complete", checkpoint_epoch=12000,
            checkpoint_sha256_before=runner.CHECKPOINT_SHA, checkpoint_sha256_after=runner.CHECKPOINT_SHA,
            checkpoint_unchanged=True),
        reference / "seed91/report/metrics.json": dict(data_sha256=runner.DATA_SHA,
            checkpoint_epoch=12000, labels_used_in_training=False),
    }
    for path, value in values.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        runner.save(path, value, exclusive=True)
    monkeypatch.setattr(runner, "SIDECAR_SHA", runner.sha(cache.with_suffix(".json")))
    monkeypatch.setattr(runner.socket, "gethostname", lambda: runner.HOST)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(runner, "source_hashes", lambda code: {"synthetic.py": "frozen"})
    original_sha = runner.sha

    def no_physical_hash(path):
        if Path(path) == cache:
            raise AssertionError("Do not open/re-hash the existing physical PT")
        return original_sha(path)

    monkeypatch.setattr(runner, "sha", no_physical_hash)
    root = tmp_path / "results/new_baselines"
    plan = runner.prepare(root)
    return root, plan


def _profile():
    return dict(status="complete", projected_gpu_seconds=1000., projected_cpu_seconds=2120.,
        peak_gpu_bytes=2 * 1024**3,
        projected_gpu_seconds_by_method={method: 200. for method in runner.METHODS},
        projected_cpu_seconds_by_method={method: 400. for method in runner.METHODS},
        comparison_cpu_seconds=120., public_files=[])


def test_default_policy_covers_all_time_and_resource_safety():
    policy = runner.policy()
    assert policy["gpu_seconds"] == 7200
    assert policy["cpu_seconds"] == 14400
    assert policy["gpu_peak_bytes"] == 8 * 1024**3
    assert policy["directory_bytes"] == 4 * 1024**3
    assert policy["cpu_threads"] == 2 and policy["nice"] == 10
    assert policy["two_gpu_checks_seconds_apart"] >= 10
    assert policy["minimum_host_available_gib"] == 36
    assert policy["gpu_free_mib"] == 12288
    assert not policy["automatic_retry"] and not policy["resume"]


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf")])
def test_invalid_budget_rejected(value):
    with pytest.raises(ValueError):
        runner.policy(gpu_hours=value)


def test_prepare_pins_read_only_sources_and_uses_new_root(prepared):
    root, plan = prepared
    assert plan["methods"] == list(runner.METHODS)
    assert plan["source_data_stat"] == runner.stat_identity(plan["source_data"])
    assert plan["seed"] == 91 and plan["updates"] == 12000 and plan["C_draws"] == 3
    assert plan["device"] == "cuda:0" and plan["gpu_uuid"] == runner.GPU_UUID
    assert plan["profile"]["updates"] == 40
    assert plan["profile"]["collection_physical_chains"] == 4
    assert plan["source_code_root"] == plan["code_root"]
    assert plan["protocol"]["methods_are_same_data_adaptations_not_native_paper_reproductions"]
    assert plan["protocol"]["labels_used_in_training"] is False
    assert runner.read(root / "job.json")["status"] == "prepared"
    with pytest.raises(FileExistsError):
        runner.prepare(root)
    assert runner.verify(root)[0] == plan


def test_original_source_change_fails_closed(prepared):
    root, plan = prepared
    Path(plan["source_report"]).write_text('{"status":"modified"}')
    with pytest.raises(RuntimeError, match="identity"):
        runner.verify(root)


def test_frozen_plan_change_fails_closed(prepared):
    root, plan = prepared
    plan["updates"] = 6000
    runner.save(root / "plan.json", plan)
    with pytest.raises(RuntimeError, match="Frozen"):
        runner.verify(root)


def test_wrong_host_or_cuda_mask_forbidden(prepared, monkeypatch):
    root, _ = prepared
    monkeypatch.setattr(runner.socket, "gethostname", lambda: "different-host")
    with pytest.raises(ValueError, match="physical host"):
        runner.verify(root)
    monkeypatch.setattr(runner.socket, "gethostname", lambda: runner.HOST)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="unmasked"):
        runner.verify(root)


def test_commands_are_fixed_stages_without_recipe_bypass_or_resume(tmp_path):
    for stage, method in [("profile", None), *(("arm", m) for m in runner.METHODS), ("compare", None)]:
        command = runner.stage_command(tmp_path, stage, method)
        assert command[:4] == [runner.sys.executable, "-u", "-m", runner.CHILD_MODULE]
        assert command[command.index("--plan") + 1] == str(tmp_path / "plan.json")
        assert command[command.index("--root") + 1] == str(tmp_path)
        assert not any("resume" in arg or "allow-" in arg for arg in command)
    for stage, method in [("train", None), ("arm", None), ("compare", "nmcrg"), ("profile", "unknown")]:
        with pytest.raises(ValueError):
            runner.stage_command(tmp_path, stage, method)


def test_projection_keeps_actual_usage_and_only_remaining_arm_forecasts():
    plan = dict(resource_policy=runner.policy())
    job = dict(gpu_elapsed_seconds=430., cpu_seconds=700.)
    result = runner.budget_projection(plan, job, _profile(), ["majority", "qcr_c_pca"])
    assert result["projected_total_gpu_seconds"] == 830
    assert result["projected_total_cpu_seconds"] == 1620
    assert result["budget_passed"] and not result["profile_is_scientific_result"]
    comparison = runner.budget_projection(plan, job, _profile(), [])
    assert comparison["projected_total_gpu_seconds"] == 430
    assert comparison["projected_total_cpu_seconds"] == 820


def test_missing_method_timings_never_assumes_equal_arm_cost():
    profile = _profile()
    del profile["projected_gpu_seconds_by_method"], profile["projected_cpu_seconds_by_method"]
    result = runner.budget_projection(dict(resource_policy=runner.policy()),
        dict(gpu_elapsed_seconds=430., cpu_seconds=700.), profile, ["majority"])
    assert result["projected_total_gpu_seconds"] == 1430
    assert result["projected_total_cpu_seconds"] == 2820


@pytest.mark.parametrize("field,value", [("projected_gpu_seconds", float("nan")),
    ("projected_cpu_seconds", -1), ("peak_gpu_bytes", True)])
def test_profile_refuses_nonfinite_or_invalid_measurements(field, value):
    profile = _profile()
    profile[field] = value
    with pytest.raises(ValueError):
        runner.validate_profile(profile)


def test_peak_or_total_projection_triggers_hold_not_reduced_updates():
    profile = _profile()
    profile["peak_gpu_bytes"] = 9 * 1024**3
    assert not runner.budget_projection(dict(resource_policy=runner.policy()), {}, profile, runner.METHODS)["budget_passed"]
    assert runner.UPDATES == 12000


def test_admission_requires_two_successes_ten_seconds_apart(prepared, monkeypatch):
    root, plan = prepared
    checks = iter([False, True, True])
    snapshots = []
    sleeps = []

    def snapshot(_):
        ready = next(checks)
        snapshots.append(ready)
        return dict(gpu_uuid=runner.GPU_UUID, gpu_ready=ready, host_ready=True)

    monkeypatch.setattr(runner, "resource_snapshot", snapshot)
    monkeypatch.setattr(runner.time, "sleep", sleeps.append)
    runner.admission(root, plan, runner.read(root / "job.json"), "nmcrg", gpu=True)
    assert snapshots == [False, True, True]
    assert sleeps == [30, 10]


def test_admission_refuses_uuid_change_without_any_migration(prepared, monkeypatch):
    root, plan = prepared
    monkeypatch.setattr(runner, "resource_snapshot", lambda _: dict(gpu_uuid="other"))
    with pytest.raises(RuntimeError, match="UUID"):
        runner.admission(root, plan, runner.read(root / "job.json"), "nmcrg", gpu=True)


@pytest.mark.parametrize("stage,method,gpu_charge", [("arm", "nmcrg", 2.), ("compare", None, 0.)])
def test_child_process_accounting_charges_entire_gpu_arm_but_not_cpu_comparison(
        prepared, monkeypatch, stage, method, gpu_charge):
    root, plan = prepared
    job = runner.read(root / "job.json")
    seen = {}
    clock = iter((10., 11., 12.))
    monkeypatch.setattr(runner.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(runner, "cpu_seconds", lambda: 100.)
    monkeypatch.setattr(runner, "live_child_cpu_seconds", lambda pid: 1.5)
    monkeypatch.setattr(runner, "admission", lambda *args, **kwargs: None)
    folder = runner.stage_folder(root, stage, method)
    folder.mkdir(parents=True)
    runner.save(folder / "status.json", dict(status="complete", peak_gpu_bytes=42.))

    class Process:
        pid = 901

        def __init__(self, command, **kwargs):
            seen.update(command=command, **kwargs)
            self.polls = 0

        def poll(self):
            self.polls += 1
            return None if self.polls == 1 else 0

        def wait(self):
            return 0

    monkeypatch.setattr(runner.subprocess, "Popen", Process)
    status = runner._run_stage(root, plan, job, stage, method, 100.)
    assert status["status"] == "complete"
    assert job["gpu_elapsed_seconds"] == gpu_charge
    assert job["child_pid"] is None
    assert job["stages"][0]["wall_seconds"] == 2.
    assert job["stages"][0]["live_child_cpu_seconds"] == 1.5
    assert job["stages"][0]["returncode"] == 0
    assert seen["cwd"] == plan["code_root"]
    assert seen["env"]["OMP_NUM_THREADS"] == ("2" if stage == "arm" else "1")
    assert seen["env"]["OPENBLAS_NUM_THREADS"] == "1"
    assert "CUDA_VISIBLE_DEVICES" not in seen["env"]


def test_child_nonzero_with_budget_receipt_is_hold_not_execution_failure(prepared, monkeypatch):
    root, plan = prepared
    job = runner.read(root / "job.json")
    monkeypatch.setattr(runner, "admission", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "cpu_seconds", lambda: 100.)
    clock = iter((10., 14.))
    monkeypatch.setattr(runner.time, "perf_counter", lambda: next(clock))
    folder = runner.stage_folder(root, "arm", "nmcrg")
    folder.mkdir(parents=True)
    runner.save(folder / "status.json", dict(status="budget_hold", error="measured stage clock"))

    class Process:
        pid = 902

        def __init__(self, *args, **kwargs):
            pass

        def poll(self):
            return 75

        def wait(self):
            return 75

    monkeypatch.setattr(runner.subprocess, "Popen", Process)
    assert runner._run_stage(root, plan, job, "arm", "nmcrg", 100.) is None
    assert job["status"] == "budget_hold" and job["requires_user_action"]
    assert job["stages"][0]["returncode"] == 75
    assert job["stages"][0]["status"] == "budget_hold"
    assert job["stages"][0]["child_error"] == "measured stage clock"
    assert job["gpu_elapsed_seconds"] == 4.


def _mock_execution(monkeypatch, root, *, slow=False, fail=None):
    called = []
    monkeypatch.setattr(runner, "cpu_seconds", lambda: 100.)
    monkeypatch.setattr(runner.os, "getpriority", lambda *args: 0)
    monkeypatch.setattr(runner.os, "nice", lambda value: value)

    def run_stage(output, plan, job, stage, method, cpu_origin):
        name = method or stage
        called.append(name)
        job["stages"].append(dict(name=name, status="complete", returncode=0))
        job["gpu_elapsed_seconds"] += 3. if stage != "compare" else 0.
        if name == fail:
            raise RuntimeError("synthetic child failure")
        status = _profile() if stage == "profile" else dict(status="complete", peak_gpu_bytes=1., public_files=[])
        if slow and stage == "profile":
            status["projected_gpu_seconds"] = 9000.
            status["projected_gpu_seconds_by_method"] = {m: 1800. for m in runner.METHODS}
        folder = runner.stage_folder(output, stage, method)
        folder.mkdir(parents=True)
        runner.save(folder / ("profile.json" if stage == "profile" else "status.json"), status)
        return status

    monkeypatch.setattr(runner, "_run_stage", run_stage)
    return called


def test_serial_workflow_completes_seven_stages_and_never_restarts(prepared, monkeypatch):
    root, _ = prepared
    called = _mock_execution(monkeypatch, root)
    job = runner.execute(root)
    assert called == ["profile", *runner.METHODS, "compare"]
    assert job["status"] == "complete" and job["workflow_complete"]
    assert job["gpu_elapsed_seconds"] == 18
    assert all(row["returncode"] == 0 for row in job["stages"])
    assert (root / "execution.json").is_file()
    with pytest.raises(RuntimeError, match="no resume"):
        runner.execute(root)


def test_slow_profile_stops_before_any_formal_arm(prepared, monkeypatch):
    root, _ = prepared
    called = _mock_execution(monkeypatch, root, slow=True)
    job = runner.execute(root)
    assert called == ["profile"]
    assert job["status"] == "budget_hold" and job["requires_user_action"]
    assert not job["workflow_complete"]
    assert not runner.read(root / "profile_receipt.json")["budget_passed"]


def test_actual_stage_overrun_is_held_even_with_completed_receipt(prepared, monkeypatch):
    root, _ = prepared
    called = _mock_execution(monkeypatch, root)
    original = runner._run_stage

    def overrun(*args, **kwargs):
        status = original(*args, **kwargs)
        args[2]["gpu_elapsed_seconds"] = 8000.
        return status

    monkeypatch.setattr(runner, "_run_stage", overrun)
    job = runner.execute(root)
    assert called == ["profile"]
    assert job["status"] == "budget_hold"
    assert not job["workflow_complete"]


def test_child_failure_preserved_without_retry_or_further_arms(prepared, monkeypatch):
    root, _ = prepared
    called = _mock_execution(monkeypatch, root, fail="rsmine")
    with pytest.raises(RuntimeError, match="synthetic child"):
        runner.execute(root)
    job = runner.read(root / "job.json")
    assert called == ["profile", "nmcrg", "rsmine"]
    assert job["status"] == "failed" and not job["workflow_complete"]
    assert job["requires_user_action"]
    assert (root / "arms/nmcrg/status.json").is_file()


def test_no_parent_kill_and_no_double_nice_in_source():
    source = Path(runner.__file__).read_text()
    assert ".kill(" not in source and ".terminate(" not in source
    assert "os.nice(target_nice - inherited_nice)" in source


def test_exclusive_output_receipts_and_safe_public_whitelist(tmp_path):
    runner.save(tmp_path / "plan.json", {}, exclusive=True)
    with pytest.raises(FileExistsError):
        runner.save(tmp_path / "plan.json", {}, exclusive=True)
    folder = tmp_path / "arms/nmcrg"
    folder.mkdir(parents=True)
    runner.save(folder / "status.json", dict(public_files=["metrics.json"]))
    runner.save(folder / "metrics.json", {})
    (folder / "state.pt").write_bytes(b"not public")
    assert {row["path"] for row in runner.public_files(tmp_path)} == {
        "plan.json", "arms/nmcrg/status.json", "arms/nmcrg/metrics.json"}
    runner.save(folder / "status.json", dict(public_files=["state.pt"]))
    with pytest.raises(ValueError, match="namespace"):
        runner.public_files(tmp_path)
    runner.save(folder / "status.json", dict(public_files=["../../plan.json"]))
    with pytest.raises(ValueError, match="namespace"):
        runner.public_files(tmp_path)


def test_public_symlink_rejected(tmp_path):
    outside = tmp_path / "outside.json"
    runner.save(outside, {})
    folder = tmp_path / "arms/nmcrg"
    folder.mkdir(parents=True)
    (folder / "linked.json").symlink_to(outside)
    runner.save(folder / "status.json", dict(public_files=["linked.json"]))
    with pytest.raises(ValueError, match="symlink"):
        runner.public_files(tmp_path)


def _exited_profile_only_hold(root, plan, *, gpu=180., cpu=90.):
    """Create synthetic exited receipts only; never start a process."""
    plan_sha = runner.sha(root / "plan.json")
    parent_pid, child_pid = 99999881, 99999882
    folder = root / "profile"
    folder.mkdir()
    runner.save(folder / "profile.json", dict(_profile(), scientific_result=False))
    runner.save(root / "profile_receipt.json", dict(budget_passed=False,
        actual_gpu_seconds=gpu, actual_cpu_seconds=cpu - 1))
    runner.save(root / "launch.json", dict(status="launched", output=str(root),
        parent_pid=parent_pid, plan_sha256=plan_sha, device=plan["device"], gpu_uuid=plan["gpu_uuid"]))
    runner.save(root / "execution.json", dict(status="started", parent_pid=parent_pid,
        plan_sha256=plan_sha, device=plan["device"], gpu_uuid=plan["gpu_uuid"]))
    job = runner.read(root / "job.json")
    job.update(status="budget_hold", workflow_complete=False, requires_user_action=True,
        parent_pid=parent_pid, child_pid=None, ended_at="synthetic-end", gpu_elapsed_seconds=gpu,
        cpu_seconds=cpu, stages=[dict(name="profile", stage="profile", method=None,
            pid=child_pid, status="complete", returncode=0, gpu_stage=True,
            ended_at="synthetic-end", wall_seconds=170., cpu_seconds=80.)])
    runner.save(root / "job.json", job)
    return job


def test_prior_profile_cost_pinned_and_carried_through_prepare_verify_and_execute(prepared, monkeypatch):
    prior, plan = prepared
    _exited_profile_only_hold(prior, plan)
    before = {str(path): path.read_bytes() for path in prior.rglob("*.json")}
    root = prior.with_name("new_timing_only_v2")
    new_plan = runner.prepare(root, prior_profile_root=prior)
    assert new_plan["budget_carry_in"] == dict(gpu_seconds=180., cpu_seconds=90.)
    assert new_plan["prior_profile_root"] == str(prior)
    assert len(new_plan["prior_profile_receipt"]["public_json"]) == 6
    assert runner.read(root / "job.json")["gpu_elapsed_seconds"] == 180.
    assert runner.read(root / "job.json")["cpu_seconds"] == 90.
    assert runner.verify(root)[0] == new_plan
    called = _mock_execution(monkeypatch, root)
    job = runner.execute(root)
    assert called == ["profile", *runner.METHODS, "compare"]
    assert job["status"] == "complete"
    assert job["gpu_elapsed_seconds"] == 198.
    assert job["cpu_seconds"] == 90.
    assert runner.read(root / "execution.json")["budget_carry_in"] == new_plan["budget_carry_in"]
    assert before == {str(path): path.read_bytes() for path in prior.rglob("*.json")}


def test_prior_profile_receipt_modification_or_accounting_reset_fails_closed(prepared):
    prior, plan = prepared
    _exited_profile_only_hold(prior, plan)
    root = prior.with_name("new_timing_only_v2")
    runner.prepare(root, prior_profile_root=prior)
    job = runner.read(root / "job.json")
    job["cpu_seconds"] = 0.
    runner.save(root / "job.json", job)
    with pytest.raises(RuntimeError, match="dropped prior"):
        runner.verify(root)
    job["cpu_seconds"] = 90.
    runner.save(root / "job.json", job)
    runner.save(prior / "profile/profile.json", dict(_profile(), scientific_result=False, changed=True))
    with pytest.raises(RuntimeError, match="Pinned prior"):
        runner.verify(root)


@pytest.mark.parametrize("mutation", ["failed", "running", "formal_stage", "formal_directory", "nonzero",
    "data", "methods", "seed", "updates", "draws", "policy", "profile_failed", "launch_hash", "execution_hash"])
def test_prior_profile_strictly_rejects_ineligible_or_changed_authority(prepared, mutation):
    prior, plan = prepared
    job = _exited_profile_only_hold(prior, plan)
    if mutation in ("failed", "running"):
        job["status"] = mutation
        runner.save(prior / "job.json", job)
    elif mutation == "formal_stage":
        job["stages"].append(dict(name="nmcrg", stage="arm", status="running"))
        runner.save(prior / "job.json", job)
    elif mutation == "formal_directory":
        (prior / "arms").mkdir()
    elif mutation == "nonzero":
        job["stages"][0]["returncode"] = 1
        runner.save(prior / "job.json", job)
    elif mutation == "profile_failed":
        runner.save(prior / "profile/profile.json", dict(_profile(), status="failed", scientific_result=False))
    elif mutation in ("launch_hash", "execution_hash"):
        path = prior / ("launch.json" if mutation == "launch_hash" else "execution.json")
        value = runner.read(path)
        value["plan_sha256"] = "wrong"
        runner.save(path, value)
    else:
        updated = deepcopy(plan)
        key, value = {
            "data": ("data_sha256", "different-data"), "methods": ("methods", ["nmcrg"]),
            "seed": ("seed", 92), "updates": ("updates", 6000), "draws": ("C_draws", 1),
            "policy": ("resource_policy", dict(plan["resource_policy"], gpu_seconds=14400.)),
        }[mutation]
        updated[key] = value
        runner.save(prior / "plan.json", updated)
    with pytest.raises((ValueError, RuntimeError)):
        runner.prepare(prior.with_name("new_timing_only_v2"), prior_profile_root=prior)
    assert not prior.with_name("new_timing_only_v2").exists()


def test_prior_profile_running_declared_process_rejected_without_signals(prepared, monkeypatch):
    prior, plan = prepared
    _exited_profile_only_hold(prior, plan)
    monkeypatch.setattr(runner, "_prior_active_processes", lambda root, pids: [pids[0]])
    with pytest.raises(ValueError, match="running declared processes"):
        runner.prepare(prior.with_name("new_timing_only_v2"), prior_profile_root=prior)


def test_prior_profile_different_new_budget_policy_rejected(prepared):
    prior, plan = prepared
    _exited_profile_only_hold(prior, plan)
    with pytest.raises(ValueError, match="resource_policy"):
        runner.prepare(prior.with_name("new_timing_only_v2"), prior_profile_root=prior, gpu_hours=3.)


def test_live_child_cpu_updates_include_prior_cost(prepared, monkeypatch):
    prior, plan = prepared
    _exited_profile_only_hold(prior, plan)
    root = prior.with_name("new_timing_only_v2")
    new_plan = runner.prepare(root, prior_profile_root=prior)
    monkeypatch.setattr(runner, "cpu_seconds", lambda: 112.)
    assert runner._cumulative_cpu(new_plan, 100.) == 102.


def test_prior_cost_can_hold_new_profile_without_budget_reset(prepared, monkeypatch):
    prior, plan = prepared
    _exited_profile_only_hold(prior, plan, gpu=7199., cpu=14000.)
    root = prior.with_name("new_timing_only_v2")
    runner.prepare(root, prior_profile_root=prior)
    called = _mock_execution(monkeypatch, root)
    job = runner.execute(root)
    assert called == ["profile"]
    assert job["status"] == "budget_hold" and not job["workflow_complete"]
    assert job["gpu_elapsed_seconds"] == 7202.
    assert job["cpu_seconds"] == 14000.


def _failed_nmcrg_label_run(root, plan, *, gpu=366.33100264519453, cpu=383.784937):
    """Synthetic complete endpoints and exact evaluation-only failure receipts."""
    job = _exited_profile_only_hold(root, plan, gpu=gpu, cpu=cpu)
    runner.save(root / "profile_receipt.json", dict(budget_passed=True))
    pilot = job["stages"][0]
    pilot.update(wall_seconds=20., cpu_seconds=10.)
    arm = root / "arms/nmcrg"
    arm.mkdir(parents=True)
    runner.save(arm / "status.json", dict(status="failed", method="nmcrg", C_step=12000, R_step=12000,
        current="evaluation_and_full5D_roots", error=runner.LABEL_EVALUATION_ERROR, requires_user_action=True))
    manifest = dict(method="nmcrg", seed=91, data_sha256=runner.DATA_SHA, source_data=plan["source_data"],
        labels_used_in_training=False, C_draws=3, scales=[108, 36, 12],
        native_C=dict(filter_width=9, stride=3, overlapping=True, batch_configurations=50,
            lr=.001, adam_epsilon=1e-8, updates=12000, dtype="float32"),
        common_readout=dict(dimension=5, R_updates=12000, R_seed=91,
            R_milestones=[6000, 9000], gamma=.3, batch_physical_chains=16, one_step_weight=5,
            rollout_weight=5, target_scale_weights=[1, 1], coordinate_reduction="sum", dtype="float64",
            C_or_PCA_gradients_from_R=False))
    runner.save(arm / "manifest.json", manifest)
    steps = [1, *range(100, 12001, 100)]
    c_history = [dict(stage="C", step=step, native_loss=0.) for step in steps]
    r_history = [dict(stage="R", step=step, loss=0.) for step in steps]
    runner.save(arm / "history_progress.json", c_history + r_history)
    runner.save(arm / "native_C_history.json", c_history)
    runner.save(arm / "readout.json", dict(
        PCA=dict(kind="TRAIN_shared_3x3_PCA_v1", fit_split="train", labels_used=False,
            dimension=5, center=[0.] * 512, basis=[[0.] * 5 for _ in range(512)], singular_values=[1.] * 5),
        affine=dict(kind="TRAIN_affine_16_motif_readout_v1", fit_split="train", labels_used=False,
            weight=[[0.] * 16 for _ in range(5)], bias=[0.] * 16, singular_values=[1.] * 5)))
    (arm / "native_C.pt").write_bytes(b"synthetic complete native C; NEVER LOAD")
    (arm / "common_R.pt").write_bytes(b"synthetic complete common R; NEVER LOAD")
    job.update(status="failed", current="nmcrg", error=runner.LABEL_PARENT_ERROR)
    job["stages"].append(dict(name="nmcrg", stage="arm", method="nmcrg", pid=99999883,
        status="failed", returncode=1, gpu_stage=True, ended_at="synthetic-end",
        child_status="failed", child_error=runner.LABEL_EVALUATION_ERROR,
        wall_seconds=120., cpu_seconds=100.))
    runner.save(root / "job.json", job)
    return job


def test_manual_fresh_retraining_freezes_failed_run_without_loading_weights(prepared, monkeypatch):
    prior, plan = prepared
    _failed_nmcrg_label_run(prior, plan)
    before = {str(path): path.read_bytes() for path in prior.rglob("*") if path.is_file()}
    root = prior.with_name("manual_fresh_v3")
    new = runner.prepare(root, retrain_from_root=prior)
    assert new["manual_retraining_authorized"] is True
    assert new["manual_retraining_reason"] == runner.MANUAL_RETRAINING_REASON
    assert new["all_methods_from_scratch"] is True
    assert new["read_failed_training_weights"] is False
    assert new["prior_state_used_for_initialization"] is False and new["resume"] is False
    assert new["prior_profile_root"] is None
    assert len(new["retraining_receipt"]["public_json"]) == 11
    assert len(new["retraining_receipt"]["model_file_fingerprints"]) == 2
    assert new["budget_carry_in"] == dict(gpu_seconds=366.33100264519453, cpu_seconds=383.784937)
    assert runner.verify(root)[0] == new
    called = _mock_execution(monkeypatch, root)
    completed = runner.execute(root)
    assert called == ["profile", *runner.METHODS, "compare"]
    assert completed["gpu_elapsed_seconds"] == 384.33100264519453
    assert completed["cpu_seconds"] == 383.784937
    assert before == {str(path): path.read_bytes() for path in prior.rglob("*") if path.is_file()}


def test_manual_retraining_carries_already_cumulative_cost_only_once(prepared):
    v1, plan1 = prepared
    _exited_profile_only_hold(v1, plan1)
    v2 = v1.with_name("timing_v2")
    plan2 = runner.prepare(v2, prior_profile_root=v1)
    assert plan2["budget_carry_in"] == dict(gpu_seconds=180., cpu_seconds=90.)
    _failed_nmcrg_label_run(v2, plan2, gpu=500., cpu=450.)
    v3 = v1.with_name("fresh_v3")
    plan3 = runner.prepare(v3, retrain_from_root=v2)
    assert plan3["budget_carry_in"] == dict(gpu_seconds=500., cpu_seconds=450.)
    assert plan3["retraining_receipt"]["carry_is_already_cumulative"]
    assert plan3["retraining_receipt"]["earlier_cost_added_again"] is False


@pytest.mark.parametrize("mutation", ["running", "budget_hold", "different_failure", "different_parent_failure",
    "C_incomplete", "R_incomplete", "other_arm", "extra_stage", "nonzero_profile", "missing_C", "missing_R",
    "short_history", "short_native_history", "readout_missing", "readout_truncated", "manifest_changed",
    "data", "methods", "seed", "updates", "C_draws", "resource_policy", "launch_hash", "execution_hash"])
def test_manual_retraining_rejects_other_failures_or_protocol_changes(prepared, mutation):
    prior, plan = prepared
    job = _failed_nmcrg_label_run(prior, plan)
    arm = prior / "arms/nmcrg"
    if mutation in ("running", "budget_hold"):
        job["status"] = mutation
        runner.save(prior / "job.json", job)
    elif mutation == "different_parent_failure":
        job["error"] = "RuntimeError: some other failure"
        runner.save(prior / "job.json", job)
    elif mutation in ("different_failure", "C_incomplete", "R_incomplete"):
        value = runner.read(arm / "status.json")
        value[{"different_failure": "error", "C_incomplete": "C_step", "R_incomplete": "R_step"}[mutation]] = (
            "IndexError: an unrelated error" if mutation == "different_failure" else 11900)
        runner.save(arm / "status.json", value)
    elif mutation == "other_arm":
        (prior / "arms/rsmine").mkdir()
    elif mutation == "extra_stage":
        job["stages"].append(dict(name="rsmine"))
        runner.save(prior / "job.json", job)
    elif mutation == "nonzero_profile":
        job["stages"][0]["returncode"] = 1
        runner.save(prior / "job.json", job)
    elif mutation in ("missing_C", "missing_R"):
        (arm / ("native_C.pt" if mutation == "missing_C" else "common_R.pt")).unlink()
    elif mutation in ("short_history", "short_native_history"):
        path = arm / ("history_progress.json" if mutation == "short_history" else "native_C_history.json")
        runner.save(path, runner.read(path)[:-1])
    elif mutation == "readout_missing":
        (arm / "readout.json").unlink()
    elif mutation == "readout_truncated":
        value = runner.read(arm / "readout.json")
        value["PCA"]["basis"] = []
        runner.save(arm / "readout.json", value)
    elif mutation == "manifest_changed":
        value = runner.read(arm / "manifest.json")
        value["native_C"]["filter_width"] = 3
        runner.save(arm / "manifest.json", value)
    elif mutation in ("launch_hash", "execution_hash"):
        path = prior / ("launch.json" if mutation == "launch_hash" else "execution.json")
        value = runner.read(path)
        value["plan_sha256"] = "wrong"
        runner.save(path, value)
    else:
        changed = deepcopy(plan)
        key, value = {
            "data": ("data_sha256", "wrong"), "methods": ("methods", ["nmcrg"]),
            "seed": ("seed", 92), "updates": ("updates", 6000), "C_draws": ("C_draws", 1),
            "resource_policy": ("resource_policy", dict(plan["resource_policy"], gpu_seconds=14400.)),
        }[mutation]
        changed[key] = value
        runner.save(prior / "plan.json", changed)
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        runner.prepare(prior.with_name("manual_fresh_v3"), retrain_from_root=prior)
    assert not prior.with_name("manual_fresh_v3").exists()


def test_manual_retraining_refuses_live_source_and_mutually_exclusive_options(prepared, monkeypatch):
    prior, plan = prepared
    _failed_nmcrg_label_run(prior, plan)
    root = prior.with_name("manual_fresh_v3")
    with pytest.raises(ValueError, match="mutually exclusive"):
        runner.prepare(root, retrain_from_root=prior, prior_profile_root=prior)
    monkeypatch.setattr(runner, "_prior_active_processes", lambda root, pids: [pids[-1]])
    with pytest.raises(ValueError, match="running declared processes"):
        runner.prepare(root, retrain_from_root=prior)


@pytest.mark.parametrize("target", ["arms/nmcrg/common_R.pt", "arms/nmcrg/manifest.json"])
def test_manual_retraining_verifies_frozen_json_and_small_model_fingerprints(prepared, target):
    prior, plan = prepared
    _failed_nmcrg_label_run(prior, plan)
    root = prior.with_name("manual_fresh_v3")
    runner.prepare(root, retrain_from_root=prior)
    (prior / target).write_bytes(b"modified")
    with pytest.raises(RuntimeError, match="Pinned failed-run"):
        runner.verify(root)


def test_manual_retraining_does_not_loosen_profile_only_mode(prepared):
    prior, plan = prepared
    _failed_nmcrg_label_run(prior, plan)
    with pytest.raises(ValueError, match="profile-only"):
        runner.prepare(prior.with_name("wrong_profile_mode"), prior_profile_root=prior)
