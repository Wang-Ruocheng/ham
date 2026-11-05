"""Exact continuation orchestration contracts; no SSH, CUDA, production data or training."""
import copy
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts import run_phi4_rg_reference_continue as runner


CODE = Path(__file__).resolve().parents[1]


@pytest.fixture
def provenance(tmp_path, monkeypatch):
    source = tmp_path/"original"
    (source/"run").mkdir(parents=True)
    runner.save(source/"run"/"summary.json", dict(timing=dict(evaluation_seconds=4.)))
    receipt = dict(source=str(source), source_cpu_seconds=816.394323,
        source_gpu_seconds=556.737061, source_directory_bytes=82_000_000,
        plan_sha256=runner.SOURCE_PLAN_SHA, checkpoint_sha256=runner.SOURCE_CHECKPOINT_SHA,
        receipt=dict(source_steps=3000, checkpoint_sha256=runner.SOURCE_CHECKPOINT_SHA),
        config=runner.original.default_config())
    monkeypatch.setattr(runner, "_source_contract", lambda root: copy.deepcopy(receipt))
    monkeypatch.setattr(runner, "validate_host", lambda **kwargs: None)
    return receipt


@pytest.fixture
def prepared(tmp_path, monkeypatch, provenance):
    root = tmp_path/"continuation"
    runner.prepare(root, code_root=CODE, source=provenance["source"])
    monkeypatch.setattr(runner, "unchanged", lambda plan: None)
    return root


def test_plan_is_read_only_exact_continuation(tmp_path, provenance):
    root = tmp_path/"not_created"
    plan = runner.build_plan(root, CODE, provenance["source"])
    assert not root.exists()
    assert plan["training"]["source_steps"] == 3000
    assert plan["training"]["additional_steps"] == 3000
    assert plan["training"]["total_steps"] == 6000
    assert plan["training"]["restore_all_three_Adam_states"]
    assert plan["training"]["restore_training_RNG_states"]
    assert not plan["training"]["resample"]
    assert not plan["training"]["recalibrate_Q"]
    assert not plan["training"]["labels_used"]
    assert plan["training"]["closure_objective"] == "all_start_anchor"
    assert plan["training"]["closure_scale_weights"] is None
    assert plan["resource_policy"]["count_original_completed_experiment"]
    assert plan["resource_policy"]["gpu_seconds"] == 3600
    assert plan["resource_policy"]["cpu_seconds"] == 7200


@pytest.mark.parametrize("kind", ["source", "inside_source", "source_parent", "code", "code_parent"])
def test_overlapping_output_paths_rejected(provenance, kind):
    source = Path(provenance["source"])
    target = dict(source=source, inside_source=source/"new", source_parent=source.parent,
        code=CODE, code_parent=CODE.parent)[kind]
    with pytest.raises(ValueError, match="separate"):
        runner.build_plan(target, CODE, source)


@pytest.mark.parametrize("field,value", [("source_cpu_seconds",7200), ("source_gpu_seconds",3600),
    ("source_directory_bytes",runner.DIRECTORY_BYTES)])
def test_original_consumption_cannot_be_reset(tmp_path, monkeypatch, provenance, field, value):
    provenance[field] = value
    monkeypatch.setattr(runner, "_source_contract", lambda root: copy.deepcopy(provenance))
    with pytest.raises(runner.BudgetHold):
        runner.build_plan(tmp_path/"unused",CODE,provenance["source"])


def test_source_science_must_match_exact_original(tmp_path, monkeypatch, provenance):
    monkeypatch.setattr(runner, "SOURCE_SCIENCE_SHA", {"experiments/phi4_qcr.py":"0"*64})
    with pytest.raises(RuntimeError, match="science code"):
        runner.build_plan(tmp_path/"unused",CODE,provenance["source"])


def test_prepared_output_cannot_be_overwritten(prepared):
    with pytest.raises(FileExistsError):
        runner.prepare(prepared)


def test_frozen_plan_change_rejected(prepared):
    plan = runner.read(prepared/"plan.json")
    plan["training"]["additional_steps"] = 1
    runner.save(prepared/"plan.json",plan)
    with pytest.raises(RuntimeError, match="Frozen"):
        runner.load_plan(prepared)


def test_source_change_rejected(prepared, monkeypatch, provenance):
    provenance["checkpoint_sha256"] = "changed"
    monkeypatch.setattr(runner,"_source_contract",lambda source:provenance)
    with pytest.raises(RuntimeError, match="source fingerprints"):
        runner.load_plan(prepared)


def test_stage_sequence_has_no_sampling_and_profile_is_discarded(prepared):
    plan,_ = runner.load_plan(prepared)
    stages = runner.planned_commands(prepared,plan)
    assert [name for name,_,_ in stages] == ["training_preflight","training","report"]
    for name,cmd,gpu in stages[:2]:
        assert gpu
        assert cmd[cmd.index("--source")+1] == plan["source"]
        assert cmd[cmd.index("--additional-steps")+1] == "3000"
        assert cmd[cmd.index("--expected-gpu-uuid")+1] == runner.GPU_UUID
        assert cmd[cmd.index("--device")+1] == "cuda:2"
        assert "--seed" not in cmd and "--data" not in cmd
        assert ("--profile-only" in cmd) == (name == "training_preflight")
        assert "profile_run" in cmd[cmd.index("--output")+1] if name == "training_preflight" else True
    assert "scripts.report_phi4_rg_reference_continue" in stages[-1][1]


def test_two_idle_checks_with_ten_seconds_separation(prepared,monkeypatch):
    calls,sleeps = [],[]
    monkeypatch.setattr(runner,"resource_snapshot",lambda plan:calls.append(1) or
        dict(gpu_uuid=runner.GPU_UUID,gpu_ready=True,host_ready=True))
    monkeypatch.setattr(runner.time,"sleep",sleeps.append)
    plan,job=runner.load_plan(prepared)
    receipt=runner.admission(prepared,plan,job,"training",True)
    assert len(calls)==2 and sleeps==[10]
    assert receipt["status"]=="admitted"


def test_occupied_gpu_does_not_count_low_utilization_as_idle(prepared,monkeypatch):
    calls,sleeps = [],[]
    states=iter([False,True,True])
    monkeypatch.setattr(runner,"resource_snapshot",lambda plan:calls.append(1) or
        dict(gpu_uuid=runner.GPU_UUID,gpu_ready=next(states),host_ready=True,gpu_compute_pids=[123]))
    monkeypatch.setattr(runner.time,"sleep",sleeps.append)
    plan,job=runner.load_plan(prepared)
    runner.admission(prepared,plan,job,"training",True)
    assert len(calls)==3 and sleeps==[30,10]


def test_gpu_identity_mismatch_never_migrates(prepared,monkeypatch):
    monkeypatch.setattr(runner,"resource_snapshot",lambda plan:
        dict(gpu_uuid="wrong",gpu_ready=True,host_ready=True))
    plan,job=runner.load_plan(prepared)
    with pytest.raises(RuntimeError,match="no migration"):
        runner.admission(prepared,plan,job,"training",True)


def test_admission_wait_counts_original_gpu_budget(prepared,monkeypatch):
    plan,job=runner.load_plan(prepared)
    job["gpu_stage_wall_seconds"]=runner.GPU_SECONDS-plan["source_provenance"]["source_gpu_seconds"]
    query=Mock()
    monkeypatch.setattr(runner,"resource_snapshot",query)
    with pytest.raises(runner.BudgetHold):
        runner.admission(prepared,plan,job,"training",True)
    query.assert_not_called()


def summary(profile):
    steps=3020 if profile else 6000
    return dict(actual_steps=steps,source_steps=3000,profile_only=profile,
        labels_used_in_training=False,frozen_evaluation_verified=not profile,
        peak_cuda_bytes=1024,device=dict(uuid=runner.GPU_UUID),
        optimizer_steps={key:dict(min=steps,max=steps) for key in ("Q","C","R")},
        timing=dict(initialization_seconds=1.,training_seconds=2.,evaluation_seconds=0.,wall_seconds=3.))


@pytest.mark.parametrize("field,value", [("actual_steps",3000),("source_steps",0),
    ("profile_only",True),("labels_used_in_training",True),
    ("peak_cuda_bytes",runner.GPU_PEAK_BYTES+1),("frozen_evaluation_verified",False)])
def test_training_summary_rejects_inexact_endpoint(field,value):
    row=summary(False)
    row[field]=value
    with pytest.raises(RuntimeError):
        runner.validate_training_summary(row,profile=False)


def test_adam_clocks_must_all_be_6000():
    row=summary(False)
    row["optimizer_steps"]["Q"]["min"]=3000
    with pytest.raises(RuntimeError):
        runner.validate_training_summary(row,profile=False)


def test_profile_projection_includes_original_full_init_eval_and_reserves(prepared):
    plan,_=runner.load_plan(prepared)
    receipt=runner.profile_projection(summary(True),5.,20.,5.,plan)
    expected_formal=1.5*(1+2+3000*.1+4)+300
    assert receipt["projected_formal_gpu_seconds_conservative"]==pytest.approx(expected_formal)
    assert receipt["projected_total_gpu_seconds_conservative"]==pytest.approx(556.737061+20+expected_formal+10)
    assert receipt["budget_passed"]
    assert receipt["profile_updates_discarded"]==20


def test_slow_profile_cannot_shrink_formal_updates(prepared):
    plan,_=runner.load_plan(prepared)
    row=summary(True)
    row["timing"]["training_seconds"]=200
    result=runner.profile_projection(row,205.,210.,205.,plan)
    assert not result["budget_passed"]
    assert result["formal_additional_steps"]==3000


def fake_execution(root,monkeypatch,*,fail=None,slow=False):
    # CPU-only test invocations may mask CUDA globally; the authorized parent
    # rejects any mask before execution, so simulate that validated environment.
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES",raising=False)
    monkeypatch.setattr(runner,"admission",lambda *args:None)
    monkeypatch.setattr(runner.os,"getpriority",lambda *args:0)
    nice=Mock()
    monkeypatch.setattr(runner.os,"nice",nice)
    calls=[]
    def popen(command,**kwargs):
        output=Path(command[command.index("--output")+1])
        output.mkdir()
        if "scripts.report_phi4_rg_reference_continue" in command:
            name="report"
            runner.save(output/"status.json",dict(workflow_complete=True))
        else:
            profile="--profile-only" in command
            name="training_preflight" if profile else "training"
            row=summary(profile)
            if profile and slow:
                row["timing"]["training_seconds"]=200
            runner.save(output/"summary.json",row)
            runner.save(output/"status.json",dict(status="complete",workflow_complete=not profile,
                returncode=0,profile_complete=profile))
        calls.append((name,command,kwargs))
        return Mock(pid=1234,poll=lambda:2 if fail==name else 0,
            wait=lambda **kwargs:2 if fail==name else 0,returncode=2 if fail==name else 0)
    monkeypatch.setattr(runner.subprocess,"Popen",popen)
    return calls,nice


def test_full_workflow_is_new_root_and_source_unchanged(prepared,monkeypatch):
    calls,nice=fake_execution(prepared,monkeypatch)
    result=runner.execute(prepared)
    assert result["status"]=="complete" and result["workflow_complete"]
    assert [row[0] for row in calls]==["training_preflight","training","report"]
    assert all(stage["returncode"]==0 for stage in result["stages"])
    assert runner.read(prepared/"source_before.json")==runner.read(prepared/"source_after.json")
    assert result["cumulative_gpu_seconds"]>=556.737061
    assert result["cumulative_cpu_seconds"]>=816.394323
    nice.assert_called_once_with(10)
    assert all(call[2]["env"]["OMP_NUM_THREADS"]=="1" for call in calls)
    assert "CUDA_VISIBLE_DEVICES" not in calls[0][2]["env"]
    assert calls[-1][2]["env"]["CUDA_VISIBLE_DEVICES"]==""
    with pytest.raises(ValueError,match="already executed"):
        runner.execute(prepared)


def test_profile_failure_preserves_outputs_and_never_launches_formal(prepared,monkeypatch):
    calls,_=fake_execution(prepared,monkeypatch,fail="training_preflight")
    result=runner.execute(prepared)
    assert result["status"]=="failed" and result["requires_user_action"]
    assert [row[0] for row in calls]==["training_preflight"]
    assert (prepared/"profile_run"/"summary.json").exists()
    assert not (prepared/"run").exists()


def test_projection_hold_never_launches_formal(prepared,monkeypatch):
    calls,_=fake_execution(prepared,monkeypatch,slow=True)
    result=runner.execute(prepared)
    assert result["status"]=="budget_hold" and not result["workflow_complete"]
    assert [row[0] for row in calls]==["training_preflight"]


def test_formal_resource_wait_rechecks_conservative_projection(prepared,monkeypatch):
    calls,_=fake_execution(prepared,monkeypatch)
    clock=[0.]
    def monotonic():
        clock[0]+=.001
        return clock[0]
    def admission(root,plan,job,name,gpu):
        if name=="training":
            clock[0]+=2300.
    monkeypatch.setattr(runner.time,"monotonic",monotonic)
    monkeypatch.setattr(runner,"admission",admission)
    result=runner.execute(prepared)
    assert result["status"]=="budget_hold"
    assert "admission wait" in result["error"]
    assert [row[0] for row in calls]==["training_preflight"]
    assert not (prepared/"run").exists()


def test_public_whitelist_excludes_checkpoints_and_profile_arrays(prepared):
    for folder in ("run","report","profile_run"):
        (prepared/folder).mkdir()
    for name in ("run/training_step_6000.pt","run/evaluation_arrays.npz","report/R_vs_g.png",
                 "profile_run/evaluation_arrays.npz","profile_run/summary.json"):
        (prepared/name).write_bytes(b"fixture")
    files=runner.final_public_files(prepared)
    assert "run/evaluation_arrays.npz" in files
    assert "report/R_vs_g.png" in files
    assert "profile_run/summary.json" in files
    assert all(not name.endswith(".pt") for name in files)
    assert "profile_run/evaluation_arrays.npz" not in files


def test_budget_stops_only_own_child(prepared,monkeypatch):
    plan,job=runner.load_plan(prepared)
    monkeypatch.setattr(runner,"process_snapshot",lambda pid:dict(pid=pid,cpu_seconds=7200))
    child=Mock(pid=4321,poll=lambda:None,returncode=-15)
    result=runner._wait_child(child,prepared,job,plan,cpu_start=runner.cpu_seconds(),
        stage_start=runner.time.monotonic(),gpu_elapsed=0.,gpu=True)
    assert result==(-15,True)
    child.terminate.assert_called_once_with()
    child.kill.assert_not_called()
