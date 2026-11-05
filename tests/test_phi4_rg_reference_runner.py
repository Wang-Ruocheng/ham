"""Frozen u=1 orchestration contracts; no production sampling, SSH or CUDA."""
import copy
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts import run_phi4_rg_reference as runner


CODE = Path(__file__).resolve().parents[1]


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root = tmp_path / "fresh_reference"
    runner.prepare(root, code_root=CODE)
    monkeypatch.setattr(runner, "unchanged", lambda plan: None)
    monkeypatch.setattr(runner.socket, "gethostname", lambda: runner.HOST)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    return root


def test_pure_plan_fixed_design_and_actual_config(tmp_path):
    root = tmp_path / "not_created"
    config = runner.read(CODE/"configs/phi4_u1_rg_reference_b3_g1_seed91.json")
    plan = runner.build_plan(root, CODE, config)
    assert not root.exists()
    assert config == plan["config"]
    p, t = plan["physical"], plan["training"]
    assert p["options"]["u_values"] == [1.]
    assert p["options"]["r_values"] == pytest.approx([-1.6724682+i*.8/12 for i in range(13)])
    assert p["total_hmc_trajectories"] == 106496
    assert p["test_cell_indices"] == [0,3,6,9,12]
    assert plan["sampler_preflight"]["cell_indices"] == [1,5,7,11]
    assert set(plan["sampler_preflight"]["cell_indices"]).issubset(p["train_cell_indices"])
    assert plan["sampler_preflight"]["expected_pilots"] == 12
    assert plan["total_planned_force_site_evaluations_including_preflight"] == 1_826_592_768
    assert plan["total_planned_force_site_evaluations_including_preflight"] < 2_000_000_000
    assert plan["total_planned_hmc_trajectories_including_preflight"] == 106496+1920
    assert t["steps"] == 3000 and t["seed"] == 91 and t["profile_seed"] == 191
    assert t["closure_objective"] == "all_start_anchor" and t["closure_scale_weights"] is None
    assert t["lattice_sizes"] == [36,12,4] and t["theory_dim"] == 1
    assert plan["resource_policy"]["minimum_host_available_gib"] == 36
    assert plan["resource_policy"]["gpu_seconds"] == 3600
    assert plan["evaluation"]["extrapolation_test_cell_indices"] == [0,12]
    assert "recorded_physical_work_ledger" not in plan


@pytest.mark.parametrize("section,key,value", [
    ("reference","u",.2), ("data","u_values",[.2]), ("data","seed",91),
    ("data","test_offset",1), ("training","steps",2999),
    ("training","q_initialization","random_legacy"), ("model","block_size",2),
    ("loss","closure_scale_weights",[2,1]), ("loss","coupling_supervision",True),
])
def test_design_changes_are_rejected(tmp_path, section, key, value):
    config = runner.default_config()
    config[section][key] = value
    with pytest.raises(ValueError, match="contract"):
        runner.build_plan(tmp_path/"unused", CODE, config)
    assert not (tmp_path/"unused").exists()


def test_prepared_plan_and_config_are_immutable(prepared):
    with pytest.raises(FileExistsError):
        runner.prepare(prepared)
    plan, _ = runner.load_plan(prepared)
    assert plan["config_sha256"] == runner.sha(prepared/"config.json")
    changed = copy.deepcopy(plan)
    changed["training"]["steps"] = 1
    runner.save(prepared/"plan.json", changed)
    with pytest.raises(RuntimeError, match="Frozen"):
        runner.load_plan(prepared)


def test_sampling_cli_matches_frozen_plan(prepared, monkeypatch):
    import scripts.prepare_phi4_hybrid as data_cli
    from recurrent_rg.data.phi4_hybrid_pipeline import generation_plan
    captured = []
    monkeypatch.setattr(data_cli, "generate_dataset", lambda output, options:
        captured.append(options) or dict(schema="fixture",shards=[1],warning_count=0))
    plan = runner.read(prepared/"plan.json")
    cmd = runner.sampling_command(prepared, plan)
    data_cli.main(cmd[cmd.index("scripts.prepare_phi4_hybrid")+1:])
    import json
    assert json.loads(json.dumps(generation_plan(captured[0]))) == plan["physical"]


def test_stage_sequence_and_label_free_fresh_initialization(prepared):
    plan = runner.read(prepared/"plan.json")
    stages = runner.planned_commands(prepared, plan)
    assert [name for name,_,_ in stages] == ["sampler_preflight","physical_sampling","training_preflight","training","report"]
    assert [name for name,_,gpu in stages if gpu] == ["training_preflight","training"]
    for index, seed in ((2,"191"),(3,"91")):
        cmd = stages[index][1]
        assert cmd[cmd.index("--seed")+1] == seed
        assert cmd[cmd.index("--device")+1] == "cuda:2"
        assert cmd[cmd.index("--expected-gpu-uuid")+1] == runner.GPU_UUID
        assert cmd[cmd.index("--q-initialization")+1] == "train_mean_bias"
        assert not {"--source","--resume","--expected-random-construction-sha256","--labels"}.intersection(cmd)
        assert not any("r-critical" in part for part in cmd)
    assert "scripts.report_phi4_rg_reference" in stages[-1][1]
    assert stages[-1][1][-4:] == ["--data",str(prepared/"data"),"--output",str(prepared/"report")]


def test_resource_snapshot_uses_required_host_and_gpu_limits(prepared, monkeypatch):
    plan = runner.read(prepared/"plan.json")
    original = Path.read_text
    state = dict(gib=36, load=30., io=19.99, free=12288, pids="")
    def read_text(path, *args, **kwargs):
        if str(path) == "/proc/meminfo":
            return f"MemAvailable: {int(state['gib']*1024**2)} kB\n"
        if str(path) == "/proc/pressure/io":
            return f"some avg10=0\nfull avg10={state['io']} avg60=0\n"
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,"read_text",read_text)
    monkeypatch.setattr(runner.os,"cpu_count",lambda:24)
    monkeypatch.setattr(runner.os,"getloadavg",lambda:(state["load"],0,0))
    def query(cmd, **kwargs):
        assert "--query-gpu=uuid,memory.free" in cmd or "--query-compute-apps=gpu_uuid,pid" in cmd
        return f"{runner.GPU_UUID}, {state['free']}\n" if "-i" in cmd else state["pids"]
    monkeypatch.setattr(runner.subprocess,"check_output",query)
    assert runner.resource_snapshot(plan)["gpu_ready"]
    for key, bad in (("gib",35.9),("load",30.01),("io",20.),("free",12287),("pids",f"{runner.GPU_UUID}, 42")):
        previous = state[key]
        state[key] = bad
        assert not runner.resource_snapshot(plan)["gpu_ready"]
        state[key] = previous


def test_gpu_requires_two_idle_checks_with_ten_second_separation(prepared, monkeypatch):
    calls, sleeps = [], []
    monkeypatch.setattr(runner,"resource_snapshot",lambda plan:
        calls.append(1) or dict(gpu_uuid=runner.GPU_UUID,gpu_ready=True,host_ready=True))
    monkeypatch.setattr(runner.time,"sleep",sleeps.append)
    plan, job = runner.load_plan(prepared)
    runner.admission(prepared,plan,job,"training",True)
    assert len(calls) == 2 and sleeps == [10]


def fake_chain(*args, **kwargs):
    assert args[1] == 1. and kwargs["h"] == 0.
    assert kwargs["samples"] == 8 and kwargs["burn_in"] == 128 and kwargs["spacing"] == 4
    assert kwargs["step_size"] == .08 and kwargs["leapfrog_steps"] == 12
    return dict(cpu_seconds=.3,wall_seconds=.3,hmc_trajectories=160,
        force_site_evaluations=160*13*36**2,production_acceptance=.9,max_absolute_delta_h=.5,
        warnings=["finite-chain warning"])


def test_pilot_is_train_only_and_counts_all_new_work(prepared, monkeypatch):
    from recurrent_rg.data import phi4_hybrid as kernel
    seen = []
    def sample(*args,**kwargs):
        seen.append(args[0])
        return fake_chain(*args,**kwargs)
    monkeypatch.setattr(kernel,"sample_hybrid_chain",sample)
    result = runner.sampler_preflight(prepared)
    assert result["status"] == "complete" and len(seen) == 12
    assert set(seen) == set(runner.read(prepared/"plan.json")["sampler_preflight"]["r_values"])
    assert result["calibration_uses_only_TRAIN"] and not result["production_data_reused"]
    assert all(row["warnings"] for row in result["rows"])
    assert not (prepared/"data").exists()


@pytest.mark.parametrize("case",["zero","nonfinite","cpu"])
def test_invalid_or_overbudget_pilot_never_starts_production(prepared,monkeypatch,case):
    from recurrent_rg.data import phi4_hybrid as kernel
    def sample(*args,**kwargs):
        row = fake_chain(*args,**kwargs)
        row[{"zero":"production_acceptance","nonfinite":"max_absolute_delta_h","cpu":"cpu_seconds"}[case]] = (
            float("nan") if case == "nonfinite" else 0 if case == "zero" else 100.)
        return row
    monkeypatch.setattr(kernel,"sample_hybrid_chain",sample)
    result = runner.sampler_preflight(prepared)
    assert result["status"] == ("budget_hold" if case == "cpu" else "invalid_sampling_hold")
    assert not result["physical_production_started"] and not (prepared/"data").exists()


def fake_data(root,plan,*,zero=False):
    data = root/"data"
    data.mkdir()
    runner.save(data/"plan.json",plan["physical"])
    shards = []
    for cell in range(13):
        folder = data/"conditions"/f"L36_c{cell:04d}"
        folder.mkdir(parents=True)
        path = folder/"sidecar.json"
        runner.save(path,dict(chains=[dict(chain_index=i,production_acceptance=0. if zero else .3,
            max_absolute_delta_h=.5,warnings=["low ESS; retained"]) for i in range(4)]))
        shards.append(dict(cell_index=cell,sidecar=str(path.relative_to(data)),sidecar_sha256=runner.sha(path)))
    runner.save(data/"manifest.json",dict(status="complete",shards=shards,warning_count=52))
    runner.save(data/"status.json",dict(status="complete",workflow_complete=True,returncode=0,cpu_seconds=1))
    runner.save(data/"quality_summary.json",dict(warning_count=52))


def install_fake_execution(root,monkeypatch,*,zero=False,pilot="complete",slow_profile=False,fail=None):
    monkeypatch.setattr(runner,"admission",lambda *args:None)
    monkeypatch.setattr(runner.os,"getpriority",lambda *args:0)
    nice = Mock()
    monkeypatch.setattr(runner.os,"nice",nice)
    plan = runner.read(root/"plan.json")
    calls = []
    def popen(cmd,**kwargs):
        module = cmd[cmd.index("-m")+1]
        output = Path(cmd[cmd.index("--output")+1])
        if module == runner.MODULE:
            name = "sampler_preflight"
            runner.save(root/"sampler_preflight.json",dict(status=pilot))
        elif module == "scripts.prepare_phi4_hybrid":
            name = "physical_sampling"
            fake_data(root,plan,zero=zero)
        elif module == "experiments.phi4_qcr":
            profile = cmd[cmd.index("--steps")+1] == "20"
            name = "training_preflight" if profile else "training"
            steps = 20 if profile else 3000
            output.mkdir()
            runner.save(output/"summary.json",dict(actual_steps=steps,device={"uuid":runner.GPU_UUID},
                optimizer_steps={key:dict(min=steps,max=steps) for key in ("Q","C","R")},
                peak_cuda_bytes=1024,labels_used_in_training=False,frozen_evaluation_verified=True,
                timing=dict(initialization_seconds=10,training_seconds=20,evaluation_seconds=1,wall_seconds=31)))
            runner.save(output/"history_progress.json",dict(history=[dict(step=1,elapsed_seconds=1.),
                dict(step=20,elapsed_seconds=191. if slow_profile else 2.)]))
            (output/"training_step_20.pt").touch()
            (output/"evaluation_arrays.npz").touch()
        else:
            name = "report"
            output.mkdir()
            runner.save(output/"status.json",dict(status="complete",workflow_complete=True))
        calls.append((name,cmd,kwargs))
        return Mock(pid=100+len(calls),returncode=int(name==fail),wait=lambda:int(name==fail))
    monkeypatch.setattr(runner.subprocess,"Popen",popen)
    monkeypatch.setattr(runner,"_wait_child",lambda child,*args,**kwargs:(child.wait(),False))
    return calls,nice


def test_complete_exploratory_workflow_keeps_warnings_and_excludes_private_files(prepared,monkeypatch):
    calls,nice = install_fake_execution(prepared,monkeypatch)
    runner.save(prepared/"launch.json",dict(parent_pid=123,code_bundle_sha256="fixture"))
    job = runner.execute(prepared)
    assert job["status"] == "complete" and job["workflow_complete"]
    assert job["returncode"] == 0
    assert "job.json" in job["public_files"] and "status.json" in job["public_files"]
    assert [row[0] for row in calls] == ["sampler_preflight","physical_sampling","training_preflight","training","report"]
    nice.assert_called_once_with(10)
    for _,_,kwargs in calls:
        assert all(kwargs["env"][key] == "1" for key in runner.THREAD_VARIABLES)
    validity = runner.read(prepared/"sampling_validity.json")
    assert validity["warnings"] and validity["quality_warnings_retained"]
    public = runner.read(prepared/"public_files.json")
    assert "launch.json" in public
    assert "run/evaluation_arrays.npz" in public
    assert not any("profile_run" in name or name.endswith(".pt") or name.endswith("data.npz") for name in public)
    assert runner.read(prepared/"status.json")["workflow_complete"]
    with pytest.raises(ValueError,match="already executed"):
        runner.execute(prepared)


@pytest.mark.parametrize("case,expected,names",[
    ("zero","invalid_sampling_hold",["sampler_preflight","physical_sampling"]),
    ("pilot","invalid_sampling_hold",["sampler_preflight"]),
    ("profile","budget_hold",["sampler_preflight","physical_sampling","training_preflight"]),
    ("fail","failed",["sampler_preflight","physical_sampling","training_preflight"]),
])
def test_hard_stops_preserve_evidence_without_retry(prepared,monkeypatch,case,expected,names):
    calls,_ = install_fake_execution(prepared,monkeypatch,zero=case=="zero",
        pilot="invalid_sampling_hold" if case=="pilot" else "complete",slow_profile=case=="profile",
        fail="training_preflight" if case=="fail" else None)
    job = runner.execute(prepared)
    assert job["status"] == expected and not job["workflow_complete"]
    assert job["requires_user_action"] and [row[0] for row in calls] == names
    assert (prepared/"execution.json").exists()
    assert not (prepared/"run").exists()


def test_profile_projection_counts_initialization_once_and_profile_cumulatively():
    summary = dict(actual_steps=20,timing=dict(initialization_seconds=100.,training_seconds=20.,
        evaluation_seconds=5.,wall_seconds=125.))
    history = dict(history=[dict(step=1,elapsed_seconds=1.),dict(step=20,elapsed_seconds=20.)])
    receipt = runner.profile_projection(summary,history,130.,130.)
    assert receipt["profile_initialization_seconds"] == 100.
    assert receipt["profile_steady_seconds_per_update"] == 1.
    assert receipt["projected_total_gpu_seconds_conservative"] == 130+1.5*(100+5+1+2999+5)+300+10
    assert receipt["remaining_gpu_admission_reserve_seconds"] == 10
    assert receipt["gpu_budget_seconds"] == 3600 and not receipt["budget_passed"]


def test_gpu_admission_and_both_full_stage_walls_are_cumulative(prepared,monkeypatch):
    calls,_ = install_fake_execution(prepared,monkeypatch)
    clock = [100.]
    monkeypatch.setattr(runner.time,"monotonic",lambda:clock[0])
    def admit(root,plan,job,stage,gpu):
        clock[0] += 10. if gpu else 0.
    def wait(child,*args,**kwargs):
        clock[0] += 32. if kwargs["gpu"] else 1.
        return child.wait(),False
    monkeypatch.setattr(runner,"admission",admit)
    monkeypatch.setattr(runner,"_wait_child",wait)
    job = runner.execute(prepared)
    assert job["status"] == "complete"
    assert job["gpu_stage_wall_seconds"] == 84.
    assert job["gpu_admission_wall_seconds"] == 20.
    profile,formal = calls[2][1],calls[3][1]
    assert float(profile[profile.index("--max-training-seconds")+1]) == 3590.
    assert float(formal[formal.index("--max-training-seconds")+1]) == 3548.


def test_failed_gpu_admission_wait_is_still_charged(prepared,monkeypatch):
    calls,_ = install_fake_execution(prepared,monkeypatch)
    clock = [100.]
    monkeypatch.setattr(runner.time,"monotonic",lambda:clock[0])
    def admit(root,plan,job,stage,gpu):
        if gpu:
            clock[0] += 3600.
            raise runner.BudgetHold("admission timeout")
    monkeypatch.setattr(runner,"admission",admit)
    job = runner.execute(prepared)
    assert job["status"] == "budget_hold" and job["returncode"] == 3
    assert job["gpu_stage_wall_seconds"] == 3600.
    assert [row[0] for row in calls] == ["sampler_preflight","physical_sampling"]


def test_monitor_terminates_only_its_own_overbudget_child(prepared,monkeypatch):
    child = Mock(pid=123,returncode=-15)
    child.poll.return_value = None
    monkeypatch.setattr(runner,"process_snapshot",lambda pid:dict(pid=pid,cpu_seconds=7201.))
    monkeypatch.setattr(runner,"cpu_seconds",lambda:0.)
    code,stopped = runner._wait_child(child,prepared,{},cpu_start=0.,stage_start=0.,gpu_elapsed=0.,gpu=False)
    assert code == -15 and stopped
    child.terminate.assert_called_once_with()
    child.kill.assert_not_called()


def test_smoke_is_plan_only_and_does_not_create_output(tmp_path):
    root = tmp_path/"smoke_only"
    plan = runner.main(["--output",str(root),"--smoke"])
    assert plan["training"]["steps"] == 3000 and not root.exists()
