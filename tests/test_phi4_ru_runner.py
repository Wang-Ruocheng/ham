"""Independent r-u orchestration contracts; CPU mocks only, no scientific runs."""
import copy
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts import run_phi4_ru_qcr as runner


CODE = Path(__file__).resolve().parents[1]


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    root = tmp_path/"fresh_ru"
    runner.prepare(root,code_root=CODE)
    monkeypatch.setattr(runner,"unchanged",lambda plan:None)
    monkeypatch.setattr(runner.socket,"gethostname",lambda:runner.HOST)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES",raising=False)
    return root


def test_pure_plan_45_condition_authorized_counts(tmp_path):
    root = tmp_path/"not_created"
    plan = runner.build_plan(root,CODE)
    assert not root.exists()
    p,t = plan["physical"],plan["training"]
    assert p["conditions"] == 45 and p["configurations"] == 23040
    assert p["options"]["r_values"] == pytest.approx([-1.55+1.05*i/8 for i in range(9)])
    assert p["options"]["u_values"] == [.5,.625,.75,.875,1.]
    assert p["total_hmc_trajectories"] == 368640
    assert p["total_force_site_evaluations"] == 6210846720
    assert len(p["train_cell_indices"]) == 34 and len(p["test_cell_indices"]) == 11
    assert p["test_cell_indices"] == list(range(1,45,4))
    assert plan["sampler_preflight"]["cell_indices"] == [0,4,40,44]
    assert plan["total_planned_force_site_evaluations_including_preflight"] == 6243194880
    assert plan["total_planned_force_site_evaluations_including_preflight"] <= runner.FORCE_SITE_CAP
    assert t["steps"] == 3000 and t["theory_dim"] == 2 and t["profile_steps"] == 20
    assert t["lattice_sizes"] == [36,12,4] and not t["labels_used"]
    assert t["closure_scale_weights"] is None and t["independent_gradients"]
    assert plan["resource_policy"]["minimum_host_available_gib"] == 36
    assert plan["resource_policy"]["gpu_compute_processes"] == 0
    assert all(plan["sampler_source_pins"][name]==digest for name,digest in p["source_pins"].items())
    assert "recurrent_rg/data/phi4_ru_pipeline.py" in plan["sampler_source_pins"]
    assert plan["evaluation"]["thermal_nu"] is None


@pytest.mark.parametrize("section,key,value",[
    ("data","r_values",[-1.5]),("data","u_values",[.5]),("data","seed",91),
    ("data","test_offset",0),("training","steps",2999),
    ("training","q_initialization","random_legacy"),("model","block_size",2),
    ("model","d_g",1),("loss","closure_scale_weights",[1.8,.2]),
    ("loss","coupling_supervision",True),("loss","prediction_prediction_loss",True),
])
def test_science_changes_rejected(tmp_path,section,key,value):
    config = runner.default_config()
    config[section][key] = value
    with pytest.raises(ValueError,match="contract"):
        runner.build_plan(tmp_path/"unused",CODE,config)
    assert not (tmp_path/"unused").exists()


def test_configuration_and_plan_frozen_no_overwrite(prepared):
    with pytest.raises(FileExistsError):
        runner.prepare(prepared)
    plan,_ = runner.load_plan(prepared)
    changed = copy.deepcopy(plan)
    changed["training"]["steps"] = 1
    runner.save(prepared/"plan.json",changed)
    with pytest.raises(RuntimeError,match="Frozen"):
        runner.load_plan(prepared)


def test_stage_cli_contract_and_profile_discard(prepared):
    plan = runner.read(prepared/"plan.json")
    stages = runner.planned_commands(prepared,plan)
    assert [n for n,_,_ in stages] == ["sampler_preflight","physical_sampling","training_preflight","training","report"]
    assert [n for n,_,g in stages if g] == ["training_preflight","training"]
    assert "scripts.prepare_phi4_ru" in stages[1][1]
    assert "scripts.report_phi4_ru_qcr" in stages[4][1]
    for i,seed in ((2,"191"),(3,"91")):
        command = stages[i][1]
        assert command[command.index("--seed")+1] == seed
        assert command[command.index("--expected-gpu-uuid")+1] == runner.GPU_UUID
        assert command[command.index("--q-initialization")+1] == "train_mean_bias"
        assert command[command.index("--device")+1] == "cuda:2"
        assert not {"--source","--resume","--labels","--reference"}.intersection(command)
    assert "--profile-only" in stages[2][1] and "--profile-only" not in stages[3][1]


def test_sampling_cli_exactly_freezes_options(prepared,monkeypatch):
    from scripts import prepare_phi4_ru as cli
    from recurrent_rg.data.phi4_ru_pipeline import generation_plan
    seen=[]
    monkeypatch.setattr(cli,"generate_dataset",lambda output,options:seen.append(options) or dict(schema="mock",shards=[],warning_count=0))
    plan=runner.read(prepared/"plan.json")
    command=runner.sampling_command(prepared,plan)
    cli.main(command[command.index("scripts.prepare_phi4_ru")+1:])
    import json
    assert json.loads(json.dumps(generation_plan(seen[0]))) == plan["physical"]


def test_two_gpu_idle_checks_and_ten_second_gap(prepared,monkeypatch):
    calls,sleeps=[],[]
    monkeypatch.setattr(runner,"resource_snapshot",lambda plan:calls.append(1) or dict(gpu_uuid=runner.GPU_UUID,gpu_ready=True,host_ready=True))
    monkeypatch.setattr(runner.time,"sleep",sleeps.append)
    plan,job=runner.load_plan(prepared)
    runner.admission(prepared,plan,job,"training",True)
    assert len(calls)==2 and sleeps==[10]


def test_live_gpu_even_low_utilization_not_admitted(prepared,monkeypatch):
    original=Path.read_text
    state=dict(gib=36,load=30.,io=19.99,free=12288,pids="")
    def read_text(path,*args,**kwargs):
        if str(path)=="/proc/meminfo": return f"MemAvailable: {int(state['gib']*1024**2)} kB\n"
        if str(path)=="/proc/pressure/io": return f"full avg10={state['io']} avg60=0\n"
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,"read_text",read_text)
    monkeypatch.setattr(runner.os,"cpu_count",lambda:24)
    monkeypatch.setattr(runner.os,"getloadavg",lambda:(state["load"],0,0))
    monkeypatch.setattr(runner.subprocess,"check_output",lambda cmd,**kw:
        f"{runner.GPU_UUID}, {state['free']}\n" if "-i" in cmd else state["pids"])
    plan=runner.read(prepared/"plan.json")
    assert runner.resource_snapshot(plan)["gpu_ready"]
    for key,bad in (("gib",35.9),("load",30.01),("io",20.),("free",12287),("pids",f"{runner.GPU_UUID}, 42")):
        previous=state[key]
        state[key]=bad
        assert not runner.resource_snapshot(plan)["gpu_ready"]
        state[key]=previous


def fake_chain(*args,**kwargs):
    assert args[0] in (-1.55,-.5) and args[1] in (.5,1.)
    assert kwargs["samples"]==8 and kwargs["burn_in"]==128 and kwargs["spacing"]==4
    return dict(cpu_seconds=.3,wall_seconds=.3,hmc_trajectories=160,
        force_site_evaluations=160*13*36**2,production_acceptance=.9,max_absolute_delta_h=.5,
        warnings=["finite chain warning"])


def test_pilots_use_only_train_corners_keep_warnings(prepared,monkeypatch):
    from recurrent_rg.data import phi4_hybrid as kernel
    seen=[]
    def sample(*args,**kwargs):
        seen.append((args,kwargs))
        return fake_chain(*args,**kwargs)
    monkeypatch.setattr(kernel,"sample_hybrid_chain",sample)
    result=runner.sampler_preflight(prepared)
    assert result["status"]=="complete" and len(seen)==12
    assert {(a[0],a[1]) for a,_ in seen}=={(-1.55,.5),(-1.55,1.),(-.5,.5),(-.5,1.)}
    assert all(row["warnings"] for row in result["rows"])
    assert not (prepared/"data").exists()


@pytest.mark.parametrize("case",["zero","nonfinite","cpu"])
def test_bad_or_overbudget_pilot_prevents_production(prepared,monkeypatch,case):
    from recurrent_rg.data import phi4_hybrid as kernel
    def sample(*args,**kwargs):
        row=fake_chain(*args,**kwargs)
        row[{"zero":"production_acceptance","nonfinite":"max_absolute_delta_h","cpu":"cpu_seconds"}[case]]=(
            float("nan") if case=="nonfinite" else 0. if case=="zero" else 100.)
        return row
    monkeypatch.setattr(kernel,"sample_hybrid_chain",sample)
    result=runner.sampler_preflight(prepared)
    assert result["status"]==("budget_hold" if case=="cpu" else "invalid_sampling_hold")
    assert not result["physical_production_started"] and not (prepared/"data").exists()


def test_gpu_projection_one_time_init_and_explicit_final_evaluation():
    summary=dict(actual_steps=20,timing=dict(training_seconds=3.,evaluation_seconds=0.,wall_seconds=13.))
    history=dict(history=[dict(step=1,elapsed_seconds=1.),dict(step=20,elapsed_seconds=3.)])
    projection=runner.profile_projection(summary,history,14.,15.)
    assert projection["profile_initialization_seconds"]==10.
    assert projection["profile_process_startup_seconds"]==1.
    assert projection["profile_steady_seconds_per_update"]==pytest.approx(2/19)
    assert projection["formal_final_evaluation_reserve_seconds"]==180.
    assert projection["final_evaluation_ensembles"]==180
    assert projection["projected_formal_gpu_seconds_conservative"]==pytest.approx(1.5*(10+1+1+2999*2/19+180)+300)
    assert projection["budget_passed"]
    history["history"][-1]["elapsed_seconds"]=100.
    assert not runner.profile_projection(summary,history,14.,15.)["budget_passed"]


def fake_data(root,plan,*,zero=False):
    data=root/"data"
    data.mkdir()
    runner.save(data/"plan.json",plan["physical"])
    runner.save(data/"wrapper_receipt.json",dict(workflow_complete=True,returncode=0,
        source_pins_before=plan["sampler_source_pins"],source_pins_after=plan["sampler_source_pins"]))
    shards=[]
    for cell in range(45):
        folder=data/"conditions"/f"L36_c{cell:04d}"
        folder.mkdir(parents=True)
        path=folder/"sidecar.json"
        runner.save(path,dict(chains=[dict(chain_index=i,production_acceptance=0. if zero else .9,
            max_absolute_delta_h=.5,warnings=["low ESS retained"]) for i in range(4)]))
        (folder/"data.npz").touch()
        shards.append(dict(cell_index=cell,sidecar=str(path.relative_to(data)),sidecar_sha256=runner.sha(path)))
    runner.save(data/"manifest.json",dict(status="complete",shards=shards,warning_count=180))
    runner.save(data/"status.json",dict(status="complete",workflow_complete=True,returncode=0))
    runner.save(data/"quality_summary.json",dict(warning_count=180))


def fake_execution(root,monkeypatch,*,zero=False,pilot="complete",slow=False,fail=None,formal_projection_hold=False):
    plan=runner.read(root/"plan.json")
    def admission(*args):
        if formal_projection_hold and args[3]=="training":
            gate=runner.read(root/"gpu_preflight.json")
            gate["projected_formal_gpu_seconds_conservative"]=4000.
            runner.save(root/"gpu_preflight.json",gate)
    monkeypatch.setattr(runner,"admission",admission)
    monkeypatch.setattr(runner.os,"getpriority",lambda *args:0)
    nice=Mock()
    monkeypatch.setattr(runner.os,"nice",nice)
    calls=[]
    def popen(command,**kwargs):
        module=command[command.index("-m")+1]
        output=Path(command[command.index("--output")+1])
        if module==runner.MODULE:
            name="sampler_preflight"
            runner.save(root/"sampler_preflight.json",dict(status=pilot))
        elif module=="scripts.prepare_phi4_ru":
            name="physical_sampling"
            fake_data(root,plan,zero=zero)
        elif module=="experiments.phi4_ru_qcr":
            profile="--profile-only" in command
            name="training_preflight" if profile else "training"
            steps=20 if profile else 3000
            output.mkdir()
            runner.save(output/"summary.json",dict(actual_steps=steps,device={"uuid":runner.GPU_UUID},
                optimizer_steps={k:dict(min=steps,max=steps) for k in ("Q","C","R")},peak_cuda_bytes=1024,
                profile_only=profile,labels_used_in_training=False,frozen_evaluation_verified=not profile,
                timing=dict(training_seconds=2.,evaluation_seconds=0. if profile else 3.,wall_seconds=12.)))
            runner.save(output/"history_progress.json",dict(history=[dict(step=1,elapsed_seconds=1.),
                dict(step=20,elapsed_seconds=191. if slow else 2.)]))
            if not profile:
                (output/"training_step_3000.pt").touch()
                (output/"evaluation_arrays.npz").touch()
        else:
            name="report"
            output.mkdir()
            runner.save(output/"status.json",dict(status="complete",workflow_complete=True,returncode=0))
        calls.append((name,command,kwargs))
        return Mock(pid=100+len(calls),returncode=int(name==fail),wait=lambda:int(name==fail))
    monkeypatch.setattr(runner.subprocess,"Popen",popen)
    monkeypatch.setattr(runner,"_wait_child",lambda child,*args,**kw:(child.wait(),False))
    return calls,nice


def test_mocked_full_execution_retains_warnings_private_files_excluded(prepared,monkeypatch):
    calls,nice=fake_execution(prepared,monkeypatch)
    result=runner.execute(prepared)
    assert result["status"]=="complete" and result["workflow_complete"]
    assert [n for n,_,_ in calls]==["sampler_preflight","physical_sampling","training_preflight","training","report"]
    assert all(row["returncode"]==0 for row in result["stages"])
    assert runner.read(prepared/"sampling_validity.json")["warning_count"]==180
    assert "data/wrapper_receipt.json" in result["public_files"]
    assert "run/evaluation_arrays.npz" in result["public_files"]
    assert not any(name.endswith(".pt") or (name.startswith("data/") and name.endswith(".npz")) for name in result["public_files"])
    inventory=runner.read(prepared/"public_files.json")
    assert all(runner.sha(prepared/name)==digest for name,digest in inventory.items())
    nice.assert_called_once_with(10)
    with pytest.raises(ValueError,match="already executed"):
        runner.execute(prepared)


@pytest.mark.parametrize("case,expected_calls,status",[
    ("pilot",1,"budget_hold"),("zero",2,"invalid_sampling_hold"),
    ("slow",3,"budget_hold"),("formal_admission",3,"budget_hold"),("fail",4,"failed")])
def test_holds_and_failures_do_not_proceed_retry_or_overwrite(prepared,monkeypatch,case,expected_calls,status):
    calls,_=fake_execution(prepared,monkeypatch,zero=case=="zero",pilot="budget_hold" if case=="pilot" else "complete",
        slow=case=="slow",formal_projection_hold=case=="formal_admission",fail="training" if case=="fail" else None)
    result=runner.execute(prepared)
    assert result["status"]==status and not result["workflow_complete"]
    assert result["requires_user_action"] and len(calls)==expected_calls
    assert not (prepared/"report").exists()


def test_profile_summary_does_not_claim_frozen_evaluation(prepared):
    plan=runner.read(prepared/"plan.json")
    summary=dict(actual_steps=20,device={"uuid":runner.GPU_UUID},profile_only=True,
        frozen_evaluation_verified=False,labels_used_in_training=False,peak_cuda_bytes=1024,
        optimizer_steps={k:dict(min=20,max=20) for k in ("Q","C","R")})
    runner.validate_training_summary(summary,plan,profile=True)
    with pytest.raises(RuntimeError):
        runner.validate_training_summary(summary,plan,profile=False)


def test_physical_provenance_checks_wrapper_and_unique_conditions(prepared):
    plan=runner.read(prepared/"plan.json")
    fake_data(prepared,plan)
    assert runner.sampling_validity(prepared,plan)["status"]=="complete"
    wrapper=runner.read(prepared/"data/wrapper_receipt.json")
    wrapper["source_pins_after"]={}
    runner.save(prepared/"data/wrapper_receipt.json",wrapper)
    with pytest.raises(RuntimeError,match="source"):
        runner.sampling_validity(prepared,plan)
