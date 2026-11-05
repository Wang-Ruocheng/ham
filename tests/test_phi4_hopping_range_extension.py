"""New range workflow: mock-only execution; no real sampling/training/CUDA."""
import copy
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_phi4_hopping_range_extension as run

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/run.DEFAULT_CONFIG
SAMPLING = ROOT/run.DEFAULT_SAMPLING_PLAN
UUID = "GPU-a167decd-ce49-6c8f-6bcf-136fde596683"


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    run.base.save(path,value)


def test_fifteen_condition_work_counts_and_preserved_membership():
    from recurrent_rg.data.phi4_hopping_refined_pipeline import Phi4HoppingRefinedOptions,generation_plan,chain_seed
    new,old = run.physical_options(SAMPLING),Phi4HoppingRefinedOptions()
    assert new.kappa_values == (.32,*old.kappa_values,1.04)
    assert new.kappa_values[7] == .6799999999999999
    assert new.test_offset == 1
    for cell,kappa in enumerate(old.kappa_values):
        assert (cell%3 == 0) == ((cell+1)%3 == 1)
        for chain in range(4):
            assert chain_seed(old,36,cell,chain) == chain_seed(new,36,cell+1,chain)
    definition = generation_plan(new)
    assert definition["conditions"] == 15 and definition["configurations"] == 7680
    assert definition["total_hybrid_cycles"] == 122880
    assert definition["total_force_site_evaluations"] == 3981312000
    assert definition["train_cell_indices"] == [0,2,3,5,6,8,9,11,12,14]
    assert definition["test_cell_indices"] == [1,4,7,10,13]
    assert new.max_force_site_evaluations == 4_000_000_000


def test_only_endpoint_pilot_and_purpose_separation():
    from recurrent_rg.data.phi4_hopping_refined_pipeline import chain_seed
    options,cases = run.physical_options(SAMPLING),run.pilot_cases(SAMPLING)
    production = {chain_seed(options,36,cell,chain) for cell in range(15) for chain in range(4)}
    assert len(cases) == 6 and len({row["seed"] for row in cases}) == 6
    assert {row["kappa"] for row in cases} == {.32,1.04}
    assert all(row["seed"] not in production for row in cases)
    assert {row["initial"] for row in cases} == {"hot","cold_positive","cold_negative"}
    plan = run.pilot_plan(SAMPLING)
    assert plan["total_hybrid_cycles"] == 960
    assert plan["total_force_site_evaluations"] == 31104000
    assert plan["cuda_used"] is plan["equilibrium_certified"] is plan["arrays_reused_in_production"] is False


def test_dry_no_compute_writes_or_historical_dependency(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(run,"run",lambda *_:pytest.fail("dry run executed"))
    output = tmp_path/"new"
    assert run.main(["--output",str(output),"--gpu-index","3","--gpu-uuid",UUID,
        "--config",str(CONFIG),"--sampling-plan",str(SAMPLING)]) == 0
    value = json.loads(capsys.readouterr().out)
    assert not output.exists() and value["execution_requested"] is False
    assert value["stages"] == ["sampler_pilot","sampling","gpu_preflight","training"]
    assert value["resources"]["cpu_seconds"] == 21600
    assert value["resources"]["gpu_card_seconds"] == 14400
    assert value["work_accounting"]["total_including_endpoint_pilot_force_site_evaluations"] == 4012416000
    assert value["work_accounting"]["four_billion_is_production_dataset_cap_not_entire_workflow_cap"]
    assert value["historical_data_read"] is value["historical_benchmark_reused"] is False
    assert "failed_root" not in value
    assert value["training"]["options"]["steps"] == 3000
    assert value["training"]["options"]["system_batch"] == 32
    assert value["training"]["options"]["sample_batch"] == 64
    command = value["commands"]["training"]
    assert command[2:5] == ["-m","recurrent_rg","train"]
    assert "--source" not in command and "--resume" not in command
    assert "--allow-legacy-training" not in str(value)
    sampling = value["commands"]["sampling"]
    assert sampling[3] == "scripts.prepare_phi4_hopping_refined"
    assert sampling[sampling.index("--test-offset")+1] == "1"
    assert sampling[sampling.index("--step-size")+1] == "0.04"
    assert sampling[sampling.index("--leapfrog-steps")+1] == "24"


@pytest.mark.parametrize("key,value",[("kappa_values",[.32,.50,1.04]),("test_offset",0),
    ("step_size",.08),("samples",64),("max_force_site_evaluations",5_000_000_000)])
def test_changed_sampling_recipe_rejected(tmp_path,key,value):
    altered = json.loads(SAMPLING.read_text()); altered["options"][key] = value
    path = tmp_path/"plan.json";write(path,altered)
    with pytest.raises(ValueError,match="fifteen"):
        run.physical_options(path)


@pytest.mark.parametrize("key",["steps","system_batch","sample_batch","seed","threads",
    "log_every","checkpoint_every","information_pairs","max_training_seconds","max_cuda_bytes"])
def test_changed_training_recipe_rejected(tmp_path,key):
    altered = json.loads(CONFIG.read_text()); altered["training"][key] += 1
    path = tmp_path/"training.json";write(path,altered)
    with pytest.raises(ValueError,match="unchanged"):
        run.plan(tmp_path/"new",path,3,UUID,SAMPLING)


def test_fresh_output_and_explicit_gpu_required(tmp_path):
    with pytest.raises(ValueError,match="fresh output"):
        run.plan(tmp_path,CONFIG,3,UUID,SAMPLING)
    for index,uuid in ((None,UUID),(3,None),(True,UUID),(3,"3090")):
        with pytest.raises(ValueError,match="explicit physical"):
            run.plan(tmp_path/"out",CONFIG,index,uuid,SAMPLING)


def mock_chain(acceptance=1.,finite=True):
    from recurrent_rg.data.phi4_hopping_refined_pipeline import WORK_FIELDS
    return dict(configurations=np.full((32,2,2),0. if finite else np.nan),
        raw_observables=np.zeros((32,8)),delta_h=np.zeros(160),trajectory_magnetization=np.zeros(160),
        accepted=np.ones(160,dtype=bool),production_acceptance=acceptance,burn_in_acceptance=1.,
        max_absolute_delta_h=0.,diagnostics={},warnings=["short pilot warning"],
        **{name:(160 if name in ("hybrid_cycles","hmc_trajectories") else
                 160*25*36**2 if name=="force_site_evaluations" else 0) for name in WORK_FIELDS})


def test_endpoint_pilot_mock_finite_acceptance_and_receipt(tmp_path,monkeypatch):
    from recurrent_rg.data import phi4_hopping
    calls = []
    monkeypatch.setattr(phi4_hopping,"sample_chain",lambda *a,**kw:(calls.append((a,kw)) or mock_chain()))
    path = tmp_path/"sampler_pilot.json"
    assert run.pilot(path,SAMPLING) == 0
    receipt = json.loads(path.read_text())
    assert len(calls) == 6 and len(receipt["rows"]) == 6
    assert all(kw["step_size"] == .04 and kw["leapfrog_steps"] == 24 for _,kw in calls)
    assert all(kw["burn_in"] == 128 and kw["samples"] == 32 and kw["spacing"] == 1 for _,kw in calls)
    assert all("configurations" not in row and row["warnings"] == ["short pilot warning"] for row in receipt["rows"])
    verified = run.validate_pilot(tmp_path,SAMPLING)
    assert verified["pilot_min_production_acceptance"] == 1.
    assert verified["remaining_production_cpu_projection_seconds"] == pytest.approx(
        2*max(row["cpu_seconds"] for row in receipt["rows"])*122880/160)
    with pytest.raises(FileExistsError):run.pilot(path,SAMPLING)


@pytest.mark.parametrize("acceptance,finite",[(.59,True),(1.,False),(0.,True),(float("nan"),True)])
def test_pilot_gate_stops_without_retry(tmp_path,monkeypatch,acceptance,finite):
    from recurrent_rg.data import phi4_hopping
    calls=[]
    monkeypatch.setattr(phi4_hopping,"sample_chain",lambda *a,**kw:(calls.append(kw) or mock_chain(acceptance,finite)))
    path=tmp_path/"sampler_pilot.json"
    assert run.pilot(path,SAMPLING) == 4 and len(calls) == 1
    receipt=json.loads(path.read_text())
    assert receipt["status"] == "scientific_stop" and receipt["workflow_complete"] is True
    assert receipt["requires_user_action"] and not (tmp_path/"data").exists()
    with pytest.raises(RuntimeError):run.validate_pilot(tmp_path,SAMPLING)


def test_pilot_numerical_overflow_is_endpoint_stop_not_training_failure(tmp_path,monkeypatch):
    from recurrent_rg.data import phi4_hopping
    def overflow(*args,**kwargs):raise FloatingPointError("overflow encountered in power")
    monkeypatch.setattr(phi4_hopping,"sample_chain",overflow)
    path=tmp_path/"sampler_pilot.json"
    assert run.pilot(path,SAMPLING)==4
    receipt=json.loads(path.read_text())
    assert receipt["status"]=="scientific_stop" and receipt["numerical_instability"]
    assert receipt["rows"][0]["finite"] is False
    assert receipt["rows"][0]["error"].startswith("FloatingPointError")


def test_initialization_not_extrapolated_per_step():
    times=[1.,1.1,1.,1.2,.9]
    assert run.benchmark_projection(10.,times) == 2*10+1.5*1.2*3000+500
    assert run.benchmark_projection(20.,times)-run.benchmark_projection(10.,times) == 20.
    for initialization,updates in ((float("nan"),times),(1.,times[:4]),(1.,[0.]*5),(1.,[float("inf")]*5)):
        with pytest.raises(ValueError):run.benchmark_projection(initialization,updates)


@pytest.mark.parametrize("cpu,gpu,size,cp,gp,expected",[
    (0,0,0,0,0,True),(21600,14400,1024**3,0,0,True),
    (21600.1,0,0,0,0,False),(0,14400.1,0,0,0,False),
    (0,0,1024**3+1,0,0,False),(21000,100,0,400,300,False),
    (100,14000,0,0,500,False),(0,0,0,-1,0,False),
])
def test_total_resources_account_actual_plus_projection(cpu,gpu,size,cp,gp,expected):
    assert run.budget_ok(cpu,gpu,size,cpu_projection=cp,gpu_projection=gp) is expected


def fake_benchmark(root,timed=1.):
    receipt=dict(status="complete",workflow_complete=True,returncode=0,error=None,
        requires_user_action=False,synthetic=True,physical_result=False,physical_data_read=False,
        checkpoint_saved=False,formal_model_reused=False,actual_updates=7,warmup_updates=2,timed_updates=5,
        training_batch=[32,64,36,36,1],device={"CUDA_VISIBLE_DEVICES":UUID},peak_cuda_bytes=100,
        training_cap_seconds=run.GPU_SECONDS,evaluation_report_reserve_seconds=500.,
        initialization_seconds=1.,timed_update_seconds=[timed]*5,
        conservative_full_training_projection_seconds=run.benchmark_projection(1.,[timed]*5),
        optimizer_steps={name:dict(min=7,max=7,parameters=1) for name in ("Q","C","R")})
    write(root/"gpu_preflight.json",receipt)
    return receipt


def test_benchmark_binding_enforces_same_batch_all_clocks_and_uuid(tmp_path):
    good=fake_benchmark(tmp_path)
    assert run.validate_benchmark(tmp_path,UUID)==5002.
    for key,value in (("actual_updates",6),("training_batch",[16,64,36,36,1]),
        ("physical_data_read",True),("evaluation_report_reserve_seconds",300.),
        ("optimizer_steps",{"R":dict(min=7,max=7,parameters=1)})):
        bad=copy.deepcopy(good);bad[key]=value;write(tmp_path/"gpu_preflight.json",bad)
        with pytest.raises(RuntimeError):run.validate_benchmark(tmp_path,UUID)


def test_production_validation_checks_range_shape_and_endpoints(tmp_path,monkeypatch):
    from recurrent_rg.data import phi4_hopping_loader
    from recurrent_rg.data.phi4_hopping_refined_pipeline import generation_plan
    definition=json.loads(json.dumps(generation_plan(run.physical_options(SAMPLING))))
    write(tmp_path/"data/plan.json",definition)
    manifest=dict(options=definition["options"],source_pins=definition["source_pins"],
        total_hmc_trajectories=122880,total_force_site_evaluations=3981312000)
    write(tmp_path/"data/manifest.json",manifest)
    quality=dict(conditions=[dict(production_acceptance=[.8]*4) for _ in range(15)])
    write(tmp_path/"data/quality_summary.json",quality)
    dataset=SimpleNamespace(configurations=SimpleNamespace(shape=(60,128,36,36,1)),
        train_indices=list(range(40)),test_indices=list(range(20)),manifest_sha256="a"*64,warning_count=17)
    monkeypatch.setattr(phi4_hopping_loader,"load_dataset",lambda _:dataset)
    assert run.validate_sampling(tmp_path,SAMPLING)["data_warning_count"]==17
    manifest["options"]["kappa_values"][-1]=1.01;write(tmp_path/"data/manifest.json",manifest)
    with pytest.raises(RuntimeError,match="fifteen-condition"):run.validate_sampling(tmp_path,SAMPLING)


def fake_training(root):
    from experiments.phi4_hopping_qcr import RECIPE
    folder=root/"training";folder.mkdir()
    (folder/"evaluation_arrays.npz").write_bytes(b"mock scalar evaluation")
    summary=dict(recipe=RECIPE,actual_steps=3000,theory_dim=1,thermal_nu=None,
        data_manifest_sha256="a"*64,peak_cuda_bytes=100,
        optimizer_steps={name:dict(min=3000,max=3000,parameters=1) for name in ("Q","C","R")})
    write(folder/"summary.json",summary)
    write(folder/"manifest.json",dict(recipe=RECIPE,data_manifest_sha256="a"*64,
        closure_objective="all_start_anchor",closure_scale_weights=None))
    write(folder/"status.json",dict(status="complete",workflow_complete=True,returncode=0,error=None,
        requires_user_action=False,actual_steps=3000,latest_checkpoint_step=3000,
        summary_sha256=run.base.sha(folder/"summary.json"),evaluation_arrays_sha256=run.base.sha(folder/"evaluation_arrays.npz"),
        public_files=[dict(path=name,sha256=run.base.sha(folder/name))
            for name in ("summary.json","manifest.json","evaluation_arrays.npz")]))


@pytest.fixture
def fake_runtime(tmp_path,monkeypatch):
    root,calls,modes=tmp_path/"result",[],{}
    monkeypatch.setattr(run.socket,"gethostname",lambda:run.base.HOST)
    monkeypatch.setattr(run.os,"getpriority",lambda *_:10)
    monkeypatch.setattr(run,"_real_python_files",lambda _: {"runner":Path(run.__file__)})
    monkeypatch.setattr(run.base,"host_resources",lambda:dict(cpus=24,load1=8.,mem_available_bytes=120*1024**3,io_full_avg10=.1))
    monkeypatch.setattr(run.base,"gpu_resources",lambda *_:dict(index=3,uuid=UUID,free_bytes=24*1024**3,compute_pids=[]))
    ticks=iter(range(0,100000,10))
    monkeypatch.setattr(run.time,"monotonic",lambda:next(ticks))
    monkeypatch.setattr(run.time,"sleep",lambda _:None)
    monkeypatch.setattr(run,"validate_pilot",lambda *_:dict(remaining_production_cpu_projection_seconds=modes.get("cpu_projection",10.)))
    monkeypatch.setattr(run,"validate_sampling",lambda *_:dict(data_manifest_sha256="a"*64,data_warning_count=17))
    class Process:
        def __init__(self,command,**kwargs):
            self.pid,self.returncode=98765,0
            if "--pilot-output" in command:
                name="sampler_pilot"
                write(root/"sampler_pilot.json",dict(status="complete",rows=[dict(actual_work=dict(force_site_evaluations=31104000))]))
            elif "scripts.prepare_phi4_hopping_refined" in command:
                name="sampling"
                write(root/"data/manifest.json",dict(public_files=["manifest.json","status.json"],total_force_site_evaluations=3981312000))
                write(root/"data/status.json",dict(status="complete"))
            elif "--benchmark-output" in command:
                name="gpu_preflight";fake_benchmark(root,modes.get("timed",1.))
            else:
                name="training";fake_training(root)
            calls.append((name,command,kwargs))
            self.returncode=modes.get(name,0)
        def poll(self):return self.returncode
        def wait(self):return self.returncode
    monkeypatch.setattr(run.subprocess,"Popen",Process)
    return root,calls,modes


def test_parent_sequential_complete_public_frozen_success(fake_runtime):
    root,calls,modes=fake_runtime
    assert run.run(root,CONFIG,3,UUID,SAMPLING)==0
    assert [name for name,*_ in calls]==["sampler_pilot","sampling","gpu_preflight","training"]
    assert [kwargs["env"]["CUDA_VISIBLE_DEVICES"] for _,_,kwargs in calls]==["","",UUID,UUID]
    job=json.loads((root/"job.json").read_text())
    assert job["workflow_complete"] and job["status"]=="complete" and job["formal_optimizer_updates"]==3000
    assert job["gpu_idle_checks"]["spacing_seconds"]>=10
    assert job["actual_total_force_site_evaluations_including_endpoint_pilot"]==4012416000
    assert job["cpu_seconds"]>=0 and job["gpu_seconds"]>0 and job["data_warning_count"]==17
    public=json.loads((root/"public_files.json").read_text())
    assert all(run.base.sha(root/name)==digest for name,digest in public.items())
    assert not any(name.endswith(".pt") or name.endswith("data.npz") for name in public)


@pytest.mark.parametrize("stage,rc,status,expected",[
    ("sampler_pilot",4,"scientific_stop",["sampler_pilot"]),
    ("sampling",1,"failed",["sampler_pilot","sampling"]),
    ("gpu_preflight",3,"budget_hold",["sampler_pilot","sampling","gpu_preflight"]),
    ("training",1,"failed",["sampler_pilot","sampling","gpu_preflight","training"]),
])
def test_failure_no_retry_resume_or_later_stage(fake_runtime,stage,rc,status,expected):
    root,calls,modes=fake_runtime;modes[stage]=rc
    assert run.run(root,CONFIG,3,UUID,SAMPLING)==rc
    assert [name for name,*_ in calls]==expected
    job=json.loads((root/"job.json").read_text())
    assert job["status"]==status and job["requires_user_action"]
    assert job["workflow_complete"] is (stage=="sampler_pilot" and rc==4)
    if stage=="sampler_pilot":assert not (root/"data").exists() and job["formal_optimizer_updates"]==0


def test_cpu_projection_hold_before_production(fake_runtime):
    root,calls,modes=fake_runtime;modes["cpu_projection"]=21601.
    assert run.run(root,CONFIG,3,UUID,SAMPLING)==3
    assert [name for name,*_ in calls]==["sampler_pilot"]
    assert not (root/"data").exists() and not (root/"training").exists()


def test_cumulative_gpu_projection_hold_preserves_data(fake_runtime):
    root,calls,modes=fake_runtime;modes["timed"]=3.085
    # Projection passes alone; actual benchmark/check intervals exhaust the remainder.
    assert run.run(root,CONFIG,3,UUID,SAMPLING)==3
    assert [name for name,*_ in calls]==["sampler_pilot","sampling","gpu_preflight"]
    assert (root/"data").exists() and not (root/"training").exists()


def test_internal_modes_disallow_ambiguous_execution(tmp_path):
    with pytest.raises(SystemExit):run.main(["--pilot-output",str(tmp_path/"pilot"),"--execute"])
    with pytest.raises(SystemExit):run.main(["--benchmark-output",str(tmp_path/"benchmark")])
