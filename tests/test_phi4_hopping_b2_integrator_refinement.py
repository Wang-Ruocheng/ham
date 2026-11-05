"""No real sampling, CUDA or model construction in refinement contracts."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_phi4_hopping_b2_integrator_refinement as driver

REPO=Path(__file__).resolve().parents[1]
SAMPLING=REPO/driver.large.DEFAULT_SAMPLING_PLAN
MIRROR=REPO/"results/phi4_hopping_b2_L128_seed91_vm225_20261101_monitor"


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def evidence(tmp_path,monkeypatch):
    code,old_code,stopped,preflight=[tmp_path/p for p in ("code","old_code","stopped","preflight")]
    for path in (code,old_code,stopped,preflight):path.mkdir()
    pins={}
    for name in ("scripts/continue_phi4_hopping_b2_large_budget.py","recurrent_rg/phi4_aligned_qcr.py"):
        for folder in (code,old_code):
            path=folder/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text("# frozen\n")
        pins[str(old_code/name)]=driver.base.sha(old_code/name)
    for name in driver.NEW_SOURCES:
        path=code/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text("# refinement\n")
    config=tmp_path/"config.json"
    config.write_bytes((REPO/driver.previous.DEFAULT_CONFIG).read_bytes())
    write(preflight/"plan.json",dict(original=True))
    write(preflight/"job.json",dict(status="budget_hold",parent_pid=None,stages=[],cpu_seconds=96.086822453,gpu_seconds=115.847737390548))
    (preflight/"gpu_preflight.json").write_bytes((MIRROR/"gpu_preflight.json").read_bytes())
    (stopped/"reused_preflight.json").write_bytes((MIRROR/"gpu_preflight.json").read_bytes())
    (stopped/"sampler_pilot.json").write_bytes((MIRROR/"budget16h/sampler_pilot.json").read_bytes())
    old_plan=dict(host=driver.base.HOST,output=str(stopped),code_root=str(old_code),GPU=dict(index=3,uuid=driver.GPU_UUID),
        config_source_sha256=driver.base.sha(config),sampling_plan_sha256=driver.base.sha(SAMPLING),
        source_sha256=pins,budget_extension=dict(prior_root=str(preflight),
            prior_plan_sha256=driver.base.sha(preflight/"plan.json"),prior_job_sha256=driver.base.sha(preflight/"job.json")))
    write(stopped/"plan.json",old_plan)
    write(stopped/"job.json",dict(status="scientific_stop",workflow_complete=True,parent_pid=None,current=None,
        returncode=4,requires_user_action=True,formal_training_started=False,formal_optimizer_updates=0,
        cpu_seconds=107.725502106,gpu_seconds=115.847737390548,plan_sha256=driver.base.sha(stopped/"plan.json"),
        stages=[dict(name="sampler_pilot",status="scientific_stop",returncode=4,pid=None)]))
    monkeypatch.setattr(driver,"STOPPED_SHA256",{name:driver.base.sha(stopped/f"{name}.json")
        for name in ("job","plan","sampler_pilot")})
    monkeypatch.setattr(driver,"_code_root",lambda:code)
    return SimpleNamespace(code=code,old_code=old_code,stopped=stopped,preflight=preflight,config=config,
        sampling_plan=SAMPLING,output=tmp_path/"fresh")


def args(e):return e.stopped,e.preflight,e.output,e.config,e.sampling_plan,3,driver.GPU_UUID


def prerequisite(e):return driver.prerequisites(e.stopped,e.preflight,e.config,e.sampling_plan,3,driver.GPU_UUID)


def test_pilot_exact_three_candidates_seeds_work_and_threshold():
    p=driver.pilot_plan(SAMPLING)
    assert [(x["step_size"],x["leapfrog_steps"]) for x in p["candidates"]]==[(.02,48),(.01,96),(.005,192)]
    assert p["cases"]==driver.large.pilot_cases(SAMPLING)
    assert len(p["cases"])==15 and p["maximum_case_executions"]==45
    assert p["maximum_force_site_evaluations"]==13330022400
    assert p["minimum_production_acceptance"]==.6 and p["arrays_reused_in_production"] is False


def test_prerequisite_charges_stopped_cumulative_once_and_unique_old_bytes(evidence):
    result=prerequisite(evidence)
    assert result["prior_cpu_seconds"]==107.725502106
    assert result["prior_gpu_seconds"]==115.847737390548
    assert result["preflight_added_again"] is False
    assert result["prior_failed_pilot_force_site_evaluations"]==131072000
    expected=sum(p.stat().st_size for folder in (evidence.stopped,evidence.preflight) for p in folder.rglob("*") if p.is_file())
    assert result["prior_directory_bytes"]==expected
    assert result["additional_benchmark_updates"]==0 and result["model_loaded"] is False


def test_dry_cli_creates_nothing_and_no_compute(evidence,monkeypatch,capsys):
    monkeypatch.setattr(driver,"run",lambda *_:pytest.fail("dry executed"))
    assert driver.main(["--stopped-root",str(evidence.stopped),"--preflight-root",str(evidence.preflight),
        "--output",str(evidence.output),"--config",str(evidence.config),"--sampling-plan",str(evidence.sampling_plan)])==0
    p=json.loads(capsys.readouterr().out)
    assert not evidence.output.exists() and p["execution_requested"] is False
    assert p["stages"]==["integrator_pilots","sampling","training"]
    assert p["resources"]["cpu_seconds"]==86400 and p["resources"]["gpu_card_seconds"]==57600
    assert p["resources"]["directory_bytes"]==2*1024**3
    assert p["work_accounting"]["candidate_production_force_site_evaluations"]==[98650030080,195286794240,388560322560]
    assert p["training"]["options"]["steps"]==3000 and p["training"]["options"]["sample_batch"]==64


@pytest.mark.parametrize("field,value",[("status","failed"),("workflow_complete",False),("returncode",1),
    ("formal_training_started",True),("formal_optimizer_updates",1)])
def test_prior_terminal_sha_is_hard_bound(evidence,field,value):
    path=evidence.stopped/"job.json";row=json.loads(path.read_text());row[field]=value;write(path,row)
    with pytest.raises(ValueError,match="fingerprint"):
        prerequisite(evidence)


def test_old_sources_and_only_three_additions(evidence):
    (evidence.code/"scripts/unreviewed.py").write_text("# extra\n")
    with pytest.raises(ValueError,match="three authorized"):
        prerequisite(evidence)


@pytest.mark.parametrize("folder",["data","training"])
def test_no_prior_production_even_empty_directory(evidence,folder):
    (evidence.stopped/folder).mkdir()
    with pytest.raises(ValueError,match="before production"):
        prerequisite(evidence)


def test_configuration_cannot_change(evidence):
    row=json.loads(evidence.config.read_text());row["training"]["steps"]=3001;write(evidence.config,row)
    with pytest.raises(ValueError,match="configuration"):
        prerequisite(evidence)


def fake_sampling(monkeypatch,acceptance_by_candidate,*,numerical=False,exception=False):
    from recurrent_rg.data import phi4_hopping as kernel
    from recurrent_rg.data.phi4_hopping_pipeline import WORK_FIELDS
    calls=[];clock=[0.]
    monkeypatch.setattr(driver.time,"process_time",lambda:clock[0])
    monkeypatch.setattr(driver,"_pilot_pins",lambda:{"mock":"a"*64})
    def sample(kappa,lam,**kw):
        calls.append((kappa,kw));clock[0]+=1.
        if exception:raise RuntimeError("mock kernel execution failure")
        candidate=[x[1] for x in driver.CANDIDATES].index(kw["leapfrog_steps"])
        work={name:0 for name in WORK_FIELDS};work.update(hybrid_cycles=160,hmc_trajectories=160,
            force_site_evaluations=160*(kw["leapfrog_steps"]+1)*128**2)
        values=np.array([float("nan") if numerical else 1.])
        return dict(configurations=values,raw_observables=np.array([1.]),delta_h=np.array([.1]),
            trajectory_magnetization=np.array([1.]),production_acceptance=acceptance_by_candidate[candidate],
            burn_in_acceptance=.8,max_absolute_delta_h=.1,accepted=np.array([True]),diagnostics={},warnings=[],**work)
    monkeypatch.setattr(kernel,"sample_chain",sample)
    return calls,clock


def test_first_all_fifteen_pass_selected_and_no_later_candidate(tmp_path,monkeypatch):
    calls,_=fake_sampling(monkeypatch,[.7,.8,.9])
    assert driver.pilot_selection(tmp_path,SAMPLING,100)==0
    assert len(calls)==15 and all(row[1]["step_size"]==.02 for row in calls)
    row=json.loads((tmp_path/"integrator_selection.json").read_text())
    assert row["selected_candidate"]==0 and len(row["candidate_receipts"])==1
    result=driver.validate_selection(tmp_path,SAMPLING)
    assert result["selected_candidate"]==0
    assert result["remaining_production_cpu_projection_seconds"]==1536
    assert result["actual_endpoint_pilot_force_site_evaluations"]==1926758400


def test_finite_rejection_advances_only_to_next_preregistered_candidate(tmp_path,monkeypatch):
    calls,_=fake_sampling(monkeypatch,[0.,.7,.9])
    assert driver.pilot_selection(tmp_path,SAMPLING,100)==0
    assert len(calls)==16 and calls[0][1]["step_size"]==.02
    assert all(row[1]["step_size"]==.01 for row in calls[1:])
    assert calls[0][1]["seed"]==calls[1][1]["seed"]
    assert not (tmp_path/"pilot_02.json").exists()
    assert driver.validate_selection(tmp_path,SAMPLING)["selected_candidate"]==1
    rejected=json.loads((tmp_path/"pilot_00.json").read_text())
    assert rejected["returncode"]==4 and rejected["finite_acceptance_failure"] is True


def test_all_fixed_candidates_reject_scientific_stop_no_extra_candidate(tmp_path,monkeypatch):
    calls,_=fake_sampling(monkeypatch,[0.,.4,.59])
    assert driver.pilot_selection(tmp_path,SAMPLING,100)==4
    assert len(calls)==3
    row=json.loads((tmp_path/"integrator_selection.json").read_text())
    assert row["status"]=="scientific_stop" and row["workflow_complete"] is True
    assert row["selected_candidate"] is None and row["training_started"] is False


@pytest.mark.parametrize("numerical,exception",[(True,False),(False,True)])
def test_numerical_or_execution_failure_never_advances(tmp_path,monkeypatch,numerical,exception):
    calls,_=fake_sampling(monkeypatch,[.7,.8,.9],numerical=numerical,exception=exception)
    assert driver.pilot_selection(tmp_path,SAMPLING,100)==1
    assert len(calls)==1 and not (tmp_path/"pilot_01.json").exists()
    row=json.loads((tmp_path/"integrator_selection.json").read_text())
    assert row["status"]=="failed" and row["workflow_complete"] is False
    failed=json.loads((tmp_path/"pilot_00.json").read_text())
    assert failed["finite_acceptance_failure"] is False
    if numerical:assert failed["actual_force_site_evaluations"]==128450560
    if exception:
        assert failed["partial_work_unknown"] is True
        assert "interrupted case work unavailable" in failed["force_work_accounting"]
        assert row["partial_work_unknown"] is True


def test_budget_hold_is_not_acceptance_rejection_or_retry(tmp_path,monkeypatch):
    calls,_=fake_sampling(monkeypatch,[.7,.8,.9])
    assert driver.pilot_selection(tmp_path,SAMPLING,.5)==3
    assert len(calls)==1
    assert not (tmp_path/"pilot_01.json").exists()


@pytest.mark.parametrize("field,value",[("production_acceptance",.2),("finite",False),("seed",123)])
def test_selection_validation_rejects_tampered_pass_case(tmp_path,monkeypatch,field,value):
    fake_sampling(monkeypatch,[.7,.8,.9]);assert driver.pilot_selection(tmp_path,SAMPLING,100)==0
    path=tmp_path/"pilot_00.json";row=json.loads(path.read_text());row["rows"][3][field]=value;write(path,row)
    binding=json.loads((tmp_path/"integrator_selection.json").read_text())
    binding["candidate_receipts"][0]["sha256"]=driver.base.sha(path);write(tmp_path/"integrator_selection.json",binding)
    with pytest.raises(RuntimeError):driver.validate_selection(tmp_path,SAMPLING)


def mock_resources(monkeypatch):
    monkeypatch.setattr(driver.socket,"gethostname",lambda:driver.base.HOST)
    monkeypatch.setattr(driver.os,"getpriority",lambda *_:10)
    monkeypatch.setattr(driver,"prior_live_processes",lambda *_:[])
    monkeypatch.setattr(driver.base,"host_resources",lambda:dict(cpus=24,load1=1,mem_available_bytes=100*1024**3,io_full_avg10=0))
    gpu_reads=[]
    def gpu(i,u):
        gpu_reads.append((i,u));return dict(index=i,uuid=u,free_bytes=24*1024**3,compute_pids=[])
    monkeypatch.setattr(driver.base,"gpu_resources",gpu)
    clock=[0.]
    monkeypatch.setattr(driver.time,"monotonic",lambda:clock[0])
    monkeypatch.setattr(driver.time,"sleep",lambda s:clock.__setitem__(0,clock[0]+s))
    from scripts import report_phi4_hopping_qcr
    monkeypatch.setattr(report_phi4_hopping_qcr,"select_chinese_font",lambda _:Path("/mock/font"))
    return gpu_reads,clock


@pytest.mark.parametrize("rc,state,complete",[(1,"failed",False),(3,"budget_hold",False),(4,"scientific_stop",True)])
def test_parent_stop_keeps_old_roots_unchanged_and_no_production(evidence,monkeypatch,rc,state,complete):
    mock_resources(monkeypatch);calls=[]
    class Child:
        pid=99999999
        returncode=rc
        def poll(self):return rc
    def popen(command,**kw):
        calls.append(command);assert kw["env"]["CUDA_VISIBLE_DEVICES"]=="";return Child()
    monkeypatch.setattr(driver.subprocess,"Popen",popen)
    before={str(p):p.read_bytes() for folder in (evidence.stopped,evidence.preflight) for p in folder.rglob("*") if p.is_file()}
    assert driver.run(*args(evidence))==rc
    job=json.loads((evidence.output/"job.json").read_text())
    assert job["status"]==state and job["workflow_complete"] is complete
    assert len(calls)==1 and job["production_started"] is False and job["formal_training_started"] is False
    assert all(Path(p).read_bytes()==v for p,v in before.items())


def mock_selected(evidence,monkeypatch,projection=10000.):
    monkeypatch.setattr(driver,"validate_selection",lambda *_:dict(selected_candidate=0,
        selected_integrator=dict(step_size=.02,leapfrog_steps=48),selected_physical_plan={"selected":True},
        actual_endpoint_pilot_force_site_evaluations=1926758400,remaining_production_cpu_projection_seconds=projection))
    monkeypatch.setattr(driver,"validate_sampling",lambda *_:dict(data_manifest_sha256="a"*64,data_warning_count=8))
    class Child:
        pid=99999999
        returncode=0
        def poll(self):return 0
    calls=[]
    def popen(command,**kw):
        calls.append((command,kw["env"]["CUDA_VISIBLE_DEVICES"]))
        if len(calls)==1:write(evidence.output/"integrator_selection.json",{"complete":True})
        if len(calls)==2:write(evidence.output/"data/manifest.json",dict(total_force_site_evaluations=98650030080))
        return Child()
    monkeypatch.setattr(driver.subprocess,"Popen",popen)
    return calls


def test_parent_success_standard_train_runtime_budget_and_two_idle_checks(evidence,monkeypatch):
    gpu,_=mock_resources(monkeypatch);calls=mock_selected(evidence,monkeypatch)
    trained=[];monkeypatch.setattr(driver.base,"validate_training",lambda *a:trained.append(a))
    monkeypatch.setattr(driver,"_public_files",lambda *_a,**_k:None)
    assert driver.run(*args(evidence))==0
    job=json.loads((evidence.output/"job.json").read_text())
    assert job["status"]=="complete" and job["formal_optimizer_updates"]==3000
    assert job["cumulative_total_force_site_evaluations"]==131072000+1926758400+98650030080
    assert job["prior_cpu_seconds"]==107.725502106 and job["prior_gpu_seconds"]==115.847737390548
    assert [x[1] for x in calls]==["","",driver.GPU_UUID] and len(gpu)==2
    assert "scripts.prepare_phi4_hopping_b2_refined" in calls[1][0]
    assert calls[-1][0][2:5]==["-m","recurrent_rg","train"]
    original=json.loads((evidence.output/"training_config.json").read_text())
    runtime=json.loads((evidence.output/"training_runtime_config.json").read_text())
    cap=runtime["training"]["max_training_seconds"]
    assert cap==pytest.approx(57414.15226260945)
    runtime["training"]["max_training_seconds"]=57600
    assert runtime==original and trained==[(evidence.output,"a"*64)]


def test_production_gate_combines_remaining_cpu_sampling_and_training(evidence,monkeypatch):
    mock_resources(monkeypatch);calls=mock_selected(evidence,monkeypatch,projection=40000.)
    assert driver.run(*args(evidence))==3
    job=json.loads((evidence.output/"job.json").read_text())
    assert len(calls)==1 and job["production_started"] is False
    assert not (evidence.output/"data").exists()


def test_resources_wait_snapshot_rechecks_budget(evidence,monkeypatch):
    mock_resources(monkeypatch);clock=[0.]
    monkeypatch.setattr(driver.time,"process_time",lambda:clock[0])
    def host():
        clock[0]=86400.;return dict(cpus=24,load1=100,mem_available_bytes=100*1024**3,io_full_avg10=0)
    monkeypatch.setattr(driver.base,"host_resources",host)
    monkeypatch.setattr(driver.subprocess,"Popen",lambda *_a,**_k:pytest.fail("exhausted budget started child"))
    assert driver.run(*args(evidence))==3
    assert json.loads((evidence.output/"job.json").read_text())["stages"]==[]


def test_second_gpu_idle_snapshot_rechecks_budget_before_training(evidence,monkeypatch):
    mock_resources(monkeypatch);calls=mock_selected(evidence,monkeypatch)
    cpu=[0.];gpu=[]
    monkeypatch.setattr(driver.time,"process_time",lambda:cpu[0])
    def read_gpu(index,uuid):
        gpu.append(index)
        if len(gpu)==2:cpu[0]=40000.
        return dict(index=index,uuid=uuid,free_bytes=24*1024**3,compute_pids=[])
    monkeypatch.setattr(driver.base,"gpu_resources",read_gpu)
    assert driver.run(*args(evidence))==3
    job=json.loads((evidence.output/"job.json").read_text())
    assert len(calls)==2 and len(gpu)==2
    assert job["formal_training_started"] is False and job["gpu_idle_checks"]["spacing_seconds"]==10
    assert not (evidence.output/"training_runtime_config.json").exists()


def test_parent_error_waits_for_training_child_and_charges_until_exit(evidence,monkeypatch):
    _,clock=mock_resources(monkeypatch)
    monkeypatch.setattr(driver,"validate_selection",lambda *_:dict(selected_candidate=0,selected_integrator={},
        selected_physical_plan={},actual_endpoint_pilot_force_site_evaluations=1926758400,
        remaining_production_cpu_projection_seconds=10000.))
    monkeypatch.setattr(driver,"validate_sampling",lambda *_:dict(data_manifest_sha256="a"*64,data_warning_count=8))
    calls=[]
    class Complete:
        pid=99999999;returncode=0
        def poll(self):return 0
    class Running:
        pid=99999999;returncode=None
        def poll(self):return self.returncode
        def wait(self):clock[0]+=20;self.returncode=0;return 0
    def popen(command,**kw):
        calls.append(command)
        if len(calls)==1:write(evidence.output/"integrator_selection.json",dict(actual_force_site_evaluations=1926758400))
        if len(calls)==2:write(evidence.output/"data/manifest.json",dict(total_force_site_evaluations=98650030080))
        return Complete() if len(calls)<3 else Running()
    monkeypatch.setattr(driver.subprocess,"Popen",popen)
    def sleep(seconds):
        clock[0]+=seconds
        if seconds==5:raise RuntimeError("mock parent interruption")
    monkeypatch.setattr(driver.time,"sleep",sleep)
    with pytest.raises(RuntimeError,match="mock parent interruption"):driver.run(*args(evidence))
    job=json.loads((evidence.output/"job.json").read_text())
    assert len(calls)==3 and job["status"]=="failed"
    assert job["stages"][-1]["status"]=="child_exited_after_parent_error"
    assert job["stages"][-1]["returncode"]==0 and job["stages"][-1]["wall_seconds"]==25
    assert job["gpu_seconds"]==pytest.approx(115.847737390548+10+25)
