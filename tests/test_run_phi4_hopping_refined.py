"""Authorized fixed integrator refinement; mocked workflows and disposable CPU pilot."""
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_phi4_hopping_refined as run
from recurrent_rg.data.phi4_hopping_pipeline import Phi4HoppingOptions, chain_seed, generation_plan, WORK_FIELDS
from recurrent_rg.data.phi4_hopping_refined_pipeline import generation_plan as refined_plan


CONFIG = Path(__file__).resolve().parents[1]/"configs/phi4_hopping_b3_g1_aligned_seed91_budget4h.json"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    old_code, code, failed, preflight, queue = [tmp_path/name for name in
                                               ("old_code","code","failed","preflight","queue")]
    for folder in (old_code,code,failed,preflight,queue):
        folder.mkdir()
    pins = {}
    for relative in ("recurrent_rg/phi4_aligned_qcr.py","scripts/run_phi4_hopping_qcr.py",
                     "scripts/continue_phi4_hopping_budget.py"):
        for folder in (old_code,code):
            (folder/relative).parent.mkdir(exist_ok=True)
            (folder/relative).write_text("# frozen computational source\n")
        pins[str(old_code/relative)] = run.base.sha(old_code/relative)
    (code/"scripts/run_phi4_hopping_refined.py").write_text("# authorized new wrapper\n")
    config = tmp_path/"config.json"
    config.write_bytes(CONFIG.read_bytes())
    write(preflight/"plan.json",dict(old=True))
    write(preflight/"job.json",dict(status="budget_hold",parent_pid=None,stages=[]))
    write(preflight/"gpu_preflight.json",dict(status="budget_hold",returncode=3,synthetic=True))
    monkeypatch.setattr(run,"PREFLIGHT_SHA256",run.base.sha(preflight/"gpu_preflight.json"))
    write(queue/"plan.json",dict(queue=True))
    write(queue/"job.json",dict(status="cancelled_before_compute"))
    extension = dict(prior_root=str(preflight),preflight_sha256=run.PREFLIGHT_SHA256,
        prior_plan_sha256=run.base.sha(preflight/"plan.json"),
        prior_job_sha256=run.base.sha(preflight/"job.json"),
        prior_cpu_seconds=26.0818944,prior_gpu_seconds=50.38484049588442,
        projection_seconds=12669.778172127903,
        prior_queue=dict(root=str(queue),cpu_seconds=.185175389,gpu_seconds=0.,executed_stages=0))
    (failed/"training_config.json").write_bytes(CONFIG.read_bytes())
    (failed/"reused_preflight.json").write_bytes((preflight/"gpu_preflight.json").read_bytes())
    write(failed/"budget_extension_receipt.json",extension)
    pins[str(config)] = run.base.sha(config)
    definition = dict(code_root=str(old_code),output=str(failed),host=run.base.HOST,
        GPU=dict(index=3,uuid=run.GPU_UUID),config_source=str(config),
        config_source_sha256=run.base.sha(config),source_sha256=pins,
        physical=generation_plan(Phi4HoppingOptions()),budget_extension=extension)
    write(failed/"plan.json",definition)
    write(failed/"job.json",dict(status="failed",workflow_complete=False,returncode=1,current=None,
        requires_user_action=True,parent_pid=None,cpu_seconds=32.966527042,gpu_seconds=50.38484049588442,
        plan_sha256=run.base.sha(failed/"plan.json"),stages=[dict(name="sampling",status="failed",returncode=1,pid=None)]))
    write(failed/"data/status.json",dict(status="failed",returncode=1,completed_conditions=0,
        error="FloatingPointError: overflow encountered in power"))
    monkeypatch.setattr(run,"_code_root",lambda:code)
    return SimpleNamespace(code=code,old_code=old_code,failed=failed,preflight=preflight,queue=queue,
                           config=config,output=tmp_path/"fresh")


def args(row):
    return (row.failed,row.preflight,row.output,row.config,3,run.GPU_UUID,row.queue)


def test_only_authorized_physical_fields_change_and_exact_work():
    before,after = asdict(Phi4HoppingOptions()),asdict(run.physical_options())
    changed = {key for key in before if before[key] != after[key]}
    assert changed == {"step_size","leapfrog_steps","max_force_site_evaluations"}
    assert before["step_size"]*before["leapfrog_steps"] == after["step_size"]*after["leapfrog_steps"] == .96
    plan = refined_plan(run.physical_options())
    assert plan["total_hybrid_cycles"] == 106496
    assert plan["total_force_site_evaluations"] == 3450470400
    assert plan["configurations"] == 6656
    assert plan["conditions"] == 13
    assert plan["options"]["seed"] == 20261031
    assert plan["train_cell_indices"] == [1,2,4,5,7,8,10,11]


def test_fixed_pilot_purpose_separation_and_failed_stream_regression():
    options,cases = run.physical_options(),run.pilot_cases()
    production = {chain_seed(options,36,cell,chain) for cell in range(13) for chain in range(4)}
    assert len(cases) == 27 and len({row["seed"] for row in cases}) == 27
    assert all(row["seed"] not in production for row in cases[:26])
    assert cases[-1]["seed"] == chain_seed(options,36,0,1)
    assert cases[-1]["shared_initial_stream_regression"] is True
    assert cases[-1]["pilot_not_independent_of_future_chain"] is True
    assert run.pilot_plan()["total_hybrid_cycles"] == 4320
    assert run.pilot_plan()["total_force_site_evaluations"] == 139968000


def test_dry_plan_no_compute_or_output_and_no_double_charge(evidence,monkeypatch,capsys):
    monkeypatch.setattr(run,"run",lambda *_:pytest.fail("dry executed"))
    assert run.main(["--failed-root",str(evidence.failed),"--preflight-root",str(evidence.preflight),
        "--prior-queue-root",str(evidence.queue),"--output",str(evidence.output),"--config",str(evidence.config)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert not evidence.output.exists() and result["execution_requested"] is False
    assert result["stages"] == ["sampler_pilot","sampling","training"]
    assert result["refinement"]["prior_cpu_seconds"] == pytest.approx(32.966527042)
    assert result["refinement"]["prior_gpu_seconds"] == pytest.approx(50.38484049588442)
    assert result["refinement"]["additional_benchmark_updates"] == 0
    assert result["training"]["options"]["steps"] == 3000
    assert result["training"]["options"]["system_batch"] == 32
    assert result["training"]["options"]["sample_batch"] == 64
    command = result["commands"]["sampling"]
    assert command[-6:] == ["--step-size",".04","--leapfrog-steps","24","--max-force-site-evaluations","4000000000"]
    assert result["resources"]["cpu_seconds"] == 21600
    assert result["resources"]["gpu_card_seconds"] == 14400
    assert "gpu_preflight" not in result["commands"]
    assert command[3] == "scripts.prepare_phi4_hopping_refined" and command[4] == "--execute"
    assert "--allow-legacy-training" not in str(result)


@pytest.mark.parametrize("field,value",[("status","running"),("workflow_complete",True),
    ("returncode",0),("current","sampling"),("requires_user_action",False),("cpu_seconds",float("inf"))])
def test_bad_prior_job_rejected(evidence,field,value):
    path=evidence.failed/"job.json"
    row=json.loads(path.read_text());row[field]=value;write(path,row)
    with pytest.raises(ValueError):
        run.plan(*args(evidence))


@pytest.mark.parametrize("field,value",[("error","other failure"),("completed_conditions",1),("returncode",0)])
def test_nonmatching_overflow_evidence_rejected(evidence,field,value):
    path=evidence.failed/"data/status.json"
    row=json.loads(path.read_text());row[field]=value;write(path,row)
    with pytest.raises(ValueError,match="documented sampling-only"):
        run.plan(*args(evidence))


@pytest.mark.parametrize("location",["old_code","code"])
def test_model_sampler_and_old_wrapper_frozen(evidence,location):
    (getattr(evidence,location)/"scripts/continue_phi4_hopping_budget.py").write_text("# altered\n")
    with pytest.raises(ValueError,match="source"):
        run.plan(*args(evidence))


def test_added_unreviewed_source_rejected(evidence):
    (evidence.code/"scripts/extra.py").write_text("# new\n")
    with pytest.raises(ValueError,match="inventory"):
        run.plan(*args(evidence))


def test_changed_training_config_rejected(evidence):
    row=json.loads(evidence.config.read_text());row["training"]["steps"]=20;write(evidence.config,row)
    with pytest.raises(ValueError,match="bindings"):
        run.plan(*args(evidence))


@pytest.mark.parametrize("folder",["failed","preflight","queue"])
def test_old_root_never_modified_by_nested_output(evidence,folder):
    with pytest.raises(ValueError,match="fresh output"):
        run.plan(evidence.failed,evidence.preflight,getattr(evidence,folder)/"nested",evidence.config,
                 prior_queue_root=evidence.queue)


def test_actual_prior_process_guard_checks_root_and_command(monkeypatch):
    original = Path.read_bytes
    monkeypatch.setattr(Path,"read_bytes",lambda p: b"python\0-m\0scripts.prepare_phi4_hopping\0--output\0/old/data\0"
                        if str(p)=="/proc/42/cmdline" else original(p))
    assert run._old_process_running(42,"/old")
    assert not run._old_process_running(42,"/other")
    assert not run._old_process_running(42,"/ol")
    assert not run._old_process_running(None,"/old")


def mock_chain(acceptance=1.,finite=True):
    return dict(configurations=np.full((32,36,36),0. if finite else np.nan),
        raw_observables=np.zeros((32,8)),delta_h=np.zeros(160),trajectory_magnetization=np.zeros(160),
        accepted=np.ones(160,dtype=bool),production_acceptance=acceptance,burn_in_acceptance=1.,
        max_absolute_delta_h=0.,diagnostics={},warnings=["short pilot ESS warning"],
        **{name:(160 if name in ("hybrid_cycles","hmc_trajectories") else
                 160*25*36**2 if name=="force_site_evaluations" else 0) for name in WORK_FIELDS})


def test_pilot_receipt_all_cases_fixed_options_no_cuda_no_arrays(tmp_path,monkeypatch):
    from recurrent_rg.data import phi4_hopping
    calls=[]
    monkeypatch.setattr(phi4_hopping,"sample_chain",lambda *a,**kw:(calls.append((a,kw)) or mock_chain()))
    path=tmp_path/"pilot.json"
    assert run.pilot(path)==0
    row=json.loads(path.read_text())
    assert len(row["rows"])==27 and row["status"]=="complete"
    assert all(kw["step_size"]==.04 and kw["leapfrog_steps"]==24 for _,kw in calls)
    assert all(kw["samples"]==32 and kw["burn_in"]==128 and kw["spacing"]==1 for _,kw in calls)
    assert all("short pilot ESS warning" in r["warnings"] for r in row["rows"])
    assert all("configurations" not in r and "accepted" in r and "delta_h" in r for r in row["rows"])
    path.rename(tmp_path/"sampler_pilot.json")
    verification = run._validate_pilot(tmp_path)
    assert verification["pilot_min_production_acceptance"]==1.
    assert verification["production_cpu_projection_seconds"] == pytest.approx(2*row["cpu_seconds"]*106496/4320)
    assert verification["production_cpu_projection"]["conservative_factor"] == 2.


@pytest.mark.parametrize("field,value",[("requires_user_action",True),("source_pins",{}),("cpu_seconds",0.)])
def test_pilot_receipt_gate_and_source_integrity(tmp_path,monkeypatch,field,value):
    from recurrent_rg.data import phi4_hopping
    monkeypatch.setattr(phi4_hopping,"sample_chain",lambda *a,**kw:mock_chain())
    path=tmp_path/"sampler_pilot.json"
    assert run.pilot(path)==0
    row=json.loads(path.read_text());row[field]=value;write(path,row)
    with pytest.raises(RuntimeError):run._validate_pilot(tmp_path)


@pytest.mark.parametrize("acceptance,finite",[(.59,True),(1.,False)])
def test_pilot_quality_failure_hard_stops_without_adaptation(tmp_path,monkeypatch,acceptance,finite):
    from recurrent_rg.data import phi4_hopping
    calls=[]
    monkeypatch.setattr(phi4_hopping,"sample_chain",lambda *a,**kw:(calls.append(kw) or mock_chain(acceptance,finite)))
    path=tmp_path/"pilot.json"
    assert run.pilot(path)==4 and len(calls)==1
    row=json.loads(path.read_text())
    assert row["status"]=="scientific_stop" and row["requires_user_action"]
    assert not (tmp_path/"data").exists()


def test_pilot_numerical_exception_is_execution_failure(tmp_path,monkeypatch):
    from recurrent_rg.data import phi4_hopping
    def fail(*a,**kw):
        raise FloatingPointError("overflow")
    monkeypatch.setattr(phi4_hopping,"sample_chain",fail)
    assert run.pilot(tmp_path/"pilot.json")==1
    row=json.loads((tmp_path/"pilot.json").read_text())
    assert row["status"]=="failed" and "FloatingPointError" in row["error"]


def setup_execution(monkeypatch):
    monkeypatch.setattr(run.socket,"gethostname",lambda:run.base.HOST)
    monkeypatch.setattr(run.os,"getpriority",lambda *_:10)
    monkeypatch.setattr(run.base,"host_resources",lambda:dict(cpus=24,load1=1,
        mem_available_bytes=100*1024**3,io_full_avg10=0))
    from scripts import report_phi4_hopping_qcr
    monkeypatch.setattr(report_phi4_hopping_qcr,"select_chinese_font",lambda _:Path("/mock/font"))
    clock=[0.]
    monkeypatch.setattr(run.time,"monotonic",lambda:clock[0])
    monkeypatch.setattr(run.time,"sleep",lambda x:clock.__setitem__(0,clock[0]+x))


def test_mock_pilot_stop_never_starts_production(evidence,monkeypatch):
    setup_execution(monkeypatch)
    calls=[]
    class Child:
        pid=999999999;returncode=4
        def poll(self):return self.returncode
    monkeypatch.setattr(run.subprocess,"Popen",lambda command,**kw:(calls.append((command,kw)) or Child()))
    before={str(p):p.read_bytes() for p in evidence.failed.rglob("*") if p.is_file()}
    assert run.run(*args(evidence))==4 and len(calls)==1
    assert calls[0][1]["env"]["CUDA_VISIBLE_DEVICES"]==""
    assert not (evidence.output/"data").exists() and not (evidence.output/"training").exists()
    job=json.loads((evidence.output/"job.json").read_text())
    assert job["status"]=="scientific_stop" and job["prior_cpu_seconds"]==pytest.approx(32.966527042)
    assert all(Path(path).read_bytes()==value for path,value in before.items())


def test_mock_success_order_two_gpu_checks_no_double_charge(evidence,monkeypatch):
    setup_execution(monkeypatch)
    gpu_reads=[]
    monkeypatch.setattr(run.base,"gpu_resources",lambda index,uuid:(gpu_reads.append((index,uuid)) or
        dict(index=index,uuid=uuid,free_bytes=24*1024**3,compute_pids=[])))
    calls=[]
    class Child:
        pid=999999999;returncode=0
        def poll(self):return self.returncode
    def popen(command,**kw):
        calls.append((command,kw["env"]["CUDA_VISIBLE_DEVICES"]))
        if "--pilot-output" in command:write(evidence.output/"sampler_pilot.json",dict(mock=True))
        return Child()
    monkeypatch.setattr(run.subprocess,"Popen",popen)
    monkeypatch.setattr(run,"_validate_pilot",lambda _:dict(pilot_min_production_acceptance=.9))
    monkeypatch.setattr(run.previous,"_validate_sampling",lambda _:dict(data_manifest_sha256="a"*64,data_warning_count=7))
    trained,public=[],[]
    monkeypatch.setattr(run.base,"validate_training",lambda *a:trained.append(a))
    monkeypatch.setattr(run,"_public_files",lambda root:public.append(root))
    assert run.run(*args(evidence))==0
    job=json.loads((evidence.output/"job.json").read_text())
    assert job["status"]=="complete" and job["workflow_complete"] is True
    assert [r["name"] for r in job["stages"]]==["sampler_pilot","sampling","training"]
    assert len(calls)==3 and [c[1] for c in calls]==["","",run.GPU_UUID]
    assert len(gpu_reads)==2 and job["gpu_idle_checks"]["spacing_seconds"]==10
    assert job["gpu_seconds"]==pytest.approx(60.38484049588442)
    assert job["data_warning_count"]==7 and job["additional_benchmark_updates"]==0
    assert trained==[(evidence.output,"a"*64)] and public==[evidence.output]


def test_no_host_authority_or_prior_live_process_no_outputs(evidence,monkeypatch):
    monkeypatch.setattr(run.socket,"gethostname",lambda:"not225")
    with pytest.raises(ValueError,match="pinned 225"):
        run.run(*args(evidence))
    assert not evidence.output.exists()
    setup_execution(monkeypatch)
    monkeypatch.setattr(run,"_old_process_running",lambda *_:True)
    with pytest.raises(ValueError,match="prior workflow process"):
        run.run(*args(evidence))
    assert not evidence.output.exists()


def test_pilot_measured_cpu_projection_holds_before_production(evidence,monkeypatch):
    setup_execution(monkeypatch)
    calls=[]
    class Child:
        pid=999999999;returncode=0
        def poll(self):return self.returncode
    def popen(command,**kw):
        calls.append(command)
        write(evidence.output/"sampler_pilot.json",dict(mock=True))
        return Child()
    monkeypatch.setattr(run.subprocess,"Popen",popen)
    monkeypatch.setattr(run,"_validate_pilot",lambda _:dict(remaining_production_cpu_projection_seconds=10000.))
    assert run.run(*args(evidence))==3
    job=json.loads((evidence.output/"job.json").read_text())
    assert job["status"]=="budget_hold" and len(calls)==1
    assert job["stages"][0]["status"]=="complete"
    assert not (evidence.output/"data").exists() and not (evidence.output/"training").exists()
