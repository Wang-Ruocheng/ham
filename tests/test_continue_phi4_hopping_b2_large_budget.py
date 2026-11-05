"""Budget-extension orchestration contracts: no CUDA/model/physical execution."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import continue_phi4_hopping_b2_large_budget as continuation

REPO = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    old_code, code, prior = [tmp_path / name for name in ("old_code", "code", "prior")]
    for folder in (old_code, code, prior):
        folder.mkdir()
    pins = {}
    for relative in ("scripts/run_phi4_hopping_b2_large.py", "recurrent_rg/phi4_aligned_qcr.py"):
        for folder in (old_code, code):
            path = folder / relative
            path.parent.mkdir(exist_ok=True)
            path.write_text("# frozen\n")
        pins[str(old_code / relative)] = continuation.base.sha(old_code / relative)
    (code / continuation.NEW_SOURCE).write_text("# new orchestration only\n")
    config = json.loads((REPO / "configs/phi4_hopping_b2_L128_g1_seed91.json").read_text())
    old_config, new_config = old_code / "config.json", tmp_path / "new_config.json"
    write(old_config, config)
    write(prior / "training_config.json", config)
    config["training"]["max_training_seconds"] = continuation.GPU_SECONDS
    write(new_config, config)
    old_sampling = old_code / "sampling.json"
    old_sampling.write_bytes((REPO / continuation.large.DEFAULT_SAMPLING_PLAN).read_bytes())
    from recurrent_rg.data.phi4_hopping_b2_large_pipeline import generation_plan
    physical = generation_plan(continuation.large.physical_options(old_sampling))
    original = dict(code_root=str(old_code), output=str(prior), host=continuation.base.HOST,
        GPU=dict(index=3, uuid=continuation.GPU_UUID), config_source=str(old_config),
        config_source_sha256=continuation.base.sha(old_config), sampling_plan_source=str(old_sampling),
        sampling_plan_sha256=continuation.base.sha(old_sampling), source_sha256=pins, physical=physical,
        pilot=continuation.large.pilot_plan(old_sampling), work_accounting={"production_force_site_evaluations": 1})
    write(prior / "plan.json", original)
    projection = continuation.large.benchmark_projection(20., [11.6]*5)
    receipt = dict(status="budget_hold", returncode=3, workflow_complete=False, error=None,
        requires_user_action=True, current=None, synthetic=True, physical_result=False,
        checkpoint_saved=False, formal_model_reused=False, physical_data_read=False,
        actual_updates=7, warmup_updates=2, timed_updates=5, training_batch=[32,64,128,128,1],
        geometry=dict(block_size=2, depth=3, sizes=[128,64,32,16]), training_cap_seconds=14400.,
        peak_cuda_bytes=2069889024, device=dict(CUDA_VISIBLE_DEVICES=continuation.GPU_UUID),
        evaluation_report_reserve_seconds=500., initialization_seconds=20., timed_update_seconds=[11.6]*5,
        conservative_full_training_projection_seconds=projection,
        optimizer_steps={name:dict(min=7,max=7,parameters=1) for name in ("Q","C","R")})
    write(prior / "gpu_preflight.json", receipt)
    monkeypatch.setattr(continuation,"PREFLIGHT_SHA256",continuation.base.sha(prior / "gpu_preflight.json"))
    write(prior / "job.json", dict(status="budget_hold", workflow_complete=False, parent_pid=None,
        returncode=3, current=None, requires_user_action=True, formal_training_started=False,
        formal_optimizer_updates=0, cpu_seconds=96.086822453, gpu_seconds=115.847737390548,
        plan_sha256=continuation.base.sha(prior / "plan.json"),
        stages=[dict(name="gpu_preflight", status="budget_hold", returncode=3, pid=None)]))
    monkeypatch.setattr(continuation, "_code_root", lambda: code)
    return SimpleNamespace(prior=prior, config=new_config, code=code, old_code=old_code,
        sampling_plan=old_sampling, output=tmp_path / "fresh")


def args(row):
    return row.prior, row.output, row.config, 3, continuation.GPU_UUID, row.sampling_plan


def prerequisite(row):
    return continuation.prerequisites(row.prior, row.config, 3, continuation.GPU_UUID, row.sampling_plan)


def test_complete_seven_step_false_workflow_receipt_is_valid_timing_only(evidence):
    row = prerequisite(evidence)
    assert row["original_receipt_workflow_complete"] is False
    assert row["prior_cpu_seconds"] == 96.086822453
    assert row["prior_gpu_seconds"] == 115.847737390548
    assert row["additional_benchmark_updates"] == 0 and row["checkpoint_loaded"] is False
    assert row["projection_seconds"] == continuation.large.benchmark_projection(20., [11.6]*5)
    assert row["prior_directory_bytes"] > 0


def test_dry_plan_no_compute_output_or_benchmark(evidence, monkeypatch, capsys):
    monkeypatch.setattr(continuation, "run", lambda *_: pytest.fail("dry plan executed"))
    assert continuation.main(["--prior-root",str(evidence.prior),"--output",str(evidence.output),
        "--config",str(evidence.config),"--sampling-plan",str(evidence.sampling_plan)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert not evidence.output.exists()
    assert result["stages"] == ["sampler_pilot","sampling","training"]
    assert "gpu_preflight" not in result["commands"]
    assert result["resources"]["cpu_seconds"] == 86400
    assert result["resources"]["gpu_card_seconds"] == 57600
    assert result["resources"]["directory_bytes"] == 2*1024**3
    assert result["training"]["options"]["steps"] == 3000
    assert result["training"]["options"]["block_size"] == 2
    assert result["training"]["options"]["depth"] == 3
    assert result["physical"]["options"]["samples"] == 128
    assert result["training_budget_admission"]["only_changed_field"] == "training.max_training_seconds"


@pytest.mark.parametrize("section,key,value",[("training","steps",2999),("training","sample_batch",32),
    ("training","seed",92),("training","max_training_seconds",14400),("model","depth",2),
    ("loss","commutator",6),("loss","closure_scale_weights",[.1,.3,.6])])
def test_only_budget_configuration_change_allowed(evidence, section, key, value):
    row = json.loads(evidence.config.read_text())
    row[section][key] = value
    write(evidence.config, row)
    with pytest.raises(ValueError, match="Only the authorized"):
        prerequisite(evidence)


@pytest.mark.parametrize("field,value",[("workflow_complete",True),("formal_training_started",True),
    ("formal_optimizer_updates",1),("returncode",1),("current","sampling"),("status","running")])
def test_prior_nonterminal_or_nonzero_production_rejected(evidence, field, value):
    path = evidence.prior / "job.json"
    row = json.loads(path.read_text()); row[field] = value; write(path,row)
    with pytest.raises(ValueError, match="exclusive zero-production"):
        prerequisite(evidence)


@pytest.mark.parametrize("name",["data","training","sampler_pilot.json"])
def test_prior_production_or_pilot_blocks_fresh_continuation(evidence, name):
    (evidence.prior / name).mkdir()
    with pytest.raises(ValueError, match="exclusive zero-production"):
        prerequisite(evidence)


def test_preflight_sha_status_not_falsified(evidence):
    path = evidence.prior / "gpu_preflight.json"
    row = json.loads(path.read_text()); row.update(status="complete",returncode=0); write(path,row)
    with pytest.raises(ValueError, match="Synthetic receipt"):
        prerequisite(evidence)


@pytest.mark.parametrize("where",["old_code","code"])
def test_computational_bytes_frozen(evidence, where):
    (getattr(evidence,where) / "recurrent_rg/phi4_aligned_qcr.py").write_text("# changed\n")
    with pytest.raises(ValueError, match="source"):
        prerequisite(evidence)


def test_new_unreviewed_source_rejected(evidence):
    (evidence.code / "scripts/new.py").write_text("# new\n")
    with pytest.raises(ValueError, match="inventory"):
        prerequisite(evidence)


def test_gpu_pin_and_fresh_output(evidence):
    with pytest.raises(ValueError, match="exclusive zero-production"):
        continuation.prerequisites(evidence.prior,evidence.config,2,"GPU-other",evidence.sampling_plan)
    evidence.output.mkdir()
    with pytest.raises(ValueError, match="fresh output"):
        continuation.plan(*args(evidence))


def test_budget_separate_2gib_and_combined_cpu_gpu_projection():
    assert continuation.budget_ok(100,100,1.5*1024**3,cpu_projection=10000,gpu_projection=53000)
    assert not continuation.budget_ok(100,100,2*1024**3+1)
    assert not continuation.budget_ok(100,100,0,cpu_projection=34000,gpu_projection=53000)
    assert not continuation.budget_ok(100,5000,0,gpu_projection=53000)
    assert not continuation.budget_ok(float("nan"),0,0)


def test_live_prior_requires_matching_command_not_reused_pid(evidence, tmp_path):
    proc = tmp_path / "proc"
    (proc / "123").mkdir(parents=True)
    row = dict(parent_pid=123, stages=[])
    (proc / "123/cmdline").write_bytes(b"python\0-m\0unrelated\0")
    assert continuation.prior_live_processes(evidence.prior,row,proc) == []
    command = ["python","-m",continuation.large.MODULE,"--output",str(evidence.prior)]
    (proc / "123/cmdline").write_bytes("\0".join(command).encode())
    assert continuation.prior_live_processes(evidence.prior,row,proc)[0]["pid"] == 123


def mock_resources(monkeypatch):
    monkeypatch.setattr(continuation.socket,"gethostname",lambda:continuation.base.HOST)
    monkeypatch.setattr(continuation.os,"getpriority",lambda *_:10)
    monkeypatch.setattr(continuation.base,"host_resources",lambda:dict(cpus=24,load1=1,
        mem_available_bytes=100*1024**3,io_full_avg10=0))
    reads = []
    def gpu(index,uuid):
        reads.append((index,uuid))
        return dict(index=index,uuid=uuid,free_bytes=24*1024**3,compute_pids=[])
    monkeypatch.setattr(continuation.base,"gpu_resources",gpu)
    clock = [0.]
    monkeypatch.setattr(continuation.time,"monotonic",lambda:clock[0])
    monkeypatch.setattr(continuation.time,"sleep",lambda seconds:clock.__setitem__(0,clock[0]+seconds))
    from scripts import report_phi4_hopping_qcr
    monkeypatch.setattr(report_phi4_hopping_qcr,"select_chinese_font",lambda _:Path("/mock/font"))
    return reads, clock


@pytest.mark.parametrize("returncode,state,complete",[(7,"failed",False),(3,"budget_hold",False),(4,"scientific_stop",True)])
def test_mock_pilot_stop_retained_no_retry_no_production(evidence,monkeypatch,returncode,state,complete):
    mock_resources(monkeypatch)
    calls = []
    class Child:
        pid = 99999999
        def poll(self): return returncode
    def popen(command,**kwargs):
        calls.append(command)
        child=Child();child.returncode=returncode
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        return child
    monkeypatch.setattr(continuation.subprocess,"Popen",popen)
    before={str(path):path.read_bytes() for path in evidence.prior.rglob("*") if path.is_file()}
    assert continuation.run(*args(evidence)) == returncode
    job=json.loads((evidence.output / "job.json").read_text())
    assert len(calls)==1 and "--pilot-output" in calls[0]
    assert job["status"]==state and job["workflow_complete"] is complete
    assert job["formal_training_started"] is False and job["formal_optimizer_updates"]==0
    assert continuation.base.sha(evidence.output / "reused_preflight.json")==continuation.PREFLIGHT_SHA256
    assert all(Path(path).read_bytes()==value for path,value in before.items())
    assert not (evidence.output / "training").exists()
    assert "reused_preflight.json" in json.loads((evidence.output / "public_files.json").read_text())


def test_mock_success_runtime_cap_two_idle_checks_and_cumulative_accounting(evidence,monkeypatch):
    reads,clock=mock_resources(monkeypatch)
    monkeypatch.setattr(continuation.large,"validate_pilot",lambda *_:dict(remaining_production_cpu_projection_seconds=10000.))
    monkeypatch.setattr(continuation.large,"validate_sampling",lambda *_:dict(data_manifest_sha256="a"*64,data_warning_count=8))
    trained,public,calls=[],[],[]
    monkeypatch.setattr(continuation.base,"validate_training",lambda *x:trained.append(x))
    monkeypatch.setattr(continuation,"_public_files",lambda root,**kw:public.append((root,kw)))
    class Child:
        pid=99999999
        returncode=0
        def poll(self):return 0
    def popen(command,**kwargs):
        calls.append((command,kwargs["env"]["CUDA_VISIBLE_DEVICES"]))
        if len(calls)==1:
            write(evidence.output / "sampler_pilot.json",dict(rows=[dict(actual_work=dict(force_site_evaluations=123))]))
        if len(calls)==2:
            write(evidence.output / "data/manifest.json",dict(total_force_site_evaluations=456))
        return Child()
    monkeypatch.setattr(continuation.subprocess,"Popen",popen)
    assert continuation.run(*args(evidence))==0
    job=json.loads((evidence.output / "job.json").read_text())
    assert job["status"]=="complete" and job["formal_optimizer_updates"]==3000
    assert [row[1] for row in calls]==["","",continuation.GPU_UUID]
    assert len(reads)==2 and job["gpu_idle_checks"]["spacing_seconds"]==10
    assert job["gpu_seconds"]==pytest.approx(125.847737390548)
    assert job["data_warning_count"]==8
    assert job["actual_total_force_site_evaluations_including_endpoint_pilot"]==579
    assert job["directory_bytes"]==job["new_directory_bytes"]+job["prior_directory_bytes"]
    template=json.loads((evidence.output / "training_config.json").read_text())
    runtime=json.loads((evidence.output / "training_runtime_config.json").read_text())
    admission=json.loads((evidence.output / "training_admission.json").read_text())
    assert runtime["training"]["max_training_seconds"]==pytest.approx(57414.15226260945)
    runtime["training"]["max_training_seconds"]=57600
    assert runtime==template
    assert calls[-1][0][calls[-1][0].index("--config")+1]==str(evidence.output / "training_runtime_config.json")
    assert admission["planned_optimizer_updates"]==3000
    assert trained==[(evidence.output,"a"*64)] and public==[(evidence.output,dict(completed=True))]


def test_wait_snapshot_budget_exhaustion_stops_without_child(evidence,monkeypatch):
    mock_resources(monkeypatch)
    clock=[0.]
    monkeypatch.setattr(continuation.time,"process_time",lambda:clock[0])
    def host():
        clock[0]=continuation.CPU_SECONDS
        return dict(cpus=24,load1=100,mem_available_bytes=100*1024**3,io_full_avg10=0)
    monkeypatch.setattr(continuation.base,"host_resources",host)
    monkeypatch.setattr(continuation.subprocess,"Popen",lambda *_a,**_k:pytest.fail("budget hold started child"))
    assert continuation.run(*args(evidence))==3
    job=json.loads((evidence.output / "job.json").read_text())
    assert job["stages"]==[] and job["status"]=="budget_hold"
    assert "while waiting" in job["error"]


def test_production_cpu_projection_must_fit_before_sampling(evidence,monkeypatch):
    mock_resources(monkeypatch)
    monkeypatch.setattr(continuation.large,"validate_pilot",lambda *_:dict(remaining_production_cpu_projection_seconds=40000.))
    calls=[]
    class Child:
        pid=99999999
        returncode=0
        def poll(self):return 0
    def popen(command,**kwargs):
        calls.append(command)
        write(evidence.output / "sampler_pilot.json",dict(rows=[dict(actual_work=dict(force_site_evaluations=123))]))
        return Child()
    monkeypatch.setattr(continuation.subprocess,"Popen",popen)
    assert continuation.run(*args(evidence))==3
    assert len(calls)==1
    job=json.loads((evidence.output / "job.json").read_text())
    assert job["status"]=="budget_hold" and not job["formal_training_started"]
    assert not (evidence.output / "data").exists()


def prepare_successful_cpu_children(evidence,monkeypatch):
    monkeypatch.setattr(continuation.large,"validate_pilot",lambda *_:dict(remaining_production_cpu_projection_seconds=10000.))
    monkeypatch.setattr(continuation.large,"validate_sampling",lambda *_:dict(data_manifest_sha256="a"*64,data_warning_count=8))
    class Complete:
        pid=99999999
        returncode=0
        def poll(self):return 0
    def emit(index):
        if index==1:
            write(evidence.output / "sampler_pilot.json",dict(rows=[dict(actual_work=dict(force_site_evaluations=123))]))
        elif index==2:
            write(evidence.output / "data/manifest.json",dict(total_force_site_evaluations=456))
    return Complete,emit


def test_gpu_second_snapshot_rechecks_remaining_budget(evidence,monkeypatch):
    mock_resources(monkeypatch)
    Complete,emit=prepare_successful_cpu_children(evidence,monkeypatch)
    calls=[]
    def popen(command,**kwargs):
        calls.append(command);emit(len(calls));return Complete()
    monkeypatch.setattr(continuation.subprocess,"Popen",popen)
    cpu_clock=[0.]
    monkeypatch.setattr(continuation.time,"process_time",lambda:cpu_clock[0])
    gpu_calls=[]
    def gpu(index,uuid):
        gpu_calls.append(index)
        if len(gpu_calls)==2:
            cpu_clock[0]=40000.
        return dict(index=index,uuid=uuid,free_bytes=24*1024**3,compute_pids=[])
    monkeypatch.setattr(continuation.base,"gpu_resources",gpu)
    assert continuation.run(*args(evidence))==3
    job=json.loads((evidence.output / "job.json").read_text())
    assert len(calls)==2 and len(gpu_calls)==2
    assert job["gpu_idle_checks"]["spacing_seconds"]==10
    assert job["formal_training_started"] is False and job["status"]=="budget_hold"
    assert not (evidence.output / "training_runtime_config.json").exists()


def test_parent_error_preserves_running_training_and_charges_until_exit(evidence,monkeypatch):
    _,clock=mock_resources(monkeypatch)
    Complete,emit=prepare_successful_cpu_children(evidence,monkeypatch)
    calls=[]
    class Running:
        pid=99999999
        returncode=None
        def poll(self):return self.returncode
        def wait(self):
            clock[0]+=20.
            self.returncode=0
            return 0
    def popen(command,**kwargs):
        calls.append(command);emit(len(calls))
        return Complete() if len(calls)<3 else Running()
    monkeypatch.setattr(continuation.subprocess,"Popen",popen)
    def sleep(seconds):
        clock[0]+=seconds
        if seconds==5:
            raise RuntimeError("mock parent error")
    monkeypatch.setattr(continuation.time,"sleep",sleep)
    with pytest.raises(RuntimeError,match="mock parent error"):
        continuation.run(*args(evidence))
    job=json.loads((evidence.output / "job.json").read_text())
    assert len(calls)==3 and job["status"]=="failed"
    assert job["stages"][-1]["status"]=="child_exited_after_parent_error"
    assert job["stages"][-1]["wall_seconds"]==25
    assert job["stages"][-1]["returncode"]==0
    assert job["gpu_seconds"]==pytest.approx(115.847737390548+10+25)
