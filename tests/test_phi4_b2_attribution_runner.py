"""No physical sampling, device access or formal training in orchestration tests."""
from pathlib import Path
import json
from unittest.mock import Mock

import pytest
import torch

from experiments.phi4_qcr import state_fingerprint
from scripts import run_phi4_b2_attribution_controls as runner

ORIGINAL_UNCHANGED = runner.unchanged


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    monkeypatch.setattr(runner.socket,"gethostname",lambda:runner.HOST)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES",raising=False)
    monkeypatch.setattr(runner,"source_inputs",lambda:{"frozen":"digest"})
    root=tmp_path/"diagnostic"
    runner.prepare(root)
    monkeypatch.setattr(runner,"unchanged",lambda p:None)
    return root


def test_plan_explicit_frozen_controls_no_sampling_or_default_change(prepared):
    plan=runner.read(prepared/"plan.json")
    assert plan["schema_version"]==5 and plan["source"]==str(runner.SOURCE)
    assert plan["device"]=="cuda:2" and plan["gpu_uuid"]==runner.GPU_UUID
    assert [x["name"] for x in plan["commands"]]==["extract_R_state","decoder_controls","frozen_R_controls","frozen_C_diagnostic"]
    assert plan["resource_policy"]["cpu_seconds"]==900
    assert all("prepare_phi4" not in " ".join(x["command"]) for x in plan["commands"])


def test_never_overwrite_or_change_host_or_mask(prepared,tmp_path,monkeypatch):
    with pytest.raises(FileExistsError):runner.prepare(prepared)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","2")
    with pytest.raises(RuntimeError,match="unmasked"):runner.prepare(tmp_path/"masked")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    monkeypatch.setattr(runner.socket,"gethostname",lambda:"other")
    with pytest.raises(RuntimeError,match="vm225"):runner.prepare(tmp_path/"wrong")


def test_frozen_inventory_change_rejected(prepared,monkeypatch):
    plan=runner.read(prepared/"plan.json")
    ORIGINAL_UNCHANGED(plan)
    plan["source_sha256"][next(iter(plan["source_sha256"]))]="wrong"
    with pytest.raises(RuntimeError,match="code changed"):
        ORIGINAL_UNCHANGED(plan)


def test_r_extraction_exact_state_and_no_qc_payload(prepared,tmp_path,monkeypatch):
    source=tmp_path/"source"
    monkeypatch.setattr(runner,"SOURCE",source)
    modules={}
    modes=dict(legacy="one_step_and_rollout",all_start="all_start_anchor")
    for arm,mode in modes.items():
        target=source/"training"/arm;target.mkdir(parents=True)
        r=torch.nn.Sequential(torch.nn.Linear(1,96),torch.nn.Tanh(),torch.nn.Linear(96,96),torch.nn.Tanh(),torch.nn.Linear(96,1)).double()
        modules[arm]=dict(final_module_sha256=dict(R=state_fingerprint(r)))
        model={"R."+k:v for k,v in r.state_dict().items()}
        model["Q.forbidden"]=torch.tensor([123.])
        torch.save(dict(step=3000,closure_objective=mode,model=model),target/"training_step_3000.pt")
    runner.save(source/"training/summary.json",dict(arms=modules))
    runner.save(source/"training/manifest.json",dict(test_only=True))
    result=runner.extract_r_state(prepared)
    assert result["workflow_complete"] and result["returncode"]==0
    state=torch.load(prepared/"R_only_state.pt",weights_only=True)
    assert state["recipe"]=="phi4_b2_R_only_extraction_v1"
    for arm in modes:
        assert all(not k.startswith(("Q.","C.")) for k in state["arms"][arm]["state_dict"])
        assert state["arms"][arm]["source_checkpoint_path"]==str(source/"training"/arm/"training_step_3000.pt")
    with pytest.raises(FileExistsError):runner.extract_r_state(prepared)


def test_launch_once_and_no_duplicate_execution(prepared,monkeypatch):
    monkeypatch.setattr(runner.subprocess,"Popen",Mock(return_value=Mock(pid=129)))
    result=runner.launch(prepared)
    assert result["parent_pid"]==129
    with pytest.raises(FileExistsError):runner.launch(prepared)
    job=runner.read(prepared/"job.json");job["status"]="running";runner.save(prepared/"job.json",job)
    with pytest.raises(RuntimeError):runner.launch(prepared)


def test_cpu_resource_check_never_queries_gpu(monkeypatch):
    original=Path.read_text
    monkeypatch.setattr(Path,"read_text",lambda p,*a,**k:"MemAvailable: 100000000 kB" if str(p)=="/proc/meminfo" else "some avg10=0\nfull avg10=0.0 avg60=0.0" if str(p)=="/proc/pressure/io" else original(p,*a,**k))
    monkeypatch.setattr(runner.os,"getloadavg",lambda:(1.,1.,1.))
    monkeypatch.setattr(runner.os,"cpu_count",lambda:24)
    query=Mock(side_effect=AssertionError("CPU admission queried GPU"))
    monkeypatch.setattr(runner.subprocess,"check_output",query)
    assert runner.snapshot(False)["ready"]
    query.assert_not_called()


def test_gpu_admission_requires_two_separated_checks(prepared,monkeypatch):
    monkeypatch.setattr(runner,"snapshot",Mock(return_value=dict(ready=True)))
    sleeps=Mock();monkeypatch.setattr(runner.time,"sleep",sleeps)
    runner.admission(prepared,runner.read(prepared/"job.json"),"C",True)
    assert runner.snapshot.call_count==2
    sleeps.assert_called_once_with(10)


def test_failed_stage_never_retried_or_later_stage_started(prepared,monkeypatch):
    monkeypatch.setattr(runner,"admission",lambda *a:None)
    monkeypatch.setattr(runner.os,"getpriority",lambda *a:10)
    popen=Mock(return_value=Mock(pid=123,wait=Mock(return_value=1)))
    monkeypatch.setattr(runner.subprocess,"Popen",popen)
    result=runner.execute(prepared)
    assert result["status"]=="failed" and result["requires_user_action"]
    assert popen.call_count==1 and len(result["stages"])==1
    with pytest.raises(RuntimeError):runner.execute(prepared)


def test_budget_hold_not_execution_failure(prepared,monkeypatch):
    monkeypatch.setattr(runner,"admission",lambda *a:None)
    monkeypatch.setattr(runner.os,"getpriority",lambda *a:10)
    monkeypatch.setattr(runner.subprocess,"Popen",Mock(return_value=Mock(pid=123,wait=Mock(return_value=3))))
    result=runner.execute(prepared)
    assert result["status"]=="budget_hold" and result["returncode"]==3
    assert not result["workflow_complete"] and result["requires_user_action"]


def test_complete_receipts_and_public_inventory_exclude_tensor_state(prepared,monkeypatch):
    monkeypatch.setattr(runner,"admission",lambda *a:None)
    monkeypatch.setattr(runner.os,"getpriority",lambda *a:10)
    def spawn(command,**kwargs):
        def wait():
            if "--extract-r-state" in command:
                runner.save(prepared/"extraction.json",dict(status="complete",workflow_complete=True,returncode=0))
                torch.save(dict(test_only=True),prepared/"R_only_state.pt")
            else:
                output=Path(command[command.index("--output")+1]);output.mkdir()
                runner.save(output/"status.json",dict(status="complete",workflow_complete=True,returncode=0))
            return 0
        return Mock(pid=123,wait=wait)
    monkeypatch.setattr(runner.subprocess,"Popen",spawn)
    result=runner.execute(prepared)
    assert result["status"]=="complete" and result["workflow_complete"] and result["returncode"]==0
    assert len(result["stages"])==4 and all(s["returncode"]==0 for s in result["stages"])
    assert "R_only_state.pt" not in result["public_files"]
    assert "frozen_C/status.json" in result["public_files"]


def test_returncode_zero_but_missing_completion_receipt_fails(prepared,monkeypatch):
    monkeypatch.setattr(runner,"admission",lambda *a:None)
    monkeypatch.setattr(runner.os,"getpriority",lambda *a:10)
    def spawn(*args,**kwargs):
        runner.save(prepared/"extraction.json",dict(status="running",workflow_complete=False,returncode=0))
        return Mock(pid=123,wait=Mock(return_value=0))
    monkeypatch.setattr(runner.subprocess,"Popen",spawn)
    result=runner.execute(prepared)
    assert result["status"]=="failed" and "completion receipt" in result["error"]
    assert len(result["stages"])==1
