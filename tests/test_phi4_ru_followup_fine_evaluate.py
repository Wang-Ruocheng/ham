import json
from pathlib import Path
import pytest

from experiments import phi4_ru_followup_fine_evaluate as fine


def plan_at(root):
    rows=[dict(size=108 if i<3 else 36,physical=dict(options=dict(
        step_size=.04 if i<3 else .08,leapfrog_steps=24 if i<3 else 12))) for i in range(13)]
    plan=dict(source_root=str(root/"source"),validation=str(root/"r_validation"),subsets=rows)
    (root/"plan.json").write_text(json.dumps(plan))
    return plan


def test_fine_integrator_contract_before_loading_data(tmp_path):
    plan=plan_at(tmp_path)
    assert fine.verify_fine_plan(tmp_path/"source",tmp_path,tmp_path/"r_validation")==plan
    plan["subsets"][0]["physical"]["options"]["step_size"] = .08
    (tmp_path/"plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError,match="integration"):
        fine.verify_fine_plan(tmp_path/"source",tmp_path,tmp_path/"r_validation")


def test_fine_wrapper_preserves_evaluator_and_adds_provenance(tmp_path,monkeypatch):
    plan_at(tmp_path)
    output=tmp_path/"evaluation"
    calls=[]
    def inherited(*args):
        calls.append(args)
        output.mkdir()
        (output/"summary.json").write_text('{"status":"complete"}')
        (output/"status.json").write_text('{"status":"complete"}')
    monkeypatch.setattr(fine.frozen,"run",inherited)
    fine.run(tmp_path/"source",tmp_path,tmp_path/"r_validation",output)
    assert len(calls)==1
    inventory=json.loads((output/"public_files.json").read_text())
    assert "fine_integration_receipt.json" in inventory
    receipt=json.loads((output/"fine_integration_receipt.json").read_text())
    assert receipt["Q_updates"]==receipt["C_updates"]==receipt["R_updates"]==0
    assert receipt["integration_by_size"]["108"]==dict(step_size=.04,leapfrog_steps=24)
