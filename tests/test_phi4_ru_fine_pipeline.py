from dataclasses import replace
import pytest

from recurrent_rg.data.phi4_ru_pipeline import Phi4RUOptions,generation_plan as old_plan
from recurrent_rg.data.phi4_ru_fine_pipeline import Phi4FineOptions,generation_plan


def test_fine_work_is_explicit_and_old_envelope_unchanged():
    fine=Phi4FineOptions()
    plan=generation_plan(fine)
    assert plan["total_hmc_trajectories"]==12*2048
    assert plan["total_force_site_evaluations"]==12*2048*25*108**2
    assert 7_000_000_000<plan["total_force_site_evaluations"]<=8_000_000_000
    assert plan["options"]["step_size"]*plan["options"]["leapfrog_steps"]==.96
    with pytest.raises(ValueError,match="force-site"):
        old_plan(Phi4RUOptions(sizes=(108,),r_values=fine.r_values,u_values=fine.u_values,
                              step_size=.04,leapfrog_steps=24))
    with pytest.raises(ValueError,match="cap"):
        old_plan(Phi4RUOptions(max_force_site_evaluations=8_000_000_000))


@pytest.mark.parametrize("change",[dict(h=.1),dict(dimension=3),dict(cluster_probability=1.),
    dict(step_size=.02,leapfrog_steps=48),dict(max_force_site_evaluations=8_000_000_001)])
def test_unplanned_changes_rejected(change):
    with pytest.raises(ValueError):
        generation_plan(replace(Phi4FineOptions(),**change))


def test_small_size_keeps_original_numerical_schedule():
    plan=generation_plan(replace(Phi4FineOptions(),sizes=(36,),step_size=.08,leapfrog_steps=12))
    assert plan["total_force_site_evaluations"]==12*2048*13*36**2


def test_cli_parameter_contract_does_not_sample(monkeypatch,tmp_path):
    from scripts import prepare_phi4_ru_fine as cli
    seen=[]
    monkeypatch.setattr(cli,"generate_dataset",lambda output,options:seen.append(options) or dict(shards=[],warning_count=0))
    cli.main(["--execute","--output",str(tmp_path/"unused"),"--sizes","108",
              "--r-values","-.7625","-.696875","-.63125","--u-values",".5"])
    assert seen[0].step_size==.04 and seen[0].leapfrog_steps==24
