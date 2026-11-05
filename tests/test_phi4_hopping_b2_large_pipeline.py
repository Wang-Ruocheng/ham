"""Count, contract and mock tests only: no physical sampling or GPU work."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from recurrent_rg.data import phi4_hopping_b2_large_pipeline as data
from scripts import run_phi4_hopping_b2_large as run

ROOT = Path(__file__).resolve().parents[1]
UUID = "GPU-a167decd-ce49-6c8f-6bcf-136fde596683"


def test_exact_fresh_large_work_and_whole_cell_split():
    options = data.Phi4HoppingB2LargeOptions()
    plan = data.generation_plan(options)
    assert options.sizes == (128,) and options.seed == 20261101
    assert plan["configurations"] == 7680
    assert plan["total_hmc_trajectories"] == 122880
    assert plan["total_force_site_evaluations"] == 50331648000
    assert plan["maximum_cluster_neighbor_visits"] == 8053063680
    assert plan["projected_output_bytes"] == 1028145152 < 1024**3
    assert plan["projected_working_bytes"] < 512*1024**2
    assert plan["requested_geometry"]["sizes"] == [128,64,32,16]
    assert plan["test_cell_indices"] == [1,4,7,10,13]
    assert plan["labels_used_in_training"] is False
    assert plan["equilibrium_certified"] is False


@pytest.mark.parametrize("changed", [dict(sizes=(36,)),dict(dimension=3),
    dict(step_size=.02),dict(leapfrog_steps=48),dict(chains=8),
    dict(max_force_site_evaluations=4_000_000_000),dict(max_cluster_neighbor_visits=1_000_000_000)])
def test_no_silent_geometry_integrator_or_work_change(changed):
    with pytest.raises(ValueError):
        data.generation_plan(replace(data.Phi4HoppingB2LargeOptions(), **changed))


def test_old_physical_kernel_stays_pinned_and_larger_seed_is_separate():
    from recurrent_rg.data.phi4_hopping_refined_pipeline import source_pins
    before = source_pins()
    pins = data.source_pins()
    for name,digest in data.FROZEN_SOURCE_PINS.items():
        assert pins[name] == before[name] == digest
    assert "recurrent_rg/data/phi4_hopping_b2_large_pipeline.py" in pins


def test_new_config_and_pilot_are_fixed_no_sampling():
    options = run.physical_options(ROOT/run.DEFAULT_SAMPLING_PLAN)
    assert options == data.Phi4HoppingB2LargeOptions()
    cases = run.pilot_cases(ROOT/run.DEFAULT_SAMPLING_PLAN)
    assert len(cases) == 15 and len({x["seed"] for x in cases}) == 15
    assert {x["kappa"] for x in cases} == {.32,.65,.6799999999999999,.71,1.04}
    assert {x["initial"] for x in cases} == {"hot","cold_positive","cold_negative"}
    production = {data.chain_seed(options,128,c,r) for c in range(15) for r in range(4)}
    assert all(x["seed"] not in production for x in cases)
    plan = run.pilot_plan(ROOT/run.DEFAULT_SAMPLING_PLAN)
    assert plan["total_force_site_evaluations"] == 983040000
    assert plan["cuda_used"] is False


def test_parent_dry_gates_new_geometry_before_physical_production(tmp_path):
    definition = run.plan(tmp_path/"fresh",ROOT/run.DEFAULT_CONFIG,3,UUID,ROOT/run.DEFAULT_SAMPLING_PLAN)
    assert definition["stages"] == ["gpu_preflight","sampler_pilot","sampling","training"]
    assert definition["training"]["model"]["block_size"] == 2
    assert definition["training"]["options"]["depth"] == 3
    assert definition["training"]["options"]["steps"] == 3000
    assert definition["benchmark"]["training_batch"] == [32,64,128,128,1]
    assert definition["resources"]["gpu_card_seconds"] == 14400
    assert definition["resources"]["cpu_seconds"] == 21600
    assert definition["resources"]["max_cuda_bytes"] == 8*1024**3
    assert not (tmp_path/"fresh").exists()
    assert "--allow" not in json.dumps(definition)


def test_fixed_pilot_stops_on_rejection_without_retry(tmp_path,monkeypatch):
    from recurrent_rg.data import phi4_hopping as kernel
    calls=[]
    def rejected(*args,**kwargs):
        calls.append(kwargs)
        return dict(configurations=np.zeros((32,2,2)),raw_observables=np.zeros((32,8)),
            delta_h=np.zeros(160),trajectory_magnetization=np.zeros(160),accepted=np.zeros(160,dtype=bool),
            production_acceptance=0.,burn_in_acceptance=0.,max_absolute_delta_h=0.,
            diagnostics={},warnings=[],**{k:0 for k in data.WORK_FIELDS})
    monkeypatch.setattr(kernel,"sample_chain",rejected)
    path=tmp_path/"sampler_pilot.json"
    assert run.pilot(path,ROOT/run.DEFAULT_SAMPLING_PLAN)==4
    assert len(calls)==1 and calls[0]["size"]==128
    assert not (tmp_path/"data").exists()
    assert json.loads(path.read_text())["status"]=="scientific_stop"


def test_benchmark_budget_is_not_a_training_failure():
    assert run.benchmark_projection(10.,[40.]*5)>run.GPU_SECONDS
    assert not run.budget_ok(0,0,0,gpu_projection=run.benchmark_projection(10.,[40.]*5))


def test_loader_requires_explicit_b2_depth_for_l128(data_root):
    # Reuse only a synthetic fixture and rebind its declared dimensions; no sampler.
    from recurrent_rg.data import phi4_hopping_loader as loader
    from test_phi4_hopping_loader import _read,_write,_refresh
    plan,manifest=_read(data_root/"plan.json"),_read(data_root/"manifest.json")
    for value in (plan,manifest): value["options"]["sizes"]=[128]
    _write(data_root/"plan.json",plan)
    for cell,shard in enumerate(manifest["shards"]):
        shard["size"]=128
        path=data_root/shard["path"]
        np.savez(path,configurations=np.ones((4,4,128,128,1)))
        shard["sha256"]=loader._sha256_file(path)
        sidepath=data_root/shard["sidecar"]
        side=_read(sidepath);side.update(size=128,data_sha256=shard["sha256"])
        _write(sidepath,side);shard["sidecar_sha256"]=loader._sha256_file(sidepath)
    manifest["plan_sha256"]=loader._sha256_file(data_root/"plan.json")
    _write(data_root/"manifest.json",manifest);_refresh(data_root)
    with pytest.raises(ValueError,match="geometry"):
        loader.load_dataset(data_root)
    loaded=loader.load_dataset(data_root,block_size=2,depth=3)
    assert loaded.configurations.shape==(12,4,128,128,1)
    assert loaded.scale==1.


from test_phi4_hopping_loader import data_root
