"""Pure plans and mocked array contracts; never run HMC or a model."""
from dataclasses import asdict, replace
import inspect
import json

import numpy as np
import pytest

from recurrent_rg.data import phi4_hopping_b2_large_pipeline as large
from recurrent_rg.data import phi4_hopping_b2_refined_pipeline as data
from scripts import prepare_phi4_hopping_b2_refined as cli


@pytest.mark.parametrize("step,steps,work", [
    (.02, 48, 98_650_030_080),
    (.01, 96, 195_286_794_240),
    (.005, 192, 388_560_322_560),
])
def test_candidate_exact_work_and_unchanged_physics(step, steps, work):
    options = data.Phi4HoppingB2RefinedOptions(step_size=step, leapfrog_steps=steps)
    before = large.generation_plan(large.Phi4HoppingB2LargeOptions())
    plan = data.generation_plan(options)
    assert options.max_force_site_evaluations == work
    assert plan["total_force_site_evaluations"] == 122880 * (steps + 1) * 128**2 == work
    assert plan["total_force_evaluations"] == 122880 * (steps + 1)
    assert step * steps == .96
    for name in ("configurations", "conditions", "parameter_cells", "total_hmc_trajectories",
                 "total_hybrid_cycles", "maximum_cluster_neighbor_visits", "train_cell_indices",
                 "test_cell_indices", "projected_output_bytes", "projected_working_bytes",
                 "action", "sampling_coordinate_maps", "schema", "requested_geometry"):
        assert plan[name] == before[name]
    changed = {name for name, value in asdict(options).items()
               if value != asdict(large.Phi4HoppingB2LargeOptions())[name]}
    assert changed == {"step_size", "leapfrog_steps", "max_force_site_evaluations"}
    refinement = plan["sampler_refinement"]
    assert refinement["changes_physical_target"] is False
    assert refinement["changes_integrator_proposal"] is True
    assert refinement["changes_sample_seed"] is False
    assert refinement["changes_train_test_split"] is False
    assert plan["production_adaptation"] is False
    assert plan["equilibrium_certified"] is False
    assert plan["labels_used_in_training"] is False
    assert plan["training_member"] == "configurations"


@pytest.mark.parametrize("change", [
    dict(sizes=(64,)), dict(dimension=3), dict(kappa_values=(.32,.68,1.04)),
    dict(quartic_lambda=.5), dict(h=.1), dict(chains=8), dict(samples=64),
    dict(burn_in=2048), dict(spacing=16), dict(seed=20261102), dict(test_every=4),
    dict(test_offset=0), dict(cluster_probability=1.), dict(max_trajectories=200001),
    dict(max_cluster_updates=2000001), dict(max_cluster_neighbor_visits=8100000001),
    dict(max_output_bytes=2*1024**3), dict(max_memory_bytes=1024**3),
    dict(step_size=.04,leapfrog_steps=24), dict(step_size=.02,leapfrog_steps=96),
    dict(step_size=.005,leapfrog_steps=48), dict(step_size=float("nan")),
    dict(max_force_site_evaluations=51_000_000_000),
    dict(max_force_site_evaluations=data.MAX_FORCE_SITE_EVALUATIONS+1),
])
def test_only_authorized_integrator_and_force_cap_change(change):
    with pytest.raises(ValueError):
        data.generation_plan(replace(data.Phi4HoppingB2RefinedOptions(), **change))


def test_counter_integer_and_positive_cap_guards():
    for value in (True, 48.5, 0, -1):
        with pytest.raises(ValueError):
            data.production_force_site_evaluations(value)
    for value in (True, .5, 0, -1):
        with pytest.raises(ValueError):
            data.generation_plan(data.Phi4HoppingB2RefinedOptions(max_force_site_evaluations=value))


def test_parameter_keyed_production_seeds_and_historical_pins_unchanged():
    baseline = large.Phi4HoppingB2LargeOptions()
    pins = data.source_pins()
    assert len(pins) == 9
    assert set(pins) == set(data.FROZEN_SOURCE_PINS) | {
        "recurrent_rg/data/phi4_hopping_b2_refined_pipeline.py",
        "scripts/prepare_phi4_hopping_b2_refined.py",
    }
    for name, digest in data.FROZEN_SOURCE_PINS.items():
        assert pins[name] == large.source_pins()[name] == digest
    for step, steps in data.INTEGRATOR_CANDIDATES:
        options = data.Phi4HoppingB2RefinedOptions(step_size=step, leapfrog_steps=steps)
        assert [data.chain_seed(options,128,c,r) for c in range(15) for r in range(4)] == [
            large.chain_seed(baseline,128,c,r) for c in range(15) for r in range(4)]


def test_old_file_pin_failure_is_not_ignored(monkeypatch):
    altered = dict(large.source_pins())
    altered["scripts/prepare_phi4_hopping_b2_large.py"] = "0"*64
    monkeypatch.setattr(large,"source_pins",lambda: altered)
    with pytest.raises(RuntimeError,match="fingerprint"):
        data.source_pins()


def test_production_loop_is_frozen_copy_not_old_global_monkeypatch():
    # Ensures the independent wrapper did not silently alter original generation.
    assert inspect.getsource(data.generate_dataset) == inspect.getsource(large.generate_dataset)


def test_cli_pure_plan_no_output_or_sampling(tmp_path,monkeypatch,capsys):
    def forbidden(*args,**kwargs):
        raise AssertionError("Dry plan must not sample")
    monkeypatch.setattr(data.kernel,"sample_chain",forbidden)
    destination = tmp_path/"not-created"
    plan = cli.main(["--output",str(destination),"--step-size",".01","--leapfrog-steps","96"])
    assert plan["total_force_site_evaluations"] == 195_286_794_240
    assert not destination.exists()
    assert json.loads(capsys.readouterr().out)["labels_used_in_training"] is False


def test_cli_execute_missing_output_fails_without_sampling(monkeypatch):
    monkeypatch.setattr(cli,"generate_dataset",lambda *a,**k: pytest.fail("Must require output"))
    with pytest.raises(SystemExit):
        cli.main(["--execute"])


def _mock_setup(monkeypatch):
    """Tiny artificial arrays solely exercise the unchanged write/loader contract."""
    options = replace(data.Phi4HoppingB2RefinedOptions(), sizes=(8,),kappa_values=(.32,1.04),
                      samples=4,burn_in=2,spacing=1)
    plan = data.generation_plan(data.Phi4HoppingB2RefinedOptions())
    cycles = options.burn_in + options.samples*options.spacing
    plan.update(options=asdict(options),conditions=2,parameter_cells=2,configurations=32,
                total_hmc_trajectories=8*cycles,total_hybrid_cycles=8*cycles,
                total_force_site_evaluations=8*cycles*(options.leapfrog_steps+1)*8**2,
                train_cell_indices=[0],test_cell_indices=[1])
    monkeypatch.setattr(data,"generation_plan",lambda received: plan)
    calls=[]
    def mocked(kappa,quartic_lambda,**kw):
        calls.append((kappa,quartic_lambda,kw))
        n = kw["burn_in"]+kw["samples"]*kw["spacing"]
        volume = kw["size"]**kw["dimension"]
        raw = np.ones((kw["samples"],8),dtype=np.float64)
        work = {name:0 for name in data.WORK_FIELDS}
        work.update(hmc_trajectories=n,hybrid_cycles=n,lazy_cluster_skips=n,
                    force_evaluations=n*(kw["leapfrog_steps"]+1),
                    force_site_evaluations=n*(kw["leapfrog_steps"]+1)*volume)
        result=dict(schema=data.SCHEMA,kernel_schema=data.counts.SCHEMA,
            cluster_probability=.5,production_adaptation=False,symmetry_augmentation=False,
            artificial_global_sign_flips=0,production_acceptance=1.,burn_in_acceptance=1.,
            saved_magnetization_sign_changes=0,initial=kw["initial"],seed=kw["seed"],
            configurations=np.ones((kw["samples"],kw["size"],kw["size"],1),dtype=np.float64),
            raw_observables=raw,accepted=np.ones(n,dtype=bool),delta_h=np.zeros(n),
            trajectory_magnetization=np.ones(n),cluster_attempted=np.zeros(n,dtype=bool),
            diagnostics={name:dict(ess=4.,mean=1.) for name in data.base.OBSERVABLES},
            warnings=["mock quality warning retained"],**work)
        for name in ("cluster_sizes","cluster_changed_sites","cluster_neighbor_visits",
                     "cluster_bond_tests","cluster_activated_bonds"):
            result[name]=np.zeros(n,dtype=np.int64)
        return result
    monkeypatch.setattr(data.kernel,"sample_chain",mocked)
    return options,plan,calls,mocked


def test_mock_generation_loader_configuration_only_contract(tmp_path,monkeypatch):
    from recurrent_rg.data.phi4_hopping_loader import load_dataset
    options,plan,calls,_ = _mock_setup(monkeypatch)
    root=tmp_path/"fresh"
    manifest=data.generate_dataset(root,options)
    assert len(calls)==8
    assert all(call[2]["step_size"]==.02 and call[2]["leapfrog_steps"]==48 for call in calls)
    assert [call[2]["initial"] for call in calls[:4]] == ["hot","hot","cold_positive","cold_negative"]
    assert manifest["warning_count"]==8
    assert manifest["labels_used_in_training"] is False
    assert all(not name.endswith("data.npz") for name in manifest["public_files"])
    assert manifest["actual_work"]["force_site_evaluations"]==plan["total_force_site_evaluations"]
    loaded=load_dataset(root,block_size=2,depth=1)
    assert loaded.configurations.shape==(8,4,8,8,1)
    assert loaded.train_indices.tolist()==[0,1,2,3]
    assert loaded.test_indices.tolist()==[4,5,6,7]
    with np.load(root/manifest["shards"][0]["path"],allow_pickle=False) as arrays:
        assert set(arrays.files)==set(data.ARRAY_MEMBERS)
        assert not set(arrays.files)&{"kappa","quartic_lambda","h","labels"}
    before=(root/"manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        data.generate_dataset(root,options)
    assert (root/"manifest.json").read_bytes()==before
    assert len(calls)==8


def test_mock_zero_acceptance_preserves_failed_chain_no_retry(tmp_path,monkeypatch):
    options,_,calls,mocked = _mock_setup(monkeypatch)
    def rejected(*args,**kwargs):
        result=mocked(*args,**kwargs)
        result["production_acceptance"]=0.
        return result
    monkeypatch.setattr(data.kernel,"sample_chain",rejected)
    root=tmp_path/"fresh"
    with pytest.raises(RuntimeError,match="Zero production"):
        data.generate_dataset(root,options)
    status=json.loads((root/"status.json").read_text())
    assert len(calls)==1
    assert status["status"]=="failed" and status["workflow_complete"] is False
    assert status["requires_user_action"] is True
    assert not (root/"manifest.json").exists()


def test_mock_source_pin_change_during_chain_is_blocked(tmp_path,monkeypatch):
    options,_,calls,mocked = _mock_setup(monkeypatch)
    pins=dict(data.source_pins())
    pins["recurrent_rg/data/phi4_hopping_b2_refined_pipeline.py"]="0"*64
    def altered(*args,**kwargs):
        result=mocked(*args,**kwargs)
        monkeypatch.setattr(data,"source_pins",lambda: pins)
        return result
    monkeypatch.setattr(data.kernel,"sample_chain",altered)
    root=tmp_path/"fresh"
    with pytest.raises(RuntimeError,match="source changed during"):
        data.generate_dataset(root,options)
    assert len(calls)==1
    assert not (root/"manifest.json").exists()


def test_mock_force_counter_mismatch_is_blocked_without_retry(tmp_path,monkeypatch):
    options,_,calls,mocked = _mock_setup(monkeypatch)
    def altered(*args,**kwargs):
        result=mocked(*args,**kwargs)
        result["force_site_evaluations"]-=1
        return result
    monkeypatch.setattr(data.kernel,"sample_chain",altered)
    root=tmp_path/"fresh"
    with pytest.raises(RuntimeError,match="force counters"):
        data.generate_dataset(root,options)
    assert len(calls)==1
    assert not (root/"manifest.json").exists()

