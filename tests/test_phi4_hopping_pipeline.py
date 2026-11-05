"""Disposable tiny CPU integration tests; never production physical evidence."""
import json

import numpy as np
import pytest

from recurrent_rg.data import phi4 as base
from recurrent_rg.data import phi4_hopping as kernel
from recurrent_rg.data import phi4_hopping_pipeline as pipeline


def options(**changes):
    values = dict(sizes=(9,), dimension=2, kappa_values=(.5, .68, .86),
        quartic_lambda=1., samples=4, burn_in=2, spacing=1, step_size=.04,
        leapfrog_steps=2, seed=780, test_every=3, test_offset=0)
    values.update(changes)
    return pipeline.Phi4HoppingOptions(**values)


def test_default_plan_has_correct_physics_resource_counts_and_no_sampling(tmp_path, monkeypatch):
    monkeypatch.setattr(kernel, "sample_chain", lambda *a, **kw: pytest.fail("Plan sampled"))
    plan = pipeline.generation_plan(pipeline.Phi4HoppingOptions())
    assert plan["schema_version"] == 5 and plan["schema"] == pipeline.SCHEMA
    assert plan["field_representation"] == "phi"
    assert plan["evaluation_label_columns"] == ["kappa", "quartic_lambda", "h"]
    assert plan["options"]["quartic_lambda"] == 1. and plan["options"]["h"] == 0.
    np.testing.assert_allclose(plan["options"]["kappa_values"], np.linspace(.5, .86, 13))
    assert plan["options"]["sizes"] == (36,)
    assert plan["conditions"] == 13 and plan["configurations"] == 6656
    assert plan["total_hybrid_cycles"] == plan["total_hmc_trajectories"] == 106496
    assert plan["total_force_site_evaluations"] == 106496*13*36**2
    assert plan["maximum_cluster_neighbor_visits"] == 106496*36**2*4
    assert plan["test_cell_indices"] == [0, 3, 6, 9, 12]
    assert plan["reference"]["kappa_critical"] == .680601
    assert plan["reference_used_for_training"] is False
    assert not plan["labels_used_in_training"] and plan["training_updates"] == 0
    assert not plan["legacy_r_u_loader_compatible"]
    assert plan["projected_output_bytes"] < 1024**3
    assert plan["projected_working_bytes"] < 512*1024**2
    assert plan["source_pins"] == pipeline.source_pins()
    assert len(plan["sampling_coordinate_maps"]) == 13
    json.dumps(plan, allow_nan=False)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("changes", [
    {"kappa_values": ()}, {"kappa_values": (0., .5)}, {"kappa_values": (-.1, .5)},
    {"kappa_values": (.5, .5)}, {"kappa_values": (True, .5)},
    {"quartic_lambda": 0.}, {"quartic_lambda": -1.}, {"quartic_lambda": True},
    {"h": .1}, {"chains": 8}, {"samples": 5}, {"sizes": (12,)}, {"sizes": (3,)},
    {"dimension": 4}, {"cluster_probability": 1.}, {"cluster_probability": 0.},
    {"burn_in": -1}, {"spacing": 0}, {"step_size": 0.}, {"leapfrog_steps": 0},
    {"test_offset": 3}, {"max_trajectories": 1}, {"max_force_site_evaluations": 1},
    {"max_cluster_updates": 1}, {"max_cluster_neighbor_visits": 1},
    {"max_output_bytes": 1}, {"max_memory_bytes": 1},
])
def test_invalid_or_over_budget_plan_is_rejected_before_directory_or_rng(tmp_path, monkeypatch, changes):
    monkeypatch.setattr(kernel, "sample_chain", lambda *a, **kw: pytest.fail("Rejected plan sampled"))
    target = tmp_path/"absent"
    with pytest.raises(ValueError):
        pipeline.generate_dataset(target, options(**changes))
    assert not target.exists()


def test_seed_is_bound_to_physical_parameters_not_list_order_or_size():
    first = options()
    reordered = options(kappa_values=(.86, .5, .68))
    assert pipeline.chain_seed(first, 9, 0, 0) == pipeline.chain_seed(reordered, 9, 1, 0)
    assert len({pipeline.chain_seed(first, size, cell, chain)
                for size in (9, 18) for cell in range(3) for chain in range(4)}) == 24


def test_tiny_generation_to_strict_loader_uses_original_phi_and_retains_diagnostics(tmp_path, monkeypatch):
    from recurrent_rg.data.phi4_hopping_loader import load_dataset
    root = tmp_path/"fresh"
    before = pipeline.source_pins()
    manifest = pipeline.generate_dataset(root, options())
    assert manifest["source_pins"] == before == pipeline.source_pins()
    assert manifest["warning_count"] > 0  # Four-sample diagnostics cannot certify equilibrium.
    assert not manifest["equilibrium_certified"] and not manifest["production_adaptation"]
    total = {name: 0 for name in pipeline.WORK_FIELDS}
    for shard in manifest["shards"]:
        meta = json.loads((root/shard["sidecar"]).read_text())
        assert meta["kappa"] == options().kappa_values[shard["cell_index"]]
        assert "r" not in meta and "u" not in meta
        assert base.sha256_file(root/shard["sidecar"]) == shard["sidecar_sha256"]
        assert base.sha256_file(root/shard["path"]) == shard["sha256"]
        assert [r["initial"] for r in meta["chains"]] == ["hot", "hot", "cold_positive", "cold_negative"]
        with np.load(root/shard["path"], allow_pickle=False) as data:
            assert set(data.files) == set(pipeline.ARRAY_MEMBERS)
            assert not set(data.files) & {"kappa", "quartic_lambda", "h", "r", "u"}
            for chain, receipt in enumerate(meta["chains"]):
                for name in total:
                    total[name] += receipt[name]
                raw = data["raw_observables"][chain]
                fields = data["configurations"][chain, ..., 0]
                for i, field in enumerate(fields):
                    action, _ = kernel.action_and_gradient(field, meta["kappa"], meta["quartic_lambda"])
                    assert raw[i, 0] == pytest.approx(action/field.size, abs=1e-12)
                    assert raw[i, 1] == pytest.approx(field.mean(), abs=1e-12)
                    assert raw[i, 5] == pytest.approx(np.mean(field**2), abs=1e-12)
                np.testing.assert_array_equal(data["trajectory_magnetization"][chain, 2:], raw[:, 1])
    assert total == manifest["actual_work"]
    assert all((root/name).is_file() and name.endswith(".json") for name in manifest["public_files"])
    # Loader must not materialize raw trajectories, even though they are retained.
    original = np.load
    accessed = []
    class Archive:
        def __init__(self, value): self.value = value
        def __enter__(self): return self
        def __exit__(self, *args): self.value.close()
        def __getitem__(self, key):
            accessed.append(key)
            return self.value[key]
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: Archive(original(*args, **kwargs)))
    dataset = load_dataset(root)
    assert dataset.configurations.shape == (12, 4, 9, 9, 1)
    assert dataset.evaluation_label_columns == ("kappa", "quartic_lambda", "h")
    assert len(dataset.train_indices) == 8 and len(dataset.test_indices) == 4
    assert accessed == ["configurations"]*3


def test_existing_output_never_overwritten_and_failure_not_retried(tmp_path, monkeypatch):
    old = tmp_path/"old"
    old.mkdir()
    sentinel = old/"preserve.json"
    sentinel.write_text('{"user":true}')
    with pytest.raises(FileExistsError):
        pipeline.generate_dataset(old, options())
    assert sentinel.read_text() == '{"user":true}' and len(list(old.iterdir())) == 1
    calls = []
    def fail(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("injected failure")
    monkeypatch.setattr(kernel, "sample_chain", fail)
    failed = tmp_path/"failed"
    with pytest.raises(RuntimeError, match="injected"):
        pipeline.generate_dataset(failed, options())
    assert calls == [1] and not (failed/"manifest.json").exists()
    status = json.loads((failed/"status.json").read_text())
    assert status["status"] == "failed" and status["requires_user_action"]
    assert status["returncode"] == 1 and not status["workflow_complete"]
    assert status["failure_scope"] == "sampling_condition"


def test_generation_source_change_fails_closed_after_preserving_completed_conditions(tmp_path, monkeypatch):
    pins = pipeline.source_pins()
    values = [pins, {**pins, "recurrent_rg/data/phi4_hopping.py": "changed"}]
    monkeypatch.setattr(pipeline, "source_pins", lambda: values.pop(0))
    root = tmp_path/"changed"
    with pytest.raises(RuntimeError, match="source changed"):
        pipeline.generate_dataset(root, options())
    assert len(list((root/"conditions").glob("*/data.npz"))) == 3
    assert not (root/"manifest.json").exists()
    failed = json.loads((root/"status.json").read_text())
    assert failed["failure_scope"] == "global_finalization" and failed["current"] is None
    for path in (root/"conditions").glob("*/status.json"):
        assert json.loads(path.read_text())["status"] == "complete"


def test_cli_plan_is_dry_and_execute_requires_new_output(tmp_path, monkeypatch, capsys):
    from scripts.prepare_phi4_hopping import main
    monkeypatch.setattr(kernel, "sample_chain", lambda *a, **kw: pytest.fail("Dry CLI sampled"))
    root = tmp_path/"not_created"
    result = main(["--output", str(root)])
    assert result["options"]["quartic_lambda"] == 1. and result["training_updates"] == 0
    assert not root.exists()
    with pytest.raises(SystemExit):
        main(["--execute"])
    assert '"schema_version": 5' in capsys.readouterr().out
