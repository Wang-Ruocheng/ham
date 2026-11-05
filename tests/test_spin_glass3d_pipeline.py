"""3D shard compatibility, frozen plan, split integrity and fail-closed writes."""

from dataclasses import replace
import json

import numpy as np
import pytest

from recurrent_rg.data.spin_glass import generate_bonds
from recurrent_rg.data.spin_glass_pipeline import GenerationOptions, _bond_sha, _seed, iter_spin_glass
from recurrent_rg.data.spin_glass3d_pipeline import NUMERIC_LIMIT, SAMPLER_SCHEMA, generation_plan, generate_dataset
from scripts.generate_spin_glass3d import main


def options(**changes):
    base = GenerationOptions(shape=(3, 3, 3), temperatures=(.5, 1., 2.), disorders=3,
        replicas=4, samples=4, burn_in=2, spacing=1, seed=553)
    return replace(base, **changes)


def test_plan_fixed_3d_work_mask_and_no_independent_hot_cold_claim():
    plan = generation_plan(options(max_spin_proposals=1, max_memory_bytes=1,
        max_output_bytes=1, max_sites=1))
    assert plan["dimension"] == 3
    assert plan["sampler_schema"] == SAMPLER_SCHEMA
    assert plan["sampler_algorithm"] == "icm_pt"
    assert plan["options"]["max_memory_bytes"] == NUMERIC_LIMIT
    assert plan["total_single_spin_visits"] == 3 * 4 * 3 * 27 * 6
    assert plan["cluster_pair_attempts_by_temperature"] == [12, 12, 0]
    assert plan["sampler"]["saved_overlap_signs_unfolded"]
    assert plan["sampler"]["field"] == 0
    assert plan["train_disorders"] == [0, 2]
    assert plan["test_disorders"] == [1]
    assert len(set(plan["sampling_seeds"])) == 3
    assert not plan["automatic_extension"]
    assert not plan["administrative_budget_caps_enabled"]


def test_formal_plan_is_finite_9j_20t_4rep_512_not_doubled_for_cold():
    ts = tuple(np.geomspace(.5, 3.5, 20))
    plan = generation_plan(options(shape=(16, 16, 16), temperatures=ts, disorders=9,
        samples=512, burn_in=8192, spacing=16))
    assert plan["total_single_spin_visits"] == 9 * 20 * 4 * 4096 * 16384
    assert plan["trace_count_per_disorder"] == 1025
    assert plan["sampler"]["replicas"] == 4
    assert plan["sampler"]["initial"] == "hot"
    assert sum(plan["sampler"]["icm_active_temperatures"]) == sum(T <= 1 for T in ts)
    assert not plan["labels_used_in_training"]


@pytest.mark.parametrize("algorithm", ["pt", "icm_pt"])
def test_3d_dataset_is_loadable_with_existing_validated_loader_and_retains_diagnostics(tmp_path, algorithm):
    root = tmp_path / algorithm
    manifest = generate_dataset(root, options(), algorithm=algorithm, trace_interval=2)
    assert manifest["status"] == "complete"
    assert manifest["sampler_schema"] == SAMPLER_SCHEMA
    assert manifest["sampler"]["algorithm"] == algorithm
    assert len(manifest["sampler"]["kernel_source_sha256"]) == 64
    assert "spin_glass3d.py" in manifest["code_sha256"]
    status = json.loads((root / "status.json").read_text())
    assert status["workflow_complete"] and not status["equilibrium_certified"]
    assert status["completed_disorders"] == 3
    assert len(status["manifest_sha256"]) == 64
    plan = json.loads((root / "plan.json").read_text())
    assert plan["sampler"] == manifest["sampler"]
    records = list(iter_spin_glass(root))
    assert len(records) == 3
    assert [r.disorder_id for r in iter_spin_glass(root, "test")] == [1]
    for index, record in enumerate(records):
        expected = generate_bonds((3, 3, 3), _seed(553, "quenched_bonds", index, (3, 3, 3)))
        assert _bond_sha(record.bonds) == _bond_sha(expected)
        assert record.configurations.shape == (3, 4, 4, 3, 3, 3)
        assert record.overlap_view().shape == (3, 4, 3, 3, 3, 1)
        with np.load(root / manifest["shards"][index]["path"], allow_pickle=False) as saved:
            assert saved["sampler_algorithm"].item() == algorithm
            assert saved["cluster_winding_count"].shape == (3, 3)
            assert saved["cluster_size_histogram"].shape == (3, 28)
            assert saved["trace_sweeps"].tolist() == [0, 2, 4, 6]
            assert saved["icm_max_temperature"].item() == 1.
    sampling = json.loads((root / "sampling.json").read_text())
    assert not sampling["equilibrium_certified"]
    assert not sampling["independent_initialization_comparison_performed"]
    row = sampling["disorders"][0]
    assert row["raw_overlap_signs_preserved"]
    assert row["independent_physical_group_count"] == 1
    assert row["group_mean_series"][0]["q2_pair_mean"]["samples"] == 4
    assert row["cluster_statistics"][2]["attempts"] == 0


def test_deterministic_arrays_never_overwrite_or_resume(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    a, b = generate_dataset(first, options()), generate_dataset(second, options())
    for ra, rb in zip(a["shards"], b["shards"]):
        with np.load(first / ra["path"]) as xa, np.load(second / rb["path"]) as xb:
            assert xa.files == xb.files
            for key in xa.files:
                np.testing.assert_array_equal(xa[key], xb[key])
    with pytest.raises(FileExistsError):
        generate_dataset(first, options())
    with (first / "sampling.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="sampling receipt SHA256"):
        list(iter_spin_glass(first))


@pytest.mark.parametrize("change, extras", [
    ({"shape": (3, 3)}, {}), ({"field": .1}, {}), ({"samples": 3}, {}),
    ({"samples": NUMERIC_LIMIT}, {}), ({}, {"icm_max_temperature": 0}),
    ({}, {"icm_max_temperature": True}), ({}, {"initial": "mixed"}),
    ({}, {"algorithm": "houdayer_pt"}), ({}, {"trace_interval": 0}),
])
def test_invalid_plans_do_not_create_destination(tmp_path, change, extras):
    root = tmp_path / "invalid"
    with pytest.raises(ValueError):
        generate_dataset(root, options(**change), **extras)
    assert not root.exists()


def test_failed_sampling_keeps_partial_evidence(tmp_path, monkeypatch):
    import recurrent_rg.data.spin_glass3d_pipeline as pipeline

    def fail(*args, **kwargs):
        raise RuntimeError("deliberate kernel failure")

    monkeypatch.setattr(pipeline, "sample_icm_parallel_tempering", fail)
    root = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="deliberate kernel failure"):
        generate_dataset(root, options())
    status = json.loads((root / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["requires_user_action"]
    assert not status["workflow_complete"]
    assert (root / "plan.json").exists()


def test_cli_dry_run_leaves_no_files(tmp_path, capsys):
    output = tmp_path / "dry_only"
    plan = main(["--output", str(output), "--length", "4", "--temperatures", ".5", "1", "2",
                 "--disorders", "3", "--samples", "4", "--burn-in", "2", "--dry-run"])
    assert plan["dimension"] == 3
    assert json.loads(capsys.readouterr().out)["dry_run"]
    assert not output.exists()
