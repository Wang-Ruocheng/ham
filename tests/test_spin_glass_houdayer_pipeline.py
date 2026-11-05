"""Storage compatibility, provenance and fail-closed Houdayer generation tests."""

from dataclasses import replace
import json

import numpy as np
import pytest

from recurrent_rg.data.spin_glass import generate_bonds
from recurrent_rg.data.spin_glass_pipeline import GenerationOptions, _bond_sha, _seed, iter_spin_glass
from recurrent_rg.data.spin_glass_houdayer_pipeline import (
    NUMERIC_LIMIT,
    SAMPLER_SCHEMA,
    generate_dataset,
    generation_plan,
)
from scripts.generate_spin_glass_houdayer import main


def _options(**changes):
    options = GenerationOptions(shape=(3, 3), temperatures=(0.9, 1.7), disorders=3,
                                replicas=4, samples=4, burn_in=2, spacing=1, seed=553,
                                distribution="gaussian", bond_scale=.3)
    return replace(options, **changes)


def test_plan_disables_administrative_caps_but_preserves_fixed_work_and_old_bond_seeds():
    options = _options(max_spin_proposals=1, max_memory_bytes=1, max_output_bytes=1, max_sites=1)
    plan = generation_plan(options)
    assert plan["sampler_schema"] == SAMPLER_SCHEMA
    assert not plan["administrative_budget_caps_enabled"]
    assert plan["numeric_representability_checks_enabled"]
    assert plan["options"]["max_memory_bytes"] == NUMERIC_LIMIT
    assert plan["total_single_spin_visits"] == 3 * 4 * 2 * 9 * 6
    assert plan["cluster_pair_attempts_per_temperature"] == 12
    assert plan["train_disorders"] == [0, 2]
    assert plan["test_disorders"] == [1]
    assert not plan["automatic_extension"]
    assert len(set(plan["sampling_seeds"])) == 3
    assert all(seed != _seed(options.seed, "thermal_ladders", i, options.shape)
               for i, seed in enumerate(plan["sampling_seeds"]))


def test_formal_size_plan_does_not_reinstate_historical_small_work_budget():
    options = GenerationOptions(
        shape=(36, 36), temperatures=tuple(.5 * 7 ** (i / 19) for i in range(20)),
        disorders=9, replicas=4, samples=512, burn_in=8192, spacing=16, seed=20261019,
    )
    plan = generation_plan(options)
    assert plan["total_single_spin_visits"] == 9 * 4 * 20 * 1296 * 16384
    assert plan["total_single_spin_visits"] > options.max_spin_proposals
    assert plan["sweeps_per_temperature_per_real_replica"] == 16384
    assert plan["trace_count_per_disorder"] == 1025
    assert not plan["administrative_budget_caps_enabled"]


@pytest.mark.parametrize("algorithm", ["pt", "houdayer_pt"])
def test_generated_shards_load_with_unmodified_loader_and_match_original_bond_fingerprints(tmp_path, algorithm):
    options = _options()
    root = tmp_path / algorithm
    manifest = generate_dataset(root, options, algorithm=algorithm, trace_interval=2)
    assert manifest["status"] == "complete"
    assert manifest["schema"] == "edwards_anderson_pt_v1"
    assert manifest["sampler_schema"] == SAMPLER_SCHEMA
    assert manifest["sampler"]["algorithm"] == algorithm
    assert manifest["sampler"]["initial"] == "hot"
    assert manifest["sampler"]["replicas"] == 4
    assert manifest["sampler"]["real_replicas_coupled"] == (algorithm == "houdayer_pt")
    assert len(manifest["sampler"]["kernel_source_sha256"]) == 64
    assert manifest["training_updates"] == 0
    assert "spin_glass_houdayer.py" in manifest["code_sha256"]
    records = list(iter_spin_glass(root))
    assert len(records) == 3
    assert [record.disorder_id for record in iter_spin_glass(root, "train")] == [0, 2]
    assert [record.disorder_id for record in iter_spin_glass(root, "test")] == [1]
    for i, record in enumerate(records):
        expected = generate_bonds(options.shape, _seed(options.seed, "quenched_bonds", i, options.shape),
                                  distribution=options.distribution, scale=options.bond_scale)
        assert _bond_sha(record.bonds) == _bond_sha(expected)
        assert record.configurations.shape == (2, 4, 4, 3, 3)
        assert record.overlap_view().shape == (2, 4, 3, 3, 1)
        with np.load(root / manifest["shards"][i]["path"], allow_pickle=False) as saved:
            assert saved["trace_sweeps"].tolist() == [0, 2, 4, 6]
            assert saved["trace_energy_density"].shape == (2, 4)
            assert saved["cluster_size_histogram"].shape == (2, 10)
            assert saved["sampler_algorithm"].item() == algorithm
            assert "replica_seeds" not in saved.files
    sampling = json.loads((root / "sampling.json").read_text())
    assert not sampling["equilibrium_certified"]
    assert not sampling["independent_initialization_comparison_performed"]
    assert sampling["disorders"][0]["pair_combinations_are_not_independent"]
    assert sampling["disorders"][0]["group_mean_series"][0]["q2_pair_mean"]["samples"] == 4
    assert "do not retain intact" in sampling["disorders"][0]["round_trip_label_caveat"]
    assert sampling["disorders"][0]["trace_summary"]["no_stationary_ess_inferred_from_burn_in_trace"]


def test_same_seed_reproduces_all_arrays_but_never_overwrites_a_destination(tmp_path):
    options = _options()
    first, second = tmp_path / "first", tmp_path / "second"
    a, b = generate_dataset(first, options), generate_dataset(second, options)
    for ra, rb in zip(a["shards"], b["shards"]):
        with np.load(first / ra["path"], allow_pickle=False) as xa, np.load(second / rb["path"], allow_pickle=False) as xb:
            assert xa.files == xb.files
            for key in xa.files:
                np.testing.assert_array_equal(xa[key], xb[key])
    with pytest.raises(FileExistsError):
        generate_dataset(first, options)


def test_actual_houdayer_generator_integrates_with_new_pair_zero_one_qcr_adapter(tmp_path):
    from scripts.prepare_spin_glass_houdayer_qcr import prepare_overlap_cache
    from recurrent_rg.data.ensembles import load_ensembles

    root, cache = tmp_path / "physical", tmp_path / "cache"
    generated = generate_dataset(root, _options(shape=(12, 12)), trace_interval=2)
    metadata = prepare_overlap_cache(root, cache)
    assert metadata["sampler"] == generated["sampler"]
    assert metadata["overlap_pair"] == [0, 1]
    assert metadata["real_replica_count"] == 4
    assert not metadata["equilibrium_certified"]
    assert not metadata["labels_used_in_training"]
    train, test = load_ensembles(cache / "ensembles.pt")
    assert train.configurations.shape == (4, 4, 12, 12, 1)
    assert test.configurations.shape == (2, 4, 12, 12, 1)
    assert train.couplings is None
    assert all(row["overlap_replicas"] == [0, 1] for row in metadata["train"] + metadata["test"])


def test_tampering_public_sampling_receipt_is_detected_by_original_loader(tmp_path):
    root = tmp_path / "data"
    generate_dataset(root, _options())
    with (root / "sampling.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="sampling receipt SHA256"):
        list(iter_spin_glass(root))


def test_failure_preserves_partial_files_and_does_not_restart(tmp_path, monkeypatch):
    import recurrent_rg.data.spin_glass_houdayer_pipeline as pipeline
    calls = []

    def fail(*args, **kwargs):
        calls.append(kwargs["seed"])
        raise RuntimeError("deliberate failure")

    monkeypatch.setattr(pipeline, "sample_houdayer_parallel_tempering", fail)
    root = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="deliberate failure"):
        generate_dataset(root, _options())
    status = json.loads((root / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["requires_user_action"]
    assert not status["workflow_complete"]
    assert len(calls) == 1
    assert (root / "plan.json").exists()


@pytest.mark.parametrize("change,extras", [
    ({"shape": (3,)}, {}), ({"samples": 3}, {}),
    ({"burn_in": -1}, {}), ({"samples": NUMERIC_LIMIT}, {}),
    ({}, {"initial": "mixed"}), ({}, {"cluster_interval": 0}),
    ({}, {"trace_interval": 0}), ({}, {"algorithm": "other"}),
])
def test_invalid_numeric_settings_do_not_create_output(tmp_path, change, extras):
    output = tmp_path / "not_created"
    with pytest.raises(ValueError):
        generate_dataset(output, _options(**change), **extras)
    assert not output.exists()


def test_cli_dry_run_never_writes_output_and_rejects_invalid_length(tmp_path, capsys):
    target = tmp_path / "not_written"
    plan = main(["--output", str(target), "--shape", "3", "3", "--temperatures", ".9", "1.7",
                 "--disorders", "3", "--samples", "4", "--burn-in", "2", "--dry-run"])
    assert not target.exists()
    assert plan["sampler_algorithm"] == "houdayer_pt"
    assert json.loads(capsys.readouterr().out)["dry_run"]
    with pytest.raises(SystemExit):
        main(["--output", str(target), "--length", "-1", "--temperatures", ".9", "1.7", "--dry-run"])
