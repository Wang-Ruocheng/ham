"""Tiny CPU checks of the isolated sampler orchestration, not physical results."""
from dataclasses import asdict, replace
import json

import numpy as np
import pytest

from recurrent_rg.data import phi4_hopping_pipeline as historical
from recurrent_rg.data import phi4_hopping_refined_pipeline as refined
from recurrent_rg.data.phi4_hopping_loader import load_dataset
from scripts import prepare_phi4_hopping_refined as cli


def _tiny(**changes):
    values = dict(sizes=(9,), kappa_values=(.68, .74), samples=4,
                  burn_in=2, spacing=1, test_every=2, test_offset=0)
    values.update(changes)
    return refined.Phi4HoppingRefinedOptions(**values)


def test_formal_defaults_change_only_integrator_and_work_guard():
    original = asdict(historical.Phi4HoppingOptions())
    current = asdict(refined.Phi4HoppingRefinedOptions())
    assert {name for name in current if current[name] != original[name]} == {
        "step_size", "leapfrog_steps", "max_force_site_evaluations"}
    assert current["step_size"] * current["leapfrog_steps"] == pytest.approx(.96)
    plan = refined.generation_plan(refined.Phi4HoppingRefinedOptions())
    assert plan["total_force_site_evaluations"] == 3_450_470_400
    assert plan["total_hmc_trajectories"] == 106496
    assert plan["conditions"] == 13
    assert plan["configurations"] == 6656
    assert len(plan["train_cell_indices"]) == 8
    assert len(plan["test_cell_indices"]) == 5
    assert plan["labels_used_in_training"] is False
    assert plan["schema_version"] == 5
    assert plan["sampler_refinement"]["changes_kernel_implementation"] is False
    assert plan["sampler_refinement"]["changes_integrator_proposal"] is True
    assert plan["source_pins"] == refined.source_pins()


def test_old_hard_cap_is_preserved_not_monkeypatched():
    with pytest.raises(ValueError, match="cannot exceed 2,000,000,000"):
        historical.generation_plan(replace(historical.Phi4HoppingOptions(),
            step_size=.04, leapfrog_steps=24, max_force_site_evaluations=4_000_000_000))
    assert historical.Phi4HoppingOptions().max_force_site_evaluations == 2_000_000_000
    assert historical.source_pins() == refined.FROZEN_SOURCE_PINS


@pytest.mark.parametrize("changes", [
    {"step_size": .08}, {"leapfrog_steps": 12}, {"chains": 8},
    {"max_force_site_evaluations": 4_000_000_001}, {"cluster_probability": 1.},
    {"max_output_bytes": 1024**3+1}, {"max_memory_bytes": 0},
    {"max_cluster_updates": False}, {"h": .01}, {"samples": 5},
    {"sizes": (10,)}, {"dimension": 4}, {"kappa_values": (0., .68)},
    {"quartic_lambda": 0.}, {"seed": True}, {"spacing": 0},
])
def test_semantic_validation_not_relaxed(changes):
    with pytest.raises(ValueError):
        refined.generation_plan(_tiny(**changes))


def test_force_projection_checked_against_new_cap_before_creation(tmp_path):
    root = tmp_path / "not-created"
    options = replace(refined.Phi4HoppingRefinedOptions(), max_force_site_evaluations=3_000_000_000)
    with pytest.raises(ValueError, match="Projected force-site evaluation budget exceeded"):
        refined.generate_dataset(root, options)
    assert not root.exists()


def test_chain_seeds_bit_identical_to_frozen_policy():
    original, current = historical.Phi4HoppingOptions(), refined.Phi4HoppingRefinedOptions()
    for cell in range(13):
        for chain in range(4):
            assert refined.chain_seed(current, 36, cell, chain) == historical.chain_seed(original, 36, cell, chain)


def test_source_fingerprint_mismatch_fails_before_output(tmp_path, monkeypatch):
    monkeypatch.setattr(historical, "source_pins", lambda: {**refined.FROZEN_SOURCE_PINS,
                        "recurrent_rg/data/phi4.py": "0" * 64})
    root = tmp_path / "never"
    with pytest.raises(RuntimeError, match="fingerprint differs"):
        refined.generate_dataset(root, _tiny())
    assert not root.exists()


def test_small_real_kernel_dataset_contract_and_strict_loader(tmp_path):
    options = _tiny()
    root = tmp_path / "tiny"
    manifest = refined.generate_dataset(root, options)
    status = json.loads((root / "status.json").read_text())
    assert status["status"] == "complete" and status["workflow_complete"]
    assert status["returncode"] == 0 and status["error"] is None
    assert manifest["total_force_site_evaluations"] == 2 * 4 * 6 * 25 * 81
    assert manifest["source_pins"] == refined.source_pins()
    dataset = load_dataset(root)
    assert tuple(dataset.configurations.shape) == (8, 4, 9, 9, 1)
    assert len(dataset.train_indices) == len(dataset.test_indices) == 4
    assert dataset.warning_count == manifest["warning_count"] > 0
    assert not manifest["equilibrium_certified"]
    for cell in range(2):
        sidecar = json.loads((root / "conditions" / f"L9_c{cell:04d}" / "sidecar.json").read_text())
        assert [row["initial"] for row in sidecar["chains"]] == [
            "hot", "hot", "cold_positive", "cold_negative"]
        for chain, row in enumerate(sidecar["chains"]):
            assert row["seed"] == historical.chain_seed(options, 9, cell, chain)
        assert all(value > 0 for value in [row["production_acceptance"] for row in sidecar["chains"]])
    with pytest.raises(FileExistsError):
        refined.generate_dataset(root, options)


def test_kernel_failure_is_recorded_without_retry_or_completed_claim(tmp_path, monkeypatch):
    calls = []
    def fail(*args, **kwargs):
        calls.append(kwargs)
        raise FloatingPointError("deliberate tiny-test failure")
    monkeypatch.setattr(refined.kernel, "sample_chain", fail)
    root = tmp_path / "failure"
    with pytest.raises(FloatingPointError, match="deliberate"):
        refined.generate_dataset(root, _tiny())
    assert len(calls) == 1
    assert calls[0]["step_size"] == .04 and calls[0]["leapfrog_steps"] == 24
    status = json.loads((root / "status.json").read_text())
    assert status["status"] == "failed" and not status["workflow_complete"]
    assert status["requires_user_action"] and status["completed_conditions"] == 0
    assert status["current"] == "L9_c0000"
    assert not (root / "manifest.json").exists()


def test_zero_acceptance_is_an_execution_stop_not_a_retry(tmp_path, monkeypatch):
    original = refined.kernel.sample_chain
    calls = []
    def zero(*args, **kwargs):
        calls.append(kwargs)
        result = original(*args, **kwargs)
        result["production_acceptance"] = 0.
        return result
    monkeypatch.setattr(refined.kernel, "sample_chain", zero)
    root = tmp_path / "zero"
    with pytest.raises(RuntimeError, match="Zero production HMC acceptance"):
        refined.generate_dataset(root, _tiny())
    assert len(calls) == 1
    status = json.loads((root / "status.json").read_text())
    assert status["actual_work"]["hmc_trajectories"] == 6
    assert status["requires_user_action"]


def test_cli_defaults_are_pure_plan_only(tmp_path, capsys):
    root = tmp_path / "untouched"
    result = cli.main(["--output", str(root)])
    printed = json.loads(capsys.readouterr().out)
    assert printed["total_force_site_evaluations"] == result["total_force_site_evaluations"] == 3_450_470_400
    assert not root.exists()
    with pytest.raises(SystemExit):
        cli.main(["--execute"])
