"""细积分预检合同；仅 mock kernel，不生成物理构型。"""
import json
from unittest.mock import patch

import numpy as np
import pytest

from scripts import phi4_ru_fine_sampler_pilot as pilot


def mock_result(*args, **kwargs):
    from recurrent_rg.data.phi4_hybrid import OBSERVABLES
    accepted = np.ones(1152, dtype=bool)
    raw = np.repeat(np.linspace(.1, .2, 32)[:, None], len(OBSERVABLES), axis=1)
    return dict(accepted=accepted, delta_h=np.ones(1152) * .3,
        raw_observables=raw, raw_observable_names=list(OBSERVABLES), configurations=np.zeros((1,)),
        diagnostics={name: dict(mean=float(raw[:, i].mean()), ess=10., warning="low ESS")
            for i, name in enumerate(OBSERVABLES)},
        initial=kwargs["initial"], seed=kwargs["seed"], cluster_probability=.5,
        production_acceptance=1., burn_in_acceptance=1., hmc_trajectories=1152,
        hybrid_cycles=1152, force_evaluations=1152 * 25,
        force_site_evaluations=1152 * 25 * 108**2, wolff_cluster_updates=576,
        total_cluster_sites=10, total_cluster_changed_sites=10, total_cluster_neighbor_visits=40,
        total_cluster_bond_tests=20, total_cluster_activated_bonds=4,
        saved_magnetization_sign_changes=2, cpu_seconds=.01, wall_seconds=.02,
        warnings=["low ESS"])


def test_plan_is_fixed_and_side_effect_free(tmp_path):
    output = tmp_path / "never_created"
    with patch("recurrent_rg.data.phi4_hybrid.sample_hybrid_chain") as kernel:
        plan = pilot.build_plan(output)
    kernel.assert_not_called()
    assert not output.exists()
    assert plan["burn_in"] == 1024 and plan["samples"] == 32 and plan["spacing"] == 4
    assert plan["step_size"] * plan["leapfrog_steps"] == pytest.approx(.96)
    assert plan["total_force_site_evaluations"] == 18 * 1152 * 25 * 108**2
    seeds = [value for row in plan["conditions"] for value in row["seeds"].values()]
    assert len(set(seeds)) == len(seeds) == 18
    assert not plan["gate"]["ess_is_hard_gate"]


def test_default_cli_plans_without_execution(tmp_path, capsys):
    output = tmp_path / "not_created"
    assert pilot.main(["--output", str(output)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["independent_pilot_chains"] == 18
    assert not output.exists()


def test_all_fixed_pilots_and_warning_retention(tmp_path):
    with patch("recurrent_rg.data.phi4_hybrid.sample_hybrid_chain", side_effect=mock_result) as kernel:
        result = pilot.run(tmp_path / "fresh")
    assert kernel.call_count == 18
    assert result["production_allowed"] and result["stage_accepted"]
    assert result["total_force_site_evaluations"] == 18 * 1152 * 25 * 108**2
    assert len(result["warnings"]) == 18 and not result["equilibrium_certified"]
    assert all(row["first_accepted_hmc_cycle"] == 1 for row in result["chains"])
    assert all(call.kwargs["burn_in"] == 1024 and call.kwargs["step_size"] == .04
               and call.kwargs["leapfrog_steps"] == 24 for call in kernel.call_args_list)
    root = tmp_path / "fresh"
    assert not any(path.suffix in (".npz", ".npy", ".pt") for path in root.iterdir())
    inventory = json.loads((root / "public_files.json").read_text())["files"]
    assert all(pilot.sha(root / name) == value for name, value in inventory.items())
    with pytest.raises(FileExistsError):
        pilot.run(root)


@pytest.mark.parametrize("case", ["zero_acceptance", "frozen_cold", "poor_hot"])
def test_gate_holds_without_parameter_selection_or_resampling(tmp_path, case):
    def altered(*args, **kwargs):
        row = mock_result(*args, **kwargs)
        if case == "zero_acceptance" and kwargs["initial"] != "hot":
            row["accepted"][:] = False
            row["production_acceptance"] = row["burn_in_acceptance"] = 0.
        if case == "frozen_cold" and kwargs["initial"] != "hot":
            index = row["raw_observable_names"].index("mean_field_squared")
            row["raw_observables"][:, index] = .5
        if case == "poor_hot" and kwargs["initial"] == "hot":
            row["accepted"][1024:] = False
            row["production_acceptance"] = 0.
        return row
    with patch("recurrent_rg.data.phi4_hybrid.sample_hybrid_chain", side_effect=altered) as kernel:
        result = pilot.run(tmp_path / case)
    assert kernel.call_count == 18
    assert result["status"] == "preflight_hold" and result["returncode"] == 3
    assert result["workflow_complete"] and not result["production_allowed"]
    assert result["gate_failures"]


def test_cpu_guard_checks_outer_timer_and_keeps_completed_chain(tmp_path, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(pilot.time, "process_time", lambda: clock[0])
    def slow(*args, **kwargs):
        clock[0] += 1201.
        return mock_result(*args, **kwargs)
    with patch("recurrent_rg.data.phi4_hybrid.sample_hybrid_chain", side_effect=slow) as kernel:
        result = pilot.run(tmp_path / "budget")
    assert kernel.call_count == 1
    assert result["status"] == "budget_hold" and not result["production_allowed"]
    assert result["cpu_seconds"] == 1201.
    assert result["chains"][0]["cpu_seconds"] == 1201.
    assert result["chains"][0]["kernel_reported_cpu_seconds"] == .01


def test_source_mutation_fails_even_if_every_chain_passes(tmp_path):
    before = pilot.source_pins()
    after = dict(before, **{"recurrent_rg/data/phi4_hybrid.py": "changed"})
    with patch.object(pilot, "source_pins", side_effect=[before, after]), patch(
            "recurrent_rg.data.phi4_hybrid.sample_hybrid_chain", side_effect=mock_result):
        result = pilot.run(tmp_path / "bad_pin")
    assert result["status"] == "failed" and result["returncode"] == 1
    assert not result["production_allowed"] and not result["source_pins_match"]


def test_nonfinite_raw_stops_without_repeating_the_failed_chain(tmp_path):
    def invalid(*args, **kwargs):
        result = mock_result(*args, **kwargs)
        result["raw_observables"][0, 0] = np.nan
        return result
    with patch("recurrent_rg.data.phi4_hybrid.sample_hybrid_chain", side_effect=invalid) as kernel:
        result = pilot.run(tmp_path / "nonfinite")
    assert kernel.call_count == 1
    assert result["status"] == "failed" and not result["production_allowed"]
    assert result["completed_work_is_lower_bound"]
