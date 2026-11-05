"""The literature-reference parameter scan does not redefine the physical action."""
import json
from pathlib import Path

import numpy as np

from recurrent_rg.data.phi4 import action_and_gradient
from recurrent_rg.data.phi4_hybrid_pipeline import Phi4HybridOptions, generation_plan


CONFIG = Path(__file__).resolve().parents[1] / "configs/phi4_u1_rg_reference_b3_g1_seed91.json"


def config():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_reference_action_matches_mc_equation_three_without_factorial_conversion():
    settings = config()
    r, u = settings["reference"]["r_critical"], settings["reference"]["u"]
    field = np.random.default_rng(528).normal(size=(9, 9))
    value, gradient = action_and_gradient(field, r, u)
    # Eq. (3): every forward edge once; r=mu_0_hat^2 and u=lambda_hat.
    expected = ((2. + r / 2.) * field ** 2 + u / 4. * field ** 4).sum()
    for axis in range(2):
        expected -= (field * np.roll(field, -1, axis=axis)).sum()
    np.testing.assert_allclose(value, expected, rtol=1e-14, atol=1e-12)
    direction = np.random.default_rng(529).normal(size=field.shape)
    eps = 1e-5
    plus = action_and_gradient(field + eps * direction, r, u)[0]
    minus = action_and_gradient(field - eps * direction, r, u)[0]
    np.testing.assert_allclose((plus - minus) / (2 * eps), (gradient * direction).sum(),
                               rtol=1e-8, atol=1e-8)


def test_fixed_scan_and_heldout_reference_do_not_expand_resource_caps():
    settings = config()
    data = settings["data"].copy()
    for key in ("sizes", "r_values", "u_values"):
        data[key] = tuple(data[key])
    plan = generation_plan(Phi4HybridOptions(**data))
    reference = settings["reference"]
    r = np.array(plan["options"]["r_values"])
    np.testing.assert_allclose(r, np.linspace(reference["r_critical"] - .4,
                                             reference["r_critical"] + .4, 13), atol=1e-14)
    assert r[6] == reference["r_critical"]
    assert reference["r_uncertainty"] == 4.8e-6
    assert plan["test_cell_indices"] == [0, 3, 6, 9, 12]
    assert plan["train_cell_indices"] == [1, 2, 4, 5, 7, 8, 10, 11]
    assert plan["configurations"] == 6656
    assert plan["total_force_site_evaluations"] == 1_794_244_608
    preflight_force_sites = 12 * (128 + 8 * 4) * 13 * 36 ** 2
    assert plan["total_force_site_evaluations"] + preflight_force_sites < 2_000_000_000
    assert not plan["labels_used_in_training"]
    assert not plan["equilibrium_certified"]
    assert "u=.2" not in plan["qualification"]


def test_default_geometry_and_independent_objective_are_explicit_not_old_b2_ablation():
    settings = config()
    model, train, loss = (settings[key] for key in ("model", "training", "loss"))
    assert settings["schema_version"] == 5
    assert model["lattice_sizes"] == [36, 12, 4]
    assert model["block_size"] == model["stride"] == 3
    assert model["d_g"] == 1 and model["decoder"] == "shared_affine"
    assert train["steps"] == 3000 and train["seed"] == 91
    assert train["learning_rate"] == .001 and train["q_initialization"] == "train_mean_bias"
    assert loss["closure_objective"] == "all_start_anchor"
    assert loss["closure_scale_weights"] is None
    assert loss["commutator"] == loss["rollout_weight"] == 5
    assert loss["Q_reconstruction_weight"] == 10 and loss["Q_subset_weights"] == [20, 5]
    assert not loss["future_observable_loss"] and not loss["prediction_prediction_loss"]
    assert not loss["coupling_supervision"]
    assert "not discrete" in loss["continuous_C_information"]
    assert settings["evaluation"]["thermal_nu"] is None


def test_tiny_u1_sampler_to_continuous_training_execution_only(tmp_path):
    """A disposable L9/two-update check, never a production scientific result."""
    from experiments.phi4_qcr import TrainingOptions, run_training
    from recurrent_rg.data.phi4_hybrid_pipeline import generate_dataset
    data, output = tmp_path / "tiny_data", tmp_path / "tiny_run"
    options = Phi4HybridOptions(sizes=(9,), dimension=2, r_values=(-1.6, -1.27, -.94),
        u_values=(1.,), chains=4, samples=4, burn_in=4, spacing=2,
        step_size=.08, leapfrog_steps=12, seed=20261028, test_every=3, test_offset=1)
    manifest = generate_dataset(data, options)
    assert manifest["status"] == "complete"
    summary = run_training(TrainingOptions(data=str(data), output=str(output), device="cpu",
        steps=2, system_batch=2, sample_batch=4, seed=91, threads=1,
        log_every=1, checkpoint_every=2, information_pairs=4, max_training_seconds=60))
    assert summary["actual_steps"] == 2
    assert summary["lattice_sizes"] == [9, 3, 1]
    assert not summary["labels_used_in_training"]
    assert summary["frozen_evaluation_verified"] and summary["thermal_nu"] is None
    for counts in summary["optimizer_steps"].values():
        assert counts["min"] == counts["max"] == 2
    saved = json.loads((output / "manifest.json").read_text())
    assert saved["closure_objective"] == "all_start_anchor"
    assert saved["closure_scale_weights"] is None
    assert saved["q_initialization_receipt"]["optimizer_updates"] == {"Q": 0, "C": 0, "R": 0}
