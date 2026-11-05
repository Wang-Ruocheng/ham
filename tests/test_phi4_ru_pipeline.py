"""New work envelope, unchanged hybrid physics; only disposable tiny fields."""
import json
from dataclasses import replace

import numpy as np
import pytest

from recurrent_rg.data import phi4 as base
from recurrent_rg.data import phi4_hybrid_pipeline as old
from recurrent_rg.data import phi4_ru_pipeline as ru


def tiny(**kwargs):
    values = dict(sizes=(9,), r_values=(-.3, .1), u_values=(.5, 1.),
                  samples=4, burn_in=2, spacing=1, leapfrog_steps=2,
                  step_size=.04, test_every=3, test_offset=1)
    values.update(kwargs)
    return ru.Phi4RUOptions(**values)


def test_authorized_plan_exact_geometry_work_and_cell_split_no_sampling(monkeypatch):
    monkeypatch.setattr(old.kernel, "sample_hybrid_chain", lambda *a, **k: pytest.fail("planning sampled"))
    p = ru.generation_plan(ru.Phi4RUOptions())
    assert p["conditions"] == p["parameter_cells"] == 45
    np.testing.assert_allclose(p["options"]["r_values"], np.linspace(-1.55, -.5, 9), rtol=0, atol=2e-16)
    np.testing.assert_array_equal(p["options"]["u_values"], np.linspace(.5, 1., 5))
    assert p["total_hmc_trajectories"] == 368640
    assert p["total_force_site_evaluations"] == 6210846720
    assert p["maximum_cluster_neighbor_visits"] == 1911029760
    assert p["configurations"] == 23040
    assert len(p["train_cell_indices"]) == 34 and len(p["test_cell_indices"]) == 11
    assert all(c in p["train_cell_indices"] for c in (0, 4, 40, 44))
    assert p["projected_output_bytes"] < 1024 ** 3
    assert p["projected_working_bytes"] < 512 * 1024 ** 2
    assert not p["labels_used_in_training"] and not p["equilibrium_certified"]


def test_historical_work_envelope_still_rejects_large_scan():
    opts = old.Phi4HybridOptions(r_values=ru.R_VALUES, u_values=ru.U_VALUES)
    with pytest.raises(ValueError, match="force-site"):
        old.generation_plan(opts)
    with pytest.raises(ValueError, match="cannot exceed"):
        replace(opts, max_force_site_evaluations=ru.FORCE_SITE_CAP).validate()


def test_checked_in_config_matches_new_plan_and_unchanged_loss():
    from pathlib import Path
    from scripts.run_phi4_ru_qcr import default_config
    path = Path(__file__).resolve().parents[1] / "configs/phi4_ru_b3_g2_seed91.json"
    config = json.loads(path.read_text())
    assert config == default_config()
    assert config["schema_version"] == 5
    assert config["model"]["d_g"] == 2
    assert config["model"]["block_size"] == config["model"]["stride"] == 3
    assert config["training"]["steps"] == 3000
    assert config["training"]["learning_rate"] == .001
    assert config["loss"]["closure_objective"] == "all_start_anchor"
    assert config["loss"]["closure_scale_weights"] is None
    assert not config["loss"]["coupling_supervision"]
    assert not config["loss"]["future_observable_loss"]
    assert not config["loss"]["prediction_prediction_loss"]


@pytest.mark.parametrize("change", [dict(h=.1), dict(dimension=1), dict(u_values=(0.,)),
    dict(cluster_probability=1.), dict(cluster_probability=True),
    dict(max_force_site_evaluations=7000000001), dict(max_trajectories=500001),
    dict(max_cluster_updates=500001), dict(max_cluster_neighbor_visits=2000000001),
    dict(max_output_bytes=1024 ** 3 + 1), dict(max_memory_bytes=512 * 1024 ** 2 + 1),
    dict(max_force_site_evaluations=1), dict(max_trajectories=1), dict(max_cluster_neighbor_visits=1)])
def test_bad_or_overbudget_plans_never_create_files(tmp_path, change):
    root = tmp_path / "rejected"
    with pytest.raises(ValueError):
        ru.generate_dataset(root, tiny(**change))
    assert not root.exists()


def test_same_frozen_generator_and_original_configuration_only_loader(tmp_path):
    from experiments.phi4_qcr import load_dataset
    root = tmp_path / "new"
    pins = ru.source_pins()
    opts = tiny()
    m = ru.generate_dataset(root, opts)
    assert ru.source_pins() == pins and m["source_pins"] == old.source_pins()
    receipt = json.loads((root / "wrapper_receipt.json").read_text())
    assert receipt["source_pins_before"] == receipt["source_pins_after"] == pins
    assert receipt["historical_kernel_unchanged"] and not receipt["labels_used_in_training"]
    assert len(m["shards"]) == 4
    data = load_dataset(root)
    assert data.configurations.shape == (16, 4, 9, 9, 1)
    assert len(data.train_indices) == 12 and len(data.test_indices) == 4
    for shard in m["shards"]:
        side = json.loads((root / shard["sidecar"]).read_text())
        assert base.sha256_file(root / shard["path"]) == side["data_sha256"]
        assert [c["initial"] for c in side["chains"]] == ["hot", "hot", "cold_positive", "cold_negative"]
        assert not side["production_adaptation"] and not side["symmetry_augmentation"]
        with np.load(root / shard["path"], allow_pickle=False) as a:
            assert set(a.files) == set(old.ARRAY_MEMBERS)
    assert json.loads((root / "status.json").read_text())["workflow_complete"]


def test_existing_output_never_modified(tmp_path):
    root = tmp_path / "preserved"
    root.mkdir()
    with pytest.raises(FileExistsError):
        ru.generate_dataset(root, tiny())
    assert not list(root.iterdir())


def test_new_envelope_does_not_change_kernel_or_seed_stream(tmp_path):
    new_options = tiny()
    legacy_options = old.Phi4HybridOptions(
        sizes=new_options.sizes, dimension=2, r_values=new_options.r_values,
        u_values=new_options.u_values, chains=4, samples=4, burn_in=2,
        spacing=1, leapfrog_steps=2, step_size=.04, seed=new_options.seed,
        test_every=3, test_offset=1)
    left, right = tmp_path / "new_envelope", tmp_path / "historical_kernel"
    new, legacy = ru.generate_dataset(left, new_options), old.generate_dataset(right, legacy_options)
    for n, p in zip(new["shards"], legacy["shards"]):
        with np.load(left / n["path"], allow_pickle=False) as a, np.load(right / p["path"], allow_pickle=False) as b:
            for member in old.ARRAY_MEMBERS:
                np.testing.assert_array_equal(a[member], b[member])
