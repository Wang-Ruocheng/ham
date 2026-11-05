"""New 3D overlap serialization and frozen descriptive report, not physics results."""
from dataclasses import asdict
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from recurrent_rg.checkpoints import build_model
from recurrent_rg.config import RunConfig, ModelConfig
from recurrent_rg.data.ensembles import load_ensembles
from recurrent_rg.data.spin_glass import energy
from recurrent_rg.data.spin_glass_pipeline import GenerationOptions, SCHEMA, _bond_sha, _json, _sha
from scripts.prepare_spin_glass3d_qcr import prepare_overlap_cache, SAMPLER_SCHEMA, _plaquette_frustration
from scripts.report_spin_glass3d import (validate_metadata, collect, root_candidates,
    inverse_temperature_branches, coordinate_diagnostics, mi_null_diagnostic, figures)
from scripts.report_spin_glass1d import _model_state_sha256, compute_closure_metrics


@pytest.fixture(autouse=True)
def one_thread():
    original = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


def synthetic_source(tmp_path, *, samples=8, side=8):
    root = tmp_path / "synthetic_icm3d_source"; (root / "shards").mkdir(parents=True)
    options = GenerationOptions(shape=(side,) * 3, temperatures=(.6, 1., 2.), disorders=3,
        replicas=4, samples=samples, burn_in=0, spacing=1)
    sampler = dict(schema=SAMPLER_SCHEMA, algorithm="icm_pt", initial="hot", replicas=4,
        real_replicas_coupled=True, kernel_source_sha256="c" * 64, icm_max_temperature=1., saved_overlap_signs_unfolded=True)
    _json(root / "plan.json", dict(schema=SCHEMA, options=asdict(options), sampler=sampler, synthetic_fixture=True))
    _json(root / "sampling.json", dict(schema=SCHEMA, synthetic_fixture=True, equilibrium_certified=False))
    rng, originals, shards = np.random.default_rng(3399), {}, []
    for disorder in range(3):
        bonds = np.ones((3, side, side, side)); bonds[0, 0, 0, disorder] = -1
        spins = rng.choice(np.array([-1, 1], np.int8), size=(3, 4, samples, side, side, side))
        spins[:, 2] = 1; spins[:, 3] = -1
        path = root / "shards" / f"disorder_{disorder:06d}.npz"
        np.savez(path, configurations=spins, bonds=bonds, energies=energy(spins, bonds),
            temperatures=np.asarray(options.temperatures), disorder_id=np.array(disorder))
        originals[disorder] = spins
        shards.append(dict(disorder_id=disorder, split="test" if disorder == 1 else "train",
            path=str(path.relative_to(root)), sha256=_sha(path), bytes=path.stat().st_size, bond_sha256=_bond_sha(bonds)))
    manifest = dict(schema=SCHEMA, status="complete", options=asdict(options), shards=shards,
        plan_sha256=_sha(root / "plan.json"), sampling_sha256=_sha(root / "sampling.json"),
        sampler=sampler, equilibrium_certified=False)
    _json(root / "manifest.json", manifest)
    _json(root / "status.json", dict(schema=SCHEMA, status="complete", workflow_complete=True,
        manifest_sha256=_sha(root / "manifest.json"), equilibrium_certified=False))
    return root, manifest, originals


def update_manifest(source, manifest):
    plan = json.loads((source / "plan.json").read_text()); plan["sampler"] = manifest.get("sampler")
    _json(source / "plan.json", plan); manifest["plan_sha256"] = _sha(source / "plan.json")
    _json(source / "manifest.json", manifest)
    status = json.loads((source / "status.json").read_text()); status["manifest_sha256"] = _sha(source / "manifest.json")
    _json(source / "status.json", status)


def test_formal_configuration_is_default_unsupervised_3d_cube():
    path = Path(__file__).resolve().parents[1] / "configs/spin_glass3d_b2_overlap_g1_seed91.json"
    cfg = RunConfig.load(path)
    assert (cfg.model.dimension, cfg.model.width, cfg.model.block_factor, cfg.model.theory_dim) == (3, 2, 2, 1)
    assert cfg.model.hidden == 96 and not cfg.model.fixed_kernel
    assert cfg.model.kernel_initialization == "random" and cfg.model.kernel_sampling == "bernoulli"
    assert cfg.model.affine_reconstruction
    assert cfg.training.epochs == 6000 and cfg.training.learning_rate == .001
    assert tuple(cfg.training.train_modules) == ("Q", "C", "R") and cfg.training.gradient_mode == "qc_r_separate"
    assert cfg.loss.closure_scale_weights is None and cfg.evaluation.samples == 64
    assert cfg.loss.weights.anchor == 2 and cfg.loss.kernel_sign_weight == 1


@pytest.mark.parametrize("samples", (8, 512))
def test_adapter_pair01_only_and_entire_disorder_split(tmp_path, samples):
    source, _, originals = synthetic_source(tmp_path, samples=samples, side=4)
    output = tmp_path / "cache"; metadata = prepare_overlap_cache(source, output)
    train, test = load_ensembles(output / "ensembles.pt")
    assert train.configurations.shape == (6, samples, 4, 4, 4, 1)
    assert test.configurations.shape == (3, samples, 4, 4, 4, 1)
    assert train.couplings is None and test.couplings is None
    assert train.group_ids.tolist() == [[0, t] for t in range(3)] + [[2, t] for t in range(3)]
    for ds in (train, test):
        for (j, t), actual in zip(ds.group_ids.tolist(), ds.configurations):
            np.testing.assert_array_equal(actual[..., 0], originals[j][t, 0] * originals[j][t, 1])
    assert metadata["training_ensemble_count_per_disorder_and_temperature"] == 1
    assert metadata["real_replicas_coupled_by_icm"] and metadata["replica_pairs_are_not_independent"]
    assert not metadata["source_thermal_independence_certified"]
    assert not metadata["labels_used_in_training"] and not metadata["equilibrium_certified"]
    validate_metadata([("train", train), ("test", test)], metadata)
    assert json.loads((output / "status.json").read_text())["workflow_complete"]
    with pytest.raises(FileExistsError):
        prepare_overlap_cache(source, output)


@pytest.mark.parametrize("bad", ("ordinary_pt", "2d_receipt", "cold", "independent", "equilibrium", "field", "fingerprint", "samples_odd", "gaussian", "folded_signs"))
def test_adapter_rejects_wrong_or_unverified_source(tmp_path, bad):
    source, manifest, _ = synthetic_source(tmp_path, samples=5 if bad == "samples_odd" else 8)
    if bad == "ordinary_pt": manifest["sampler"]["algorithm"] = "pt"
    elif bad == "2d_receipt": manifest["sampler"]["schema"] = "houdayer_parallel_tempering_v1"
    elif bad == "cold": manifest["sampler"]["initial"] = "cold"
    elif bad == "independent": manifest["sampler"]["real_replicas_coupled"] = False
    elif bad == "equilibrium": manifest["equilibrium_certified"] = True
    elif bad == "field": manifest["options"]["field"] = .1
    elif bad == "fingerprint": manifest["sampler"]["kernel_source_sha256"] = "x"
    elif bad == "gaussian": manifest["options"]["distribution"] = "gaussian"
    elif bad == "folded_signs": manifest["sampler"]["saved_overlap_signs_unfolded"] = False
    update_manifest(source, manifest)
    with pytest.raises(ValueError): prepare_overlap_cache(source, tmp_path / "cache")
    assert not (tmp_path / "cache").exists()


def test_3d_plaquette_planes_are_all_included():
    bonds = np.ones((3, 4, 4, 4)); bonds[0, 0, 0, 0] = -1
    # One negative edge frustrates four plaquettes out of 3*64.
    assert _plaquette_frustration(bonds) == (4 / 192, 0.)


def fake_arrays():
    arrays = {}
    for split, j in (("train", 0), ("test", 1)):
        # Nonmonotone beta->g gives two roots; an average-based inverse would hide them.
        arrays[f"group_ids_{split}"] = np.array([[j, 0], [j, 1], [j, 2]])
        arrays[f"temperatures_{split}"] = np.array([2., 1., .5])
        arrays[f"inverse_temperatures_{split}"] = np.array([.5, 1., 2.])
        arrays[f"direct_{split}"] = np.array([[[-1.], [-.5]], [[1.], [.5]], [[-1.], [-.5]]])
    return arrays


def test_root_inverse_preserves_nonmonotone_beta_branches_per_J_and_scale():
    class Model:
        def advance_theory(self, x): return 2 * x
    arrays = fake_arrays(); grid = np.linspace(-1, 1, 101)[:, None]
    result = root_candidates(Model(), grid, 2 * grid, arrays, "cpu")
    assert result["thermal_nu"] is None and len(result["candidates"]) == 1
    root = result["candidates"][0]
    assert root["g"] == 0. and root["R_derivative"] == 2. and root["candidate_nu"] == 1.
    assert root["thermal_identification_certified"] is False
    assert "k0_interpolated_K" not in root
    assert len(root["inverse_temperature_branches"]) == 4
    for branch in root["inverse_temperature_branches"]:
        assert not branch["strictly_monotone"]
        assert [row["inverse_temperature"] for row in branch["all_piecewise_linear_crossings"]] == [.75, 1.5]


def test_flat_inverse_branch_not_returned_as_unique_temperature():
    arrays = fake_arrays()
    arrays["direct_train"][:, 0] = 0
    first = inverse_temperature_branches(arrays, 0)[0]
    assert len(first["all_piecewise_linear_crossings"]) == 2
    assert all(row["ambiguous_flat_segment"] for row in first["all_piecewise_linear_crossings"])


def test_frozen_collection_same_C_full_half_and_first_sample_subset(tmp_path):
    source, _, _ = synthetic_source(tmp_path)
    metadata = prepare_overlap_cache(source, tmp_path / "cache")
    train, test = load_ensembles(tmp_path / "cache/ensembles.pt")
    model = build_model(ModelConfig(dimension=3, width=2, hidden=4, theory_dim=1), 1).eval().requires_grad_(False)
    before = _model_state_sha256(model)
    arrays, contract = collect(model, [("train", train), ("test", test)], metadata, "cpu", 4, 2, 1)
    assert arrays["direct_draws_train"].shape == (6, 2, 2, 1)
    assert arrays["motif_target_draws_train"].shape == (6, 2, 2, 256)
    assert contract["samples"] == 4 and "first requested" in contract["source_sample_selection"]
    np.testing.assert_allclose(arrays["Q_full_minus_half_draws_test"],
        arrays["full_ensemble_direct_draws_test"] - arrays["direct_draws_test"])
    assert _model_state_sha256(model) == before and all(p.grad is None for p in model.parameters())
    metrics = compute_closure_metrics(arrays)
    assert metrics["Cmean"]["train"]["independent_disorders"] == 2
    assert metrics["Cmean"]["test"]["independent_disorders"] == 1
    coordinates = coordinate_diagnostics(arrays)
    assert len(coordinates["train"]["by_temperature"]) == 3
    with pytest.raises(ValueError, match="exceeds saved"):
        collect(model, [("train", train), ("test", test)], metadata, "cpu", 10, 1, 1)


def test_mi_null_selection_and_matching_E_not_theory_selected(tmp_path):
    source, _, _ = synthetic_source(tmp_path)
    metadata = prepare_overlap_cache(source, tmp_path / "cache")
    train, _ = load_ensembles(tmp_path / "cache/ensembles.pt")
    model = build_model(ModelConfig(dimension=3, width=2, hidden=4, theory_dim=1), 1).eval().requires_grad_(False)
    null = mi_null_diagnostic(model, train, metadata, "cpu")
    assert null["TRAIN_ensemble_indices"] == [0, 1, 2]
    assert [r["temperature_index"] for r in null["evaluated_cells"]] == [0, 1, 2]
    assert all(r["disorder_id"] == 0 for r in null["evaluated_cells"])
    assert null["samples_per_overlap_ensemble"] == 8 and null["translations_per_configuration"] == 512
    assert null["observation_counts_by_cell"] == [8 * 512] * 3
    assert null["diagnostics"]["observed_environment_states_by_ensemble"] == null["shuffled_diagnostics"]["observed_environment_states_by_ensemble"]
    assert len(null["learned_C_MI_bits"]) == len(null["shuffled_V_MI_bits"]) == 3


def test_metadata_rejects_wrong_pair_or_disorder_label_leak(tmp_path):
    source, _, _ = synthetic_source(tmp_path)
    metadata = prepare_overlap_cache(source, tmp_path / "cache")
    train, test = load_ensembles(tmp_path / "cache/ensembles.pt")
    bad = copy.deepcopy(metadata); bad["test"][0]["overlap_replicas"] = [2, 3]
    with pytest.raises(ValueError, match="metadata mismatch"):
        validate_metadata([("train", train), ("test", test)], bad)
