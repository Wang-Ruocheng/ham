"""Fixed-J selection preserves saved fields and holds out whole temperatures."""
from dataclasses import asdict
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from recurrent_rg.data.ensembles import EnsembleDataset, load_ensembles
from scripts.prepare_spin_glass3d_fixed_j import SPLIT, prepare, select_fixed_disorder
from scripts.prepare_spin_glass3d_qcr import SAMPLER_SCHEMA
from scripts.report_spin_glass1d import compute_closure_metrics
from scripts.report_spin_glass3d import closure_metrics, validate_metadata
from scripts.run_spin_glass2d import sha, write_json


def source_fixture(tmp_path, *, samples=8):
    """Two original whole-J splits, reverse ordered to expose index assumptions."""
    root = tmp_path / "source"
    (root / "data").mkdir(parents=True)
    datasets, rows, originals = {}, {}, {}
    side = 2
    rng = torch.Generator().manual_seed(30620)
    for split, disorder in (("train", 0), ("test", 1)):
        fields, groups, records = [], [], []
        for temperature_index in reversed(range(20)):
            field = (torch.randint(0, 2, (samples, side, side, side, 1), generator=rng)
                     * 2 - 1).to(torch.int8)
            originals[disorder, temperature_index] = field.clone()
            fields.append(field)
            groups.append([disorder, temperature_index])
            records.append(dict(disorder_id=disorder, temperature_index=temperature_index,
                temperature=.5 + temperature_index * .15, bond_sha256=str(disorder) * 64,
                samples=samples, dimension=3, lattice_shape=[side] * 3, sites=side ** 3,
                overlap_replicas=[0, 1], equilibrium_certified=False))
        datasets[split] = EnsembleDataset(torch.stack(fields), couplings=None,
            group_ids=torch.tensor(groups, dtype=torch.long)).validate()
        rows[split] = records
    cache = root / "data/ensembles.pt"
    torch.save(dict(schema_version=1, **{split: asdict(ds) for split, ds in datasets.items()}), cache)
    metadata = dict(schema_version=1, dimension=3, lattice_shape=[side] * 3,
        overlap_pair=[0, 1], labels_used_in_training=False, equilibrium_certified=False,
        training_ensemble_count_per_disorder_and_temperature=1,
        sampler=dict(schema=SAMPLER_SCHEMA), split_unit="entire quenched disorder, all temperatures and replicas",
        source_thermal_independence_certified=False, replica_pairs_are_not_independent=True,
        cache_sha256=sha(cache), **rows)
    write_json(root / "data/evaluation_metadata.json", metadata)
    write_json(root / "config.json", dict(data=dict(cache=str(cache))))
    write_json(root / "job.json", dict(status="complete", workflow_complete=True))
    return root, cache, metadata, originals


def save_metadata(root, metadata):
    write_json(root / "data/evaluation_metadata.json", metadata)
    return root / "data/evaluation_metadata.json"


@pytest.mark.parametrize("disorder_id", (0, 1))
@pytest.mark.parametrize("samples", (8, 512))
def test_selector_uses_requested_J_from_either_source_split_and_preserves_all_samples(tmp_path, disorder_id, samples):
    root, cache, source_metadata, originals = source_fixture(tmp_path, samples=samples)
    datasets, metadata = select_fixed_disorder(cache, root / "data/evaluation_metadata.json", disorder_id)
    expected = {"train": [t for t in range(20) if t % 5 != 2], "test": [2, 7, 12, 17]}
    assert metadata["split_unit"] == SPLIT
    assert metadata["fixed_disorder_id"] == disorder_id
    assert metadata["fixed_bond_sha256"] == str(disorder_id) * 64
    assert metadata["independent_disorder_count"] == 1
    assert metadata["physical_sampling_updates"] == 0
    assert metadata["source_cache_sha256"] == source_metadata["cache_sha256"]
    assert metadata["source_metadata_sha256"] == sha(root / "data/evaluation_metadata.json")
    assert metadata["labels_used_in_training"] is False
    assert metadata["couplings_in_training_cache"] is False
    assert metadata["equilibrium_certified"] is False
    assert metadata["source_thermal_independence_certified"] is False
    assert metadata["replica_pairs_are_not_independent"] is True
    for split, ds in datasets.items():
        assert ds.couplings is None
        assert ds.configurations.shape == (len(expected[split]), samples, 2, 2, 2, 1)
        assert ds.group_ids.tolist() == [[disorder_id, t] for t in expected[split]]
        assert [row["temperature_index"] for row in metadata[split]] == expected[split]
        for (j, t), actual in zip(ds.group_ids.tolist(), ds.configurations):
            torch.testing.assert_close(actual, originals[j, t], rtol=0, atol=0)
    assert set(expected["train"]).isdisjoint(expected["test"])
    validate_metadata(list(datasets.items()), metadata)


def test_prepare_writes_new_label_free_cache_and_does_not_touch_source_or_overwrite(tmp_path):
    root, cache, _, originals = source_fixture(tmp_path)
    watched = [cache, root / "data/evaluation_metadata.json", root / "config.json", root / "job.json"]
    before = {path: sha(path) for path in watched}
    output = tmp_path / "selected"
    metadata = prepare(root, output, disorder_id=0)
    train, test = load_ensembles(output / "ensembles.pt")
    validate_metadata([("train", train), ("test", test)], metadata)
    assert metadata["cache_sha256"] == sha(output / "ensembles.pt")
    assert json.loads((output / "evaluation_metadata.json").read_text()) == metadata
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "complete" and status["workflow_complete"] is True
    assert status["returncode"] == status["physical_sampling_updates"] == 0
    assert train.couplings is test.couplings is None
    assert train.group_ids[:, 0].unique().tolist() == test.group_ids[:, 0].unique().tolist() == [0]
    torch.testing.assert_close(test.configurations[0], originals[0, 2], rtol=0, atol=0)
    selected_sha = sha(output / "ensembles.pt")
    with pytest.raises(FileExistsError):
        prepare(root, output, disorder_id=1)
    assert sha(output / "ensembles.pt") == selected_sha
    assert {path: sha(path) for path in watched} == before


def test_prepare_rejects_dangling_output_symlink_without_creating_its_target(tmp_path):
    root, _, _, _ = source_fixture(tmp_path)
    target = tmp_path / "must_not_be_created"
    output = tmp_path / "selected_symlink"
    output.symlink_to(target, target_is_directory=True)
    with pytest.raises(FileExistsError):
        prepare(root, output)
    assert output.is_symlink()
    assert not target.exists()


def test_stage_commands_keep_default_RSMI_recipe_and_fresh_6000_updates(tmp_path):
    from experiments.rsmi_separate_qcr import build_parser, validate_recipe
    from recurrent_rg.config import RunConfig
    from scripts.run_spin_glass3d_fixed_j import DEVICE, OVERRIDES, stage_commands

    root, baseline = tmp_path / "new", tmp_path / "baseline"
    commands = stage_commands(root, baseline)
    assert "report_spin_glass1d.py" in OVERRIDES
    assert [name for name, _, _ in commands] == [
        "prepare_fixed_J", "full_shell_MI", "gpu_preflight", "train", "report"]
    cfg = RunConfig.load(Path(__file__).resolve().parents[1] / "configs/spin_glass3d_b2_overlap_g1_seed91.json")
    assert cfg.training.epochs == 6000 and cfg.loss.closure_scale_weights is None
    for name, command, gpu in commands:
        assert "--allow-nondefault-recipe" not in command
        assert "--allow-legacy-training" not in command
        assert "--resume-checkpoint" not in command and "--source" not in command
        if name in ("gpu_preflight", "train"):
            args = build_parser().parse_args(command[3:])
            validate_recipe(args, cfg)
            assert args.config == root / "config.json"
            assert args.device == DEVICE == "cuda:2" and gpu
            assert args.rsmi_weight == args.variance_weight == 2
            assert args.color_weight == args.variance_ratio == 1
            assert args.estimator == "histogram" and args.environment == "full_shell"
            assert args.independent_c and args.variance_per_ensemble and not args.mi_initialize
            assert args.epochs == (5 if name == "gpu_preflight" else None)
            assert args.output == root / ("gpu_preflight" if name == "gpu_preflight" else "run")
    report = next(command for name, command, _ in commands if name == "report")
    assert "--skip-mi-diagnostic" not in report


@pytest.mark.parametrize("bad_id", (2, -1, True, 0.0, "0"))
def test_selector_rejects_unknown_or_noninteger_disorder(tmp_path, bad_id):
    root, cache, _, _ = source_fixture(tmp_path)
    with pytest.raises(ValueError):
        select_fixed_disorder(cache, root / "data/evaluation_metadata.json", bad_id)


def test_selector_rejects_wrong_cache_fingerprint_before_selection(tmp_path):
    root, cache, metadata, _ = source_fixture(tmp_path)
    metadata["cache_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="fingerprint"):
        select_fixed_disorder(cache, save_metadata(root, metadata), 0)


def test_selector_rejects_conflicting_bond_identity(tmp_path):
    root, cache, metadata, _ = source_fixture(tmp_path)
    metadata["train"][0]["bond_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="conflicting bond"):
        select_fixed_disorder(cache, save_metadata(root, metadata), 0)


def test_prepare_refuses_incomplete_source_without_creating_output(tmp_path):
    root, _, _, _ = source_fixture(tmp_path)
    write_json(root / "job.json", dict(status="running", workflow_complete=False))
    output = tmp_path / "selected"
    with pytest.raises(ValueError, match="must be complete"):
        prepare(root, output)
    assert not output.exists()


def test_original_report_validator_still_requires_whole_disorder_separation(tmp_path):
    root, cache, metadata, _ = source_fixture(tmp_path)
    original = load_ensembles(cache)
    validate_metadata(list(zip(("train", "test"), original)), metadata)
    datasets, fixed_metadata = select_fixed_disorder(cache, root / "data/evaluation_metadata.json", 0)
    fixed_metadata["split_unit"] = metadata["split_unit"]
    with pytest.raises(ValueError, match="leaked between splits"):
        validate_metadata(list(datasets.items()), fixed_metadata)


@pytest.mark.parametrize("missing", ("fixed_disorder_id", "fixed_bond_sha256"))
def test_fixed_J_report_requires_explicit_identity_declaration(tmp_path, missing):
    root, cache, _, _ = source_fixture(tmp_path)
    datasets, metadata = select_fixed_disorder(cache, root / "data/evaluation_metadata.json", 0)
    metadata.pop(missing)
    with pytest.raises(ValueError, match="same one declared J"):
        validate_metadata(list(datasets.items()), metadata)


def test_fixed_J_report_rejects_overlap_in_temperature_values(tmp_path):
    root, cache, _, _ = source_fixture(tmp_path)
    datasets, metadata = select_fixed_disorder(cache, root / "data/evaluation_metadata.json", 0)
    metadata["test"][0]["temperature"] = metadata["train"][0]["temperature"]
    with pytest.raises(ValueError, match="temperature values overlap"):
        validate_metadata(list(datasets.items()), metadata)


def test_fixed_J_report_rejects_overlapping_temperature_cells(tmp_path):
    root, cache, _, _ = source_fixture(tmp_path)
    datasets, metadata = select_fixed_disorder(cache, root / "data/evaluation_metadata.json", 0)
    datasets["test"].group_ids[0] = datasets["train"].group_ids[0].clone()
    metadata["test"][0] = copy.deepcopy(metadata["train"][0])
    with pytest.raises(ValueError, match="temperature cells overlap"):
        validate_metadata(list(datasets.items()), metadata)


def fixed_closure_fixture(tmp_path):
    """Saved-array algebra only: no model construction, inference or sampling."""
    root, cache, _, _ = source_fixture(tmp_path)
    datasets, metadata = select_fixed_disorder(cache, root / "data/evaluation_metadata.json", 0)
    arrays = {}
    for split, dataset in datasets.items():
        groups = dataset.group_ids.numpy().copy()
        direct = (.1 * groups[:, None, None, 1:2]
                  + np.array([-.03, .03])[None, :, None, None]
                  + .3 * np.arange(3)[None, None, :, None])
        mapped, rollout = direct.copy(), direct.copy()
        mapped[:, :, :-1] = direct[:, :, 1:] + np.array([.1, .2])[None, None, :, None]
        mapped[:, :, :-1] += np.array([-.04, .04])[None, :, None, None]
        rollout[:, :, 1:] += np.array([.3, .4])[None, None, :, None]
        rollout[:, :, 1:] += np.array([-.08, .08])[None, :, None, None]
        arrays.update({f"direct_draws_{split}": direct,
            f"R_g_draws_{split}": mapped, f"rollout_draws_{split}": rollout,
            f"group_ids_{split}": groups,
            f"bond_sha256_{split}": np.array([r["bond_sha256"] for r in metadata[split]], dtype="U64"),
            f"temperatures_{split}": np.array([r["temperature"] for r in metadata[split]])})
    return arrays, metadata


def test_fixed_J_closure_metrics_matches_hand_algebra_and_uses_TRAIN_normalization(tmp_path):
    arrays, metadata = fixed_closure_fixture(tmp_path)
    result = closure_metrics(arrays, metadata)
    assert result == compute_closure_metrics(arrays, split_metadata=metadata)
    assert result["split_unit"] == SPLIT and result["fixed_disorder_id"] == 0
    assert "not independent physical chains" in result["independent_unit"]
    assert result["confidence_intervals"] is None
    train_spread = np.std(arrays["group_ids_train"][:, 1] * .1, ddof=0)
    for mode, extra_variance in (("Cmean", 0.), ("per_C_draw", .03 ** 2)):
        expected_std = np.sqrt(train_spread ** 2 + extra_variance)
        np.testing.assert_allclose(result[mode]["train_target_std"], [expected_std] * 2)
        for split in ("train", "test"):
            for branch, expected, extra_error in (("one_step", [.1, .2], .04),
                                                   ("microscopic_rollout", [.3, .4], .08)):
                rmse = np.sqrt(np.square(expected) + (extra_error ** 2 if mode == "per_C_draw" else 0.))
                np.testing.assert_allclose(result[mode][split][branch]["rmse"], rmse)
                np.testing.assert_allclose(result[mode][split][branch]["nrmse"], rmse / expected_std)
            assert result[mode][split]["independent_disorders"] == 1
    with pytest.raises(ValueError, match="Whole disorders must be held out"):
        compute_closure_metrics(arrays)
    old_metadata = dict(metadata, split_unit="entire quenched disorder, all temperatures and replicas")
    with pytest.raises(ValueError, match="Whole disorders must be held out"):
        closure_metrics(arrays, old_metadata)


@pytest.mark.parametrize("bad", ("undeclared_J", "undeclared_bond", "wrong_J", "wrong_bond_array",
    "missing_bond_array", "missing_temperature_array", "overlapping_index", "overlapping_temperature",
    "row_identity", "row_temperature", "duplicate_cell", "unrecognized_contract"))
def test_fixed_J_closure_exception_is_fail_closed(tmp_path, bad):
    arrays, metadata = fixed_closure_fixture(tmp_path)
    if bad == "undeclared_J": metadata.pop("fixed_disorder_id")
    elif bad == "undeclared_bond": metadata.pop("fixed_bond_sha256")
    elif bad == "wrong_J": arrays["group_ids_test"][:, 0] = 1
    elif bad == "wrong_bond_array": arrays["bond_sha256_test"][0] = "f" * 64
    elif bad == "missing_bond_array": arrays.pop("bond_sha256_test")
    elif bad == "missing_temperature_array": arrays.pop("temperatures_test")
    elif bad == "overlapping_index":
        arrays["group_ids_test"][0, 1] = arrays["group_ids_train"][0, 1]
        metadata["test"][0]["temperature_index"] = int(arrays["group_ids_test"][0, 1])
    elif bad == "overlapping_temperature":
        arrays["temperatures_test"][0] = arrays["temperatures_train"][0]
        metadata["test"][0]["temperature"] = float(arrays["temperatures_test"][0])
    elif bad == "row_identity": metadata["test"][0]["temperature_index"] = 99
    elif bad == "row_temperature": metadata["test"][0]["temperature"] = .123
    elif bad == "duplicate_cell":
        arrays["group_ids_test"][0] = arrays["group_ids_test"][1]
        arrays["temperatures_test"][0] = arrays["temperatures_test"][1]
        metadata["test"][0] = copy.deepcopy(metadata["test"][1])
    elif bad == "unrecognized_contract": metadata["split_unit"] = "allow_same_J"
    with pytest.raises(ValueError):
        compute_closure_metrics(arrays, split_metadata=metadata)
