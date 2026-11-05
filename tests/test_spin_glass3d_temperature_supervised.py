"""Synthetic-only checks for the explicit 3D raw-temperature Q ablation.

No physical sampling, remote execution, or historical model/data is used here.
"""
from dataclasses import asdict
import json

import numpy as np
import pytest
import torch

from experiments import rsmi_separate_qcr as driver
from recurrent_rg.checkpoints import build_model
from recurrent_rg.config import ModelConfig, RunConfig
from recurrent_rg.data.ensembles import EnsembleDataset, load_ensembles
from recurrent_rg.data.statistics import cube_distribution
from recurrent_rg.losses import qcr_objective
from scripts.prepare_spin_glass3d_qcr import SAMPLER_SCHEMA
from scripts.run_spin_glass2d import sha, write_json


@pytest.fixture(autouse=True)
def one_thread():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


def source_fixture(tmp_path, *, side=4, samples=8):
    """Original label-free, whole-J held-out cache with identical T grids.

    Reverse row order deliberately: labels must follow group metadata, not an
    assumed sorted temperature grid. The fields are synthetic random +/-1.
    """
    root = tmp_path / "source"
    (root / "data").mkdir(parents=True)
    rng = torch.Generator().manual_seed(20261020)
    temperatures = [.6, 1.1, 2.0]
    datasets, rows = {}, {}
    for split, disorders in (("train", [2, 0]), ("test", [1])):
        fields, groups, records = [], [], []
        for disorder in disorders:
            for ti in reversed(range(len(temperatures))):
                field = (2 * torch.randint(2, (samples, side, side, side, 1), generator=rng) - 1).to(torch.int8)
                fields.append(field)
                groups.append([disorder, ti])
                records.append(dict(disorder_id=disorder, temperature_index=ti,
                    temperature=temperatures[ti], bond_sha256=str(disorder) * 64,
                    samples=samples, dimension=3, lattice_shape=[side] * 3,
                    sites=side ** 3, overlap_replicas=[0, 1], equilibrium_certified=False))
        datasets[split] = EnsembleDataset(torch.stack(fields), couplings=None,
            group_ids=torch.tensor(groups, dtype=torch.long)).validate()
        rows[split] = records
    cache = root / "data/ensembles.pt"
    torch.save(dict(schema_version=1, **{name: asdict(ds) for name, ds in datasets.items()}), cache)
    metadata = dict(schema_version=1, dimension=3, lattice_shape=[side] * 3,
        overlap_pair=[0, 1], labels_used_in_training=False, couplings_in_training_cache=False,
        equilibrium_certified=False, source_thermal_independence_certified=False,
        replica_pairs_are_not_independent=True,
        training_ensemble_count_per_disorder_and_temperature=1,
        sampler=dict(schema=SAMPLER_SCHEMA),
        split_unit="entire quenched disorder, all temperatures and replicas",
        cache_sha256=sha(cache), **rows)
    write_json(root / "data/evaluation_metadata.json", metadata)
    write_json(root / "config.json", dict(data=dict(cache=str(cache))))
    write_json(root / "job.json", dict(status="complete", workflow_complete=True))
    return root, cache, metadata, datasets


def save_metadata(root, metadata):
    path = root / "data/evaluation_metadata.json"
    # Deliberately permits a NaN fixture to exercise the adapter's fail-closed
    # reader; production write_json correctly forbids non-JSON numbers.
    path.write_text(json.dumps(metadata))
    return path


def test_raw_temperature_adapter_preserves_all_fields_groups_and_original_split(tmp_path):
    from scripts.prepare_spin_glass3d_temperature_supervised import temperature_supervised_datasets

    root, cache, old, original = source_fixture(tmp_path)
    before = {path: sha(path) for path in (cache, root / "data/evaluation_metadata.json")}
    datasets, metadata = temperature_supervised_datasets(cache, root / "data/evaluation_metadata.json")
    assert metadata["labels_used_in_training"] is True
    assert metadata["couplings_in_training_cache"] is True
    assert metadata["split_unit"] == old["split_unit"]
    assert metadata["source_cache_sha256"] == old["cache_sha256"]
    assert metadata["source_metadata_sha256"] == before[root / "data/evaluation_metadata.json"]
    assert metadata["physical_sampling_updates"] == 0
    assert metadata["equilibrium_certified"] is False
    contract = metadata["temperature_supervision"]
    assert contract["weight"] == 10.
    assert contract["scale"] == 0 and contract["target"] == "raw_temperature_T"
    assert contract["estimator"] == "Q(full microscopic batch)"
    assert contract["fitting_split"] == "TRAIN only" and contract["test_labels_evaluation_only"] is True
    for split in ("train", "test"):
        torch.testing.assert_close(datasets[split].configurations, original[split].configurations, atol=0, rtol=0)
        torch.testing.assert_close(datasets[split].group_ids, original[split].group_ids, atol=0, rtol=0)
        assert metadata[split] == old[split]
        expected = torch.tensor([[row["temperature"]] for row in old[split]], dtype=torch.float64)
        assert datasets[split].couplings.dtype == torch.float64
        torch.testing.assert_close(datasets[split].couplings, expected, atol=0, rtol=0)
    assert set(datasets["train"].couplings[:, 0].tolist()) == set(datasets["test"].couplings[:, 0].tolist())
    assert set(datasets["train"].group_ids[:, 0].tolist()).isdisjoint(datasets["test"].group_ids[:, 0].tolist())
    assert {path: sha(path) for path in before} == before


def test_prepare_new_supervised_cache_is_loadable_with_overlapping_T_but_disjoint_J(tmp_path):
    from scripts.prepare_spin_glass3d_temperature_supervised import prepare
    from scripts.report_spin_glass3d_temperature_supervised import validate_supervised_metadata

    root, cache, _, originals = source_fixture(tmp_path)
    watched = [cache, root / "data/evaluation_metadata.json", root / "config.json", root / "job.json"]
    before = {path: sha(path) for path in watched}
    output = tmp_path / "supervised"
    metadata = prepare(root, output)
    train, test = load_ensembles(output / "ensembles.pt")
    payload = torch.load(output / "ensembles.pt", map_location="cpu", weights_only=True)
    contract = payload["split_contract"]
    assert contract["kind"] == "whole_disorder_temperature_supervision"
    assert contract["group_columns"] == ["disorder_id", "temperature_index"]
    assert contract["supervision_columns"] == ["T"]
    assert contract["bond_sha256_by_disorder"] == {str(j): str(j) * 64 for j in range(3)}
    assert metadata["cache_sha256"] == sha(output / "ensembles.pt")
    assert json.loads((output / "evaluation_metadata.json").read_text()) == metadata
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "complete" and status["workflow_complete"] is True
    assert status["labels_used_in_training"] is True and status["returncode"] == 0
    assert status["physical_sampling_updates"] == 0
    views, validation_metadata = validate_supervised_metadata([("train", train), ("test", test)], metadata)
    for split, ds in views:
        assert ds.couplings is None
        torch.testing.assert_close(ds.configurations, originals[split].configurations, atol=0, rtol=0)
    # The label-free view is an internal report compatibility layer, never the
    # published training contract or a mutation of the saved source metadata.
    assert validation_metadata["labels_used_in_training"] is False
    assert metadata["labels_used_in_training"] is True
    assert train.couplings is not None and test.couplings is not None
    with pytest.raises(FileExistsError):
        prepare(root, output)
    assert {path: sha(path) for path in watched} == before


def test_prepare_refuses_existing_or_dangling_output_and_incomplete_source(tmp_path):
    from scripts.prepare_spin_glass3d_temperature_supervised import prepare

    root, _, _, _ = source_fixture(tmp_path)
    target = tmp_path / "absent_target"
    symlink = tmp_path / "output_symlink"
    symlink.symlink_to(target, target_is_directory=True)
    with pytest.raises(FileExistsError):
        prepare(root, symlink)
    assert symlink.is_symlink() and not target.exists()
    write_json(root / "job.json", dict(status="running", workflow_complete=False))
    with pytest.raises(ValueError, match="complete"):
        prepare(root, tmp_path / "incomplete_output")
    assert not (tmp_path / "incomplete_output").exists()


@pytest.mark.parametrize("bad", ["fingerprint", "nonpositive_T", "nonfinite_T", "wrong_identity", "wrong_pair", "overlap_J", "overlap_bond"])
def test_temperature_adapter_rejects_bad_source_before_writing(tmp_path, bad):
    from scripts.prepare_spin_glass3d_temperature_supervised import prepare

    root, cache, metadata, _ = source_fixture(tmp_path)
    if bad == "fingerprint":
        metadata["cache_sha256"] = "f" * 64
    elif bad == "nonpositive_T":
        metadata["train"][0]["temperature"] = 0.
    elif bad == "nonfinite_T":
        metadata["train"][0]["temperature"] = float("nan")
    elif bad == "wrong_identity":
        metadata["train"][0]["temperature_index"] = 17
    elif bad == "wrong_pair":
        metadata["train"][0]["overlap_replicas"] = [2, 3]
    else:
        payload = torch.load(cache, map_location="cpu", weights_only=True)
        if bad == "overlap_J":
            payload["test"]["group_ids"][:, 0] = 0
            for row in metadata["test"]:
                row["disorder_id"] = 0
                row["bond_sha256"] = "0" * 64
        else:
            for row in metadata["test"]:
                row["bond_sha256"] = "0" * 64
        torch.save(payload, cache)
        metadata["cache_sha256"] = sha(cache)
    save_metadata(root, metadata)
    with pytest.raises(ValueError):
        prepare(root, tmp_path / "bad_output")
    assert not (tmp_path / "bad_output").exists()


@pytest.mark.parametrize("bad", ["missing", "flat", "two_columns", "wrong_T", "nonfinite", "nonpositive",
                                 "false_label_contract", "coarse_contract", "bond_leak"])
def test_supervised_report_rejects_malformed_targets_and_contracts(tmp_path, bad):
    from scripts.prepare_spin_glass3d_temperature_supervised import temperature_supervised_datasets
    from scripts.report_spin_glass3d_temperature_supervised import validate_supervised_metadata

    source, cache, _, _ = source_fixture(tmp_path)
    datasets, metadata = temperature_supervised_datasets(cache, source / "data/evaluation_metadata.json")
    if bad == "missing":
        datasets["train"].couplings = None
    elif bad == "flat":
        datasets["train"].couplings = datasets["train"].couplings[:, 0]
    elif bad == "two_columns":
        datasets["train"].couplings = datasets["train"].couplings.repeat(1, 2)
    elif bad == "wrong_T":
        datasets["train"].couplings[0, 0] += .1
    elif bad == "nonfinite":
        datasets["train"].couplings[0, 0] = float("inf")
    elif bad == "nonpositive":
        datasets["train"].couplings[0, 0] = 0.
    elif bad == "false_label_contract":
        metadata["labels_used_in_training"] = False
    elif bad == "coarse_contract":
        metadata["temperature_supervision"]["scale"] = 1
    elif bad == "bond_leak":
        for row in metadata["test"]:
            row["bond_sha256"] = "0" * 64
        metadata["split_contract"]["bond_sha256_by_disorder"]["1"] = "0" * 64
    with pytest.raises(ValueError):
        validate_supervised_metadata(list(datasets.items()), metadata)


def supervision_arrays():
    """Full Q fits raw T exactly while the half-Q diagnostic is offset by 2.

    Coarse values intentionally bear no numerical relation to bare T. They must
    neither enter the temperature objective nor be reported as coarse T errors.
    """
    arrays = {}
    for split, disorders in (("train", [0, 2]), ("test", [1])):
        groups = np.array([[j, ti] for j in disorders for ti in range(3)])
        temperatures = np.tile([.6, 1.1, 2.], len(disorders))
        full = np.zeros((len(groups), 3, 1))
        full[:, 0, 0] = temperatures
        full[:, 1:, 0] = [71., -53.]
        half = full.copy()
        half[:, 0, 0] += 2
        arrays[f"full_ensemble_direct_{split}"] = full
        arrays[f"direct_{split}"] = half
        arrays[f"temperatures_{split}"] = temperatures
        arrays[f"group_ids_{split}"] = groups
    return arrays


def test_report_T_regression_primary_estimator_is_full_micro_and_never_coarse():
    from scripts.report_spin_glass3d_temperature_supervised import temperature_supervision_metrics

    arrays = supervision_arrays()
    before = {key: value.copy() for key, value in arrays.items()}
    metrics = temperature_supervision_metrics(arrays)
    assert metrics["labels_used_in_training"] is True
    assert metrics["weight"] == 10. and metrics["scale"] == 0
    assert metrics["target"] == "raw_temperature_T"
    assert "full_ensemble_direct" in metrics["primary_estimator"]
    assert metrics["coarse_temperature_error"] is None
    for split, disorder_count in (("train", 2), ("test", 1)):
        full, half = (metrics[name][split] for name in ("full_ensemble_micro", "half_subset_micro"))
        assert full["overall"]["mse"] == full["overall"]["rmse"] == full["overall"]["mae"] == 0.
        assert half["overall"]["mse"] == pytest.approx(4.)
        assert half["overall"]["rmse"] == half["overall"]["mae"] == pytest.approx(2.)
        assert len(full["per_disorder"]) == disorder_count
        assert all(row["rmse"] == 0. for row in full["per_disorder"])
        assert len(full["by_temperature"]) == 3
        assert all(row["disorder_population_std"] == 0. for row in full["by_temperature"])
    # Changing coarse coordinates cannot change any temperature-fit metric.
    for split in ("train", "test"):
        for prefix in ("", "full_ensemble_"):
            arrays[f"{prefix}direct_{split}"][:, 1:] *= -1000.
    assert temperature_supervision_metrics(arrays) == metrics
    for key in before:
        if "direct" not in key:
            np.testing.assert_array_equal(arrays[key], before[key])


@pytest.mark.parametrize("bad", ["nonfinite_prediction", "nonfinite_T", "nonpositive_T", "wrong_count", "wrong_dim", "wrong_groups"])
def test_report_T_regression_rejects_malformed_saved_arrays(bad):
    from scripts.report_spin_glass3d_temperature_supervised import temperature_supervision_metrics

    arrays = supervision_arrays()
    if bad == "nonfinite_prediction":
        arrays["full_ensemble_direct_test"][0, 0, 0] = np.nan
    elif bad == "nonfinite_T":
        arrays["temperatures_test"][0] = np.inf
    elif bad == "nonpositive_T":
        arrays["temperatures_test"][0] = -1.
    elif bad == "wrong_count":
        arrays["temperatures_test"] = arrays["temperatures_test"][:-1]
    elif bad == "wrong_dim":
        arrays["full_ensemble_direct_test"] = arrays["full_ensemble_direct_test"][..., 0]
    elif bad == "wrong_groups":
        arrays["group_ids_test"] = arrays["group_ids_test"][:, :1]
    with pytest.raises(ValueError):
        temperature_supervision_metrics(arrays)


def test_3d_temperature_supervision_has_nonzero_Q_but_no_C_or_R_gradient():
    torch.manual_seed(411)
    model = build_model(ModelConfig(dimension=3, theory_dim=1, hidden=5, width=2,
        block_factor=2, affine_reconstruction=True, kernel_sampling="bernoulli"), 1)
    model.detach_kernel_representation = True
    batch = (2 * torch.randint(2, (3, 4, 4, 4, 4, 1)) - 1).double()
    temperatures = torch.tensor([[.6], [1.1], [2.]], dtype=torch.float64, requires_grad=True)
    base, terms = qcr_objective(model, batch, pattern_fn=cube_distribution)
    closure = 5 * (terms["commutator"] + terms["rollout"])
    supervised = driver.microscopic_coupling_loss(model.theory_encoder, batch, temperatures)
    torch.testing.assert_close(supervised, (model.theory_encoder(batch) - temperatures.detach()).square().mean())
    q, c, r = (list(part.parameters()) for part in (model.theory_encoder, model.rg_kernel, model.theory_rg))
    supervision_gradients = torch.autograd.grad(supervised, q + c + r, retain_graph=True, allow_unused=True)
    assert any(g is not None and bool(g.abs().sum() > 0) for g in supervision_gradients[:len(q)])
    assert all(g is None for g in supervision_gradients[len(q):])
    c_objective = sum(parameter.square().sum() for parameter in c)
    expected_q = torch.autograd.grad(base - closure + 10 * supervised, q, retain_graph=True, allow_unused=True)
    expected_c = torch.autograd.grad(c_objective, c, retain_graph=True)
    expected_r = torch.autograd.grad(closure, r, retain_graph=True)
    driver.independent_component_gradients(base + 10 * supervised, closure, c_objective, q, c, r)
    for parameters, gradients in ((q, expected_q), (c, expected_c), (r, expected_r)):
        for parameter, gradient in zip(parameters, gradients):
            if gradient is None:
                assert parameter.grad is None
            else:
                torch.testing.assert_close(parameter.grad, gradient)
    assert temperatures.grad is None


def test_3d_supervised_trainer_cpu_one_update_accepts_new_protocol(tmp_path, monkeypatch):
    from scripts.prepare_spin_glass3d_temperature_supervised import prepare

    source, source_cache, _, _ = source_fixture(tmp_path, side=8)
    original_sha = sha(source_cache)
    cache_root = tmp_path / "data"
    prepare(source, cache_root)
    cfg = RunConfig.from_dict(dict(schema_version=5,
        data=dict(cache=str(cache_root / "ensembles.pt"), physical_model="synthetic 3D EA overlap", coupling_names=["T"], spin_flip_augment=False),
        model=dict(dimension=3, theory_dim=1, hidden=5, width=2, block_factor=2, affine_reconstruction=True),
        loss=dict(kernel_sign_weight=1., kernel_noncollapse_mode="variance_floor_ensemble",
            kernel_variance_ratio=1., weights=dict(anchor=2., separation=10., coarse_consistency=5.,
                commutator=5., kernel_noncollapse=2.)),
        training=dict(max_rg_steps=1, epochs=1, system_batch=2, sample_batch=4,
            checkpoint_every=1, log_every=1, gradient_mode="qc_r_separate", threads=1),
        evaluation=dict(max_rg_steps=1, samples=4, systems=2)))
    config = tmp_path / "config.json"
    write_json(config, cfg.to_dict())
    output = tmp_path / "run"
    monkeypatch.setattr(driver, "evaluate_run", lambda *a, **k: pytest.fail("Explicit final report skip must be honored"))
    driver.main(["--config", str(config), "--output", str(output), "--device", "cpu",
                 "--q-coupling-weight", "10", "--skip-final-report"])
    manifest = json.loads((output / "manifest.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    assert manifest["labels_used_in_training"] is True
    assert manifest["q_supervision"]["coupling_names"] == ["T"]
    assert manifest["q_supervision"]["scale"] == 0
    assert summary["contract"]["labels_used_in_training"] is True
    assert summary["actual_epochs"] == 1 and summary["final_report_skipped"] is True
    assert summary["history"][0]["train_coupling_supervision"] >= 0
    assert summary["history"][0]["test_coupling_supervision"] >= 0
    assert (output / "checkpoint_last.pt").is_file() and (output / "qcr_model.pt").is_file()
    assert sha(source_cache) == original_sha and not (output / "report").exists()

    # Exercise the real collector and JSON report on this synthetic model. Only
    # plots and root diagnostics are stubbed; no MI/sampling pipeline is called.
    from scripts import report_spin_glass3d_temperature_supervised as reporter
    monkeypatch.setattr(reporter, "figures", lambda *a, **k: "synthetic-font-not-used")
    monkeypatch.setattr(reporter, "_supervision_figures", lambda *a, **k: [])
    monkeypatch.setattr(reporter, "root_candidates", lambda *a, **k: dict(candidates=[], thermal_nu=None))
    final = reporter.report(tmp_path, draws=1)
    saved_report = json.loads((tmp_path / "report/report.json").read_text())
    saved_status = json.loads((tmp_path / "report/status.json").read_text())
    assert final == saved_report
    assert saved_report["labels_used_in_training"] is saved_status["labels_used_in_training"] is True
    assert saved_report["temperature_supervision"]["labels_used_in_training"] is True
    assert saved_report["temperature_supervision"]["coarse_temperature_error"] is None
    assert saved_report["thermal_nu"] is None and saved_report["physical_sampling_updates"] == 0
    assert "显式温度监督消融" in saved_report["qualification"]
    assert "温度与随机键不监督模型" not in saved_report["qualification"]
    assert sha(source_cache) == original_sha
