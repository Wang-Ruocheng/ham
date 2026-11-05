"""Synthetic sampling-report checks; no chain generation or model calls."""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.report_spin_glass_sampling_trial import (
    _finite_json, _sha, aggregate_observables, batch_means, compare_cases,
    describe_series, load_case, report,
)


def write_json(path, payload):
    path.write_text(json.dumps(payload, allow_nan=False))


def synthetic_trial(tmp_path, *, samples=512, constant=False):
    root = tmp_path / "trial"
    (root / "cases").mkdir(parents=True)
    rng = np.random.default_rng(812)
    cases = []
    temperatures = np.array([.5, 2.])
    for algorithm in ("pt", "houdayer_pt"):
        for initial in ("hot", "cold"):
            key = f"j000_{algorithm}_{initial}"
            declared = dict(key=key, algorithm=algorithm, initial=initial, disorder_id=0,
                expected_bond_sha256="a" * 64)
            cases.append(declared)
            spins = (np.ones((2, 4, samples, 3, 3), dtype=np.int8) if constant
                else rng.choice(np.array([-1, 1], np.int8), size=(2, 4, samples, 3, 3)))
            energies = (np.full((2, 4, samples), -9.) if constant
                else rng.normal(-9, 2, size=(2, 4, samples)))
            path = root / "cases" / f"{key}.npz"
            np.savez(path, configurations=spins, energies=energies, temperatures=temperatures,
                round_trips=np.zeros((4, 2), np.int64),
                swap_attempts=np.zeros((4, 1), np.int64), swap_accepts=np.zeros((4, 1), np.int64),
                trace_sweeps=np.array([0, 16, 32, 48], np.int64),
                trace_energy_density=np.full((2, 4), -1.), trace_q2=np.ones((2, 4)),
                cluster_size_histogram=np.array([[2] + [0] * 8 + [1], [0] * 10], np.int64))
            receipt = dict(**declared, status="complete", returncode=0, source_unchanged=True,
                replicas=4, samples=samples, burn_in=16, spacing=2,
                sampling_wall_seconds=10., jit_execution_check_wall_seconds=99.,
                npz_path=f"cases/{key}.npz", npz_sha256=_sha(path), npz_bytes=path.stat().st_size,
                bond_sha256="a" * 64, training_updates=0)
            write_json(root / "cases" / f"{key}.json", receipt)
    plan = dict(schema_version=1, shape=[3, 3], temperatures=temperatures.tolist(),
        replicas=4, samples=samples, burn_in=16, spacing=2, cases=cases,
        execution_smoke=True, scientific_result=False)
    write_json(root / "plan.json", plan)
    return root, plan


def test_pairs_averaged_per_time_before_diagnostics_not_independent_samples():
    rng = np.random.default_rng(42)
    spins = rng.choice(np.array([-1, 1], np.int8), size=(2, 4, 256, 3, 3))
    energies = rng.normal(size=(2, 4, 256))
    values = aggregate_observables(spins, energies)
    expected_q2, expected_q4, expected_corr = [], [], []
    for a in range(4):
        for b in range(a + 1, 4):
            overlap = spins[:, a].astype(float) * spins[:, b]
            q = overlap.mean(axis=(2, 3))
            expected_q2.append(q ** 2)
            expected_q4.append(q ** 4)
            expected_corr.append(.5 * (
                (overlap * np.roll(overlap, 1, axis=2)).mean(axis=(2, 3))
                + (overlap * np.roll(overlap, 1, axis=3)).mean(axis=(2, 3))))
    assert len(expected_q2) == 6
    np.testing.assert_allclose(values["q_squared"], np.mean(expected_q2, axis=0))
    np.testing.assert_allclose(values["q_fourth"], np.mean(expected_q4, axis=0))
    np.testing.assert_allclose(values["overlap_nearest_neighbor"], np.mean(expected_corr, axis=0), atol=1e-15)
    np.testing.assert_allclose(values["energy_density"], energies.mean(axis=1) / 9)
    assert values["q_squared"].shape == (2, 256)
    details = describe_series(values["q_squared"][0], spacing=16, wall_seconds=10.)
    assert details["samples"] == 256  # not 256 * six shared pairs
    assert details["ess"] <= 256
    assert details["descriptive_ess_per_sampling_wall_second"] == details["ess"] / 10
    assert details["iat_sweeps"] == details["iat_saved_intervals"] * 16
    assert details["batch_means"]["32"]["blocks"] == 8


def test_batch_means_minimum_blocks_and_no_false_resolution_for_constant():
    short = batch_means(np.arange(255.), 32)
    assert short["blocks"] == 7 and short["mcse"] is None
    values = np.arange(257.)
    result = batch_means(values, 32)
    means = values[:256].reshape(8, 32).mean(axis=1)
    assert result["discarded_tail_samples"] == 1
    assert result["mcse"] == pytest.approx(np.std(means, ddof=1) / np.sqrt(8))
    constant = describe_series(np.ones(512), spacing=2, wall_seconds=10)
    assert constant["ess"] is None and constant["chosen_mcse"] == 0
    assert constant["first_second_half_drift"]["available"]
    assert constant["first_second_half_drift"]["descriptive_z"] is None


def test_first_second_half_drift_is_descriptive_and_short_half_unresolved():
    rng = np.random.default_rng(7)
    values = rng.normal(size=512)
    values[256:] += 1
    result = describe_series(values, spacing=1, wall_seconds=2)
    drift = result["first_second_half_drift"]
    assert drift["available"]
    assert drift["difference"] == pytest.approx(values[256:].mean() - values[:256].mean())
    assert drift["descriptive_z"] > 3
    short = describe_series(values[:256], spacing=1, wall_seconds=2)
    assert short["first_second_half_drift"]["combined_mcse"] is None


def test_finite_json_handles_nested_numpy_nan_without_invalid_json():
    result = _finite_json({"a": np.array([np.nan, np.inf, -np.inf]), "b": np.float64(np.nan), "c": 1.})
    assert result == {"a": [None, None, None], "b": None, "c": 1.}
    assert "NaN" not in json.dumps(result, allow_nan=False)


def test_report_public_arrays_no_spins_and_all_json_strict(tmp_path, monkeypatch):
    root, plan = synthetic_trial(tmp_path, constant=True)
    def fake_figure(cases, comparison, arrays, output, *, font=None):
        (output / "sampling_comparison.png").write_bytes(b"synthetic figure")
        return dict(path="synthetic-font", sha256="b" * 64)
    monkeypatch.setattr("scripts.report_spin_glass_sampling_trial.make_figure", fake_figure)
    result = report(root)
    assert result["status"] == "complete" and result["scientific_result"] is False
    diagnostics = json.loads((root / "report/diagnostics.json").read_text())
    assert diagnostics["independent_disorders"] == 1
    assert diagnostics["case_count"] == 4 and not diagnostics["equilibrium_certified"]
    for case in diagnostics["cases"]:
        assert case["pairs"] == 6 and case["scalar_time_series_length"] == 512
        assert case["pair_combinations_are_not_independent"]
        assert case["temperatures"][0]["energy_density"]["mean"] == -1
        assert case["temperatures"][0]["q_squared"]["ess"] is None
        assert case["cluster"]["empty_cluster_counts"] == [2, 0]
        assert case["cluster"]["full_lattice_cluster_counts"] == [1, 0]
        assert case["swap_acceptance"] == [[None], [None], [None], [None]]
        assert case["jit_execution_check_wall_seconds"] == 99
    comparison = json.loads((root / "report/comparison.json").read_text())
    assert len(comparison["hot_cold"]) == 4
    assert len(comparison["algorithm_comparison"]) == 2
    assert not comparison["joint_confidence_claim"]
    for path in (root / "report").glob("*.json"):
        assert "NaN" not in path.read_text() and "Infinity" not in path.read_text()
        json.loads(path.read_text(), parse_constant=lambda value: pytest.fail(value))
    with np.load(root / "report/series.npz", allow_pickle=False) as series:
        assert not any(word in name for name in series.files for word in ("configurations", "bonds", "spins"))
        assert series["j000_pt_hot__q_squared"].shape == (2, 512)
        assert all(series[name].dtype.kind != "O" for name in series.files)
    for item in result["public_files"]:
        assert _sha(root / "report" / item["path"]) == item["sha256"]
    with pytest.raises(FileExistsError):
        report(root)


def test_comparison_uses_independent_start_mcse_quadrature(tmp_path):
    root, plan = synthetic_trial(tmp_path)
    cases = []
    for declared in plan["cases"]:
        receipt, observables, public, cluster, source = load_case(root, declared, plan)
        rows = []
        for t in range(2):
            rows.append(dict(temperature=plan["temperatures"][t],
                **{name: describe_series(value[t], spacing=2, wall_seconds=10)
                   for name, value in observables.items()}))
        cases.append(dict(**declared, temperatures=rows))
    comparison = compare_cases(cases)
    row = comparison["hot_cold"][0]["observables"]["q_squared"]
    hot, cold = cases[0]["temperatures"][0]["q_squared"], cases[1]["temperatures"][0]["q_squared"]
    assert row["difference"] == pytest.approx(cold["mean"] - hot["mean"])
    assert row["combined_mcse"] == pytest.approx(np.hypot(hot["chosen_mcse"], cold["chosen_mcse"]))


def test_actual_chinese_figure_and_font_provenance(tmp_path):
    # Synthetic arrays only. No physical sampling and no model is involved.
    from scripts.report_ising_zh import select_chinese_font
    try:
        font = select_chinese_font()
    except (FileNotFoundError, RuntimeError, ValueError):
        pytest.skip("No Chinese font installed on this test host")
    root, _ = synthetic_trial(tmp_path, samples=16)
    result = report(root, font=font)
    assert result["chinese_font"]["sha256"] == _sha(font)
    figure = root / "report/sampling_comparison.png"
    assert figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert figure.stat().st_size > 10_000


@pytest.mark.parametrize("bad", ("sha", "outside", "source", "identity", "replicas", "samples", "shape", "bond", "energy_nan"))
def test_case_validation_rejects_corruption_or_frozen_plan_mismatch(tmp_path, bad):
    root, plan = synthetic_trial(tmp_path, samples=16)
    declared = plan["cases"][0]
    receipt_path = root / "cases" / f"{declared['key']}.json"
    receipt = json.loads(receipt_path.read_text())
    if bad == "sha":
        receipt["npz_sha256"] = "f" * 64
    elif bad == "outside":
        receipt["npz_path"] = "../outside.npz"
    elif bad == "source":
        receipt["source_unchanged"] = False
    elif bad == "identity":
        receipt["algorithm"] = "houdayer_pt"
    elif bad == "replicas":
        receipt["replicas"] = plan["replicas"] = 3
    elif bad == "samples":
        receipt["samples"] = plan["samples"] = 15
    elif bad == "shape":
        plan["shape"] = [4, 4]
    elif bad == "bond":
        receipt["bond_sha256"] = "f" * 64
    elif bad == "energy_nan":
        path = root / receipt["npz_path"]
        with np.load(path, allow_pickle=False) as original:
            arrays = {name: original[name] for name in original.files}
        arrays["energies"][0, 0, 0] = np.nan
        np.savez(path, **arrays)
        receipt["npz_sha256"], receipt["npz_bytes"] = _sha(path), path.stat().st_size
    write_json(receipt_path, receipt)
    with pytest.raises(ValueError):
        load_case(root, declared, plan)


def test_report_rejects_conflicting_bonds_across_same_disorder(tmp_path, monkeypatch):
    root, plan = synthetic_trial(tmp_path, samples=16)
    declared = plan["cases"][1]
    declared["expected_bond_sha256"] = "b" * 64
    write_json(root / "plan.json", plan)
    path = root / "cases" / f"{declared['key']}.json"
    receipt = json.loads(path.read_text())
    receipt["bond_sha256"] = receipt["expected_bond_sha256"] = "b" * 64
    write_json(path, receipt)
    with pytest.raises(ValueError, match="conflicting bond"):
        report(root)
    assert json.loads((root / "report/status.json").read_text())["status"] == "failed"
