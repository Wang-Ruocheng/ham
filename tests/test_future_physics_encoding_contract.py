"""Toy-only collection and public-artifact estimator compatibility checks."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import prepare_ising_future_physics as prepare
from scripts import report_ising_future_physics as report
from scripts.ising_qcr_evaluation import (
    FULL_ENSEMBLE_ESTIMATOR, HALF_SUBSET_ESTIMATOR, encoding_contract,
    validate_encoding_arrays,
)
from test_ising_future_physics_report import fixture_run, write_json


def corrected_collection(root, arm):
    """Convert only synthetic fixture arrays to a declared half-Q collection."""
    path = root / ("run_" + arm) / "report/encodings.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    for split in ("train", "test"):
        direct = arrays["direct_" + split]
        half = np.stack((direct - .01, direct + .01), axis=1)
        full = half + .02
        for prefix, g in (("", half), ("full_ensemble_", full)):
            rg = g ** 2 + .2
            rollout = np.stack((g[:, :, 0], rg[:, :, 0], rg[:, :, 0] ** 2 + .2), axis=2)
            for key, value in (("direct", g), ("R_g", rg), ("rollout", rollout)):
                arrays[prefix + key + "_draws_" + split] = value
                arrays[prefix + key + "_" + split] = value.mean(1)
            arrays[prefix + "R_of_mean_g_" + split] = g.mean(1) ** 2 + .2
        arrays["Q_full_minus_half_draws_" + split] = full - half
        arrays["Q_full_minus_half_" + split] = (full - half).mean(1)
    np.savez_compressed(path, **arrays)
    metrics = dict(schema_version=2, actual_steps=6, scales=[108, 36, 12], samples=64, C_draws=2,
                   encoding_estimator=HALF_SUBSET_ESTIMATOR, encoding_protocol=encoding_contract(64),
                   encoding_matches_training=True)
    write_json(path.with_name("metrics.json"), metrics)
    return arrays, metrics


def test_legacy_report_is_allowed_but_explicitly_marked(tmp_path):
    fixture_run(tmp_path)
    result, _ = report.build_report(tmp_path, expected_steps=6)
    assert result["legacy_encoding_comparison"] is True
    assert result["comparison_encoding_estimator"] == FULL_ENSEMBLE_ESTIMATOR
    for arm in result["arms"].values():
        assert arm["encoding_estimator"] == FULL_ENSEMBLE_ESTIMATOR
        assert arm["encoding_legacy_unmarked"] is True
        assert arm["encoding_matches_training"] is False
        assert "历史全体 Q" in arm["encoding_qualification"]


def test_corrected_report_records_the_shared_contract(tmp_path):
    fixture_run(tmp_path)
    for arm in report.ARMS:
        corrected_collection(tmp_path, arm)
    result, _ = report.build_report(tmp_path, expected_steps=6)
    assert result["status"] == "complete"
    assert result["legacy_encoding_comparison"] is False
    assert result["comparison_encoding_estimator"] == HALF_SUBSET_ESTIMATOR
    for arm in result["arms"].values():
        assert arm["encoding_estimator"] == HALF_SUBSET_ESTIMATOR
        assert arm["encoding_matches_training"] is True
        assert arm["encoding_legacy_unmarked"] is False
        assert arm["encoding_protocol"] == encoding_contract(64)


def test_mixed_report_fails_before_cross_arm_array_comparison(tmp_path):
    fixture_run(tmp_path)
    arrays, _ = corrected_collection(tmp_path, "future")
    # Deliberately mismatched labels would fail the later cross-arm check;
    # the encoding mismatch must be rejected first instead.
    arrays["couplings_test"] = arrays["couplings_test"] + .1
    np.savez_compressed(tmp_path / "run_future/report/encodings.npz", **arrays)
    with pytest.raises(ValueError, match="编码估计器不同"):
        report.build_report(tmp_path, expected_steps=6)


@pytest.mark.parametrize("missing", ["full_ensemble_direct_draws_test", "Q_full_minus_half_test"])
def test_corrected_report_rejects_missing_saved_diagnostics(tmp_path, missing):
    fixture_run(tmp_path)
    arrays, _ = corrected_collection(tmp_path, "future")
    arrays.pop(missing)
    np.savez_compressed(tmp_path / "run_future/report/encodings.npz", **arrays)
    with pytest.raises(ValueError, match="半子集|差值"):
        report.build_report(tmp_path, expected_steps=6)


def test_corrected_report_rejects_falsified_saved_q_gap(tmp_path):
    fixture_run(tmp_path)
    arrays, _ = corrected_collection(tmp_path, "future")
    arrays["Q_full_minus_half_draws_test"] += 1
    np.savez_compressed(tmp_path / "run_future/report/encodings.npz", **arrays)
    with pytest.raises(AssertionError):
        report.build_report(tmp_path, expected_steps=6)


class ToyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1., dtype=torch.float64))

    def forward(self, x):
        return x.mean(tuple(range(1, x.ndim))).square()[:, None] * self.weight


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rg_kernel = torch.nn.Linear(1, 1).double()
        self.theory_encoder = ToyEncoder()
        self.c_calls = 0

    def coarse_grain(self, batch):
        self.c_calls += 1
        # A single sampled trajectory is shared by half/full Q and observables.
        sign = torch.where(torch.rand(()) > .5, 1., -1.)
        return [batch, batch * sign, batch * sign]

    def advance_theory(self, g):
        return g.square() + .2


def test_future_collector_uses_one_c_draw_and_restores_rng(tmp_path, monkeypatch):
    from recurrent_rg import checkpoints, execution
    from recurrent_rg.data import ensembles

    cache = tmp_path / "toy-cache.bin"
    cache.write_bytes(b"synthetic cache identity; never loaded")
    run = tmp_path / "toy-run"
    run.mkdir()
    model = ToyModel()
    cfg = prepare.make_config(cache, epochs=6)
    loaded = SimpleNamespace(model=model, config=cfg,
        payload=dict(epoch=6, metadata=dict(data_sha256=prepare.sha(cache))))
    spins = torch.ones((2, 4, 12, 12, 1), dtype=torch.float64)
    spins[:, -1] = -1
    train = SimpleNamespace(configurations=spins, couplings=torch.zeros(2, 7), group_ids=torch.arange(2))
    test = SimpleNamespace(configurations=spins, couplings=torch.ones(2, 7), group_ids=torch.arange(2) + 2)
    monkeypatch.setattr(checkpoints, "load_run", lambda path: loaded)
    monkeypatch.setattr(ensembles, "load_ensembles", lambda path: (train, test))
    monkeypatch.setattr(execution, "configure_cuda_execution", lambda *args, **kwargs: None)
    rng = torch.get_rng_state().clone()
    threads = torch.get_num_threads()
    try:
        prepare.collect(tmp_path, run, "cpu")
    finally:
        torch.set_num_threads(threads)
    assert torch.equal(torch.get_rng_state(), rng)
    assert model.c_calls == 2 * 5
    metrics = json.loads((run / "report/metrics.json").read_text())
    with np.load(run / "report/encodings.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    assert metrics["schema_version"] == 2
    assert metrics["encoding_matches_training"] is True
    assert metrics["samples"] == 4  # The actual synthetic input, not hard-coded64.
    assert metrics["encoding_protocol"] == encoding_contract(4)
    assert metrics["model_sha256_before"] == metrics["model_sha256_after"]
    assert validate_encoding_arrays(metrics, arrays) == HALF_SUBSET_ESTIMATOR
    assert np.max(np.abs(arrays["Q_full_minus_half_test"])) > .1
