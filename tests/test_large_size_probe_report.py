"""Synthetic saved JSON/NPZ only: no model, training or remote data is opened."""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import report_ising_large_size_probe as report


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False))


def fixture_probe(root):
    rows = [(size, k, replica) for size in report.SIZES for k in report.COUPLINGS for replica in (0, 1)]
    direct = np.arange(12 * 5 * 4, dtype=float).reshape(12, 5, 4, 1) / 500
    rollout = direct + .01
    rollout[:, :, 0] = direct[:, :, 0]
    observed = np.ones((12, 5, 4, 16)) / 16
    decoded = observed + .005
    predicted = decoded - .003
    arrays = dict(row_size0=np.array([row[0] for row in rows]), row_K0=np.array([row[1] for row in rows]),
        row_replica=np.array([row[2] for row in rows]), direct_g=direct, R_rollout_g=rollout,
        R_one_step_g=direct + .1, p_observed=observed, p_decoded_direct=decoded,
        p_decoded_rollout=predicted, scales=np.array([[size // 3 ** k for k in range(4)] for size, _, _ in rows]))
    arrays["error_total"] = predicted - observed
    arrays["error_R"] = predicted - decoded
    arrays["error_repr"] = decoded - observed
    (root / "collection").mkdir(parents=True)
    np.savez_compressed(root / "collection/arrays.npz", **arrays)
    # Keep full required dimensions, but compression makes these synthetic snapshots tiny.
    snapshots = {f"row_{row:03d}_scale_{k}": np.ones((2, int(L), int(L)), dtype=np.int8)
                 for row in range(12) if rows[row][2] == 0 for k, L in enumerate(arrays["scales"][row])}
    np.savez_compressed(root / "collection/snapshots.npz", **snapshots)
    for i in range(3):
        _json(root / f"data/c{i:03d}/sidecar.json", dict(warnings=["少量构型不能证明平衡"], samples_are_iid=False))
    receipt = dict(status="complete", arrays_sha256=report.public_sha(root / "collection/arrays.npz"),
        snapshots_sha256=report.public_sha(root / "collection/snapshots.npz"),
        training_updates=0, mutual_information_evaluations=0, all_parameters_frozen=True, all_grad_none=True,
        model_sha256_before="source_sha", model_sha256_after="source_sha",
        module_sha256_before=dict(Q="q", C="c", R="r"), module_sha256_after=dict(Q="q", C="c", R="r"),
        encoding_estimator=report.ENCODING_ESTIMATOR,
        encoding_protocol=dict(version=1, estimator=report.ENCODING_ESTIMATOR, samples=64, half_samples=32,
                               sample_partition=list(range(64))),
        dataset_rows=[dict(sidecar_path=f"data/c{i:03d}/sidecar.json") for i in range(3)])
    _json(root / "collection/receipt.json", receipt)
    _json(root / "collection/status.json", dict(status="complete"))
    _json(root / "source_record/summary.json", dict(actual_epochs=6000, history=[dict(epoch=1, train_commutator=.4, test_commutator=.5), dict(epoch=6000, train_commutator=.02, test_commutator=.03)]))
    _json(root / "source_record/job.json", dict(status="failed", workflow_complete=False))
    return arrays, receipt


def _rewrite_arrays(root, arrays):
    np.savez_compressed(root / "collection/arrays.npz", **arrays)
    path = root / "collection/receipt.json"
    receipt = json.loads(path.read_text())
    receipt["arrays_sha256"] = report.public_sha(root / "collection/arrays.npz")
    _json(path, receipt)


def test_report_preserves_scope_errors_and_snapshots_without_copy(tmp_path):
    arrays, _ = fixture_probe(tmp_path)
    original_job = (tmp_path / "source_record/job.json").read_bytes()
    result = report.report_run(tmp_path, make_plots=False)
    assert result["status"] == "complete" and result["workflow_complete"]
    assert result["training_updates"] == result["mutual_information_evaluations"] == 0
    assert result["thermal_nu"] is None and not result["root_search_performed"]
    assert not result["confidence_intervals_computed"] and not result["bootstrap_performed"]
    assert not result["decoder_refitted"] and not result["baseline_constant_fitted"]
    assert result["snapshot_selection"]["sample"] == result["snapshot_selection"]["C_draw"] == 0
    assert result["large_snapshots_copied"] is False
    assert not (tmp_path / "report/evaluation_arrays.npz").exists()
    assert not (tmp_path / "report/snapshots.npz").exists()
    assert (tmp_path / "source_record/job.json").read_bytes() == original_job
    assert result["original_source_workflow_status"] == "failed"
    assert len(result["sampling_sidecars"]) == 3
    for size in report.SIZES:
        info = result["sizes"][str(size)]
        assert info["scales"] == [size // 3 ** k for k in range(4)]
        for condition in info["conditions"]:
            assert condition["physical_chain_count"] == 2 and condition["C_draw_count"] == 5
            assert [row["trained_depth"] for row in condition["scales"]] == [True, True, True, False]
            for row in condition["scales"]:
                e = row["motif_of_C_means"]
                assert e["total_mse"] == pytest.approx(e["readout_mse"] + e["R_transport_mse"] + e["cross_term"])
                assert e["cross_term"] < 0
                assert e["total_rmse"] == pytest.approx(.002)
    status = json.loads((tmp_path / "report/status.json").read_text())
    assert status["report_sha256"] == report.public_sha(tmp_path / "report/report.json")
    assert set(result["input_sha256"]) >= {"collection/arrays.npz", "collection/snapshots.npz"}


def test_C_draws_averaged_before_chain_error_and_cross_not_hidden():
    observed = np.zeros((2, 2, 3))
    direct = np.array([[[1., 1., 1.], [-1., -1., -1.]], [[2., 2., 2.], [-2., -2., -2.]]])
    rollout = direct / 2
    after_mean = report.error_decomposition(observed, direct, rollout)
    per_draw = report.error_decomposition(observed, direct, rollout, average_C=False)
    assert after_mean["total_rmse"] == after_mean["readout_rmse"] == 0
    assert per_draw["total_mse"] == pytest.approx(.625)
    assert per_draw["readout_mse"] == pytest.approx(2.5)
    assert per_draw["R_transport_mse"] == pytest.approx(.625)
    assert per_draw["cross_term"] == pytest.approx(-2.5)
    assert len(per_draw["per_physical_chain"]["total_mse"]) == 2


def test_raw_affine_invalid_probability_is_reported_not_clipped(tmp_path):
    arrays, _ = fixture_probe(tmp_path)
    arrays["p_decoded_direct"][..., 0] = -.2
    arrays["p_decoded_rollout"][..., 1] = 1.2
    for key in ("error_total", "error_R", "error_repr"): arrays.pop(key)
    _rewrite_arrays(tmp_path, arrays)
    result, _, _ = report.build_report(tmp_path)
    raw = result["sizes"]["1080"]["conditions"][0]["scales"][0]["raw_affine_decoder"]
    assert raw["direct"]["minimum"] == -.2 and raw["direct"]["negative_fraction"] > 0
    assert raw["rollout"]["above_one_fraction"] > 0
    assert not raw["direct"]["clipped"] and not raw["direct"]["renormalized"]


@pytest.mark.parametrize("mutation,match", [("incomplete", "尚未"), ("training", "零训练"),
    ("MI", "零互信息"), ("frozen", "参数冻结"), ("fingerprint", "指纹发生"),
    ("modules", "模块指纹"), ("encoding", "半子集"), ("sha", "数组指纹"),
    ("snapshot_sha", "快照指纹"), ("sidecar", "sidecar"), ("escape", "相对路径"),
    ("partition", "编码划分协议")])
def test_invalid_receipts_rejected(tmp_path, mutation, match):
    _, receipt = fixture_probe(tmp_path)
    if mutation == "incomplete": receipt["status"] = "running"
    elif mutation == "training": receipt["training_updates"] = 1
    elif mutation == "MI": receipt["mutual_information_evaluations"] = 1
    elif mutation == "frozen": receipt["all_parameters_frozen"] = False
    elif mutation == "fingerprint": receipt["model_sha256_after"] = "changed"
    elif mutation == "modules":
        receipt["module_sha256_before"] = receipt["module_sha256_after"] = dict(Q="q")
    elif mutation == "encoding": receipt["encoding_estimator"] = "Q_of_full_ensemble_v1"
    elif mutation == "sha": receipt["arrays_sha256"] = "wrong"
    elif mutation == "snapshot_sha": receipt["snapshots_sha256"] = "wrong"
    elif mutation == "sidecar": receipt["dataset_rows"] = []
    elif mutation == "escape": receipt["dataset_rows"][0]["sidecar_path"] = "../old_data/sidecar.json"
    elif mutation == "partition": receipt["encoding_protocol"]["sample_partition"][1] = 0
    _json(tmp_path / "collection/receipt.json", receipt)
    with pytest.raises(ValueError, match=match): report.build_report(tmp_path)


@pytest.mark.parametrize("mutation", ["row", "shape", "R0", "probability", "nonfinite", "error"])
def test_invalid_saved_arrays_rejected(tmp_path, mutation):
    arrays, _ = fixture_probe(tmp_path)
    if mutation == "row": arrays["row_replica"][1] = 0
    elif mutation == "shape": arrays["direct_g"] = arrays["direct_g"][:, :, :3]
    elif mutation == "R0": arrays["R_rollout_g"][:, :, 0] = 7.
    elif mutation == "probability": arrays["p_observed"][..., 0] = -.1
    elif mutation == "nonfinite": arrays["R_one_step_g"][0, 0, 0, 0] = np.nan
    elif mutation == "error": arrays["error_R"] += .1
    _rewrite_arrays(tmp_path, arrays)
    with pytest.raises((ValueError, AssertionError)): report.build_report(tmp_path)


def test_no_overwrite_even_empty_report_or_path_escape(tmp_path):
    fixture_probe(tmp_path)
    (tmp_path / "report").mkdir()
    with pytest.raises(FileExistsError): report.report_run(tmp_path, make_plots=False)
    with pytest.raises(ValueError, match="输出根内"):
        report.report_run(tmp_path, output=tmp_path.parent / "other_probe", make_plots=False)


def test_pure_saved_artifact_report_no_model_or_root_search():
    source = Path(report.__file__).read_text()
    assert "import torch" not in source and "load_run(" not in source and "brentq" not in source
    assert "ssh " not in source


def test_plan_module_pin_checked_without_conflating_checkpoint_state_sha(tmp_path):
    _, receipt = fixture_probe(tmp_path)
    _json(tmp_path / "plan.json", dict(source=dict(module_sha256=receipt["module_sha256_before"],
        checkpoint_sha256="different_file_sha_not_state_sha")))
    result, _, _ = report.build_report(tmp_path)
    assert result["status"] == "complete"
    _json(tmp_path / "plan.json", dict(source=dict(module_sha256=dict(Q="changed", C="c", R="r"))))
    with pytest.raises(ValueError, match="冻结计划"):
        report.build_report(tmp_path)


def test_long_range_statistics_never_labeled_R_predictions(tmp_path):
    arrays, _ = fixture_probe(tmp_path)
    arrays["long_range_observed"] = np.zeros((12, 5, 4, 5))
    _rewrite_arrays(tmp_path, arrays)
    result, _, _ = report.build_report(tmp_path)
    diagnostic = result["sizes"]["1080"]["conditions"][0]["scales"][3]["long_range_C_observed_only"]
    assert diagnostic["R_prediction_available"] is False
    assert len(diagnostic["per_physical_chain"]) == 2


def test_chinese_figures_and_nearest_grid_render(tmp_path):
    from scripts.report_ising_zh import select_chinese_font
    try: font = select_chinese_font()
    except RuntimeError: pytest.skip("No installed Chinese font")
    fixture_probe(tmp_path)
    result = report.report_run(tmp_path, font=font)
    assert all((tmp_path / "report" / name).stat().st_size > 1000 for name in (*report.FIGURES, "loss_reference.png"))
    assert "loss_reference.png" in result["public_files"]
    assert 'interpolation="nearest", resample=False' in Path(report.__file__).read_text()
