"""Four-scale public-cache checks only; no physical data or model inference."""
import json

import numpy as np
import pytest

from scripts import report_phi4_hopping_qcr as report


def fixture_run(tmp_path):
    root = tmp_path / "b2_run"
    root.mkdir()
    rng = np.random.default_rng(31)
    direct = rng.normal(size=(12, 4, 1)) * .1
    advance = lambda value: 1.03 * value + .02
    one_step = advance(direct[:, :-1])
    rollout = [direct[:, 0]]
    for _ in range(3):
        rollout.append(advance(rollout[-1]))
    predictions, pairs = [], []
    for target in range(1, 4):
        for source in range(target):
            prediction = direct[:, source]
            for _ in range(target - source):
                prediction = advance(prediction)
            predictions.append(prediction)
            pairs.append((source, target))
    target_features = rng.uniform(size=(12, 4, 625))
    target_features /= target_features.sum(-1, keepdims=True)
    arrays = dict(direct_g=direct, one_step_g=one_step,
        rollout_g=np.stack(rollout, axis=1),
        all_start_g=np.stack(predictions, axis=1), all_start_pairs=np.asarray(pairs),
        labels=np.column_stack((np.repeat([.5, .68, .86], 4), np.ones(12), np.zeros(12))),
        splits=np.repeat(["train", "test", "train"], 4), cell_ids=np.repeat(np.arange(3), 4),
        target_features=target_features, reconstructed_features=target_features + .0001,
        flow_grid=np.linspace(-.3, .3, 21)[:, None],
        flow_next=advance(np.linspace(-.3, .3, 21)[:, None]))
    np.savez(root / "evaluation_arrays.npz", **arrays)
    sizes = [128, 64, 32, 16]
    manifest = dict(recipe=report.RECIPE, schema_version=5,
        options=dict(steps=2, block_size=2, depth=3), model=dict(depth=3),
        block_size=2, stride=2, coarse_graining_depth=3, lattice_sizes=sizes,
        labels_used_in_training=False, field_representation="phi",
        evaluation_label_columns=["kappa", "quartic_lambda", "h"])
    summary = dict(recipe=report.RECIPE, status="complete", labels_used_in_training=False,
        actual_steps=2, thermal_nu=None, lattice_sizes=sizes,
        optimizer_steps={name: dict(min=2, max=2, parameters=4) for name in ("Q", "C", "R")})
    history = [dict(step=step, q_total=1. / step, c_total=-.1, r_total=.2,
                    q_reconstruction=.3, c_information=.2,
                    closure_one_step=.1, closure_rollout=.2) for step in (1, 2)]
    for name, value in (("manifest", manifest), ("summary", summary), ("history", history)):
        (root / f"{name}.json").write_text(json.dumps(value))
    return root, arrays


def test_four_scale_loader_and_six_pair_metrics_match_direct_reference(tmp_path):
    root, expected = fixture_run(tmp_path)
    _, _, _, arrays, _ = report.load_public_run(root)
    pairs = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]
    assert arrays["direct_g"].shape == (12, 4, 1)
    actual = report.closure_metrics(arrays)
    train = arrays["splits"] == "train"
    train_std = arrays["direct_g"][train].std(axis=0, ddof=0)
    for split in ("train", "test"):
        mask = arrays["splits"] == split
        rows = actual[split]["all_start"]
        assert len(rows) == 6
        target_mses = []
        for index, (source, target) in enumerate(pairs):
            error = expected["all_start_g"][mask, index] - expected["direct_g"][mask, target]
            rmse = np.sqrt(np.mean(error ** 2))
            assert rows[index]["source_scale"] == source and rows[index]["target_scale"] == target
            assert rows[index]["rmse"] == pytest.approx(rmse)
            assert rows[index]["nrmse"] == pytest.approx(rmse / train_std[target, 0])
            identity_rmse = np.sqrt(np.mean((expected["direct_g"][mask, source]
                                            - expected["direct_g"][mask, target]) ** 2))
            assert actual[split]["identity_all_start"][index]["rmse"] == pytest.approx(identity_rmse)
        for target in range(1, 4):
            selected = [i for i, (_, t) in enumerate(pairs) if t == target]
            errors = expected["all_start_g"][mask][:, selected] - expected["direct_g"][mask, target, None]
            target_mses.append(float(np.mean(np.sum(errors ** 2, axis=-1))))
        assert actual[split]["all_start_target_mse_coordinate_sum"] == pytest.approx(target_mses)
        assert actual[split]["weighted_all_start_objective"] == pytest.approx(10. * np.mean(target_mses))


@pytest.mark.parametrize("mutation", ["missing", "pair_order", "adjacent", "rollout", "geometry", "size"])
def test_four_scale_complete_cache_and_geometry_corruption_rejected(tmp_path, mutation):
    root, arrays = fixture_run(tmp_path)
    if mutation == "missing":
        del arrays["all_start_g"]
        del arrays["all_start_pairs"]
    elif mutation == "pair_order":
        arrays["all_start_pairs"] = arrays["all_start_pairs"][::-1]
    elif mutation == "adjacent":
        arrays["all_start_g"][0, 5] += 1.
    elif mutation == "rollout":
        arrays["all_start_g"][0, 3] += 1.
    else:
        path = root / "manifest.json"
        metadata = json.loads(path.read_text())
        if mutation == "geometry":
            metadata["options"]["block_size"] = 3
        else:
            metadata["lattice_sizes"][-1] = 4
        path.write_text(json.dumps(metadata))
    np.savez(root / "evaluation_arrays.npz", **arrays)
    with pytest.raises(ValueError):
        report.load_public_run(root)


def test_four_scale_chinese_png_report_covers_all_scales_and_pairs_without_model_io(tmp_path, monkeypatch):
    root, _ = fixture_run(tmp_path)
    before = {name: report._sha(root / name) for name in report.INPUT_FILES}
    import torch
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("report loaded a checkpoint"))
    import matplotlib.figure
    original_savefig = matplotlib.figure.Figure.savefig
    figure_evidence = {}
    def capture(figure, filename, *args, **kwargs):
        figure_evidence[str(filename).split("/")[-1]] = {
            "yticks": [axis.get_yticks().tolist() for axis in figure.axes],
            "xticklabels": [[label.get_text() for label in axis.get_xticklabels()] for axis in figure.axes],
            "legend_labels": [axis.get_legend_handles_labels()[1] for axis in figure.axes]}
        return original_savefig(figure, filename, *args, **kwargs)
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", capture)
    result = report.render_report(root)
    assert result["scales"] == 4 and result["depth"] == 3
    assert result["lattice_sizes"] == [128, 64, 32, 16]
    assert result["all_start_pairs"] == [[0, 1], [0, 2], [1, 2], [0, 3], [1, 3], [2, 3]]
    assert result["all_start_predictions_source"] == "saved_complete_cache"
    assert result["thermal_nu"] is None and result["critical_kappa"] is None
    assert len(result["metrics"]["test"]["closure"]["all_start"]) == 6
    assert len(result["metrics"]["test"]["one_step_coordinate_rmse_by_scale"]) == 3
    assert len(result["metrics"]["test"]["joint_probability_rmse_by_scale"]) == 4
    assert figure_evidence["R_vs_g.png"]["yticks"][1] == [0, 1, 2]
    labels = figure_evidence["coupling_by_scale.png"]["legend_labels"][0]
    assert any(label.startswith("k=3，") for label in labels)
    assert figure_evidence["all_start_closure.png"]["xticklabels"][0] == [
        "0→1", "0→2", "1→2", "0→3", "1→3", "2→3"]
    assert all((root / "report" / name).stat().st_size > 1000 for name in result["plots"])
    assert (root / "report/report.json").is_file()
    assert before == {name: report._sha(root / name) for name in report.INPUT_FILES}
