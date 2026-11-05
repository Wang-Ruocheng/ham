"""Frozen-model 3D overlap-QCR report for the explicit raw-T Q ablation.

Only microscopic full-ensemble Q has a temperature-regression interpretation.
Coarse coordinates are plotted, never penalized against the unchanged bare T.
All roots remain descriptive and thermal_nu is always null.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from recurrent_rg.checkpoints import isolated_rng, load_run
from recurrent_rg.config import RunConfig
from recurrent_rg.data.ensembles import EnsembleDataset
from recurrent_rg.runner import file_fingerprint, prepared_data
from scripts.prepare_spin_glass3d_temperature_supervised import QUALIFICATION, SPLIT_CONTRACT_KIND, SUPERVISION
from scripts.report_spin_glass1d import _model_state_sha256
from scripts.report_spin_glass3d import (
    collect, closure_metrics, coordinate_diagnostics, figures, root_candidates, validate_metadata,
)
from scripts.run_spin_glass2d import write_json


def validate_supervised_metadata(datasets, metadata):
    """Validate true supervised metadata, then make an internal unlabelled view.

    The old collector's scientific geometry checks are reused without changing
    or saving the source cache or falsely relabelling the formal experiment.
    """
    datasets = list(datasets)
    if (metadata.get("labels_used_in_training") is not True
            or metadata.get("couplings_in_training_cache") is not True
            or metadata.get("temperature_supervision") != SUPERVISION):
        raise ValueError("Require explicit microscopic TRAIN raw-T supervision contract")
    if [split for split, _ in datasets] != ["train", "test"]:
        raise ValueError("Require ordered TRAIN and TEST datasets")
    expected_bonds, ladder, views = {}, {}, []
    for split, ds in datasets:
        labels = ds.couplings
        rows = metadata.get(split, [])
        if (labels is None or labels.dtype != torch.float64 or labels.shape != (len(rows), 1)
                or not torch.isfinite(labels).all() or not (labels > 0).all()):
            raise ValueError("Expected positive finite float64 raw temperature column")
        expected = torch.tensor([[row["temperature"]] for row in rows], dtype=torch.float64)
        if not torch.equal(labels.cpu(), expected):
            raise ValueError("Raw-T targets do not align with evaluation metadata")
        if (ds.group_ids is None
                or ds.group_ids.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
                or torch.any(ds.group_ids < 0)):
            raise ValueError("Expected nonnegative integer disorder and temperature IDs")
        for row in rows:
            j, bond = str(row["disorder_id"]), row["bond_sha256"]
            if j in expected_bonds and expected_bonds[j] != bond:
                raise ValueError("Conflicting bond identity")
            expected_bonds[j] = bond
            ti, temperature = row["temperature_index"], row["temperature"]
            if ti in ladder and ladder[ti] != temperature:
                raise ValueError("Temperature-index ladder differs between disorders")
            ladder[ti] = temperature
        views.append((split, EnsembleDataset(ds.configurations, None, ds.group_ids)))
    expected_contract = dict(kind=SPLIT_CONTRACT_KIND,
        group_columns=["disorder_id", "temperature_index"], supervision_columns=["T"],
        bond_sha256_by_disorder=expected_bonds)
    if metadata.get("split_contract") != expected_contract:
        raise ValueError("Missing or inconsistent whole-disorder supervision split contract")
    if len(set(expected_bonds.values())) != len(expected_bonds):
        raise ValueError("Different disorder IDs share the same bond fingerprint")
    if metadata.get("split_unit") == "whole_temperature_within_fixed_disorder":
        raise ValueError("This ablation holds out whole J, not temperatures within a single J")
    internal = copy.deepcopy(metadata)
    internal["labels_used_in_training"] = False
    internal["couplings_in_training_cache"] = False
    validate_metadata(views, internal)
    return views, internal


def _regression(values, targets):
    error = np.asarray(values, dtype=np.float64) - np.asarray(targets, dtype=np.float64)
    x, y = np.asarray(values), np.asarray(targets)
    variance = float(np.mean((y - np.mean(y)) ** 2)) if np.ptp(y) > 0 else 0.
    correlation = (float(np.corrcoef(x, y)[0, 1])
                   if len(x) > 1 and np.ptp(x) > 0 and np.ptp(y) > 0 else None)
    mse = float(np.mean(error ** 2))
    return dict(ensembles=len(error), mse=mse, rmse=float(np.sqrt(mse)),
        mae=float(np.mean(np.abs(error))), bias=float(np.mean(error)),
        r_squared=(1 - mse / variance if variance > 0 else None), pearson=correlation)


def temperature_supervision_metrics(arrays):
    """Compute microscopic full-Q target errors; half-Q is a separate diagnostic."""
    result = dict(labels_used_in_training=True, target="raw_temperature_T", weight=10., scale=0,
        primary_estimator="full_ensemble_direct[:,0,0], Q of all requested microscopic configurations",
        half_subset_estimator="direct[:,0,0], separate subset diagnostic, not the supervised training estimator",
        coarse_temperature_error=None,
        coarse_qualification="No same-bare-T target or error is defined for k>=1",
        uncertainty="Per-J and cross-J population spread only; not independent temperature chains or a confidence interval")
    for name, prefix in (("full_ensemble_micro", "full_ensemble_"), ("half_subset_micro", "")):
        measurements = {}
        for split in ("train", "test"):
            direct = np.asarray(arrays[f"{prefix}direct_{split}"])
            targets = np.asarray(arrays[f"temperatures_{split}"], dtype=np.float64)
            groups = np.asarray(arrays[f"group_ids_{split}"])
            if (direct.ndim != 3 or direct.shape[2] != 1 or direct.shape[0] != len(targets)
                    or groups.shape != (len(targets), 2) or not np.isfinite(direct).all()
                    or not np.isfinite(targets).all() or np.any(targets <= 0)):
                raise ValueError("Temperature regression arrays do not align")
            predictions = direct[:, 0, 0]
            per_j, by_temperature = [], []
            for j in np.unique(groups[:, 0]):
                ids = np.flatnonzero(groups[:, 0] == j)
                per_j.append(dict(disorder_id=int(j), **_regression(predictions[ids], targets[ids])))
            for temperature in np.unique(targets):
                ids = np.flatnonzero(targets == temperature)
                values = predictions[ids]
                by_temperature.append(dict(temperature=float(temperature),
                    disorder_ids=groups[ids, 0].astype(int).tolist(), predictions=values.tolist(),
                    mean_g=float(values.mean()), disorder_population_std=float(values.std(ddof=0)),
                    residual_population_std=float((values - temperature).std(ddof=0)),
                    **_regression(values, targets[ids])))
            measurements[split] = dict(overall=_regression(predictions, targets), per_disorder=per_j,
                by_temperature=by_temperature,
                per_disorder_rmse_population_std=float(np.std([row["rmse"] for row in per_j], ddof=0)),
                per_disorder_mae_population_std=float(np.std([row["mae"] for row in per_j], ddof=0)),
                aggregation="Each saved (J,T) cell equally weighted; same full temperature ladder per J")
        result[name] = measurements
    return result


def _baseline_comparison(arrays, new_report, baseline):
    if baseline is None:
        return None
    baseline = Path(baseline).resolve()
    report_path, arrays_path = baseline / "report/report.json", baseline / "report/encodings.npz"
    before = {str(p): file_fingerprint(p) for p in (report_path, arrays_path)}
    old = json.loads(report_path.read_text())
    if (old.get("status") != "complete" or old.get("workflow_complete") is not True
            or old.get("labels_used_in_training") is not False
            or old.get("public_arrays_sha256") != before[str(arrays_path)]):
        raise ValueError("Require a complete, fingerprint-bound unsupervised baseline report")
    with np.load(arrays_path, allow_pickle=False) as z:
        old_arrays = {key: z[key] for key in z.files}
    for split in ("train", "test"):
        for prefix in ("group_ids", "temperatures", "bond_sha256"):
            if not np.array_equal(old_arrays[f"{prefix}_{split}"], arrays[f"{prefix}_{split}"]):
                raise ValueError("Baseline comparison must preserve all original cells, J and temperatures")
    if (old.get("samples") != new_report["samples"] or old.get("C_draws") != new_report["C_draws"]
            or not np.array_equal(old_arrays["sample_partition"], arrays["sample_partition"])):
        raise ValueError("Baseline evaluation protocol differs")
    after = {str(p): file_fingerprint(p) for p in (report_path, arrays_path)}
    if before != after:
        raise RuntimeError("Read-only baseline report fingerprints changed")
    return dict(source_root=str(baseline), source_files_sha256=before, source_files_unchanged=True,
        matched_whole_disorder_cells=True, matched_evaluation_samples_and_C_draws=True,
        unsupervised=dict(actual_training_updates=old.get("actual_training_updates"),
            closure=old["closure"], coordinates=old.get("coordinates"),
            fixed_point_candidates=old.get("fixed_point_candidates")),
        temperature_supervised=dict(actual_training_updates=new_report["actual_training_updates"],
            closure=new_report["closure"], coordinates=new_report["coordinates"],
            fixed_point_candidates=new_report["fixed_point_candidates"]),
        raw_latent_coordinate_comparison=False,
        qualification="两模型独立学习 Q/C，坐标与目标方差可能改变。闭合 RMSE/NRMSE 并列展示，不把 g 绝对值或不同坐标中的误差差值当作同一物理误差；无监督 g 不以原始 T 为监督标尺。",
        thermal_nu=None)


def _supervision_figures(arrays, history, output, font_path):
    """New Chinese raw-T plots; do not relabel a coarse coordinate as T."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ft2font import FT2Font
    from matplotlib.text import Text
    available = set(FT2Font(font_path).get_charmap())
    outputs = []
    def save(fig, filename):
        for text in fig.findobj(match=Text):
            absent = {c for c in text.get_text() if "\u3400" <= c <= "\u9fff" and ord(c) not in available}
            if absent:
                raise ValueError(f"中文字体缺少字形：{''.join(sorted(absent))}")
        fig.tight_layout(); fig.savefig(output / filename, dpi=160); plt.close(fig); outputs.append(filename)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for split, marker, color, label in (("train", "o", "C0", "训练无序实现"), ("test", "x", "C1", "留出无序实现")):
        groups, T = arrays[f"group_ids_{split}"], arrays[f"temperatures_{split}"]
        values = arrays[f"full_ensemble_direct_{split}"][:, 0, 0]
        for count, j in enumerate(np.unique(groups[:, 0])):
            ids = np.flatnonzero(groups[:, 0] == j); ids = ids[np.argsort(T[ids])]
            kwargs = dict(color=color, marker=marker, markersize=3, linewidth=.8, alpha=.7,
                          label=label if count == 0 else None)
            axes[0].plot(T[ids], values[ids], **kwargs)
            axes[1].plot(T[ids], values[ids] - T[ids], **kwargs)
    low = min(arrays["temperatures_train"].min(), arrays["temperatures_test"].min())
    high = max(arrays["temperatures_train"].max(), arrays["temperatures_test"].max())
    axes[0].plot([low, high], [low, high], "k--", label="微观监督目标")
    axes[1].axhline(0, color="gray", linestyle="--")
    axes[0].set(xlabel="原始温度 $T$", ylabel="$g_0=Q(D_0)$", title="微观完整系综温度拟合（每条线为一个 J）")
    axes[1].set(xlabel="原始温度 $T$", ylabel="$g_0-T$", title="微观温度拟合残差（不是粗尺度目标）")
    for ax in axes: ax.legend()
    save(fig, "temperature_supervision_micro.png")
    levels = arrays["direct_train"].shape[1]
    fig, axes = plt.subplots(1, levels, figsize=(4.5 * levels, 4))
    for split, marker, color, label in (("train", "o", "C0", "训练无序实现"), ("test", "x", "C1", "留出无序实现")):
        groups, T = arrays[f"group_ids_{split}"], arrays[f"temperatures_{split}"]
        for k, ax in enumerate(np.atleast_1d(axes)):
            for count, j in enumerate(np.unique(groups[:, 0])):
                ids = np.flatnonzero(groups[:, 0] == j); ids = ids[np.argsort(T[ids])]
                ax.plot(T[ids], arrays[f"full_ensemble_direct_{split}"][ids, k, 0],
                    color=color, marker=marker, markersize=3, linewidth=.7, alpha=.65,
                    label=label if count == 0 else None)
            ax.set(xlabel="原始温度 $T$", ylabel="$g_k$", title=("微观尺度 k=0（受 T 监督）" if k == 0 else f"粗尺度 k={k}（无同 T 目标）")); ax.legend()
    save(fig, "temperature_by_scale.png")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for split, label in (("train", "训练"), ("test", "留出（只评价）")):
        for ax, key in zip(axes, ("coupling_supervision", "weighted_coupling_supervision")):
            available_rows = [row for row in history if f"{split}_{key}" in row]
            ax.plot([row["epoch"] for row in available_rows], [row[f"{split}_{key}"] for row in available_rows], label=label)
    axes[0].set(title="微观完整系综 T 监督损失", xlabel="优化步数", ylabel="MSE")
    axes[1].set(title="加权 T 监督损失（权重10）", xlabel="优化步数", ylabel="加权 MSE")
    for ax in axes: ax.legend()
    save(fig, "temperature_supervision_loss.png")
    return outputs


def report(root, *, device="cpu", draws=5, baseline=None, font=None):
    root = Path(root).resolve(); output = root / "report"
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.mkdir()
    write_json(output / "status.json", dict(status="running", workflow_complete=False, labels_used_in_training=True))
    try:
        torch.set_num_threads(1)
        cfg = RunConfig.load(root / "config.json")
        if (cfg.model.dimension, cfg.model.width, cfg.model.block_factor, cfg.model.theory_dim) != (3, 2, 2, 1):
            raise ValueError("Require scalar 3D anchored-cube b2 model")
        if list(cfg.data.coupling_names) != ["T"]:
            raise ValueError("Temperature-supervised configuration must declare coupling_names=['T']")
        train, test = prepared_data(cfg)
        metadata = json.loads((root / "data/evaluation_metadata.json").read_text())
        if metadata.get("cache_sha256") != file_fingerprint(cfg.data.cache):
            raise ValueError("Evaluation metadata does not bind actual cache")
        views, internal = validate_supervised_metadata([("train", train), ("test", test)], metadata)
        loaded = load_run(root / "run", checkpoint="checkpoint_last.pt")
        checkpoint_metadata = loaded.payload["metadata"]
        manifest = json.loads((root / "run/manifest.json").read_text())
        if (checkpoint_metadata.get("data_sha256") != metadata["cache_sha256"]
                or manifest.get("data_sha256") != metadata["cache_sha256"]
                or manifest.get("labels_used_in_training") is not True
                or checkpoint_metadata.get("labels_used_in_training") is not True
                or checkpoint_metadata.get("q_supervision") != manifest.get("q_supervision")
                or manifest.get("q_supervision", {}).get("weight") != 10.
                or manifest.get("q_supervision", {}).get("scale") != 0):
            raise ValueError("Training supervised objective/data receipt mismatch")
        saved = RunConfig.from_dict(loaded.config)
        if json.loads(json.dumps(saved.to_dict())) != json.loads(json.dumps(cfg.to_dict())):
            raise ValueError("Checkpoint architecture/objective differs from root config")
        if loaded.payload["epoch"] != cfg.training.epochs:
            raise ValueError("Requested formal training length is incomplete")
        model = loaded.model.to(device=device, dtype=torch.float64).eval().requires_grad_(False)
        before = _model_state_sha256(model)
        with isolated_rng(device, 20261019), torch.no_grad():
            arrays, contract = collect(model, views, internal, device,
                cfg.evaluation.samples, draws, cfg.training.max_rg_steps)
            source = np.concatenate([arrays[f"direct_{split}"][:, :-1].reshape(-1) for split in ("train", "test")])
            grid = torch.linspace(float(source.min()), float(source.max()), 1001, device=device, dtype=torch.float64)[:, None]
            arrays["R_grid"], arrays["R_mapped"] = grid.cpu().numpy(), model.advance_theory(grid).cpu().numpy()
            arrays["R_displacement"] = arrays["R_mapped"] - arrays["R_grid"]
        roots = root_candidates(model, arrays["R_grid"], arrays["R_mapped"], arrays, device)
        if before != _model_state_sha256(model) or any(p.grad is not None for p in model.parameters()):
            raise RuntimeError("Report changed model state or parameter gradients")
        summary = json.loads((root / "run/summary.json").read_text())
        font_path = figures(arrays, summary["history"], output, font=font)
        additional_figures = _supervision_figures(arrays, summary["history"], output, font_path)
        np.savez_compressed(output / "encodings.npz", **arrays)
        result = dict(status="complete", workflow_complete=True, dimension=3, b=2, g_dimension=1,
            actual_training_updates=loaded.payload["epoch"], labels_used_in_training=True,
            temperature_supervision=temperature_supervision_metrics(arrays),
            model_state_unchanged=True, model_gradients_none=True, model_sha256=before,
            C_draws=draws, samples=cfg.evaluation.samples,
            saved_source_samples_per_overlap_ensemble=train.configurations.shape[1],
            evaluation_sample_selection="First requested saved times per overlap ensemble; no source resampling",
            encoding_protocol=contract, closure=closure_metrics(arrays, metadata),
            coordinates=coordinate_diagnostics(arrays), fixed_point_candidates=roots,
            thermal_nu=None, overlap_pair=[0, 1], equilibrium_certified=False,
            split_unit=metadata["split_unit"], split_contract=metadata["split_contract"],
            uncertainty_unit="Whole quenched disorder, not temperature, C draw, replica pair or saved time",
            C_reduction="Mean across C draws before closure RMSE; mean R(g_draw) and R(mean g) remain separate",
            MI_shuffle_diagnostic=None,
            MI_shuffle_diagnostic_reason="Not recomputed: existing baseline MI cache is reused for training; no new MI null evaluation",
            source_sampling_warning_receipt=metadata.get("source_sampling_warning_receipt"),
            physical_sampling_updates=0, qualification=QUALIFICATION,
            scale_shapes=[[n // (2 ** k) for n in train.configurations.shape[2:-1]] for k in range(cfg.training.max_rg_steps + 1)],
            font_path=font_path, figures=["R_vs_g.png", "coupling_by_scale.png", "loss.png", *additional_figures],
            public_arrays_sha256=file_fingerprint(output / "encodings.npz"))
        result["baseline_comparison"] = _baseline_comparison(arrays, result, baseline)
        write_json(output / "report.json", result)
        write_json(output / "status.json", dict(status="complete", workflow_complete=True,
            labels_used_in_training=True, model_state_unchanged=True, thermal_nu=None))
        return result
    except Exception as exc:
        write_json(output / "status.json", dict(status="failed", workflow_complete=False,
            labels_used_in_training=True, error=str(exc)))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--draws", default=5, type=int)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--font", type=Path)
    args = parser.parse_args(argv)
    result = report(args.root, device=args.device, draws=args.draws, baseline=args.baseline, font=args.font)
    print(json.dumps(dict(status=result["status"], labels_used_in_training=True, thermal_nu=None), ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
