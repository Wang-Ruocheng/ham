"""Frozen scalar-QCR report for two-dimensional EA real-replica overlaps.

Only encodings, predictions and descriptive evaluation metadata are public.
No roots, critical temperatures or exponents are fitted by this report.
"""
from __future__ import annotations

import argparse
import json
from math import prod
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recurrent_rg.checkpoints import isolated_rng, load_run
from recurrent_rg.config import RunConfig
from recurrent_rg.data.statistics import plaquette_distribution
from recurrent_rg.encoding import half_subset_theory
from recurrent_rg.runner import prepared_data
from scripts.ising_qcr_evaluation import encoding_contract, validate_encoding_arrays
from scripts.report_spin_glass1d import (
    DRAW_SEED, SPLIT_SEED, _disorder_values, _expectation, _group_ids,
    _model_state_sha256, _sha256, _write_json, compute_closure_metrics,
)


QUALIFICATION = (
    "二维零场最近邻 EA ±J 重叠场的有限正温、有限尺寸探索；统计仅作描述，未认证平衡采样。"
    "二维存在规范不变的局域受挫方格，不能像一维开链那样消去全部随机键。"
    "T_c=0 是该模型的热力学理论背景，不是本次有限尺寸学习的测量结论。"
    "未使用地面态，不搜索不动点、相变温度或热临界指数；d_g=1 只是表示维度假设。"
)


def _validate_metadata(datasets, metadata):
    """Check the evaluation-only identity/geometry without loading bond arrays."""
    if metadata.get("schema_version") != 1 or metadata.get("dimension") != 2:
        raise ValueError("Expected two-dimensional evaluation metadata schema 1")
    all_bonds, disorder_frustration, temperatures = {}, {}, {}
    split_ids = {}
    for split, dataset in datasets:
        configurations = dataset.configurations
        if configurations.ndim != 5 or configurations.shape[-1] != 1:
            raise ValueError("Expected binary overlap configurations[E,S,H,W,1]")
        shape = list(configurations.shape[2:-1])
        if metadata.get("lattice_shape") != shape:
            raise ValueError("Evaluation metadata lattice geometry differs from cache")
        if dataset.couplings is not None:
            raise ValueError("Temperature/J labels must be absent from the training cache")
        if dataset.group_ids is None:
            raise ValueError("Explicit disorder/temperature-index grouping is required")
        rows = metadata.get(split)
        if not isinstance(rows, list) or len(rows) != len(configurations):
            raise ValueError("Evaluation sidecar must match every cache ensemble in order")
        groups = _group_ids(dataset.group_ids.numpy(), len(rows))
        split_ids[split] = set(groups[:, 0].tolist())
        for row, group in zip(rows, groups):
            if [row["disorder_id"], row["temperature_index"]] != group.tolist():
                raise ValueError("Evaluation sidecar order differs from training cache")
            if row.get("overlap_replicas") != [0, 1]:
                raise ValueError("Require predeclared same-disorder real-replica overlap pair 0/1")
            if row.get("dimension") != 2 or row.get("lattice_shape") != shape:
                raise ValueError("Each evaluation row must carry the correct 2D lattice geometry")
            if row.get("sites") != prod(shape) or row.get("samples") != configurations.shape[1]:
                raise ValueError("Sidecar sample/site count differs from cache")
            if "ring_flux" in row:
                raise ValueError("The 1D ring-flux grouping is not a 2D frustration diagnostic")
            temperature = row.get("temperature")
            if not isinstance(temperature, (int, float)) or not np.isfinite(temperature) or temperature <= 0:
                raise ValueError("Require finite positive evaluation temperatures")
            bond = row.get("bond_sha256", "")
            if len(bond) != 64 or any(c not in "0123456789abcdef" for c in bond):
                raise ValueError("Missing/invalid disorder bond fingerprint")
            disorder = int(group[0])
            if disorder in all_bonds and all_bonds[disorder] != bond:
                raise ValueError("One disorder has conflicting bond fingerprints")
            all_bonds[disorder] = bond
            frustration = row.get("frustrated_plaquette_fraction")
            zero = row.get("zero_plaquette_fraction")
            if any(not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1
                   for value in (frustration, zero)) or frustration + zero > 1 + 1e-12:
                raise ValueError("Invalid local plaquette frustration fractions")
            pair = (frustration, zero)
            if disorder in disorder_frustration and disorder_frustration[disorder] != pair:
                raise ValueError("One disorder has conflicting local frustration metadata")
            disorder_frustration[disorder] = pair
            cell = tuple(group.tolist())
            if cell in temperatures and temperatures[cell] != temperature:
                raise ValueError("One (disorder,temperature-index) cell has conflicting temperatures")
            temperatures[cell] = temperature
    if split_ids["train"] & split_ids["test"]:
        raise ValueError("Whole disorders must be held out between TRAIN and TEST")
    train_bonds = {all_bonds[disorder] for disorder in split_ids["train"]}
    if train_bonds & {all_bonds[disorder] for disorder in split_ids["test"]}:
        raise ValueError("Identical bond realizations leak between TRAIN and TEST")


def collect_encodings(model, datasets, metadata, device, *, samples, draws, depth, batch_size=8):
    """Reuse one sampled 2D C trajectory for half-Q/full-Q and 16-bin motifs."""
    if type(draws) is not int or draws < 1 or type(depth) is not int or depth < 1:
        raise ValueError("Require positive integer C draws and RG depth")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    contract = encoding_contract(samples, split_seed=SPLIT_SEED)
    partition = torch.tensor(contract["sample_partition"], dtype=torch.long, device=device)
    arrays = {"sample_partition": np.array(contract["sample_partition"], dtype=np.int64)}
    decoder = getattr(model.theory_encoder, "distribution_decoder", None)
    for split, dataset in datasets:
        if samples > dataset.configurations.shape[1]:
            raise ValueError("Requested evaluation samples exceed the actual saved ensemble")
        pieces = {}
        for start in range(0, len(dataset.configurations), batch_size):
            batch = dataset.configurations[start:start + batch_size, :samples].to(device=device, dtype=torch.float64)
            current = {}
            for _ in range(draws):
                levels = model.coarse_grain(batch, max_rg_steps=depth)
                half = half_subset_theory(model, [level.index_select(1, partition) for level in levels])
                full = torch.stack([model.theory_encoder(level) for level in levels], dim=1)
                for prefix, direct in (("", half), ("full_ensemble_", full)):
                    rollout = [direct[:, 0]]
                    for _ in range(depth):
                        rollout.append(model.advance_theory(rollout[-1]))
                    for key, value in (("direct", direct), ("R_g", model.advance_theory(direct)),
                                       ("rollout", torch.stack(rollout, dim=1))):
                        current.setdefault(prefix + key, []).append(value.cpu())
                if decoder is not None:
                    target = torch.stack([plaquette_distribution(level) for level in levels], dim=1)
                    reconstruction = decoder(half)
                    if reconstruction.shape != target.shape or target.shape[-1] != 16:
                        raise ValueError("2D binary affine reconstruction must use the training 16-bin plaquette target")
                    current.setdefault("plaquette_distribution_target", []).append(target.cpu())
                    current.setdefault("plaquette_distribution_reconstruction", []).append(reconstruction.cpu())
            for key, values in current.items():
                pieces.setdefault(key, []).append(torch.stack(values, dim=1))
        for key, chunks in pieces.items():
            value = torch.cat(chunks).numpy()
            arrays[f"{key}_draws_{split}"] = value
            arrays[f"{key}_{split}"] = value.mean(axis=1)
        for prefix in ("", "full_ensemble_"):
            mean = torch.from_numpy(arrays[f"{prefix}direct_{split}"]).to(device)
            arrays[f"{prefix}R_of_mean_g_{split}"] = model.advance_theory(mean).cpu().numpy()
        for key in ("direct", "R_g", "rollout"):
            arrays[f"{key}_Cmean_{split}"] = arrays[f"{key}_{split}"]
        gap = arrays[f"full_ensemble_direct_draws_{split}"] - arrays[f"direct_draws_{split}"]
        arrays[f"Q_full_minus_half_draws_{split}"] = gap
        arrays[f"Q_full_minus_half_{split}"] = gap.mean(axis=1)
        rows = metadata[split]
        arrays[f"group_ids_{split}"] = dataset.group_ids.numpy().copy()
        arrays[f"temperatures_{split}"] = np.array([row["temperature"] for row in rows])
        arrays[f"bond_sha256_{split}"] = np.array([row["bond_sha256"] for row in rows], dtype="U64")
        for key in ("frustrated_plaquette_fraction", "zero_plaquette_fraction"):
            arrays[f"{key}_{split}"] = np.array([row[key] for row in rows])
    validate_encoding_arrays(dict(schema_version=2, samples=samples, C_draws=draws,
        encoding_estimator=contract["estimator"], encoding_protocol=contract), arrays)
    return arrays, contract


def describe_coordinates(arrays):
    """Temperature cells and local frustration, always balanced by whole J."""
    result = {}
    for split in ("train", "test"):
        groups, direct = arrays[f"group_ids_{split}"], arrays[f"direct_{split}"]
        temperatures = arrays[f"temperatures_{split}"]
        rows = []
        for temperature in np.unique(temperatures):
            mask = temperatures == temperature
            ids, values = _disorder_values(direct[mask], groups[mask])
            rows.append(dict(temperature=float(temperature), disorder_ids=ids.tolist(),
                independent_disorders=len(ids), mean_g=values.mean(axis=0).tolist(),
                disorder_population_std=values.std(axis=0, ddof=0).tolist(),
                per_disorder_g=values.tolist()))
        gap = arrays[f"Q_full_minus_half_draws_{split}"]
        jensen = arrays[f"R_g_{split}"] - arrays[f"R_of_mean_g_{split}"]
        ids, frustration = _disorder_values(arrays[f"frustrated_plaquette_fraction_{split}"], groups)
        result[split] = dict(by_temperature=rows,
            local_frustration=dict(disorder_ids=ids.tolist(), per_disorder_fraction=frustration.tolist(),
                descriptive_mean=float(frustration.mean()), interpretation="Fraction of elementary plaquettes with negative bond-sign product; not a 1D ring flux"),
            Q_full_minus_half_rmse_by_scale=np.sqrt(_expectation(np.square(gap).mean(axis=(1, 3)), groups)).tolist(),
            R_Cmean_minus_R_of_mean_g_rmse_by_scale=np.sqrt(_expectation(np.square(jensen).mean(axis=-1), groups)).tolist())
        if f"plaquette_distribution_target_draws_{split}" in arrays:
            error = arrays[f"plaquette_distribution_reconstruction_draws_{split}"] - arrays[f"plaquette_distribution_target_draws_{split}"]
            result[split]["plaquette_distribution_reconstruction_rmse_by_scale"] = np.sqrt(
                _expectation(np.square(error).mean(axis=(1, 3)), groups)).tolist()
            result[split]["reconstruction_note"] = "Raw saved affine decoder of half-Q against the training 16-bin plaquette distribution; no softmax or refit; per-bin RMSE"
    return result


def make_figures(arrays, coordinates, history, output, *, lattice_shape, block_factor, font=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ft2font import FT2Font
    from matplotlib.text import Text
    from scripts.report_ising_zh import select_chinese_font

    font_path = select_chinese_font(font)
    properties = font_manager.FontProperties(fname=str(font_path))
    available = set(FT2Font(str(font_path)).get_charmap())
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({"axes.unicode_minus": False, "font.family": properties.get_name()})
    output = Path(output)

    def save(figure, name):
        for artist in figure.findobj(match=Text):
            text = artist.get_text()
            missing = {char for char in text if "\u3400" <= char <= "\u9fff" and ord(char) not in available}
            if missing:
                raise ValueError(f"Chinese font lacks figure glyphs: {''.join(sorted(missing))}")
            copy = artist.get_fontproperties().copy()
            copy.set_file(str(font_path))
            artist.set_fontproperties(copy)
        figure.tight_layout()
        figure.savefig(output / name, dpi=170)
        plt.close(figure)

    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    panels = (("total", "总损失（含互信息）"), ("rsmi_bits", "微观缓冲全壳互信息（比特）"),
              ("separation", "Q 局域方格分布重构损失"), ("anchor", "Q 微观子集一致性损失"),
              ("coarse_consistency", "Q 粗尺度子集一致性损失"), ("commutator", "R 单步闭合损失"),
              ("rollout", "R 微观起点多步闭合损失"), ("kernel_noncollapse", "C 系综方差下限损失"),
              ("kernel_sign", "C 均匀块颜色保持损失"))
    if not history:
        raise ValueError("Saved loss history is required")
    for ax, (key, title) in zip(axes.flat, panels):
        for split, label in (("train", "训练集"), ("test", "测试集")):
            rows = [row for row in history if row.get(f"{split}_{key}") is not None]
            if rows:
                ax.plot([row["epoch"] for row in rows], [row[f"{split}_{key}"] for row in rows], label=label)
        ax.set(title=title, xlabel="训练更新次数", ylabel="已记录数值")
        ax.grid(alpha=.2)
        if ax.lines:
            ax.legend()
        else:
            ax.text(.5, .5, "历史未记录此项", transform=ax.transAxes, ha="center")
    fig.suptitle("二维重叠场训练历史：直接读取已保存的损失")
    save(fig, "loss.png")

    grid, mapped = arrays["R_grid"][:, 0], arrays["R_mapped"][:, 0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(grid, mapped, label="已保存 R 的映射")
    axes[0].plot(grid, grid, "k--", label="恒等映射")
    axes[1].plot(grid, mapped - grid, label="离散位移 R(g)-g")
    axes[1].axhline(0, color="black", ls="--", lw=.8)
    for split, label, color in (("train", "训练集", "tab:blue"), ("test", "测试集", "tab:orange")):
        direct = arrays[f"direct_{split}"][:, :-1, 0].reshape(-1)
        prediction = arrays[f"R_of_mean_g_{split}"][:, :-1, 0].reshape(-1)
        axes[0].scatter(direct, prediction, s=9, alpha=.35, color=color, label=f"{label}观测编码")
        axes[1].quiver(direct, np.zeros_like(direct), prediction - direct, np.zeros_like(direct),
            color=color, alpha=.3, angles="xy", scale_units="xy", scale=1, label=f"{label}真实位移箭头")
    axes[0].set(xlabel="输入 g", ylabel="输出 R(g)", title="冻结的二维系统标量映射")
    axes[1].set(xlabel="输入 g", ylabel="R(g)-g", title="箭头长度为实际 R(g)-g")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    save(fig, "R_vs_g.png")

    scales = arrays["direct_train"].shape[1]
    fig, axes = plt.subplots(2, scales, squeeze=False, figsize=(4 * scales, 7.5))
    for row_index, (split, name) in enumerate((("train", "训练集"), ("test", "测试集"))):
        rows = sorted(coordinates[split]["by_temperature"], key=lambda row: row["temperature"])
        for k in range(scales):
            ax = axes[row_index, k]
            ax.plot([row["temperature"] for row in rows], [row["mean_g"][k][0] for row in rows],
                "o-", markersize=3, label="独立无序样本均值", lw=2)
            ax.scatter(arrays[f"temperatures_{split}"], arrays[f"direct_{split}"][:, k, 0],
                s=7, color="gray", alpha=.25, label="全部系综")
            shape = [length // block_factor ** k for length in lattice_shape]
            ax.set(title=f"{name}：k={k}，{shape[0]}×{shape[1]}", xlabel="温度 T", ylabel="直接编码 g")
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    fig.suptitle("二维重叠场温度到各尺度 g：温度仅供评价，不参与训练")
    save(fig, "coupling_by_scale.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, split, name in zip(axes, ("train", "test"), ("训练集", "测试集")):
        temperatures = arrays[f"temperatures_{split}"]
        colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(np.unique(temperatures))))
        for temperature, color in zip(np.unique(temperatures), colors):
            mask = temperatures == temperature
            groups = arrays[f"group_ids_{split}"][mask]
            direct, rollout = arrays[f"direct_{split}"][mask], arrays[f"rollout_{split}"][mask]
            for ensemble in direct:
                ax.plot(range(scales), ensemble[:, 0], color=color, alpha=.08, lw=.6)
            ax.plot(range(scales), _expectation(direct, groups)[:, 0], "o-", color=color, label=f"T={temperature:g}")
            ax.plot(range(scales), _expectation(rollout, groups)[:, 0], "x--", color=color)
        ax.set(title=f"{name}：实线直接编码，虚线 R 多步迭代", xlabel="粗粒化尺度 k", ylabel="g")
        ax.set_xticks(range(scales))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, ncol=2)
    fig.suptitle("全部温度与局域受挫无序样本均保留；描述性均值不代表平衡认证")
    save(fig, "trajectories.png")
    return str(font_path)


def report(root, *, device="cpu", draws=5, output=None, font=None):
    root = Path(root).resolve()
    output = Path(output).resolve() if output else root / "report"
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "status.json", dict(status="running", workflow_complete=False))
    try:
        if type(draws) is not int or draws < 1:
            raise ValueError("draws must be a positive integer")
        torch.set_num_threads(1)
        config = RunConfig.load(root / "config.json")
        if (config.model.dimension != 2 or config.model.theory_dim != 1
                or config.model.categorical_states or config.model.block_factor != 3):
            raise ValueError("This report requires binary two-dimensional scalar QCR with block factor 3")
        train, test = prepared_data(config)
        datasets = [("train", train), ("test", test)]
        metadata_path = root / "data/evaluation_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        _validate_metadata(datasets, metadata)
        fingerprint = _sha256(config.data.cache)
        if metadata.get("cache_sha256") != fingerprint:
            raise ValueError("Evaluation metadata cache fingerprint mismatch")
        run_dir = root / "run"
        loaded = load_run(run_dir, checkpoint="checkpoint_last.pt")
        saved_config = RunConfig.from_dict(loaded.config)
        current, saved = config.to_dict(), saved_config.to_dict()
        current["data"]["cache"] = saved["data"]["cache"] = str(Path(config.data.cache).resolve())
        if current != saved:
            raise ValueError("Root config differs from saved checkpoint architecture/objective")
        manifest = json.loads((run_dir / "manifest.json").read_text())
        for saved_metadata in (manifest, loaded.payload.get("metadata", {})):
            if saved_metadata.get("data_sha256") is not None and saved_metadata["data_sha256"] != fingerprint:
                raise ValueError("Training manifest/checkpoint data fingerprint mismatch")
        summary = json.loads((run_dir / "summary.json").read_text())
        history = summary.get("history", loaded.payload.get("history", []))
        model = loaded.model.to(device=device, dtype=torch.float64).eval()
        model.requires_grad_(False)
        before_model_sha256 = _model_state_sha256(model)
        samples, depth = int(config.evaluation.samples), int(config.training.max_rg_steps)
        with isolated_rng(device, DRAW_SEED), torch.no_grad():
            arrays, contract = collect_encodings(model, datasets, metadata, device,
                samples=samples, draws=draws, depth=depth)
            values = np.concatenate([arrays[f"direct_{split}"].reshape(-1) for split in ("train", "test")])
            low, high = float(values.min()), float(values.max())
            padding = max(.02, .05 * (high - low))
            grid = torch.linspace(low - padding, high + padding, 401, device=device, dtype=torch.float64)[:, None]
            arrays["R_grid"] = grid.cpu().numpy()
            arrays["R_mapped"] = model.advance_theory(grid).cpu().numpy()
            arrays["R_displacement"] = arrays["R_mapped"] - arrays["R_grid"]
        after_model_sha256 = _model_state_sha256(model)
        gradients_none = all(parameter.grad is None for parameter in model.parameters())
        if before_model_sha256 != after_model_sha256 or not gradients_none:
            raise RuntimeError("Frozen evaluation changed model state or produced parameter gradients")
        closure, coordinates = compute_closure_metrics(arrays), describe_coordinates(arrays)
        lattice_shape = list(train.configurations.shape[2:-1])
        font_path = make_figures(arrays, coordinates, history, output,
            lattice_shape=lattice_shape, block_factor=config.model.block_factor, font=font)
        public_path = output / "encodings.npz"
        np.savez_compressed(public_path, **arrays)
        result = dict(schema_version=1, status="complete", workflow_complete=True,
            checkpoint=str(loaded.checkpoint), checkpoint_sha256=_sha256(loaded.checkpoint),
            checkpoint_epoch=loaded.payload.get("epoch"), training_cache_sha256=fingerprint,
            source_dataset=metadata.get("source_dataset"), evaluation_metadata_sha256=_sha256(metadata_path),
            source_manifest_sha256=metadata.get("source_manifest_sha256"), spatial_dimension=2, theory_dimension=1,
            labels_used_in_training=False, independent_unit="disorder", confidence_intervals=None,
            encoding_estimator=contract["estimator"], encoding_protocol=contract,
            C_draws=draws, C_draw_seed=DRAW_SEED, samples=samples, device=str(device), torch_threads=1,
            model_eval=True, model_parameters_frozen=all(not p.requires_grad for p in model.parameters()),
            model_gradients_none=gradients_none, model_state_sha256_before=before_model_sha256,
            model_state_sha256_after=after_model_sha256, model_state_unchanged=True,
            scale_shapes=[[length // config.model.block_factor ** k for length in lattice_shape] for k in range(depth + 1)],
            closure=closure, coordinates=coordinates,
            aggregation="Average C draws; merge repeated (J,T) cells; balance temperatures within J and then independent J; population std is descriptive, not a confidence interval",
            flow=dict(grid_points=len(arrays["R_grid"]), observed_g_support=[low, high],
                convention="Saved R(g); arrows are actual R(g)-g in coordinate units",
                roots_searched=False, exponent_search=False, grid_extension=padding),
            thermal_nu=None, equilibrium_certified=False, ground_state_used=False,
            thermodynamic_transition_certified=False, qualification=QUALIFICATION, chinese_font=font_path,
            public_npz=dict(path=public_path.name, sha256=_sha256(public_path), contains_spins=False,
                contains_bonds=False, contains_model_parameters=False,
                arrays={key: list(value.shape) for key, value in arrays.items()}),
            figures=["loss.png", "R_vs_g.png", "coupling_by_scale.png", "trajectories.png"])
        _write_json(output / "report.json", result)
        _write_json(output / "status.json", dict(status="complete", workflow_complete=True,
            report="report.json", thermal_nu=None, qualification=QUALIFICATION))
        return result
    except Exception as exc:
        _write_json(output / "status.json", dict(status="failed", workflow_complete=False,
            error=str(exc), thermal_nu=None))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--draws", default=5, type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--font", type=Path)
    args = parser.parse_args(argv)
    result = report(args.root, device=args.device, draws=args.draws, output=args.output, font=args.font)
    print(json.dumps(dict(status=result["status"], C_draws=result["C_draws"], samples=result["samples"],
        thermal_nu=None, qualification=QUALIFICATION), ensure_ascii=False))


if __name__ == "__main__":
    main()
