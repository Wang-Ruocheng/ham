"""Frozen-checkpoint, disorder-aware reports for one-dimensional EA overlap QCR.

Run with ``python -m scripts.report_spin_glass1d --root EXPERIMENT``.  The
public NPZ contains encodings, predictions and evaluation metadata, never
configurations, bonds or model parameters.  No roots or exponents are fitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recurrent_rg.checkpoints import isolated_rng, load_run
from recurrent_rg.config import RunConfig
from recurrent_rg.encoding import half_subset_theory
from recurrent_rg.runner import prepared_data
from scripts.ising_qcr_evaluation import encoding_contract, validate_encoding_arrays


DRAW_SEED = 20261019
SPLIT_SEED = 20261019
QUALIFICATION = (
    "有限正温、有限长度的一维 EA 重叠场探索；统计仅作描述，未认证平衡采样。"
    "一维零场 ±J 链的局域键符号可由规范变换消去；周期边界的环通量仍可保留。"
    "热力学临界温度为 T=0；这里不认证高维玻璃行为，不搜索不动点或估计热指数。"
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_state_sha256(model):
    """Read-only parameter/buffer fingerprint; never publish tensor values."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(json.dumps([name, str(array.dtype), list(array.shape)]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def _group_ids(value, ensembles):
    groups = np.asarray(value)
    if (groups.shape != (ensembles, 2) or not np.isfinite(groups).all()
            or not np.equal(groups, np.floor(groups)).all()):
        raise ValueError("Expected integral group_ids[E,2]=[disorder_id,temperature_index]")
    return groups.astype(np.int64)


def _disorder_values(values, groups):
    """Average duplicate (J,T) ensembles, then temperatures within each J.

    C draws have already been averaged by the caller.  This function never
    counts temperature cells or C draws as independent disorder samples.
    """
    values = np.asarray(values)
    if len(values) != len(groups) or not np.isfinite(values).all():
        raise ValueError("Finite ensemble values must match the grouping metadata")
    ids, result = np.unique(groups[:, 0]), []
    for disorder in ids:
        mask = groups[:, 0] == disorder
        cells = [values[mask & (groups[:, 1] == temperature)].mean(axis=0)
                 for temperature in np.unique(groups[mask, 1])]
        result.append(np.stack(cells).mean(axis=0))
    return ids, np.stack(result)


def _expectation(values, groups):
    return _disorder_values(values, groups)[1].mean(axis=0)


def _train_target_std(direct, groups, *, per_draw):
    # Population spread of the actual direct-Q TRAIN targets.  No epsilon,
    # rescaling of g, test-derived denominator, or artificial std is used.
    if per_draw:
        mean = _expectation(direct.mean(axis=1), groups)
        second = _expectation(np.square(direct).mean(axis=1), groups)
    else:
        value = direct.mean(axis=1)
        mean = _expectation(value, groups)
        second = _expectation(np.square(value), groups)
    variance = np.maximum(second - np.square(mean), 0)
    return np.sqrt(variance.mean(axis=-1))


def _closure_split_contract(arrays, groups, split_metadata):
    """Keep whole-J holdout strict unless one fixed-J protocol is explicit.

    The fixed-J exception checks saved evaluation metadata, not a boolean that
    could silently bypass the original disorder-leakage guard. Temperatures
    remain repeated conditions of one J, never independent physical chains.
    """
    if split_metadata is None:
        if set(groups["train"][:, 0]) & set(groups["test"][:, 0]):
            raise ValueError("Whole disorders must be held out between TRAIN and TEST")
        return False
    if (not isinstance(split_metadata, dict)
            or split_metadata.get("split_unit") != "whole_temperature_within_fixed_disorder"):
        raise ValueError("Require an explicit fixed-J whole-temperature split contract")
    disorder = split_metadata.get("fixed_disorder_id")
    bond = split_metadata.get("fixed_bond_sha256")
    if (type(disorder) is not int or disorder < 0 or not isinstance(bond, str)
            or len(bond) != 64 or any(c not in "0123456789abcdef" for c in bond)):
        raise ValueError("Fixed-J closure requires a declared integer J and valid bond fingerprint")
    temperature_indices, temperatures = {}, {}
    for split in ("train", "test"):
        group = groups[split]
        if set(group[:, 0]) != {disorder} or np.any(group[:, 1] < 0):
            raise ValueError("Fixed-J closure requires the same one declared J in both splits")
        rows = split_metadata.get(split)
        bonds = np.asarray(arrays.get(f"bond_sha256_{split}"))
        values = np.asarray(arrays.get(f"temperatures_{split}"))
        if (not isinstance(rows, list) or len(rows) != len(group)
                or bonds.shape != (len(group),) or bonds.dtype.kind not in "US"
                or not np.all(bonds == bond)):
            raise ValueError("Fixed-J saved rows/bond fingerprints do not match the declared J")
        if (values.shape != (len(group),) or values.dtype.kind not in "fiu"
                or not np.isfinite(values).all() or np.any(values <= 0)):
            raise ValueError("Fixed-J closure needs positive finite saved temperatures")
        for row, identity, value in zip(rows, group, values):
            if (not isinstance(row, dict)
                    or type(row.get("disorder_id")) is not int
                    or type(row.get("temperature_index")) is not int
                    or [row["disorder_id"], row["temperature_index"]] != identity.tolist()
                    or row.get("bond_sha256") != bond or row.get("temperature") != value):
                raise ValueError("Fixed-J saved cell identity/temperature differs from split metadata")
        temperature_indices[split] = set(group[:, 1].tolist())
        temperatures[split] = set(values.tolist())
        if (len(temperature_indices[split]) != len(group)
                or len(temperatures[split]) != len(group)):
            raise ValueError("Fixed-J closure requires unique whole temperature cells")
    if temperature_indices["train"] & temperature_indices["test"]:
        raise ValueError("Fixed-J temperature indices overlap between TRAIN and TEST")
    if temperatures["train"] & temperatures["test"]:
        raise ValueError("Fixed-J temperature values overlap between TRAIN and TEST")
    return True


def compute_closure_metrics(arrays, *, split_metadata=None):
    """Pure NumPy, shape-checked closure metrics for saved public arrays.

    Required per split: direct_draws, R_g_draws, rollout_draws [E,C,K,d]
    and group_ids [E,2].  The train denominators are target-scale population
    standard deviations, with disorder-balanced empirical weights.  Both
    mean-C-prediction error and mean squared per-C-draw error are retained.
    All returned standard deviations and errors are descriptive, not CIs.
    The default requires whole-J holdout. The explicit fixed-J metadata
    exception requires one matching bond realization and disjoint whole
    temperature indices/values, and does not infer physical independence.
    """
    directs, groups = {}, {}
    common_shape = None
    for split in ("train", "test"):
        direct = np.asarray(arrays[f"direct_draws_{split}"])
        if direct.ndim != 4 or min(direct.shape) < 1 or direct.shape[2] < 2:
            raise ValueError("Expected nonempty direct_draws[E,C,K>=2,d]")
        if not np.isfinite(direct).all():
            raise ValueError("Nonfinite direct encodings")
        if common_shape is not None and direct.shape[1:] != common_shape:
            raise ValueError("TRAIN/TEST must use identical C draws, scales and coordinates")
        common_shape = direct.shape[1:]
        for key in ("R_g", "rollout"):
            value = np.asarray(arrays[f"{key}_draws_{split}"])
            if value.shape != direct.shape or not np.isfinite(value).all():
                raise ValueError(f"{key} shape/finite mismatch")
        np.testing.assert_allclose(arrays[f"rollout_draws_{split}"][:, :, 0],
                                   direct[:, :, 0], rtol=1e-10, atol=1e-12)
        directs[split] = direct
        groups[split] = _group_ids(arrays[f"group_ids_{split}"], len(direct))
    fixed_disorder = _closure_split_contract(arrays, groups, split_metadata)
    result = dict(
        target_scales=list(range(1, common_shape[1])),
        aggregation="Average squared error over C draws and coordinates; merge repeated (J,T) cells; "
                    "average T cells within each J; average independent J equally; take square root last",
        normalization="Actual TRAIN direct-Q target population std at the corresponding target scale; "
                      "same TRAIN denominator for TRAIN and TEST; zero spread gives null NRMSE",
        independent_unit="disorder", confidence_intervals=None,
        Cmean={}, per_C_draw={})
    if fixed_disorder:
        result.update(split_unit="whole_temperature_within_fixed_disorder",
            fixed_disorder_id=split_metadata["fixed_disorder_id"],
            fixed_bond_sha256=split_metadata["fixed_bond_sha256"],
            aggregation="Average squared error over C draws and coordinates; average whole temperature "
                        "cells equally within the one fixed J; take square root last",
            independent_unit="One fixed quenched disorder; temperatures and C draws are repeated "
                             "conditions, not independent physical chains",
            uncertainty="Descriptive within-J temperature summaries only; no independent-disorder, "
                        "independent-chain or training-seed confidence interval")
    for mode, per_draw in (("Cmean", False), ("per_C_draw", True)):
        std = _train_target_std(directs["train"], groups["train"], per_draw=per_draw)[1:]
        result[mode]["train_target_std"] = std.tolist()
        result[mode]["definition"] = (
            "Compare mean_C prediction to mean_C direct target" if not per_draw else
            "Average prediction-minus-direct squared errors within each C draw")
        for split in ("train", "test"):
            direct = directs[split]
            target = direct[:, :, 1:] if per_draw else direct.mean(axis=1)[:, 1:]
            metrics = {}
            for branch, key in (("one_step", "R_g"), ("microscopic_rollout", "rollout")):
                value = np.asarray(arrays[f"{key}_draws_{split}"])
                value = value[:, :, :-1] if key == "R_g" else value[:, :, 1:]
                prediction = value if per_draw else value.mean(axis=1)
                error = np.square(prediction - target)
                mse = error.mean(axis=(1, 3)) if per_draw else error.mean(axis=2)
                ids, disorder_mse = _disorder_values(mse, groups[split])
                rmse = np.sqrt(disorder_mse.mean(axis=0))
                metrics[branch] = dict(
                    rmse=rmse.tolist(),
                    nrmse=[float(value / spread) if spread > 0 else None
                           for value, spread in zip(rmse, std)],
                    disorder_ids=ids.tolist(), per_disorder_rmse=np.sqrt(disorder_mse).tolist(),
                    uniform_target_scale_rmse=float(np.sqrt(disorder_mse.mean())),
                )
                if f"ring_flux_{split}" in arrays:
                    flux = np.asarray(arrays[f"ring_flux_{split}"])
                    if flux.shape != (len(direct),) or not np.isin(flux, (-1, 1)).all():
                        raise ValueError("Invalid ring-flux grouping")
                    sectors = {}
                    for sign in np.unique(flux):
                        mask = flux == sign
                        sector_ids, sector_mse = _disorder_values(mse[mask], groups[split][mask])
                        sector_rmse = np.sqrt(sector_mse.mean(axis=0))
                        sectors[str(int(sign))] = dict(disorder_ids=sector_ids.tolist(),
                            independent_disorders=len(sector_ids), rmse=sector_rmse.tolist(),
                            nrmse=[float(value / spread) if spread > 0 else None
                                   for value, spread in zip(sector_rmse, std)])
                    metrics[branch]["by_ring_flux"] = sectors
            metrics.update(ensembles=len(direct), independent_disorders=len(np.unique(groups[split][:, 0])))
            result[mode][split] = metrics
    return result


def _validate_metadata(datasets, metadata):
    if metadata.get("schema_version") != 1:
        raise ValueError("Unsupported evaluation metadata schema")
    all_bonds, disorder_flux, cell_temperatures = {}, {}, {}
    for split, dataset in datasets:
        if dataset.couplings is not None:
            raise ValueError("Temperature/J labels must be absent from this training cache")
        if dataset.group_ids is None:
            raise ValueError("Explicit disorder/temperature-index grouping is required")
        rows = metadata.get(split)
        if not isinstance(rows, list) or len(rows) != len(dataset.configurations):
            raise ValueError("Evaluation sidecar must match every cache ensemble in order")
        groups = _group_ids(dataset.group_ids.numpy(), len(rows))
        for row, group in zip(rows, groups):
            if [row["disorder_id"], row["temperature_index"]] != group.tolist():
                raise ValueError("Evaluation sidecar order differs from training cache")
            if row.get("overlap_replicas") != [0, 1]:
                raise ValueError("Expected the predeclared real-replica overlap pair 0/1")
            if not np.isfinite(row["temperature"]) or row["temperature"] <= 0:
                raise ValueError("Require finite positive evaluation temperatures")
            bond = row.get("bond_sha256", "")
            if len(bond) != 64 or any(c not in "0123456789abcdef" for c in bond):
                raise ValueError("Missing/invalid disorder bond fingerprint")
            if int(group[0]) in all_bonds and all_bonds[int(group[0])] != bond:
                raise ValueError("One disorder ID has conflicting bond fingerprints")
            all_bonds[int(group[0])] = bond
            if "ring_flux" in row and row["ring_flux"] not in (-1, 1):
                raise ValueError("Ring flux must be -1 or +1")
            if "ring_flux" in row:
                if (int(group[0]) in disorder_flux
                        and disorder_flux[int(group[0])] != row["ring_flux"]):
                    raise ValueError("One disorder has conflicting ring-flux metadata")
                disorder_flux[int(group[0])] = row["ring_flux"]
            cell = tuple(group.tolist())
            if cell in cell_temperatures and cell_temperatures[cell] != row["temperature"]:
                raise ValueError("One (disorder, temperature-index) cell has conflicting temperatures")
            cell_temperatures[cell] = row["temperature"]
            if "samples" in row and row["samples"] != dataset.configurations.shape[1]:
                raise ValueError("Sidecar sample count differs from cache")
            if "sites" in row and row["sites"] != dataset.configurations.shape[2]:
                raise ValueError("Sidecar lattice size differs from cache")
    if ({int(row["disorder_id"]) for row in metadata["train"]}
            & {int(row["disorder_id"]) for row in metadata["test"]}):
        raise ValueError("Disorders overlap between TRAIN and TEST")


def collect_encodings(model, datasets, metadata, device, *, samples, draws, depth, batch_size=8):
    """Exactly one C trajectory per batch/draw, reused for both Q estimators."""
    contract = encoding_contract(samples, split_seed=SPLIT_SEED)
    partition = torch.tensor(contract["sample_partition"], dtype=torch.long, device=device)
    arrays = {"sample_partition": np.array(contract["sample_partition"], dtype=np.int64)}
    decoder = getattr(model.theory_encoder, "distribution_decoder", None)
    from recurrent_rg.data.statistics import two_spin_distribution

    for split, dataset in datasets:
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
                    target = torch.stack([two_spin_distribution(level) for level in levels], dim=1)
                    prediction = decoder(half)
                    current.setdefault("pair_distribution_target", []).append(target.cpu())
                    current.setdefault("pair_distribution_reconstruction", []).append(prediction.cpu())
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
        if all("ring_flux" in row for row in rows):
            arrays[f"ring_flux_{split}"] = np.array([row["ring_flux"] for row in rows], dtype=np.int64)
    validate_encoding_arrays(dict(schema_version=2, samples=samples, C_draws=draws,
                                  encoding_estimator=contract["estimator"], encoding_protocol=contract), arrays)
    return arrays, contract


def describe_coordinates(arrays):
    """All temperature cells and flux sectors, with J as the repeated unit."""
    result = {}
    for split in ("train", "test"):
        groups = arrays[f"group_ids_{split}"]
        direct = arrays[f"direct_{split}"]
        temperatures = arrays[f"temperatures_{split}"]
        flux = arrays.get(f"ring_flux_{split}")
        rows = []
        for temperature in np.unique(temperatures):
            sectors = [("all", np.ones(len(groups), dtype=bool))]
            if flux is not None:
                sectors += [(str(sign), flux == sign) for sign in np.unique(flux)]
            for sector, sector_mask in sectors:
                mask = (temperatures == temperature) & sector_mask
                if not mask.any():
                    continue
                ids, values = _disorder_values(direct[mask], groups[mask])
                rows.append(dict(temperature=float(temperature), ring_flux=sector,
                    disorder_ids=ids.tolist(), independent_disorders=len(ids),
                    mean_g=values.mean(axis=0).tolist(),
                    disorder_population_std=values.std(axis=0, ddof=0).tolist(),
                    per_disorder_g=values.tolist()))
        gap = arrays[f"Q_full_minus_half_draws_{split}"]
        jensen = arrays[f"R_g_{split}"] - arrays[f"R_of_mean_g_{split}"]
        result[split] = dict(by_temperature_and_ring_flux=rows,
            Q_full_minus_half_rmse_by_scale=np.sqrt(_expectation(np.square(gap).mean(axis=(1, 3)), groups)).tolist(),
            R_Cmean_minus_R_of_mean_g_rmse_by_scale=np.sqrt(_expectation(np.square(jensen).mean(axis=-1), groups)).tolist())
        if f"pair_distribution_target_draws_{split}" in arrays:
            error = arrays[f"pair_distribution_reconstruction_draws_{split}"] - arrays[f"pair_distribution_target_draws_{split}"]
            result[split]["pair_distribution_reconstruction_rmse_by_scale"] = np.sqrt(
                _expectation(np.square(error).mean(axis=(1, 3)), groups)).tolist()
            result[split]["pair_reconstruction_note"] = "Raw saved affine decoder of half-Q; no softmax or refit; per-probability-bin RMSE"
    return result


def make_figures(arrays, coordinates, history, output, *, sites, block_factor, font=None):
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
    plt.rcParams.update({"axes.unicode_minus": False, "font.family": properties.get_name()})
    font_manager.fontManager.addfont(str(font_path))

    def save(figure, name):
        # Check every Chinese glyph including generated legend annotations.
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
              ("separation", "Q 局域分布重构损失"), ("anchor", "Q 微观子集一致性损失"),
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
    fig.suptitle("训练历史：直接读取已保存的损失")
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
        # Every observed source ensemble/scale is retained; no root or cell selection.
        axes[1].quiver(direct, np.zeros_like(direct), prediction - direct, np.zeros_like(direct),
                       color=color, alpha=.3, angles="xy", scale_units="xy", scale=1,
                       label=f"{label}真实位移箭头")
    axes[0].set(xlabel="输入 g", ylabel="输出 R(g)", title="冻结的离散标量映射")
    axes[1].set(xlabel="输入 g", ylabel="R(g)-g", title="箭头长度为实际 R(g)-g")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    save(fig, "R_vs_g.png")

    scales = arrays["direct_train"].shape[1]
    fig, axes = plt.subplots(2, scales, squeeze=False, figsize=(4 * scales, 7.5))
    for row_index, (split, name) in enumerate((("train", "训练集"), ("test", "测试集"))):
        rows = coordinates[split]["by_temperature_and_ring_flux"]
        sectors = sorted({row["ring_flux"] for row in rows})
        for k in range(scales):
            ax = axes[row_index, k]
            for sector in sectors:
                selected = sorted((row for row in rows if row["ring_flux"] == sector), key=lambda row: row["temperature"])
                label = "全部环通量" if sector == "all" else f"环通量 {int(sector):+d}"
                ax.plot([row["temperature"] for row in selected], [row["mean_g"][k][0] for row in selected],
                        "o-", markersize=3, label=label, lw=2 if sector == "all" else 1)
            temperatures = arrays[f"temperatures_{split}"]
            ax.scatter(temperatures, arrays[f"direct_{split}"][:, k, 0], s=7, color="gray", alpha=.2)
            ax.set(title=f"{name}：k={k}，L={sites // block_factor ** k}", xlabel="温度 T", ylabel="直接编码 g")
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    fig.suptitle("温度到各尺度 g：先在同温度内按独立无序样本平均；灰点保留全部系综")
    save(fig, "coupling_by_scale.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, split, name in zip(axes, ("train", "test"), ("训练集", "测试集")):
        temperatures = arrays[f"temperatures_{split}"]
        colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(np.unique(temperatures))))
        for temperature, color in zip(np.unique(temperatures), colors):
            mask = temperatures == temperature
            groups = arrays[f"group_ids_{split}"][mask]
            direct = arrays[f"direct_{split}"][mask]
            rollout = arrays[f"rollout_{split}"][mask]
            d = _expectation(direct, groups)[:, 0]
            r = _expectation(rollout, groups)[:, 0]
            for ensemble in direct:
                ax.plot(range(scales), ensemble[:, 0], color=color, alpha=.08, lw=.6)
            ax.plot(range(scales), d, "o-", color=color, label=f"T={temperature:g}")
            ax.plot(range(scales), r, "x--", color=color)
        ax.set(title=f"{name}：实线为直接编码，虚线为 R 多步迭代", xlabel="粗粒化尺度 k", ylabel="g")
        ax.set_xticks(range(scales))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, ncol=2)
    fig.suptitle("所有温度与无序样本均保留；曲线为独立无序样本均值，浅线为各系综")
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
        if config.model.dimension != 1 or config.model.theory_dim != 1:
            raise ValueError("This report requires one-dimensional scalar QCR")
        train, test = prepared_data(config)
        datasets = [("train", train), ("test", test)]
        metadata_path = root / "data/evaluation_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        _validate_metadata(datasets, metadata)
        fingerprint = _sha256(config.data.cache)
        if metadata.get("cache_sha256") is not None and metadata["cache_sha256"] != fingerprint:
            raise ValueError("Evaluation metadata cache fingerprint mismatch")
        run_dir = root / "run"
        loaded = load_run(run_dir, checkpoint="checkpoint_last.pt")
        saved_config = RunConfig.from_dict(loaded.config)
        # Paths are resolved separately because config.json may use a relative cache.
        current, saved = config.to_dict(), saved_config.to_dict()
        current["data"]["cache"] = saved["data"]["cache"] = str(Path(config.data.cache).resolve())
        if current != saved:
            raise ValueError("Root config differs from the saved checkpoint architecture/objective")
        manifest = json.loads((run_dir / "manifest.json").read_text())
        for saved_metadata in (manifest, loaded.payload.get("metadata", {})):
            if saved_metadata.get("data_sha256") is not None and saved_metadata["data_sha256"] != fingerprint:
                raise ValueError("Saved training manifest/checkpoint data fingerprint mismatch")
        summary = json.loads((run_dir / "summary.json").read_text())
        history = summary.get("history", loaded.payload.get("history", []))
        model = loaded.model.to(device=device, dtype=torch.float64).eval()
        model.requires_grad_(False)
        before_model_sha256 = _model_state_sha256(model)
        samples = int(config.evaluation.samples)
        depth = int(config.training.max_rg_steps)
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
            raise RuntimeError("Frozen evaluation changed the saved model or produced parameter gradients")
        closure = compute_closure_metrics(arrays)
        coordinates = describe_coordinates(arrays)
        font_path = make_figures(arrays, coordinates, history, output,
            sites=train.configurations.shape[2], block_factor=config.model.block_factor, font=font)
        public_path = output / "encodings.npz"
        np.savez_compressed(public_path, **arrays)
        result = dict(schema_version=1, status="complete", workflow_complete=True,
            checkpoint=str(loaded.checkpoint), checkpoint_sha256=_sha256(loaded.checkpoint),
            checkpoint_epoch=loaded.payload.get("epoch"), training_cache_sha256=fingerprint,
            source_dataset=metadata.get("source_dataset"),
            evaluation_metadata_sha256=_sha256(metadata_path),
            source_manifest_sha256=metadata.get("source_manifest_sha256"),
            labels_used_in_training=False, independent_unit="disorder", confidence_intervals=None,
            encoding_estimator=contract["estimator"], encoding_protocol=contract,
            C_draws=draws, C_draw_seed=DRAW_SEED, samples=samples, device=str(device), torch_threads=1,
            model_eval=True, model_parameters_frozen=all(not parameter.requires_grad for parameter in model.parameters()),
            model_gradients_none=gradients_none, model_state_sha256_before=before_model_sha256,
            model_state_sha256_after=after_model_sha256, model_state_unchanged=True,
            scale_sizes=[train.configurations.shape[2] // config.model.block_factor ** k for k in range(depth + 1)],
            closure=closure, coordinates=coordinates,
            aggregation="Repeated (J,T) ensembles and C draws are averaged before disorder-level summaries; "
                        "same-J temperatures are repeated conditions, not independent disorder samples; "
                        "population std is descriptive, never an error bar or confidence interval",
            flow=dict(grid_points=len(arrays["R_grid"]), observed_g_support=[low, high],
                      convention="Saved R(g); displacement arrows are R(g)-g in coordinate units",
                      roots_searched=False, exponent_search=False, grid_extension=padding),
            thermal_nu=None, equilibrium_certified=False, higher_dimensional_glass_certified=False,
            qualification=QUALIFICATION, chinese_font=font_path,
            public_npz=dict(path=public_path.name, sha256=_sha256(public_path),
                contains_spins=False, contains_bonds=False, contains_model_parameters=False,
                arrays={key: list(value.shape) for key, value in arrays.items()}),
            figures=["loss.png", "R_vs_g.png", "coupling_by_scale.png", "trajectories.png"],
        )
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
