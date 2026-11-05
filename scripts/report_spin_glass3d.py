"""Frozen-model, finite-size 3D EA overlap-QCR report; no thermal certification.

Roots are descriptive crossings on the observed source envelope only. Every
inverse-temperature branch is retained separately by disorder and scale; an
unqualified disorder-average inverse map is never substituted for them.
"""
from __future__ import annotations

import argparse
import json
from math import prod
from pathlib import Path

import numpy as np
import torch

from recurrent_rg.checkpoints import isolated_rng, load_run
from recurrent_rg.config import RunConfig
from recurrent_rg.data.statistics import cube_distribution
from recurrent_rg.encoding import half_subset_theory
from recurrent_rg.runner import file_fingerprint, prepared_data
from scripts.ising_qcr_evaluation import encoding_contract, validate_encoding_arrays
from scripts.report_spin_glass1d import _model_state_sha256, compute_closure_metrics, _expectation
from scripts.prepare_spin_glass3d_qcr import SAMPLER_SCHEMA
from scripts.run_spin_glass2d import write_json


QUALIFICATION = (
    "三维零场 EA ±J 的有限尺寸、单训练种子探索；只有少量独立无序实现，未作多尺寸标度外推。"
    "每个 J 只有一组全随机初态，四真实副本由 ICM 耦合；未认证热化或副本独立性。"
    "固定副本 0/1 重叠场是训练输入，温度与随机键不监督模型。"
    "一维 g 是表示假设，不保证完整重整化耦合空间闭合。"
    "神经不动点与正斜率候选指数不能单独认证热方向、相变温度或普适热指数。"
    "完整 152 自旋环境的经验 MI 可能严重受有限样本与稀有环境偏差影响。"
)


def validate_metadata(datasets, metadata):
    if (metadata.get("schema_version") != 1 or metadata.get("dimension") != 3
            or metadata.get("overlap_pair") != [0, 1]
            or metadata.get("labels_used_in_training") is not False
            or metadata.get("equilibrium_certified") is not False
            or metadata.get("training_ensemble_count_per_disorder_and_temperature") != 1
            or metadata.get("sampler", {}).get("schema") != SAMPLER_SCHEMA):
        raise ValueError("Require new label-free 3D ICM+PT pair-01 evaluation metadata")
    bonds, identities = {}, {}
    for split, ds in datasets:
        rows = metadata.get(split, [])
        shape = list(ds.configurations.shape[2:-1])
        if (ds.configurations.ndim != 6 or ds.configurations.shape[-1] != 1
                or len(shape) != 3 or len(set(shape)) != 1
                or shape != metadata.get("lattice_shape")
                or ds.couplings is not None or ds.group_ids is None
                or ds.group_ids.shape != (len(rows), 2)
                or len(rows) != len(ds.configurations)):
            raise ValueError("3D overlap cache geometry/grouping differs from evaluation rows")
        cells, split_bonds = set(), set()
        for group, row in zip(ds.group_ids.tolist(), rows):
            j, ti = group
            if ([row.get("disorder_id"), row.get("temperature_index")] != group
                    or row.get("dimension") != 3 or row.get("lattice_shape") != shape
                    or row.get("sites") != prod(shape) or row.get("samples") != ds.configurations.shape[1]
                    or row.get("overlap_replicas") != [0, 1]
                    or row.get("equilibrium_certified") is not False):
                raise ValueError("Pair, identity, sample or geometry metadata mismatch")
            t, bond = row.get("temperature"), row.get("bond_sha256", "")
            if not isinstance(t, (int, float)) or not np.isfinite(t) or t <= 0:
                raise ValueError("Require positive finite evaluation-only temperatures")
            if (not isinstance(bond, str) or len(bond) != 64
                    or any(c not in "0123456789abcdef" for c in bond)):
                raise ValueError("Missing valid disorder fingerprint")
            if (j, ti) in cells or (j in bonds and bonds[j] != bond):
                raise ValueError("Duplicate cell or conflicting disorder fingerprint")
            cells.add((j, ti)); bonds[j] = bond; split_bonds.add(bond)
        identities[split] = (set(ds.group_ids[:, 0].tolist()), split_bonds)
    if metadata.get("split_unit") == "whole_temperature_within_fixed_disorder":
        j, bond = metadata.get("fixed_disorder_id"), metadata.get("fixed_bond_sha256")
        for split in ("train", "test"):
            if identities[split] != ({j}, {bond}):
                raise ValueError("Fixed-J report requires the same one declared J and bond fingerprint")
        train_cells = {(r["disorder_id"], r["temperature_index"]) for r in metadata["train"]}
        test_cells = {(r["disorder_id"], r["temperature_index"]) for r in metadata["test"]}
        if train_cells & test_cells:
            raise ValueError("Fixed-J temperature cells overlap")
        if {r["temperature"] for r in metadata["train"]} & {r["temperature"] for r in metadata["test"]}:
            raise ValueError("Fixed-J temperature values overlap")
    elif identities["train"][0] & identities["test"][0] or identities["train"][1] & identities["test"][1]:
        raise ValueError("Whole disorder or identical bond realization leaked between splits")


def closure_metrics(arrays, metadata):
    """Use the explicit fixed-J exception only for its declared split type."""
    if metadata.get("split_unit") == "whole_temperature_within_fixed_disorder":
        return compute_closure_metrics(arrays, split_metadata=metadata)
    return compute_closure_metrics(arrays)


@torch.no_grad()
def collect(model, datasets, metadata, device, samples, draws, depth):
    datasets = list(datasets)
    validate_metadata(datasets, metadata)
    if type(draws) is not int or draws < 1 or type(depth) is not int or depth < 1:
        raise ValueError("Require positive C draws and RG depth")
    contract = encoding_contract(samples, split_seed=20261019)
    contract["source_sample_selection"] = "first requested saved times in every overlap ensemble, not all saved configurations"
    partition = torch.tensor(contract["sample_partition"], dtype=torch.long, device=device)
    arrays = {"sample_partition": partition.cpu().numpy()}
    for split, ds in datasets:
        if samples > ds.configurations.shape[1]:
            raise ValueError("Evaluation sample count exceeds saved overlap configurations")
        chunks = {}
        for start in range(0, len(ds.configurations), 4):
            batch = ds.configurations[start:start + 4, :samples].to(device=device, dtype=torch.float64)
            current = {}
            for _ in range(draws):
                levels = model.coarse_grain(batch, max_rg_steps=depth)
                half = half_subset_theory(model, [level.index_select(1, partition) for level in levels])
                full = torch.stack([model.theory_encoder(level) for level in levels], dim=1)
                for prefix, direct in (("", half), ("full_ensemble_", full)):
                    rollout = [direct[:, 0]]
                    for _ in range(depth):
                        rollout.append(model.advance_theory(rollout[-1]))
                    for key, value in dict(direct=direct, R_g=model.advance_theory(direct),
                            rollout=torch.stack(rollout, dim=1)).items():
                        current.setdefault(prefix + key, []).append(value.cpu())
                target = torch.stack([cube_distribution(level) for level in levels], dim=1)
                reconstruction = model.theory_encoder.distribution_decoder(half)
                if target.shape != reconstruction.shape or target.shape[-1] != 256:
                    raise ValueError("Anchored-cube affine decoder must reconstruct 256 oriented motifs")
                current.setdefault("motif_target", []).append(target.cpu())
                current.setdefault("motif_reconstruction", []).append(reconstruction.cpu())
            for key, values in current.items():
                chunks.setdefault(key, []).append(torch.stack(values, dim=1))
        for key, chunks_for_key in chunks.items():
            arrays[f"{key}_draws_{split}"] = torch.cat(chunks_for_key).numpy()
            arrays[f"{key}_{split}"] = arrays[f"{key}_draws_{split}"].mean(axis=1)
        for prefix in ("", "full_ensemble_"):
            mean = torch.from_numpy(arrays[f"{prefix}direct_{split}"]).to(device)
            arrays[f"{prefix}R_of_mean_g_{split}"] = model.advance_theory(mean).cpu().numpy()
        arrays[f"Q_full_minus_half_draws_{split}"] = arrays[f"full_ensemble_direct_draws_{split}"] - arrays[f"direct_draws_{split}"]
        arrays[f"Q_full_minus_half_{split}"] = arrays[f"Q_full_minus_half_draws_{split}"].mean(axis=1)
        arrays[f"temperatures_{split}"] = np.array([r["temperature"] for r in metadata[split]], dtype=np.float64)
        arrays[f"inverse_temperatures_{split}"] = 1 / arrays[f"temperatures_{split}"]
        arrays[f"group_ids_{split}"] = ds.group_ids.numpy().copy()
        arrays[f"bond_sha256_{split}"] = np.array([r["bond_sha256"] for r in metadata[split]], dtype="U64")
    validate_encoding_arrays(dict(schema_version=2, samples=samples, C_draws=draws,
        encoding_estimator=contract["estimator"], encoding_protocol=contract), arrays)
    return arrays, contract


def inverse_temperature_branches(arrays, g):
    """Retain every piecewise-linear crossing separately for each J and k.

    Beta is the interpolation coordinate, not a claim of exact beta recursion.
    No monotonicity assumption or averaging over disorders is used.
    """
    result = []
    for split in ("train", "test"):
        groups = arrays[f"group_ids_{split}"]
        for disorder in np.unique(groups[:, 0]):
            selection = np.flatnonzero(groups[:, 0] == disorder)
            order = np.argsort(arrays[f"inverse_temperatures_{split}"][selection])
            selection = selection[order]
            beta = arrays[f"inverse_temperatures_{split}"][selection]
            direct = arrays[f"direct_{split}"][selection, :, 0]
            if len(np.unique(beta)) != len(beta):
                raise ValueError("A disorder has duplicate inverse-temperature cells")
            for scale in range(direct.shape[1]):
                values, crossings = direct[:, scale], []
                difference = np.diff(direct[:, scale])
                for i in range(len(beta) - 1):
                    a, b = float(values[i]), float(values[i + 1])
                    if a == b:
                        if a == g:
                            crossings.append(dict(inverse_temperature_interval=[float(beta[i]), float(beta[i + 1])],
                                temperature_interval=[float(1 / beta[i + 1]), float(1 / beta[i])],
                                ambiguous_flat_segment=True))
                    elif min(a, b) <= g <= max(a, b):
                        at = float(beta[i] + (g - a) / (b - a) * (beta[i + 1] - beta[i]))
                        if not any(abs(at - c.get("inverse_temperature", float("inf"))) < 1e-12 for c in crossings):
                            crossings.append(dict(inverse_temperature=at, temperature=float(1 / at),
                                bracket_temperature_indices=groups[selection[i:i + 2], 1].tolist(),
                                bracket_inverse_temperature=[float(beta[i]), float(beta[i + 1])],
                                ambiguous_flat_segment=False))
                result.append(dict(split=split, disorder_id=int(disorder), scale=scale,
                    strictly_monotone=bool(np.all(difference > 0) or np.all(difference < 0)),
                    all_piecewise_linear_crossings=crossings,
                    observed_g_range=[float(values.min()), float(values.max())]))
    return result


def root_candidates(model, grid, mapped, arrays, device):
    delta, flat = (mapped - grid).reshape(-1), grid.reshape(-1)
    intervals = [(float(flat[i]), float(flat[i + 1])) for i in np.flatnonzero(delta[:-1] * delta[1:] < 0)]
    intervals += [(float(flat[i]), float(flat[i])) for i in np.flatnonzero(delta == 0)]
    candidates = []
    def displacement(g):
        with torch.no_grad():
            x = torch.tensor([[g]], dtype=torch.float64, device=device)
            return float((model.advance_theory(x) - x).item())
    for low, high in intervals:
        bracket = [low, high]
        left = displacement(low)
        for _ in range(60 if high > low else 0):
            mid = .5 * (low + high)
            value = displacement(mid)
            if left * value <= 0:
                high = mid
            else:
                low, left = mid, value
        root = .5 * (low + high)
        if any(abs(root - r["g"]) < 1e-9 for r in candidates):
            continue
        x = torch.tensor([[root]], dtype=torch.float64, device=device, requires_grad=True)
        slope = float(torch.autograd.grad(model.advance_theory(x).sum(), x)[0].item())
        candidates.append(dict(g=root, R_derivative=slope, initial_grid_bracket=bracket,
            displacement_residual=displacement(root),
            stability="attracting" if abs(slope) < 1 else "repelling" if abs(slope) > 1 else "marginal",
            candidate_nu=float(np.log(2) / np.log(slope)) if slope > 1 else None,
            thermal_identification_certified=False,
            nearest_observed_source_g_distance=float(min(np.min(abs(arrays[f"direct_{s}"][:, :-1] - root)) for s in ("train", "test"))),
            inverse_temperature_branches=inverse_temperature_branches(arrays, root)))
    return dict(candidates=candidates, thermal_nu=None, search_support=[float(flat.min()), float(flat.max())],
        TRAIN_source_g_envelope=[float(arrays["direct_train"][:, :-1].min()), float(arrays["direct_train"][:, :-1].max())],
        selection="All observed-source-envelope sign-change brackets and exact grid zeros; none selected by theory or stability",
        search_method="1001-point grid with bracket bisection; can miss tangent roots or close pairs; envelope need not be densely supported",
        inverse_map="Every disorder and every scale separately, all beta=1/T linear segments retained; no inverse of a silently averaged g(T)",
        qualification="Positive-slope candidate nu is descriptive only, not a certified thermal eigenvalue or critical temperature")


def coordinate_diagnostics(arrays):
    result = {}
    for split in ("train", "test"):
        groups = arrays[f"group_ids_{split}"]
        by_t = []
        for temperature in np.unique(arrays[f"temperatures_{split}"]):
            indices = np.flatnonzero(arrays[f"temperatures_{split}"] == temperature)
            values = arrays[f"direct_{split}"][indices]
            by_t.append(dict(temperature=float(temperature), inverse_temperature=float(1 / temperature),
                disorder_ids=groups[indices, 0].tolist(), per_disorder_g=values.tolist(),
                mean_g=values.mean(axis=0).tolist(), disorder_population_std=values.std(axis=0, ddof=0).tolist()))
        gap = arrays[f"Q_full_minus_half_draws_{split}"]
        jensen = arrays[f"R_g_{split}"] - arrays[f"R_of_mean_g_{split}"]
        motif = arrays[f"motif_reconstruction_draws_{split}"] - arrays[f"motif_target_draws_{split}"]
        result[split] = dict(independent_disorder_count=len(np.unique(groups[:, 0])), overlap_ensemble_count=len(groups),
            by_temperature=by_t, Q_full_minus_half_RMSE_by_scale=np.sqrt(_expectation(np.mean(gap ** 2, axis=(1, 3)), groups)).tolist(),
            mean_R_minus_R_of_mean_g_RMSE_by_scale=np.sqrt(_expectation(np.mean(jensen ** 2, axis=2), groups)).tolist(),
            cube_256_motif_reconstruction_per_bin_RMSE_by_scale=np.sqrt(_expectation(np.mean(motif ** 2, axis=(1, 3)), groups)).tolist(),
            uncertainty="Descriptive population spread over the listed J only; no disorder or training-seed confidence interval")
    return result


def mi_null_diagnostic(model, dataset, metadata, device):
    """Predeclared first-J low/middle/high ladder cells; not selected by Tc.

    Visible labels are permuted within the same observations of each cell,
    retaining identical E and V marginals and the same learned C table.
    """
    from recurrent_rg.empirical_mi import lattice_mi_statistics
    from recurrent_rg.rsmi import kernel_probabilities
    rows = metadata["train"]
    first_j = min(r["disorder_id"] for r in rows)
    cells = sorted((r["temperature_index"], i) for i, r in enumerate(rows) if r["disorder_id"] == first_j)
    selected = [cells[k][1] for k in sorted({0, len(cells) // 2, len(cells) - 1})]
    samples = min(64, dataset.configurations.shape[1])
    x = dataset.configurations[selected, :samples]
    original = lattice_mi_statistics(x, states=2, block=2, buffer=1)
    shuffled = lattice_mi_statistics(x, states=2, block=2, buffer=1, shuffle_visible_seed=20261019)
    code = torch.arange(256, device=device)[:, None]
    patterns = (2 * ((code // (2 ** torch.arange(8, device=device))) % 2) - 1).double()[None]
    with torch.no_grad():
        table = kernel_probabilities(model.rg_kernel, patterns)[0].cpu()
        regular, null = original.information(table) / np.log(2), shuffled.information(table) / np.log(2)
    return dict(TRAIN_ensemble_indices=selected, samples_per_overlap_ensemble=samples,
        evaluated_cells=[{k: rows[i][k] for k in ("disorder_id", "temperature_index", "temperature")} for i in selected],
        predeclared_selection="Smallest TRAIN J, positions 0, N_TRAIN//2, N_TRAIN-1 in its sorted TRAIN-cell list; evaluated_cells gives original ladder indices; no theory/Tc or observed-MI selection",
        all_translations=True, translations_per_configuration=prod(dataset.configurations.shape[2:-1]),
        observation_counts_by_cell=original.diagnostics["observations_by_ensemble"],
        learned_C_MI_bits=regular.tolist(), shuffled_V_MI_bits=null.tolist(),
        diagnostics=original.diagnostics, shuffled_diagnostics=shuffled.diagnostics,
        scope="Only listed three TRAIN (J,T) overlap cells and first <=64 saved configurations; not full disorder/temperature domain",
        note="Descriptive matched finite-cache null, not a training loss or population MI; near-unique E can make shuffled MI high too")


def figures(arrays, history, output, *, font=None, fixed_disorder=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ft2font import FT2Font
    from matplotlib.text import Text
    from scripts.report_ising_zh import select_chinese_font
    font_path = select_chinese_font(font)
    available = set(FT2Font(str(font_path)).get_charmap())
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font_path)).get_name(), "axes.unicode_minus": False})
    def save(fig, name):
        for text in fig.findobj(match=Text):
            missing = {c for c in text.get_text() if "\u3400" <= c <= "\u9fff" and ord(c) not in available}
            if missing:
                raise ValueError(f"中文字体缺少字形：{''.join(sorted(missing))}")
        fig.tight_layout(); fig.savefig(output / name, dpi=160); plt.close(fig)
    grid, mapped = arrays["R_grid"].reshape(-1), arrays["R_mapped"].reshape(-1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(grid, mapped, label="学习映射"); axes[0].plot(grid, grid, "--", label="恒等映射")
    axes[0].set(xlabel="$g$", ylabel="$R(g)$", title="三维自旋玻璃重整化映射")
    selected = np.linspace(0, len(grid) - 1, 30).astype(int)
    axes[1].quiver(grid[selected], np.zeros(len(selected)), mapped[selected] - grid[selected], np.zeros(len(selected)),
        angles="xy", scale_units="xy", scale=1)
    axes[1].plot(grid, mapped - grid, label="实际位移"); axes[1].axhline(0, linestyle="--", color="gray")
    xmin, xmax = min(grid.min(), mapped.min()), max(grid.max(), mapped.max())
    margin = max(1e-4, .05 * (xmax - xmin)); axes[1].set_xlim(xmin - margin, xmax + margin)
    axes[1].set(xlabel="$g$", ylabel="$R(g)-g$", title="真实坐标位移（箭头未归一化）")
    axes[0].legend(); axes[1].legend(); save(fig, "R_vs_g.png")
    levels = arrays["direct_train"].shape[1]
    fig, axes = plt.subplots(1, levels, figsize=(4.5 * levels, 4))
    for split, marker, color in (("train", "o", "C0"), ("test", "x", "C1")):
        label = (("训练温度（同一 J）" if split == "train" else "留出温度（同一 J）")
                 if fixed_disorder else ("训练无序实现" if split == "train" else "留出无序实现"))
        groups = arrays[f"group_ids_{split}"]
        for scale, ax in enumerate(np.atleast_1d(axes)):
            for count, j in enumerate(np.unique(groups[:, 0])):
                chosen = np.flatnonzero(groups[:, 0] == j)
                chosen = chosen[np.argsort(arrays[f"inverse_temperatures_{split}"][chosen])]
                ax.plot(arrays[f"inverse_temperatures_{split}"][chosen], arrays[f"direct_{split}"][chosen, scale, 0],
                    color=color, marker=marker, markersize=3, linewidth=.7, alpha=.65, label=label if count == 0 else None)
            ax.set(xlabel="$\\beta=1/T$（仅评价）", ylabel="$g$", title=f"尺度 k={scale}（每条线为一个 J）")
            ax.legend()
    save(fig, "coupling_by_scale.png")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, metric, name in zip(axes.flat,
            ("separation", "rsmi", "kernel_noncollapse", "kernel_sign", "commutator", "rollout"),
            ("Q 立方体概率重建", "C 负经验互信息", "C 方差下限", "C 颜色保持", "R 一步闭合", "R 多步闭合")):
        for split, label in (("train", "训练"), ("test", "留出")):
            ax.plot([r["epoch"] for r in history], [r[f"{split}_{metric}"] for r in history], label=label)
        ax.set(xlabel="优化步数", ylabel="损失", title=name); ax.legend()
    save(fig, "loss.png")
    return str(font_path)


def report(root, *, device="cpu", draws=5, font=None):
    root = Path(root).resolve(); output = root / "report"; output.mkdir(exist_ok=False)
    write_json(output / "status.json", dict(status="running", workflow_complete=False))
    try:
        torch.set_num_threads(1)
        cfg = RunConfig.load(root / "config.json")
        if (cfg.model.dimension, cfg.model.width, cfg.model.block_factor, cfg.model.theory_dim) != (3, 2, 2, 1):
            raise ValueError("Require explicit scalar 3D anchored-cube b2 model")
        train, test = prepared_data(cfg)
        metadata = json.loads((root / "data/evaluation_metadata.json").read_text())
        if metadata.get("cache_sha256") != file_fingerprint(cfg.data.cache):
            raise ValueError("Evaluation metadata does not bind actual cache fingerprint")
        loaded = load_run(root / "run", checkpoint="checkpoint_last.pt")
        if loaded.payload["metadata"]["data_sha256"] != metadata["cache_sha256"]:
            raise ValueError("Training checkpoint data fingerprint mismatch")
        saved = RunConfig.from_dict(loaded.config)
        if json.loads(json.dumps(saved.to_dict())) != json.loads(json.dumps(cfg.to_dict())):
            raise ValueError("Checkpoint architecture/objective differs from root config")
        model = loaded.model.to(device=device, dtype=torch.float64).eval().requires_grad_(False)
        before = _model_state_sha256(model)
        with isolated_rng(device, 20261019), torch.no_grad():
            arrays, contract = collect(model, [("train", train), ("test", test)], metadata, device,
                cfg.evaluation.samples, draws, cfg.training.max_rg_steps)
            source = np.concatenate([arrays[f"direct_{s}"][:, :-1].reshape(-1) for s in ("train", "test")])
            grid = torch.linspace(float(source.min()), float(source.max()), 1001, device=device, dtype=torch.float64)[:, None]
            arrays["R_grid"] = grid.cpu().numpy(); arrays["R_mapped"] = model.advance_theory(grid).cpu().numpy()
            arrays["R_displacement"] = arrays["R_mapped"] - arrays["R_grid"]
            null = mi_null_diagnostic(model, train, metadata, device)
        # Parameter gradients remain disabled; scalar input derivatives are only descriptive.
        roots = root_candidates(model, arrays["R_grid"], arrays["R_mapped"], arrays, device)
        unchanged = before == _model_state_sha256(model) and all(p.grad is None for p in model.parameters())
        if not unchanged:
            raise RuntimeError("Report changed model state or parameter gradients")
        summary = json.loads((root / "run/summary.json").read_text())
        fixed_disorder = metadata.get("split_unit") == "whole_temperature_within_fixed_disorder"
        font_path = figures(arrays, summary["history"], output, font=font, fixed_disorder=fixed_disorder)
        np.savez_compressed(output / "encodings.npz", **arrays)
        result = dict(status="complete", workflow_complete=True, dimension=3, b=2, g_dimension=1,
            physical_model="zero-field simple-cubic Edwards-Anderson bimodal +/-J spin glass",
            actual_training_updates=loaded.payload["epoch"], labels_used_in_training=False,
            model_state_unchanged=True, model_gradients_none=True, model_sha256=before,
            C_draws=draws, samples=cfg.evaluation.samples,
            saved_source_samples_per_overlap_ensemble=train.configurations.shape[1],
            evaluation_sample_selection="First requested saved times per overlap ensemble; not all 512 source configurations",
            encoding_protocol=contract, closure=closure_metrics(arrays, metadata), coordinates=coordinate_diagnostics(arrays),
            fixed_point_candidates=roots, thermal_nu=None, overlap_pair=[0, 1],
            uncertainty_unit=("One fixed disorder only; no disorder uncertainty or independent-chain CI inferred"
                if fixed_disorder else "Whole quenched disorder, not temperature, replica pair, C draw or saved time"),
            C_reduction="Mean across C draws before primary RMSE; mean R(g_draw) and R(mean g) saved separately",
            MI_shuffle_diagnostic=null, equilibrium_certified=False,
            split_unit=metadata.get("split_unit"), fixed_disorder_id=metadata.get("fixed_disorder_id"),
            qualification=metadata["qualification"] if fixed_disorder else QUALIFICATION,
            scale_shapes=[[n // (2 ** k) for n in train.configurations.shape[2:-1]] for k in range(cfg.training.max_rg_steps + 1)],
            font_path=font_path, figures=["R_vs_g.png", "coupling_by_scale.png", "loss.png"],
            public_arrays_sha256=file_fingerprint(output / "encodings.npz"))
        write_json(output / "report.json", result)
        write_json(output / "status.json", dict(status="complete", workflow_complete=True, thermal_nu=None))
        return result
    except Exception as exc:
        write_json(output / "status.json", dict(status="failed", workflow_complete=False, error=str(exc)))
        raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True); p.add_argument("--device", default="cpu")
    p.add_argument("--draws", default=5, type=int); p.add_argument("--font", type=Path)
    args = p.parse_args(argv)
    result = report(args.root, device=args.device, draws=args.draws, font=args.font)
    print(json.dumps(dict(status=result["status"], thermal_nu=None), ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
