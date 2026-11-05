"""Frozen-model Chinese report for the first finite-size 3D Ising QCR run.

All physical couplings are evaluation-only sidecar data. Root slopes are
descriptive neural candidates, never certified universal thermal exponents.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from recurrent_rg.checkpoints import isolated_rng, load_run
from recurrent_rg.config import RunConfig
from recurrent_rg.data.statistics import cube_distribution
from recurrent_rg.encoding import half_subset_theory
from recurrent_rg.runner import file_fingerprint, prepared_data
from scripts.ising_qcr_evaluation import encoding_contract, validate_encoding_arrays
from scripts.report_spin_glass1d import _model_state_sha256
from scripts.run_spin_glass2d import write_json

QUALIFICATION = ("单一训练种子的有限尺寸探索；未进行多尺寸标度外推或平衡认证。"
    "一维 g 为表示假设，不保证粗粒化生成的全部耦合闭合。"
    "神经不动点斜率仅给候选标度量，不能单独认证热方向或普适热指数。"
    "完整152自旋环境的经验互信息可能严重受稀有环境和有限样本偏差影响。")


def mi_diagnostic_policy(manifest, contract, checkpoint_metadata, *, skip_mi_diagnostic=False):
    """Bind the no-MI report mode to all three saved objective contracts.

    The ablation must not construct statistics merely to fill a plotting field.
    Requiring an explicit skip flag also catches an incorrectly wired runner
    before model/data evaluation starts.
    """
    weights = [record.get("rsmi_weight") for record in (manifest, contract, checkpoint_metadata)]
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 for value in weights):
        raise ValueError("Missing or invalid saved rsmi_weight objective contract")
    if len(set(weights)) != 1:
        raise ValueError("Manifest, summary and checkpoint disagree on rsmi_weight")
    weight = float(weights[0])
    if weight == 0 and not skip_mi_diagnostic:
        raise ValueError("Zero-RSMI ablation requires --skip-mi-diagnostic; no MI will be computed")
    if skip_mi_diagnostic and weight != 0:
        raise ValueError("--skip-mi-diagnostic is reserved for the explicit rsmi_weight=0 ablation")
    return dict(rsmi_weight=weight, enabled=weight > 0, computed=not skip_mi_diagnostic,
        reason=("rsmi_weight=0 的显式无互信息消融：未构建、加载或评价任何 MI 统计；未计算，不是实测为零。"
            if skip_mi_diagnostic else "固定 TRAIN 子集上的描述性配对置乱诊断，不是总体互信息。"))


def closure_metrics(arrays):
    std = arrays["direct_train"][:, 1:, :].std(axis=0, ddof=0).reshape(-1)
    result = dict(TRAIN_target_std_by_scale=std.tolist(), normalization="Original final-model TRAIN target population std, no gauge rescaling")
    for split in ("train", "test"):
        direct = arrays[f"direct_{split}"]
        error = arrays[f"R_g_{split}"][:, :-1] - direct[:, 1:]
        rollout = arrays[f"rollout_{split}"][:, 1:] - direct[:, 1:]
        one_rmse = np.sqrt(np.mean(error * error, axis=(0, 2)))
        multi_rmse = np.sqrt(np.mean(rollout * rollout, axis=(0, 2)))
        result[split] = dict(one_step_RMSE=one_rmse.tolist(), rollout_RMSE=multi_rmse.tolist(),
            one_step_NRMSE=[float(r / s) if s > 0 else None for r, s in zip(one_rmse, std)],
            rollout_NRMSE=[float(r / s) if s > 0 else None for r, s in zip(multi_rmse, std)])
    return result


def root_candidates(model, grid, mapped, arrays, device):
    """Bracket all crossings in observed one-step-source support, not near theory.

    No root is selected by Kc, closeness to a known eigenvalue, or attraction.
    The interpolation is at k=0; it is not an exact Hamiltonian recursion.
    """
    delta = mapped.reshape(-1) - grid.reshape(-1)
    roots = []
    brackets = np.flatnonzero(delta[:-1] * delta[1:] < 0).tolist()
    exact = np.flatnonzero(delta == 0).tolist()
    flat_grid = grid.reshape(-1)
    intervals = [(float(flat_grid[i]), float(flat_grid[i + 1])) for i in brackets]
    intervals += [(float(flat_grid[i]), float(flat_grid[i])) for i in exact]
    labels = np.concatenate([arrays[f"K_{s}"] for s in ("train", "test")])
    direct = np.concatenate([arrays[f"direct_{s}"][:, 0, 0] for s in ("train", "test")])
    kvals = np.unique(labels)
    means = np.array([direct[labels == k].mean() for k in kvals])
    difference = np.diff(means)
    monotone = bool(np.all(difference > 0) or np.all(difference < 0))
    for low, high in intervals:
        bracket = [low, high]
        def displacement(x):
            with torch.no_grad():
                t = torch.tensor([[x]], dtype=torch.float64, device=device)
                return float((model.advance_theory(t) - t).item())
        left = displacement(low)
        for _ in range(60 if high > low else 0):
            middle = .5 * (low + high)
            value = displacement(middle)
            if left * value <= 0:
                high = middle
            else:
                low, left = middle, value
        root = .5 * (low + high)
        if any(abs(root - row["g"]) < 1e-9 for row in roots):
            continue
        x = torch.tensor([[root]], dtype=torch.float64, device=device, requires_grad=True)
        value = model.advance_theory(x)
        slope = float(torch.autograd.grad(value.sum(), x)[0].item())
        interpolation = None
        if monotone and means.min() <= root <= means.max():
            order = np.argsort(means)
            interpolation = float(np.interp(root, means[order], kvals[order]))
        roots.append(dict(g=root, R_derivative=slope, initial_grid_bracket=bracket,
            displacement_residual=displacement(root),
            nearest_observed_source_g_distance=float(min(np.min(abs(arrays[f"direct_{s}"][:, :-1] - root)) for s in ("train", "test"))),
            stability="attracting" if abs(slope) < 1 else "repelling" if abs(slope) > 1 else "marginal",
            k0_interpolated_K=interpolation,
            candidate_nu=float(np.log(2) / np.log(slope)) if slope > 1 else None,
            thermal_identification_certified=False))
    return dict(candidates=roots, k0_mean_gK_strictly_monotone=monotone,
        search_support=[float(grid.min()), float(grid.max())], selection="All bracketed crossings in observed source envelope; none selected by theoretical Kc/lambda",
        TRAIN_source_g_envelope=[float(arrays["direct_train"][:, :-1].min()), float(arrays["direct_train"][:, :-1].max())],
        search_method="Sign-change grid brackets and exact grid zeros inside observed source envelope; can miss tangent roots or close root pairs; envelope alone is not dense physical support",
        thermal_nu=None, qualification="Candidate nu only for positive R derivative>1; not a certified thermal eigenvalue")


def collect(model, datasets, metadata, device, samples, draws, depth):
    contract = encoding_contract(samples, split_seed=20261019)
    partition = torch.tensor(contract["sample_partition"], dtype=torch.long, device=device)
    arrays = {"sample_partition": partition.cpu().numpy()}
    for split, ds in datasets:
        rows = metadata[split]
        if ds.couplings is not None or len(rows) != len(ds.configurations):
            raise ValueError("Expected label-free training cache and aligned evaluation rows")
        if (metadata.get("dimension") != 3 or metadata.get("lattice_shape") != list(ds.configurations.shape[2:-1])
                or ds.group_ids is None or ds.group_ids.reshape(-1).tolist() != [r["coupling_index"] for r in rows]
                or any(r["samples"] != ds.configurations.shape[1] or r["size"] != ds.configurations.shape[2] for r in rows)):
            raise ValueError("Evaluation row geometry/order differs from training cache")
        chunks = {}
        for start in range(0, len(ds.configurations), 4):
            batch = ds.configurations[start:start + 4, :samples].to(device=device, dtype=torch.float64)
            current = {}
            for _ in range(draws):
                levels = model.coarse_grain(batch, max_rg_steps=depth)
                direct = half_subset_theory(model, [level.index_select(1, partition) for level in levels])
                full = torch.stack([model.theory_encoder(level) for level in levels], dim=1)
                rollout = [direct[:, 0]]
                full_rollout = [full[:, 0]]
                for _ in range(depth):
                    rollout.append(model.advance_theory(rollout[-1]))
                    full_rollout.append(model.advance_theory(full_rollout[-1]))
                target = torch.stack([cube_distribution(level) for level in levels], dim=1)
                for key, value in dict(direct=direct, full_ensemble_direct=full,
                        R_g=model.advance_theory(direct), rollout=torch.stack(rollout, dim=1),
                        full_ensemble_R_g=model.advance_theory(full),
                        full_ensemble_rollout=torch.stack(full_rollout, dim=1),
                        motif_target=target, motif_reconstruction=model.theory_encoder.distribution_decoder(direct)).items():
                    current.setdefault(key, []).append(value.cpu())
            for key, values in current.items():
                chunks.setdefault(key, []).append(torch.stack(values, dim=1))
        for key, values in chunks.items():
            arrays[f"{key}_draws_{split}"] = torch.cat(values).numpy()
            arrays[f"{key}_{split}"] = arrays[f"{key}_draws_{split}"].mean(axis=1)
        arrays[f"R_of_mean_g_{split}"] = model.advance_theory(torch.from_numpy(arrays[f"direct_{split}"]).to(device)).cpu().numpy()
        arrays[f"K_{split}"] = np.array([row["K"] for row in rows], dtype=np.float64)
        arrays[f"condition_index_{split}"] = np.array([row["coupling_index"] for row in rows], dtype=np.int64)
        arrays[f"Q_full_minus_half_draws_{split}"] = arrays[f"full_ensemble_direct_draws_{split}"] - arrays[f"direct_draws_{split}"]
        arrays[f"Q_full_minus_half_{split}"] = arrays[f"Q_full_minus_half_draws_{split}"].mean(axis=1)
    validate_encoding_arrays(dict(schema_version=2, samples=samples, C_draws=draws,
        encoding_estimator=contract["estimator"], encoding_protocol=contract), arrays)
    return arrays, contract


def mi_null_diagnostic(model, dataset, device):
    """A fixed, tiny TRAIN-only descriptive shuffle control; no fitting.

    Same observations, V/E regions and learned C table in both statistics.
    Its sample limit is disclosed and it never replaces training full-cache MI.
    """
    from recurrent_rg.empirical_mi import lattice_mi_statistics
    from recurrent_rg.rsmi import kernel_probabilities
    x = dataset.configurations[:2, :64]
    kwargs = dict(states=2, block=2, buffer=1)
    original = lattice_mi_statistics(x, **kwargs)
    shuffled = lattice_mi_statistics(x, shuffle_visible_seed=20261019, **kwargs)
    code = torch.arange(256, device=device)[:, None]
    patterns = (2 * ((code // (2 ** torch.arange(8, device=device))) % 2) - 1).double()[None]
    with torch.no_grad():
        table = kernel_probabilities(model.rg_kernel, patterns)[0].cpu()
        regular = original.information(table) / np.log(2)
        null = shuffled.information(table) / np.log(2)
    return dict(TRAIN_ensemble_indices=[0, 1], samples_per_chain=min(64, dataset.configurations.shape[1]),
        all_translations=True, learned_C_MI_bits=regular.tolist(), shuffled_V_MI_bits=null.tolist(),
        diagnostics=original.diagnostics, shuffled_diagnostics=shuffled.diagnostics,
        note="Descriptive matched finite-cache null, not a loss, not population MI; near-unique E can make shuffled MI high too")


def figures(arrays, history, output, *, font=None, mi_enabled=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from scripts.report_ising_zh import select_chinese_font
    font_path = select_chinese_font(font)
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font_path)).get_name(), "axes.unicode_minus": False})
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(output / name, dpi=160)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    grid, mapped = arrays["R_grid"].reshape(-1), arrays["R_mapped"].reshape(-1)
    axes[0].plot(grid, mapped, label="学习映射")
    axes[0].plot(grid, grid, "--", label="恒等映射")
    axes[0].set(xlabel="$g$", ylabel="$R(g)$", title="三维 Ising 重整化映射")
    selected = np.linspace(0, len(grid) - 1, 30).astype(int)
    axes[1].quiver(grid[selected], np.zeros(len(selected)), mapped[selected] - grid[selected], np.zeros(len(selected)),
        angles="xy", scale_units="xy", scale=1)
    axes[1].plot(grid, mapped - grid, label="实际位移")
    axes[1].axhline(0, color="gray", linestyle="--")
    # Keep actual arrow endpoints visible even when the map leaves the data
    # envelope. This is display extent only, never extra model evaluation.
    xmin, xmax = min(grid.min(), mapped.min()), max(grid.max(), mapped.max())
    margin = max(1e-4, .05 * (xmax - xmin))
    axes[1].set_xlim(xmin - margin, xmax + margin)
    axes[1].set(xlabel="$g$", ylabel="$R(g)-g$", title="真实坐标位移（箭头未归一化）")
    axes[0].legend(); axes[1].legend()
    save(fig, "R_vs_g.png")
    levels = arrays["direct_train"].shape[1]
    fig, axes = plt.subplots(1, levels, figsize=(4.3 * levels, 4))
    axes = np.atleast_1d(axes)
    for split, marker in (("train", "o"), ("test", "x")):
        label = "训练" if split == "train" else "留出"
        k = arrays[f"K_{split}"]
        for level, ax in enumerate(axes):
            ax.scatter(k, arrays[f"direct_{split}"][:, level, 0], s=14, marker=marker, label=label)
            ax.set(xlabel="$K$（仅评价）", ylabel="$g$", title=f"尺度 k={level}")
            ax.legend()
    save(fig, "coupling_by_scale.png")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, metric, name in zip(axes.flat,
            ("separation", "rsmi", "kernel_noncollapse", "kernel_sign", "commutator", "rollout"),
            ("Q 局部概率重建", "C 负经验互信息", "C 方差下限", "C 颜色保持", "R 一步闭合", "R 多步闭合")):
        if metric == "rsmi" and not mi_enabled:
            ax.text(.5, .5, "未启用\n未计算 MI（不是实测零）", ha="center", va="center", transform=ax.transAxes)
            ax.set(xlabel="优化步数", ylabel="损失", title=name + "（未启用）")
            continue
        for split, label in (("train", "训练"), ("test", "留出")):
            ax.plot([row["epoch"] for row in history], [row[f"{split}_{metric}"] for row in history], label=label)
        ax.set(xlabel="优化步数", ylabel="损失", title=name)
        ax.legend()
    save(fig, "loss.png")
    return str(font_path)


def report(root, *, device="cpu", draws=5, font=None, skip_mi_diagnostic=False):
    root = Path(root).resolve()
    output = root / "report"
    output.mkdir(exist_ok=False)
    write_json(output / "status.json", dict(status="running", workflow_complete=False))
    try:
        if type(draws) is not int or draws < 1:
            raise ValueError("draws must be a positive integer")
        manifest = json.loads((root / "run/manifest.json").read_text())
        summary = json.loads((root / "run/summary.json").read_text())
        torch.set_num_threads(1)
        cfg = RunConfig.load(root / "config.json")
        if (cfg.model.dimension, cfg.model.width, cfg.model.block_factor, cfg.model.theory_dim) != (3, 2, 2, 1):
            raise ValueError("Report requires explicit anchored-cube 3D b2 scalar model")
        train, test = prepared_data(cfg)
        metadata = json.loads((root / "data/evaluation_metadata.json").read_text())
        if metadata.get("cache_sha256") != file_fingerprint(cfg.data.cache):
            raise ValueError("Evaluation metadata must bind to the actual label-free cache")
        loaded = load_run(root / "run", checkpoint="checkpoint_last.pt")
        mi_policy = mi_diagnostic_policy(manifest, summary.get("contract", {}),
            loaded.payload["metadata"], skip_mi_diagnostic=skip_mi_diagnostic)
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
            arrays["R_grid"] = grid.cpu().numpy()
            arrays["R_mapped"] = model.advance_theory(grid).cpu().numpy()
            arrays["R_displacement"] = arrays["R_mapped"] - arrays["R_grid"]
        roots = root_candidates(model, arrays["R_grid"], arrays["R_mapped"], arrays, device)
        null = None if skip_mi_diagnostic else mi_null_diagnostic(model, train, device)
        unchanged = before == _model_state_sha256(model) and all(p.grad is None for p in model.parameters())
        if not unchanged:
            raise RuntimeError("Report changed model or parameter gradients")
        font_path = figures(arrays, summary["history"], output, font=font, mi_enabled=mi_policy["enabled"])
        np.savez_compressed(output / "encodings.npz", **arrays)
        result = dict(status="complete", workflow_complete=True, dimension=3, b=2, g_dimension=1,
            actual_training_updates=loaded.payload["epoch"], labels_used_in_training=False,
            model_state_unchanged=True, model_gradients_none=True, model_sha256=before,
            C_draws=draws, samples=cfg.evaluation.samples, encoding_protocol=contract,
            closure=closure_metrics(arrays), fixed_point_candidates=roots, thermal_nu=None,
            C_reduction="Mean across C draws before primary RMSE; mean R(g_draw) and R(mean g) are saved separately",
            MI_shuffle_diagnostic=null, MI_diagnostic_policy=mi_policy,
            equilibrium_certified=False, qualification=(QUALIFICATION if mi_policy["enabled"] else
                QUALIFICATION.replace("完整152自旋环境的经验互信息可能严重受稀有环境和有限样本偏差影响。",
                    "本次消融未启用或计算互信息；保色与方差防塌缩本身不保证保留长程关联。")),
            scale_shapes=[[n // (2 ** k) for n in train.configurations.shape[2:-1]] for k in range(cfg.training.max_rg_steps + 1)],
            font_path=font_path, figures=["R_vs_g.png", "coupling_by_scale.png", "loss.png"],
            public_arrays_sha256=file_fingerprint(output / "encodings.npz"))
        write_json(output / "report.json", result)
        write_json(output / "status.json", dict(status="complete", workflow_complete=True, thermal_nu=None))
        return result
    except Exception as exc:
        write_json(output / "status.json", dict(status="failed", workflow_complete=False, error=str(exc)))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--draws", default=5, type=int)
    p.add_argument("--font", type=Path)
    p.add_argument("--skip-mi-diagnostic", action="store_true",
        help="Explicit rsmi_weight=0 ablation only: do not construct/load/evaluate MI statistics")
    args = p.parse_args()
    result = report(args.root, device=args.device, draws=args.draws, font=args.font,
        skip_mi_diagnostic=args.skip_mi_diagnostic)
    print(json.dumps(dict(status=result["status"], thermal_nu=None), ensure_ascii=False))


if __name__ == "__main__":
    main()
