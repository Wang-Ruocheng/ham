"""Paired R-only optimizer diagnosis from fixed scaled96/6000 endpoints.

Only FIT TRAIN g is decompressed before every optimization arm has finished.
The fixed Q/C is never loaded.  Original-g closure and internal normalization
are preserved. Adam updates and L-BFGS gradient evaluations are not matched
units of work; both the actual counts and the CPU costs are reported.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from experiments.phi4_qcr import sha256_file, state_fingerprint, write_json
from experiments.phi4_ru_r_calibration import endpoint_metrics, predict, PATHS
from recurrent_rg.phi4_r_calibration import (
    FrozenResidualMap, all_start_closure_loss, fit_input_normalizer,
)
from scripts.phi4_ru_empirical_flow_diagnostic import chain_and_cell_means, vector_metrics


RECIPE = "phi4_ru_fixed_QC_scaled96_endpoint_optimizer_probe_v1"
SEEDS = (91, 92, 93)
ARMS = ("noopt", "adam", "lbfgs")
NAMES = {"noopt": "原6000步端点", "adam": "重新初始化 Adam，400步",
         "lbfgs": "重新初始化 L-BFGS"}


@dataclass
class Options:
    cache: str
    calibration: str
    output: str
    cpu_budget_seconds: float = 600.
    output_budget_bytes: int = 100 * 1024 ** 2


class BudgetHold(RuntimeError):
    pass


def verify_public(root, hashes):
    """Hash public files without decompressing TEST or coupling-label arrays."""
    root = Path(root).resolve()
    status = json.loads((root / "status.json").read_text())
    if status.get("status") != "complete" or not status.get("workflow_complete"):
        raise ValueError(f"Immutable input is incomplete: {root}")
    inventory_path = root / "public_files.json"
    inventory = json.loads(inventory_path.read_text())
    for name, digest in inventory.items():
        path = (root / name).resolve()
        if path.parent != root or path.suffix not in (".json", ".npz", ".png"):
            raise ValueError("Public inventory contains an unexpected file")
        if sha256_file(path) != digest:
            raise ValueError(f"Immutable input SHA mismatch: {path}")
        hashes[str(path)] = digest
    for path in (inventory_path, root / "status.json"):
        hashes[str(path)] = sha256_file(path)
    return json.loads((root / "manifest.json").read_text())


def load_fit_only(cache):
    with np.load(Path(cache) / "cache.npz", allow_pickle=False) as archive:
        fit = archive["fit_direct"].copy()
    if fit.shape != (4, 136, 3, 2) or not np.isfinite(fit).all():
        raise ValueError("Expected four fixed TRAIN repeats of 136 chains")
    return torch.tensor(fit, dtype=torch.float64)


def restore_endpoints(calibration, summary, fit):
    """Restore exact endpoints and fixed buffers, without source Q/C/PT files."""
    models = {}
    center, scale = fit_input_normalizer(fit)
    with np.load(Path(calibration) / "R_parameters.npz", allow_pickle=False) as archive:
        for seed in SEEDS:
            key = f"scaled96_seed{seed}"
            if summary["actual_steps_by_model"][key] != 6000:
                raise ValueError("Expected the fixed6000 scaled96 endpoint")
            state = {name[len(key) + 1:]: torch.tensor(archive[name])
                     for name in archive.files if name.startswith(key + ".")}
            if state["net.0.weight"].shape != (96, 2):
                raise ValueError("The registered diagnostic requires hidden96")
            model = FrozenResidualMap(2, 96, center=state["center"], scale=state["scale"])
            model.load_state_dict(state, strict=True)
            if state_fingerprint(model) != summary["final_R_sha256"][key]:
                raise ValueError(f"Restored endpoint SHA mismatch: {key}")
            if not torch.equal(model.center, center) or not torch.equal(model.scale, scale):
                raise ValueError("Endpoint normalizer disagrees with FIT TRAIN inputs")
            models[seed] = model
    return models


def optimizer_arm(source, direct, arm, *, guard=lambda: None,
                  adam_steps=400, lbfgs_max_iter=100, lbfgs_max_eval=400,
                  on_progress=lambda row: None):
    """Return a fixed optimizer endpoint; restore initial state on interruption.

    max_eval is the PyTorch L-BFGS outer-step stopping budget. Strong-Wolfe
    line searches may exceed it before returning. Actual calls are recorded.
    Tests may use smaller counts; the registered run pins all counts below.
    """
    if arm not in ARMS:
        raise ValueError("Unknown optimizer arm")
    model = deepcopy(source).train()
    initial = deepcopy(model.state_dict())
    initial_sha = state_fingerprint(model)
    history = []
    calls = 0
    forward_calls = 0
    cpu, wall = time.process_time(), time.perf_counter()
    optimizer = None

    def objective(*, gradient):
        nonlocal calls, forward_calls
        guard()
        if gradient:
            optimizer.zero_grad(set_to_none=True)
        loss, terms = all_start_closure_loss(model, direct)
        forward_calls += 1
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite R closure loss")
        if gradient:
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("Invalid R gradient")
            calls += 1
        row = dict(gradient_calls=calls, loss=float(loss.detach()),
                   **{key: float(value.detach()) for key, value in terms.items()},
                   cpu_seconds=time.process_time() - cpu,
                   wall_seconds=time.perf_counter() - wall)
        history.append(row)
        if calls % 25 == 0 or not gradient:
            on_progress(row)
        return loss

    try:
        with torch.no_grad():
            objective(gradient=False)
        if arm == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            for _ in range(adam_steps):
                objective(gradient=True)
                optimizer.step()
            clocks = [int(value["step"].item()) for value in optimizer.state.values()]
            if not clocks or min(clocks) != adam_steps or max(clocks) != adam_steps:
                raise RuntimeError("Adam parameter clocks disagree with its fixed endpoint")
            iterations = adam_steps
            stopping = "fixed Adam update count"
        elif arm == "lbfgs":
            optimizer = torch.optim.LBFGS(model.parameters(), lr=1.,
                max_iter=lbfgs_max_iter, max_eval=lbfgs_max_eval, history_size=20,
                tolerance_grad=1e-9, tolerance_change=1e-12,
                line_search_fn="strong_wolfe")
            optimizer.step(lambda: objective(gradient=True))
            state = optimizer.state[next(iter(model.parameters()))]
            iterations = int(state.get("n_iter", 0))
            if int(state.get("func_evals", 0)) != calls:
                raise RuntimeError("L-BFGS function evaluations disagree with gradient calls")
            stopping = ("max_iter" if iterations >= lbfgs_max_iter else
                        "max_eval (soft; line search may exceed)" if calls >= lbfgs_max_eval
                        else "optimizer tolerance or strong-Wolfe stopping")
        else:
            iterations = 0
            stopping = "no optimization"
        with torch.no_grad():
            endpoint_loss = float(objective(gradient=False))
        guard()
        if any(not torch.equal(model.state_dict()[name], initial[name]) for name in ("center", "scale")):
            raise RuntimeError("Fixed normalizer changed")
        for parameter in model.parameters():
            parameter.grad = None
        model.eval()
        record = dict(arm=arm, initial_R_sha256=initial_sha,
            final_R_sha256=state_fingerprint(model), endpoint_train_loss=endpoint_loss,
            optimizer_iterations=iterations, gradient_calls=calls,
            objective_forward_evaluations=forward_calls, stopping=stopping,
            max_eval_is_soft_under_strong_wolfe=arm == "lbfgs",
            cpu_seconds=time.process_time() - cpu, wall_seconds=time.perf_counter() - wall,
            history=history, normalizer_unchanged=True, optimizer_state_restarted=arm != "noopt")
        return model, record
    except BaseException:
        # L-BFGS may temporarily overwrite parameters during line search.
        # An interrupted step is not an accepted endpoint.
        model.load_state_dict(initial, strict=True)
        for parameter in model.parameters():
            parameter.grad = None
        raise


def direction_metrics(direct, prediction, cell_ids):
    empirical = direct[..., 1:, :] - direct[..., :2, :]
    predicted = prediction["one_step"] - direct[..., :2, :]
    chain_truth, _, cell_truth, cells = chain_and_cell_means(empirical, cell_ids)
    chain_prediction, _, cell_prediction, _ = chain_and_cell_means(predicted, cell_ids)
    rows = []
    for k in (0, 1):
        chain, _ = vector_metrics(chain_truth[:, k], chain_prediction[:, k])
        condition, angles = vector_metrics(cell_truth[:, k], cell_prediction[:, k])
        rows.append(dict(path=("0_to_1", "1_to_2")[k], physical_chains=chain,
            parameter_conditions=condition, cell_ids=cells.tolist(),
            cell_angles_degrees=[float(x) if np.isfinite(x) else None for x in angles],
            cell_opposite=[bool(np.isfinite(x) and x > 90) for x in angles],
            encoding_repeats_averaged_first=True,
            conditional_on_fixed_QC_and_existing_physical_chains=True))
    return rows


def write_plots(output, arrays, records, metrics):
    from scripts.report_phi4_qcr import _plotting
    plt = _plotting(None)
    colors = {"noopt": "#777777", "adam": "#4b93b7", "lbfgs": "#b76b75"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for ax, seed in zip(axes, SEEDS):
        for arm in ARMS:
            rows = records[f"{arm}_seed{seed}"]["history"]
            ax.plot([x["gradient_calls"] for x in rows], [x["loss"] for x in rows],
                    color=colors[arm], label=NAMES[arm], linewidth=1.1)
        ax.set(title=f"R 初始端点种子 {seed}", xlabel="累计梯度计算次数", ylabel="FIT TRAIN 闭合目标")
        ax.set_yscale("log")
    axes[0].legend(fontsize=8, frameon=False)
    fig.savefig(output / "loss.png", dpi=170)
    plt.close(fig)

    direct = arrays["train_direct"]
    points = direct.mean(0)
    extent = direct.reshape(-1, 2)
    lower, upper = extent.min(0), extent.max(0)
    padding = .08 * (upper - lower)
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 13), sharex=True, sharey=True,
                             constrained_layout=True)
    for row, seed in enumerate(SEEDS):
        for col, arm in enumerate(ARMS):
            ax = axes[row, col]
            prediction = arrays[f"eval_{arm}_seed{seed}_train_one_step"].mean(0)
            for k, color in enumerate(("#4d94bd", "#d89b48", "#a071ac")):
                ax.scatter(*extent.reshape(4, 136, 3, 2)[..., k, :].reshape(-1, 2).T,
                           s=5, alpha=.16, color=color, label=f"数据 k={k}")
            for k in (0, 1):
                delta = prediction[:, k] - points[:, k]
                ax.quiver(*points[:, k].T, *delta.T, angles="xy", scale_units="xy", scale=1,
                    color=("#246388", "#a46b25")[k], width=.004, alpha=.6)
            ax.set(title=f"{NAMES[arm]}，种子 {seed}", xlabel="g₁", ylabel="g₂",
                xlim=(lower[0]-padding[0], upper[0]+padding[0]),
                ylim=(lower[1]-padding[1], upper[1]+padding[1]))
            ax.set_aspect("equal", adjustable="box")
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("实际数据点上的 R(g)−g；先对编码重复平均，箭长为原始位移", fontsize=13)
    fig.savefig(output / "R_vs_g.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for index, (ax, path) in enumerate(zip(axes, ("36→12", "12→4", "36→4"))):
        for arm in ARMS:
            values = [metrics[f"{arm}_seed{s}"]["test"][index]["nrmse"] for s in SEEDS]
            ax.scatter(SEEDS, values, color=colors[arm], label=NAMES[arm])
        ax.set(title=f"留出条件 {path}", xlabel="R 初始端点种子", ylabel="原 TRAIN 目标标准差 NRMSE")
        ax.set_xticks(SEEDS)
    axes[0].legend(fontsize=8, frameon=False)
    fig.savefig(output / "metrics.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    labels = arrays["train_labels"]
    for k, ax in enumerate(axes):
        cloud = ax.scatter(*direct[:, :, k].mean(0).T, c=labels[:, 0], cmap="viridis", s=20)
        ax.set(title=f"实际编码 k={k}", xlabel="g₁", ylabel="g₂")
        fig.colorbar(cloud, ax=ax, label="裸参数 r（仅评价）")
    fig.savefig(output / "coupling_by_scale.png", dpi=170)
    plt.close(fig)


def run(options):
    output, cache, calibration = map(lambda value: Path(value).resolve(),
        (options.output, options.cache, options.calibration))
    if any(output == path or path in output.parents for path in (cache, calibration)):
        raise ValueError("Output cannot overwrite or nest inside an immutable input")
    if options.cpu_budget_seconds != 600. or options.output_budget_bytes != 100 * 1024 ** 2:
        raise ValueError("Registered resource limits are 600 CPU seconds and 100 MiB")
    output.mkdir(parents=True, exist_ok=False)
    started, cpu_started = time.perf_counter(), time.process_time()
    records, models, hashes = {}, {}, {}
    optimization_complete = False

    def guard():
        if time.process_time() - cpu_started > options.cpu_budget_seconds:
            raise BudgetHold("600 CPU-second diagnostic budget reached")
        if sum(p.stat().st_size for p in output.iterdir() if p.is_file()) > options.output_budget_bytes:
            raise BudgetHold("100 MiB diagnostic output budget reached")

    def status(state, **extra):
        write_json(output / "status.json", dict(recipe=RECIPE, status=state,
            workflow_complete=state == "complete", optimization_complete=optimization_complete,
            completed_models=list(records), current=current,
            returncode=0 if state == "complete" else 3 if state == "budget_hold" else 1 if state == "failed" else None,
            cpu_seconds=time.process_time()-cpu_started,
            wall_seconds=time.perf_counter()-started, **extra))

    current = "verify_inputs"
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        status("running")
        cache_manifest = verify_public(cache, hashes)
        calibration_manifest = verify_public(calibration, hashes)
        summary_path = calibration / "summary.json"
        source_summary = json.loads(summary_path.read_text())
        if (calibration_manifest["cache_manifest"]["source_state_sha256"] != cache_manifest["source_state_sha256"]
                or not source_summary.get("source_and_cache_fingerprints_unchanged")):
            raise ValueError("Frozen cache and calibration source disagree")
        fit = load_fit_only(cache)
        endpoints = restore_endpoints(calibration, source_summary, fit)
        source_paths = [Path(__file__), Path(__file__).parents[1] / "recurrent_rg/phi4_r_calibration.py",
            Path(__file__).parents[1] / "experiments/phi4_ru_r_calibration.py",
            Path(__file__).parents[1] / "experiments/phi4_qcr.py",
            Path(__file__).parents[1] / "scripts/phi4_ru_empirical_flow_diagnostic.py",
            Path(__file__).parents[1] / "scripts/report_phi4_qcr.py"]
        code_sha = {str(p.resolve()): sha256_file(p) for p in source_paths}
        manifest = dict(schema_version=5, recipe=RECIPE, options=asdict(options),
            input_sha256=hashes, source_sha256=code_sha,
            source_QCR_steps=3000, parent_R_steps=6000, R_endpoint_seeds=list(SEEDS),
            Q_loaded=False, C_loaded=False, Q_updates=0, C_updates=0,
            optimizer_initialization="fresh optimizer from identical fixed6000 scaled96 state for each seed",
            optimizer_state_from_parent_loaded=False, TEST_loaded_before_optimization=False,
            labels_loaded_before_optimization=False, labels_used_by_optimizer=False,
            closure_objective="all_start_anchor", closure_scale_weights=None,
            loss="5*e01+2.5*e12+2.5*e02 in original g; coordinates summed, full FIT TRAIN mean",
            fit_shape=list(fit.shape), hidden=96, normalization="restored unchanged TRAIN buffers",
            arms={"noopt": {}, "adam": dict(lr=.001, updates=400),
                  "lbfgs": dict(lr=1., max_iter=100, max_eval=400, history_size=20,
                    tolerance_grad=1e-9, tolerance_change=1e-12, line_search_fn="strong_wolfe")},
            lbfgs_max_eval_qualification="Soft outer-step cap; a strong-Wolfe line search can exceed max_eval before returning",
            budget_qualification="Checked before closure evaluations and between stages; interrupted L-BFGS is restored and never declared complete",
            matched_updates_claimed=False, endpoint_selection="fixed optimizer stopping rules; no TEST or checkpoint selection",
            target_stds=calibration_manifest["target_stds"], CPU_threads=1, GPU_used=False,
            new_physical_sampling=False, new_root_search=False, thermal_nu=None,
            default_recipe_modified=False)
        write_json(output / "manifest.json", manifest)
        for seed in SEEDS:
            initial_sha = state_fingerprint(endpoints[seed])
            for arm in ARMS:
                current = f"{arm}_seed{seed}"
                status("running")
                def progress(row):
                    write_json(output / "history_progress.json", dict(current=current,
                        completed=records, current_progress=row))
                model, record = optimizer_arm(endpoints[seed], fit, arm, guard=guard, on_progress=progress)
                if record["initial_R_sha256"] != initial_sha or state_fingerprint(endpoints[seed]) != initial_sha:
                    raise RuntimeError("Paired source endpoint changed")
                models[current], records[current] = model, record
                write_json(output / "history_progress.json", dict(completed=records))
                print(json.dumps(dict(model=current, **{k: record[k] for k in
                    ("gradient_calls", "optimizer_iterations", "endpoint_train_loss", "cpu_seconds")})), flush=True)
        optimization_complete = True
        current = "fixed_endpoint_evaluation"
        status("running")
        endpoint_sha = {key: state_fingerprint(model) for key, model in models.items()}
        # This is the first access to labels or any evaluation encoding.
        with np.load(cache / "cache.npz", allow_pickle=False) as archive:
            saved = {f"{split}_{name}": archive[f"{split}_{name}"].copy()
                     for split in ("train", "test") for name in ("labels", "cell_ids")}
            for split in ("train", "test"):
                saved[f"{split}_direct"] = archive[f"eval_{split}_direct"].copy()
        if set(saved["train_cell_ids"]) & set(saved["test_cell_ids"]):
            raise ValueError("TRAIN/TEST condition overlap")
        metrics, directional = {}, {}
        for key, model in models.items():
            guard()
            metrics[key], directional[key] = {}, {}
            for split in ("train", "test"):
                direct = saved[f"{split}_direct"]
                expected = (4, 136 if split == "train" else 44, 3, 2)
                if direct.shape != expected or not np.isfinite(direct).all():
                    raise ValueError("Invalid evaluation cache")
                values = predict(model, direct)
                metrics[key][split] = endpoint_metrics(direct, values, manifest["target_stds"])
                directional[key][split] = direction_metrics(direct, values, saved[f"{split}_cell_ids"])
                saved.update({f"eval_{key}_{split}_{kind}": value for kind, value in values.items()})
        np.savez_compressed(output / "evaluation_arrays.npz", **saved)
        np.savez_compressed(output / "R_parameters.npz", **{
            f"{key}.{name}": value.detach().numpy() for key, model in models.items()
            for name, value in model.state_dict().items()})
        write_json(output / "metrics.json", dict(latent=metrics, direction=directional))
        write_plots(output, saved, records, metrics)
        guard()
        if any(state_fingerprint(model) != endpoint_sha[key] for key, model in models.items()):
            raise RuntimeError("Evaluation changed an R endpoint")
        if any(sha256_file(path) != digest for path, digest in {**hashes, **code_sha}.items()):
            raise RuntimeError("Immutable input/code fingerprint changed")
        result = dict(recipe=RECIPE, status="complete", workflow_complete=True, returncode=0,
            records=records, final_R_sha256=endpoint_sha, latent_metrics=metrics,
            directional_metrics=directional, input_and_code_sha256_unchanged=True,
            Q_loaded=False, C_loaded=False, Q_updates=0, C_updates=0,
            optimization_completed_before_TEST_and_labels_loaded=True,
            CPU_threads=1, GPU_seconds=0, cpu_seconds=time.process_time()-cpu_started,
            wall_seconds=time.perf_counter()-started, thermal_nu=None, critical_r=None,
            new_root_search=False, physical_fixed_point_certified=False,
            limitations=["Diagnostic restart of R only; optimizer work units and CPU costs differ",
                "Three R endpoints share one frozen Q/C seed and the same finite physical chains",
                "Existing TEST has been inspected previously; no untouched final-test claim",
                "Source sampling warnings persist; no equilibrium or physical phase-boundary certification"])
        write_json(output / "summary.json", result)
        write_json(output / "public_files.json", {p.name: sha256_file(p) for p in output.iterdir()
            if p.is_file() and p.name not in ("status.json", "public_files.json")})
        current = "complete"
        status("complete", error=None, requires_user_action=False)
        return result
    except BaseException as error:
        status("budget_hold" if isinstance(error, BudgetHold) else "failed",
            error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(),
            requires_user_action=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "calibration", "output"):
        parser.add_argument("--" + name, required=True)
    run(Options(**vars(parser.parse_args(argv))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
