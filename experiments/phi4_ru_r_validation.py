"""Fresh R-only fit with a fixed condition-level R validation split.

Q/C was trained on all original TRAIN conditions. The eight R-validation
conditions therefore validate a new R fit only, not a fully unseen QCR model.
No old R endpoint is restored. TEST and coupling labels are decompressed only
after all three seeds finish training and validation-based endpoint selection.
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
from experiments.phi4_ru_r_calibration import endpoint_metrics, predict
from experiments.phi4_ru_r_optimizer_probe import BudgetHold, direction_metrics, verify_public
from recurrent_rg.phi4_r_calibration import (
    FrozenResidualMap, all_start_closure_loss, fit_input_normalizer,
)
from scripts.phi4_ru_empirical_flow_diagnostic import chain_and_cell_means


RECIPE = "phi4_ru_fresh_R_condition_validation_v1"
SEEDS = (91, 92, 93)
SPLIT_SEED = 20261030


@dataclass(frozen=True)
class Options:
    cache: str
    calibration: str
    output: str
    cpu_budget_seconds: float = 1200.
    output_budget_bytes: int = 100 * 1024 ** 2


@dataclass(frozen=True)
class Schedule:
    """The entry point fixes defaults; tiny schedules are for unit tests only."""
    adam_steps: int = 6000
    checkpoint_every: int = 100
    lbfgs_block_iterations: int = 100
    lbfgs_max_iterations: int = 1000
    patience: int = 3
    relative_improvement: float = .001

    def __post_init__(self):
        integers = (self.adam_steps, self.checkpoint_every, self.lbfgs_block_iterations,
                    self.lbfgs_max_iterations, self.patience)
        if any(not isinstance(x, int) or isinstance(x, bool) or x < 1 for x in integers):
            raise ValueError("Schedule counts must be positive integers")
        if self.adam_steps % self.checkpoint_every or self.lbfgs_max_iterations % self.lbfgs_block_iterations:
            raise ValueError("Fixed stage endpoints must be divisible by checkpoint/block counts")
        if not 0 < self.relative_improvement < 1:
            raise ValueError("Relative improvement must lie in (0,1)")


def split_conditions(cell_ids, *, validation_count=8, seed=SPLIT_SEED,
                     expected_conditions=34, chains_per_condition=4):
    ids = np.asarray(cell_ids)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("Condition IDs must be an integer vector")
    cells, counts = np.unique(ids, return_counts=True)
    if len(cells) != expected_conditions or np.any(counts != chains_per_condition):
        raise ValueError("Expected complete and equally sized physical-chain condition groups")
    if not 0 < validation_count < len(cells):
        raise ValueError("Validation must leave fit conditions")
    shuffled = np.random.default_rng(seed).permutation(cells)
    validation_cells = np.sort(shuffled[:validation_count])
    fit_cells = np.sort(shuffled[validation_count:])
    validation_mask = np.isin(ids, validation_cells)
    return dict(seed=seed, fit_cells=fit_cells, validation_cells=validation_cells,
                fit_mask=~validation_mask, validation_mask=validation_mask,
                shuffled_cells=shuffled)


def load_r_fit_and_validation(cache):
    """Lazy NPZ access explicitly excludes TEST, labels, decoder and old R."""
    with np.load(Path(cache) / "cache.npz", allow_pickle=False) as archive:
        ids = archive["train_cell_ids"].copy()
        fit = archive["fit_direct"].copy()
        evaluation = archive["eval_train_direct"].copy()
    for value in (fit, evaluation):
        if value.shape != (4, 136, 3, 2) or not np.isfinite(value).all():
            raise ValueError("Expected four frozen repeats of 136 TRAIN physical chains")
    split = split_conditions(ids)
    return (torch.tensor(fit[:, split["fit_mask"]], dtype=torch.float64),
            torch.tensor(evaluation[:, split["validation_mask"]], dtype=torch.float64),
            split, ids)


def initialize_model(fit, seed, *, hidden=96):
    center, scale = fit_input_normalizer(fit)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = FrozenResidualMap(2, hidden, center=center, scale=scale)
    return model


@torch.no_grad()
def validation_metrics(model, direct):
    """Square chain-mean residuals after predicting each repeat separately.

    mean(R(g)) is used, never R(mean(g)). Four stochastic encodings of a chain
    remain four measurements of one physical ensemble, not four new chains.
    """
    if direct.ndim != 4 or direct.shape[-2:] != (3, 2) or direct.requires_grad:
        raise ValueError("Expected frozen [repeat,chain,3,2] encodings")
    if not torch.isfinite(direct).all():
        raise ValueError("Nonfinite validation encodings")
    first = model.advance(direct[..., 0, :])
    outputs = (first, model.advance(direct[..., 1, :]), model.advance(first))
    errors = [p - direct[..., target, :] for p, target in zip(outputs, (1, 2, 2))]
    names, weights = ("e01", "e12", "e02"), (5., 2.5, 2.5)
    chain = {name: float(error.mean(0).square().sum(-1).mean())
             for name, error in zip(names, errors)}
    repeat = {name: error.square().sum(-1).mean(-1).tolist()
              for name, error in zip(names, errors)}
    per_repeat = [sum(weight * repeat[name][i] for name, weight in zip(names, weights))
                  for i in range(direct.shape[0])]
    return dict(chain_mean_loss=sum(weight * chain[name] for name, weight in zip(names, weights)),
        chain_mean_terms=chain, repeat_losses=per_repeat,
        repeat_terms=repeat, repeat_mean_loss=float(np.mean(per_repeat)),
        physical_chains=int(direct.shape[1]), encoding_repeats=int(direct.shape[0]),
        prediction_then_repeat_average=True, selection_uses_chain_mean_loss=True)


class ImprovementTracker:
    """Keep strict best separately from the patience-reset reference.

    Tiny successive improvements may accumulate until their total improvement
    from the last significant best reaches the predeclared relative threshold.
    """
    def __init__(self, initial, *, patience=3, relative_improvement=.001):
        if not np.isfinite(initial) or initial < 0:
            raise ValueError("Expected a finite nonnegative validation objective")
        self.best = self.reference = float(initial)
        self.patience, self.relative = patience, relative_improvement
        self.bad_blocks = 0

    def update(self, value):
        if not np.isfinite(value) or value < 0:
            raise FloatingPointError("Nonfinite or negative validation objective")
        strict = value < self.best
        if strict:
            self.best = float(value)
        significant = (value < self.reference and
                       self.reference - value >= self.relative * self.reference)
        if significant:
            self.reference, self.bad_blocks = float(value), 0
        else:
            self.bad_blocks += 1
        return dict(strict_improvement=strict, significant_improvement=significant,
                    bad_blocks=self.bad_blocks, should_stop=self.bad_blocks >= self.patience)


def train_seed(fit, validation, seed, *, schedule=Schedule(), hidden=96,
               guard=lambda: None, on_checkpoint=lambda *args: None):
    """Train from fresh R, select solely on R-validation, retain fair baselines."""
    model = initialize_model(fit, seed, hidden=hidden).train()
    initial_state = deepcopy(model.state_dict())
    initial_sha = state_fingerprint(model)
    histories, calls = {"adam": [], "lbfgs": []}, {"adam": 0, "lbfgs": 0}
    cpu_started = time.process_time()

    def checked_backward(optimizer, stage):
        guard()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = all_start_closure_loss(model, fit)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite R-fit loss")
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("Invalid R-fit gradient")
        calls[stage] += 1
        return loss

    def checkpoint(stage, step, **extra):
        guard()
        for parameter in model.parameters():
            parameter.grad = None
        before = state_fingerprint(model)
        with torch.no_grad():
            train_loss, terms = all_start_closure_loss(model, fit)
            val = validation_metrics(model, validation)
        if state_fingerprint(model) != before or any(p.grad is not None for p in model.parameters()):
            raise RuntimeError("Validation altered model state or gradients")
        if not np.isfinite(float(train_loss)) or not np.isfinite(val["chain_mean_loss"]):
            raise FloatingPointError("Nonfinite accepted endpoint objective")
        row = dict(stage=stage, step=step, fit_loss=float(train_loss),
            fit_terms={name: float(value) for name, value in terms.items()}, validation=val,
            R_sha256=before, gradient_calls=calls.copy(),
            cpu_seconds=time.process_time()-cpu_started, **extra)
        histories[stage].append(row)
        on_checkpoint(stage, step, model, row)
        return row

    try:
        adam = torch.optim.Adam(model.parameters(), lr=.001)
        row = checkpoint("adam", 0)
        adam_best_score, adam_best_step = row["validation"]["chain_mean_loss"], 0
        adam_best_state = deepcopy(model.state_dict())
        for step in range(1, schedule.adam_steps + 1):
            checked_backward(adam, "adam")
            adam.step()
            if step % schedule.checkpoint_every == 0:
                row = checkpoint("adam", step)
                if row["validation"]["chain_mean_loss"] < adam_best_score:
                    adam_best_score, adam_best_step = row["validation"]["chain_mean_loss"], step
                    adam_best_state = deepcopy(model.state_dict())
        clocks = [int(state["step"].item()) for state in adam.state.values()]
        if not clocks or min(clocks) != schedule.adam_steps or max(clocks) != schedule.adam_steps:
            raise RuntimeError("Adam parameter update clocks disagree")
        adam_final_state = deepcopy(model.state_dict())
        model.load_state_dict(adam_best_state, strict=True)
        lbfgs_start_sha = state_fingerprint(model)
        lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.,
            max_iter=schedule.lbfgs_block_iterations, max_eval=4*schedule.lbfgs_block_iterations,
            history_size=20, tolerance_grad=1e-9, tolerance_change=1e-12,
            line_search_fn="strong_wolfe")
        tracker = ImprovementTracker(adam_best_score, patience=schedule.patience,
                                     relative_improvement=schedule.relative_improvement)
        row = checkpoint("lbfgs", 0, optimizer_iterations=0, block=0)
        lbfgs_best_state, lbfgs_best_step = deepcopy(model.state_dict()), 0
        lbfgs_best_score, iterations, stop = adam_best_score, 0, "max_iterations"
        for block in range(1, schedule.lbfgs_max_iterations // schedule.lbfgs_block_iterations + 1):
            accepted = deepcopy(model.state_dict())
            calls_before, iterations_before = calls["lbfgs"], iterations
            try:
                # One optimizer instance is retained across blocks, including
                # curvature pairs and its cumulative iteration/evaluation state.
                lbfgs.step(lambda: checked_backward(lbfgs, "lbfgs"))
                guard()
            except BaseException:
                model.load_state_dict(accepted, strict=True)
                raise
            opt_state = lbfgs.state[next(iter(model.parameters()))]
            iterations = int(opt_state.get("n_iter", 0))
            if int(opt_state.get("func_evals", 0)) != calls["lbfgs"]:
                raise RuntimeError("L-BFGS cumulative evaluation counters disagree")
            row = checkpoint("lbfgs", iterations, optimizer_iterations=iterations, block=block,
                             block_gradient_calls=calls["lbfgs"]-calls_before)
            score = row["validation"]["chain_mean_loss"]
            if score < lbfgs_best_score:
                lbfgs_best_score, lbfgs_best_step = score, iterations
                lbfgs_best_state = deepcopy(model.state_dict())
            progress = tracker.update(score)
            row.update(progress)
            if progress["should_stop"]:
                stop = "validation_patience"
                break
            if iterations - iterations_before < schedule.lbfgs_block_iterations:
                stop = ("soft_max_eval" if calls["lbfgs"]-calls_before >= 4*schedule.lbfgs_block_iterations
                        else "optimizer_tolerance_or_strong_wolfe_stopping")
                break
        lbfgs_final_state = deepcopy(model.state_dict())
        selected_stage = "lbfgs" if lbfgs_best_score < adam_best_score else "adam"
        selected_state = lbfgs_best_state if selected_stage == "lbfgs" else adam_best_state
        selected_step = lbfgs_best_step if selected_stage == "lbfgs" else adam_best_step
        states = {"initial": initial_state, "adam6000": adam_final_state,
                  "adam_best": adam_best_state, "lbfgs_final": lbfgs_final_state,
                  "lbfgs_best": lbfgs_best_state, "selected": selected_state}
        models = {}
        for name, state in states.items():
            restored = deepcopy(model).eval()
            restored.load_state_dict(state, strict=True)
            for parameter in restored.parameters():
                parameter.grad = None
            if any(not torch.equal(restored.state_dict()[key], initial_state[key])
                   for key in ("center", "scale")):
                raise RuntimeError("R-fit-only normalizer changed")
            models[name] = restored
        record = dict(seed=seed, fresh_R=True, initial_R_sha256=initial_sha,
            adam_updates=schedule.adam_steps, adam_parameter_clock_min=min(clocks),
            adam_parameter_clock_max=max(clocks), adam_best_step=adam_best_step,
            adam_best_validation_loss=adam_best_score, lbfgs_start_R_sha256=lbfgs_start_sha,
            lbfgs_start_matches_adam_best=lbfgs_start_sha == state_fingerprint(models["adam_best"]),
            lbfgs_optimizer_iterations=iterations, lbfgs_stopping=stop,
            lbfgs_best_iteration=lbfgs_best_step, lbfgs_best_validation_loss=lbfgs_best_score,
            lbfgs_optimizer_history_retained_across_blocks=True,
            lbfgs_optimizer_restarted_from_adam=True,
            lbfgs_max_eval_is_soft_under_strong_wolfe=True,
            selected_stage=selected_stage, selected_step=selected_step,
            selected_validation_loss=min(adam_best_score, lbfgs_best_score),
            selected_R_sha256=state_fingerprint(models["selected"]),
            gradient_calls=calls, history=histories, normalizer_unchanged=True,
            no_validation_gradient=True, cpu_seconds=time.process_time()-cpu_started)
        return models, record
    except BaseException:
        for parameter in model.parameters():
            parameter.grad = None
        raise


def write_plots(output, arrays, records):
    from scripts.report_phi4_qcr import _plotting
    plt = _plotting(None)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4), constrained_layout=True)
    for ax, seed in zip(axes, SEEDS):
        record = records[str(seed)]
        for stage, style, offset in (("adam", "-", 0), ("lbfgs", "--", 6000)):
            rows = record["history"][stage]
            x = [offset + row["step"] for row in rows]
            ax.plot(x, [row["fit_loss"] for row in rows], style, color="#527d9c",
                    label="拟合：逐重复目标" if stage == "adam" else "L-BFGS 拟合")
            ax.plot(x, [row["validation"]["chain_mean_loss"] for row in rows], style,
                    color="#b76a70", label="验证：先平均重复残差" if stage == "adam" else "L-BFGS 验证")
        ax.axvline(6000, color=".6", linewidth=.8)
        ax.set(title=f"新 R 种子 {seed}", xlabel="Adam 更新数＋L-BFGS 累计迭代数",
               ylabel="原 g 坐标的闭合目标", yscale="log")
    axes[0].legend(fontsize=8, frameon=False)
    fig.savefig(output / "loss.png", dpi=180)
    plt.close(fig)

    g = np.concatenate([arrays[f"{s}_direct"] for s in ("fit", "validation", "test")], axis=1)
    ids = np.concatenate([arrays[f"{s}_cell_ids"] for s in ("fit", "validation", "test")])
    _, _, means, _ = chain_and_cell_means(g, ids)
    fig, axes = plt.subplots(3, 4, figsize=(15, 11), sharex=True, sharey=True,
                             constrained_layout=True)
    extent = g.reshape(-1, 2)
    lo, hi = extent.min(0), extent.max(0)
    pad = .07 * np.maximum(hi-lo, 1e-8)
    for row, seed in enumerate(SEEDS):
        repeated = np.concatenate([arrays[f"eval_selected_seed{seed}_{s}_one_step"]
                                   for s in ("fit", "validation", "test")], axis=1)
        _, _, predicted, _ = chain_and_cell_means(repeated, ids)
        for k in (0, 1):
            for j, (name, destination) in enumerate((("配对数据位移", means[:, 1:]),
                                                    ("验证选定 R 的位移", predicted))):
                ax = axes[row, 2*k+j]
                for scale, color in ((k, "#5594bd"), (k+1, "#d9aa65")):
                    ax.scatter(*g[..., scale, :].reshape(-1, 2).T, s=4, alpha=.14, color=color)
                delta = destination[:, k]-means[:, k]
                ax.quiver(*means[:, k].T, *delta.T, angles="xy", scale_units="xy", scale=1,
                          width=.004, color="#294c60")
                ax.set(title=f"种子 {seed}，k={k}：{name}", xlabel="g₁", ylabel="g₂",
                    xlim=(lo[0]-pad[0], hi[0]+pad[0]), ylim=(lo[1]-pad[1], hi[1]+pad[1]))
                ax.set_aspect("equal", adjustable="box")
    fig.suptitle("逐条件实际点流场；预测为 mean R(g)，箭长保持原始 R(g)−g", fontsize=13)
    fig.savefig(output / "R_vs_g.png", dpi=180)
    plt.close(fig)

    labels = np.concatenate([arrays[f"{s}_labels"] for s in ("fit", "validation", "test")])
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for row, (index, name) in enumerate(((0, "裸参数 r"), (1, "裸参数 u"))):
        for k in range(3):
            ax = axes[row, k]
            scatter = ax.scatter(*g[..., k, :].mean(0).T, c=labels[:, index], cmap="viridis", s=18)
            ax.set(title=f"k={k}，{name}", xlabel="g₁", ylabel="g₂")
            fig.colorbar(scatter, ax=ax, label=f"{name}（仅事后评价）")
    fig.savefig(output / "coupling_by_scale.png", dpi=180)
    plt.close(fig)


def run(options):
    output, cache, calibration = map(lambda p: Path(p).resolve(),
                                   (options.output, options.cache, options.calibration))
    if any(output == p or p in output.parents for p in (cache, calibration)):
        raise ValueError("Output must be separate from immutable inputs")
    if options.cpu_budget_seconds != 1200. or options.output_budget_bytes != 100*1024**2:
        raise ValueError("Registered limits are 1200 CPU seconds and 100 MiB")
    output.mkdir(parents=True, exist_ok=False)
    started, cpu_started = time.perf_counter(), time.process_time()
    current, records, models, hashes = "verify_inputs", {}, {}, {}
    optimization_complete, last_size_check = False, -1.

    def guard(force=False):
        nonlocal last_size_check
        used = time.process_time()-cpu_started
        if used > options.cpu_budget_seconds:
            raise BudgetHold("1200 CPU-second R validation budget reached")
        if force or used-last_size_check >= 1.:
            last_size_check = used
            if sum(p.stat().st_size for p in output.iterdir() if p.is_file()) > options.output_budget_bytes:
                raise BudgetHold("100 MiB R validation output budget reached")

    def status(state, **extra):
        write_json(output / "status.json", dict(status=state, recipe=RECIPE,
            workflow_complete=state == "complete", current=current,
            returncode=0 if state == "complete" else 3 if state == "budget_hold" else 1 if state == "failed" else None,
            optimization_complete=optimization_complete, completed_seeds=list(records),
            cpu_seconds=time.process_time()-cpu_started, wall_seconds=time.perf_counter()-started, **extra))

    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        status("running")
        cache_manifest = verify_public(cache, hashes)
        reference_manifest = verify_public(calibration, hashes)
        if reference_manifest["cache_manifest"]["source_state_sha256"] != cache_manifest["source_state_sha256"]:
            raise ValueError("Frozen cache/source reference mismatch")
        fit, validation, split, train_ids = load_r_fit_and_validation(cache)
        root = Path(__file__).parents[1]
        code_files = [Path(__file__), root / "recurrent_rg/phi4_r_calibration.py",
            root / "experiments/phi4_ru_r_calibration.py", root / "experiments/phi4_ru_r_optimizer_probe.py",
            root / "experiments/phi4_qcr.py", root / "scripts/phi4_ru_empirical_flow_diagnostic.py",
            root / "scripts/report_phi4_qcr.py"]
        code_sha = {str(p.resolve()): sha256_file(p) for p in code_files}
        center, scale = fit_input_normalizer(fit)
        manifest = dict(schema_version=5, recipe=RECIPE, options=asdict(options),
            schedule=asdict(Schedule()), seeds=list(SEEDS), input_sha256=hashes, source_sha256=code_sha,
            source_QCR_steps=3000, source_QC_seed=91, hidden=96, dtype="float64",
            source_state_sha256=cache_manifest["source_state_sha256"],
            source_module_sha256=cache_manifest["source_module_sha256"],
            initialization="new random R for each seed; no previous R endpoint loaded",
            R_validation_is_independent_of_R_fit_conditions=True,
            R_validation_is_unseen_by_source_QC=False,
            split={k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in split.items()},
            fit_shape=list(fit.shape), validation_shape=list(validation.shape),
            normalizer=dict(center=center.tolist(), scale=scale.tolist(),
                inputs="only R-fit FIT repeats, scales k=0,1; no validation/TEST/target-scale moments"),
            fit_objective="5*e01+2.5*e12+2.5*e02; original g; sum coordinates, mean FIT repeats and chains",
            closure_objective="all_start_anchor", closure_scale_weights=None,
            validation_objective="same weights; predict each EVAL repeat, average residuals within physical chain, then square and mean chains",
            endpoint_selection="strict minimum R-validation loss within each seed, across Adam and L-BFGS; no cross-seed selection",
            patience_rule="three L-BFGS blocks without >=0.001 relative improvement from last significant best; strict best saved separately",
            lbfgs_start="lowest-validation Adam checkpoint; optimizer fresh at start, history retained across blocks",
            target_stds=reference_manifest["target_stds"],
            original_TEST_qualification="already inspected in earlier experiments; exploratory only",
            labels_loaded_before_all_selection=False, TEST_loaded_before_all_selection=False,
            labels_used_by_optimizer=False, Q_loaded=False, C_loaded=False, Q_updates=0, C_updates=0,
            CPU_threads=1, GPU_used=False, new_physical_sampling=False, new_root_search=False,
            default_recipe_modified=False)
        write_json(output / "manifest.json", manifest)
        for seed in SEEDS:
            current = f"fresh_R_seed{seed}"
            status("running")
            def save_checkpoint(stage, step, model, row):
                path = output / f"checkpoint_seed{seed}_{stage}_{step:06d}.npz"
                np.savez_compressed(path, **{name: value.detach().numpy()
                                            for name, value in model.state_dict().items()})
                write_json(output / "history_progress.json", dict(current=current,
                    completed=records, current_progress=row))
                guard(True)
            trained, record = train_seed(fit, validation, seed, guard=guard,
                                          on_checkpoint=save_checkpoint)
            records[str(seed)] = record
            models.update({f"{name}_seed{seed}": model for name, model in trained.items()})
            write_json(output / "history_progress.json", dict(completed=records))
            print(json.dumps(dict(seed=seed, selected_stage=record["selected_stage"],
                selected_step=record["selected_step"], validation_loss=record["selected_validation_loss"],
                lbfgs_stop=record["lbfgs_stopping"], cpu_seconds=record["cpu_seconds"])), flush=True)
        optimization_complete = True
        current = "selected_endpoint_evaluation"
        status("running")
        # First TEST/label decompression occurs after ALL seeds are selected.
        with np.load(cache / "cache.npz", allow_pickle=False) as archive:
            train_eval = archive["eval_train_direct"].copy()
            labels = archive["train_labels"].copy()
            saved = dict(test_direct=archive["eval_test_direct"].copy(),
                         test_cell_ids=archive["test_cell_ids"].copy(),
                         test_labels=archive["test_labels"].copy())
        if set(train_ids) & set(saved["test_cell_ids"]):
            raise ValueError("Original TRAIN/TEST conditions overlap")
        if saved["test_direct"].shape != (4, 44, 3, 2) or not np.isfinite(saved["test_direct"]).all():
            raise ValueError("Invalid exploratory TEST cache")
        for name in ("fit", "validation"):
            mask = split[f"{name}_mask"]
            saved.update({f"{name}_direct": train_eval[:, mask],
                          f"{name}_cell_ids": train_ids[mask], f"{name}_labels": labels[mask]})
        metrics, directions, scores = {}, {}, {}
        fingerprints = {key: state_fingerprint(model) for key, model in models.items()}
        for key, model in models.items():
            guard()
            metrics[key], directions[key], scores[key] = {}, {}, {}
            for name in ("fit", "validation", "test"):
                direct = saved[f"{name}_direct"]
                prediction = predict(model, direct)
                metrics[key][name] = endpoint_metrics(direct, prediction, manifest["target_stds"])
                directions[key][name] = direction_metrics(direct, prediction, saved[f"{name}_cell_ids"])
                scores[key][name] = validation_metrics(model, torch.tensor(direct, dtype=torch.float64))
                saved.update({f"eval_{key}_{name}_{kind}": values for kind, values in prediction.items()})
        np.savez_compressed(output / "evaluation_arrays.npz", **saved)
        np.savez_compressed(output / "R_parameters.npz", **{
            f"{key}.{name}": value.detach().numpy() for key, model in models.items()
            for name, value in model.state_dict().items()})
        write_json(output / "history.json", records)
        write_plots(output, saved, records)
        if any(state_fingerprint(model) != fingerprints[key] for key, model in models.items()):
            raise RuntimeError("Evaluation changed R endpoint")
        if any(sha256_file(path) != digest for path, digest in {**hashes, **code_sha}.items()):
            raise RuntimeError("Immutable input or pinned source changed")
        guard(True)
        result = dict(status="complete", workflow_complete=True, returncode=0, recipe=RECIPE,
            records=records, latent_metrics=metrics, directional_metrics=directions, closure_scores=scores,
            final_R_sha256=fingerprints, input_and_code_sha256_unchanged=True,
            all_seeds_selected_before_TEST_and_labels_loaded=True,
            fair_Adam_baseline="adam6000_seed91/92/93; same new 26-condition fit and fresh initialization",
            selected_parameter_keys=[f"selected_seed{s}" for s in SEEDS],
            Q_loaded=False, C_loaded=False, Q_updates=0, C_updates=0, GPU_seconds=0,
            CPU_threads=1, cpu_seconds=time.process_time()-cpu_started,
            wall_seconds=time.perf_counter()-started, thermal_nu=None, critical_r=None,
            physical_fixed_point_certified=False, root_search=False,
            limitations=["R-validation conditions were seen by the frozen source Q/C; independence is R-fit-only",
                "All R seeds share one source Q/C and existing finite physical chains",
                "Original TEST was inspected previously and is exploratory, not a fresh final test",
                "Four repeats per physical chain remain measurement repeats, not independent ensembles",
                "Sampling warnings persist; no equilibrium or physical boundary certification"])
        write_json(output / "summary.json", result)
        write_json(output / "public_files.json", {p.name: sha256_file(p) for p in output.iterdir()
            if p.is_file() and p.name not in ("status.json", "public_files.json")})
        guard(True)
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
        parser.add_argument("--"+name, required=True)
    run(Options(**vars(parser.parse_args(argv))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
