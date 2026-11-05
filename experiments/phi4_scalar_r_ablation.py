"""Frozen-Q/C scalar closure ablation: four condition folds, four R architectures.

Only anonymous TRAIN group IDs and frozen TRAIN encodings are accessed before
all 48 endpoint selections. Coupling labels and the previously inspected TEST
remain post-selection diagnostics. No source model or physical field is loaded.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from torch import nn

from experiments.phi4_qcr import sha256_file, state_fingerprint, write_json
from experiments.phi4_ru_r_validation import ImprovementTracker, Schedule
from experiments.phi4_scalar_r_validation import (
    _direct, endpoint_metrics, load_evaluation_after_selection, verify_cache,
)
from recurrent_rg.phi4_r_calibration import FrozenResidualMap, fit_input_normalizer


RECIPE = "phi4_u1_scalar_R_architecture_fourfold_ablation_v1"
SEEDS = (91, 92, 93)
SPLIT_SEED = 20261030
OUTPUT_BUDGET_BYTES = 512 * 1024**2
ARMS = {
    "A_shared96": dict(kind="shared", hidden=96, heads=1),
    "B_separate96": dict(kind="separate", hidden=96, heads=2),
    "C_shared136": dict(kind="shared", hidden=136, heads=1),
    "D_shared16": dict(kind="shared", hidden=16, heads=1),
}


@dataclass(frozen=True)
class Options:
    cache: str
    output: str
    cpu_budget_seconds: float = 3600.


class BudgetHold(RuntimeError):
    pass


class ScalarClosureMaps(nn.Module):
    """One shared or two scale-specific maps, with identical fixed input units."""

    def __init__(self, hidden=96, kind="shared", *, center=None, scale=None):
        super().__init__()
        if kind not in ("shared", "separate"):
            raise ValueError("kind must be shared or separate")
        self.theory_dim, self.hidden, self.kind = 1, hidden, kind
        self.heads = nn.ModuleList(FrozenResidualMap(1, hidden, center=center, scale=scale)
                                  for _ in range(1 if kind == "shared" else 2))

    def step(self, g, index):
        if not isinstance(index, int) or isinstance(index, bool) or index not in (0, 1):
            raise ValueError("transition index must be 0 or 1")
        return self.heads[0 if self.kind == "shared" else index].advance(g)

    def forward(self, g, index=0):
        return self.step(g, index)


def closure_loss(model, direct):
    if (not isinstance(direct, torch.Tensor) or direct.ndim != 4
            or direct.shape[0] != 4 or direct.shape[-2:] != (3, 1)
            or not direct.shape[1] or direct.requires_grad or not torch.isfinite(direct).all()):
        raise ValueError("Expected four finite frozen scalar encoding repeats")
    p01 = model.step(direct[..., 0, :], 0)
    p12 = model.step(direct[..., 1, :], 1)
    p02 = model.step(p01, 1)
    terms = dict(e01=(p01-direct[..., 1, :]).square().sum(-1).mean(),
                 e12=(p12-direct[..., 2, :]).square().sum(-1).mean(),
                 e02=(p02-direct[..., 2, :]).square().sum(-1).mean())
    return 5*terms["e01"] + 2.5*terms["e12"] + 2.5*terms["e02"], terms


def condition_folds(cell_ids):
    ids = np.asarray(cell_ids)
    if ids.shape != (32,) or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("Expected 32 anonymous TRAIN condition IDs")
    cells, counts = np.unique(ids, return_counts=True)
    if len(cells) != 8 or np.any(counts != 4):
        raise ValueError("Require eight TRAIN conditions, four physical chains each")
    shuffled = np.random.default_rng(SPLIT_SEED).permutation(cells)
    folds = []
    for fold in range(4):
        validation_cells = np.sort(shuffled[2*fold:2*fold+2])
        mask = np.isin(ids, validation_cells)
        folds.append(dict(fold=fold, seed=SPLIT_SEED, fit_cells=np.sort(cells[~np.isin(cells, validation_cells)]),
                          validation_cells=validation_cells, fit_mask=~mask,
                          validation_mask=mask, shuffled_cells=shuffled.copy()))
    return folds


def load_train_only(cache):
    """The exhaustive pre-selection NPZ access list is intentionally short."""
    with np.load(Path(cache) / "cache.npz", allow_pickle=False) as archive:
        ids = archive["train_cell_ids"].copy()
        fit = _direct(archive["fit_direct"].copy(), 32, "fit_direct")
        evaluation = _direct(archive["eval_train_direct"].copy(), 32, "eval_train_direct")
    return torch.tensor(fit, dtype=torch.float64), torch.tensor(evaluation, dtype=torch.float64), condition_folds(ids), ids


def initialize_model(fit, seed, *, hidden=96, kind="shared"):
    center, scale = fit_input_normalizer(fit)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return ScalarClosureMaps(hidden, kind, center=center, scale=scale)


@torch.no_grad()
def predict(model, direct):
    g = torch.as_tensor(direct, dtype=torch.float64)
    first = model.step(g[..., 0, :], 0)
    return dict(one_step=torch.stack((first, model.step(g[..., 1, :], 1)), -2).numpy(),
                rollout=torch.stack((g[..., 0, :], first, model.step(first, 1)), -2).numpy())


@torch.no_grad()
def validation_metrics(model, direct):
    closure_loss(model, direct)  # shape, finiteness and frozen-input validation
    prediction = predict(model, direct)
    errors = (torch.from_numpy(prediction["one_step"][..., 0, :])-direct[..., 1, :],
              torch.from_numpy(prediction["one_step"][..., 1, :])-direct[..., 2, :],
              torch.from_numpy(prediction["rollout"][..., 2, :])-direct[..., 2, :])
    names, weights = ("e01", "e12", "e02"), (5., 2.5, 2.5)
    chain = {name: float(error.mean(0).square().sum(-1).mean()) for name, error in zip(names, errors)}
    repeat = {name: error.square().sum(-1).mean(-1).tolist() for name, error in zip(names, errors)}
    losses = [sum(w*repeat[n][i] for n, w in zip(names, weights)) for i in range(4)]
    return dict(chain_mean_loss=sum(w*chain[n] for n, w in zip(names, weights)),
                chain_mean_terms=chain, repeat_losses=losses, repeat_terms=repeat,
                repeat_mean_loss=float(np.mean(losses)), physical_chains=int(direct.shape[1]),
                encoding_repeats=4, prediction_then_repeat_average=True,
                selection_uses_chain_mean_loss=True)


def train_seed(fit, validation, seed, *, schedule=None, hidden=96, kind="shared",
               guard=lambda: None, on_checkpoint=lambda *args: None):
    """Fixed 6000-step Adam followed by within-run validation-selected L-BFGS."""
    schedule = Schedule() if schedule is None else schedule
    model = initialize_model(fit, seed, hidden=hidden, kind=kind).train()
    initial_state = deepcopy(model.state_dict())
    initial_sha = state_fingerprint(model)
    histories, calls = {"adam": [], "lbfgs": []}, {"adam": 0, "lbfgs": 0}
    cpu_started = time.process_time()

    def checked_backward(optimizer, stage):
        guard()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = closure_loss(model, fit)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite scalar ablation fit loss")
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("Invalid scalar ablation gradient")
        calls[stage] += 1
        return loss

    def checkpoint(stage, step, **extra):
        guard()
        for parameter in model.parameters():
            parameter.grad = None
        before = state_fingerprint(model)
        with torch.no_grad():
            loss, terms = closure_loss(model, fit)
            val = validation_metrics(model, validation)
        if state_fingerprint(model) != before or any(p.grad is not None for p in model.parameters()):
            raise RuntimeError("Validation altered R or gradients")
        if not np.isfinite(float(loss)) or not np.isfinite(val["chain_mean_loss"]):
            raise FloatingPointError("Nonfinite scalar ablation endpoint")
        row = dict(stage=stage, step=step, fit_loss=float(loss),
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
        if len(clocks) != len(list(model.parameters())) or min(clocks) != schedule.adam_steps or max(clocks) != schedule.adam_steps:
            raise RuntimeError("Adam parameter clocks disagree")
        adam_final_state = deepcopy(model.state_dict())
        model.load_state_dict(adam_best_state, strict=True)
        lbfgs_start_sha = state_fingerprint(model)
        lbfgs = torch.optim.LBFGS(model.parameters(), lr=1., max_iter=schedule.lbfgs_block_iterations,
            max_eval=4*schedule.lbfgs_block_iterations, history_size=20, tolerance_grad=1e-9,
            tolerance_change=1e-12, line_search_fn="strong_wolfe")
        tracker = ImprovementTracker(adam_best_score, patience=schedule.patience,
                                     relative_improvement=schedule.relative_improvement)
        checkpoint("lbfgs", 0, optimizer_iterations=0, block=0)
        lbfgs_best_state, lbfgs_best_step = deepcopy(model.state_dict()), 0
        lbfgs_best_score, iterations, stop = adam_best_score, 0, "max_iterations"
        for block in range(1, schedule.lbfgs_max_iterations // schedule.lbfgs_block_iterations + 1):
            accepted = deepcopy(model.state_dict())
            calls_before, iterations_before = calls["lbfgs"], iterations
            try:
                lbfgs.step(lambda: checked_backward(lbfgs, "lbfgs"))
                guard()
            except BaseException:
                model.load_state_dict(accepted, strict=True)
                raise
            opt_state = lbfgs.state[next(iter(model.parameters()))]
            iterations = int(opt_state.get("n_iter", 0))
            if int(opt_state.get("func_evals", 0)) != calls["lbfgs"]:
                raise RuntimeError("L-BFGS evaluation counters disagree")
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
            if iterations-iterations_before < schedule.lbfgs_block_iterations:
                stop = ("soft_max_eval" if calls["lbfgs"]-calls_before >= 4*schedule.lbfgs_block_iterations
                        else "optimizer_tolerance_or_strong_wolfe_stopping")
                break
        selected_stage = "lbfgs" if lbfgs_best_score < adam_best_score else "adam"
        selected_state = lbfgs_best_state if selected_stage == "lbfgs" else adam_best_state
        selected_step = lbfgs_best_step if selected_stage == "lbfgs" else adam_best_step
        models = {}
        for name, state in (("adam6000", adam_final_state), ("selected", selected_state)):
            restored = deepcopy(model).eval()
            restored.load_state_dict(state, strict=True)
            for parameter in restored.parameters():
                parameter.grad = None
            for key in initial_state:
                if key.endswith((".center", ".scale")) and not torch.equal(restored.state_dict()[key], initial_state[key]):
                    raise RuntimeError("Fixed common input normalizer changed")
            models[name] = restored
        return models, dict(seed=seed, kind=kind, hidden=hidden,
            parameter_count=sum(p.numel() for p in model.parameters()), fresh_R=True,
            initial_R_sha256=initial_sha, adam_updates=schedule.adam_steps,
            adam_parameter_clock_min=min(clocks), adam_parameter_clock_max=max(clocks),
            adam_parameter_count=len(clocks), adam_best_step=adam_best_step,
            adam_best_validation_loss=adam_best_score, lbfgs_start_R_sha256=lbfgs_start_sha,
            lbfgs_start_matches_adam_best=True, lbfgs_optimizer_iterations=iterations,
            lbfgs_stopping=stop, lbfgs_best_iteration=lbfgs_best_step,
            lbfgs_best_validation_loss=lbfgs_best_score,
            lbfgs_optimizer_history_retained_across_blocks=True, lbfgs_optimizer_restarted_from_adam=True,
            lbfgs_max_eval_is_soft_under_strong_wolfe=True, selected_stage=selected_stage,
            selected_step=selected_step, selected_validation_loss=min(adam_best_score, lbfgs_best_score),
            selected_R_sha256=state_fingerprint(models["selected"]), gradient_calls=calls,
            history=histories, normalizer_unchanged=True, no_validation_gradient=True,
            cpu_seconds=time.process_time()-cpu_started)
    finally:
        for parameter in model.parameters():
            parameter.grad = None


def _source_hashes():
    import experiments.phi4_qcr as qcr_source
    import experiments.phi4_ru_r_validation as schedule_source
    import experiments.phi4_scalar_r_validation as helper_source
    import recurrent_rg.phi4_r_calibration as model_source
    return {str(Path(module.__file__).resolve()): sha256_file(module.__file__)
            for module in (qcr_source, schedule_source, helper_source, model_source)} | {
                str(Path(__file__).resolve()): sha256_file(__file__)}


def _json_split(split):
    return {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in split.items()}


def run(options):
    if not math.isfinite(options.cpu_budget_seconds) or not 0 < options.cpu_budget_seconds <= 7200:
        raise ValueError("CPU budget must be positive and no greater than 7200 seconds")
    output, cache = (Path(value).resolve() for value in (options.output, options.cache))
    if output == cache or output in cache.parents or cache in output.parents:
        raise ValueError("Output must be separate from immutable cache")
    output.mkdir(parents=True, exist_ok=False)
    started, cpu_started = time.perf_counter(), time.process_time()
    records, models, model_folds = {}, {}, {}
    current, optimization_complete, last_size_check = "verify_inputs", False, -1.

    def guard(force=False):
        nonlocal last_size_check
        used = time.process_time()-cpu_started
        if used > options.cpu_budget_seconds:
            raise BudgetHold("Scalar R ablation CPU budget reached")
        if force or used-last_size_check >= 1.:
            last_size_check = used
            if sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) > OUTPUT_BUDGET_BYTES:
                raise BudgetHold("Scalar R ablation 512 MiB output budget reached")

    def status(state, **extra):
        write_json(output / "status.json", dict(status=state, recipe=RECIPE,
            workflow_complete=state == "complete", current=current,
            returncode=0 if state == "complete" else 3 if state == "budget_hold" else 1 if state == "failed" else None,
            optimization_complete=optimization_complete, completed_runs=list(records),
            completed_count=len(records), planned_runs=48, cpu_seconds=time.process_time()-cpu_started,
            wall_seconds=time.perf_counter()-started, **extra))

    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        status("running")
        cache_manifest, hashes = verify_cache(cache)
        fit_all, evaluation_all, splits, train_ids = load_train_only(cache)
        code_sha = _source_hashes()
        normalizers = []
        for split in splits:
            center, scale = fit_input_normalizer(fit_all[:, split["fit_mask"]])
            normalizers.append(dict(fold=split["fold"], center=center.tolist(), scale=scale.tolist()))
        manifest = dict(schema_version=5, recipe=RECIPE, options=asdict(options),
            schedule=asdict(Schedule()), seeds=list(SEEDS), arms=ARMS, planned_runs=48,
            theory_dim=1, dtype="float64", input_sha256=hashes, source_sha256=code_sha,
            cache_manifest=cache_manifest, source_state_sha256=cache_manifest["source_state_sha256"],
            source_module_sha256=cache_manifest["source_module_sha256"],
            source_QCR_steps=3000, source_QC_seed=91, splits=[_json_split(s) for s in splits],
            normalizers=normalizers, normalizer_policy="one common pooled input-scale-0/1 fit-only normalizer per fold for all arms and heads",
            fit_objective="5*e01+2.5*e12+2.5*e02 in original g; B uses R1(R0(g0))",
            closure_objective="all_start_anchor", closure_scale_weights=None,
            validation_objective="predict each repeat, average paired residual within physical chain, then square",
            endpoint_selection="strict minimum R-validation loss within each fold/arm/seed; no TEST selection",
            target_stds=cache_manifest["target_stds"],
            target_std_scope="immutable original joint-QCR TRAIN evaluation, all 128 fields, one saved encoding; scales 0/1/2",
            R_validation_is_unseen_by_source_QC=False, labels_loaded_before_all_selection=False,
            TEST_loaded_before_all_selection=False, labels_used_by_optimizer=False,
            Q_loaded=False, C_loaded=False, Q_updates=0, C_updates=0, new_physical_sampling=False,
            new_root_search=False, CPU_threads=1, GPU_used=False, default_recipe_modified=False,
            explicit_architecture_ablation=True, source_sampling_warnings_retained=True)
        write_json(output / "manifest.json", manifest)
        for split in splits:
            fold = split["fold"]
            fit = fit_all[:, split["fit_mask"]]
            validation = evaluation_all[:, split["validation_mask"]]
            for arm, spec in ARMS.items():
                for seed in SEEDS:
                    key = f"fold{fold}_{arm}_seed{seed}"
                    current = key
                    status("running", error=None, requires_user_action=False)

                    def progress(stage, step, model, row):
                        write_json(output / "history_progress.json", dict(current=key, completed_runs=list(records),
                            completed_count=len(records), planned_runs=48, current_progress=row))
                        guard(True)

                    trained, record = train_seed(fit, validation, seed, hidden=spec["hidden"],
                        kind=spec["kind"], guard=guard, on_checkpoint=progress)
                    record.update(fold=fold, arm=arm, fit_cells=split["fit_cells"].tolist(),
                                  validation_cells=split["validation_cells"].tolist())
                    records[key] = record
                    for endpoint, model in trained.items():
                        model_key = f"{key}_{endpoint}"
                        models[model_key], model_folds[model_key] = model, fold
                    write_json(output / "history.json", records)
                    write_json(output / "history_progress.json", dict(current=key, completed_runs=list(records),
                        completed_count=len(records), planned_runs=48, selected_stage=record["selected_stage"],
                        selected_step=record["selected_step"]))
                    print(json.dumps(dict(run=key, selected_stage=record["selected_stage"],
                        selected_step=record["selected_step"], validation_loss=record["selected_validation_loss"],
                        cpu_seconds=record["cpu_seconds"])), flush=True)
        optimization_complete = True
        current = "all_selections_complete_evaluation"
        status("running", error=None, requires_user_action=False)
        # This is the first access to labels and TEST in the complete workflow.
        saved, source_predictions = load_evaluation_after_selection(cache, splits[0], train_ids)
        for name in list(saved):
            if name.startswith(("fit_", "validation_")):
                del saved[name]
        metrics, scores = {}, {}
        fingerprints = {key: state_fingerprint(model) for key, model in models.items()}
        for key, model in models.items():
            guard()
            split = splits[model_folds[key]]
            metrics[key], scores[key] = {}, {}
            for name in ("train", "fit", "validation", "test"):
                direct = saved[f"{name}_direct"] if name in ("train", "test") else saved["train_direct"][:, split[f"{name}_mask"]]
                prediction = predict(model, direct)
                metrics[key][name] = endpoint_metrics(direct, prediction, manifest["target_stds"])
                scores[key][name] = validation_metrics(model, torch.tensor(direct, dtype=torch.float64))
                if name in ("train", "test"):
                    saved.update({f"eval_{key}_{name}_{kind}": value for kind, value in prediction.items()})
        if source_predictions:
            metrics["source_joint3000"] = {}
            for name, prediction in source_predictions.items():
                metrics["source_joint3000"][name] = endpoint_metrics(saved[f"{name}_direct"], prediction, manifest["target_stds"])
                saved.update({f"eval_source_joint3000_{name}_{kind}": value for kind, value in prediction.items()})
        np.savez_compressed(output / "evaluation_arrays.npz", **saved)
        np.savez_compressed(output / "R_parameters.npz", **{
            f"{key}.{name}": value.detach().numpy() for key, model in models.items()
            for name, value in model.state_dict().items()})
        if any(state_fingerprint(model) != fingerprints[key] for key, model in models.items()):
            raise RuntimeError("Frozen endpoint evaluation changed R")
        if any(p.grad is not None for model in models.values() for p in model.parameters()):
            raise RuntimeError("Frozen endpoint evaluation left gradients")
        if any(sha256_file(path) != digest for path, digest in {**hashes, **code_sha}.items()):
            raise RuntimeError("Immutable cache or pinned source changed")
        guard(True)
        result = dict(status="complete", workflow_complete=True, returncode=0, recipe=RECIPE,
            records=records, latent_metrics=metrics, closure_scores=scores, final_R_sha256=fingerprints,
            target_stds=manifest["target_stds"], arms=ARMS, completed_runs=len(records),
            all_runs_selected_before_TEST_and_labels_loaded=True, input_and_code_sha256_unchanged=True,
            Q_loaded=False, C_loaded=False, Q_updates=0, C_updates=0, GPU_seconds=0, CPU_threads=1,
            cpu_seconds=time.process_time()-cpu_started, wall_seconds=time.perf_counter()-started,
            thermal_nu=None, critical_r=None, physical_fixed_point_certified=False, root_search=False,
            limitations=["All fits share one frozen Q/C; validation conditions were previously seen by source Q/C",
                "Four folds reuse the same eight TRAIN conditions and are not independent datasets",
                "Previously inspected TEST is exploratory and is not used for selection",
                "Four encoding repeats reuse physical chains; all source sampling warnings remain",
                "B has two scale-specific maps and cannot be interpreted as one autonomous RG flow",
                "C only approximately matches B parameter count; no training seed or architecture selected by TEST"])
        write_json(output / "summary.json", result)
        current = "complete"
        status("complete", error=None, requires_user_action=False)
        write_json(output / "public_files.json", {p.name: sha256_file(p) for p in output.iterdir()
            if p.is_file() and p.name != "public_files.json"})
        guard(True)
        return result
    except BaseException as error:
        status("budget_hold" if isinstance(error, BudgetHold) else "failed",
               error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(), requires_user_action=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cpu-budget-seconds", type=float, default=3600.)
    run(Options(**vars(parser.parse_args(argv))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
