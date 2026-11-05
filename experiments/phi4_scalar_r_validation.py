"""Fixed-u phi4: fresh scalar R fits on an immutable, repeated Q/C cache.

Only TRAIN encodings are decompressed before every endpoint is selected.  The
condition-level validation is independent of R fitting, not of source Q/C.
The original coordinate closure objective and source TRAIN target scales are
unchanged; internal R normalization is fixed using R-fit inputs alone.
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

from experiments.phi4_qcr import sha256_file, state_fingerprint, write_json
from experiments.phi4_ru_r_validation import ImprovementTracker, Schedule
from recurrent_rg.phi4_r_calibration import (
    FrozenResidualMap, all_start_closure_loss, fit_input_normalizer,
)


RECIPE = "phi4_u1_fresh_scalar_R_condition_validation_v1"
SEEDS = (91, 92, 93)
SPLIT_SEED = 20261030
OUTPUT_BUDGET_BYTES = 100 * 1024**2
PATHS = (("0_to_1", 0, 1, "one_step", 0),
         ("1_to_2", 1, 2, "one_step", 1),
         ("0_to_2", 0, 2, "rollout", 2))


@dataclass(frozen=True)
class Options:
    cache: str
    output: str
    cpu_budget_seconds: float = 1200.


class BudgetHold(RuntimeError):
    pass


def verify_cache(cache):
    """Hash public containers without decompressing NPZ labels or TEST."""
    cache = Path(cache).resolve()
    status_path, inventory_path = cache / "status.json", cache / "public_files.json"
    status = json.loads(status_path.read_text())
    if (status.get("status") != "complete" or status.get("workflow_complete") is not True
            or status.get("returncode") != 0 or status.get("error") is not None):
        raise ValueError("A completed scalar frozen cache is required")
    inventory = json.loads(inventory_path.read_text())
    if not isinstance(inventory, dict) or not {"cache.npz", "manifest.json"} <= set(inventory):
        raise ValueError("Cache inventory must pin cache.npz and manifest.json")
    hashes = {}
    for name, digest in inventory.items():
        path = (cache / name).resolve()
        if (path.parent != cache or path.suffix not in (".json", ".npz")
                or not isinstance(digest, str) or len(digest) != 64):
            raise ValueError("Invalid public scalar-cache member")
        if sha256_file(path) != digest:
            raise ValueError(f"Scalar cache SHA256 mismatch: {name}")
        hashes[str(path)] = digest
    for path in (status_path, inventory_path):
        hashes[str(path)] = sha256_file(path)
    manifest = json.loads((cache / "manifest.json").read_text())
    stds = np.asarray(manifest.get("target_stds", []), dtype=np.float64)
    if stds.shape != (3,) or not np.isfinite(stds).all() or np.any(stds <= 0):
        raise ValueError("Require three immutable original-TRAIN target standard deviations")
    source_modules = manifest.get("source_module_sha256", {})
    if (not isinstance(manifest.get("source_state_sha256"), str)
            or len(manifest["source_state_sha256"]) != 64
            or not isinstance(source_modules, dict)
            or any(not isinstance(source_modules.get(key), str)
                   or len(source_modules[key]) != 64 for key in ("Q", "C", "R"))):
        raise ValueError("Frozen source Q/C/R fingerprints are required")
    return manifest, hashes


def split_conditions(cell_ids):
    ids = np.asarray(cell_ids)
    if ids.shape != (32,) or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("Expected 32 TRAIN physical-chain condition IDs")
    cells, counts = np.unique(ids, return_counts=True)
    if len(cells) != 8 or np.any(counts != 4):
        raise ValueError("Require eight complete TRAIN conditions with four physical chains each")
    shuffled = np.random.default_rng(SPLIT_SEED).permutation(cells)
    validation_cells, fit_cells = np.sort(shuffled[:2]), np.sort(shuffled[2:])
    validation_mask = np.isin(ids, validation_cells)
    return dict(seed=SPLIT_SEED, fit_cells=fit_cells, validation_cells=validation_cells,
                fit_mask=~validation_mask, validation_mask=validation_mask,
                shuffled_cells=shuffled)


def _direct(value, chains, name):
    if (value.shape != (4, chains, 3, 1) or value.dtype != np.float64
            or not np.isfinite(value).all()):
        raise ValueError(f"Invalid {name}: expected four repeats of {chains} scalar three-scale chains")
    return value


def load_r_fit_and_validation(cache):
    """The only preselection member reads: TRAIN IDs and two TRAIN arrays."""
    with np.load(Path(cache) / "cache.npz", allow_pickle=False) as archive:
        ids = archive["train_cell_ids"].copy()
        fit = _direct(archive["fit_direct"].copy(), 32, "fit_direct")
        evaluation = _direct(archive["eval_train_direct"].copy(), 32, "eval_train_direct")
    split = split_conditions(ids)
    return (torch.tensor(fit[:, split["fit_mask"]], dtype=torch.float64),
            torch.tensor(evaluation[:, split["validation_mask"]], dtype=torch.float64),
            split, ids)


def initialize_model(fit, seed, *, hidden=96):
    center, scale = fit_input_normalizer(fit)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return FrozenResidualMap(1, hidden, center=center, scale=scale)


@torch.no_grad()
def validation_metrics(model, direct):
    if (direct.ndim != 4 or direct.shape[0] != 4 or direct.shape[-2:] != (3, 1)
            or direct.requires_grad or not torch.isfinite(direct).all()):
        raise ValueError("Expected four frozen scalar encoding repeats")
    first = model.advance(direct[..., 0, :])
    outputs = (first, model.advance(direct[..., 1, :]), model.advance(first))
    errors = [p - direct[..., target, :] for p, target in zip(outputs, (1, 2, 2))]
    names, weights = ("e01", "e12", "e02"), (5., 2.5, 2.5)
    chain = {name: float(error.mean(0).square().sum(-1).mean())
             for name, error in zip(names, errors)}
    repeat = {name: error.square().sum(-1).mean(-1).tolist()
              for name, error in zip(names, errors)}
    losses = [sum(weight * repeat[name][i] for name, weight in zip(names, weights))
              for i in range(direct.shape[0])]
    return dict(chain_mean_loss=sum(weight * chain[name] for name, weight in zip(names, weights)),
        chain_mean_terms=chain, repeat_losses=losses, repeat_terms=repeat,
        repeat_mean_loss=float(np.mean(losses)), physical_chains=int(direct.shape[1]),
        encoding_repeats=4, prediction_then_repeat_average=True,
        selection_uses_chain_mean_loss=True)


def train_seed(fit, validation, seed, *, schedule=None, hidden=96,
               guard=lambda: None, on_checkpoint=lambda *args: None):
    """Same frozen-input schedule/selection rule as the two-coordinate run."""
    schedule = Schedule() if schedule is None else schedule
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
            raise FloatingPointError("Nonfinite scalar R-fit loss")
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("Invalid scalar R-fit gradient")
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
        if (state_fingerprint(model) != before
                or any(parameter.grad is not None for parameter in model.parameters())):
            raise RuntimeError("Scalar validation altered model state or gradients")
        if not np.isfinite(float(train_loss)) or not np.isfinite(val["chain_mean_loss"]):
            raise FloatingPointError("Nonfinite accepted scalar endpoint")
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
            raise RuntimeError("Scalar Adam parameter clocks disagree")
        adam_final_state = deepcopy(model.state_dict())
        model.load_state_dict(adam_best_state, strict=True)
        lbfgs_start_sha = state_fingerprint(model)
        lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.,
            max_iter=schedule.lbfgs_block_iterations, max_eval=4*schedule.lbfgs_block_iterations,
            history_size=20, tolerance_grad=1e-9, tolerance_change=1e-12,
            line_search_fn="strong_wolfe")
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
                raise RuntimeError("Scalar L-BFGS evaluation counters disagree")
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
                raise RuntimeError("Fixed scalar input normalizer changed")
            models[name] = restored
        return models, dict(seed=seed, fresh_R=True, initial_R_sha256=initial_sha,
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
    except BaseException:
        for parameter in model.parameters():
            parameter.grad = None
        raise


@torch.no_grad()
def predict(model, direct):
    g = torch.as_tensor(direct, dtype=torch.float64)
    first = model.advance(g[..., 0, :])
    return dict(one_step=torch.stack((first, model.advance(g[..., 1, :])), -2).numpy(),
                rollout=torch.stack((g[..., 0, :], first, model.advance(first)), -2).numpy())


def endpoint_metrics(direct, predictions, target_stds):
    """Raw repeats and mean paired residuals share immutable TRAIN scales."""
    stds = np.asarray(target_stds, dtype=np.float64)
    if stds.shape != (3,) or not np.isfinite(stds).all() or np.any(stds <= 0):
        raise ValueError("Expected three positive source-TRAIN standard deviations")
    rows = []
    for path, start, target, kind, index in PATHS:
        output = np.asarray(predictions[kind])[..., index, :]
        residual = output-direct[..., target, :]
        baseline = direct[..., start, :]-direct[..., target, :]
        if output.shape != direct[..., target, :].shape or not np.isfinite(output).all():
            raise ValueError("Invalid scalar R prediction")
        raw_rmse = float(np.sqrt(np.mean(residual**2)))
        chain_rmse = float(np.sqrt(np.mean(residual.mean(0)**2)))
        baseline_rmse = float(np.sqrt(np.mean(baseline**2)))
        baseline_chain = float(np.sqrt(np.mean(baseline.mean(0)**2)))
        per_repeat = np.sqrt(np.mean(residual**2, axis=(1, 2)))
        std = float(stds[target])
        rows.append(dict(path=path, rmse=raw_rmse, nrmse=raw_rmse/std,
            raw_repeat_rmse=raw_rmse, raw_repeat_nrmse=raw_rmse/std,
            chain_mean_rmse=chain_rmse, chain_mean_nrmse=chain_rmse/std,
            train_target_std=std, target_scale=target,
            identity_rmse=baseline_rmse, identity_chain_mean_rmse=baseline_chain,
            relative_displacement_error=raw_rmse/baseline_rmse if baseline_rmse > 0 else None,
            repeat_rmse=per_repeat.tolist(), repeat_nrmse=(per_repeat/std).tolist(),
            physical_chains=int(direct.shape[1]), encoding_repeats=int(direct.shape[0]),
            prediction_then_repeat_average=True, repeats_are_not_independent_physical_chains=True))
    return rows


def load_evaluation_after_selection(cache, split, train_ids):
    """Call only once every seed has been selected; read labels and TEST now."""
    with np.load(Path(cache) / "cache.npz", allow_pickle=False) as archive:
        train = _direct(archive["eval_train_direct"].copy(), 32, "eval_train_direct")
        test = _direct(archive["eval_test_direct"].copy(), 20, "eval_test_direct")
        labels, test_labels = archive["train_labels"].copy(), archive["test_labels"].copy()
        test_ids = archive["test_cell_ids"].copy()
        source = {}
        for name, chains in (("train", 32), ("test", 20)):
            keys = {kind: f"eval_{name}_source_{kind}" for kind in ("one_step", "rollout")}
            present = [key in archive.files for key in keys.values()]
            if any(present) and not all(present):
                raise ValueError("Incomplete original-R comparison predictions")
            if all(present):
                source[name] = {kind: archive[key].copy() for kind, key in keys.items()}
                for kind, scales in (("one_step", 2), ("rollout", 3)):
                    value = source[name][kind]
                    if value.shape != (4, chains, scales, 1) or not np.isfinite(value).all():
                        raise ValueError("Invalid original-R comparison prediction")
    if (labels.shape != (32, 3) or test_labels.shape != (20, 3)
            or not np.isfinite(labels).all() or not np.isfinite(test_labels).all()
            or test_ids.shape != (20,) or not np.issubdtype(test_ids.dtype, np.integer)):
        raise ValueError("Invalid scalar evaluation labels or IDs")
    if set(train_ids) & set(test_ids):
        raise ValueError("Whole-condition TRAIN/TEST overlap")
    _, counts = np.unique(test_ids, return_counts=True)
    if counts.shape != (5,) or np.any(counts != 4):
        raise ValueError("Expected five TEST conditions with four chains each")
    saved = dict(train_direct=train, train_cell_ids=train_ids, train_labels=labels,
                 test_direct=test, test_cell_ids=test_ids, test_labels=test_labels)
    for name in ("fit", "validation"):
        mask = split[f"{name}_mask"]
        saved.update({f"{name}_direct": train[:, mask], f"{name}_cell_ids": train_ids[mask],
                      f"{name}_labels": labels[mask]})
    if source and set(source) != {"train", "test"}:
        raise ValueError("Original-R comparison must include both TRAIN and TEST")
    return saved, source


def _source_hashes():
    import experiments.phi4_qcr as qcr_source
    import experiments.phi4_ru_r_validation as schedule_source
    import recurrent_rg.phi4_r_calibration as model_source
    return {str(Path(module.__file__).resolve()): sha256_file(module.__file__)
            for module in (qcr_source, schedule_source, model_source)} | {
                str(Path(__file__).resolve()): sha256_file(__file__)}


def run(options):
    if (not math.isfinite(options.cpu_budget_seconds)
            or not 0 < options.cpu_budget_seconds <= 1200.):
        raise ValueError("CPU budget must be finite, positive, and no greater than 1200 seconds")
    output, cache = (Path(value).resolve() for value in (options.output, options.cache))
    if output == cache or output in cache.parents or cache in output.parents:
        raise ValueError("Output must be separate from immutable cache")
    output.mkdir(parents=True, exist_ok=False)
    started, cpu_started = time.perf_counter(), time.process_time()
    records, models, current = {}, {}, "verify_inputs"
    optimization_complete, last_size_check = False, -1.

    def guard(force=False):
        nonlocal last_size_check
        used = time.process_time()-cpu_started
        if used > options.cpu_budget_seconds:
            raise BudgetHold("Scalar R validation CPU budget reached")
        if force or used-last_size_check >= 1.:
            last_size_check = used
            if sum(path.stat().st_size for path in output.iterdir() if path.is_file()) > OUTPUT_BUDGET_BYTES:
                raise BudgetHold("Scalar R validation 100 MiB output budget reached")

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
        cache_manifest, hashes = verify_cache(cache)
        fit, validation, split, train_ids = load_r_fit_and_validation(cache)
        code_sha = _source_hashes()
        center, scale = fit_input_normalizer(fit)
        manifest = dict(schema_version=5, recipe=RECIPE, options=asdict(options),
            schedule=asdict(Schedule()), seeds=list(SEEDS), theory_dim=1, hidden=96, dtype="float64",
            input_sha256=hashes, source_sha256=code_sha, cache_manifest=cache_manifest,
            source_state_sha256=cache_manifest["source_state_sha256"],
            source_module_sha256=cache_manifest["source_module_sha256"],
            source_QCR_steps=3000, source_QC_seed=91,
            initialization="fresh random R; no old R endpoint loaded; frozen source Q/C",
            R_validation_is_independent_of_R_fit_conditions=True,
            R_validation_is_unseen_by_source_QC=False,
            split={key: value.tolist() if isinstance(value, np.ndarray) else value
                   for key, value in split.items()}, fit_shape=list(fit.shape),
            validation_shape=list(validation.shape),
            normalizer=dict(center=center.tolist(), scale=scale.tolist(),
                inputs="R-fit TRAIN repeats, input scales 0/1 only; fixed before optimization"),
            fit_objective="5*e01+2.5*e12+2.5*e02; original g; mean over repeats and physical chains",
            closure_objective="all_start_anchor", closure_scale_weights=None,
            validation_objective="predict each repeat, average paired residual within physical chain, then square",
            endpoint_selection="strict minimum R-validation loss within each seed; no cross-seed selection",
            target_stds=cache_manifest["target_stds"],
            target_std_scope="immutable original joint-QCR TRAIN evaluation, all 128 fields, one saved encoding; scales 0/1/2",
            labels_loaded_before_all_selection=False, TEST_loaded_before_all_selection=False,
            labels_used_by_optimizer=False, Q_loaded=False, C_loaded=False, Q_updates=0, C_updates=0,
            new_physical_sampling=False, new_root_search=False, CPU_threads=1, GPU_used=False,
            default_recipe_modified=False, source_sampling_warnings_retained=True)
        write_json(output / "manifest.json", manifest)
        for seed in SEEDS:
            current = f"fresh_scalar_R_seed{seed}"
            status("running")

            def save_checkpoint(stage, step, model, row):
                np.savez_compressed(output / f"checkpoint_seed{seed}_{stage}_{step:06d}.npz",
                    **{key: value.detach().numpy() for key, value in model.state_dict().items()})
                write_json(output / "history_progress.json", dict(current=current,
                    completed=records, current_progress=row))
                guard(True)

            trained, record = train_seed(fit, validation, seed, guard=guard,
                                          on_checkpoint=save_checkpoint)
            records[str(seed)] = record
            models.update({f"{name}_seed{seed}": model for name, model in trained.items()
                           if name in ("adam6000", "selected")})
            write_json(output / "history_progress.json", dict(completed=records))
            print(json.dumps(dict(seed=seed, selected_stage=record["selected_stage"],
                selected_step=record["selected_step"], validation_loss=record["selected_validation_loss"],
                lbfgs_stop=record["lbfgs_stopping"], cpu_seconds=record["cpu_seconds"])), flush=True)
        optimization_complete = True
        current = "selected_endpoint_evaluation"
        status("running")
        saved, source_predictions = load_evaluation_after_selection(cache, split, train_ids)
        metrics, scores = {}, {}
        fingerprints = {key: state_fingerprint(model) for key, model in models.items()}
        for key, model in models.items():
            guard()
            metrics[key], scores[key] = {}, {}
            for name in ("train", "fit", "validation", "test"):
                direct = saved[f"{name}_direct"]
                prediction = predict(model, direct)
                metrics[key][name] = endpoint_metrics(direct, prediction, manifest["target_stds"])
                scores[key][name] = validation_metrics(model, torch.tensor(direct, dtype=torch.float64))
                saved.update({f"eval_{key}_{name}_{kind}": value for kind, value in prediction.items()})
        if source_predictions:
            metrics["source_joint3000"] = {}
            for name, prediction in source_predictions.items():
                metrics["source_joint3000"][name] = endpoint_metrics(
                    saved[f"{name}_direct"], prediction, manifest["target_stds"])
                saved.update({f"eval_source_joint3000_{name}_{kind}": value
                              for kind, value in prediction.items()})
        np.savez_compressed(output / "evaluation_arrays.npz", **saved)
        np.savez_compressed(output / "R_parameters.npz", **{
            f"{key}.{name}": value.detach().numpy() for key, model in models.items()
            for name, value in model.state_dict().items()})
        write_json(output / "history.json", records)
        if any(state_fingerprint(model) != fingerprints[key] for key, model in models.items()):
            raise RuntimeError("Frozen scalar evaluation changed R")
        if any(parameter.grad is not None for model in models.values() for parameter in model.parameters()):
            raise RuntimeError("Frozen scalar evaluation left gradients")
        if any(sha256_file(path) != digest for path, digest in {**hashes, **code_sha}.items()):
            raise RuntimeError("Scalar cache or pinned source changed")
        guard(True)
        result = dict(status="complete", workflow_complete=True, returncode=0, recipe=RECIPE,
            records=records, latent_metrics=metrics, closure_scores=scores,
            final_R_sha256=fingerprints, target_stds=manifest["target_stds"],
            input_and_code_sha256_unchanged=True,
            all_seeds_selected_before_TEST_and_labels_loaded=True,
            selected_parameter_keys=[f"selected_seed{seed}" for seed in SEEDS],
            fair_Adam_baseline="adam6000_seed91/92/93; same six-condition fit and fresh initialization",
            original_R_comparison=bool(source_predictions),
            Q_loaded=False, C_loaded=False, Q_updates=0, C_updates=0, GPU_seconds=0,
            CPU_threads=1, cpu_seconds=time.process_time()-cpu_started,
            wall_seconds=time.perf_counter()-started, thermal_nu=None, critical_r=None,
            physical_fixed_point_certified=False, root_search=False,
            limitations=["R-validation conditions were seen by the frozen source Q/C; R-fit-only independence",
                "All R seeds share one source Q/C and the existing finite physical chains",
                "Only two condition-level R-validation points; model-selection uncertainty remains",
                "Previously inspected TEST is exploratory, not a new independent final test",
                "Four encoding repeats are not four independent physical ensembles",
                "All source sampling warnings remain; no equilibrium or critical-point certification"])
        write_json(output / "summary.json", result)
        current = "complete"
        status("complete", error=None, requires_user_action=False)
        write_json(output / "public_files.json", {path.name: sha256_file(path) for path in output.iterdir()
            if path.is_file() and path.name != "public_files.json"})
        guard(True)
        return result
    except BaseException as error:
        status("budget_hold" if isinstance(error, BudgetHold) else "failed",
            error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(),
            requires_user_action=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cpu-budget-seconds", type=float, default=1200.)
    run(Options(**vars(parser.parse_args(argv))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
