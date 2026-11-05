"""Frozen-Q/C R calibration on repeated, training-size phi4 encodings.

The source experiment is immutable.  This new CPU stage sees fixed TRAIN g
only, retains the raw-coordinate all-start objective, and evaluates all fixed
6000-step endpoints.  Couplings, TEST encodings and decoder diagnostics do not
enter optimization.  The source QCR seed remains 91; the three new R seeds are
optimization repeats, not three independently trained Q/C models.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from experiments.phi4_qcr import sha256_file, state_fingerprint, write_json
from recurrent_rg.phi4_r_calibration import (
    FrozenResidualMap, all_start_closure_loss, conditioned_copy,
    fit_input_normalizer, function_preserving_widen,
)


RECIPE = "phi4_ru_frozen_QC_repeated64_R_calibration_v1"
ARMS = ("raw96", "scaled96", "scaled256")
SEEDS = (91, 92, 93)
PATHS = (("0_to_1", 0, 1, "one_step", 0),
         ("1_to_2", 1, 2, "one_step", 1),
         ("0_to_2", 0, 2, "rollout", 2))
DISPLAY = {"source_seed0": "原联合训练 R"} | {
    f"{arm}_seed{seed}": f"{name}，种子{seed}"
    for arm, name in (("raw96", "原坐标96"), ("scaled96", "归一化96"),
                      ("scaled256", "归一化256")) for seed in SEEDS}


@dataclass
class Options:
    cache: str
    reference: str
    output: str
    steps: int = 6000
    cpu_budget_seconds: float = 1200.
    output_budget_bytes: int = 100 * 1024 ** 2


class BudgetHold(RuntimeError):
    pass


def load_cache(root):
    root = Path(root).resolve()
    state = json.loads((root / "status.json").read_text())
    if state.get("status") != "complete" or not state.get("workflow_complete"):
        raise ValueError("Collection must be complete before R calibration")
    inventory = json.loads((root / "public_files.json").read_text())
    hashes = {}
    for name, digest in inventory.items():
        path = (root / name).resolve()
        if path.parent != root or path.suffix not in (".json", ".npz"):
            raise ValueError("Invalid public cache member")
        if sha256_file(path) != digest:
            raise ValueError(f"Cache SHA mismatch: {name}")
        hashes[str(path)] = digest
    hashes[str(root / "public_files.json")] = sha256_file(root / "public_files.json")
    hashes[str(root / "status.json")] = sha256_file(root / "status.json")
    with np.load(root / "cache.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    for name, chains in (("fit", 136), ("eval_train", 136), ("eval_test", 44)):
        direct = arrays[f"{name}_direct"]
        if direct.shape != (4, chains, 3, 2) or not np.isfinite(direct).all():
            raise ValueError(f"Invalid {name} g shape or values")
        target = arrays[f"{name}_feature_target"]
        if target.shape != (4, chains, 3, 105) or not np.isfinite(target).all():
            raise ValueError(f"Invalid {name} feature targets")
        subsets = arrays[f"{name}_subset_indices"]
        if subsets.shape != (4, chains, 64):
            raise ValueError("Subset recording incomplete")
        if any(len(np.unique(row)) != 64 for row in subsets.reshape(-1, 64)):
            raise ValueError("Subsets must contain 64 distinct saved configurations")
    if set(arrays["train_cell_ids"]) & set(arrays["test_cell_ids"]):
        raise ValueError("Whole-condition TRAIN/TEST split violated")
    return arrays, hashes, json.loads((root / "manifest.json").read_text())


@torch.no_grad()
def predict(model, direct):
    g = torch.as_tensor(direct, dtype=torch.float64)
    first = model.advance(g[..., 0, :])
    one = torch.stack((first, model.advance(g[..., 1, :])), dim=-2)
    roll = torch.stack((g[..., 0, :], first, model.advance(first)), dim=-2)
    return {"one_step": one.numpy(), "rollout": roll.numpy()}


def endpoint_metrics(g, predictions, stds, feature_target=None, decoder=None):
    rows = []
    if feature_target is not None:
        weight, bias = decoder
    for path, start, target, kind, index in PATHS:
        prediction = predictions[kind][..., index, :]
        truth = g[..., target, :]
        identity = g[..., start, :]
        if feature_target is not None:
            prediction = prediction @ weight.T + bias
            truth = feature_target[..., target, :]
            identity = g[..., start, :] @ weight.T + bias
        residual = prediction - truth
        rmse = float(np.sqrt(np.mean(residual ** 2)))
        baseline = float(np.sqrt(np.mean((identity - truth) ** 2)))
        # All seeds, paths and noise repeats use the same immutable source
        # TRAIN reference standard deviation, not a seed-adapted denominator.
        std = float(stds[target - 1])
        per_repeat = np.sqrt(np.mean(residual ** 2, axis=(1, 2)))
        rows.append(dict(path=path, rmse=rmse, nrmse=rmse / std if std > 0 else None,
            train_target_std=std, identity_rmse=baseline,
            relative_displacement_error=rmse / baseline if baseline > 0 else None,
            repeat_rmse=per_repeat.tolist(), repeat_nrmse=(per_repeat / std).tolist() if std > 0 else None,
            coordinate_rmse=np.sqrt(np.mean(residual ** 2, axis=(0, 1))).tolist(),
            physical_chains=g.shape[1], C_and_subset_repeats=g.shape[0],
            repeats_are_not_independent_physical_chains=True))
    return rows


def jacobian_statistics(model, points):
    x = torch.as_tensor(points, dtype=torch.float64).reshape(-1, 2).detach().requires_grad_(True)
    y = model.advance(x)
    jac = torch.stack([torch.autograd.grad(y[:, j].sum(), x, retain_graph=j == 0)[0]
                       for j in range(2)], dim=1).detach().numpy()
    singular = np.linalg.svd(jac, compute_uv=False)[:, 0]
    return dict(full_DR_includes_identity=True, count=len(singular),
        largest_singular_value_median=float(np.median(singular)),
        largest_singular_value_p95=float(np.quantile(singular, .95)),
        largest_singular_value_max=float(singular.max()),
        values_are_not_fixed_point_eigenvalues=True), jac


def bootstrap_differences(arrays, predictions, *, count=1000):
    """Resample TEST parameter conditions, retaining chains and C repeats."""
    cells = np.asarray(arrays["test_cell_ids"])
    unique = np.unique(cells)
    rng = np.random.default_rng(20261030)
    sampled = rng.integers(len(unique), size=(count, len(unique)))
    truth = arrays["eval_test_direct"]
    source = predictions["source_seed0"]["test"]
    result = {}
    for key, split_predictions in predictions.items():
        if key == "source_seed0":
            continue
        rows = []
        for path, _, target, kind, index in PATHS:
            def condition_mse(p):
                squared = (p[kind][..., index, :] - truth[..., target, :]) ** 2
                return np.asarray([squared[:, cells == cell].mean() for cell in unique])
            a, b = condition_mse(split_predictions["test"]), condition_mse(source)
            delta = np.sqrt(a[sampled].mean(1)) - np.sqrt(b[sampled].mean(1))
            rows.append(dict(path=path, rmse_difference=float(np.sqrt(a.mean()) - np.sqrt(b.mean())),
                interval95=np.quantile(delta, [.025, .975]).tolist()))
        result[key] = rows
    return dict(replicates=count, independent_units="11 TEST parameter conditions; 4 physical chains per condition",
        C_repeats_kept_together=True, paired_across_arms=True, by_model=result,
        qualification="Conditional on one trained Q/C and existing physical chains; no seed or equilibrium certification; no multiplicity correction")


def repeat_stability(g, prediction):
    rows = []
    for path, start, target, kind, index in PATHS:
        output = prediction[kind][..., index, :]
        error = output - g[..., target, :]
        variation = lambda x: float(np.sqrt(np.mean(np.var(x, axis=0, ddof=0))))
        numerator, denominator = variation(output), variation(g[..., start, :])
        rows.append(dict(path=path, input_repeat_rms=denominator,
            target_repeat_rms=variation(g[..., target, :]), prediction_repeat_rms=numerator,
            paired_error_repeat_rms=variation(error),
            prediction_to_input_repeat_rms=numerator / denominator if denominator > 0 else None,
            qualification="Joint effect of resampled 64-field subsets and C noise; not a physical-chain error bar"))
    return rows


def _directory_bytes(root):
    return sum(p.stat().st_size for p in root.iterdir() if p.is_file())


def run(options):
    if options.steps != 6000:
        raise ValueError("This registered calibration fixes 6000 R updates")
    output, cache, reference = map(lambda p: Path(p).resolve(),
                                   (options.output, options.cache, options.reference))
    if any(output == p or p in output.parents for p in (cache, reference)):
        raise ValueError("Output must not overwrite or nest under immutable inputs")
    output.mkdir(parents=True, exist_ok=False)
    started, cpu_started = time.perf_counter(), time.process_time()
    models, actual, histories = {}, {}, []
    def status(state, **extra):
        write_json(output / "status.json", dict(recipe=RECIPE, status=state,
            workflow_complete=state == "complete", actual_steps_by_model=actual.copy(),
            returncode=0 if state == "complete" else 3 if state == "budget_hold" else 1 if state == "failed" else None,
            cpu_seconds=time.process_time() - cpu_started,
            wall_seconds=time.perf_counter() - started, **extra))
    def budget():
        if time.process_time() - cpu_started > options.cpu_budget_seconds:
            raise BudgetHold("Fixed CPU budget reached")
        if _directory_bytes(output) > options.output_budget_bytes:
            raise BudgetHold("Fixed output budget reached")
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        status("running", current="verify_frozen_cache")
        arrays, hashes, cache_manifest = load_cache(cache)
        reference_file = reference / "evaluation_arrays.npz"
        reference_inventory = reference.parent / "public_files.json"
        inventory = json.loads(reference_inventory.read_text())
        if inventory.get("run/evaluation_arrays.npz") != sha256_file(reference_file):
            raise ValueError("Source reference arrays disagree with their public inventory")
        source_summary_path = reference / "summary.json"
        source_summary = json.loads(source_summary_path.read_text())
        if (cache_manifest.get("source_state_sha256") != source_summary["final_state_sha256"]
                or cache_manifest.get("source_module_sha256") != source_summary["final_module_sha256"]):
            raise ValueError("Cache Q/C/R source differs from the immutable reference run")
        hashes[str(reference_inventory)] = sha256_file(reference_inventory)
        hashes[str(source_summary_path)] = sha256_file(source_summary_path)
        with np.load(reference_file, allow_pickle=False) as archive:
            old_train = archive["train_direct"].copy()
            old_features = archive["train_feature_target"].copy()
        hashes[str(reference_file)] = sha256_file(reference_file)
        stds = [float(np.sqrt(np.var(old_train[:, k], axis=0, ddof=0).mean())) for k in (1, 2)]
        feature_stds = [float(np.sqrt(np.var(old_features[:, k], axis=0, ddof=0).mean())) for k in (1, 2)]
        direct = torch.tensor(arrays["fit_direct"], dtype=torch.float64)
        center, scale = fit_input_normalizer(direct)
        initial, paired = {}, {}
        for seed in SEEDS:
            torch.manual_seed(seed)
            raw = FrozenResidualMap(theory_dim=2, hidden=96, center=torch.zeros(2), scale=torch.ones(2))
            normalized = conditioned_copy(raw, center, scale)
            wide = conditioned_copy(function_preserving_widen(raw, 256, seed=seed + 10000), center, scale)
            for arm, model in zip(ARMS, (raw, normalized, wide)):
                key = f"{arm}_seed{seed}"
                models[key], actual[key] = model, 0
                initial[key] = state_fingerprint(model)
            with torch.no_grad():
                probe = direct.reshape(-1, 2)
                paired[str(seed)] = {"scaled96": float((raw.advance(probe) - normalized.advance(probe)).abs().max()),
                                      "scaled256": float((raw.advance(probe) - wide.advance(probe)).abs().max())}
        if max(v for row in paired.values() for v in row.values()) > 1e-12:
            raise RuntimeError("Matched physical initial functions failed")
        optimizers = {key: torch.optim.Adam(model.parameters(), lr=.001) for key, model in models.items()}
        source_paths = [Path(__file__), Path(__file__).parents[1] / "recurrent_rg/phi4_r_calibration.py",
                        Path(__file__).parents[1] / "scripts/report_phi4_ru_r_calibration.py",
                        Path(__file__).parents[1] / "scripts/report_phi4_ru_frozen_r.py",
                        Path(__file__).parents[1] / "scripts/report_phi4_qcr.py",
                        Path(__file__).parents[1] / "experiments/phi4_qcr.py"]
        code_sha = {str(p.resolve()): sha256_file(p) for p in source_paths}
        manifest = dict(schema_version=5, recipe=RECIPE, options=asdict(options),
            source_QCR_steps=3000, QCR_seed=91, R_optimizer_seeds=list(SEEDS),
            requested_R_updates_per_model=options.steps, input_sha256=hashes, source_sha256=code_sha,
            cache_manifest=cache_manifest, Q_updates=0, C_updates=0, Q_loaded=False, C_loaded=False,
            R_initialization="fresh paired physical function for all three capacities per seed; wide extra output weights start at zero",
            center=center.tolist(), scale=scale.tolist(), normalizer_scope="FIT TRAIN repetitions, input scales 0/1 only; pooled, frozen",
            closure_objective="all_start_anchor", closure_scale_weights=None,
            closure_loss="5*e01+2.5*e12+2.5*e02; sum coordinates then mean physical chains/repeats; raw g",
            learning_rate=.001, optimizer="fresh Adam, fixed lr; no scheduler, clipping, checkpoint selection or weight decay",
            batch="full 4 repeats x 136 TRAIN physical chains; C repeats are repeated measurements, not independent chains",
            labels_used_by_optimizer=False, TEST_used_by_optimizer=False, full_batch=True,
            paired_initial_max_error=paired, initial_R_sha256=initial,
            parameter_counts={key: sum(p.numel() for p in model.parameters()) for key, model in models.items()},
            metric_reference="immutable source128-field TRAIN target std, common across seeds and arms",
            target_stds=stds, feature_target_stds=feature_stds,
            evaluation_plan="four disjoint RNG-repeat caches per split; all fixed6000 endpoints; existing TEST has been inspected before",
            CPU_threads=1, GPU_used=False, default_recipe_modified=False,
            no_new_physical_sampling=True, new_root_search=False, thermal_nu=None)
        write_json(output / "manifest.json", manifest)
        for step in range(options.steps + 1):
            if step % 25 == 0:
                budget()
            if step:
                for key, model in models.items():
                    loss, _ = all_start_closure_loss(model, direct)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite {key} loss")
                    optimizers[key].zero_grad(set_to_none=True)
                    loss.backward()
                    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                        raise FloatingPointError(f"Invalid {key} gradient")
                    optimizers[key].step()
                    actual[key] = step
            if step % 100 == 0 or step == options.steps:
                with torch.no_grad():
                    row = dict(step=step, arms={}, cpu_seconds=time.process_time() - cpu_started)
                    for key, model in models.items():
                        loss, terms = all_start_closure_loss(model, direct)
                        row["arms"][key] = dict(loss=float(loss), **{k: float(v) for k, v in terms.items()})
                histories.append(row)
                write_json(output / "history_progress.json", dict(history=histories, actual_steps_by_model=actual.copy()))
                status("running", current="R_only_calibration")
                if step % 500 == 0:
                    print(json.dumps(dict(step=step, cpu_seconds=row["cpu_seconds"],
                        losses={k: v["loss"] for k, v in row["arms"].items()})), flush=True)
        endpoint_sha = {key: state_fingerprint(model) for key, model in models.items()}
        for model in models.values():
            for parameter in model.parameters():
                parameter.grad = None
        status("running", current="fixed_endpoint_evaluation")
        decoder = arrays["decoder_weight"], arrays["decoder_bias"]
        predictions = {"source_seed0": {split: {kind: arrays[f"eval_{split}_source_{kind}"]
            for kind in ("one_step", "rollout")} for split in ("train", "test")}}
        for key, model in models.items():
            predictions[key] = {split: predict(model, arrays[f"eval_{split}_direct"])
                                for split in ("train", "test")}
        metrics, native, repeat, derivatives = {}, {}, {}, {}
        saved = {f"{split}_{name}": arrays[f"{split}_{name}"] for split in ("train", "test")
                 for name in ("labels", "cell_ids")}
        for split in ("train", "test"):
            saved[f"{split}_direct"] = arrays[f"eval_{split}_direct"]
        # Arrows are drawn at actual TRAIN input points, not through unsupported
        # holes of the joint convex hull.  No root interpolation/search occurs.
        flow = arrays["fit_direct"][0, :, :2].reshape(-1, 2)
        saved.update(flow_g=flow, flow_in_train_hull=np.ones(len(flow), dtype=bool),
                     flow_near_train_support=np.ones(len(flow), dtype=bool))
        # Recover source flow at the same observed points from collector output.
        saved["source_seed0_flow_R"] = arrays["fit_source_one_step"][0].reshape(-1, 2)
        for key, split_predictions in predictions.items():
            metrics[key], native[key], repeat[key] = [], [], {}
            for split, values in split_predictions.items():
                g, target = arrays[f"eval_{split}_direct"], arrays[f"eval_{split}_feature_target"]
                metrics[key].extend(dict(split=split, **row) for row in endpoint_metrics(g, values, stds))
                native[key].extend(dict(split=split, **row) for row in endpoint_metrics(
                    g, values, feature_stds, feature_target=target, decoder=decoder))
                repeat[key][split] = repeat_stability(g, values)
                for name, value in values.items():
                    saved[f"eval_{key}_{split}_{name}"] = value
            if key in models:
                with torch.no_grad():
                    saved[f"{key}_flow_R"] = models[key].advance(torch.tensor(flow)).numpy()
                derivatives[key] = {}
                for split in ("train", "test"):
                    info, jac = jacobian_statistics(models[key], arrays[f"eval_{split}_direct"][..., :2, :])
                    derivatives[key][split] = info
                    saved[f"{key}_{split}_full_DR"] = jac
            budget()
        q_floor = {}
        for split in ("train", "test"):
            g, target = arrays[f"eval_{split}_direct"], arrays[f"eval_{split}_feature_target"]
            q_floor[split] = [{"target_scale": k,
                "rmse": float(np.sqrt(np.mean((g[..., k, :] @ decoder[0].T + decoder[1] - target[..., k, :]) ** 2))),
                "nrmse": float(np.sqrt(np.mean((g[..., k, :] @ decoder[0].T + decoder[1] - target[..., k, :]) ** 2)) / feature_stds[k - 1])}
                for k in (1, 2)]
        bootstrap = bootstrap_differences(arrays, predictions)
        clocks = {}
        for key, optimizer in optimizers.items():
            values = [int(state["step"].item()) for state in optimizer.state.values()]
            if not values or min(values) != options.steps or max(values) != options.steps:
                raise RuntimeError("Every R Adam parameter must reach the exact endpoint")
            clocks[key] = dict(min=min(values), max=max(values), parameter_tensors=len(values))
        np.savez_compressed(output / "evaluation_arrays.npz", **saved)
        np.savez_compressed(output / "R_parameters.npz", **{f"{key}.{name}": tensor.detach().numpy()
            for key, model in models.items() for name, tensor in model.state_dict().items()})
        write_json(output / "metrics.json", dict(latent=metrics, native_features=native,
            Q_feature_reconstruction=q_floor, repeat_stability=repeat, derivatives=derivatives,
            paired_condition_bootstrap=bootstrap))
        from scripts.report_phi4_ru_r_calibration import write_plots
        write_plots(output, saved, metrics, histories, DISPLAY)
        budget()
        if (any(state_fingerprint(model) != endpoint_sha[key] for key, model in models.items())
                or any(p.grad is not None for model in models.values() for p in model.parameters())):
            raise RuntimeError("R state or parameter gradients changed during frozen evaluation")
        if any(sha256_file(path) != digest for path, digest in {**hashes, **code_sha}.items()):
            raise RuntimeError("An immutable source/cache/code file changed")
        summary = dict(recipe=RECIPE, status="complete", workflow_complete=True, returncode=0,
            source_QCR_steps=3000, QCR_seed=91, R_optimizer_seeds=list(SEEDS),
            actual_steps_by_model=actual.copy(), optimizer_steps=clocks,
            source_and_cache_fingerprints_unchanged=True,
            final_R_sha256=endpoint_sha, frozen_R_evaluation_verified=True,
            Q_updates=0, C_updates=0, labels_used_by_optimizer=False,
            source_collection_verified=cache_manifest,
            latent_metrics=metrics, native_feature_metrics=native, Q_feature_reconstruction=q_floor,
            repeat_stability=repeat, full_DR_statistics=derivatives,
            paired_condition_bootstrap=bootstrap, thermal_nu=None, critical_r=None,
            new_root_search=False, physical_fixed_point_certified=False,
            cpu_seconds=time.process_time() - cpu_started, wall_seconds=time.perf_counter() - started,
            directory_bytes=_directory_bytes(output), GPU_seconds=0,
            limitations=["One fixed trained Q/C seed and finite physical chains; R seeds do not measure full-QCR robustness",
                "Four fixed fit subset/C repetitions approximate, but do not exhaust, the original stochastic training distribution",
                "Existing TEST parameter conditions have been inspected previously; not an untouched final test",
                "Source physical sampling warnings persist; no new equilibrium certification",
                "Small latent closure errors do not certify physical statistics, phase boundaries or fixed-point spectra"])
        write_json(output / "summary.json", summary)
        write_json(output / "public_files.json", {p.name: sha256_file(p) for p in output.iterdir()
            if p.is_file() and p.name not in ("status.json", "public_files.json")})
        status("complete", current="complete", error=None, requires_user_action=False,
               summary_sha256=sha256_file(output / "summary.json"))
        print(json.dumps(dict(status="complete", cpu_seconds=summary["cpu_seconds"], output=str(output))), flush=True)
        return summary
    except BaseException as error:
        status("budget_hold" if isinstance(error, BudgetHold) else "failed",
            current="stopped", error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(), requires_user_action=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "reference", "output"):
        parser.add_argument("--" + name, required=True)
    run(Options(**vars(parser.parse_args(argv))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
