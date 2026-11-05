"""Conditional fixed-C Q-dimension ablation on immutable paired soft features.

Fresh dg=1/2 encoders see identical finite cached TRAIN batches. Both Q fits
and all six R selections finish before TEST or coupling labels are accessed.
The frozen-C finite-cache experiment is not an exact joint online QCR rerun.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from experiments.phi4_qcr import sha256_file, state_fingerprint, write_json
from experiments.phi4_ru_r_validation import Schedule, train_seed as train_vector_r
from experiments.phi4_scalar_r_validation import split_conditions, train_seed as train_scalar_r
from recurrent_rg.phi4_qcr import ContinuousTheoryEncoder, FEATURE_DIM


RECIPE = "phi4_u1_fixed_C_Q_dimension_comparison_v1"
Q_STEPS = 3000
R_SEEDS = (91, 92, 93)
OUTPUT_BUDGET_BYTES = 256*1024**2
PATHS = (("0_to_1", 0, 1, "one_step", 0),
         ("1_to_2", 1, 2, "one_step", 1),
         ("0_to_2", 0, 2, "rollout", 2))


@dataclass(frozen=True)
class Options:
    features: str
    output: str
    q_seed: int = 91
    cpu_budget_seconds: float = 1200.


class BudgetHold(RuntimeError):
    pass


def verify_features(root):
    """Hash feature containers without decompressing labels or TEST."""
    root = Path(root).resolve()
    inventory_path = root / "public_files.json"
    inventory = json.loads(inventory_path.read_text())
    if set(inventory) != {"features.npz", "manifest.json", "receipt.json", "status.json"}:
        raise ValueError("Expected the complete four-file paired-feature cache")
    pins = {str(inventory_path): sha256_file(inventory_path)}
    for name, digest in inventory.items():
        path = (root / name).resolve()
        if (path.parent != root or not isinstance(digest, str) or len(digest) != 64
                or sha256_file(path) != digest):
            raise ValueError(f"Paired-feature SHA256 mismatch: {name}")
        pins[str(path)] = digest
    manifest = json.loads((root / "manifest.json").read_text())
    status = json.loads((root / "status.json").read_text())
    receipt = json.loads((root / "receipt.json").read_text())
    if (status.get("status") != "complete" or status.get("workflow_complete") is not True
            or status.get("returncode") != 0 or status.get("error") is not None
            or status.get("requires_user_action") is not False
            or manifest.get("recipe") != "phi4_u1_frozen_C_paired_half_features_v1"
            or manifest.get("lattice_sizes") != [36, 12, 4]
            or manifest.get("feature_dim") != 105
            or manifest.get("train_chains") != 32 or manifest.get("test_chains") != 20
            or manifest.get("fit_repeats") != 4 or manifest.get("evaluation_repeats") != 4
            or manifest.get("source_training_steps") != 3000
            or receipt.get("source_unchanged") is not True
            or receipt.get("features_sha256") != inventory["features.npz"]
            or not isinstance(manifest.get("source_module_sha256", {}).get("C"), str)
            or len(manifest["source_module_sha256"]["C"]) != 64):
        raise ValueError("Require the completed pinned fixed-C feature cache")
    return manifest, pins


def half_features(value, chains, name):
    if (value.shape != (4, chains, 3, 2, FEATURE_DIM) or value.dtype != np.float64
            or not np.isfinite(value).all() or np.any(value < 0)):
        raise ValueError(f"Invalid paired half features: {name}")
    if not np.allclose(value.sum(-1), 1., rtol=0., atol=1e-10):
        raise ValueError("Each soft-feature vector must have unit mass")
    return value


def load_train_only(root):
    with np.load(Path(root) / "features.npz", allow_pickle=False) as archive:
        ids = archive["train_cell_ids"].copy()
        fit = half_features(archive["fit_half_features"].copy(), 32, "fit")
        evaluation = half_features(archive["eval_train_half_features"].copy(), 32, "eval_train")
    return torch.tensor(fit), torch.tensor(evaluation), ids, split_conditions(ids)


def initialize_paired_q(fit, seed):
    """Pair every shared hidden layer and the first output/decoder direction."""
    if (fit.requires_grad or fit.shape[-2:] != (2, 105) or not torch.isfinite(fit).all()
            or not isinstance(seed, int) or isinstance(seed, bool) or seed < 0):
        raise ValueError("Require frozen half features and a nonnegative Q seed")
    mean = fit.mean(tuple(range(fit.ndim-1)))
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        two = ContinuousTheoryEncoder(96, 2)
        one = ContinuousTheoryEncoder(96, 1)
        with torch.no_grad():
            for index in (0, 2, 4):
                one.net[index].load_state_dict(two.net[index].state_dict())
            one.net[6].weight.copy_(two.net[6].weight[:1])
            one.net[6].bias.copy_(two.net[6].bias[:1])
            one.decoder.weight.copy_(two.decoder.weight[:, :1])
            for model in (one, two):
                model.decoder.bias.copy_(mean)
                model.normalize_decoder_()
    return {1: one, 2: two}


def q_objective(q, features):
    if (features.requires_grad or features.ndim < 4 or features.shape[-3:] != (3, 2, 105)
            or not torch.isfinite(features).all()):
        raise ValueError("Require frozen paired three-scale soft features")
    encoded = q.from_features(features)
    direct = encoded.mean(-2)
    target = features.mean(-2)
    reconstruction = (q.decode(direct)-target).square().sum(-1).mean()
    subset = (encoded[..., 0, :]-encoded[..., 1, :]).square().sum(-1)
    micro = subset[..., 0].mean()
    coarse = subset[..., 1:].mean()
    return 10*reconstruction+20*micro+5*coarse, dict(
        reconstruction=reconstruction, subset_micro=micro, subset_coarse=coarse)


def train_q(q, fit, seed, *, steps=Q_STEPS, checkpoint_every=100,
            guard=lambda: None, on_checkpoint=lambda *args: None):
    if steps < 1 or checkpoint_every < 1 or steps % checkpoint_every:
        raise ValueError("Q update endpoint must be divisible by checkpoint interval")
    if fit.shape != (4, 32, 3, 2, 105) or fit.requires_grad:
        raise ValueError("Q fits use four frozen repeats of all 32 TRAIN physical chains")
    q.train()
    initial = state_fingerprint(q)
    optimizer = torch.optim.Adam(q.parameters(), lr=.001)
    batch_rng = np.random.default_rng(int(seed)+710000)
    repeat_rng = np.random.default_rng(int(seed)+720000)
    digest = hashlib.sha256()
    history = []

    def checkpoint(step):
        guard()
        for parameter in q.parameters():
            parameter.grad = None
        before = state_fingerprint(q)
        with torch.no_grad():
            loss, terms = q_objective(q, fit)
        if not torch.isfinite(loss) or state_fingerprint(q) != before:
            raise FloatingPointError("Invalid frozen Q checkpoint")
        row = dict(step=step, loss=float(loss), Q_sha256=before,
                   terms={key: float(value) for key, value in terms.items()})
        history.append(row)
        on_checkpoint(row)

    checkpoint(0)
    for step in range(1, steps+1):
        guard()
        indices = batch_rng.choice(32, size=8, replace=False).astype(np.int64)
        repeats = repeat_rng.integers(0, 4, size=8, dtype=np.int64)
        digest.update(indices.tobytes())
        digest.update(repeats.tobytes())
        batch = fit[torch.from_numpy(repeats), torch.from_numpy(indices)]
        optimizer.zero_grad(set_to_none=True)
        loss, _ = q_objective(q, batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite Q-only loss")
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in q.parameters()):
            raise FloatingPointError("Invalid Q-only gradient")
        optimizer.step()
        q.normalize_decoder_()
        if step % checkpoint_every == 0:
            checkpoint(step)
    clocks = [int(state["step"].item()) for state in optimizer.state.values()]
    if len(clocks) != len(list(q.parameters())) or min(clocks) != steps or max(clocks) != steps:
        raise RuntimeError("Q Adam parameter clocks disagree")
    q.eval().requires_grad_(False)
    for parameter in q.parameters():
        parameter.grad = None
    return dict(actual_steps=steps, adam_parameter_clock_min=min(clocks),
        adam_parameter_clock_max=max(clocks), initial_Q_sha256=initial,
        final_Q_sha256=state_fingerprint(q), batch_stream_sha256=digest.hexdigest(),
        batch_size=8, distinct_physical_chains_per_batch=True,
        private_chain_rng_seed=int(seed)+710000, private_repeat_rng_seed=int(seed)+720000,
        decoder_columns_normalized=True, history=history, C_updates=0, R_updates=0)


@torch.no_grad()
def encode(q, features):
    # Averaging two encodings is deliberate: nonlinear Q(mean f) is different.
    return q.from_features(features).mean(-2).detach()


@torch.no_grad()
def predict_r(model, direct):
    first = model.advance(direct[..., 0, :])
    return dict(one_step=torch.stack((first, model.advance(direct[..., 1, :])), -2).numpy(),
                rollout=torch.stack((direct[..., 0, :], first, model.advance(first)), -2).numpy())


def residual_metrics(residual):
    value = np.asarray(residual, dtype=np.float64)
    if value.ndim != 3 or value.shape[0] != 4 or not np.isfinite(value).all():
        raise ValueError("Expected finite [repeat,physical_chain,coordinate] residuals")
    return dict(raw_repeat_rmse=float(np.sqrt(np.mean(value**2))),
        chain_mean_rmse=float(np.sqrt(np.mean(value.mean(0)**2))),
        repeat_rmse=np.sqrt(np.mean(value**2, axis=(1, 2))).tolist(),
        coordinate_rmse=np.sqrt(np.mean(value**2, axis=(0, 1))).tolist(),
        physical_chains=int(value.shape[1]), encoding_repeats=4,
        prediction_then_repeat_average=True, repeats_are_not_independent_physical_chains=True)


@torch.no_grad()
def endpoint_metrics(q, direct, prediction, target, latent_stds):
    g = direct.detach().numpy()
    weight = q.normalized_decoder_weight().numpy()
    bias = q.decoder.bias.numpy()
    target = np.asarray(target, dtype=np.float64)
    if target.shape != (*g.shape[:-1], 105) or not np.isfinite(target).all():
        raise ValueError("Common feature targets must match direct encodings")
    feature_rows, latent_rows = [], []
    for path, start, end, kind, index in PATHS:
        output = prediction[kind][..., index, :]
        decoded = output @ weight.T+bias
        reconstruction = g[..., end, :] @ weight.T+bias
        identity_decoded = g[..., start, :] @ weight.T+bias
        feature_rows.append(dict(path=path, target_scale=end,
            prediction=residual_metrics(decoded-target[..., end, :]),
            target_Q_reconstruction=residual_metrics(reconstruction-target[..., end, :]),
            identity_decoded=residual_metrics(identity_decoded-target[..., end, :]),
            identity_feature=residual_metrics(target[..., start, :]-target[..., end, :]),
            feature_dim=105, directly_comparable_across_dg=True))
        latent = residual_metrics(output-g[..., end, :])
        std = np.asarray(latent_stds[end], dtype=np.float64)
        latent_rows.append(dict(path=path, target_scale=end, **latent,
            target_coordinate_stds=std.tolist(),
            raw_repeat_coordinate_nrmse=(np.sqrt(np.mean((output-g[..., end, :])**2, axis=(0, 1)))/std).tolist(),
            directly_comparable_across_dg=False))
    return feature_rows, latent_rows


def load_postselection(root, train_ids):
    with np.load(Path(root) / "features.npz", allow_pickle=False) as archive:
        test = half_features(archive["eval_test_half_features"].copy(), 20, "eval_test")
        targets = {name: archive[f"eval_{name}_feature_target"].copy() for name in ("train", "test")}
        labels = {name: archive[f"{name}_labels"].copy() for name in ("train", "test")}
        test_ids = archive["test_cell_ids"].copy()
    if (test_ids.shape != (20,) or not np.issubdtype(test_ids.dtype, np.integer)
            or len(np.unique(test_ids)) != 5 or np.any(np.unique(test_ids, return_counts=True)[1] != 4)
            or set(train_ids) & set(test_ids)):
        raise ValueError("Invalid whole-condition TRAIN/TEST groups")
    for name, count in (("train", 32), ("test", 20)):
        if (targets[name].shape != (4, count, 3, 105) or labels[name].shape != (count, 3)
                or not np.isfinite(targets[name]).all() or not np.isfinite(labels[name]).all()):
            raise ValueError("Invalid postselection feature target or evaluation label")
    return torch.tensor(test), targets, labels, test_ids


def _source_hashes():
    import experiments.phi4_qcr as base
    import experiments.phi4_ru_r_validation as vector
    import experiments.phi4_scalar_r_validation as scalar
    import recurrent_rg.phi4_qcr as q_source
    import recurrent_rg.phi4_r_calibration as r_source
    return {str(Path(module.__file__).resolve()): sha256_file(module.__file__)
            for module in (base, vector, scalar, q_source, r_source)} | {
                str(Path(__file__).resolve()): sha256_file(__file__)}


def run(options):
    if (not math.isfinite(options.cpu_budget_seconds) or not 0 < options.cpu_budget_seconds <= 3600
            or not isinstance(options.q_seed, int) or isinstance(options.q_seed, bool) or options.q_seed < 0):
        raise ValueError("Require nonnegative integer Q seed and CPU budget in (0,3600]")
    output, features = (Path(p).resolve() for p in (options.output, options.features))
    if output == features or output in features.parents or features in output.parents:
        raise ValueError("Output must be separate from immutable feature cache")
    output.mkdir(parents=True, exist_ok=False)
    started, cpu_started = time.perf_counter(), time.process_time()
    current, complete, last_size_check = "verify_inputs", False, -1.
    q_records, r_records, models = {}, {}, {}

    def guard(force=False):
        nonlocal last_size_check
        used = time.process_time()-cpu_started
        if used > options.cpu_budget_seconds:
            raise BudgetHold("Fixed-C dimension CPU budget reached")
        if force or used-last_size_check >= 1.:
            last_size_check = used
            if sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) > OUTPUT_BUDGET_BYTES:
                raise BudgetHold("Fixed-C dimension 256 MiB output budget reached")

    def status(state, **extra):
        write_json(output / "status.json", dict(status=state, recipe=RECIPE,
            workflow_complete=state == "complete", optimization_complete=complete, current=current,
            returncode=0 if state == "complete" else 3 if state == "budget_hold" else 1 if state == "failed" else None,
            completed_Q=list(q_records), completed_R=list(r_records),
            cpu_seconds=time.process_time()-cpu_started, wall_seconds=time.perf_counter()-started, **extra))

    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        status("running", error=None, requires_user_action=False)
        feature_manifest, pins = verify_features(features)
        fit, evaluation, ids, split = load_train_only(features)
        codepins = _source_hashes()
        qs = initialize_paired_q(fit, options.q_seed)
        initial_q = {str(d): state_fingerprint(q) for d, q in qs.items()}
        manifest = dict(schema_version=5, recipe=RECIPE, options=asdict(options), theory_dims=[1, 2],
            q_steps=Q_STEPS, q_hidden=96, r_hidden=96, r_seeds=list(R_SEEDS), r_schedule=asdict(Schedule()),
            feature_manifest=feature_manifest, input_sha256=pins, source_sha256=codepins,
            source_C_sha256=feature_manifest["source_module_sha256"]["C"], dtype="float64",
            initial_Q_sha256=initial_q, initialization="fresh paired shared layers, first output coordinate and decoder direction; TRAIN fit-feature mean decoder bias",
            decoder_mean_bias=fit.mean((0, 1, 2, 3)).tolist(),
            mean_bias_scope="equal weight over four cached TRAIN fit repeats, 32 chains, all three scales, both halves",
            Q_objective="10*mean_three_scale_reconstruction+20*subset_k0+5*mean_subset_k1_k2",
            Q_fit_conditions=8, R_fit_conditions=6, R_validation_conditions=2,
            split={k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in split.items()},
            objective="all_start_anchor", closure_scale_weights=None,
            R_objective="5*e01+2.5*e12+2.5*e02 in each learned g coordinate",
            primary_metric="common 105-feature decoded RMSE, with Q reconstruction and identity baselines",
            labels_loaded_before_all_selections=False, TEST_loaded_before_all_selections=False,
            labels_used_by_optimizer=False, source_Q_loaded=False, source_C_loaded=False,
            C_updates=0, no_Q_gradients_from_R=True, CPU_threads=1, GPU_used=False,
            explicit_fixed_C_dimension_ablation=True, exact_online_QCR_reproduction=False,
            new_physical_sampling=False, physical_warning_count=feature_manifest.get("data_warning_count"),
            previously_inspected_TEST_is_exploratory=True, default_recipe_modified=False)
        write_json(output / "manifest.json", manifest)
        # Both Q fits finish before any R fit or TEST/label access.
        for dim, q in qs.items():
            current = f"Q_dg{dim}"
            status("running", error=None, requires_user_action=False)
            q_records[str(dim)] = train_q(q, fit, options.q_seed, guard=guard,
                on_checkpoint=lambda row: write_json(output / "history_progress.json", dict(current=current,
                    completed_Q=list(q_records), completed_R=list(r_records), progress=row)))
            write_json(output / "history.json", dict(Q=q_records, R=r_records))
        if q_records["1"]["batch_stream_sha256"] != q_records["2"]["batch_stream_sha256"]:
            raise RuntimeError("Paired Q dimensions received different batch streams")
        frozen_q = {dim: state_fingerprint(q) for dim, q in qs.items()}
        train_direct, fit_direct, latent_stds = {}, {}, {}
        for dim, q in qs.items():
            fit_direct[dim], train_direct[dim] = encode(q, fit), encode(q, evaluation)
            std = train_direct[dim].reshape(-1, 3, dim).std(0, unbiased=False)
            if not torch.isfinite(std).all() or torch.any(std <= 0):
                raise FloatingPointError("Learned TRAIN latent coordinate has zero or nonfinite scale")
            latent_stds[dim] = std.numpy()
            trainer = train_scalar_r if dim == 1 else train_vector_r
            for seed in R_SEEDS:
                current = f"dg{dim}_seed{seed}"
                status("running", error=None, requires_user_action=False)
                trained, record = trainer(fit_direct[dim][:, split["fit_mask"]],
                    train_direct[dim][:, split["validation_mask"]], seed, guard=guard,
                    on_checkpoint=lambda stage, step, model, row: write_json(output / "history_progress.json",
                        dict(current=current, completed_Q=list(q_records), completed_R=list(r_records), progress=row)))
                r_records[current] = dict(theory_dim=dim, **record)
                models.update({f"{current}_{name}": model for name, model in trained.items()
                               if name in ("selected", "adam6000")})
                write_json(output / "history.json", dict(Q=q_records, R=r_records))
        complete = True
        current = "postselection_TEST_and_common_feature_evaluation"
        status("running", error=None, requires_user_action=False)
        test_features, targets, labels, test_ids = load_postselection(features, ids)
        for name, halves in (("train", evaluation), ("test", test_features)):
            if not np.allclose(halves.mean(-2).numpy(), targets[name], rtol=0., atol=1e-12):
                raise ValueError("Common feature target differs from exact paired-half average")
        arrays = dict(train_cell_ids=ids, test_cell_ids=test_ids, train_labels=labels["train"],
                      test_labels=labels["test"], train_feature_target=targets["train"], test_feature_target=targets["test"])
        direct = {}
        for dim, q in qs.items():
            direct[dim] = dict(train=train_direct[dim], test=encode(q, test_features))
            for name, value in direct[dim].items():
                arrays[f"dg{dim}_{name}_direct"] = value.numpy()
                arrays[f"dg{dim}_{name}_Q_reconstruction"] = q.decode(value).numpy()
        feature_metrics, latent_metrics = {}, {}
        fingerprints = {key: state_fingerprint(model) for key, model in models.items()}
        for key, model in models.items():
            guard()
            dim = int(key[2])
            q = qs[dim]
            feature_metrics[key], latent_metrics[key] = {}, {}
            for name in ("train", "fit", "validation", "test"):
                mask = split[f"{name}_mask"] if name in ("fit", "validation") else slice(None)
                base = "test" if name == "test" else "train"
                g = direct[dim][base][:, mask]
                target = targets[base][:, mask]
                prediction = predict_r(model, g)
                f_metric, g_metric = endpoint_metrics(q, g, prediction, target, latent_stds[dim])
                feature_metrics[key][name], latent_metrics[key][name] = f_metric, g_metric
                if name in ("train", "test"):
                    arrays.update({f"eval_{key}_{name}_{kind}": value for kind, value in prediction.items()})
        np.savez_compressed(output / "evaluation_arrays.npz", **arrays)
        np.savez_compressed(output / "Q_parameters.npz", **{
            f"dg{dim}.{name}": value.detach().numpy() for dim, q in qs.items() for name, value in q.state_dict().items()})
        np.savez_compressed(output / "R_parameters.npz", **{
            f"{key}.{name}": value.detach().numpy() for key, model in models.items() for name, value in model.state_dict().items()})
        if any(state_fingerprint(q) != frozen_q[dim] for dim, q in qs.items()):
            raise RuntimeError("R training or evaluation changed frozen Q")
        if any(state_fingerprint(model) != fingerprints[key] for key, model in models.items()):
            raise RuntimeError("Frozen endpoint evaluation changed R")
        if any(p.grad is not None for m in (*qs.values(), *models.values()) for p in m.parameters()):
            raise RuntimeError("Frozen evaluation left Q/R gradients")
        if any(sha256_file(path) != sha for path, sha in (pins | codepins).items()):
            raise RuntimeError("Pinned feature or code input changed")
        guard(True)
        result = dict(status="complete", workflow_complete=True, returncode=0, recipe=RECIPE,
            Q_records=q_records, R_records=r_records, feature_metrics=feature_metrics, latent_metrics=latent_metrics,
            final_Q_sha256={str(d): sha for d, sha in frozen_q.items()}, final_R_sha256=fingerprints,
            latent_TRAIN_coordinate_stds={str(d): value.tolist() for d, value in latent_stds.items()},
            all_Q_fits_and_R_selections_before_TEST_and_labels=True, common_feature_targets_identical=True,
            Q_frozen_during_R=True, C_updates=0, GPU_seconds=0, CPU_threads=1,
            cpu_seconds=time.process_time()-cpu_started, wall_seconds=time.perf_counter()-started,
            input_and_code_sha256_unchanged=True, thermal_nu=None, critical_r=None,
            physical_fixed_point_certified=False, root_search=False,
            limitations=["Frozen-C finite-cache dimension ablation; not an exact fresh joint online QCR reproduction",
                "Q sees all eight TRAIN conditions, including the two used for R-only validation",
                "Both dimensions share finite physical data and C; R seeds are optimizer repeats only",
                "Latent-coordinate NRMSE is not comparable across dimensions; use common feature metrics",
                "Four repeated encodings reuse physical chains; all source quality warnings remain",
                "TEST was previously inspected and is exploratory; no TEST-based model selection"])
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
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--q-seed", type=int, default=91)
    parser.add_argument("--cpu-budget-seconds", type=float, default=1200.)
    run(Options(**vars(parser.parse_args(argv))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
