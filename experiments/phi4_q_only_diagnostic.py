"""Paired, frozen-C diagnostics of the continuous-field Q objective.

This deliberately separate entry point is not a default QCR recipe. It reads
statistics collected with one immutable completed C, optimizes only copies of
Q on CPU, and never evaluates R, MI, or the physical sampler. Coupling metadata
is copied to the evaluation archive only after optimization.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from torch import nn

from experiments.phi4_qcr import sha256_file, state_fingerprint, write_json
from recurrent_rg.phi4_qcr import FEATURE_DIM, ContinuousTheoryEncoder, Phi4QCR


RECIPE = "phi4_frozenC_paired_Q_only_five_arm_v1"
ARM_NAMES = ("A", "B", "C", "D", "E")
ARRAY_KEYS = ("train_per_sample", "eval_train_first", "eval_train_second",
              "eval_test_first", "eval_test_second")
EVALUATION_METADATA = tuple(f"eval_{split}_{name}" for split in ("train", "test")
                            for name in ("labels", "cell_ids", "chain_ids"))


class BudgetHold(RuntimeError):
    """The fixed diagnostic cannot continue within its resource budget."""


@dataclass
class Options:
    source_run: str
    cache: str
    output: str
    steps: int = 3000
    seed: int = 91
    system_batch: int = 8
    sample_batch: int = 64
    log_every: int = 100
    cpu_budget_seconds: float = 600.

    def validate(self):
        for name in ("steps", "system_batch", "sample_batch", "log_every"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.sample_batch % 2:
            raise ValueError("sample_batch must have equally sized halves")
        if not math.isfinite(self.cpu_budget_seconds) or self.cpu_budget_seconds <= 0:
            raise ValueError("cpu_budget_seconds must be finite and positive")


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_cache(cache):
    """Load only the explicit statistic members, never metadata into training."""
    root = Path(cache).resolve()
    path = root / "arrays.npz"
    receipt = _json(root / "receipt.json")
    expected = receipt.get("arrays_sha256")
    if not expected or sha256_file(path) != expected:
        raise ValueError("Frozen-C statistics archive SHA256 mismatch")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], dtype=np.float64, copy=True) for key in ARRAY_KEYS}
    train = arrays["train_per_sample"]
    if (train.ndim != 5 or train.shape[0] != 2 or train.shape[2] != 3
            or train.shape[-1] != FEATURE_DIM or min(train.shape) < 1):
        raise ValueError("train_per_sample must be [2,TRAIN,3,samples,105]")
    for split in ("train", "test"):
        first, second = (arrays[f"eval_{split}_{name}"] for name in ("first", "second"))
        if (first.ndim != 3 or first.shape != second.shape or first.shape[1:] != (3, FEATURE_DIM)
                or not len(first)):
            raise ValueError(f"Invalid {split} fixed evaluation half statistics")
    if len(arrays["eval_train_first"]) != train.shape[1]:
        raise ValueError("TRAIN evaluation and optimization physical-ensemble counts differ")
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise FloatingPointError("Non-finite frozen-C statistics")
    return {key: torch.from_numpy(value) for key, value in arrays.items()}, receipt


def load_source(source_run, receipt):
    root = Path(source_run).resolve()
    manifest, summary = _json(root / "manifest.json"), _json(root / "summary.json")
    if summary.get("status") != "complete":
        raise ValueError("The source model must be from a completed immutable run")
    if int(manifest.get("theory_dim", 0)) != 1:
        raise ValueError("This paired diagnostic is explicitly a d_g=1 ablation")
    latest = int(summary.get("latest_checkpoint_step", summary["actual_steps"]))
    checkpoint_path = root / f"training_step_{latest}.pt"
    checkpoint_sha = sha256_file(checkpoint_path)
    if checkpoint_sha != receipt.get("source_checkpoint_sha256"):
        raise ValueError("Source checkpoint differs from frozen-C receipt")
    if receipt.get("source_run") and Path(receipt["source_run"]).resolve() != root:
        raise ValueError("Statistics belong to a different source run")
    if receipt.get("source_manifest_sha256") != sha256_file(root / "manifest.json"):
        raise ValueError("Source manifest differs from frozen-C receipt")
    scale = float(manifest["scale"])
    if scale != float(receipt["scale"]):
        raise ValueError("Frozen-C feature scale differs from source")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = Phi4QCR(scale, hidden=int(manifest.get("hidden", 96)), theory_dim=1)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.requires_grad_(False).eval()
    fingerprint = state_fingerprint(model)
    if fingerprint != summary["final_state_sha256"] or fingerprint != receipt["source_state_sha256"]:
        raise ValueError("Source model state fingerprint mismatch")
    return model, manifest, summary, checkpoint_path, checkpoint_sha


def make_arms(source, pooled_train_mean, seed=91):
    """Matched initialization: C/D/E differ from B in named factors only."""
    if pooled_train_mean.shape != (FEATURE_DIM,):
        raise ValueError("The pooled TRAIN feature mean must have 105 coordinates")
    hidden = source.C.net[0].out_features
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        # Preserve the original C-before-Q construction RNG order exactly.
        fresh = Phi4QCR(float(source.scale), hidden=hidden, theory_dim=1)
        original = copy.deepcopy(fresh.Q)
        centered = copy.deepcopy(original)
        with torch.no_grad():
            centered.decoder.bias.copy_(pooled_train_mean)
        linear = copy.deepcopy(centered)
        torch.manual_seed(seed)
        linear.net = nn.Linear(FEATURE_DIM, 1).double()
    arms = {"A": copy.deepcopy(source.Q), "B": original, "C": centered,
            "D": linear, "E": copy.deepcopy(centered)}
    for encoder in arms.values():
        encoder.requires_grad_(True).train()
    return arms


def q_objective(encoder, first, second, reconstruction_only=False):
    """The original full Q loss, without C/R or coupling-label arguments."""
    first, second = first.detach(), second.detach()
    if first.shape != second.shape or first.ndim != 3 or first.shape[1:] != (3, FEATURE_DIM):
        raise ValueError("Q half features must have shape [physical ensembles,3,105]")
    ga, gb = encoder.from_features(first), encoder.from_features(second)
    g = .5 * (ga + gb)
    target = .5 * (first + second)
    reconstruction_by_scale = (encoder.decode(g) - target).square().sum(-1).mean(0)
    subset_by_scale = (ga - gb).square().sum(-1).mean(0)
    reconstruction = reconstruction_by_scale.mean()
    full = 10. * reconstruction + 20. * subset_by_scale[0] + 5. * subset_by_scale[1:].mean()
    optimized = 10. * reconstruction if reconstruction_only else full
    return optimized, dict(q_full_objective=full, optimization_objective=optimized,
        reconstruction_l2=reconstruction, reconstruction_l2_by_scale=reconstruction_by_scale,
        subset_l2_by_scale=subset_by_scale)


def schedule(train, options):
    generator = torch.Generator(device="cpu").manual_seed(options.seed + 101)
    draws, systems, samples = [], [], []
    for _ in range(options.steps):
        draws.append(torch.randint(train.shape[0], (), generator=generator).item())
        systems.append(torch.randperm(train.shape[1], generator=generator)[:options.system_batch])
        samples.append(torch.randperm(train.shape[3], generator=generator)[:options.sample_batch])
    return draws, torch.stack(systems), torch.stack(samples)


def statistic_halves(train, draw, systems, samples):
    batch = train[draw, systems][:, :, samples]
    half = len(samples) // 2
    return batch[:, :, :half].mean(2), batch[:, :, half:].mean(2)


def _scalar_metrics(terms):
    return {key: value.detach().cpu().tolist() for key, value in terms.items()}


def _optimizer_counts(optimizer, expected):
    counts = [int(value["step"].item()) for value in optimizer.state.values()]
    if not counts or min(counts) != expected or max(counts) != expected:
        raise RuntimeError("Every Adam parameter must receive the fixed number of updates")
    return dict(min=min(counts), max=max(counts), parameters=len(counts))


def _metric_arrays(first, second, g_first, g_second, reconstruction, train_std,
                   reconstruction_only=False):
    target = .5 * (first + second)
    l2 = np.mean(np.sum((reconstruction - target) ** 2, axis=-1), axis=0)
    rmse = np.sqrt(l2 / FEATURE_DIM)
    subset = np.mean(np.sum((g_first - g_second) ** 2, axis=-1), axis=0)
    full = float(10 * l2.mean() + 20 * subset[0] + 5 * subset[1:].mean())
    return dict(reconstruction_rmse_by_scale=rmse.tolist(),
        reconstruction_nrmse_by_scale=[float(a / b) if b > 0 else None for a, b in zip(rmse, train_std)],
        reconstruction_l2=float(l2.mean()), reconstruction_l2_by_scale=l2.tolist(),
        subset_l2_by_scale=subset.tolist(), q_full_objective=full,
        optimization_objective=float(10 * l2.mean()) if reconstruction_only else full)


def reconstruction_references(features):
    target = .5 * (features["eval_train_first"].numpy() + features["eval_train_second"].numpy())
    flattened = target.reshape(-1, FEATURE_DIM)
    mean = flattened.mean(0)
    centered = flattened - mean
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    return mean, vh, singular, np.sqrt(np.mean((target - target.mean(0, keepdims=True)) ** 2, axis=(0, 2)))


@torch.no_grad()
def evaluate_encoder(encoder, features, train_std, reconstruction_only=False):
    metrics, arrays = {}, {}
    for split in ("train", "test"):
        first, second = (features[f"eval_{split}_{key}"] for key in ("first", "second"))
        ga, gb = encoder.from_features(first), encoder.from_features(second)
        g = .5 * (ga + gb)
        target = .5 * (first + second)
        decoded = encoder.decode(g)
        weight = encoder.normalized_decoder_weight()
        oracle = (target - encoder.decoder.bias) @ weight
        oracle_decoded = encoder.decode(oracle)
        arrays.update({f"{split}_direct": g.numpy(),
            f"{split}_direct_full": encoder.from_features(target).numpy(),
            f"{split}_reconstruction": decoded.numpy(),
            f"{split}_oracle_reconstruction": oracle_decoded.numpy()})
        metrics[split] = _metric_arrays(first.numpy(), second.numpy(), ga.numpy(), gb.numpy(),
            decoded.numpy(), train_std, reconstruction_only)
    return metrics, arrays


def evaluate_references(features, mean, components, train_std):
    summaries, arrays = {}, {}
    for name, dimension in (("constant", 0), ("PCA1", 1), ("PCA3", 3)):
        axes = components[:dimension].T
        metrics = {}
        for split in ("train", "test"):
            first, second = (features[f"eval_{split}_{key}"].numpy() for key in ("first", "second"))
            ga, gb = (first - mean) @ axes, (second - mean) @ axes
            direct = .5 * (ga + gb)
            reconstruction = mean + direct @ axes.T
            metrics[split] = _metric_arrays(first, second, ga, gb, reconstruction, train_std)
            arrays[f"{name}_{split}_direct"] = direct
            arrays[f"{name}_{split}_reconstruction"] = reconstruction
        summaries[name] = dict(metrics=metrics, reconstruction_only=dimension == 3,
            qualification="TRAIN-only analytic reconstruction reference, not initialization or learned RG")
    return summaries, arrays


def encoder_diagnostic(encoder, arrays, train_mean, components, features):
    weight = encoder.normalized_decoder_weight().detach().numpy()[:, 0]
    train_target = .5 * (features["eval_train_first"].numpy() + features["eval_train_second"].numpy())
    centered = train_target.reshape(-1, FEATURE_DIM) - train_mean
    total_variation = float(np.sum(centered ** 2))
    diagnostic = dict(decoder_alignment_abs_pc1=float(abs(weight @ components[0])),
        decoder_column_norms=np.linalg.norm(encoder.decoder.weight.detach().numpy(), axis=0).tolist(),
        decoder_explained_fraction=float(np.sum((centered @ weight) ** 2) / total_variation)
            if total_variation else None,
        oracle_rmse_by_scale={}, latent_stats={}, jensen_rms_by_scale={},
        encoder_correlation_with_TRAIN_PC1={})
    for split in ("train", "test"):
        target = .5 * (features[f"eval_{split}_first"].numpy() + features[f"eval_{split}_second"].numpy())
        oracle = arrays[f"{split}_oracle_reconstruction"]
        g = arrays[f"{split}_direct"]
        diagnostic["oracle_rmse_by_scale"][split] = np.sqrt(np.mean((oracle - target) ** 2, axis=(0, 2))).tolist()
        diagnostic["latent_stats"][split] = dict(mean_by_scale=g.mean(0).tolist(),
            std_by_scale=g.std(0, ddof=0).tolist(), minimum_by_scale=g.min(0).tolist(),
            maximum_by_scale=g.max(0).tolist())
        diagnostic["jensen_rms_by_scale"][split] = np.sqrt(np.mean(
            (arrays[f"{split}_direct_full"] - g) ** 2, axis=(0, 2))).tolist()
        projected = (target - train_mean) @ components[0]
        correlation = []
        for level in range(3):
            x, y = projected[:, level], g[:, level, 0]
            correlation.append(float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else None)
        diagnostic["encoder_correlation_with_TRAIN_PC1"][split] = correlation
    return diagnostic


def train(options):
    options.validate()
    root = Path(options.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    started, cpu_started = time.perf_counter(), time.process_time()
    steps = {name: 0 for name in ARM_NAMES}
    source = None
    history = []

    def status(value, **extra):
        write_json(root / "status.json", dict(status=value, workflow_complete=value == "complete",
            actual_steps_by_arm=steps.copy(), returncode=0 if value == "complete" else None,
            error=None, requires_user_action=value == "budget_hold", recipe=RECIPE, **extra))

    def budget():
        used = time.process_time() - cpu_started
        if used > options.cpu_budget_seconds:
            raise BudgetHold(f"CPU budget exceeded: {used:.6f} > {options.cpu_budget_seconds:.6f} seconds")

    status("initializing")
    try:
        torch.set_num_threads(1)
        features, receipt = load_cache(options.cache)
        budget()
        source, manifest, source_summary, checkpoint_path, checkpoint_sha = load_source(options.source_run, receipt)
        source_before = state_fingerprint(source)
        frozen_before = {key: state_fingerprint(getattr(source, key)) for key in ("Q", "C", "R")}
        if options.system_batch > features["train_per_sample"].shape[1]:
            raise ValueError("system_batch exceeds TRAIN physical-ensemble count")
        if options.sample_batch > features["train_per_sample"].shape[3]:
            raise ValueError("sample_batch exceeds saved samples per physical ensemble")
        pooled_mean = features["train_per_sample"].mean((0, 1, 2, 3))
        arms = make_arms(source, pooled_mean, options.seed)
        initial = {name: state_fingerprint(encoder) for name, encoder in arms.items()}
        optimizers = {name: torch.optim.Adam(encoder.parameters(), lr=.001) for name, encoder in arms.items()}
        batches = schedule(features["train_per_sample"], options)
        schedule_hash = hashlib.sha256()
        schedule_hash.update(np.asarray(batches[0], dtype=np.int64).tobytes())
        for indices in batches[1:]:
            schedule_hash.update(indices.numpy().tobytes())
        train_mean, components, singular, train_std = reconstruction_references(features)
        run_manifest = dict(recipe=RECIPE, schema_version=5, options=asdict(options),
            source_run=str(Path(options.source_run).resolve()), source_checkpoint=str(checkpoint_path),
            source_checkpoint_sha256=checkpoint_sha, source_state_sha256=source_before,
            cache_receipt=receipt, cache_receipt_sha256=sha256_file(Path(options.cache) / "receipt.json"),
            source_manifest_sha256=sha256_file(Path(options.source_run) / "manifest.json"),
            source_summary_sha256=sha256_file(Path(options.source_run) / "summary.json"),
            source_code_sha256={str(Path(__file__).resolve()): sha256_file(__file__)},
            arms={"A": "source Q warm start, fresh Adam; original full Q objective",
                  "B": "fresh original architecture and objective; seed91 C-before-Q RNG construction",
                  "C": "B with decoder bias set only to pooled TRAIN feature mean",
                  "D": "C decoder and bias; seed91 random linear encoder replaces nonlinear encoder",
                  "E": "C initialization; optimize reconstruction only, evaluate original full objective"},
            initial_Q_sha256=initial, paired_schedule_sha256=schedule_hash.hexdigest(),
            actual_source_steps=source_summary["actual_steps"], additional_updates_per_arm=options.steps,
            optimizer="independent fresh Adam", learning_rate=.001, device="cpu", threads=1,
            C_updates=0, R_updates=0, source_Q_updates=0, physical_sampling_calls=0, MI_calls=0,
            labels_used_in_training=False, optimizer_input_members=list(ARRAY_KEYS),
            full_Q_objective="10*mean(scale reconstruction squared L2)+20*micro subset squared L2+5*mean(coarse subset squared L2)",
            NRMSE_definition="RMSE divided by same-scale original TRAIN target per-feature population RMS std",
            PCA_reference="TRAIN fixed evaluation target only; analytic reconstruction diagnostic, never initialization",
            training_C_draws="two fixed draws, not independent physical ensembles",
            equilibrium_certified=False, thermal_nu=None, computes_R=False,
            qualification="Isolated frozen-C Q-only diagnostic; changed Q coordinates cannot be used with the old R")
        write_json(root / "manifest.json", run_manifest)
        budget()

        def record(step, batch_metrics=None):
            row = dict(step=step, elapsed_seconds=time.perf_counter() - started, arms={})
            for name, encoder in arms.items():
                metric, _ = evaluate_encoder(encoder, features, train_std, name == "E")
                row["arms"][name] = dict(**metric, batch_metrics=(batch_metrics or {}).get(name))
            history.append(row)
            write_json(root / "history_progress.json", dict(recipe=RECIPE, latest_step=step,
                actual_steps_by_arm=steps.copy(), history=history))
            print(json.dumps(dict(event="phi4_Q_only_progress", **row), allow_nan=False), flush=True)
            budget()

        record(0)
        status("training")
        for update in range(1, options.steps + 1):
            budget()
            first, second = statistic_halves(features["train_per_sample"], batches[0][update - 1],
                batches[1][update - 1], batches[2][update - 1])
            batch_metrics = {}
            for name, encoder in arms.items():
                budget()
                optimizer = optimizers[name]
                optimizer.zero_grad(set_to_none=True)
                loss, terms = q_objective(encoder, first, second, reconstruction_only=name == "E")
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite Q objective in arm {name}")
                loss.backward()
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in encoder.parameters()):
                    raise FloatingPointError(f"Missing/non-finite Q gradient in arm {name}")
                optimizer.step()
                encoder.normalize_decoder_()
                steps[name] = update
                batch_metrics[name] = _scalar_metrics(terms)
            if update == 1 or update % options.log_every == 0 or update == options.steps:
                record(update, batch_metrics)
                status("training", cpu_seconds=time.process_time() - cpu_started)
        optimizer_counts = {name: _optimizer_counts(optimizers[name], options.steps) for name in ARM_NAMES}
        for encoder in arms.values():
            for parameter in encoder.parameters():
                parameter.grad = None
        status("evaluating")
        summaries, arrays = {}, {}
        for name, encoder in {"source": source.Q, **arms}.items():
            budget()
            metrics, encoder_arrays = evaluate_encoder(encoder, features, train_std, name == "E")
            summaries[name] = dict(metrics=metrics, actual_steps=steps.get(name, 0),
                initial_Q_sha256=frozen_before["Q"] if name == "source" else initial[name],
                final_Q_sha256=state_fingerprint(encoder),
                diagnostic=encoder_diagnostic(encoder, encoder_arrays, train_mean, components, features))
            arrays.update({f"{name}_{key}": value for key, value in encoder_arrays.items()})
        references, reference_arrays = evaluate_references(features, train_mean, components, train_std)
        arrays.update(reference_arrays)
        arrays["train_std_by_scale"] = train_std
        arrays["TRAIN_reference_mean"] = train_mean
        arrays["TRAIN_reference_components"] = components
        arrays["TRAIN_reference_singular_values"] = singular
        for split in ("train", "test"):
            arrays[f"{split}_target"] = .5 * (features[f"eval_{split}_first"].numpy() +
                features[f"eval_{split}_second"].numpy())
        # Labels enter only this post-optimization, evaluation-only copy stage.
        with np.load(Path(options.cache) / "arrays.npz", allow_pickle=False) as archive:
            for key in EVALUATION_METADATA:
                if key in archive.files:
                    arrays[key] = np.array(archive[key], copy=True)
        for split in ("train", "test"):
            key = f"eval_{split}_cell_ids"
            if key not in arrays:
                continue
            cell_ids = arrays[key]
            for name in summaries:
                g = arrays[f"{name}_{split}_direct"]
                summaries[name].setdefault("per_condition", {})[split] = [
                    dict(cell_id=int(cell), physical_ensemble_count=int(np.sum(cell_ids == cell)),
                        mean_g_by_scale=g[cell_ids == cell].mean(0).tolist(),
                        variance_g_by_scale=g[cell_ids == cell].var(0, ddof=0).tolist())
                    for cell in np.unique(cell_ids)]
        if state_fingerprint(source) != source_before or any(p.grad is not None for p in source.parameters()):
            raise RuntimeError("Immutable source state or gradients changed")
        frozen_after = {key: state_fingerprint(getattr(source, key)) for key in ("Q", "C", "R")}
        if frozen_after != frozen_before:
            raise RuntimeError("Immutable source Q/C/R fingerprint changed")
        if sha256_file(checkpoint_path) != checkpoint_sha:
            raise RuntimeError("Source checkpoint file changed during diagnostic")
        for name in ("manifest", "summary"):
            if sha256_file(Path(options.source_run) / f"{name}.json") != run_manifest[f"source_{name}_sha256"]:
                raise RuntimeError(f"Immutable source {name} changed during diagnostic")
        if (sha256_file(Path(options.cache) / "arrays.npz") != receipt["arrays_sha256"]
                or sha256_file(Path(options.cache) / "receipt.json") != run_manifest["cache_receipt_sha256"]):
            raise RuntimeError("Frozen-C statistics cache changed during diagnostic")
        budget()
        for name, encoder in arms.items():
            arm_root = root / name
            arm_root.mkdir()
            torch.save(dict(recipe=RECIPE, step=steps[name], Q=encoder.state_dict(),
                optimizer=optimizers[name].state_dict(), initial_Q_sha256=initial[name],
                source_checkpoint_sha256=checkpoint_sha),
                arm_root / f"Q_step_{steps[name]}.pt")
        np.savez(root / "evaluation_arrays.npz", **arrays)
        summary = dict(recipe=RECIPE, schema_version=5, status="complete", actual_steps=options.steps,
            actual_steps_by_arm=steps,
            arms=summaries, baselines=references, optimizer_steps=optimizer_counts,
            arm_optimizer_steps=optimizer_counts,
            frozen_source_before=frozen_before, frozen_source_after=frozen_after,
            frozen_source_verified=True, source_unchanged=True, source_gradients_none=True,
            source_pins=dict(checkpoint_sha256=checkpoint_sha, state_sha256=source_before,
                manifest_sha256=run_manifest["source_manifest_sha256"],
                summary_sha256=run_manifest["source_summary_sha256"],
                cache_arrays_sha256=receipt["arrays_sha256"]),
            source_quality={key: source_summary.get(key) for key in
                ("equilibrium_certified", "data_warning_count", "sampler_schema", "sampler_qualification")},
            labels_used_in_training=False, C_updates=0, R_updates=0, source_Q_updates=0,
            source_parameters_updated=dict(Q=0, C=0, R=0),
            train_C_draws=2, training_C_draws=2,
            train_physical_ensembles=features["train_per_sample"].shape[1],
            test_physical_ensembles=len(features["eval_test_first"]),
            timing=dict(wall_seconds=time.perf_counter() - started, cpu_seconds=time.process_time() - cpu_started),
            cpu_budget_seconds=options.cpu_budget_seconds, thermal_nu=None, critical_r=None,
            computes_R=False, equilibrium_certified=False, qualification=run_manifest["qualification"])
        variance = singular ** 2
        summary["pooled_TRAIN_PC_variance_fraction"] = (variance / variance.sum()).tolist() if variance.sum() else None
        summary["pooled_TRAIN_PC_cumulative_fraction"] = (np.cumsum(variance) / variance.sum()).tolist() if variance.sum() else None
        write_json(root / "summary.json", summary)
        budget()
        status("complete", summary_sha256=sha256_file(root / "summary.json"),
            evaluation_arrays_sha256=sha256_file(root / "evaluation_arrays.npz"),
            history_sha256=sha256_file(root / "history_progress.json"),
            history_progress_sha256=sha256_file(root / "history_progress.json"),
            cpu_seconds=time.process_time() - cpu_started)
        return summary
    except Exception as error:
        hold = isinstance(error, BudgetHold)
        write_json(root / "status.json", dict(recipe=RECIPE, status="budget_hold" if hold else "failed",
            workflow_complete=False, actual_steps_by_arm=steps, returncode=None if hold else 1,
            error=f"{type(error).__name__}: {error}", requires_user_action=True,
            cpu_seconds=time.process_time() - cpu_started, traceback=traceback.format_exc(),
            source_gradients_none=source is None or all(p.grad is None for p in source.parameters())))
        if hold:
            return dict(status="budget_hold", actual_steps_by_arm=steps)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-run", "cache", "output"):
        parser.add_argument(f"--{name}", required=True)
    for name, default in (("steps", 3000), ("seed", 91), ("system-batch", 8),
                          ("sample-batch", 64), ("log-every", 100)):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--cpu-budget-seconds", type=float, default=600.)
    options = Options(**vars(parser.parse_args()))
    result = train(options)
    if result["status"] == "budget_hold":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
