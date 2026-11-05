"""Fresh two-coordinate QCR experiment on an unlabeled r-by-u phi4 grid.

The explicitly authorized continuous-field recipe is reused without changes;
this entry point changes only the grid dimensionality and theory_dim=2.  A
discarded seed-191 profile never seeds or resumes the formal seed-91 run.
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

from experiments import phi4_qcr as original
from recurrent_rg.phi4_qcr import Phi4QCR
from recurrent_rg.phi4_ru_flow import bounded_flow


RECIPE = original.RECIPE
EXPERIMENT_SCHEMA = "phi4_ru_qcr_dg2_v1"


@dataclass
class TrainingOptions(original.TrainingOptions):
    profile_only: bool = False
    profile_steps: int = 20

    def validate(self):
        super().validate()
        if self.q_initialization != "train_mean_bias":
            raise ValueError("This fresh ru experiment pins TRAIN mean-bias initialization")
        if (isinstance(self.profile_steps, bool) or not isinstance(self.profile_steps, int)
                or self.profile_steps < 1):
            raise ValueError("profile_steps must be a positive integer")
        if self.profile_only and self.seed != 191:
            raise ValueError("The discarded profile uses its independent seed 191")


def _sources():
    import recurrent_rg.phi4_qcr as model_source
    import recurrent_rg.phi4_ru_flow as flow_source
    return {str(Path(module.__file__).resolve()): original.sha256_file(module.__file__)
            for module in (original, model_source, flow_source)} | {
                str(Path(__file__).resolve()): original.sha256_file(__file__)}


def _manifest(options, dataset, device_receipt, initialization, sources):
    return dict(schema_version=5, experiment_schema=EXPERIMENT_SCHEMA, recipe=RECIPE,
        options=asdict(options), data_root=str(Path(options.data).resolve()),
        data_manifest_sha256=dataset.manifest_sha256, data_shards=dataset.receipts,
        source_sha256=sources, theory_dim=2, hidden=96,
        lattice_sizes=[int(dataset.configurations.shape[2] // 3 ** k) for k in range(3)],
        block_size=3, stride=3, dtype="float64", scale=dataset.scale,
        scale_definition="sqrt(mean(TRAIN microscopic field**2)); frozen label-free scalar",
        initialization="random trainable Gaussian C/Q encoder/decoder directions/R; TRAIN mean Q decoder bias only",
        q_initialization=options.q_initialization, q_initialization_receipt=initialization,
        random_construction_state_sha256=initialization["constructor_state_sha256"],
        random_construction_module_sha256=initialization["constructor_module_sha256"],
        training_start_state_sha256=initialization["training_start_state_sha256"],
        training_start_module_sha256=initialization["training_start_module_sha256"],
        initial_state_sha256=initialization["training_start_state_sha256"],
        initial_state_sha256_semantics="post TRAIN-only decoder-bias calibration, before fresh Adam",
        learning_rate=.001, learning_rate_policy="constant", optimizer="three independent fresh Adam",
        closure_objective="all_start_anchor", closure_scale_weights=None, closure_target_scales=2,
        closure_implementation="5*mean(one_step)+5*mean(microscopic_rollout); S=2 exact all-start equivalence",
        closure_branch_normalization="uniform target-scale average; mean over earlier starts per target",
        closure_effective_weights=dict(e01=5., e12=2.5, e02=2.5),
        continuous_recipe=dict(C="-2*full-shell smoothed information proxy + 2*variance floor + uniform amplitude MSE",
            C_noise="Gaussian, standard deviation 0.1 times frozen TRAIN microscopic RMS",
            Q="10*shared affine 105-soft-motif reconstruction + 20*microscopic subset + 5*coarse subset",
            Q_decoder_columns="unit norm; full rank not forced", future_loss=False,
            prediction_prediction_penalty=False, pca_initialization=False,
            discrete_histogram_MI_or_color_NLL=False),
        gradient_routing="simultaneous independent Q-only/C-only/R-only gradients; all gradients before all optimizer steps",
        labels_used_in_training=False, labels_generation_and_evaluation_only=True,
        model_input_members=["configurations"], evaluation_label_columns=["r", "u", "h"],
        evaluation_seed=original.EVALUATION_SEED,
        evaluation_encoding="mean of Q(first-half features) and Q(second-half features); direct_full=Q(pooled features) kept separately",
        evaluation_sample_count=int(dataset.configurations.shape[1]),
        training_sample_count=options.sample_batch,
        training_encoding="mean of two half-ensemble Q encodings; full-feature reconstruction",
        data_warning_count=dataset.warning_count, equilibrium_certified=False,
        sampler_schema=dataset.sampler_schema, sample_semantics=dataset.sample_semantics,
        sampler_qualification=dataset.qualification, device=device_receipt,
        peak_cuda_bytes_semantics="max of own allocated and reserved high-water marks",
        max_training_seconds_semantics="device/model/mean-bias/optimizer initialization + all updates + frozen final evaluation and bounded flow",
        flow_plan=dict(grid_side=33, bound_padding=.1, bounds_hull_PCA="TRAIN scales 0,1 only",
            maximum_starts=49, max_nfev_per_start=64, full_DR_includes_identity=True,
            root_policy="all accepted distinct bounded numerical roots; no eigenmode selection"),
        profile_only=options.profile_only, profile_seed=191 if options.profile_only else None,
        profile_endpoint_discarded=options.profile_only, thermal_nu=None)


def run_training(options):
    options.validate()
    output = Path(options.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started, cpu_started = time.perf_counter(), time.process_time()
    step = latest_checkpoint = 0
    original.write_json(output / "status.json", dict(status="initializing", workflow_complete=False,
        actual_steps=0, returncode=None, error=None, requires_user_action=False, recipe=RECIPE,
        experiment_schema=EXPERIMENT_SCHEMA, profile_only=options.profile_only))
    try:
        torch.set_num_threads(options.threads)
        torch.manual_seed(options.seed)
        dataset = original.load_dataset(options.data)
        if options.system_batch > len(dataset.train_indices) or options.sample_batch > dataset.configurations.shape[1]:
            raise ValueError("Requested batch exceeds available TRAIN fields")
        device = torch.device(options.device)
        compute_started = time.perf_counter()
        receipt = original._device_receipt(device, options.expected_gpu_uuid)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        check = lambda: original._check_budget(options, device, compute_started)
        model = Phi4QCR(dataset.scale, hidden=96, theory_dim=2).to(device=device, dtype=torch.float64)
        random_sha = original.state_fingerprint(model)
        if (options.expected_random_construction_sha256 is not None
                and random_sha != options.expected_random_construction_sha256):
            raise RuntimeError("Random construction SHA256 mismatch before calibration or updates")
        initialization = original.initialize_q_decoder(model, dataset.configurations,
            dataset.train_indices, seed=options.seed, mode=options.q_initialization, budget_check=check)
        sources = _sources()
        manifest = _manifest(options, dataset, receipt, initialization, sources)
        original.write_json(output / "manifest.json", manifest)
        optimizers = {name: torch.optim.Adam(getattr(model, name).parameters(), lr=.001)
                      for name in ("Q", "C", "R")}
        batch_rng = torch.Generator(device="cpu").manual_seed(options.seed + 101)
        field_rng = torch.Generator(device=device).manual_seed(options.seed + 202)
        train_indices = torch.as_tensor(dataset.train_indices, dtype=torch.long)
        training_started = time.perf_counter()
        initialization_seconds = training_started - compute_started
        history, update_seconds = [], []
        final_step = options.profile_steps if options.profile_only else options.steps
        model.train()
        for update in range(1, final_step + 1):
            check()
            update_started = time.perf_counter()
            systems = train_indices[torch.randperm(len(train_indices), generator=batch_rng)[:options.system_batch]]
            samples = torch.randperm(dataset.configurations.shape[1], generator=batch_rng)[:options.sample_batch]
            batch = dataset.configurations[systems][:, samples].to(device=device, dtype=torch.float64)
            metrics = original.independent_step(model, batch, optimizers,
                generator=field_rng, information_pairs=options.information_pairs)
            step = update
            peak = check()
            update_seconds.append(time.perf_counter() - update_started)
            logged = step == 1 or step % options.log_every == 0 or step == final_step
            if logged:
                history.append(dict(step=step, **metrics,
                                    elapsed_seconds=time.perf_counter() - training_started))
                print(json.dumps(dict(event="phi4_ru_qcr_progress", **history[-1]), allow_nan=False), flush=True)
            if not options.profile_only and (step % options.checkpoint_every == 0 or step == final_step):
                torch.save(dict(recipe=RECIPE, experiment_schema=EXPERIMENT_SCHEMA, step=step,
                    model=model.state_dict(), optimizers={name: value.state_dict() for name, value in optimizers.items()},
                    batch_rng=batch_rng.get_state(), field_rng=field_rng.get_state(), manifest=manifest),
                    output / f"training_step_{step}.pt")
                latest_checkpoint = step
            if logged:
                original.write_json(output / "history_progress.json", dict(history=history,
                    latest_step=step, latest_checkpoint_step=latest_checkpoint, recipe=RECIPE))
                original.write_json(output / "status.json", dict(status="profiling" if options.profile_only else "training",
                    workflow_complete=False, actual_steps=step, latest_checkpoint_step=latest_checkpoint,
                    peak_cuda_bytes=peak, returncode=None, error=None, requires_user_action=False,
                    recipe=RECIPE, profile_only=options.profile_only))
        original._sync(device)
        training_seconds = time.perf_counter() - training_started
        optimizer_steps = original.optimizer_step_counts(optimizers, final_step)
        for parameter in model.parameters():
            parameter.grad = None
        final_sha = original.state_fingerprint(model)
        evaluation_started = time.perf_counter()
        arrays, roots, flow = None, [], None
        if not options.profile_only:
            original.write_json(output / "status.json", dict(status="evaluating", workflow_complete=False,
                actual_steps=step, latest_checkpoint_step=latest_checkpoint, returncode=None,
                error=None, requires_user_action=False, recipe=RECIPE))
            arrays = original.evaluate_model(model, dataset, device, budget_check=check)
            check()
            original.cell_grid_evaluation(model, arrays, device)
            roots, flow = bounded_flow(model, arrays, device, budget_check=check)
        peak = check()
        original._sync(device)
        evaluation_seconds = time.perf_counter() - evaluation_started
        if final_sha != original.state_fingerprint(model) or any(p.grad is not None for p in model.parameters()):
            raise RuntimeError("Model or parameter gradients changed during frozen evaluation")
        if original.sha256_file(Path(options.data) / "manifest.json") != dataset.manifest_sha256:
            raise RuntimeError("Physical data manifest changed")
        if any(original.sha256_file(path) != digest for path, digest in sources.items()):
            raise RuntimeError("Scientific source changed during execution")
        if arrays is not None:
            np.savez(output / "evaluation_arrays.npz", **arrays)
        steady = update_seconds[1:] or update_seconds
        summary = dict(status="complete", recipe=RECIPE, experiment_schema=EXPERIMENT_SCHEMA,
            actual_steps=step, latest_checkpoint_step=latest_checkpoint, optimizer_steps=optimizer_steps,
            theory_dim=2, lattice_sizes=manifest["lattice_sizes"], scale=dataset.scale,
            initial_state_sha256=initialization["training_start_state_sha256"],
            random_construction_state_sha256=random_sha,
            q_initialization=options.q_initialization, q_initialization_receipt=initialization,
            training_start_state_sha256=initialization["training_start_state_sha256"],
            final_state_sha256=final_sha, final_module_sha256=original.module_fingerprints(model),
            frozen_evaluation_verified=not options.profile_only,
            evaluation_skipped_for_profile=options.profile_only,
            data_manifest_sha256=dataset.manifest_sha256, labels_used_in_training=False,
            train_chains=int(len(dataset.train_indices)), test_chains=int(len(dataset.test_indices)),
            samples_per_chain=int(dataset.configurations.shape[1]), evaluation_seed=original.EVALUATION_SEED,
            closure=({split: original.closure_metrics(arrays[f"{split}_direct"],
                arrays[f"{split}_one_step"], arrays[f"{split}_rollout"], arrays["train_direct"])
                for split in ("train", "test")} if arrays is not None else None),
            g_statistics=({split: dict(mean=arrays[f"{split}_direct"].mean(0).tolist(),
                std=arrays[f"{split}_direct"].std(0, ddof=0).tolist(),
                minimum=arrays[f"{split}_direct"].min(0).tolist(),
                maximum=arrays[f"{split}_direct"].max(0).tolist()) for split in ("train", "test")}
                if arrays is not None else None),
            numerical_fixed_points=roots, flow_analysis=flow, thermal_nu=None,
            profile_only=options.profile_only, profile_endpoint_discarded=options.profile_only,
            peak_cuda_bytes=peak, device=receipt, equilibrium_certified=False,
            data_warning_count=dataset.warning_count, sampler_schema=dataset.sampler_schema,
            sample_semantics=dataset.sample_semantics, sampler_qualification=dataset.qualification,
            timing=dict(initialization_seconds=initialization_seconds, training_seconds=training_seconds,
                evaluation_seconds=evaluation_seconds, compute_seconds=time.perf_counter() - compute_started,
                wall_seconds=time.perf_counter() - started, cpu_seconds=time.process_time() - cpu_started,
                first_update_seconds=update_seconds[0], steady_update_seconds_mean=float(np.mean(steady)),
                steady_update_seconds_p95=float(np.quantile(steady, .95)), update_seconds=update_seconds),
            qualification="Single seed, finite data and finite basis; bounded numerical roots are not certified physical fixed points; no thermal mode selected")
        original.write_json(output / "summary.json", summary)
        original.write_json(output / "status.json", dict(status="complete", workflow_complete=True,
            actual_steps=step, latest_checkpoint_step=latest_checkpoint, returncode=0, error=None,
            requires_user_action=False, recipe=RECIPE, profile_only=options.profile_only,
            peak_cuda_bytes=peak, summary_sha256=original.sha256_file(output / "summary.json"),
            evaluation_arrays_sha256=(original.sha256_file(output / "evaluation_arrays.npz") if arrays is not None else None)))
        print(json.dumps(dict(event="phi4_ru_qcr_complete", actual_steps=step,
            profile_only=options.profile_only, training_seconds=training_seconds,
            evaluation_seconds=evaluation_seconds)), flush=True)
        return summary
    except Exception as exc:
        original.write_json(output / "status.json", dict(status="budget_hold" if isinstance(exc, original.BudgetHold) else "failed",
            workflow_complete=False, actual_steps=step, latest_checkpoint_step=latest_checkpoint,
            returncode=3 if isinstance(exc, original.BudgetHold) else 1,
            error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(),
            requires_user_action=True, recipe=RECIPE, profile_only=options.profile_only,
            wall_seconds=time.perf_counter() - started))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:2")
    for name, default in (("steps", 3000), ("system_batch", 8), ("sample_batch", 64),
        ("seed", 91), ("threads", 1), ("log_every", 100), ("checkpoint_every", 1000),
        ("information_pairs", 128), ("max_cuda_bytes", 8 * 1024 ** 3), ("profile_steps", 20)):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    parser.add_argument("--max-training-seconds", type=float, default=1800.)
    parser.add_argument("--expected-gpu-uuid")
    parser.add_argument("--q-initialization", default="train_mean_bias", choices=("train_mean_bias",))
    parser.add_argument("--expected-random-construction-sha256")
    parser.add_argument("--profile-only", action="store_true")
    try:
        run_training(TrainingOptions(**vars(parser.parse_args(argv))))
    except original.BudgetHold:
        return 3
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
