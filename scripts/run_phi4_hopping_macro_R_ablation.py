"""Explicit frozen-Q/C, CPU-only R source-scale weighting ablation.

Only an already public latent cache and the pinned R state are consumed.  This
is not a fresh QCR run, a physical sampler, a root finder, or a default change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch


SOURCE = Path("/mnt/nas-new/home/yange/ham/results/phi4_hopping_b2_L128_g1_seed91_gpu3_vm225_20261101_integrator_refined_v1")
CHECKPOINT_SHA = "1aec98b9d9c9d562ca32b6bef972cb79fa46ccc5f06819bb752129709dcf50d7"
DATA_SHA = "3d2023b401b15e20a6dca6c6b602959c237a030ff0e2f8f89676a7ca9ed30fa5"
CACHE_SHA = "1af96e4b1ad9e4d4a516d8552be80f4c924ba3f53677ad4c1b4c58127d91ae4f"
PAIRS = ((0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3))
ARMS = ("A_equal_target", "B_macro_source")
STEPS, BATCH, SEED, LR = 3000, 32, 91, .001
MAX_BYTES = 100 * 1024**2


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def state_sha(state):
    """Same sorted dtype/shape/byte fingerprint as the source trainer."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous().numpy()
        item = hashlib.sha256()
        for part in (str(value.dtype).encode(), str(value.shape).encode(), value.tobytes()):
            item.update(part)
        digest.update(name.encode())
        digest.update(item.hexdigest().encode())
    return digest.hexdigest()


def make_r(state):
    # Constructor randomness is private, not a restored checkpoint RNG stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        model = torch.nn.Sequential(torch.nn.Linear(1, 96), torch.nn.Tanh(),
            torch.nn.Linear(96, 96), torch.nn.Tanh(), torch.nn.Linear(96, 1)).double()
    model.load_state_dict(state, strict=True)
    return model


def predict_pairs(model, direct):
    """Detached data anchors, but uninterrupted recurrent R gradients."""
    direct = direct.detach()
    predictions = {}
    for start in range(3):
        value = direct[:, start]
        for target in range(start + 1, 4):
            value = value + model(value)
            predictions[start, target] = value
    return torch.stack([predictions[pair] for pair in PAIRS], dim=1)


def pair_weights(arm):
    if arm == "A_equal_target":
        return np.asarray([10 / (3 * target) for _, target in PAIRS])
    if arm == "B_macro_source":
        return np.asarray([10 * (start + 1) / (6 * (3 - start)) for start, _ in PAIRS])
    raise ValueError("Unknown explicit ablation arm")


def pair_errors(predicted, direct):
    target = torch.stack([direct[:, target].detach() for _, target in PAIRS], dim=1)
    return (predicted - target).square().mean(dim=(0, 2))


def objective(model, direct, arm):
    errors = pair_errors(predict_pairs(model, direct), direct)
    weights = torch.as_tensor(pair_weights(arm), dtype=errors.dtype, device=errors.device)
    return torch.sum(weights * errors)


def optimizer_counts(optimizer, model):
    counts = [int(optimizer.state[param]["step"].item()) for param in model.parameters()]
    return dict(min=min(counts), max=max(counts), parameters=len(counts))


def train_update(models, optimizers, direct_train, generator):
    """Training takes only TRAIN latent anchors, never labels, cells or TEST."""
    indices = torch.randint(len(direct_train), (BATCH,), generator=generator)
    batch = direct_train[indices].detach()
    for arm in ARMS:
        optimizers[arm].zero_grad(set_to_none=True)
        loss = objective(models[arm], batch, arm)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite R objective")
        loss.backward()
        if any(param.grad is None or not torch.isfinite(param.grad).all() for param in models[arm].parameters()):
            raise FloatingPointError("Missing/nonfinite R gradient")
        optimizers[arm].step()
        if any(not torch.isfinite(param).all() for param in models[arm].parameters()):
            raise FloatingPointError("Nonfinite R parameter")
    return indices


def validate_cache(arrays):
    for name in ("direct_g", "direct_full_g"):
        if arrays[name].shape != (60, 4, 1) or arrays[name].dtype != np.float64 or not np.isfinite(arrays[name]).all():
            raise ValueError(f"Unexpected {name} axes/dtype/values")
    if not np.array_equal(arrays["all_start_pairs"], PAIRS):
        raise ValueError("Source all-start pair order differs")
    splits, cells, labels = arrays["splits"], arrays["cell_ids"], arrays["labels"]
    if (splits.shape != (60,) or cells.shape != (60,) or labels.shape != (60, 3)
            or not np.isfinite(labels).all() or set(splits) != {"train", "test"}
            or (splits == "train").sum() != 40 or (splits == "test").sum() != 20
            or not np.array_equal(np.unique(cells), np.arange(15))):
        raise ValueError("Source physical-chain or holdout geometry differs")
    for cell in range(15):
        mask = cells == cell
        if mask.sum() != 4 or len(set(splits[mask])) != 1 or not np.all(labels[mask] == labels[mask][0]):
            raise ValueError("Cell split leaks or physical-chain count differs")
    if np.std(arrays["direct_g"][splits == "train"], axis=0, ddof=0).min() <= 0:
        raise ValueError("TRAIN target variation must be nonzero")


def calculate_metrics(direct, prediction, target_std, train_target_mean=None):
    errors = np.mean((prediction - direct[:, [t for _, t in PAIRS]])**2, axis=(0, 2))
    rows = []
    for i, (start, target) in enumerate(PAIRS):
        identity = float(np.sqrt(np.mean((direct[:, start] - direct[:, target])**2)))
        rmse = float(np.sqrt(errors[i]))
        rows.append(dict(start=start, target=target, rmse=rmse, nrmse=rmse / target_std[target],
            identity_rmse=identity, identity_nrmse=identity / target_std[target]))
        if train_target_mean is not None:
            constant_rmse = float(np.sqrt(np.mean((direct[:, target] - train_target_mean[target])**2)))
            rows[-1].update(constant_train_mean_rmse=constant_rmse,
                constant_train_mean_nrmse=constant_rmse / target_std[target])
    totals = {arm: float(pair_weights(arm) @ errors) for arm in ARMS}
    contributions = {arm: {str(start): float(sum(pair_weights(arm)[i] * errors[i]
        for i, pair in enumerate(PAIRS) if pair[0] == start)) for start in range(3)} for arm in ARMS}
    return dict(pairs=rows, equal_target_objective=totals[ARMS[0]],
        macro_source_objective=totals[ARMS[1]], source_group_contributions=contributions)


def complete(value, name):
    if (value.get("status") != "complete" or value.get("workflow_complete") is not True
            or value.get("returncode") != 0 or value.get("error") or value.get("requires_user_action") is not False):
        raise ValueError(f"{name} is not cleanly complete")


def bind_sources():
    """Check JSON pins and public latent cache only; never open field shards."""
    training = SOURCE / "training"
    job, plan, status, manifest, summary = [read(path) for path in
        (SOURCE / "job.json", SOURCE / "plan.json", training / "status.json",
         training / "manifest.json", training / "summary.json")]
    complete(job, "Source parent")
    complete(status, "Source training")
    if any(stage.get("status") != "complete" or stage.get("returncode") != 0 for stage in job["stages"]):
        raise ValueError("Source execution stage incomplete")
    if (job.get("plan_sha256") != sha(SOURCE / "plan.json") or summary.get("actual_steps") != 3000
            or summary.get("status") != "complete" or manifest.get("theory_dim") != 1
            or manifest.get("block_size") != 2 or manifest.get("lattice_sizes") != [128, 64, 32, 16]
            or manifest.get("hidden") != 96 or manifest.get("labels_used_in_training") is not False
            or summary.get("labels_used_in_training") is not False
            or manifest.get("data_manifest_sha256") != DATA_SHA
            or sha(SOURCE / "data/manifest.json") != DATA_SHA
            or sha(training / "evaluation_arrays.npz") != CACHE_SHA
            or status.get("evaluation_arrays_sha256") != CACHE_SHA
            or status.get("summary_sha256") != sha(training / "summary.json")):
        raise ValueError("Source geometry/cache/endpoint binding differs")
    protected = {path: sha(path) for path in (SOURCE / "job.json", SOURCE / "plan.json",
        training / "status.json", training / "manifest.json", training / "summary.json",
        training / "evaluation_arrays.npz", SOURCE / "data/manifest.json")}
    for pins in (plan["source_sha256"], manifest["source_sha256"]):
        for name, digest in pins.items():
            path = Path(name) if Path(name).is_absolute() else Path(plan["code_root"]) / name
            if sha(path) != digest:
                raise ValueError(f"Frozen source pin differs: {path}")
            protected[path] = digest
    allowed = {}
    for row in status["public_files"]:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe source public whitelist path")
        path = training / relative
        if sha(path) != row["sha256"]:
            raise ValueError("Source public whitelist SHA differs")
        allowed[str(relative)] = row["sha256"]
        protected[path] = row["sha256"]
    if any(name not in allowed for name in ("manifest.json", "summary.json", "evaluation_arrays.npz")):
        raise ValueError("Missing required public source cache/metadata")
    checkpoint_path = training / "training_step_3000.pt"
    if sha(checkpoint_path) != CHECKPOINT_SHA:
        raise ValueError("Source checkpoint SHA differs")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("step") != 3000 or checkpoint.get("manifest") != manifest:
        raise ValueError("Source checkpoint endpoint/manifest differs")
    for name in ("Q", "C", "R"):
        state = {key[len(name) + 1:]: value for key, value in checkpoint["model"].items() if key.startswith(name + ".")}
        if state_sha(state) != summary["final_module_sha256"][name]:
            raise ValueError(f"Source {name} fingerprint differs")
    state = {key[2:]: value.detach().clone() for key, value in checkpoint["model"].items() if key.startswith("R.")}
    del checkpoint  # Optimizer and training RNG payloads are never restored.
    protected[checkpoint_path] = CHECKPOINT_SHA
    with np.load(training / "evaluation_arrays.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in ("direct_g", "direct_full_g", "labels", "splits",
            "cell_ids", "all_start_pairs", "one_step_g", "flow_grid", "flow_next")}
    validate_cache(arrays)
    return state, arrays, protected, summary["final_module_sha256"]


class BudgetHold(RuntimeError):
    pass


def run(output, max_cpu_seconds=300.):
    # process_time is process-cumulative, including interpreter/module imports.
    started, wall_started = 0., time.monotonic()
    if not 30 < max_cpu_seconds <= 300:
        raise ValueError("CPU budget must be in (30,300]")
    output = Path(output).resolve()
    if output.exists() or output == SOURCE or SOURCE in output.parents:
        raise ValueError("Require a fresh independent output outside the formal root")
    if os.uname().nodename != "thinklab-105-225" or os.getpriority(os.PRIO_PROCESS, 0) < 10:
        raise RuntimeError("Wrong host/priority; no fallback")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise RuntimeError("Launch with CUDA_VISIBLE_DEVICES empty; CUDA must remain uninitialized")
    host = dict(cpus=os.cpu_count(), load1=os.getloadavg()[0],
        mem_available_bytes=int(Path("/proc/meminfo").read_text().split("MemAvailable:")[1].split()[0]) * 1024,
        io_full_avg10=float(Path("/proc/pressure/io").read_text().split("full avg10=")[1].split()[0]))
    if host["mem_available_bytes"] < 36 * 1024**3 or host["load1"] > 1.25 * host["cpus"] or host["io_full_avg10"] >= 20:
        raise BudgetHold("Resource gate: wait without starting, restarting or migrating")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output.mkdir(parents=True)
    job = dict(status="running", workflow_complete=False, returncode=None, error=None,
        requires_user_action=False, current="source_verification", actual_steps=0,
        cpu_budget_seconds=max_cpu_seconds, cpu_seconds=0., gpu_seconds=0., pid=os.getpid())
    def status(**updates):
        job.update(updates, cpu_seconds=time.process_time() - started,
            wall_seconds=time.monotonic() - wall_started)
        write(output / "job.json", job)
        write(output / "status.json", job)
    def guard(reserve=15., check_directory=False):
        if time.process_time() - started > max_cpu_seconds - reserve:
            raise BudgetHold("CPU budget reserve reached; no automatic restart or shortened successful run")
        if check_directory and sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) > MAX_BYTES:
            raise BudgetHold("Independent directory budget exceeded")
    status()
    try:
        own_source = {str(Path(__file__).resolve()): sha(__file__)}
        reporter = Path(__file__).with_name("report_phi4_hopping_macro_R_ablation.py")
        own_source[str(reporter.resolve())] = sha(reporter)
        rng_before = torch.get_rng_state().clone()
        state, arrays, protected, modules = bind_sources()
        source = make_r(state).eval().requires_grad_(False)
        direct = torch.as_tensor(arrays["direct_g"], dtype=torch.float64)
        with torch.no_grad():
            source_one = direct[:, :-1] + source(direct[:, :-1])
            saved_grid = torch.as_tensor(arrays["flow_grid"], dtype=torch.float64)
            source_flow = saved_grid + source(saved_grid)
        reproduction_error = float(np.max(np.abs(source_one.numpy() - arrays["one_step_g"])))
        reproduction_flow_error = float(np.max(np.abs(source_flow.numpy() - arrays["flow_next"])))
        if max(reproduction_error, reproduction_flow_error) > 1e-10:
            raise ValueError("CPU R does not reproduce saved one-step/flow predictions")
        source_sha = state_sha(source.state_dict())
        if source_sha != modules["R"]:
            raise ValueError("Constructed residual R differs")
        models = {arm: make_r(state).train() for arm in ARMS}
        optimizers = {arm: torch.optim.Adam(models[arm].parameters(), lr=LR) for arm in ARMS}
        train_mask = arrays["splits"] == "train"
        direct_train = direct[train_mask].detach().clone()
        target_std = arrays["direct_g"][train_mask].std(axis=0, ddof=0)[:, 0]
        target_mean = arrays["direct_g"][train_mask].mean(axis=0)[:, 0]
        plan = dict(schema_version=5, recipe="phi4_frozen_QC_macro_source_R_ablation_v1", source_root=str(SOURCE),
            source_checkpoint_step=3000, source_checkpoint_sha256=CHECKPOINT_SHA, source_module_sha256=modules,
            source_data_manifest_sha256=DATA_SHA, source_cache_sha256=CACHE_SHA, source_sha256=own_source,
            lattice_sizes=[128, 64, 32, 16], block_size=2, theory_dim=1, hidden=96, dtype="float64",
            arms={arm: dict(pair_weights=pair_weights(arm).tolist()) for arm in ARMS},
            all_start_pairs=PAIRS, macro_source_weights=[1/6, 2/6, 3/6], closure_outer_weight=10,
            extra_R_updates=STEPS, source_QCR_updates=3000, batch_physical_chains=BATCH,
            sampling="with replacement; exactly paired batches between arms", seed=SEED, learning_rate=LR,
            Q_updates=0, C_updates=0, fresh_Adam=True, labels_used_in_training=False, test_used_in_training=False,
            train_physical_chains=40, test_physical_chains=20, CPU_only=True, gpu_seconds=0.,
            max_cpu_seconds=max_cpu_seconds, max_directory_bytes=MAX_BYTES, resource_receipt=host,
            R_normalization="unchanged residual advance(g)=g+network(g)",
            checkpoint_selection="none; fixed final 3000; no early stopping or root search",
            evaluation_std="original frozen TRAIN direct_g target-scale population std, ddof=0",
            frozen_anchor_definition="direct_g = mean of Q encodings from first/second halves, 64 existing fields each; not Q of pooled 128",
            constant_baseline="one frozen original TRAIN direct_g mean per target scale")
        write(output / "plan.json", plan)
        job["plan_sha256"] = sha(output / "plan.json")
        generator = torch.Generator(device="cpu").manual_seed(SEED)
        history = []
        def log(step):
            guard(check_directory=True)
            row = dict(step=step, arms={})
            with torch.no_grad():
                for arm in ARMS:
                    prediction = predict_pairs(models[arm], direct_train).numpy()
                    metrics = calculate_metrics(direct_train.numpy(), prediction, target_std, target_mean)
                    row["arms"][arm] = dict(metrics, objective=metrics[
                        "equal_target_objective" if arm == ARMS[0] else "macro_source_objective"])
            history.append(row)
            write(output / "history.json", history)
            status(current="R_only_training", actual_steps=step)
            print(json.dumps(dict(step=step, arms={arm: row["arms"][arm]["objective"] for arm in ARMS})), flush=True)
        log(0)
        for step in range(1, STEPS + 1):
            guard()
            train_update(models, optimizers, direct_train, generator)
            if step % 100 == 0:
                log(step)
        guard(check_directory=True)
        status(current="fixed_final_evaluation")
        low, high = float(direct_train.min()), float(direct_train.max())
        padding = .1 * max(high - low, 1e-6)
        grid = np.linspace(low - padding, high + padding, 501)[:, None]
        exported = {key: arrays[key] for key in ("direct_g", "direct_full_g", "labels", "splits", "cell_ids", "all_start_pairs")}
        exported.update(flow_grid=grid, train_target_std=target_std, train_target_mean=target_mean)
        metrics = dict(models={}, train_target_std=target_std.tolist(), train_target_mean=target_mean.tolist(), all_start_pairs=PAIRS)
        for name, model in dict(source=source, **models).items():
            model.eval()
            model.zero_grad(set_to_none=True)
            with torch.no_grad():
                prediction = predict_pairs(model, direct).numpy()
                exported[name + "_one_step"] = (direct[:, :-1] + model(direct[:, :-1])).numpy()
                exported[name + "_all_start"] = prediction
                exported[name + "_rollout"] = np.concatenate((arrays["direct_g"][:, :1], prediction[:, [0, 1, 3]]), axis=1)
                tg = torch.as_tensor(grid, dtype=torch.float64)
                exported[name + "_flow_next"] = (tg + model(tg)).numpy()
            metrics["models"][name] = {split.upper(): calculate_metrics(arrays["direct_g"][arrays["splits"] == split],
                prediction[arrays["splits"] == split], target_std, target_mean) for split in ("train", "test")}
            if name in ARMS:
                counts = optimizer_counts(optimizers[name], model)
                if counts != dict(min=STEPS, max=STEPS, parameters=6):
                    raise ValueError("R optimizer clocks differ from fixed endpoint")
                torch.save(dict(step=STEPS, source_step=3000, arm=name, R=model.state_dict(),
                    optimizer_steps=counts, optimizer_restored=False, Q_updates=0, C_updates=0), output / (name + "_R_final.pt"))
            if any(param.grad is not None for param in model.parameters()):
                raise ValueError("Evaluation must leave all R gradients None")
        if any(not np.isfinite(value).all() for value in exported.values() if value.dtype.kind in "fc"):
            raise FloatingPointError("Nonfinite final public arrays")
        np.savez_compressed(output / "evaluation_arrays.npz", **exported)
        write(output / "metrics.json", metrics)
        if state_sha(source.state_dict()) != source_sha or not torch.equal(rng_before, torch.get_rng_state()) or torch.cuda.is_initialized():
            raise ValueError("Source R/global RNG/CUDA protection differs")
        for path, digest in protected.items():
            if sha(path) != digest:
                raise ValueError(f"Protected source changed: {path}")
        for path, digest in own_source.items():
            if sha(path) != digest:
                raise ValueError("Ablation implementation changed during execution")
        summary = dict(status="complete", schema_version=5, recipe=plan["recipe"], actual_steps=STEPS,
            source_QCR_steps=3000, additional_R_steps_per_arm=STEPS, Q_updates=0, C_updates=0,
            optimizer_steps={arm: optimizer_counts(optimizers[arm], models[arm]) for arm in ARMS},
            final_R_module_sha256={arm: state_sha(models[arm].state_dict()) for arm in ARMS},
            source_module_sha256=modules, source_frozen_verified=True, source_cache_sha256=CACHE_SHA,
            cpu_R_one_step_reproduction_max_abs=reproduction_error, all_R_grad_None=True,
            cpu_R_saved_flow_reproduction_max_abs=reproduction_flow_error,
            frozen_anchor_definition=plan["frozen_anchor_definition"],
            global_RNG_unchanged=True, cuda_initialized=False, gpu_seconds=0., labels_used_in_training=False,
            test_used_for_checkpoint_selection=False, thermal_nu=None, critical_kappa=None,
            physical_sampling=0, Q_C_model_constructions=0, equilibrium_certified=False, retained_quality_warning_count=75,
            limitations=["Single fixed Q/C, cache, physical sample set and training seed.",
                "Finite-sample R-only weighting test, not new QCR training or certified RG.",
                "Cached ensemble g anchors, not new sample-minibatch C/Q inference.",
                "Flow-grid extrapolation is not evidence of a physical fixed point.",
                "Original 75 sampling warnings retained; no thermal exponent or root search."])
        write(output / "summary.json", summary)
        guard(check_directory=True)
        status(current="JSON_NPZ_report")
        from report_phi4_hopping_macro_R_ablation import render_report
        report_paths = render_report(output)
        # A report is read-only; include its whole duration in source guards.
        for path, digest in {**protected, **own_source}.items():
            if sha(path) != digest:
                raise ValueError(f"Protected source changed during report: {path}")
        if state_sha(source.state_dict()) != source_sha or not torch.equal(rng_before, torch.get_rng_state()) or torch.cuda.is_initialized():
            raise ValueError("Source R/global RNG/CUDA changed during report")
        guard(reserve=0., check_directory=True)
        status(status="complete", workflow_complete=True, returncode=0, current=None,
            error=None, requires_user_action=False, actual_steps=STEPS)
        names = ["job.json", "status.json", "plan.json", "summary.json", "history.json", "metrics.json", "evaluation_arrays.npz"] + report_paths
        public = []
        for name in sorted(set(names)):
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts or relative.suffix not in (".json", ".npz", ".png"):
                raise ValueError("Unsafe/nonpublic result path")
            path = output / relative
            public.append(dict(path=str(relative), sha256=sha(path), bytes=path.stat().st_size))
        write(output / "public_files.json", public)
        print(json.dumps(dict(status="complete", output=str(output), public_manifest_sha256=sha(output / "public_files.json"),
            cpu_seconds=time.process_time() - started), ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        status(status="budget_hold" if isinstance(exc, BudgetHold) else "failed", workflow_complete=False,
            returncode=3 if isinstance(exc, BudgetHold) else 1, error=str(exc),
            requires_user_action=True, traceback=traceback.format_exc())
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cpu-seconds", type=float, default=300.)
    args = parser.parse_args(argv)
    return run(args.output, args.max_cpu_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
