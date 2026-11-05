"""Single-seed, same-data baseline adaptations; never native-paper claims.

Native C objectives are trained without K/T labels. A common TRAIN-only PCA
and residual R are *additional comparison readouts*, not original Neural MCRG
or RSMI-NE outputs. All physical cache tensors remain read-only and mmap-backed.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch

from recurrent_rg.baselines.nmcrg import PeriodicConvRBM
from recurrent_rg.baselines.readout import (
    build_r, fit_affine_readout, fit_train_pca, path_metrics,
    physical_error_decomposition,
)
from recurrent_rg.baselines.rsmine import RSMINE
from recurrent_rg.data.ensembles import load_ensembles


METHODS = ("nmcrg", "rsmine", "decimation", "majority", "qcr_c_pca")
C_UPDATES = 12000
R_UPDATES = 12000
DIMENSION = 5
SCALES = (108, 36, 12)


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + "\n")
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def tensor_sha(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(key.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def cpu_time():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def peak_memory(device):
    # Conservatively include a 1-GiB runtime/context/library envelope outside
    # the Torch allocator. This is a budget envelope, not an NVML measurement.
    return (max(torch.cuda.max_memory_allocated(device),torch.cuda.max_memory_reserved(device))
            + 1024**3) \
        if torch.device(device).type == "cuda" else 0


def configure(device, threads=2, gpu_peak_bytes=None):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    if torch.device(device).type == "cuda":
        torch.cuda.set_device(device)
        if gpu_peak_bytes is not None:
            total = torch.cuda.get_device_properties(device).total_memory
            fraction = (gpu_peak_bytes - 1024**3) / total
            if not 0 < fraction <= 1:
                raise ValueError("Own GPU allocator budget needs a positive context reserve")
            torch.cuda.set_per_process_memory_fraction(fraction, device)
        torch.cuda.reset_peak_memory_stats(device)


def device_receipt(plan):
    from experiments.phi4_qcr import _device_receipt
    device = torch.device(plan["device"])
    return _device_receipt(device, plan.get("gpu_uuid") if device.type == "cuda" else None)


def generator(seed, device="cpu"):
    return torch.Generator(device=device).manual_seed(int(seed))


def source_C_table(checkpoint):
    """Only the frozen seed91 C is read; its Q/R are not used as readouts."""
    from recurrent_rg.checkpoints import load_run
    from recurrent_rg.binary_pooling import binary_pattern_table

    loaded = load_run(checkpoint)
    model = loaded.model.eval()
    before = tensor_sha(model)
    with torch.no_grad():
        parameter = next(model.parameters())
        _, patterns = binary_pattern_table(9, parameter, cached=True)
        kernel = model.rg_kernel
        logits = kernel.field_head.logits(kernel.local_encoder(patterns))
        probability = torch.sigmoid(2 * logits).reshape(512).cpu().double()
    if tensor_sha(model) != before or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("Frozen source Q/C/R changed while extracting C table")
    return probability, before


def construct_C(method, *, seed, device, source_table=None):
    rng = generator(seed)
    if method == "nmcrg":
        # The native models use FP32; the common PCA/R/statistics use FP64.
        return PeriodicConvRBM(filter_width=9, stride=3, generator=rng,
                               device="cpu", dtype=torch.float32).to(device)
    if method == "rsmine":
        return RSMINE(generator=rng, device=device, dtype=torch.float32)
    if method == "qcr_c_pca" and source_table is None:
        raise ValueError("Frozen source C requires its audited probability table")
    if method not in METHODS:
        raise ValueError("Unknown comparison method")
    return None


def native_C_training(configurations, method, model, *, updates, seed, device,
                      folder=None, progress=None):
    """This boundary accepts configurations ONLY, not coupling/group metadata.

    Uniformly sampling physical chains weights all 43 TRAIN conditions equally
    (each has 12 chains). RSMI negatives remain within the same selected chain.
    No temperature classifier, QCR MI/colour/variance loss or R loss is added.
    """
    if model is None:
        return [], 0.
    from recurrent_rg.baselines.statistics import native_pair_batch

    cpu_rng, device_rng = generator(seed + 1), generator(seed + 2, device)
    initial_sha = tensor_sha(model)
    if method == "nmcrg":
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, eps=1e-8)
    else:
        # Keras/TF Adam default epsilon used by the author implementation.
        optimizer = torch.optim.Adam(model.parameters(), lr=.005, eps=1e-7)
    history = []
    started = time.perf_counter()
    for step in range(1, updates + 1):
        optimizer.zero_grad(set_to_none=True)
        if method == "nmcrg":
            chains = torch.randint(len(configurations), (50,), generator=cpu_rng)
            samples = torch.randint(configurations.shape[1], (50,), generator=cpu_rng)
            visible = configurations[chains, samples].to(device=device, dtype=torch.float32)
            loss = model.cd_loss(visible, steps=3, generator=device_rng)
            extra = {}
        else:
            chains = torch.randint(len(configurations), (16,), generator=cpu_rng)
            visible, environment = native_pair_batch(configurations, chains,
                pairs_per_chain=64, generator=cpu_rng)
            result = model.mi_loss(visible.to(device=device, dtype=torch.float32),
                environment.to(device=device, dtype=torch.float32), step=step - 1,
                generator=device_rng)
            loss = result.loss
            extra = dict(mi_estimate=float(result.mi_estimate.detach().cpu()),
                         temperature=result.temperature)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite native C objective at update {step}")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError(f"Nonfinite native C gradient at update {step}")
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == updates:
            row = dict(stage="C", step=step, native_loss=float(loss.detach().cpu()),
                       **extra, elapsed_seconds=time.perf_counter() - started)
            history.append(row)
            print(json.dumps(row), flush=True)
            if progress:
                progress(current="native_C_training", C_step=step, R_step=0)
    sync(device)
    elapsed = time.perf_counter() - started
    if folder is not None:
        torch.save(dict(state_dict=model.state_dict(), method=method, updates=updates,
                        seed=seed, initial_state_sha256=initial_sha,
                        final_state_sha256=tensor_sha(model)), Path(folder) / "native_C.pt")
    model.eval()
    for parameter in model.parameters():
        parameter.grad = None
        parameter.requires_grad_(False)
    return history, elapsed


def collect_statistics(configurations, method, model, *, draws, seed, device,
                       source_table=None, progress=None):
    """Collect each independent physical chain separately, with conditional C draws.

    No lattice-sized float64 dataset is built or cached. Temporary GPU buffers
    hold at most four chains; only small ensemble motif summaries are saved.
    """
    from recurrent_rg.baselines.statistics import coarsen_tensor, motif_probabilities_tensor

    n = len(configurations)
    f512 = np.empty((n, draws, 3, 512), dtype=np.float64)
    p16 = np.empty((n, draws, 3, 16), dtype=np.float64)
    observations = np.empty((n, draws, 3, 3), dtype=np.float64)
    rng = generator(seed, device)
    table = source_table.to(device=device) if source_table is not None else None
    c_before = tensor_sha(model) if model is not None else None
    started = time.perf_counter()
    with torch.no_grad():
        for begin in range(0, n, 4):
            end = min(n, begin + 4)
            source = configurations[begin:end].to(device=device, dtype=torch.float32)
            micro_f = motif_probabilities_tensor(source, width=3).cpu().numpy()
            micro_p = motif_probabilities_tensor(source, width=2).cpu().numpy()
            for draw in range(draws):
                spins = source
                for scale in range(3):
                    f512[begin:end, draw, scale] = (micro_f if scale == 0 else
                        motif_probabilities_tensor(spins, width=3).cpu().numpy())
                    p16[begin:end, draw, scale] = (micro_p if scale == 0 else
                        motif_probabilities_tensor(spins, width=2).cpu().numpy())
                    magnetization = spins.double().mean((-2, -1))
                    observations[begin:end, draw, scale] = torch.stack((
                        magnetization.mean(-1), magnetization.abs().mean(-1),
                        .5 * (spins * torch.roll(spins, 1, -1) +
                              spins * torch.roll(spins, 1, -2)).double().mean((-3, -2, -1))),
                        dim=-1).cpu().numpy()
                    if scale < 2:
                        spins = coarsen_tensor(spins, method, model=model,
                            probabilities=table, generator=rng)
            if progress and (end == n or end % 48 == 0):
                progress(current="frozen_collection", collected_chains=end, total_chains=n)
    sync(device)
    if model is not None and (tensor_sha(model) != c_before or
                              any(p.grad is not None for p in model.parameters())):
        raise RuntimeError("Native C changed during frozen collection")
    return dict(f512=f512, p16=p16, observations=observations), time.perf_counter() - started


def closure_objective(flow, direct):
    """Uniform target k, separate one-step and microscopic-start branches."""
    one = torch.stack([flow(direct[..., k, :]) for k in (0, 1)], dim=-2)
    first = flow(direct[..., 0, :])
    rollout = torch.stack((first, flow(first)), dim=-2)
    one_loss = ((one - direct[..., 1:, :]) ** 2).sum(-1).mean()
    rollout_loss = ((rollout - direct[..., 1:, :]) ** 2).sum(-1).mean()
    return 5 * one_loss + 5 * rollout_loss, one_loss, rollout_loss


def train_R(g_train, *, updates, seed, device, folder=None, progress=None):
    """Train only the common R adapter; no graph or loss can reach C/PCA."""
    flow = build_r(5, 96, seed=seed).to(device)
    data = torch.as_tensor(g_train, dtype=torch.float64, device=device)
    optimizer = torch.optim.Adam(flow.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [6000, 9000], gamma=.3)
    rng = generator(seed + 3, device)
    history = []
    started = time.perf_counter()
    for step in range(1, updates + 1):
        indices = torch.randint(len(data), (16,), generator=rng, device=device)
        # All three fixed C draws are conditional repeats of the selected chain.
        loss, one, recurrent = closure_objective(flow, data[indices])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite common R objective at update {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(not torch.isfinite(p.grad).all() for p in flow.parameters()):
            raise FloatingPointError(f"Nonfinite common R gradient at update {step}")
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 100 == 0 or step == updates:
            row = dict(stage="R", step=step, loss=float(loss.detach().cpu()),
                one_step=float(one.detach().cpu()), rollout=float(recurrent.detach().cpu()),
                lr=optimizer.param_groups[0]["lr"], elapsed_seconds=time.perf_counter() - started)
            history.append(row)
            print(json.dumps(row), flush=True)
            if progress:
                progress(current="common_R_training", R_step=step)
    sync(device)
    elapsed = time.perf_counter() - started
    if folder is not None:
        torch.save(dict(state_dict=flow.state_dict(), seed=seed, updates=updates,
                        dimension=5, hidden=96), Path(folder) / "common_R.pt")
    flow.eval()
    for parameter in flow.parameters():
        parameter.grad = None
        parameter.requires_grad_(False)
    return flow, history, elapsed


def evaluate(flow, g, decoder, statistics, ntrain, *, device):
    data = torch.as_tensor(g, dtype=torch.float64, device=device)
    with torch.no_grad():
        one = flow(data[..., :2, :]).cpu().numpy()
        first = flow(data[..., 0, :])
        rollout = torch.stack((data[..., 0, :], first, flow(first)), dim=-2).cpu().numpy()
        r_mean = flow(data.mean(1)[..., :2, :]).cpu().numpy()
    direct_p, rollout_p = decoder.predict(g), decoder.predict(rollout)
    arrays = dict(g=g, r_one=one, rollout=rollout,
        r_of_C_mean=r_mean, mean_R=one.mean(1), p16=statistics["p16"],
        decoded_direct=direct_p, decoded_rollout=rollout_p,
        observations=statistics["observations"], split=np.r_[np.zeros(ntrain, dtype=np.int8),
            np.ones(len(g) - ntrain, dtype=np.int8)])
    metrics = {}
    for split, subset in (("train", slice(0, ntrain)), ("test", slice(ntrain, len(g)))):
        if len(g[subset]) < 2:
            continue
        paths = path_metrics(g[subset], one[subset], rollout[subset], g[:ntrain, 0])
        long_path = next(row for row in paths["paths"] if (row["start"], row["end"]) == (0, 2))
        physical = physical_error_decomposition(statistics["p16"][subset, :, 1:],
            direct_p[subset, :, 1:], rollout_p[subset, :, 1:])
        last_physical = physical_error_decomposition(statistics["p16"][subset, :, 2:3],
            direct_p[subset, :, 2:3], rollout_p[subset, :, 2:3])
        physical_identity = physical_error_decomposition(statistics["p16"][subset, :, 1:],
            direct_p[subset, :, 1:], np.repeat(direct_p[subset, :, :1], 2, axis=2))
        direct = statistics["p16"][subset]
        entropy = -(direct * np.log(np.maximum(direct, 1e-300))).sum(-1)
        metrics[split] = dict(latent=dict(rollout_rmse=long_path["R"]["rmse"],
            rollout_nrmse=long_path["R"]["nrmse"], identity_nrmse=long_path["identity"]["nrmse"],
            R_to_identity_rmse_ratio=long_path["R_to_identity_rmse_ratio"], details=paths),
            motif16=dict(rollout_rmse=last_physical["total_rmse"],
                all_target_scales=physical, last_target_scale=last_physical,
                identity_all_target_scales=physical_identity,
                direct_entropy_by_scale=entropy.mean((0, 1)).tolist(),
                direct_across_chain_std_by_scale=direct.mean(1).std(0).mean(-1).tolist(),
                affine_out_of_probability_interval_fraction=float(np.mean(
                    (rollout_p[subset] < 0) | (rollout_p[subset] > 1))),
                no_probability_clipping=True),
            physical_chain_count=len(g[subset]), C_draws=g.shape[1])
    return arrays, metrics


def find_roots(flow, g_train, *, maximum_starts=24):
    """Bounded work, full 5D roots; projection crossings are not fixed points.

    Starts and support thresholds use TRAIN encodings only. Retain every
    complex/negative eigenvalue. No eigenvalue is assigned a thermal identity.
    """
    from scipy.optimize import root
    model = copy.deepcopy(flow).cpu()
    means = g_train.mean(1).reshape(-1, 5)
    center, std = means.mean(0), means.std(0)
    norm = np.maximum(std, 1e-8)
    # Deterministic TRAIN ordering, not chosen by Kc or proximity to lambda=3.
    ids = np.linspace(0, len(means) - 1, maximum_starts - 1).astype(int)
    starts = np.concatenate((means[ids], center[None]))

    def displacement(x):
        with torch.no_grad():
            t = torch.as_tensor(x, dtype=torch.float64)
            return (model(t) - t).numpy()

    def jacobian(x):
        t = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        return torch.autograd.functional.jacobian(model, t).detach().numpy()

    found = []
    for start in starts:
        solution = root(displacement, start, jac=lambda x: jacobian(x) - np.eye(5),
                        method="hybr", options=dict(maxfev=200, xtol=1e-9))
        x = solution.x
        residual = float(np.linalg.norm(displacement(x)))
        if not np.isfinite(x).all() or residual > 1e-7:
            continue
        if any(np.linalg.norm((x - np.asarray(row["g"])) / norm) < 1e-4 for row in found):
            continue
        eigenvalues = np.linalg.eigvals(jacobian(x))
        absolute = np.abs(eigenvalues)
        contracting, expanding = int(np.sum(absolute < 1 - 1e-6)), int(np.sum(absolute > 1 + 1e-6))
        nearest = float(np.min(np.linalg.norm((means - x) / norm, axis=1)))
        found.append(dict(g=x.tolist(), residual=residual, solver_success=bool(solution.success),
            full_DR_eigenvalues=[dict(real=float(v.real), imag=float(v.imag)) for v in eigenvalues],
            contracting=contracting, expanding=expanding, neutral=5-contracting-expanding,
            numerical_type=("stable" if contracting == 5 else "repeller" if expanding == 5 else "saddle_or_neutral"),
            nearest_TRAIN_standardized_distance=nearest, physical_support_certified=False,
            thermal_nu=None, warning="Numerical root only; local data support and a thermal scaling field are not certified"))
    return found


def manifest(plan, method):
    native = dict(
        nmcrg=dict(objective="CD3 of periodic zero-bias convolutional RBM", filter_width=9,
            stride=3, overlapping=True, batch_configurations=50, lr=.001, adam_epsilon=1e-8,
            updates=C_UPDATES, dtype="float32", native_source="https://arxiv.org/abs/2010.05703",
            changes_from_paper=["same B0 TRAIN cache across K, not known-critical input only",
                "9x9 stride3 instead of 8x8 stride2", "12000 fixed updates instead of paper 20000",
                "common PCA/R added; not native MCRG operator-matrix inference"]),
        rsmine=dict(objective="grouped native RelaxedBernoulli separable-critic InfoNCE",
            block=3, stride=3, buffer=1, environment_sites=24, bias=False,
            critic_hidden=[32,32], embedding=8, groups_per_update=16, pairs_per_group=64,
            negative_policy="same condition AND same physical chain only", lr=.005,
            adam_epsilon=1e-7, temperature_initial=.5, temperature_minimum=.1,
            temperature_rate=.005, updates=C_UPDATES, dtype="float32",
            native_source="https://arxiv.org/abs/2103.16887",
            changes_from_paper=["one shared C across all TRAIN K instead of state-specific fitting",
                "3x3 stride3 instead of paper 4x4 Ising block", "grouped batch16x64,12000 updates",
                "PyTorch implementation; common PCA/R added, not a native thermometer readout"]),
        decimation=dict(objective=None, block=3, stride=3, fixed=True, updates=0),
        majority=dict(objective=None, block=3, stride=3, fixed=True, updates=0, tie_rule="none: 9 spins"),
        qcr_c_pca=dict(objective=None, block=3, stride=3, fixed=True, updates=0,
            checkpoint=plan["source_checkpoint"], checkpoint_sha256=plan["source_checkpoint_sha256"],
            meaning="Existing QCR seed91 C frozen; replace Q only by common TRAIN PCA"))
    return dict(schema_version=5, method=method, seed=plan["seed"], data_sha256=plan["data_sha256"],
        source_data=plan["source_data"], labels_used_in_training=False, labels_for_evaluation_only=True,
        scientific_claim="Single-seed same-data method adaptation, NOT original-paper exact reproduction",
        native_C=native[method], scales=list(SCALES), C_draws=plan["C_draws"],
        source_model_unchanged=True, training_cache_modified=False,
        runtime=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            CUBLAS_WORKSPACE_CONFIG=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            tf32=False, gpu_peak_accounting="max Torch allocated/reserved plus 1-GiB context envelope"),
        common_readout=dict(PCA_input="all 512 centered 3x3 binary motif probabilities",
            dimension=5, fit="TRAIN only, jointly over all three scales and conditional C draws",
            shared_scale_map=True, decoder="TRAIN affine 5->16 oriented2x2 motif minimum-norm fit",
            decoder_rcond=1e-12, no_rank_enforcement=True, R="residual 5->96tanh->96tanh->5",
            R_updates=R_UPDATES, R_seed=plan["seed"], same_R_initialization_across_baselines=True,
            not_same_initialization_as_whole_QCR_constructor=True, R_optimizer="fresh Adam .001",
            R_milestones=[6000,9000], gamma=.3, batch_physical_chains=16,
            one_step_weight=5, rollout_weight=5, target_scale_weights=[1,1],
            coordinate_reduction="sum", normalization_over_target_scales="mean per branch",
            dtype="float64", C_or_PCA_gradients_from_R=False,
            encoding="PCA of full ensemble motif mean; exact equality to mean of two half-PCAs since PCA affine",
            normalization="base C draw TRAIN target population std at each original target scale"),
        comparability_caution="Different C create different coarse targets. Do not rank methods solely by latent NRMSE or low-entropy/collapsed target closure.",
        thermal_nu=None, no_parameter_or_training_seed_search=True)


class BudgetHold(RuntimeError):
    """A declared budget gate, not an optimizer or scientific failure."""


def budget_guard(plan, root, elapsed, used_cpu):
    """Check actual cumulative usage without killing or retrying any process."""
    job = read(root / "job.json")
    policy = plan["resource_policy"]
    if (job.get("gpu_elapsed_seconds", 0.) + elapsed >= policy["gpu_seconds"]
            or job.get("cpu_seconds", 0.) + used_cpu >= policy["cpu_seconds"]
            or peak_memory(plan["device"]) > policy["gpu_peak_bytes"]):
        raise BudgetHold("Actual cumulative CPU/GPU or own allocated memory reached the fixed budget")


def validate_data_geometry(tx, vx):
    if (tuple(tx.shape[1:]) != (128,108,108) or tuple(vx.shape[1:]) != (128,108,108)
            or len(tx) != 516 or len(vx) != 252):
        raise ValueError("Same-data cache shape differs from frozen protocol")


def evaluation_K_labels(couplings, expected_count):
    """Copy scalar K metadata to [N] for evaluation, never a training input.

    Historical caches legitimately store scalar labels as either [N] or [N,1].
    Do not flatten multi-coupling metadata into a purported scalar K label.
    This small metadata preflight also runs before any formal fitting.
    """
    if isinstance(couplings, torch.Tensor):
        if couplings.dtype == torch.bool or couplings.is_complex():
            raise ValueError("Evaluation K labels must be finite real numeric metadata")
        values = couplings.detach().to(device="cpu", dtype=torch.float64).numpy()
    elif isinstance(couplings, np.ndarray) and couplings.dtype.kind in "iuf":
        values = couplings
    else:
        raise ValueError("Evaluation K labels must be finite real numeric metadata")
    if values.shape not in ((expected_count,), (expected_count, 1)):
        raise ValueError("Evaluation K labels must have shape [N] or [N,1] matching physical chains")
    if not np.isfinite(values).all():
        raise ValueError("Evaluation K labels must be finite real numeric metadata")
    return np.array(values.reshape(expected_count), dtype=np.float64, copy=True)


def run_arm(plan, root, method):
    from recurrent_rg.baselines.report import write_arm_report

    folder = root / "arms" / method
    folder.mkdir(parents=True, exist_ok=False)
    status = dict(status="running", method=method, started_at=now(), C_step=0, R_step=0,
                  error=None, requires_user_action=False,
                  physical_device_receipt=device_receipt(plan))
    start_wall, start_cpu = time.perf_counter(), cpu_time()

    def progress(**values):
        status.update(values, updated_at=now(), peak_gpu_bytes=peak_memory(plan["device"]))
        save(folder / "status.json", status)
        if status["status"] == "running":
            budget_guard(plan, root, time.perf_counter()-start_wall, cpu_time()-start_cpu)

    progress(current="loading_same_data")
    try:
        train, test = load_ensembles(plan["source_data"], mmap=True)
        tx, vx = train.configurations[..., 0], test.configurations[..., 0]
        validate_data_geometry(tx, vx)
        # Validate only evaluation metadata; neither labels nor this result are
        # passed to C training, statistics collection, PCA, decoder or R fitting.
        labels = np.concatenate((evaluation_K_labels(train.couplings, len(tx)),
                                 evaluation_K_labels(test.couplings, len(vx))))
        table, source_sha = source_C_table(plan["source_checkpoint"]) if method == "qcr_c_pca" else (None,None)
        model = construct_C(method, seed=plan["seed"], device=plan["device"], source_table=table)
        receipt = manifest(plan, method)
        receipt["source_model_tensor_sha256"] = source_sha
        save(folder / "manifest.json", receipt)
        history, c_seconds = native_C_training(tx, method, model, updates=C_UPDATES,
            seed=plan["seed"], device=plan["device"], folder=folder, progress=progress)
        save(folder / "native_C_history.json", history)
        # Keep TRAIN and TEST distinct even during collection; never pool physical chains.
        train_stats, tc = collect_statistics(tx, method, model, draws=plan["C_draws"],
            seed=plan["seed"] + 1000, device=plan["device"], source_table=table, progress=progress)
        test_stats, vc = collect_statistics(vx, method, model, draws=plan["C_draws"],
            seed=plan["seed"] + 2000, device=plan["device"], source_table=table, progress=progress)
        statistics = {key:np.concatenate((train_stats[key],test_stats[key])) for key in train_stats}
        progress(current="TRAIN_only_PCA_and_affine_fit")
        pca = fit_train_pca(train_stats["f512"], dimension=5)
        g = pca.transform(statistics["f512"])
        decoder = fit_affine_readout(g[:len(tx)], train_stats["p16"], split="train")
        save(folder / "readout.json", dict(PCA=pca.to_dict(), affine=decoder.to_dict()))
        flow, r_history, r_seconds = train_R(g[:len(tx)], updates=R_UPDATES, seed=plan["seed"],
            device=plan["device"], folder=folder, progress=progress)
        history += r_history
        save(folder / "history_progress.json", history)
        progress(current="evaluation_and_full5D_roots")
        arrays, metrics = evaluate(flow, g, decoder, statistics, len(tx), device=plan["device"])
        # The preflighted metadata is used only by post-fit evaluation plots.
        arrays["labels"] = labels
        selected_fraction = np.asarray(pca.to_dict()["explained_variance_ratio"][:5])
        retained = float(selected_fraction.sum())
        arrays["input_motif_pca_selected_fraction"] = selected_fraction
        arrays["retained_input_variance_fraction"] = np.asarray(retained)
        arrays["pca_fraction"] = selected_fraction / retained if retained > 0 else np.zeros(5)
        arrays["f512"] = statistics["f512"]
        roots = find_roots(flow, g[:len(tx)])
        metrics.update(method=method, seed=plan["seed"], data_sha256=plan["data_sha256"],
            native_C_seconds=c_seconds, collection_seconds=tc+vc, common_R_seconds=r_seconds,
            retained_g_component_fraction=arrays["pca_fraction"].tolist(),
            input_motif_PCA_selected_fraction=selected_fraction.tolist(),
            retained_input_variance_fraction=retained, PCA_rank=pca.rank,
            PCA_warnings=list(pca.warnings), fixed_points=roots, thermal_nu=None,
            comparability_caution=receipt["comparability_caution"])
        np.savez_compressed(folder / "evaluation_arrays.npz", **arrays)
        save(folder / "metrics.json", metrics)
        save(folder / "fixed_points.json", dict(fixed_points=roots, thermal_nu=None,
            maximum_TRAIN_starts=24, maximum_function_evaluations_per_start=200,
            labels_used_in_search=False, physical_support_certified=False))
        progress(current="Chinese_report_from_saved_numbers")
        public = write_arm_report(folder, arrays, metrics, history, fixed_points=roots)
        public += ["manifest.json","readout.json","history.json","history_progress.json",
                   "native_C_history.json","metrics.json",
                   "fixed_points.json","evaluation_arrays.npz"]
        progress(status="complete", current=None, C_step=C_UPDATES if model is not None else 0,
                 R_step=R_UPDATES, ended_at=now(), public_files=sorted(set(public)))
        return status
    except BudgetHold as error:
        progress(status="budget_hold", error=str(error), requires_user_action=True)
        return status
    except BaseException as error:
        progress(status="failed", error=f"{type(error).__name__}: {error}", requires_user_action=True)
        raise


def run_profile(plan, root):
    """Execution-only independent pilot; never merged with formal training."""
    folder = root / "profile"
    folder.mkdir(parents=True, exist_ok=False)
    device, seed = plan["device"], plan["profile"]["seed"]
    status = dict(status="running", scientific_result=False, started_at=now(),
                  physical_device_receipt=device_receipt(plan))
    save(folder / "profile.json", status)
    started, cpu_start = time.perf_counter(), cpu_time()
    # Fail before formal work if the declared CPU analysis/render dependencies
    # or an already installed Chinese font are unavailable; never install one.
    from scipy.optimize import root as _root_dependency
    from recurrent_rg.baselines.report import write_arm_report as _report_dependency
    from scripts.report_ising_zh import select_chinese_font
    font = select_chinese_font()
    train, test = load_ensembles(plan["source_data"], mmap=True)
    tx = train.configurations[...,0]
    evaluation_K_labels(train.couplings, len(tx))
    evaluation_K_labels(test.couplings, len(test.configurations))
    table, _ = source_C_table(plan["source_checkpoint"])
    per_method, raw = {}, {}
    small = None
    for method in METHODS:
        model = construct_C(method, seed=seed, device=device, source_table=table)
        sync(device)
        cpu_before = cpu_time()
        c_history, c_seconds = native_C_training(tx, method, model,
            updates=plan["profile"]["updates"], seed=seed, device=device)
        small, collection = collect_statistics(tx[:4], method, model, draws=plan["C_draws"],
            seed=seed+1000, device=device, source_table=table)
        # Two-stage throughput extrapolation uses fixed endpoints, never a speed-driven reduced run.
        c_timing = (project_updates_with_single_startup(c_seconds, c_history,
            plan["profile"]["updates"], C_UPDATES) if model is not None else None)
        c_forecast = c_timing["projected_seconds"] if c_timing is not None else 0.
        collection_forecast = collection * 768 / 4
        raw[method] = dict(native_C_profile_seconds=c_seconds,
            four_chain_collection_seconds=collection,
            measured_cpu_seconds=cpu_time()-cpu_before,
            native_C_startup_and_steady_timing=c_timing,
            projected_native_C_seconds=c_forecast, projected_collection_seconds=collection_forecast)
        per_method[method] = c_forecast + collection_forecast
        del model
    pca_start = time.perf_counter()
    # Shape-matched synthetic statistics for TRAIN SVD cost only, not a scientific fit.
    rng = np.random.default_rng(seed)
    fake = rng.dirichlet(np.ones(512), size=(516,3,3))
    pca = fit_train_pca(fake, dimension=5)
    fake_g = pca.transform(fake)
    fit_affine_readout(fake_g, rng.dirichlet(np.ones(16), size=(516,3,3)))
    pca_seconds = time.perf_counter() - pca_start
    _, r_history, r_seconds = train_R(fake_g, updates=50, seed=seed, device=device)
    r_timing = project_updates_with_single_startup(r_seconds,r_history,50,R_UPDATES)
    r_forecast = r_timing["projected_seconds"]
    # Root/report/load reserves are included in whole-arm wall GPU accounting.
    gpu_map = {m:1.5*(seconds+r_forecast+pca_seconds)+180+30 for m,seconds in per_method.items()}
    # Two CPU threads times whole-stage wall is a deliberately conservative CPU envelope.
    cpu_map = {m:2*seconds for m,seconds in gpu_map.items()}
    status.update(status="complete", ended_at=now(), projected_gpu_seconds=sum(gpu_map.values()),
        projected_cpu_seconds=sum(cpu_map.values())+120,
        projected_gpu_seconds_by_method=gpu_map, projected_cpu_seconds_by_method=cpu_map,
        comparison_cpu_seconds=120, peak_gpu_bytes=peak_memory(device),
        profile_wall_seconds=time.perf_counter()-started, profile_cpu_seconds=cpu_time()-cpu_start,
        timing_by_method=raw, common_R_profile_50_updates_seconds=r_seconds,
        projected_common_R_seconds=r_forecast, PCA_affine_shape_matched_seconds=pca_seconds,
        common_R_startup_and_steady_timing=r_timing,
        safety_multiplier=1.5, per_arm_load_reserve_seconds=30,
        per_arm_evaluation_root_plot_reserve_seconds=180,
        existing_Chinese_font=str(font), analysis_dependencies_preflight=True,
        evaluation_scalar_label_schema_preflight=True,
        notes="Independent execution pilot only; synthetic PCA/R timing is NOT a scientific baseline result",
        public_files=[])
    save(folder / "profile.json", status)
    return status


def project_updates_with_single_startup(total_seconds, history, measured_updates, target_updates):
    """Count the measured cold first update ONCE, not once per pilot block.

    The remaining pilot updates define steady throughput. This fixes execution
    accounting only: no C/R update, optimizer, loss, sample or endpoint changes.
    A separate 1.5 safety multiplier and per-arm reserves still apply upstream.
    """
    if (type(measured_updates) is not int or measured_updates < 2
            or type(target_updates) is not int or target_updates < measured_updates
            or isinstance(total_seconds,bool) or not math.isfinite(total_seconds)
            or total_seconds < 0):
        raise ValueError("Need finite timing, >=2 measured updates and a fixed longer endpoint")
    first_rows = [row for row in history if row.get("step") == 1]
    last_rows = [row for row in history if row.get("step") == measured_updates]
    if len(first_rows) != 1 or len(last_rows) != 1:
        raise ValueError("Pilot must save exactly its first and final update times")
    first = first_rows[0].get("elapsed_seconds")
    last = last_rows[0].get("elapsed_seconds")
    if (any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x)
            for x in (first,last)) or not 0 <= first <= last <= total_seconds):
        raise ValueError("Pilot update times must be finite, monotone and within total wall time")
    # Include post-last-update synchronization/printing in the steady envelope.
    steady = (total_seconds-first)/(measured_updates-1)
    projection = first+(target_updates-1)*steady
    return dict(first_update_including_startup_seconds=first,
        measured_total_seconds=total_seconds, measured_updates=measured_updates,
        steady_seconds_per_update=steady, target_updates=target_updates,
        projected_seconds=projection,
        formula="first_update + (target_updates-1)*(total-first_update)/(measured_updates-1)",
        startup_charged_once_per_arm=True, scientific_protocol_changed=False)


def compare(plan, root):
    from recurrent_rg.baselines.report import write_comparison_report
    folder = root / "comparison"
    folder.mkdir(parents=True, exist_ok=False)
    summaries = {}
    for method in METHODS:
        arm = root / "arms" / method
        if read(arm / "status.json")["status"] != "complete":
            raise ValueError("Cannot compare an unfinished baseline")
        summaries[method] = read(arm / "metrics.json")
    original_reference = read(plan["source_metrics"])
    reference = reference_indicators(original_reference)
    save(folder / "comparison.json", dict(status="complete", summaries=summaries,
        reference_metrics=original_reference, reference_plot_adapter=reference,
        adaptations=True, labels_used_in_training=False,
        single_seed=True, thermal_nu=None,
        caveat="No common latent coordinate across C/PCA. Physical motif target also depends on C; low closure alone is not a phase-structure or noncollapse certificate."))
    public = write_comparison_report(folder, summaries, reference=reference)
    status = dict(status="complete", ended_at=now(), public_files=sorted(set(public+["comparison.json"])))
    save(folder / "status.json", status)
    return status


def reference_indicators(metrics):
    """Adapt already saved reference endpoints, without recomputing predictions.

    The historical report has no matched 16-motif autonomous decoder error;
    leave it absent, not zero or an unrelated reconstruction-only metric.
    """
    adapted = dict(kind="saved_QCR_reference_indicator_adapter", source_metrics=metrics,
        common_coordinate_comparison=False, physical_motif_prediction_not_available=True)
    for split in ("train", "test"):
        row = metrics.get("closure",{}).get(split,{})
        rollout = row.get("rollout",{})
        if rollout.get("nrmse"):
            adapted[split] = dict(latent=dict(rollout_nrmse=rollout["nrmse"][-1],
                rollout_rmse=rollout["rmse"][-1]),motif16={})
    return adapted


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=("profile","arm","compare"), required=True)
    parser.add_argument("--method", choices=METHODS)
    args = parser.parse_args(argv)
    if (args.stage == "arm") != (args.method is not None):
        parser.error("Only an arm stage takes --method")
    if args.plan.resolve() != (args.root / "plan.json").resolve():
        raise ValueError("Only the root's exact frozen plan is accepted")
    from scripts.run_ising_baseline_comparison import verify
    plan, _ = verify(args.root)
    configure("cpu" if args.stage == "compare" else plan["device"],
              threads=1 if args.stage == "compare" else 2,
              gpu_peak_bytes=plan["resource_policy"]["gpu_peak_bytes"])
    try:
        if args.stage == "profile":
            result = run_profile(plan, args.root)
        elif args.stage == "arm":
            result = run_arm(plan, args.root, args.method)
        else:
            result = compare(plan, args.root)
    except BaseException as error:
        if args.stage != "arm":
            folder = args.root / ("profile" if args.stage == "profile" else "comparison")
            if folder.is_dir():
                file = folder / ("profile.json" if args.stage == "profile" else "status.json")
                status = read(file) if file.exists() else {}
                status.update(status="failed", error=f"{type(error).__name__}: {error}",
                    requires_user_action=True, ended_at=now())
                save(file,status)
        raise
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
