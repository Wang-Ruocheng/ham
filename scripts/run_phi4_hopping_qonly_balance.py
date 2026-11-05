"""Explicit paired Q-only scale-balancing diagnostic on one frozen phi4 run.

Not a default recipe, physical sampler, QCR continuation, or fixed-point test.
Only ``--execute`` imports the pinned model and reads the physical dataset.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch


SOURCE = Path("/mnt/nas-new/home/yange/ham/results/phi4_hopping_b2_L128_g1_seed91_gpu3_vm225_20261101_integrator_refined_v1")
SNAPSHOT = Path("/mnt/nas-new/home/yange/ham/run_code/phi4_hopping_b2_L128_seed91_20261101_integrator_refined_v1")
PINS = {
    "training/training_step_3000.pt": "1aec98b9d9c9d562ca32b6bef972cb79fa46ccc5f06819bb752129709dcf50d7",
    "data/manifest.json": "3d2023b401b15e20a6dca6c6b602959c237a030ff0e2f8f89676a7ca9ed30fa5",
    "training/evaluation_arrays.npz": "1af96e4b1ad9e4d4a516d8552be80f4c924ba3f53677ad4c1b4c58127d91ae4f",
    "training/summary.json": "022357928cbc7feba6bae112521cce826a5c3875d274c9e6fe33d35eb17be9cd",
}
ARMS = ("A_absolute", "B_relative")
STEPS, SYSTEM_BATCH, SAMPLE_BATCH, LR, SEED = 300, 8, 32, .001, 91
GPU_SECONDS, CPU_SECONDS, MAX_BYTES, MAX_CUDA_BYTES = 1800., 7200., 3 * 1024**3, 8 * 1024**3
SIZES = (128, 64, 32, 16)


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
    """Nested tensor fingerprint, exactly the hopping trainer's convention."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous().numpy()
        item = hashlib.sha256()
        for part in (str(value.dtype).encode(), str(value.shape).encode(), value.tobytes()):
            item.update(part)
        digest.update(name.encode())
        digest.update(item.hexdigest().encode())
    return digest.hexdigest()


def fit_scale_weights(target_per_field, reconstructed_source):
    """TRAIN-only, scalar per-scale weights; all descriptor bins are retained.

    Contiguous-block fluctuations are a noise heuristic, not an ESS estimate or
    an independent-sample/equilibration certificate. All constants are frozen.
    """
    p = np.asarray(target_per_field, dtype=np.float64)
    decoded = np.asarray(reconstructed_source, dtype=np.float64)
    if p.ndim != 4 or p.shape[1] != 4 or p.shape[2] % 4 or decoded.shape != p.shape[:2] + p.shape[3:]:
        raise ValueError("Expected TRAIN [chain,4,fields,descriptor] and source reconstruction")
    if len(p) < 2 or not np.isfinite(p).all() or not np.isfinite(decoded).all():
        raise ValueError("Finite TRAIN descriptors required")
    target = p.mean(axis=2)
    variance = ((target - target.mean(axis=0))**2).sum(axis=-1).mean(axis=0)
    blocks = p.reshape(len(p), 4, 4, p.shape[2] // 4, p.shape[-1]).mean(axis=3)
    noise = blocks.var(axis=2, ddof=1).sum(axis=-1).mean(axis=0) / 4
    floor = max(float(variance.max()) / 20., 1e-12)
    denominator = np.maximum.reduce([variance, noise, np.full(4, floor)])
    residual = ((decoded - target)**2).sum(axis=-1).mean(axis=0)
    if not np.isfinite(residual).all() or residual.mean() <= 0:
        raise ValueError("A positive finite initial reconstruction is required")
    alpha = float(residual.mean() / (residual / denominator).mean())
    weights = alpha / denominator
    return dict(variance=variance, noise=noise, denominator=denominator, weights=weights,
        alpha=alpha, common_floor=floor, source_reconstruction_by_scale=residual,
        target_mean=target.mean(axis=0), target_std=np.sqrt(variance / p.shape[-1]))


def q_objective(q, levels, per_field_targets, weights):
    """Only Q and detached fields/targets enter; no C, R, label, split or cell."""
    if len(levels) != 4 or per_field_targets.ndim != 4 or per_field_targets.shape[1] != 4:
        raise ValueError("Four fixed scales required")
    reconstruction, subset = [], []
    for k, level in enumerate(levels):
        field = level.detach()
        half = field.shape[1] // 2
        if half < 1 or field.shape[1] != 2 * half:
            raise ValueError("Equal, nonempty disjoint field halves required")
        first, second = q(field[:, :half]), q(field[:, half:])
        direct = .5 * (first + second)
        target = per_field_targets[:, k].detach().mean(dim=1)
        reconstruction.append((q.decode(direct) - target).square().sum(-1).mean())
        subset.append((first - second).square().sum(-1).mean())
    rec, consistency = torch.stack(reconstruction), torch.stack(subset)
    weight = torch.as_tensor(weights, device=rec.device, dtype=rec.dtype)
    if weight.shape != (4,) or not bool(torch.isfinite(weight).all()) or bool((weight <= 0).any()):
        raise ValueError("Four positive finite scalar reconstruction weights required")
    total = 10. * (rec * weight).mean() + 20. * consistency[0] + 5. * consistency[1:].mean()
    return total, rec, consistency


def batch_schedule(train_count, fields, *, generator, system_batch=SYSTEM_BATCH, sample_batch=SAMPLE_BATCH):
    if train_count < system_batch or fields < sample_batch or sample_batch % 2:
        raise ValueError("Insufficient TRAIN chains/fields for fixed distinct sampling")
    return (torch.randperm(train_count, generator=generator)[:system_batch],
            torch.randperm(fields, generator=generator)[:sample_batch])


def optimizer_counts(optimizer, q):
    values = [int(optimizer.state[parameter]["step"].item()) for parameter in q.parameters()]
    return dict(parameter_tensors=len(values), minimum=min(values), maximum=max(values))


def update(q, optimizer, levels, targets, weights):
    optimizer.zero_grad(set_to_none=True)
    total, _, _ = q_objective(q, levels, targets, weights)
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("Nonfinite Q objective")
    total.backward()
    if any(parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()) for parameter in q.parameters()):
        raise FloatingPointError("Missing/nonfinite Q gradient")
    optimizer.step()
    q.normalize_decoder_()
    if any(not bool(torch.isfinite(parameter).all()) for parameter in q.parameters()):
        raise FloatingPointError("Nonfinite Q parameter")
    optimizer.zero_grad(set_to_none=True)
    return float(total.detach())


def host_resources():
    return dict(cpus=os.cpu_count(), load1=os.getloadavg()[0],
        mem_available_bytes=int(Path("/proc/meminfo").read_text().split("MemAvailable:")[1].split()[0]) * 1024,
        io_full_avg10=float(Path("/proc/pressure/io").read_text().split("full avg10=")[1].split()[0]))


def resource_ok(value):
    return (value["mem_available_bytes"] >= 36 * 1024**3
        and value["load1"] <= 1.25 * value["cpus"] and value["io_full_avg10"] < 20)


class BudgetHold(RuntimeError):
    pass


def gpu_snapshot(index, expected_uuid):
    def smi(arguments):
        return subprocess.check_output(["nvidia-smi", *arguments], text=True, timeout=15)
    rows = [row.strip().split(", ") for row in smi([
        "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"]).strip().splitlines()]
    matching = [row for row in rows if int(row[0]) == index]
    if len(matching) != 1 or matching[0][1] != expected_uuid:
        raise RuntimeError("Pinned physical GPU index/UUID mismatch; no migration")
    processes = [row.strip().split(", ") for row in smi([
        "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"]).strip().splitlines() if row.strip()]
    occupied = [int(row[1]) for row in processes if row[0] == expected_uuid]
    value = dict(index=index, uuid=expected_uuid, free_mib=float(matching[0][2]), compute_pids=occupied,
        host=host_resources(), monotonic=time.monotonic())
    if occupied or value["free_mib"] < 12 * 1024 or not resource_ok(value["host"]):
        raise BudgetHold("Resource admission did not pass; do not interfere with other users")
    return value


def complete(value, name):
    if (value.get("status") != "complete" or value.get("workflow_complete") is not True
            or value.get("returncode") != 0 or value.get("error") or value.get("requires_user_action") is not False):
        raise ValueError(f"{name} is not cleanly complete")


def bind_sources():
    job, plan = read(SOURCE / "job.json"), read(SOURCE / "plan.json")
    status, manifest, summary = [read(SOURCE / "training" / name) for name in ("status.json", "manifest.json", "summary.json")]
    complete(job, "Source workflow")
    complete(status, "Source training")
    if any(stage.get("status") != "complete" or stage.get("returncode") != 0 for stage in job["stages"]):
        raise ValueError("Source actual execution stage incomplete")
    if (job.get("plan_sha256") != sha(SOURCE / "plan.json") or summary.get("actual_steps") != 3000
            or manifest.get("theory_dim") != 1 or manifest.get("block_size") != 2
            or manifest.get("lattice_sizes") != list(SIZES) or manifest.get("hidden") != 96
            or manifest.get("labels_used_in_training") is not False or summary.get("labels_used_in_training") is not False
            or Path(manifest["data_root"]).resolve() != (SOURCE / "data").resolve()):
        raise ValueError("Unexpected formal source endpoint/geometry/label policy")
    protected = {}
    for relative, expected in PINS.items():
        path = SOURCE / relative
        if sha(path) != expected:
            raise ValueError(f"Source SHA mismatch: {relative}")
        protected[path] = expected
    for path in (SOURCE / "job.json", SOURCE / "plan.json", SOURCE / "training/status.json",
                 SOURCE / "training/manifest.json", SOURCE / "data/status.json", SOURCE / "data/plan.json"):
        protected[path] = sha(path)
    for pins in (plan["source_sha256"], manifest["source_sha256"]):
        for name, digest in pins.items():
            path = Path(name) if Path(name).is_absolute() else Path(plan["code_root"]) / name
            if sha(path) != digest:
                raise ValueError(f"Frozen source pin changed: {path}")
            protected[path] = digest
    public = {}
    for row in status["public_files"]:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe source public whitelist")
        path = SOURCE / "training" / relative
        if sha(path) != row["sha256"]:
            raise ValueError("Source public whitelist SHA mismatch")
        protected[path] = row["sha256"]
        public[str(relative)] = row["sha256"]
    if not {"manifest.json", "summary.json", "evaluation_arrays.npz"} <= set(public):
        raise ValueError("Required source public descriptors absent")
    checkpoint = torch.load(SOURCE / "training/training_step_3000.pt", map_location="cpu", weights_only=False)
    if checkpoint.get("step") != 3000 or checkpoint.get("manifest") != manifest:
        raise ValueError("Source checkpoint endpoint binding differs")
    state = checkpoint["model"]
    for name in ("Q", "C", "R"):
        subset = {key[len(name)+1:]: value for key, value in state.items() if key.startswith(name + ".")}
        if state_sha(subset) != summary["final_module_sha256"][name]:
            raise ValueError(f"Source {name} content fingerprint differs")
    with np.load(SOURCE / "training/evaluation_arrays.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in ("direct_g", "direct_full_g", "target_features",
            "reconstructed_features", "labels", "splits", "cell_ids", "flow_grid", "flow_next")}
    if arrays["direct_g"].shape != (60, 4, 1) or arrays["target_features"].shape != (60, 4, 625):
        raise ValueError("Unexpected formal public descriptor axes")
    return state, arrays, protected, manifest, summary


def snapshot_imports():
    sys.path.insert(0, str(SNAPSHOT))
    names = ("recurrent_rg.phi4_aligned_qcr", "recurrent_rg.data.phi4_hopping_loader")
    modules = [importlib.import_module(name) for name in names]
    if any(SNAPSHOT not in Path(module.__file__).resolve().parents for module in modules):
        raise RuntimeError("A model/loader import did not resolve inside the frozen snapshot")
    return modules[0].AlignedPhi4QCR, modules[0].soft_plaquette_features, modules[1].load_dataset


@torch.no_grad()
def evaluate_q(q, fields, *, budget_check=lambda: None):
    first, second, direct, pooled, decoded = [], [], [], [], []
    device = next(q.parameters()).device
    q.eval()
    for chain in range(len(fields[0])):
        a, b, whole = [], [], []
        for level in fields:
            x = level[chain:chain+1].to(device)
            half = x.shape[1] // 2
            a.append(q(x[:, :half]).cpu().numpy()[0])
            b.append(q(x[:, half:]).cpu().numpy()[0])
            whole.append(q(x).cpu().numpy()[0])
            budget_check()
        g1, g2 = np.asarray(a), np.asarray(b)
        g = .5 * (g1 + g2)
        first.append(g1); second.append(g2); direct.append(g); pooled.append(whole)
        decoded.append(q.decode(torch.as_tensor(g, device=device)).cpu().numpy())
    return dict(g_first=np.asarray(first), g_second=np.asarray(second), direct_g=np.asarray(direct),
        direct_full_g=np.asarray(pooled), decoded=np.asarray(decoded))


@torch.no_grad()
def probe(q, fields, targets, weights, budget_check):
    """Fixed TRAIN-only first32 fields, all40 TRAIN physical chains."""
    rec, subset = [], []
    device = next(q.parameters()).device
    for chain in range(len(fields[0])):
        levels = [level[chain:chain+1, :SAMPLE_BATCH].to(device) for level in fields]
        _, r, s = q_objective(q, levels, targets[chain:chain+1, :, :SAMPLE_BATCH].to(device), weights)
        rec.append(r.cpu().numpy()); subset.append(s.cpu().numpy())
        budget_check()
    r, s = np.asarray(rec).mean(axis=0), np.asarray(subset).mean(axis=0)
    return dict(q_total=float(10 * np.mean(weights * r) + 20 * s[0] + 5 * s[1:].mean()),
        reconstruction_raw=float(r.mean()), reconstruction_relative=float(np.mean(weights * r)),
        reconstruction_by_scale=r.tolist(), subset_micro=float(s[0]), subset_coarse=float(s[1:].mean()),
        subset_by_scale=s.tolist())


def calculate_metrics(target, decoded, train_std, train_mean):
    rmse = np.sqrt(((decoded - target)**2).mean(axis=(0, 2)))
    constant = np.sqrt(((train_mean[None] - target)**2).mean(axis=(0, 2)))
    return dict(rmse=rmse.tolist(), nrmse=(rmse / train_std).tolist(),
        constant_train_mean_rmse=constant.tolist(), constant_train_mean_nrmse=(constant / train_std).tolist())


def run(output, gpu_index, expected_gpu_uuid):
    wall = time.monotonic()
    output = Path(output).resolve()
    if output.exists() or output == SOURCE or SOURCE in output.parents or output == SNAPSHOT or SNAPSHOT in output.parents:
        raise ValueError("Require fresh independent output; no overwrite/resume")
    if os.uname().nodename != "thinklab-105-225" or os.getpriority(os.PRIO_PROCESS, 0) < 10:
        raise RuntimeError("Wrong host or priority; no fallback")
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != expected_gpu_uuid
            or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID" or torch.cuda.is_initialized()):
        raise RuntimeError("Require the pinned UUID mask/PCI_BUS_ID before CUDA initialization")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output.mkdir(parents=True)
    job = dict(status="running", workflow_complete=False, current="resource_admission", actual_steps=0,
        returncode=None, error=None, requires_user_action=False, pid=os.getpid(),
        source_Q_steps=3000, additional_Q_steps_per_arm=STEPS, optimizer_updates=dict(C=0, R=0))
    def status(**updates):
        job.update(updates, elapsed_seconds=time.monotonic()-wall, cpu_seconds=time.process_time())
        write(output / "job.json", job); write(output / "status.json", job)
        print(json.dumps(job, ensure_ascii=False), flush=True)
    def check(reserve=0., *, scan_directory=False):
        if time.monotonic()-wall+reserve > GPU_SECONDS or time.process_time()+reserve > CPU_SECONDS:
            raise BudgetHold("Conservative elapsed GPU reservation/CPU budget exhausted; no shortened training")
        if scan_directory and sum(path.stat().st_size for path in output.rglob("*") if path.is_file()) > MAX_BYTES:
            raise BudgetHold("New diagnostic directory exceeds 3 GiB")
        if torch.cuda.is_initialized() and torch.cuda.max_memory_allocated() > MAX_CUDA_BYTES:
            raise BudgetHold("Own peak CUDA allocation exceeds 8 GiB")
    status()
    try:
        admission = []
        while len(admission) < 2:
            try:
                admission.append(gpu_snapshot(gpu_index, expected_gpu_uuid))
            except BudgetHold:
                admission = []
                status(current="waiting_resources",resource_wait=True)
                time.sleep(30.)
                check()
                continue
            if len(admission) == 1:
                time.sleep(10.)
                check()
        if admission[1]["monotonic"] - admission[0]["monotonic"] < 10:
            raise RuntimeError("Two resource checks must be separated by at least 10 seconds")
        status(current="source_verification")
        source_state, formal, protected, source_manifest, source_summary = bind_sources()
        protected[Path(__file__).resolve()] = sha(__file__)
        report_path = Path(__file__).with_name("report_phi4_hopping_qonly_balance.py")
        protected[report_path] = sha(report_path)
        AlignedPhi4QCR, features, load_dataset = snapshot_imports()
        check()
        device = torch.device("cuda:0")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Expected one visible fixed GPU")
        torch.cuda.set_per_process_memory_fraction(
            MAX_CUDA_BYTES / torch.cuda.get_device_properties(device).total_memory, device)
        torch.cuda.reset_peak_memory_stats(device)
        dataset = load_dataset(SOURCE / "data", block_size=2, depth=3)
        if (dataset.configurations.shape != (60, 128, 128, 128, 1)
                or len(dataset.train_indices) != 40 or len(dataset.test_indices) != 20
                or not np.array_equal(dataset.splits, formal["splits"])
                or not np.array_equal(dataset.cell_ids, formal["cell_ids"])
                or not np.array_equal(dataset.labels, formal["labels"])):
            raise ValueError("Unexpected physical chain/cache split or order")
        for receipt in dataset.receipts:
            protected[SOURCE / "data" / receipt["path"]] = receipt["sha256"]
            protected[SOURCE / "data" / receipt["sidecar"]] = receipt["sidecar_sha256"]
        model = AlignedPhi4QCR(dataset.scale, hidden=96, theory_dim=1, local_features=12,
            patch_chunk_size=262144, block_size=2, steps=3).to(device)
        model.load_state_dict(source_state, strict=True)
        del source_state
        for name in ("Q", "C", "R"):
            if state_sha(getattr(model, name).state_dict()) != source_summary["final_module_sha256"][name]:
                raise ValueError(f"Instantiated source {name} fingerprint differs")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        check()
        plan = dict(schema_version=5, experiment="explicit_Q_only_scalar_scale_balance_v1", source=str(SOURCE),
            snapshot=str(SNAPSHOT), source_checkpoint_sha256=PINS["training/training_step_3000.pt"],
            source_Q_steps=3000, extra_Q_steps_per_arm=STEPS, arms=list(ARMS),
            architecture="unchanged full Q local MLP/head/shared affine625 decoder, dg1",
            frozen_modules=["C", "R"], C_updates=0, R_updates=0, model_labels_used=False,
            batch=dict(physical_chains=SYSTEM_BATCH, distinct_saved_fields=SAMPLE_BATCH, half_fields=16,
                paired_arms=True, seed=SEED+101), optimizer=dict(name="fresh Adam", learning_rate=LR),
            reconstruction="A:10 mean scale L2; B:10 mean scalar_weight[k] L2; all625 bins retained",
            consistency="20 microscopic subset L2 + 5 mean coarse subset L2, unchanged",
            cache_policy="one original evaluation CUDA C stream seed20261031, chain1x128 in original order; no additional C trajectories",
            qualification="R-only source flow is unchanged and is NOT fitted to the new Q coordinates; no fixed-point/exponent inference",
            changed_sampling="fixed C cache;8chains x32fields instead of source32x64; not exact QCR continuation",
            budget=dict(gpu_reservation_seconds=GPU_SECONDS, cpu_seconds=CPU_SECONDS,
                directory_bytes=MAX_BYTES, own_cuda_peak_bytes=MAX_CUDA_BYTES),
            admission=admission, physical_gpu_index=gpu_index, expected_gpu_uuid=expected_gpu_uuid,
            source_module_sha256=source_summary["final_module_sha256"], source_protected_sha256={str(p):v for p,v in protected.items()},
            warnings=dataset.warning_count, equilibrium_certified=False, thermal_nu=None,
            training_started=False, overwrite=False, resume=False)
        write(output / "plan.json", plan)
        status(current="fixed_C_cache")
        frozen = [dataset.configurations] + [torch.empty((60,128,size,size,1),dtype=torch.float64) for size in SIZES[1:]]
        target_per_field = torch.empty((60,4,128,625),dtype=torch.float64)
        generator = torch.Generator(device=device).manual_seed(20261031)
        with torch.no_grad():
            for chain in range(60):
                levels = model.coarse_grain(dataset.configurations[chain:chain+1].to(device),generator=generator)
                for k, level in enumerate(levels):
                    if k:
                        frozen[k][chain].copy_(level[0].cpu())
                    # Treat each field as one ensemble; feature averaging is linear in fields.
                    target_per_field[chain,k].copy_(features(level.transpose(0,1),model.scale,
                        chunk_size=262144).cpu())
                    check()
                status(current="fixed_C_cache", cached_chains=chain+1)
        source_evaluation = evaluate_q(model.Q, frozen, budget_check=check)
        cached_target = target_per_field.mean(dim=2).numpy()
        differences = {name:float(np.max(np.abs(source_evaluation[name]-formal[formal_name])))
            for name,formal_name in (("direct_g","direct_g"),("direct_full_g","direct_full_g"),("decoded","reconstructed_features"))}
        differences["target_features"] = float(np.max(np.abs(cached_target-formal["target_features"])))
        if max(differences.values()) > 1e-10:
            raise RuntimeError(f"Frozen C/source Q reproduction mismatch: {differences}")
        torch.save(dict(fields=frozen, target_per_field=target_per_field,
            semantics="private fixed C trajectory; original correlated physical fields"), output / "frozen_fields.pt")
        check(scan_directory=True)
        train = dataset.train_indices
        balanced = fit_scale_weights(target_per_field[train].numpy(),source_evaluation["decoded"][train])
        weights = {ARMS[0]:np.ones(4),ARMS[1]:balanced["weights"]}
        receipt = {key:(value.tolist() if isinstance(value,np.ndarray) else value) for key,value in balanced.items()}
        receipt.update(source_reproduction_maxabs=differences, cache_sha256=sha(output/"frozen_fields.pt"),
            cache_chain_count=60, fields_per_chain=128, C_draws_per_field_trajectory=1,
            source_module_sha256=source_summary["final_module_sha256"],
            noise_qualification="four contiguous32-field blocks are correlated; heuristic only, not ESS/equilibration certification",
            labels_used=False, fit_train_chains=40, fit_test_chains=0)
        write(output/"cache_receipt.json",receipt)
        del dataset.configurations
        train_fields = [level[train] for level in frozen]
        train_targets = target_per_field[train]
        def get_batch(indices, samples):
            return ([level[indices][:,samples].to(device) for level in train_fields],
                train_targets[indices][:,:,samples].to(device))
        status(current="bounded_benchmark")
        benchmark_models={arm:copy.deepcopy(model.Q).requires_grad_(True) for arm in ARMS}
        benchmark_optim={arm:torch.optim.Adam(benchmark_models[arm].parameters(),lr=LR) for arm in ARMS}
        bench_rng=torch.Generator().manual_seed(SEED+101)
        times=[]
        for iteration in range(4):
            torch.cuda.synchronize(); began=time.monotonic()
            levels,targets=get_batch(*batch_schedule(40,128,generator=bench_rng))
            for arm in ARMS:
                update(benchmark_models[arm],benchmark_optim[arm],levels,targets,weights[arm])
            torch.cuda.synchronize()
            if iteration:
                times.append(time.monotonic()-began)
            check()
        del benchmark_models,benchmark_optim,levels,targets
        torch.cuda.empty_cache()
        projected=1.5*max(times)*(STEPS+100)+120.
        benchmark=dict(warmup_paired_updates=1,timed_paired_updates=3,paired_update_seconds=times,
            conservative_remaining_seconds=projected,elapsed_reservation_seconds=time.monotonic()-wall,
            cpu_seconds=time.process_time(),formal_updates=0,formal_optimizer_restored=False)
        write(output/"benchmark.json",benchmark)
        check(projected)
        models={arm:copy.deepcopy(model.Q).requires_grad_(True) for arm in ARMS}
        optimizers={arm:torch.optim.Adam(models[arm].parameters(),lr=LR) for arm in ARMS}
        if any(state_sha(q.state_dict())!=source_summary["final_module_sha256"]["Q"] for q in models.values()):
            raise RuntimeError("Formal arm did not start from identical source Q")
        history=[]
        def log(step):
            row=dict(step=step,arms={arm:probe(models[arm],train_fields,train_targets,weights[arm],check) for arm in ARMS})
            # Both common reconstruction metrics use the same fixed TRAIN weights.
            for arm in ARMS:
                row["arms"][arm]["reconstruction_relative"]=float(np.mean(
                    balanced["weights"]*np.asarray(row["arms"][arm]["reconstruction_by_scale"])))
            history.append(row);write(output/"history.json",history)
            check(scan_directory=True)
        log(0)
        plan["training_started"]=True
        write(output/"plan.json",plan)
        status(current="paired_Q_training",training_started=True)
        schedule_hash=hashlib.sha256(); rng=torch.Generator().manual_seed(SEED+101)
        for step in range(1,STEPS+1):
            indices,samples=batch_schedule(40,128,generator=rng)
            schedule_hash.update(indices.numpy().tobytes());schedule_hash.update(samples.numpy().tobytes())
            levels,targets=get_batch(indices,samples)
            for arm in ARMS:
                update(models[arm],optimizers[arm],levels,targets,weights[arm])
            del levels,targets
            check(120.)
            if step%50==0:
                log(step);status(actual_steps=step)
        status(current="fixed_endpoint_evaluation",actual_steps=STEPS)
        arrays=dict(target_features=cached_target,train_target_mean=balanced["target_mean"],
            train_target_std=balanced["target_std"],metric_denominators=balanced["denominator"],
            relative_weights=balanced["weights"],source_flow_grid=formal["flow_grid"],source_flow_next=formal["flow_next"])
        for name,value in source_evaluation.items():arrays["source_"+name]=value
        for arm in ARMS:
            evaluated=evaluate_q(models[arm],frozen,budget_check=check)
            for name,value in evaluated.items():arrays[arm+"_"+name]=value
            torch.save(dict(source_Q_steps=3000,additional_Q_steps=STEPS,Q=models[arm].state_dict()),output/(arm+"_Q_final.pt"))
        # Coupling labels enter saved evaluation/report arrays only, after all optimization.
        arrays.update({name:formal[name] for name in ("labels","splits","cell_ids")})
        if any(value.dtype.kind in "fc" and not np.isfinite(value).all() for value in arrays.values()):
            raise FloatingPointError("Nonfinite endpoint arrays")
        np.savez_compressed(output/"evaluation_arrays.npz",**arrays)
        metrics={}
        for split in ("train","test"):
            mask=formal["splits"]==split
            metrics[split]={name:calculate_metrics(cached_target[mask],arrays[name+"_decoded"][mask],
                balanced["target_std"],balanced["target_mean"]) for name in ("source",*ARMS)}
        write(output/"metrics.json",metrics)
        counts={arm:optimizer_counts(optimizers[arm],models[arm]) for arm in ARMS}
        if any(value!={"parameter_tensors":12,"minimum":STEPS,"maximum":STEPS} for value in counts.values()):
            raise RuntimeError("Not all twelve Q parameter tensors have exactly300 updates")
        frozen_ok=all(state_sha(getattr(model,name).state_dict())==source_summary["final_module_sha256"][name]
            and all(parameter.grad is None for parameter in getattr(model,name).parameters()) for name in ("C","R"))
        if not frozen_ok:raise RuntimeError("Frozen C/R state or gradients changed")
        if any(parameter.grad is not None for q in models.values() for parameter in q.parameters()):
            raise RuntimeError("Final Q gradients were not cleared")
        if any(sha(path)!=digest for path,digest in protected.items()):
            raise RuntimeError("Protected source/model/data/code bytes changed before report")
        if any(state_sha(getattr(model,name).state_dict())!=source_summary["final_module_sha256"][name] for name in ("Q","C","R")):
            raise RuntimeError("Protected source model changed before report")
        summary=dict(status="complete",actual_steps=STEPS,source_Q_steps=3000,additional_Q_steps_per_arm=STEPS,
            optimizer_parameter_steps=counts,optimizer_updates=dict(Q_per_arm=STEPS,C=0,R=0),
            final_Q_sha256={arm:state_sha(models[arm].state_dict()) for arm in ARMS},
            source_module_sha256=source_summary["final_module_sha256"],frozen_C_R_unchanged=True,
            labels_used_in_training=False,test_used_in_training=False,model_selection=False,
            early_stopping=False,all_final_parameter_gradients_none=True,
            paired_batch_schedule_sha256=schedule_hash.hexdigest(),same_schedule_both_arms=True,
            warnings=source_manifest["data_warning_count"],equilibrium_certified=False,
            cache_qualification=plan["cache_policy"],changed_sampling=plan["changed_sampling"],
            noise_qualification=receipt["noise_qualification"],physical_critical_point_certified=False,
            thermal_nu=None,R_flow_unchanged_not_refitted=True,threads=torch.get_num_threads(),
            nice=os.getpriority(os.PRIO_PROCESS,0),expected_gpu_uuid=expected_gpu_uuid,
            own_peak_cuda_bytes=torch.cuda.max_memory_allocated(),
            elapsed_reservation_seconds_before_report=time.monotonic()-wall,cpu_seconds_before_report=time.process_time(),
            source_files_unchanged_verified=True)
        write(output/"summary.json",summary)
        status(current="array_only_report")
        sys.path.insert(0,str(Path(__file__).parent))
        report=importlib.import_module("report_phi4_hopping_qonly_balance")
        public_report=report.render_report(output)
        check(scan_directory=True)
        if any(sha(path)!=digest for path,digest in protected.items()):
            raise RuntimeError("Protected source/model/data/code bytes changed")
        if any(state_sha(getattr(model,name).state_dict())!=source_summary["final_module_sha256"][name] for name in ("Q","C","R")):
            raise RuntimeError("Protected source model changed")
        # The report pins summary.json. Keep those exact bytes immutable after
        # rendering; actual end-to-end timing belongs to a separate receipt.
        execution=dict(status="complete",source_files_unchanged_verified_after_report=True,
            elapsed_reservation_seconds=time.monotonic()-wall,cpu_seconds=time.process_time(),
            own_peak_cuda_bytes=torch.cuda.max_memory_allocated(),
            report_summary_sha256=sha(output/"summary.json"),returncode=0)
        write(output/"execution.json",execution)
        status(status="complete",workflow_complete=True,returncode=0,current="complete",error=None,requires_user_action=False,
            source_files_unchanged_verified=True,actual_steps=STEPS)
        public=["plan.json","cache_receipt.json","benchmark.json","history.json","evaluation_arrays.npz","execution.json",
            "metrics.json","summary.json","job.json","status.json",*public_report]
        for relative in public:
            path=Path(relative)
            if path.is_absolute() or ".." in path.parts or path.suffix not in (".json",".png",".npz"):
                raise ValueError("Report returned an unsafe/nonpublic file")
        write(output/"public_files.json",dict(files=[dict(path=name,sha256=sha(output/name),bytes=(output/name).stat().st_size) for name in public]))
        return 0
    except Exception as exc:
        hold=isinstance(exc,BudgetHold)
        status(status="budget_hold" if hold else "failed",workflow_complete=False,
            returncode=2 if hold else 1,error=f"{type(exc).__name__}: {exc}",requires_user_action=True,
            traceback=None if hold else traceback.format_exc())
        return 2 if hold else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--gpu-index",type=int,required=True)
    parser.add_argument("--expected-gpu-uuid",required=True)
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(execution_requested=False,source=str(SOURCE),output=str(args.output),arms=ARMS,
            extra_Q_updates_per_arm=STEPS,system_batch=SYSTEM_BATCH,sample_batch=SAMPLE_BATCH,
            frozen_modules=["C","R"],budgets=dict(gpu_seconds=GPU_SECONDS,cpu_seconds=CPU_SECONDS,directory_bytes=MAX_BYTES)),indent=2))
        return 0
    return run(args.output,args.gpu_index,args.expected_gpu_uuid)


if __name__=="__main__":
    raise SystemExit(main())
