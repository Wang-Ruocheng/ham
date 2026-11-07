"""Independent, explicitly authorized continuous-field phi4 QCR experiment.

This is not the binary-spin recipe.  Physical couplings and sampler observables
are provenance/evaluation data only; the optimization receives field arrays.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from recurrent_rg.phi4_qcr import Phi4QCR, continuous_objectives


RECIPE = "phi4_continuous_gaussian_soft_statistics_v1"
EVALUATION_SEED = 20261022


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def state_fingerprint(model):
    """Stable tensor-content hash, independent of pickle/ZIP serialization."""
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def tensor_fingerprint(tensor):
    """Hash one tensor's dtype, shape and content, without serialization."""
    value = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def module_fingerprints(model):
    return {name: state_fingerprint(getattr(model, name)) for name in ("Q", "C", "R")}


def initialize_q_decoder(model, configurations, train_indices, *, seed,
                         mode="train_mean_bias", budget_check=None):
    """Initialize only the affine bias from TRAIN fields under random initial C.

    Each physical ensemble is processed separately along its own ancestral C
    trajectory.  Ensembles and the three scales receive equal weight.  The
    private generator never advances the global or subsequent training RNGs;
    no optimizer, labels, TEST fields, kernel selection or pretraining is used.
    """
    if mode not in ("train_mean_bias", "random_legacy"):
        raise ValueError("Unknown Q initialization mode")
    constructor_sha = state_fingerprint(model)
    before_modules = module_fingerprints(model)
    protected = {"Q_encoder": state_fingerprint(model.Q.net),
                 "Q_decoder_weight": tensor_fingerprint(model.Q.decoder.weight)}
    device = model.Q.decoder.bias.device
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    initial_bias_sha = tensor_fingerprint(model.Q.decoder.bias)
    receipt = dict(mode=mode, constructor_state_sha256=constructor_sha,
        constructor_module_sha256=before_modules, constructor_protected_sha256=protected,
        bias_before_sha256=initial_bias_sha, labels_used=False, test_data_used=False,
        pca_initialization=False, warm_start=False, kernel_selection=False,
        optimizer_updates=dict(Q=0, C=0, R=0), scales=[0, 1, 2],
        scale_weighting="uniform", physical_ensemble_weighting="uniform",
        fields_per_ensemble=None, physical_ensemble_count=0, total_fields=0,
        generator_seed=None, mean=None, mean_sha256=None)
    if budget_check is not None:
        budget_check()
    if mode == "train_mean_bias":
        indices = np.asarray(train_indices, dtype=np.int64)
        if (indices.ndim != 1 or not len(indices) or len(np.unique(indices)) != len(indices)
                or np.any(indices < 0) or np.any(indices >= len(configurations))):
            raise ValueError("Expected nonempty unique TRAIN physical-ensemble indices")
        generator_seed = int(seed) + 9000
        initialization_rng = torch.Generator(device=device).manual_seed(generator_seed)
        pooled = torch.zeros_like(model.Q.decoder.bias)
        with torch.no_grad():
            for index in indices:
                field = configurations[int(index)].unsqueeze(0).to(device=device, dtype=torch.float64)
                levels = model.coarse_grain(field, generator=initialization_rng)
                if len(levels) != 3:
                    raise ValueError("Mean-bias calibration requires exactly three scales")
                pooled.add_(torch.stack([model.features(level)[0] for level in levels]).mean(dim=0))
                if budget_check is not None:
                    budget_check()
            pooled.div_(len(indices))
            if not torch.isfinite(pooled).all():
                raise FloatingPointError("Nonfinite TRAIN mean-bias calibration")
            model.Q.decoder.bias.copy_(pooled)
        receipt.update(physical_ensemble_count=int(len(indices)),
            fields_per_ensemble=int(configurations.shape[1]),
            total_fields=int(len(indices) * configurations.shape[1]),
            generator_seed=generator_seed, mean=pooled.cpu().tolist(),
            mean_sha256=tensor_fingerprint(pooled))
    after_modules = module_fingerprints(model)
    after_protected = {"Q_encoder": state_fingerprint(model.Q.net),
                       "Q_decoder_weight": tensor_fingerprint(model.Q.decoder.weight)}
    cpu_unchanged = torch.equal(cpu_rng, torch.get_rng_state())
    cuda_unchanged = (torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
                      if cuda_rng is not None else None)
    if (before_modules["C"] != after_modules["C"] or before_modules["R"] != after_modules["R"]
            or protected != after_protected or not cpu_unchanged or cuda_unchanged is False
            or any(parameter.grad is not None for parameter in model.parameters())):
        raise RuntimeError("Mean-bias calibration altered protected parameters, gradients or training RNG")
    receipt.update(training_start_state_sha256=state_fingerprint(model),
        training_start_module_sha256=after_modules,
        training_start_protected_sha256=after_protected,
        bias_after_sha256=tensor_fingerprint(model.Q.decoder.bias),
        global_cpu_rng_unchanged=cpu_unchanged, global_cuda_rng_unchanged=cuda_unchanged,
        global_cpu_rng_before_sha256=tensor_fingerprint(cpu_rng),
        global_cpu_rng_after_sha256=tensor_fingerprint(torch.get_rng_state()),
        global_cuda_rng_before_sha256=tensor_fingerprint(cuda_rng) if cuda_rng is not None else None,
        global_cuda_rng_after_sha256=tensor_fingerprint(torch.cuda.get_rng_state(device)) if cuda_rng is not None else None,
        training_batch_rng_created_after_initialization=True,
        training_field_rng_created_after_initialization=True,
        all_parameter_gradients_none=True,
        C_trainable=all(parameter.requires_grad for parameter in model.C.parameters()))
    return receipt


def _inside(root, relative):
    path = (root / relative).resolve()
    if path == root or root not in path.parents:
        raise ValueError("Data manifest path escapes its root")
    return path


@dataclass
class FieldDataset:
    configurations: torch.Tensor
    splits: np.ndarray
    cell_ids: np.ndarray
    labels: np.ndarray
    manifest_sha256: str
    receipts: list
    warning_count: int
    qualification: str
    sampler_schema: str = "unspecified"
    sample_semantics: str = "physical_ensembles"

    @property
    def train_indices(self):
        return np.flatnonzero(self.splits == "train")

    @property
    def test_indices(self):
        return np.flatnonzero(self.splits == "test")

    @property
    def scale(self):
        # No coupling or diagnostic value enters this normalization.
        train = self.configurations[self.train_indices]
        return float(torch.mean(train.square()).sqrt())


def load_dataset(root):
    """Verify every declared shard, then read *only* its configurations member."""
    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Physical dataset is not complete")
    options = manifest["options"]
    if options["dimension"] != 2 or len(options["sizes"]) != 1:
        raise ValueError("This pilot requires one two-dimensional microscopic size")
    if options["samples"] < 4 or options["samples"] % 2:
        raise ValueError("Evaluation requires two equal disjoint saved-sample halves")
    side = int(options["sizes"][0])
    if side % 9 or side < 9:
        raise ValueError("Microscopic side must support two nonoverlapping b=3 steps")
    configurations, splits, cells, labels, receipts = [], [], [], [], []
    seen, cell_split = set(), {}
    explicit_cells = options.get("parameter_cells")
    if explicit_cells is not None:
        from recurrent_rg.data.phi4_hybrid_pipeline import parameter_cells
        from types import SimpleNamespace
        explicit_cells = parameter_cells(SimpleNamespace(**options))
    expected_cells = len(explicit_cells) if explicit_cells is not None else len(options["r_values"]) * len(options["u_values"])
    for shard in manifest["shards"]:
        if shard["size"] != side or shard["cell_index"] in seen:
            raise ValueError("Duplicate or unexpected microscopic shard")
        cell = int(shard["cell_index"])
        if not 0 <= cell < expected_cells:
            raise ValueError("Cell index outside the declared parameter grid")
        seen.add(cell)
        data_path, sidecar_path = _inside(root, shard["path"]), _inside(root, shard["sidecar"])
        data_sha, sidecar_sha = sha256_file(data_path), sha256_file(sidecar_path)
        if data_sha != shard["sha256"] or sidecar_sha != shard["sidecar_sha256"]:
            raise ValueError("Physical shard/sidecar SHA256 mismatch")
        metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
        expected_split = "test" if cell % options["test_every"] == options["test_offset"] else "train"
        if (metadata.get("status") != "complete" or metadata["cell_index"] != cell
                or metadata["size"] != side or metadata["dimension"] != 2
                or metadata["data_sha256"] != data_sha
                or metadata["split"] != expected_split or shard["split"] != expected_split):
            raise ValueError("Inconsistent whole-parameter-cell split or shard provenance")
        pair = explicit_cells[cell] if explicit_cells is not None else (
            options["r_values"][cell // len(options["u_values"])],
            options["u_values"][cell % len(options["u_values"])])
        expected_label = [*pair, options["h"]]
        actual_label = [metadata[name] for name in ("r", "u", "h")]
        if actual_label != expected_label:
            raise ValueError("Sidecar evaluation labels disagree with the manifest")
        # Do not materialize raw_observables, accepted, or delta_h.
        with np.load(data_path, allow_pickle=False) as archive:
            value = archive["configurations"]
        expected_shape = (options["chains"], options["samples"], side, side, 1)
        if value.shape != expected_shape or value.dtype != np.float64 or not np.isfinite(value).all():
            raise ValueError("Invalid continuous configuration array")
        configurations.append(torch.from_numpy(np.ascontiguousarray(value)))
        count = value.shape[0]
        splits.extend([expected_split] * count)
        cells.extend([cell] * count)
        labels.extend([actual_label] * count)
        cell_split[cell] = expected_split
        receipts.append(dict(path=shard["path"], sha256=data_sha,
                             sidecar=shard["sidecar"], sidecar_sha256=sidecar_sha))
    if seen != set(range(expected_cells)) or set(cell_split.values()) != {"train", "test"}:
        raise ValueError("Incomplete physical grid or missing train/test cells")
    dataset = FieldDataset(torch.cat(configurations, dim=0), np.asarray(splits),
        np.asarray(cells, dtype=np.int64), np.asarray(labels, dtype=np.float64),
        sha256_file(manifest_path), receipts, int(manifest.get("warning_count", 0)),
        manifest.get("qualification", "Finite samples do not certify physical RG."),
        manifest.get("schema", "unspecified"),
        manifest.get("sample_semantics", "physical_ensembles"))
    if not math.isfinite(dataset.scale) or dataset.scale <= 0:
        raise ValueError("TRAIN microscopic RMS must be finite and nonzero")
    return dataset


@dataclass
class TrainingOptions:
    data: str
    output: str
    device: str = "cuda:2"
    steps: int = 3000
    system_batch: int = 8
    sample_batch: int = 64
    seed: int = 91
    threads: int = 1
    log_every: int = 100
    checkpoint_every: int = 1000
    information_pairs: int = 128
    max_training_seconds: float = 1800.
    max_cuda_bytes: int = 8 * 1024 ** 3
    expected_gpu_uuid: str | None = None
    q_initialization: str = "train_mean_bias"
    expected_random_construction_sha256: str | None = None

    def validate(self):
        if self.q_initialization not in ("train_mean_bias", "random_legacy"):
            raise ValueError("q_initialization must be train_mean_bias or random_legacy")
        if (self.expected_random_construction_sha256 is not None
                and (len(self.expected_random_construction_sha256) != 64
                     or any(c not in "0123456789abcdef" for c in self.expected_random_construction_sha256))):
            raise ValueError("expected_random_construction_sha256 must be a lowercase SHA256")
        for name in ("steps", "system_batch", "threads", "log_every", "checkpoint_every",
                     "information_pairs", "max_cuda_bytes"):
            if isinstance(getattr(self, name), bool) or int(getattr(self, name)) != getattr(self, name) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.sample_batch < 4 or self.sample_batch % 2:
            raise ValueError("sample_batch must be even and at least four")
        if self.seed < 0 or not math.isfinite(self.max_training_seconds) or self.max_training_seconds <= 0:
            raise ValueError("Invalid seed or compute-time budget")
        if Path(self.output).exists():
            raise FileExistsError("A fresh output directory is required; resume/overwrite is disabled")


class BudgetHold(RuntimeError):
    pass


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _check_budget(options, device, started):
    _sync(device)
    elapsed = time.perf_counter() - started
    peak = max(int(torch.cuda.max_memory_allocated(device)),
               int(torch.cuda.max_memory_reserved(device))) if device.type == "cuda" else 0
    if elapsed > options.max_training_seconds:
        raise BudgetHold(f"Compute budget exceeded: {elapsed:.3f}s > {options.max_training_seconds}s; no step reduction or restart")
    if peak > options.max_cuda_bytes:
        raise BudgetHold(f"Own CUDA peak exceeded: {peak} > {options.max_cuda_bytes}; no automatic migration")
    return peak


def independent_step(model, batch, optimizers, *, generator, information_pairs):
    """Take all three independent gradients at the same parameter state."""
    q_loss, c_loss, r_loss, metrics = continuous_objectives(
        model, batch, generator=generator, information_pairs=information_pairs)
    parameters = {"Q": list(model.Q.parameters()), "C": list(model.C.parameters()),
                  "R": list(model.R.parameters())}
    losses = {"Q": q_loss, "C": c_loss, "R": r_loss}
    for loss in losses.values():
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite continuous-field objective")
    gradients = {name: torch.autograd.grad(losses[name], parameters[name]) for name in ("Q", "C", "R")}
    for name in ("Q", "C", "R"):
        optimizers[name].zero_grad(set_to_none=True)
        for parameter, gradient in zip(parameters[name], gradients[name]):
            if not torch.isfinite(gradient).all():
                raise FloatingPointError(f"Nonfinite {name} gradient")
            parameter.grad = gradient
    # No parameter is updated until every gradient has been computed and checked.
    for optimizer in optimizers.values():
        optimizer.step()
    model.Q.normalize_decoder_()
    row = dict(metrics)
    row.update(q_loss=float(q_loss.detach()), c_loss=float(c_loss.detach()), r_loss=float(r_loss.detach()))
    if not all(math.isfinite(float(value)) for value in row.values()):
        raise FloatingPointError("Nonfinite recorded training metric")
    return row


def optimizer_step_counts(optimizers, expected):
    receipt = {}
    for name, optimizer in optimizers.items():
        counts = [int(state["step"].item() if torch.is_tensor(state["step"]) else state["step"])
                  for state in optimizer.state.values() if "step" in state]
        total = sum(len(group["params"]) for group in optimizer.param_groups)
        if len(counts) != total or any(count != expected for count in counts):
            raise RuntimeError(f"{name} Adam parameter step counters do not all equal {expected}")
        receipt[name] = dict(min=min(counts), max=max(counts), parameters=total)
    return receipt


def closure_metrics(direct, one_step, rollout, train_direct):
    result = {}
    for name, prediction, target, target_scales in (
        ("one_step", one_step, direct[:, 1:], (1, 2)),
        ("rollout", rollout[:, 1:], direct[:, 1:], (1, 2))):
        by_scale = []
        for index, scale in enumerate(target_scales):
            rmse = float(np.sqrt(np.mean((prediction[:, index] - target[:, index]) ** 2)))
            std = float(np.sqrt(np.mean(np.var(train_direct[:, scale], axis=0, ddof=0))))
            by_scale.append(dict(target_scale=scale, rmse=rmse, train_target_std=std,
                                 nrmse=rmse / std if std > 0 else None))
        result[name] = by_scale
    return result


def evaluate_model(model, dataset, device, *, budget_check=None):
    """All chains, full saved ensemble; an RNG stream never used by training."""
    evaluation_rng = torch.Generator(device=device).manual_seed(EVALUATION_SEED)
    direct, direct_full, one_step, rollout = [], [], [], []
    feature_target, feature_reconstruction = [], []
    one_step_feature_prediction, rollout_feature_prediction = [], []
    model.eval()
    with torch.no_grad():
        for chain in dataset.configurations:
            x = chain.unsqueeze(0).to(device=device, dtype=torch.float64)
            levels = model.coarse_grain(x, generator=evaluation_rng)
            gs, full_gs, targets = [], [], []
            for level in levels:
                half = level.shape[1] // 2
                first = model.features(level[:, :half])
                second = model.features(level[:, half:])
                gs.append(.5 * (model.Q.from_features(first) + model.Q.from_features(second)))
                targets.append(.5 * (first + second))
                full_gs.append(model.Q.from_features(targets[-1]))
            encoded = torch.stack(gs, dim=1)
            one = torch.stack([model.advance(gs[index]) for index in range(2)], dim=1)
            roll = [gs[0]]
            for _ in range(2):
                roll.append(model.advance(roll[-1]))
            direct.append(encoded.cpu().numpy()[0])
            direct_full.append(torch.stack(full_gs, dim=1).cpu().numpy()[0])
            one_step.append(one.cpu().numpy()[0])
            rollout.append(torch.stack(roll, dim=1).cpu().numpy()[0])
            # Reuse the exact same C trajectory and already computed encodings.
            # These are frozen decoder diagnostics, not additional training loss.
            feature_target.append(torch.stack(targets, dim=1).cpu().numpy()[0])
            feature_reconstruction.append(model.Q.decode(encoded).cpu().numpy()[0])
            one_step_feature_prediction.append(model.Q.decode(one).cpu().numpy()[0])
            rollout_feature_prediction.append(model.Q.decode(torch.stack(roll, dim=1)).cpu().numpy()[0])
            if budget_check is not None:
                budget_check()
    direct, direct_full, one_step, rollout = map(np.asarray, (direct, direct_full, one_step, rollout))
    feature_target, feature_reconstruction, one_step_feature_prediction, rollout_feature_prediction = map(
        np.asarray, (feature_target, feature_reconstruction, one_step_feature_prediction,
                     rollout_feature_prediction))
    arrays = {}
    for split in ("train", "test"):
        mask = dataset.splits == split
        arrays.update({f"{split}_direct": direct[mask], f"{split}_direct_full": direct_full[mask],
                       f"{split}_one_step": one_step[mask],
                       f"{split}_rollout": rollout[mask], f"{split}_cell_ids": dataset.cell_ids[mask],
                       f"{split}_labels": dataset.labels[mask],
                       f"{split}_feature_target": feature_target[mask],
                       f"{split}_feature_reconstruction": feature_reconstruction[mask],
                       f"{split}_one_step_feature_prediction": one_step_feature_prediction[mask],
                       f"{split}_rollout_feature_prediction": rollout_feature_prediction[mask]})
    return arrays


def scalar_flow(model, arrays, device):
    """Frozen R only; numerical roots are not certified physical critical points."""
    from scipy.optimize import brentq

    inputs = np.concatenate([arrays[f"{split}_direct"][:, :2].reshape(-1) for split in ("train", "test")])
    low, high = float(inputs.min()), float(inputs.max())
    padding = .1 * max(high - low, 1e-6)
    grid = np.linspace(low - padding, high + padding, 256)
    with torch.no_grad():
        mapped = model.advance(torch.as_tensor(grid[:, None], device=device, dtype=torch.float64)).cpu().numpy()
    displacement = mapped[:, 0] - grid

    def drift(value):
        with torch.no_grad():
            x = torch.tensor([[value]], device=device, dtype=torch.float64)
            return float((model.advance(x) - x).item())

    candidates = [float(grid[i]) for i in range(len(grid)) if displacement[i] == 0.]
    for i in range(len(grid) - 1):
        if displacement[i] * displacement[i + 1] < 0:
            candidates.append(float(brentq(drift, grid[i], grid[i + 1], xtol=1e-13)))
    roots = []
    for value in sorted(candidates):
        if roots and abs(value - roots[-1]["g"]) < 1e-8:
            continue
        x = torch.tensor([[value]], device=device, dtype=torch.float64, requires_grad=True)
        derivative = float(torch.autograd.grad(model.advance(x).sum(), x)[0].item())
        modulus = abs(derivative)
        classification = "attracting" if modulus < 1 - 1e-6 else "repelling" if modulus > 1 + 1e-6 else "neutral"
        support = []
        for split in ("train", "test"):
            for scale in range(3):
                sample = arrays[f"{split}_direct"][:, scale, 0]
                support.append(dict(split=split, scale=scale, within_range=bool(sample.min() <= value <= sample.max()),
                                    nearest_distance=float(np.min(np.abs(sample - value)))))
        roots.append(dict(g=value, residual=abs(drift(value)), R_derivative=derivative,
            classification=classification, data_nearby=support, physical_critical_point_certified=False,
            thermal_nu=None))
    arrays.update(flow_g=grid[:, None], flow_R=mapped, flow_displacement=displacement[:, None])
    return roots


def cell_grid_evaluation(model, arrays, device):
    """Keep R(mean g) distinct from mean R(g), for the two trained inputs."""
    direct = np.concatenate([arrays["train_direct"], arrays["test_direct"]])
    one_step = np.concatenate([arrays["train_one_step"], arrays["test_one_step"]])
    cell_ids = np.concatenate([arrays["train_cell_ids"], arrays["test_cell_ids"]])
    labels = np.concatenate([arrays["train_labels"], arrays["test_labels"]])
    unique = np.unique(cell_ids)
    means = np.stack([direct[cell_ids == cell].mean(axis=0) for cell in unique])
    mean_R = np.stack([one_step[cell_ids == cell].mean(axis=0) for cell in unique])
    cell_labels = np.stack([labels[np.flatnonzero(cell_ids == cell)[0]] for cell in unique])
    with torch.no_grad():
        R_of_mean = model.advance(torch.as_tensor(means[:, :2], device=device,
                                                  dtype=torch.float64)).cpu().numpy()
    arrays.update(grid_cell_ids=unique, grid_labels=cell_labels,
        grid_mean_direct=means, grid_R_of_mean=R_of_mean, grid_mean_R=mean_R,
        grid_jensen_difference=mean_R - R_of_mean,
        grid_displacement=R_of_mean - means[:, :2])


def _device_receipt(device, expected_uuid):
    if device.type != "cuda":
        if expected_uuid is not None:
            raise ValueError("A GPU UUID cannot be checked on CPU")
        return dict(type="cpu", uuid=None)
    properties = torch.cuda.get_device_properties(device)
    raw_uuid = getattr(properties, "uuid", None)
    uuid = str(raw_uuid) if raw_uuid is not None else None
    if uuid is None:
        # Older Torch omits UUID from its Python properties. Query the loaded
        # CUDA *runtime* using the same logical device ordinal, not nvidia-smi's
        # potentially different physical ordering (or the driver ordinal).
        import ctypes
        import ctypes.util
        import uuid as uuid_module
        candidates = []
        mapped_libraries = Path("/proc/self/maps")
        if mapped_libraries.is_file():
            candidates.extend(line.split()[-1] for line in mapped_libraries.read_text().splitlines()
                              if "libcudart" in line and line.split()[-1].startswith("/"))
        found = ctypes.util.find_library("cudart")
        if found:
            candidates.append(found)
        candidates.extend(str(path) for path in (Path(torch.__file__).parent / "lib").glob("libcudart*"))
        for candidate in dict.fromkeys(candidates):
            try:
                runtime = ctypes.CDLL(candidate)
                get_uuid = runtime.cudaDeviceGetUuid
                get_uuid.argtypes = [ctypes.c_void_p, ctypes.c_int]
                get_uuid.restype = ctypes.c_int
                raw = (ctypes.c_ubyte * 16)()
                if get_uuid(ctypes.byref(raw), device.index if device.index is not None else torch.cuda.current_device()) == 0:
                    uuid = str(uuid_module.UUID(bytes=bytes(raw)))
                    break
            except (OSError, AttributeError):
                continue
    if expected_uuid is not None and (uuid is None or uuid.lower().removeprefix("gpu-") != expected_uuid.lower().removeprefix("gpu-")):
        raise RuntimeError(f"CUDA device UUID unavailable/mismatch: {uuid!r} != {expected_uuid!r}")
    return dict(type="cuda", index=device.index, name=properties.name,
                total_memory=int(properties.total_memory), uuid=uuid,
                CUDA_DEVICE_ORDER=os.environ.get("CUDA_DEVICE_ORDER"),
                CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"))


def run_training(options):
    options.validate()
    output = Path(options.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started, cpu_started = time.perf_counter(), time.process_time()
    step, latest_checkpoint = 0, 0
    history, model, manifest = [], None, None
    write_json(output / "status.json", dict(status="initializing", workflow_complete=False,
        actual_steps=0, returncode=None, error=None, requires_user_action=False, recipe=RECIPE))
    try:
        torch.set_num_threads(options.threads)
        torch.manual_seed(options.seed)
        dataset = load_dataset(options.data)
        if options.system_batch > len(dataset.train_indices) or options.sample_batch > dataset.configurations.shape[1]:
            raise ValueError("Requested physical/sample batch exceeds available TRAIN data")
        device = torch.device(options.device)
        # Include device/model setup and label-free calibration, not only the
        # update loop, in the unchanged compute-time and peak-memory budgets.
        compute_started = time.perf_counter()
        receipt = _device_receipt(device, options.expected_gpu_uuid)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model = Phi4QCR(dataset.scale, hidden=96, theory_dim=1).to(device=device, dtype=torch.float64)
        random_construction_sha = state_fingerprint(model)
        if (options.expected_random_construction_sha256 is not None
                and random_construction_sha != options.expected_random_construction_sha256):
            raise RuntimeError("Random construction SHA256 mismatch before initialization or optimizer updates")
        initialization = initialize_q_decoder(model, dataset.configurations, dataset.train_indices,
            seed=options.seed, mode=options.q_initialization,
            budget_check=lambda: _check_budget(options, device, compute_started))
        initial_sha = initialization["training_start_state_sha256"]
        import recurrent_rg.phi4_qcr as model_module
        sources = {str(Path(__file__).resolve()): sha256_file(__file__),
                   str(Path(model_module.__file__).resolve()): sha256_file(model_module.__file__)}
        manifest = dict(schema_version=5, recipe=RECIPE, options=asdict(options),
            data_root=str(Path(options.data).resolve()), data_manifest_sha256=dataset.manifest_sha256,
            data_shards=dataset.receipts, source_sha256=sources, theory_dim=1, hidden=96,
            lattice_sizes=[int(dataset.configurations.shape[2] // (3 ** k)) for k in range(3)],
            block_size=3, stride=3, dtype="float64",
            initialization=("random C/Q-encoder/Q-decoder-direction/R; TRAIN mean Q-decoder bias only; no warm start or kernel selection"
                            if options.q_initialization == "train_mean_bias" else "random; no warm start or kernel selection"),
            q_initialization=options.q_initialization, q_initialization_receipt=initialization,
            random_construction_state_sha256=random_construction_sha,
            random_construction_module_sha256=initialization["constructor_module_sha256"],
            training_start_state_sha256=initial_sha,
            training_start_module_sha256=initialization["training_start_module_sha256"],
            scale=dataset.scale, scale_definition="sqrt(mean(TRAIN microscopic field**2)); one frozen label-free scalar",
            learning_rate=.001, learning_rate_policy="constant", optimizer="three independent fresh Adam",
            closure_objective="all_start_anchor", closure_target_scales=2,
            closure_implementation="5*mean(one_step)+5*mean(microscopic_rollout); exactly equals all_start_anchor for S=2",
            closure_scale_weights=None, closure_branch_normalization="uniform average over two target scales",
            gradient_routing="simultaneous independent Q-only/C-only/R-only gradients; all gradients before all optimizer updates",
            labels_used_in_training=False, model_input_members=["configurations"],
            evaluation_label_columns=["r", "u", "h"], evaluation_seed=EVALUATION_SEED,
            evaluation_encoding="mean of Q(features(first half)) and Q(features(second half)); all saved samples; direct_full stores Q(pooled full features) separately",
            evaluation_sample_count=int(dataset.configurations.shape[1]),
            training_sample_count=options.sample_batch,
            training_encoding="mean of two half-ensemble Q encodings; full features reconstruction",
            data_warning_count=dataset.warning_count, equilibrium_certified=False,
            sampler_schema=dataset.sampler_schema, sample_semantics=dataset.sample_semantics,
            sampler_qualification=dataset.qualification, device=receipt,
            peak_cuda_bytes_semantics="maximum of own torch allocated and reserved high-water marks",
            initial_state_sha256=initial_sha,
            initial_state_sha256_semantics="post-initialization model state immediately before fresh Adam construction",
            max_training_seconds_semantics="device/model initialization + TRAIN mean-bias calibration + optimizer setup + updates + frozen final evaluation",
            thermal_nu=None)
        write_json(output / "manifest.json", manifest)
        optimizers = {name: torch.optim.Adam(getattr(model, name).parameters(), lr=.001) for name in ("Q", "C", "R")}
        batch_rng = torch.Generator(device="cpu").manual_seed(options.seed + 101)
        field_rng = torch.Generator(device=device).manual_seed(options.seed + 202)
        train_indices = torch.as_tensor(dataset.train_indices, dtype=torch.long)
        training_started = time.perf_counter()
        initialization_seconds = training_started - compute_started
        model.train()
        for update in range(1, options.steps + 1):
            _check_budget(options, device, compute_started)
            systems = train_indices[torch.randperm(len(train_indices), generator=batch_rng)[:options.system_batch]]
            samples = torch.randperm(dataset.configurations.shape[1], generator=batch_rng)[:options.sample_batch]
            batch = dataset.configurations[systems][:, samples].to(device=device, dtype=torch.float64)
            metrics = independent_step(model, batch, optimizers, generator=field_rng,
                                       information_pairs=options.information_pairs)
            step = update
            peak = _check_budget(options, device, compute_started)
            if step == 1 or step % options.log_every == 0 or step == options.steps:
                row = dict(step=step, **metrics, elapsed_seconds=time.perf_counter() - training_started)
                history.append(row)
                print(json.dumps(dict(event="phi4_qcr_progress", **row), allow_nan=False), flush=True)
            if step % options.checkpoint_every == 0 or step == options.steps:
                torch.save(dict(recipe=RECIPE, step=step, model=model.state_dict(),
                    optimizers={key: value.state_dict() for key, value in optimizers.items()},
                    batch_rng=batch_rng.get_state(), field_rng=field_rng.get_state(), manifest=manifest),
                    output / f"training_step_{step}.pt")
                latest_checkpoint = step
            if step == 1 or step % options.log_every == 0 or step == options.steps:
                write_json(output / "history_progress.json", dict(history=history, latest_step=step,
                    latest_checkpoint_step=latest_checkpoint, recipe=RECIPE))
                write_json(output / "status.json", dict(status="training", workflow_complete=False,
                    actual_steps=step, latest_checkpoint_step=latest_checkpoint, peak_cuda_bytes=peak,
                    returncode=None, error=None, requires_user_action=False, recipe=RECIPE))
        _sync(device)
        training_seconds = time.perf_counter() - training_started
        optimizer_steps = optimizer_step_counts(optimizers, options.steps)
        for parameter in model.parameters():
            parameter.grad = None
        final_sha = state_fingerprint(model)
        evaluation_started = time.perf_counter()
        write_json(output / "status.json", dict(status="evaluating", workflow_complete=False,
            actual_steps=step, latest_checkpoint_step=latest_checkpoint, returncode=None,
            error=None, requires_user_action=False, recipe=RECIPE))
        arrays = evaluate_model(model, dataset, device,
            budget_check=lambda: _check_budget(options, device, compute_started))
        cell_grid_evaluation(model, arrays, device)
        roots = scalar_flow(model, arrays, device)
        peak = _check_budget(options, device, compute_started)
        _sync(device)
        evaluation_seconds = time.perf_counter() - evaluation_started
        if final_sha != state_fingerprint(model) or any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError("Model/gradients changed during frozen evaluation")
        if sha256_file(Path(options.data) / "manifest.json") != dataset.manifest_sha256:
            raise RuntimeError("Data manifest changed during training")
        for path, expected_sha in sources.items():
            if sha256_file(path) != expected_sha:
                raise RuntimeError("Training source changed while experiment was running")
        np.savez(output / "evaluation_arrays.npz", **arrays)
        summary = dict(status="complete", recipe=RECIPE, actual_steps=step, optimizer_steps=optimizer_steps,
            latest_checkpoint_step=latest_checkpoint, initial_state_sha256=initial_sha,
            q_initialization=options.q_initialization, q_initialization_receipt=initialization,
            random_construction_state_sha256=random_construction_sha,
            random_construction_module_sha256=initialization["constructor_module_sha256"],
            training_start_state_sha256=initial_sha,
            training_start_module_sha256=initialization["training_start_module_sha256"],
            final_state_sha256=final_sha, final_module_sha256=module_fingerprints(model),
            frozen_evaluation_verified=True,
            data_manifest_sha256=dataset.manifest_sha256, labels_used_in_training=False,
            theory_dim=1, lattice_sizes=manifest["lattice_sizes"], scale=dataset.scale,
            train_chains=int(len(dataset.train_indices)), test_chains=int(len(dataset.test_indices)),
            samples_per_chain=int(dataset.configurations.shape[1]), evaluation_seed=EVALUATION_SEED,
            evaluation_direct_semantics="One shared stochastic C trajectory per physical chain; each direct g is the average of the two half-ensemble Q encodings; one-step and rollout start from these same g. direct_full is a separate pooled-feature Jensen diagnostic.",
            closure={split: closure_metrics(arrays[f"{split}_direct"], arrays[f"{split}_one_step"],
                arrays[f"{split}_rollout"], arrays["train_direct"]) for split in ("train", "test")},
            g_statistics={split: dict(mean=arrays[f"{split}_direct"].mean(axis=0).tolist(),
                std=arrays[f"{split}_direct"].std(axis=0, ddof=0).tolist(),
                minimum=arrays[f"{split}_direct"].min(axis=0).tolist(),
                maximum=arrays[f"{split}_direct"].max(axis=0).tolist()) for split in ("train", "test")},
            numerical_fixed_points=roots, thermal_nu=None, peak_cuda_bytes=peak,
            encoding_jensen_difference={split: dict(
                max_abs=float(np.max(np.abs(arrays[f"{split}_direct_full"] - arrays[f"{split}_direct"]))),
                rms=float(np.sqrt(np.mean((arrays[f"{split}_direct_full"] - arrays[f"{split}_direct"]) ** 2))))
                for split in ("train", "test")},
            timing=dict(initialization_seconds=initialization_seconds,
                        training_seconds=training_seconds, evaluation_seconds=evaluation_seconds,
                        compute_seconds=time.perf_counter() - compute_started,
                        wall_seconds=time.perf_counter() - started, cpu_seconds=time.process_time() - cpu_started),
            device=receipt, equilibrium_certified=False, data_warning_count=dataset.warning_count,
            sampler_schema=dataset.sampler_schema, sample_semantics=dataset.sample_semantics,
            sampler_qualification=dataset.qualification,
            qualification="Small single-seed continuous-field pilot; sampler-specific evidence is recorded separately; finite samples and numerical R roots are not certified physical critical points.")
        write_json(output / "summary.json", summary)
        write_json(output / "status.json", dict(status="complete", workflow_complete=True,
            actual_steps=step, latest_checkpoint_step=latest_checkpoint, peak_cuda_bytes=peak,
            returncode=0, error=None, requires_user_action=False, recipe=RECIPE,
            summary_sha256=sha256_file(output / "summary.json"),
            evaluation_arrays_sha256=sha256_file(output / "evaluation_arrays.npz")))
        print(json.dumps(dict(event="phi4_qcr_complete", actual_steps=step,
                              training_seconds=training_seconds, evaluation_seconds=evaluation_seconds)), flush=True)
        return summary
    except Exception as exc:
        write_json(output / "status.json", dict(status="budget_hold" if isinstance(exc, BudgetHold) else "failed",
            workflow_complete=False, actual_steps=step, latest_checkpoint_step=latest_checkpoint,
            returncode=3 if isinstance(exc, BudgetHold) else 1, error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(), requires_user_action=True, recipe=RECIPE,
            wall_seconds=time.perf_counter() - started))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:2")
    for name, default in (("steps", 3000), ("system_batch", 8), ("sample_batch", 64),
        ("seed", 91), ("threads", 1), ("log_every", 100), ("checkpoint_every", 1000),
        ("information_pairs", 128), ("max_cuda_bytes", 8 * 1024 ** 3)):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    parser.add_argument("--max-training-seconds", type=float, default=1800.)
    parser.add_argument("--expected-gpu-uuid")
    parser.add_argument("--q-initialization", choices=("train_mean_bias", "random_legacy"),
                        default="train_mean_bias",
                        help="Fresh Q decoder bias initialization; pin random_legacy for historical exact reproduction")
    parser.add_argument("--expected-random-construction-sha256",
                        help="Optional exact random constructor state pin, checked before calibration or updates")
    options = TrainingOptions(**vars(parser.parse_args(argv)))
    try:
        return run_training(options)
    except BudgetHold:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
