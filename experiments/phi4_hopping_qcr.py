"""Explicit continuous-field QCR alignment for hopping-kappa phi4 data.

This new recipe does not load historical checkpoints or alter binary defaults.
It restores learned raw-patch Q features and full soft 2x2 joint reconstruction.
Gaussian C and the smoothed information proxy remain continuous adaptations,
not the exact binary histogram-MI recipe. CLI planning is dry unless --execute.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
import operator
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from recurrent_rg.data.phi4_hopping_loader import (
    EVALUATION_LABEL_COLUMNS, FIELD_REPRESENTATION, PARAMETERIZATION, load_dataset,
)
from recurrent_rg.phi4_aligned_qcr import AlignedPhi4QCR, aligned_objectives


RECIPE = "phi4_hopping_raw_patch_joint625_v1"
EVALUATION_SEED = 20261031
MODEL_POLICY = dict(hidden=96, theory_dim=1, local_features=12,
    patch_chunk_size=4096, block_size=3, stride=3)
LOSS_POLICY = dict(closure_objective="all_start_anchor", closure_scale_weights=None,
    commutator=5, rollout_weight=5, q_reconstruction=10, q_subset_micro=20,
    q_subset_coarse=5, c_information=-2, c_variance=2, c_uniform=1)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
        allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def tensor_fingerprint(tensor):
    value = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    for part in (str(value.dtype).encode(), str(value.shape).encode(), value.tobytes()):
        digest.update(part)
    return digest.hexdigest()


def state_fingerprint(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor_fingerprint(tensor).encode())
    return digest.hexdigest()


def module_fingerprints(model):
    return {name: state_fingerprint(getattr(model, name)) for name in ("Q", "C", "R")}


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not bool")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(result)


def model_policy(block_size=3, depth=2):
    """Keep the b3 policy unchanged; make the requested b2/S3 variant explicit."""
    block_size = _integer(block_size, "block_size")
    depth = _integer(depth, "depth")
    if (block_size, depth) == (3, 2):
        return dict(MODEL_POLICY)
    if (block_size, depth) == (2, 3):
        return dict(MODEL_POLICY, block_size=2, stride=2, depth=3, patch_chunk_size=262144)
    raise ValueError("Supported explicit geometries are b3/depth2 or b2/depth3")


@dataclass
class TrainingOptions:
    data: str
    output: str
    device: str = "cuda:2"
    steps: int = 3000
    system_batch: int = 32
    sample_batch: int = 64
    seed: int = 91
    threads: int = 1
    log_every: int = 100
    checkpoint_every: int = 1000
    information_pairs: int = 128
    max_training_seconds: float = 1800.
    max_cuda_bytes: int = 8 * 1024 ** 3
    expected_gpu_uuid: str | None = None
    block_size: int = 3
    depth: int = 2
    early_stopping: str | None = None
    expected_initial_state_sha256: str | None = None

    def validate(self):
        model_policy(self.block_size, self.depth)
        if self.early_stopping is not None and (
                self.early_stopping != "train_closure_100_200_patience20_v1"
                or self.steps != 200 or (self.block_size, self.depth) != (2, 3)):
            raise ValueError("Explicit early stopping requires b2/depth3, max200 and the fixed TRAIN-only policy")
        if self.expected_initial_state_sha256 is not None and (
                not isinstance(self.expected_initial_state_sha256, str)
                or len(self.expected_initial_state_sha256) != 64
                or any(x not in "0123456789abcdef" for x in self.expected_initial_state_sha256)):
            raise ValueError("Expected initialization must be a lowercase SHA256")
        for name in ("steps", "system_batch", "sample_batch", "threads", "log_every",
                     "checkpoint_every", "information_pairs", "max_cuda_bytes"):
            _integer(getattr(self, name), name)
        _integer(self.seed, "seed", 0)
        if self.sample_batch < 4 or self.sample_batch % 2:
            raise ValueError("sample_batch must be even and at least four")
        if self.information_pairs < 2:
            raise ValueError("information_pairs must be at least two")
        if self.threads != 1:
            raise ValueError("This aligned continuous recipe pins CPU threads to one")
        if (isinstance(self.max_training_seconds, bool)
                or not math.isfinite(self.max_training_seconds) or self.max_training_seconds <= 0):
            raise ValueError("max_training_seconds must be finite and positive")
        if not isinstance(self.data, str) or not self.data.strip():
            raise ValueError("A hopping dataset path is required")
        if not isinstance(self.output, str) or not self.output.strip():
            raise ValueError("A fresh output path is required")
        if Path(self.output).exists():
            raise FileExistsError("Fresh output required; resume and overwrite are disabled")
        if Path(self.output).resolve() == Path(self.data).resolve():
            raise ValueError("Training output must differ from the physical data root")
        if not isinstance(self.device, str) or (self.device != "cpu" and not (
                self.device.startswith("cuda:") and self.device[5:].isdigit())):
            raise ValueError("Use cpu or an explicit fixed cuda:N device; no auto selection")
        if self.expected_gpu_uuid is not None and (not isinstance(self.expected_gpu_uuid, str)
                or not self.expected_gpu_uuid.strip() or self.device == "cpu"):
            raise ValueError("expected_gpu_uuid requires a nonempty UUID and a CUDA device")


def training_plan(options):
    """Pure plan: no data decompression, RNG draws, GPU access or output writes."""
    options.validate()
    return dict(schema_version=5, recipe=RECIPE, execution_requested=False,
        options=asdict(options), model=model_policy(options.block_size, options.depth), loss=dict(LOSS_POLICY),
        implementation_chunks=dict(Q_site_observations=model_policy(options.block_size,options.depth)["patch_chunk_size"],
            C_blocks=65536 if options.block_size==2 else None,
            activation_checkpoint=True, changes_objective=False, changes_parameter_shapes=False),
        optimizer=dict(name="three independent fresh Adam", learning_rate=.001,
            learning_rate_policy="constant", gradients="simultaneous independent Q/C/R"),
        label_policy="kappa/quartic_lambda/h are generation/evaluation only",
        evaluation_label_columns=list(EVALUATION_LABEL_COLUMNS),
        parameterization=PARAMETERIZATION, field_representation=FIELD_REPRESENTATION,
        q_target="normalized complete 2x2 soft joint, 5**4=625 bins; not exact continuous density",
        q_initialization="random encoder/decoder direction; first min(32,TRAIN) mean decoder bias",
        c_adaptation="Gaussian C; smoothed full-shell information proxy, variance floor and uniform-amplitude MSE; not literal binary empirical MI/NLL",
        formal_training_started=False, overwrite=False, resume=False,
        physical_critical_point_certified=False, thermal_nu=None)


def load_config(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) - {
            "schema_version", "recipe", "data", "output", "model", "loss", "training"}:
        raise ValueError("Unknown aligned hopping configuration fields")
    if value.get("schema_version") != 5 or value.get("recipe") != RECIPE:
        raise ValueError("Require schema_version=5 and the explicit aligned continuous recipe")
    policy = value.get("model")
    if policy == MODEL_POLICY:
        geometry = dict(block_size=3, depth=2)
    elif policy == model_policy(2, 3):
        geometry = dict(block_size=2, depth=3)
    else:
        raise ValueError("Model must match an explicit aligned hopping geometry policy")
    if value.get("loss") != LOSS_POLICY:
        raise ValueError("Model/loss must match the explicit aligned hopping policy")
    training = value.get("training", {})
    valid = {field.name for field in fields(TrainingOptions)} - {"data", "output", "block_size", "depth"}
    if not isinstance(training, dict) or set(training) - valid:
        raise ValueError("Unknown or misplaced training configuration fields")
    return dict(data=value.get("data"), output=value.get("output"), **geometry, **training)


def initialize_q_decoder(model, configurations, train_indices, *, seed, budget_check=None):
    """Only mean affine bias, on first min(32, TRAIN) ensembles, private RNG.

    Unlike the old continuous runner, this matches the discrete calibration
    cap and processes the selected TRAIN ensembles in one shared batch. No
    label, TEST field, optimizer update, PCA or C pretraining is used.
    """
    indices = np.asarray(train_indices)
    if (indices.ndim != 1 or not len(indices) or indices.dtype.kind not in "iu"
            or len(np.unique(indices)) != len(indices) or np.any(indices < 0)
            or np.any(indices >= len(configurations))):
        raise ValueError("Expected unique valid TRAIN ensemble indices")
    selected = indices[:32]
    device = model.Q.decoder.bias.device
    before, before_modules = state_fingerprint(model), module_fingerprints(model)
    protected = {name: tensor_fingerprint(tensor) for name, tensor in model.Q.state_dict().items()
        if name != "decoder.bias"}
    bias_before = tensor_fingerprint(model.Q.decoder.bias)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    generator_seed = int(seed) + 9000
    generator = torch.Generator(device=device).manual_seed(generator_seed)
    with torch.no_grad():
        x = configurations[selected].to(device=device, dtype=torch.float64)
        levels = model.coarse_grain(x, generator=generator)
        if len(levels) < 2:
            raise ValueError("Initialization requires at least two scales")
        mean = torch.stack([model.features(level).mean(dim=0) for level in levels]).mean(dim=0)
        if not torch.isfinite(mean).all():
            raise FloatingPointError("Nonfinite TRAIN affine-bias mean")
        model.Q.decoder.bias.copy_(mean)
    if budget_check is not None:
        budget_check()
    after_modules = module_fingerprints(model)
    after_protected = {name: tensor_fingerprint(tensor) for name, tensor in model.Q.state_dict().items()
        if name != "decoder.bias"}
    cpu_ok = torch.equal(cpu_rng, torch.get_rng_state())
    cuda_ok = torch.equal(cuda_rng, torch.cuda.get_rng_state(device)) if cuda_rng is not None else None
    if (protected != after_protected or before_modules["C"] != after_modules["C"]
            or before_modules["R"] != after_modules["R"] or not cpu_ok or cuda_ok is False
            or any(parameter.grad is not None for parameter in model.parameters())):
        raise RuntimeError("TRAIN mean-bias initialization changed protected state or RNG")
    return dict(mode="train_mean_bias", physical_ensemble_count=int(len(selected)),
        selected_train_indices=selected.tolist(), first_train_cap=32,
        fields_per_ensemble=int(configurations.shape[1]), scales=list(range(len(levels))),
        scale_weighting="uniform", physical_ensemble_weighting="uniform",
        generator_seed=generator_seed, labels_used=False, test_data_used=False,
        pca_initialization=False, kernel_selection=False, optimizer_updates=dict(Q=0, C=0, R=0),
        constructor_state_sha256=before, constructor_module_sha256=before_modules,
        bias_before_sha256=bias_before, bias_after_sha256=tensor_fingerprint(model.Q.decoder.bias),
        mean=mean.cpu().tolist(), mean_sha256=tensor_fingerprint(mean),
        global_cpu_rng_unchanged=cpu_ok, global_cuda_rng_unchanged=cuda_ok,
        all_parameter_gradients_none=True, training_start_state_sha256=state_fingerprint(model),
        training_start_module_sha256=after_modules)


class BudgetHold(RuntimeError):
    """A resource stop, never silently shorten, resume or migrate training."""


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _check_budget(options, device, started):
    _sync(device)
    peak = max(int(torch.cuda.max_memory_allocated(device)),
        int(torch.cuda.max_memory_reserved(device))) if device.type == "cuda" else 0
    elapsed = time.perf_counter() - started
    if elapsed > options.max_training_seconds:
        raise BudgetHold(f"Compute budget exceeded ({elapsed:.3f}s); no shortening or restart")
    if peak > options.max_cuda_bytes:
        raise BudgetHold(f"Own CUDA peak exceeded ({peak} bytes); no automatic migration")
    return peak


def _device_receipt(device, expected_uuid):
    if device.type == "cpu":
        return dict(type="cpu", uuid=None)
    properties = torch.cuda.get_device_properties(device)
    raw_uuid = getattr(properties, "uuid", None)
    uuid = str(raw_uuid) if raw_uuid is not None else None
    if expected_uuid is not None and (uuid is None or uuid.lower().removeprefix("gpu-")
            != expected_uuid.lower().removeprefix("gpu-")):
        raise RuntimeError("CUDA UUID unavailable or mismatched; no fallback device")
    return dict(type="cuda", index=device.index, name=properties.name, uuid=uuid,
        total_memory=int(properties.total_memory),
        CUDA_DEVICE_ORDER=os.environ.get("CUDA_DEVICE_ORDER"),
        CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"))


def independent_step(model, batch, optimizers, *, generator, information_pairs,
                     r_updates_per_batch=1):
    if type(r_updates_per_batch) is not int or not 1 <= r_updates_per_batch <= 32:
        raise ValueError("r_updates_per_batch must be an integer in [1, 32]")
    targets = [] if r_updates_per_batch > 1 else None
    losses = aligned_objectives(model, batch, generator=generator,
        information_pairs=information_pairs,
        **({"closure_targets_sink": targets} if targets is not None else {}))
    parameters = {name: list(getattr(model, name).parameters()) for name in ("Q", "C", "R")}
    if set(optimizers) != {"Q", "C", "R"}:
        raise ValueError("Require one fresh optimizer for each Q/C/R parameter group")
    gradients = {}
    for name, loss in zip(("Q", "C", "R"), losses[:3]):
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite {name} objective")
        gradients[name] = torch.autograd.grad(loss, parameters[name])
    for name in ("Q", "C", "R"):
        optimizers[name].zero_grad(set_to_none=True)
        for parameter, gradient in zip(parameters[name], gradients[name]):
            if not torch.isfinite(gradient).all():
                raise FloatingPointError(f"Nonfinite {name} gradient")
            parameter.grad = gradient
    # Every independent gradient is computed and validated before any update.
    for name in ("Q", "C", "R"):
        optimizers[name].step()
    model.Q.normalize_decoder_()
    result = {name: float(value) for name, value in losses[3].items()}
    if targets is not None:
        from types import SimpleNamespace
        from recurrent_rg.config import LossConfig
        from recurrent_rg.r_refinement import refine_r_on_targets
        # Adapter does not register duplicate modules or change checkpoints.
        adapter = SimpleNamespace(theory_rg=model.R, advance_theory=model.advance_theory)
        for name in ("Q", "C"):
            optimizers[name].zero_grad(set_to_none=True)
        after = refine_r_on_targets(adapter, optimizers["R"], targets[0], LossConfig(),
                                    r_updates_per_batch - 1)
        result.update(R_batch_before=losses[3]["r_total"], R_batch_after=after,
                      r_updates_per_batch=r_updates_per_batch)
    result.update({f"{name.lower()}_loss": float(loss.detach())
        for name, loss in zip(("Q", "C", "R"), losses[:3])})
    if not all(math.isfinite(value) for value in result.values()):
        raise FloatingPointError("Nonfinite training metrics")
    return result


def optimizer_step_counts(optimizers, expected, *, r_updates_per_batch=1):
    receipt = {}
    for name, optimizer in optimizers.items():
        counts = [int(state["step"].item() if torch.is_tensor(state["step"]) else state["step"])
            for state in optimizer.state.values() if "step" in state]
        count = sum(len(group["params"]) for group in optimizer.param_groups)
        target = expected * (r_updates_per_batch if name == "R" else 1)
        if not counts or len(counts) != count or any(step != target for step in counts):
            raise RuntimeError(f"{name} Adam parameter clocks differ from {target}")
        receipt[name] = dict(min=min(counts), max=max(counts), parameters=count)
    return receipt


def fixed_train_closure_probe(model, configurations, train_indices, *, seed):
    """Post-update TRAIN-only probe; independent RNG, no gradients or updates.

    This selects an exploratory stopping point, not a validation minimum or a
    certification of physical closure. Q/C still evolve, so the target changes.
    """
    from recurrent_rg.losses import all_start_anchor_closure_loss
    candidates = np.asarray(train_indices)
    if (candidates.ndim != 1 or not len(candidates) or candidates.dtype.kind not in "iu"
            or len(np.unique(candidates)) != len(candidates) or np.any(candidates < 0)
            or np.any(candidates >= len(configurations))):
        raise ValueError("Require unique valid TRAIN ensemble indices")
    indices = candidates[np.linspace(0, len(candidates)-1, min(8, len(candidates)), dtype=int)]
    if not len(indices) or indices.dtype.kind not in "iu":
        raise ValueError("A nonempty TRAIN-only index vector is required")
    samples = min(8, int(configurations.shape[1]))
    if samples < 4 or samples % 2:
        raise ValueError("Probe requires at least four and an even number of saved fields")
    device = next(model.parameters()).device
    before = state_fingerprint(model)
    gradients = [None if p.grad is None else tensor_fingerprint(p.grad) for p in model.parameters()]
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    generator = torch.Generator(device=device).manual_seed(seed + 303)
    with torch.no_grad():
        batch = configurations[indices, :samples].to(device=device, dtype=torch.float64)
        levels = model.coarse_grain(batch, generator=generator)
        half = samples // 2
        gs = [.5 * (model.Q(level[:, :half]) + model.Q(level[:, half:])) for level in levels]
        loss, _, _ = all_start_anchor_closure_loss(model, torch.stack(gs, dim=1),
                                                  coordinate_reduction="sum")
        value = float(10. * loss)
    if (not math.isfinite(value) or value < 0 or before != state_fingerprint(model)
            or gradients != [None if p.grad is None else tensor_fingerprint(p.grad) for p in model.parameters()]
            or not torch.equal(cpu_rng, torch.get_rng_state())
            or (cuda_rng is not None and not torch.equal(cuda_rng, torch.cuda.get_rng_state(device)))):
        raise RuntimeError("TRAIN closure probe is nonfinite or changed protected model/gradient/RNG state")
    return value


def evaluate_model(model, dataset, device, *, budget_check=None):
    """Frozen private stream, matching two-half Q encoding; labels attached last."""
    generator = torch.Generator(device=device).manual_seed(EVALUATION_SEED)
    direct, direct_full, one_step, rollout, targets, reconstructions = [], [], [], [], [], []
    all_start = []
    pairs = None
    model.eval()
    with torch.no_grad():
        for chain in dataset.configurations:
            levels = model.coarse_grain(chain.unsqueeze(0).to(device=device,
                dtype=torch.float64), generator=generator)
            half = chain.shape[0] // 2
            gs = [.5 * (model.Q(level[:, :half]) + model.Q(level[:, half:])) for level in levels]
            encoded = torch.stack(gs, dim=1)
            target = torch.stack([.5 * (model.features(level[:, :half])
                + model.features(level[:, half:])) for level in levels], dim=1)
            pooled = torch.stack([model.Q(level) for level in levels], dim=1)
            one = torch.stack([model.advance(g) for g in gs[:-1]], dim=1)
            roll = [gs[0]]
            for _ in range(len(gs) - 1):
                roll.append(model.advance(roll[-1]))
            chain_pairs = [(s, t) for t in range(1, len(gs)) for s in range(t)]
            if pairs is not None and pairs != chain_pairs:
                raise ValueError("Physical ensembles must use one fixed scale geometry")
            pairs = chain_pairs
            predictions = {(0, t): value for t, value in enumerate(roll) if t}
            for s in range(1, len(gs) - 1):
                value = gs[s]
                for t in range(s + 1, len(gs)):
                    value = model.advance(value)
                    predictions[s, t] = value
            all_start.append(torch.stack([predictions[pair] for pair in pairs], dim=1).cpu().numpy()[0])
            for destination, value in ((direct, encoded), (direct_full, pooled),
                    (one_step, one), (rollout, torch.stack(roll, dim=1)),
                    (targets, target), (reconstructions, model.Q.decode(encoded))):
                destination.append(value.cpu().numpy()[0])
            if budget_check is not None:
                budget_check()
        arrays = dict(direct_g=np.asarray(direct), direct_full_g=np.asarray(direct_full),
            one_step_g=np.asarray(one_step), rollout_g=np.asarray(rollout),
            all_start_g=np.asarray(all_start), all_start_pairs=np.asarray(pairs, dtype=np.int64),
            target_features=np.asarray(targets), reconstructed_features=np.asarray(reconstructions))
        dimension = arrays["direct_g"].shape[-1]
        if dimension == 1:
            inputs = arrays["direct_g"][dataset.train_indices, :-1].reshape(-1)
            low, high = float(inputs.min()), float(inputs.max())
            padding = .1 * max(high - low, 1e-6)
            grid = np.linspace(low - padding, high + padding, 501)[:, None]
            arrays["flow_grid"] = grid
            arrays["flow_next"] = model.advance(torch.as_tensor(grid,
                device=device, dtype=torch.float64)).cpu().numpy()
        else:
            arrays["flow_grid"] = np.empty((0, dimension), dtype=np.float64)
            arrays["flow_next"] = np.empty((0, dimension), dtype=np.float64)
    # No coupling/split/cell metadata is passed into Q, C, R or an objective.
    arrays.update(labels=np.asarray(dataset.labels), splits=np.asarray(dataset.splits),
        cell_ids=np.asarray(dataset.cell_ids))
    for name, value in arrays.items():
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise FloatingPointError(f"Nonfinite evaluation array {name}")
    return arrays


def closure_metrics(arrays):
    from scripts.report_phi4_hopping_qcr import closure_metrics as saved_array_metrics
    return saved_array_metrics(arrays)


def source_pins():
    import recurrent_rg.data.phi4_hopping_loader as loader_module
    import recurrent_rg.phi4_aligned_qcr as model_module
    import recurrent_rg.phi4_qcr as continuous_module
    import recurrent_rg.losses as closure_module
    import scripts.report_phi4_hopping_qcr as report_module
    import recurrent_rg.phi4_early_stopping as stopping_module
    return {str(Path(module.__file__).resolve()): sha256_file(module.__file__)
        for module in (loader_module, model_module, continuous_module, closure_module, report_module, stopping_module)} | {
        str(Path(__file__).resolve()): sha256_file(__file__)}


def verify_dataset_unchanged(data_root, dataset):
    """Recheck only the new physical dataset's public manifest/shard bindings."""
    root = Path(data_root).resolve()
    if sha256_file(root / "manifest.json") != dataset.manifest_sha256:
        raise RuntimeError("Physical data manifest changed during training")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if sha256_file(root / "plan.json") != manifest["plan_sha256"]:
        raise RuntimeError("Physical data plan changed during training")
    for receipt in dataset.receipts:
        for name, digest_name in (("path", "sha256"), ("sidecar", "sidecar_sha256")):
            path = (root / receipt[name]).resolve()
            if root not in path.parents or sha256_file(path) != receipt[digest_name]:
                raise RuntimeError("New physical shard/sidecar changed during training")


def run_training(options):
    """Explicit execution only; fresh output, no resume, no auto-repair or roots."""
    plan = training_plan(options)
    output = Path(options.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started, cpu_started = time.perf_counter(), time.process_time()
    step, latest_checkpoint, stage = 0, 0, "initializing"
    history = []
    stopping = None
    selected_step = None
    def status(state, **extra):
        write_json(output / "status.json", dict(status=state, recipe=RECIPE,
            workflow_complete=False, actual_steps=step, latest_checkpoint_step=latest_checkpoint,
            returncode=None, error=None, requires_user_action=False, **extra))
    status(stage)
    write_json(output / "plan.json", dict(plan, execution_requested=True))
    try:
        pins = source_pins()
        torch.set_num_threads(1)
        torch.manual_seed(options.seed)
        dataset = load_dataset(options.data, block_size=options.block_size, depth=options.depth)
        if options.system_batch > len(dataset.train_indices) or options.sample_batch > dataset.configurations.shape[1]:
            raise ValueError("Requested physical/sample batch exceeds available TRAIN data")
        device = torch.device(options.device)
        receipt = _device_receipt(device, options.expected_gpu_uuid)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model = AlignedPhi4QCR(dataset.scale, hidden=96, theory_dim=1,
            local_features=12, patch_chunk_size=model_policy(options.block_size,options.depth)["patch_chunk_size"], block_size=options.block_size,
            steps=options.depth).to(device=device, dtype=torch.float64)
        initialization = initialize_q_decoder(model, dataset.configurations,
            dataset.train_indices, seed=options.seed,
            budget_check=lambda: _check_budget(options, device, started))
        if (options.expected_initial_state_sha256 is not None
                and initialization["training_start_state_sha256"] != options.expected_initial_state_sha256):
            raise RuntimeError("Fresh initialization differs from the pinned original; zero optimizer updates")
        if options.early_stopping is not None:
            from recurrent_rg.phi4_early_stopping import EarlyStopping
            stopping = EarlyStopping()
        manifest = dict(schema_version=5, recipe=RECIPE, options=asdict(options),
            model=model_policy(options.block_size, options.depth), loss=dict(LOSS_POLICY), dtype="float64",
            implementation_chunks=dict(Q_site_observations=model.Q.patch_chunk_size,
                C_blocks=getattr(model.C,"block_chunk_size",None),activation_checkpoint=True,
                changes_objective=False,changes_parameter_shapes=False),
            theory_dim=1, hidden=96, local_features=12, block_size=options.block_size,
            stride=options.block_size, coarse_graining_depth=options.depth,
            lattice_sizes=[int(dataset.configurations.shape[2] // options.block_size ** index)
                for index in range(options.depth + 1)],
            scale=dataset.scale, scale_definition="one frozen TRAIN microscopic original-phi RMS",
            q_input="physical raw overlapping 3x3 patches, normalized by fixed TRAIN RMS",
            q_target="full 2x2 soft joint probabilities (5**4=625), not 105 single/pair marginals",
            decoder_projection="post-Adam unit column norm, no forced full rank",
            q_initialization_receipt=initialization, learning_rate=.001,
            learning_rate_policy="constant", optimizer="three independent fresh Adam",
            gradient_routing="all independent Q/C/R gradients computed before all Adam steps",
            closure_objective="all_start_anchor", closure_scale_weights=None,
            labels_used_in_training=False, model_input_members=["configurations"],
            evaluation_label_columns=list(EVALUATION_LABEL_COLUMNS),
            parameterization=PARAMETERIZATION, field_representation=FIELD_REPRESENTATION,
            evaluation_seed=EVALUATION_SEED,
            evaluation_encoding="mean of Q(first half fields) and Q(second half fields); direct_full_g separately encodes pooled fields",
            flow_domain=f"TRAIN input scales 0..{options.depth - 1} g bounds with 10% margin; no numerical root search",
            evaluation_all_start_pair_order="target t increasing, then source s=0..t-1",
            source_sha256=pins, data_root=str(Path(options.data).resolve()),
            data_manifest_sha256=dataset.manifest_sha256, data_shards=dataset.receipts,
            data_warning_count=dataset.warning_count, equilibrium_certified=False,
            sampler_schema=dataset.sampler_schema, sample_semantics=dataset.sample_semantics,
            sampler_qualification=dataset.qualification, device=receipt,
            continuous_adaptations=plan["c_adaptation"],
            exact_binary_recipe=False, thermal_nu=None,
            max_training_seconds_semantics="data read, setup, TRAIN bias calibration, updates, evaluation and report; not a wall-time forecast")
        if stopping is not None:
            manifest["early_stopping"] = dict(policy=options.early_stopping,
                minimum_step=100, maximum_step=200, every=10, patience=20, min_delta=0.,
                monitor="post-update 10*all_start_anchor closure on fixed TRAIN subset",
                selected_train_indices=np.asarray(dataset.train_indices)[np.linspace(0,
                    len(dataset.train_indices)-1, min(8, len(dataset.train_indices)), dtype=int)].tolist(),
                ensemble_selection="uniform ranks in TRAIN index vector, without labels or cell metadata",
                saved_fields_per_chain=min(8, int(dataset.configurations.shape[1])),
                generator_seed=options.seed+303, test_data_used=False, labels_used=False,
                preserve_training_rng=True, additional_optimizer_updates=0,
                qualification="TRAIN stopping heuristic; changing Q/C targets; not held-out model selection")
        write_json(output / "manifest.json", manifest)
        optimizers = {name: torch.optim.Adam(getattr(model, name).parameters(), lr=.001)
            for name in ("Q", "C", "R")}
        batch_rng = torch.Generator(device="cpu").manual_seed(options.seed + 101)
        field_rng = torch.Generator(device=device).manual_seed(options.seed + 202)
        train_indices = torch.as_tensor(dataset.train_indices, dtype=torch.long)
        training_started = time.perf_counter()
        stage = "training"
        model.train()
        for update in range(1, options.steps + 1):
            _check_budget(options, device, started)
            systems = train_indices[torch.randperm(len(train_indices), generator=batch_rng)[:options.system_batch]]
            samples = torch.randperm(dataset.configurations.shape[1], generator=batch_rng)[:options.sample_batch]
            batch = dataset.configurations[systems][:, samples].to(device=device, dtype=torch.float64)
            metrics = independent_step(model, batch, optimizers,
                generator=field_rng, information_pairs=options.information_pairs)
            step = update
            peak = _check_budget(options, device, started)
            stop_now = False
            if stopping is not None and step >= 100 and step % 10 == 0:
                value = fixed_train_closure_probe(model, dataset.configurations,
                    dataset.train_indices, seed=options.seed)
                peak = _check_budget(options, device, started)
                stop_now = stopping.observe(step, value)
                write_json(output / "early_stopping.json", stopping.receipt())
                print(json.dumps(dict(event="early_stopping_probe", step=step,
                    train_closure=value, **stopping.receipt()), allow_nan=False), flush=True)
            if step == 1 or step % options.log_every == 0 or step == options.steps or stop_now:
                row = dict(step=step, **metrics, elapsed_seconds=time.perf_counter() - training_started)
                history.append(row)
                print(json.dumps(dict(event="phi4_hopping_qcr_progress", **row), allow_nan=False), flush=True)
                write_json(output / "history.json", history)
            if (step % options.checkpoint_every == 0 or step == options.steps or stop_now
                    or (stopping is not None and step >= 100 and step % 10 == 0)):
                checkpoint_path = output / f"training_step_{step}.pt"
                temporary = checkpoint_path.with_suffix(".pt.partial")
                torch.save(dict(recipe=RECIPE, step=step, model=model.state_dict(),
                    optimizers={name: optimizer.state_dict() for name, optimizer in optimizers.items()},
                    batch_rng=batch_rng.get_state(), field_rng=field_rng.get_state(), manifest=manifest),
                    temporary)
                os.replace(temporary, checkpoint_path)
                latest_checkpoint = step
            if step == 1 or step % options.log_every == 0 or step == options.steps or stop_now:
                status(stage, peak_cuda_bytes=peak)
            if stop_now:
                break
        optimizer_steps = optimizer_step_counts(optimizers, step)
        selected_step = step
        if stopping is not None:
            selected_step = stopping.best_step
            selected_path = output / f"training_step_{selected_step}.pt"
            selected = torch.load(selected_path, map_location=device, weights_only=False)
            if selected["step"] != selected_step or selected["manifest"] != manifest:
                raise RuntimeError("Selected early-stop checkpoint binding differs")
            model.load_state_dict(selected["model"], strict=True)
            selected_optimizer_steps = {}
            for name in ("Q", "C", "R"):
                states = selected["optimizers"][name]["state"].values()
                clocks = [int(x["step"].item()) for x in states if "step" in x]
                if not clocks or any(x != selected_step for x in clocks):
                    raise RuntimeError("Selected early-stop optimizer clocks differ")
                selected_optimizer_steps[name] = dict(min=min(clocks), max=max(clocks), parameters=len(clocks))
            del selected  # No optimizer or RNG is resumed; evaluation only.
            early_receipt = dict(stopping.receipt(), actual_steps=step,
                evaluated_checkpoint_step=selected_step,
                selected_checkpoint_sha256=sha256_file(selected_path),
                selected_optimizer_steps=selected_optimizer_steps,
                test_data_used=False, labels_used=False)
            write_json(output / "early_stopping.json", early_receipt)
        training_seconds = time.perf_counter() - training_started
        for parameter in model.parameters():
            parameter.grad = None
        final_sha = state_fingerprint(model)
        stage = "evaluating"
        status(stage)
        arrays = evaluate_model(model, dataset, device,
            budget_check=lambda: _check_budget(options, device, started))
        peak = _check_budget(options, device, started)
        if final_sha != state_fingerprint(model) or any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError("Frozen final evaluation changed model or gradients")
        verify_dataset_unchanged(options.data, dataset)
        for path, expected in pins.items():
            if sha256_file(path) != expected:
                raise RuntimeError("Pinned training/evaluation source changed during execution")
        np.savez(output / "evaluation_arrays.npz", **arrays)
        summary = dict(status="complete", recipe=RECIPE, actual_steps=step,
            optimizer_steps=optimizer_steps, latest_checkpoint_step=latest_checkpoint,
            labels_used_in_training=False, evaluation_label_columns=list(EVALUATION_LABEL_COLUMNS),
            theory_dim=1, lattice_sizes=manifest["lattice_sizes"], scale=dataset.scale,
            q_initialization_receipt=initialization, final_state_sha256=final_sha,
            final_module_sha256=module_fingerprints(model), frozen_evaluation_verified=True,
            data_manifest_sha256=dataset.manifest_sha256, train_chains=len(dataset.train_indices),
            test_chains=len(dataset.test_indices), samples_per_chain=int(dataset.configurations.shape[1]),
            closure=closure_metrics(arrays), peak_cuda_bytes=peak, thermal_nu=None,
            physical_critical_point_certified=False, equilibrium_certified=False,
            data_warning_count=dataset.warning_count, sampler_qualification=dataset.qualification,
            timing=dict(training_seconds=training_seconds, compute_seconds=time.perf_counter() - started,
                wall_seconds=time.perf_counter() - started, cpu_seconds=time.process_time() - cpu_started),
            timing_scope="through frozen evaluation; excludes report and final public-file hashing",
            qualification="Continuous adaptation and finite smoothed motif target; no certified thermal exponent or physical RG/fixed point. Finite-chain warnings are retained.")
        if stopping is not None:
            summary.update(early_stopping=early_receipt, evaluated_checkpoint_step=selected_step,
                selected_optimizer_steps=selected_optimizer_steps,
                actual_steps_semantics="optimizer updates actually executed; evaluation restores selected checkpoint model only")
        write_json(output / "summary.json", summary)
        stage = "reporting"
        status(stage, peak_cuda_bytes=peak)
        from scripts.report_phi4_hopping_qcr import render_report
        report_started = time.perf_counter()
        render_report(output)
        report_seconds = time.perf_counter() - report_started
        peak = _check_budget(options, device, started)
        for path, expected in pins.items():
            if sha256_file(path) != expected:
                raise RuntimeError("Pinned source changed during report generation")
        public_files = []
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.suffix.lower() in (".json", ".png", ".npz") and path.name != "status.json":
                public_files.append(dict(path=str(path.relative_to(output)), sha256=sha256_file(path), bytes=path.stat().st_size))
        peak = _check_budget(options, device, started)
        write_json(output / "status.json", dict(status="complete", recipe=RECIPE,
            workflow_complete=True, actual_steps=step, latest_checkpoint_step=latest_checkpoint,
            peak_cuda_bytes=peak, returncode=0, error=None, requires_user_action=False,
            workflow_wall_seconds=time.perf_counter() - started,
            workflow_cpu_seconds=time.process_time() - cpu_started, report_seconds=report_seconds,
            timing_scope="full workflow through report and public-file hashing; excludes this final status write",
            summary_sha256=sha256_file(output / "summary.json"),
            evaluation_arrays_sha256=sha256_file(output / "evaluation_arrays.npz"),
            **(dict(evaluated_checkpoint_step=selected_step, early_stopping=early_receipt)
               if stopping is not None else {}),
            public_files=public_files))
        return summary
    except Exception as exc:
        write_json(output / "status.json", dict(status="budget_hold" if isinstance(exc, BudgetHold) else "failed",
            recipe=RECIPE, failed_stage=stage, workflow_complete=False, actual_steps=step,
            latest_checkpoint_step=latest_checkpoint, returncode=3 if isinstance(exc, BudgetHold) else 1,
            error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(),
            requires_user_action=True, wall_seconds=time.perf_counter() - started))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--execute", action="store_true")
    for name in ("data", "output", "device", "expected_gpu_uuid"):
        parser.add_argument("--" + name.replace("_", "-"))
    for name in ("steps", "system_batch", "sample_batch", "seed", "threads", "log_every",
                 "checkpoint_every", "information_pairs", "max_cuda_bytes", "block_size", "depth"):
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    parser.add_argument("--max-training-seconds", type=float)
    args = vars(parser.parse_args(argv))
    config_path, execute = args.pop("config"), args.pop("execute")
    values = load_config(config_path) if config_path is not None else {}
    values.update({name: value for name, value in args.items() if value is not None})
    if not values.get("data") or not values.get("output"):
        parser.error("--data and --output (or both in --config) are required")
    options = TrainingOptions(**values)
    if not execute:
        plan = training_plan(options)
        print(json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False))
        return plan
    try:
        return run_training(options)
    except BudgetHold:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
