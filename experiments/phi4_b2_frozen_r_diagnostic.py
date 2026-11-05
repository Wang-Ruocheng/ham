"""Bounded CPU-only R optimization controls on saved, frozen phi4 b=2 g.

This is deliberately not a QCR continuation: each physical chain has the saved
endpoint encoding of all 128 fields under one fixed C draw, rather than the
original changing Q/C and 64-field stochastic training batches. No Q/C model or
physical field archive is loaded. Labels are used only for plots, never by the
optimizer or the per-transition interpolation control.

python -m experiments.phi4_b2_frozen_r_diagnostic --training SOURCE_TRAINING \
    --r-state R_ONLY_EXTRACTION.pt --output NEW_DIRECTORY [--font FONT]

The extraction contains only R tensors and public provenance; the original full
checkpoints are only byte-hashed, not deserialized. No root search is performed.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import time
import traceback

import numpy as np
import torch
from torch import nn

from experiments.phi4_qcr import optimizer_step_counts, sha256_file, state_fingerprint, write_json
from scripts.report_phi4_b2_closure_comparison import load_public_comparison
from scripts.report_phi4_qcr import _plotting


RECIPE = "phi4_b2_frozen_R_cached_g_diagnostic_v1"
EXTRACTION_RECIPE = "phi4_b2_R_only_extraction_v1"
CONTROL_STEPS = 3000
CONTROL_NAMES = ("continue_legacy", "continue_all_start", "fresh_all_start")
MODES = dict(continue_legacy="one_step_and_rollout", continue_all_start="all_start_anchor",
             fresh_all_start="all_start_anchor")
LABELS = dict(source_legacy="原端点：早期闭环", source_all_start="原端点：全起点闭环",
              continue_legacy="固定表示：早期闭环续优化", continue_all_start="固定表示：全起点续优化",
              fresh_all_start="固定表示：随机 R 全起点")
SIZES = (32, 16, 8, 4)
PUBLIC_INPUTS = ("manifest.json", "summary.json", "history_progress.json", "status.json", "evaluation_arrays.npz")
LIMITATIONS = [
    "诊断固定端点 g，不是原 QCR 训练的严格续训，也不是重新学习 Q 或 C。",
    "每物理链 g 使用全部 128 构型和单个固定 C 实现；原训练使用随机 64 构型且 Q/C 持续变化，两者目标分布不同。",
    "固定 g 不等于新的独立物理样本；本诊断不认证采样平衡或临界点。",
    "所有 R 在原有共同一维坐标下比较，R(g) 定义为 g 加残差网络；不重定义坐标或改变 Q/C。",
    "逐转移 PCHIP 只用 TRAIN 条件的平均 g 配对，不用 r；每尺度独立，不是统一自治 R，不递归生成流场。",
    "PCHIP 禁止外推，留出覆盖率与范围外点单独报告；改善不证明物理 RG 正确。",
    "没有搜索或选择新根，critical_r 和 thermal_nu 始终为 null。",
]


class BudgetHold(RuntimeError):
    pass


@dataclass(frozen=True)
class Options:
    training: str
    r_state: str
    output: str
    font: str | None = None
    cpu_budget_seconds: float = 300.
    directory_budget_bytes: int = 100 * 1024 ** 2
    seed: int = 91
    optimizer_seed: int = 20261028
    steps: int = CONTROL_STEPS
    batch_size: int = 8
    learning_rate: float = .001

    def validate(self):
        if (self.steps != CONTROL_STEPS or self.seed != 91 or self.optimizer_seed != 20261028
                or self.batch_size != 8 or self.learning_rate != .001):
            raise ValueError("This diagnostic pins 3000 updates, seed 91, batch seed 20261028, batch 8 and Adam .001")
        if (not np.isfinite(self.cpu_budget_seconds) or not 0 < self.cpu_budget_seconds <= 300
                or not isinstance(self.directory_budget_bytes, int)
                or not 0 < self.directory_budget_bytes <= 100 * 1024 ** 2):
            raise ValueError("CPU budget must be <=300 seconds and directory <=100 MiB")


def make_r() -> nn.Sequential:
    """Exactly the original scalar residual submodule, without constructing Q/C."""
    return nn.Sequential(nn.Linear(1, 96), nn.Tanh(), nn.Linear(96, 96), nn.Tanh(), nn.Linear(96, 1)).double()


def advance(r: nn.Module, g: torch.Tensor) -> torch.Tensor:
    return g + r(g)


def closure_loss(r: nn.Module, direct: torch.Tensor, mode: str):
    if direct.ndim != 3 or direct.shape[1:] != (4, 1) or direct.requires_grad:
        raise ValueError("Only frozen [physical chain,4,1] direct g is allowed")
    if mode not in ("one_step_and_rollout", "all_start_anchor"):
        raise ValueError("Unknown closure mode")
    errors = {}
    for start in range(3):
        prediction = direct[:, start]
        for target in range(start + 1, 4):
            prediction = advance(r, prediction)
            errors[start, target] = (prediction - direct[:, target]).square().sum(-1).mean()
    adjacent = torch.stack([errors[k, k + 1] for k in range(3)]).mean()
    microscopic = torch.stack([errors[0, t] for t in range(1, 4)]).mean()
    all_start = 10. * torch.stack([torch.stack([errors[s, t] for s in range(t)]).mean()
                                  for t in range(1, 4)]).mean()
    objective = all_start if mode == "all_start_anchor" else 5. * (adjacent + microscopic)
    metrics = dict(loss=float(objective.detach()), one_step=float(adjacent.detach()),
                   micro_rollout=float(microscopic.detach()), all_start=float(all_start.detach()))
    return objective, metrics


def _digest(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a SHA256 digest")
    return value


def _file_sha(paths):
    return {str(path): sha256_file(path) for path in paths}


def load_sources(training: Path, r_state: Path):
    """Strict source success/pairing validation; deserialize only extracted R."""
    public = load_public_comparison(training)
    if public.manifest["options"]["steps"] != 3000 or public.summary["actual_steps"] != 3000:
        raise ValueError("Expected the completed 3000-update source experiment")
    if public.manifest.get("q_initialization_receipt", {}).get("fields_per_ensemble") != 128:
        raise ValueError("Saved endpoint encodings must use the 128-field protocol")
    for split, count, cells in (("train", 32, 8), ("test", 16, 4)):
        values = public.endpoints["legacy"][f"{split}_direct"]
        ids = public.endpoints["legacy"][f"{split}_cell_ids"]
        unique, counts = np.unique(ids, return_counts=True)
        if len(values) != count or len(unique) != cells or not np.all(counts == 4):
            raise ValueError("Expected 32 TRAIN and 16 TEST physical chains, four per condition")
    payload = torch.load(r_state, map_location="cpu", weights_only=True)
    if (not isinstance(payload, dict) or payload.get("recipe") != EXTRACTION_RECIPE
            or set(payload.get("arms", {})) != {"legacy", "all_start"}):
        raise ValueError("Expected an explicit R-only extraction, not a complete QCR checkpoint")
    if set(payload) - {"recipe", "source_manifest_sha256", "source_summary_sha256", "arms"}:
        raise ValueError("Unexpected members in R-only extraction")
    for key, filename in (("source_manifest_sha256", "manifest.json"), ("source_summary_sha256", "summary.json")):
        if payload.get(key) != public.fingerprints[filename]:
            raise ValueError(f"R-only provenance {key} differs from the completed source")
    models, checkpoints, receipt = {}, {}, {}
    for arm, row in payload["arms"].items():
        if (not isinstance(row, dict) or set(row) != {"state_dict", "source_checkpoint_sha256",
                "source_checkpoint_path", "step", "closure_objective"}
                or row["step"] != 3000
                or row["closure_objective"] != public.summary["arms"][arm]["closure_objective"]):
            raise ValueError("Invalid R-only arm receipt")
        checkpoint = Path(row["source_checkpoint_path"]).resolve()
        expected_checkpoint = (training.resolve() / arm / "training_step_3000.pt").resolve()
        if checkpoint != expected_checkpoint or not checkpoint.is_file():
            raise ValueError("Original checkpoint path must name this arm's completed endpoint")
        expected_sha = _digest(row["source_checkpoint_sha256"], "source checkpoint SHA")
        if sha256_file(checkpoint) != expected_sha:
            raise ValueError("Original checkpoint byte fingerprint differs from the extraction")
        state = row["state_dict"]
        model = make_r()
        if (not isinstance(state, dict) or set(state) != set(model.state_dict())
                or any(not isinstance(value, torch.Tensor) or value.dtype != torch.float64
                       or value.device.type != "cpu" or not torch.isfinite(value).all() for value in state.values())):
            raise ValueError("R-only state must contain only the six float64 R tensors")
        model.load_state_dict(state, strict=True)
        fingerprint = state_fingerprint(model)
        if fingerprint != public.summary["arms"][arm]["final_module_sha256"]["R"]:
            raise ValueError("Extracted R module fingerprint differs from source summary")
        maximum_error = 0.
        with torch.no_grad():
            for split in ("train", "test"):
                direct = torch.as_tensor(public.endpoints[arm][f"{split}_direct"], dtype=torch.float64)
                actual = advance(model, direct[:, :-1]).numpy()
                saved = public.endpoints[arm][f"{split}_one_step"]
                maximum_error = max(maximum_error, float(np.max(np.abs(actual - saved))))
        if maximum_error > 1e-10:
            raise ValueError(f"Source R cannot reproduce saved one-step outputs within 1e-10: {maximum_error}")
        models[arm] = model
        checkpoints[str(checkpoint)] = expected_sha
        receipt[arm] = dict(source_checkpoint_path=str(checkpoint), source_checkpoint_sha256=expected_sha,
            R_module_sha256=fingerprint, source_step=3000, saved_one_step_max_absolute_error=maximum_error,
            output_reproduction_tolerance=1e-10, full_checkpoint_deserialized=False)
    paths = [training / name for name in PUBLIC_INPUTS] + [r_state]
    fingerprints = _file_sha(paths)
    fingerprints.update(checkpoints)
    return public, models, receipt, fingerprints


@torch.no_grad()
def predictions(r: nn.Module, direct: np.ndarray):
    value = torch.as_tensor(direct, dtype=torch.float64)
    one_step = advance(r, value[:, :-1]).numpy()
    rollout = [value[:, 0]]
    for _ in range(3):
        rollout.append(advance(r, rollout[-1]))
    return dict(one_step=one_step, rollout=torch.stack(rollout, dim=1).numpy())


def _error(prediction, target, std):
    if len(target) == 0:
        return dict(absolute_rmse=None, fixed_train_nrmse=None, train_target_population_std=std,
                    evaluated_physical_chains=0)
    rmse = float(np.sqrt(np.mean((np.asarray(prediction) - np.asarray(target)) ** 2)))
    return dict(absolute_rmse=rmse, fixed_train_nrmse=rmse / std if std > 0 else None,
                train_target_population_std=std, evaluated_physical_chains=len(target))


def evaluate_metrics(train, test, predicted):
    rows = []
    for split, direct in (("train", train), ("test", test)):
        for branch in ("one_step", "rollout"):
            for target in range(1, 4):
                source = target - 1 if branch == "one_step" else 0
                prediction = predicted[split][branch][:, target - 1 if branch == "one_step" else target]
                target_g, std = direct[:, target], float(train[:, target].std(ddof=0))
                rows.append(dict(split=split, branch=branch, source_scale=source, target_scale=target,
                    source_lattice_size=SIZES[source], target_lattice_size=SIZES[target],
                    model=_error(prediction, target_g, std),
                    identity=_error(direct[:, source], target_g, std),
                    train_constant=_error(np.full_like(target_g, train[:, target].mean()), target_g, std)))
    return rows


def pchip_control(train, test, train_ids, test_ids):
    """Three independent TRAIN-mean regressions; no labels or extrapolation."""
    from scipy.interpolate import PchipInterpolator
    ids = np.asarray(train_ids)
    if (ids.shape != (len(train),) or np.asarray(test_ids).shape != (len(test),)
            or set(ids.tolist()) & set(np.asarray(test_ids).tolist())):
        raise ValueError("PCHIP requires whole-condition disjoint TRAIN/TEST IDs")
    cells = np.unique(ids)
    means = np.stack([train[ids == cell].mean(axis=0) for cell in cells])
    rows, arrays = [], {}
    for scale in range(3):
        x, y = means[:, scale, 0], means[:, scale + 1, 0]
        order = np.argsort(x, kind="stable")
        x, y = x[order], y[order]
        if len(x) < 2 or not np.all(np.diff(x) > 0):
            raise ValueError("PCHIP requires at least two distinct TRAIN condition-mean inputs")
        interpolation = PchipInterpolator(x, y, extrapolate=False)
        arrays[f"pchip_scale{scale}_train_condition_mean_input"] = x
        arrays[f"pchip_scale{scale}_train_condition_mean_target"] = y
        for split, values in (("train", train), ("test", test)):
            prediction = interpolation(values[:, scale, 0])[:, None]
            covered = np.isfinite(prediction[:, 0])
            std = float(train[:, scale + 1].std(ddof=0))
            arrays[f"pchip_{split}_scale{scale}_prediction"] = prediction
            arrays[f"pchip_{split}_scale{scale}_covered"] = covered
            rows.append(dict(split=split, source_scale=scale, target_scale=scale + 1,
                train_mean_input_range=[float(x[0]), float(x[-1])],
                total_physical_chains=len(values), covered_physical_chains=int(covered.sum()),
                coverage_fraction=float(covered.mean()), outside_training_range_indices=np.flatnonzero(~covered).tolist(),
                model=_error(prediction[covered], values[covered, scale + 1], std),
                identity_on_same_covered_chains=_error(values[covered, scale], values[covered, scale + 1], std)))
    return dict(method="per_transition_PCHIP_of_TRAIN_condition_mean_pairs", rows=rows,
        coupling_labels_used=False, condition_ids_used_for_grouping_only=True, train_conditions=len(cells),
        extrapolation=False, unified_autonomous_R=False, recursive_rollout=False,
        critical_r=None, thermal_nu=None, limitations=LIMITATIONS[4:6]), arrays


def _directory_bytes(root):
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _partial_optimizer_counts(optimizers):
    result = {}
    for name, optimizer in optimizers.items():
        counts = []
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                step = optimizer.state.get(parameter, {}).get("step", 0)
                counts.append(int(step.item() if isinstance(step, torch.Tensor) else step))
        result[name] = dict(min=min(counts), max=max(counts), parameters=len(counts))
    return result


def _plots(output, arrays, metrics, history, public, font):
    plt = _plotting(font)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for name in LABELS:
        g, mapped = arrays[f"{name}_flow_g"], arrays[f"{name}_flow_R"]
        line, = axes[0].plot(g, mapped, label=LABELS[name])
        axes[1].plot(g, mapped - g, color=line.get_color(), label=LABELS[name])
        index = np.linspace(0, len(g) - 1, 8).astype(int)
        axes[1].quiver(g[index], np.zeros(len(index)), (mapped - g)[index], np.zeros(len(index)),
            color=line.get_color(), angles="xy", scale_units="xy", scale=1, width=.003)
    g = arrays["source_legacy_flow_g"]
    axes[0].plot(g, g, "k--", linewidth=1, label="恒等映射")
    axes[1].axhline(0, color="k", linestyle="--", linewidth=1)
    for ax, title, ylabel in zip(axes, ("固定表示下的重整化映射（不搜索根）", "真实位移箭头"), ("R(g)", "R(g)−g")):
        ax.set(title=title, xlabel="g", ylabel=ylabel)
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.savefig(output / "R_vs_g.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
    for name in CONTROL_NAMES:
        ax.plot([row["step"] for row in history], [row["arms"][name]["full_train_loss"] for row in history], label=LABELS[name])
    ax.set(title="固定缓存表示的全 TRAIN 闭环目标", xlabel="额外 R 更新步", ylabel="闭环目标（保留 5/5 或 10 权重）")
    ax.set_yscale("log")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.savefig(output / "loss.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11.5, 4.8), constrained_layout=True)
    paths = [("one_step", 1), ("one_step", 2), ("one_step", 3), ("rollout", 2), ("rollout", 3)]
    x = np.arange(len(paths))
    for name in LABELS:
        rows = [next(row for row in metrics[name] if row["split"] == "test" and row["branch"] == branch
                     and row["target_scale"] == target) for branch, target in paths]
        ax.plot(x, [row["model"]["fixed_train_nrmse"] for row in rows], marker="o", label=LABELS[name])
    rows = [next(row for row in metrics["source_legacy"] if row["split"] == "test" and row["branch"] == branch
                 and row["target_scale"] == target) for branch, target in paths]
    ax.plot(x, [row["identity"]["fixed_train_nrmse"] for row in rows], "k--", label="恒等映射")
    ax.set_xticks(x, ["32→16", "16→8", "8→4", "32→8（两步）", "32→4（三步）"])
    ax.set(title="留出物理链的固定 TRAIN 标准差归一误差", ylabel="NRMSE", xlabel="预测路径")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.savefig(output / "metrics.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
    values = public.endpoints["legacy"]
    for scale, size in enumerate(SIZES):
        ax.plot(values["grid_labels"][:, 0], values["grid_mean_direct"][:, scale, 0], marker="o", label=f"尺度 {scale}：{size}×{size}")
    ax.set(title="固定源表示的耦合到 g 映射（仅事后评价）", xlabel="r", ylabel="物理链平均 g")
    ax.grid(alpha=.2)
    ax.legend()
    fig.savefig(output / "coupling_by_scale.png", dpi=170)
    plt.close(fig)


def run(options: Options):
    options.validate()
    output = Path(options.output).resolve()
    training, r_state = Path(options.training).resolve(), Path(options.r_state).resolve()
    if output == training or training in output.parents or output == r_state.parent:
        raise ValueError("A fresh output outside the source training/R-state directories is required")
    output.mkdir(parents=True, exist_ok=False)
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    actual = {name: 0 for name in CONTROL_NAMES}
    latest = {name: 0 for name in CONTROL_NAMES}
    history, input_hashes, optimizers = [], {}, {}
    def budget():
        if time.process_time() - start_cpu > options.cpu_budget_seconds:
            raise BudgetHold("300 CPU-second diagnostic budget reached; no automatic continuation")
        if _directory_bytes(output) > options.directory_budget_bytes:
            raise BudgetHold("100 MiB diagnostic directory budget reached")
    def status(state, **extra):
        write_json(output / "status.json", dict(recipe=RECIPE, status=state,
            workflow_complete=state == "complete", actual_steps_by_arm=actual,
            latest_checkpoint_step_by_arm=latest, cpu_seconds=time.process_time() - start_cpu,
            wall_seconds=time.perf_counter() - start_wall, device="cpu", threads=1,
            requires_user_action=state in ("budget_hold", "failed"),
            returncode=0 if state == "complete" else 3 if state == "budget_hold" else 1 if state == "failed" else None,
            thermal_nu=None, critical_r=None, **extra))
    try:
        torch.set_num_threads(1)
        torch.manual_seed(options.seed)
        status("running", current="validate_sources")
        public, sources, extraction, input_hashes = load_sources(training, r_state)
        budget()
        direct = {split: public.endpoints["legacy"][f"{split}_direct"].copy() for split in ("train", "test")}
        train_tensor = torch.as_tensor(direct["train"], dtype=torch.float64)
        models = dict(continue_legacy=copy.deepcopy(sources["legacy"]),
                      continue_all_start=copy.deepcopy(sources["all_start"]))
        torch.manual_seed(options.seed)
        models["fresh_all_start"] = make_r()
        initial_hash = {name: state_fingerprint(model) for name, model in models.items()}
        optimizers = {name: torch.optim.Adam(model.parameters(), lr=options.learning_rate) for name, model in models.items()}
        batch_rng = torch.Generator(device="cpu").manual_seed(options.optimizer_seed)
        schedule_hash = hashlib.sha256()
        own_source_sha = sha256_file(__file__)
        manifest = dict(schema_version=5, recipe=RECIPE, options=asdict(options), device="cpu", threads=1,
            input_sha256=input_hashes, script_sha256=own_source_sha, extraction=extraction,
            initial_R_module_sha256=initial_hash, closure_modes=MODES,
            architecture="scalar g + Linear(1,96)-Tanh-Linear(96,96)-Tanh-Linear(96,1), float64",
            optimizer="fresh Adam per arm", requested_additional_steps=options.steps,
            source_QCR_steps=3000, source_QCR_steps_not_added_to_new_R_optimizer_counters=True,
            Q_loaded=False, C_loaded=False, Q_updates=0, C_updates=0, GPU_used=False,
            labels_used_by_optimizer=False, raw_physical_fields_loaded=False,
            source_fingerprints_required_unchanged=True, fixed_train_target_std=True,
            training_chain_count=32, test_chain_count=16, saved_fields_per_chain=128,
            conditional_C_realizations_per_chain=1, cached_g_used=True,
            original_training_target_distribution_reproduced=False,
            default_recipe_modified=False, thermal_nu=None, critical_r=None,
            limitations=LIMITATIONS)
        write_json(output / "manifest.json", manifest)
        for step in range(options.steps + 1):
            budget()
            if step:
                indices = torch.randperm(len(train_tensor), generator=batch_rng)[:options.batch_size]
                schedule_hash.update(indices.numpy().tobytes())
                batch = train_tensor[indices]
                for name in CONTROL_NAMES:
                    loss, _ = closure_loss(models[name], batch, MODES[name])
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite {name} objective")
                    optimizers[name].zero_grad(set_to_none=True)
                    loss.backward()
                    if any(parameter.grad is None or not torch.isfinite(parameter.grad).all()
                           for parameter in models[name].parameters()):
                        raise FloatingPointError(f"Invalid {name} gradient")
                    optimizers[name].step()
                    actual[name] = step
                    if any(not torch.isfinite(parameter).all() for parameter in models[name].parameters()):
                        raise FloatingPointError(f"Nonfinite {name} R parameter")
            if step % 100 == 0 or step == options.steps:
                with torch.no_grad():
                    row = dict(step=step, cpu_seconds=time.process_time() - start_cpu, arms={})
                    for name in CONTROL_NAMES:
                        _, terms = closure_loss(models[name], train_tensor, MODES[name])
                        row["arms"][name] = dict(full_train_loss=terms.pop("loss"), **terms)
                    history.append(row)
                write_json(output / "history_progress.json", dict(recipe=RECIPE, history=history,
                    actual_steps_by_arm=actual, latest_checkpoint_step_by_arm=latest))
                status("running", current="R_optimization")
            if step and (step % 1000 == 0 or step == options.steps):
                for name in CONTROL_NAMES:
                    folder = output / name
                    folder.mkdir(exist_ok=True)
                    torch.save(dict(recipe=RECIPE, step=step, R_state_dict=models[name].state_dict(),
                        closure_objective=MODES[name], source_input_sha256=input_hashes), folder / f"training_step_{step}.pt")
                    latest[name] = step
        status("running", current="evaluate_fixed_g")
        all_models = {"source_legacy": sources["legacy"], "source_all_start": sources["all_start"], **models}
        metrics, arrays = {}, {f"{split}_direct": value for split, value in direct.items()}
        train_input = direct["train"][:, :-1].ravel()
        all_input = np.concatenate([value[:, :-1].ravel() for value in direct.values()])
        low, high = float(all_input.min()), float(all_input.max())
        pad = .1 * max(high - low, 1e-6)
        grid = np.linspace(low - pad, high + pad, 256)
        for name, model in all_models.items():
            predictions_by_split = {split: predictions(model, value) for split, value in direct.items()}
            metrics[name] = evaluate_metrics(direct["train"], direct["test"], predictions_by_split)
            for split, predicted in predictions_by_split.items():
                arrays.update({f"{name}_{split}_{key}": value for key, value in predicted.items()})
            with torch.no_grad():
                mapped = advance(model, torch.as_tensor(grid[:, None], dtype=torch.float64)).numpy()[:, 0]
            arrays.update({f"{name}_flow_g": grid.copy(), f"{name}_flow_R": mapped,
                           f"{name}_flow_displacement": mapped - grid})
            budget()
        pchip, interpolation_arrays = pchip_control(direct["train"], direct["test"],
            public.endpoints["legacy"]["train_cell_ids"], public.endpoints["legacy"]["test_cell_ids"])
        for row in pchip["rows"]:
            split, scale = row["split"], row["source_scale"]
            covered = interpolation_arrays[f"pchip_{split}_scale{scale}_covered"]
            target = direct[split][covered, scale + 1]
            std = float(direct["train"][:, scale + 1].std(ddof=0))
            row["unified_R_on_same_covered_chains"] = {
                name: _error(arrays[f"{name}_{split}_one_step"][covered, scale], target, std)
                for name in all_models}
        arrays.update(interpolation_arrays)
        np.savez_compressed(output / "evaluation_arrays.npz", **arrays)
        budget()
        _plots(output, arrays, metrics, history, public, options.font)
        budget()
        if _file_sha([Path(path) for path in input_hashes]) != input_hashes:
            raise RuntimeError("Source artifact or original checkpoint changed during diagnostic")
        if sha256_file(__file__) != own_source_sha:
            raise RuntimeError("Diagnostic source code changed during execution")
        counts = optimizer_step_counts(optimizers, options.steps)
        if any(count["min"] != options.steps or count["max"] != options.steps for count in counts.values()):
            raise RuntimeError("Every R Adam parameter must have exactly 3000 additional updates")
        summary = dict(recipe=RECIPE, status="complete", workflow_complete=True, returncode=0,
            source_fingerprints_unchanged=True, script_sha256_unchanged=True,
            actual_steps_by_arm=actual, latest_checkpoint_step_by_arm=latest, optimizer_steps=counts,
            initial_R_module_sha256=initial_hash,
            final_R_module_sha256={name: state_fingerprint(model) for name, model in models.items()},
            paired_batch_schedule_sha256=schedule_hash.hexdigest(), metrics=metrics,
            per_transition_interpolation=pchip, training_input_range=[float(train_input.min()), float(train_input.max())],
            flow_grid_scope="all saved input g range plus ten percent, 256 fixed points; no root search",
            cpu_seconds=time.process_time() - start_cpu, wall_seconds=time.perf_counter() - start_wall,
            directory_bytes=_directory_bytes(output), GPU_seconds=0, device="cpu", threads=1,
            thermal_nu=None, critical_r=None, physical_critical_point_certified=False,
            equilibrium_certified=False, source_data_warning_count=public.manifest.get("data_warning_count"),
            source_data_warnings_retained=True, limitations=LIMITATIONS)
        write_json(output / "summary.json", summary)
        write_json(output / "history_progress.json", dict(recipe=RECIPE, history=history,
            actual_steps_by_arm=actual, latest_checkpoint_step_by_arm=latest))
        status("complete", current="complete", error=None,
            manifest_sha256=sha256_file(output / "manifest.json"), summary_sha256=sha256_file(output / "summary.json"),
            history_progress_sha256=sha256_file(output / "history_progress.json"),
            evaluation_arrays_sha256=sha256_file(output / "evaluation_arrays.npz"))
        return summary
    except BaseException as error:
        held = isinstance(error, BudgetHold)
        status("budget_hold" if held else "failed", current="stopped_without_restart", error=str(error),
            traceback=traceback.format_exc(), optimizer_steps=_partial_optimizer_counts(optimizers))
        if held:
            return dict(recipe=RECIPE, status="budget_hold", workflow_complete=False, returncode=3, error=str(error))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training", "r-state", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--font")
    parser.add_argument("--cpu-budget-seconds", type=float, default=300.)
    parser.add_argument("--directory-budget-bytes", type=int, default=100 * 1024 ** 2)
    options = Options(**vars(parser.parse_args(argv)))
    result = run(options)
    print(json.dumps(dict(status=result["status"], output=options.output, thermal_nu=None), ensure_ascii=False), flush=True)
    return 0 if result["status"] == "complete" else 3


if __name__ == "__main__":
    raise SystemExit(main())
