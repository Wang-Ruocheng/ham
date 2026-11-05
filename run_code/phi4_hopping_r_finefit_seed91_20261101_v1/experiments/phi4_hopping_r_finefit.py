"""CPU-only refinement of R on immutable, already encoded phi4 ensembles.

No Q/C object is constructed, no physical configurations are opened, and no
coupling enters the optimizer.  The sole objective is the existing equal-target
all-start closure, in the unchanged physical g coordinate.  CLI planning is dry
unless --execute is explicit.  The source run is never modified.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from recurrent_rg.phi4_r_calibration import (
    FrozenResidualMap, all_start_closure_loss, conditioned_copy,
    fit_input_normalizer,
)


RECIPE = "phi4_hopping_frozen_QC_R_finefit_v1"
SOURCE_RECIPE = "phi4_hopping_raw_patch_joint625_v1"
SOURCE_STATE = "510bb18ff3c9ea2a8abebe7ae4b49959daa2c15ea1b94df8704e7b4f6efce680"
SOURCE_MODULES = {
    "Q": "b155f9174280923aee620c95930f64dfedfe7729ea5a39560f2a63d97b445ee6",
    "C": "4cadcd500903dd2da379a2e84688a4954f6c5646d7ff79ad5da60691af6c878d",
    "R": "f29fb2bd3fd7f0bba975cb00d0cb94a9de3fc181633762cc3109b53d34498220",
}
R_SHAPES = {"0.weight": (96, 1), "0.bias": (96,),
            "2.weight": (96, 96), "2.bias": (96,),
            "4.weight": (1, 96), "4.bias": (1,)}
ADAM_STEPS, LBFGS_ITERATIONS, LBFGS_EVALUATIONS = 10000, 2000, 8000
CPU_BUDGET_SECONDS, OUTPUT_BUDGET_BYTES = 600., 64 * 1024 ** 2
PATHS = (("0_to_1", 0, 1), ("1_to_2", 1, 2), ("0_to_2", 0, 2))
DISPLAY = {"source": "原联合训练 R", "adam10000": "仅 R：Adam 一万步",
           "lbfgs": "仅 R：最终精细拟合", "budget_hold_partial": "预算停止时的 R"}


@dataclass(frozen=True)
class Options:
    source_root: str
    output: str
    font: str | None = None


class BudgetHold(RuntimeError):
    pass


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


def state_fingerprint(state):
    """Match the formal source's tensor/state algorithm without model imports."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if not torch.is_tensor(tensor):
            raise ValueError("State contains a nontensor")
        array = tensor.detach().cpu().contiguous().numpy()
        item = hashlib.sha256()
        for part in (str(array.dtype).encode(), str(array.shape).encode(), array.tobytes()):
            item.update(part)
        digest.update(name.encode())
        digest.update(item.hexdigest().encode())
    return digest.hexdigest()


def extract_raw_r(state):
    """Copy precisely six R tensors; Q/C are never instantiated or invoked."""
    selected = {name[2:]: tensor for name, tensor in state.items() if name.startswith("R.")}
    if set(selected) != set(R_SHAPES):
        raise ValueError("Source R must contain precisely six expected tensors")
    for name, tensor in selected.items():
        if (not torch.is_tensor(tensor) or tuple(tensor.shape) != R_SHAPES[name]
                or tensor.dtype != torch.float64 or tensor.device.type != "cpu"
                or not torch.isfinite(tensor).all()):
            raise ValueError(f"Invalid R tensor {name}")
    # All randomly constructed values are overwritten. Preserve caller RNG.
    with torch.random.fork_rng(devices=[]):
        raw = FrozenResidualMap(theory_dim=1, hidden=96)
    raw.net.load_state_dict(selected, strict=True)
    return raw


def split_frozen_g(direct, splits, cells):
    direct, splits, cells = np.asarray(direct), np.asarray(splits), np.asarray(cells)
    if (direct.shape != (60, 3, 1) or direct.dtype != np.float64
            or not np.isfinite(direct).all() or splits.shape != (60,)
            or cells.shape != (60,) or cells.dtype.kind not in "iu"
            or set(splits.tolist()) != {"train", "test"}):
        raise ValueError("Expected source 60 frozen physical-chain encodings")
    train = splits == "train"
    if train.sum() != 40 or (~train).sum() != 20:
        raise ValueError("Expected 40 TRAIN and 20 TEST chains")
    if len(np.unique(cells)) != 15:
        raise ValueError("Expected 15 physical conditions")
    for cell in np.unique(cells):
        mask = cells == cell
        if mask.sum() != 4 or len(np.unique(splits[mask])) != 1:
            raise ValueError("Four physical chains per whole-condition split required")
    return torch.tensor(direct[train].copy(), dtype=torch.float64), train


def load_source(root):
    """Read completed public g caches and one trusted final checkpoint only.

    NPZ direct_g contains both splits; TEST may be read into memory but is not
    supplied to normalization, optimization, convergence or endpoint selection.
    torch.load necessarily deserializes checkpoint tensor storage; Q/C modules
    are not constructed. Only R tensors survive checkpoint verification.
    """
    root = Path(root).resolve()
    run = root / "training"
    names = ("job.json", "public_files.json", "training/status.json",
             "training/manifest.json", "training/summary.json",
             "training/evaluation_arrays.npz", "training/training_step_3000.pt",
             "training/report/coupling_by_scale.png")
    pins = {name: sha256_file(root / name) for name in names}
    job, status, manifest, summary = [json.loads((root / name).read_text())
        for name in ("job.json", "training/status.json", "training/manifest.json", "training/summary.json")]
    for state in (job, status):
        if (state.get("status") != "complete" or state.get("workflow_complete") is not True
                or state.get("returncode") != 0):
            raise ValueError("Source workflow/training is not successfully complete")
    if any(stage.get("returncode") != 0 for stage in job.get("stages", [])
           if stage.get("status") != "skipped"):
        raise ValueError("An executed source stage did not return zero")
    if (manifest.get("schema_version") != 5 or manifest.get("recipe") != SOURCE_RECIPE
            or summary.get("recipe") != SOURCE_RECIPE or summary.get("status") != "complete"
            or manifest.get("options", {}).get("steps") != 3000
            or summary.get("actual_steps") != 3000 or summary.get("latest_checkpoint_step") != 3000
            or manifest.get("closure_objective") != "all_start_anchor"
            or manifest.get("closure_scale_weights") is not None
            or manifest.get("labels_used_in_training") is not False
            or summary.get("labels_used_in_training") is not False
            or manifest.get("theory_dim") != 1 or manifest.get("hidden") != 96
            or manifest.get("dtype") != "float64"
            or summary.get("final_state_sha256") != SOURCE_STATE
            or summary.get("final_module_sha256") != SOURCE_MODULES
            or summary.get("frozen_evaluation_verified") is not True
            or manifest.get("evaluation_encoding") != "mean of Q(first half fields) and Q(second half fields); direct_full_g separately encodes pooled fields"):
        raise ValueError("Source recipe, endpoint, fingerprints or encoding policy mismatch")
    for name in ("Q", "C", "R"):
        clock = summary.get("optimizer_steps", {}).get(name, {})
        if clock.get("min") != 3000 or clock.get("max") != 3000:
            raise ValueError("Source optimizer clocks incomplete")
    inventory = json.loads((root / "public_files.json").read_text())
    for name in ("training/manifest.json", "training/summary.json",
                 "training/evaluation_arrays.npz", "training/report/coupling_by_scale.png"):
        if inventory.get(name) != pins[name]:
            raise ValueError(f"Public source SHA mismatch: {name}")
    checkpoint_path = run / "training_step_3000.pt"
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        loading = "weights_only=True"
    except (TypeError, __import__("pickle").UnpicklingError):
        # Restricted to the explicitly supplied, trusted formal checkpoint.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        loading = "trusted formal checkpoint; weights_only=False fallback"
    if (checkpoint.get("recipe") != SOURCE_RECIPE or checkpoint.get("step") != 3000
            or checkpoint.get("manifest") != manifest):
        raise ValueError("Checkpoint and public manifest differ")
    state = checkpoint["model"]
    if state_fingerprint(state) != SOURCE_STATE:
        raise ValueError("Complete checkpoint state fingerprint mismatch")
    for module in ("Q", "C", "R"):
        subset = {key[len(module) + 1:]: value for key, value in state.items()
                  if key.startswith(module + ".")}
        if state_fingerprint(subset) != SOURCE_MODULES[module]:
            raise ValueError(f"Checkpoint {module} fingerprint mismatch")
    raw = extract_raw_r(state)
    del checkpoint, state
    with np.load(run / "evaluation_arrays.npz", allow_pickle=False) as archive:
        # Feature reconstruction arrays/raw physical fields are not requested.
        arrays = {name: archive[name].copy() for name in (
            "direct_g", "splits", "cell_ids", "flow_grid", "flow_next",
            "one_step_g", "rollout_g")}
    train, train_mask = split_frozen_g(arrays["direct_g"], arrays["splits"], arrays["cell_ids"])
    if not np.isfinite(arrays["flow_grid"]).all() or arrays["flow_grid"].shape != (501, 1):
        raise ValueError("Invalid original TRAIN-only flow grid")
    with torch.no_grad():
        probes = {"flow": raw(torch.tensor(arrays["flow_grid"])).numpy(),
                  "one": np.stack([raw(train[:, k]).numpy() for k in (0, 1)], axis=1)}
    if (not np.allclose(probes["flow"], arrays["flow_next"], rtol=1e-11, atol=1e-12)
            or not np.allclose(probes["one"], arrays["one_step_g"][train_mask], rtol=1e-11, atol=1e-12)):
        raise ValueError("Extracted source residual R differs from formal saved predictions")
    return raw, train, train_mask, arrays, manifest, summary, pins, loading


def preservation_receipt(raw, conditioned, train):
    x = train[:, :2].reshape(-1, 1).detach().requires_grad_(True)
    a, b = raw(x), conditioned(x)
    da = torch.autograd.grad(a.sum(), x)[0]
    db = torch.autograd.grad(b.sum(), x)[0]
    result = {"max_function_error": float((a - b).detach().abs().max()),
              "max_full_derivative_error": float((da - db).detach().abs().max())}
    if max(result.values()) > 1e-11:
        raise ValueError("Internal normalization changed initial physical F or DF")
    return result


def predict(model, direct):
    with torch.no_grad():
        direct = torch.as_tensor(direct, dtype=torch.float64)
        p01, p12 = model(direct[:, 0]), model(direct[:, 1])
        return np.stack([p01.numpy(), p12.numpy(), model(p01).numpy()], axis=1)


def endpoint_metrics(direct, prediction, train_mask, target_stds):
    result = {}
    for name, mask in (("train", train_mask), ("test", ~train_mask)):
        result[name] = {}
        for column, (path, start, end) in enumerate(PATHS):
            error = prediction[mask, column] - direct[mask, end]
            identity_error = direct[mask, start] - direct[mask, end]
            rmse = float(np.sqrt(np.mean(error ** 2)))
            std = float(target_stds[end - 1])
            identity = float(np.sqrt(np.mean(identity_error ** 2)))
            result[name][path] = dict(rmse=rmse, nrmse=rmse / std if std > 0 else None,
                train_target_std=std, identity_rmse=identity,
                mean_error=float(error.mean()), max_abs_error=float(np.abs(error).max()),
                physical_chains=int(mask.sum()))
    return result


def train_full_batch(model, train, *, budget, record, checkpoint,
                     adam_steps=ADAM_STEPS, lbfgs_iterations=LBFGS_ITERATIONS,
                     lbfgs_evaluations=LBFGS_EVALUATIONS):
    """Only fixed TRAIN tensor enters this optimizer; no evaluation callback."""
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    actual = {"adam_steps": 0, "lbfgs_iterations": 0, "lbfgs_closure_calls": 0}
    def row(phase, index):
        with torch.no_grad():
            loss, terms = all_start_closure_loss(model, train)
        record(dict(phase=phase, index=index, loss=float(loss),
                    **{name: float(value) for name, value in terms.items()}, **actual))
    row("adam", 0)
    for step in range(1, adam_steps + 1):
        budget()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = all_start_closure_loss(model, train)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite TRAIN closure loss")
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("Missing/nonfinite R gradients")
        optimizer.step()
        actual["adam_steps"] = step
        if step % 100 == 0 or step == adam_steps:
            row("adam", step)
        if step % 1000 == 0 or step == adam_steps:
            checkpoint("adam_progress" if step < adam_steps else "adam10000", actual, optimizer)
    clocks = [int(value["step"]) for value in optimizer.state.values()]
    if len(clocks) != 6 or set(clocks) != {adam_steps}:
        raise RuntimeError("Not every R Adam parameter reached the fixed endpoint")
    # LBFGS line-search closures can inspect a trial parameter vector. If a
    # resource guard fires there, restore the last unambiguously accepted
    # endpoint (Adam) rather than publishing that trial as the final solution.
    accepted_adam = {key: value.detach().clone() for key, value in model.state_dict().items()}
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1., max_iter=lbfgs_iterations,
        max_eval=lbfgs_evaluations, history_size=20, tolerance_grad=1e-9,
        tolerance_change=1e-12, line_search_fn="strong_wolfe")
    last = {}
    def closure():
        budget()
        optimizer.zero_grad(set_to_none=True)
        loss, terms = all_start_closure_loss(model, train)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite LBFGS TRAIN closure loss")
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("Missing/nonfinite LBFGS R gradients")
        actual["lbfgs_closure_calls"] += 1
        state = optimizer.state.get(next(iter(model.parameters())), {})
        actual["lbfgs_iterations"] = int(state.get("n_iter", 0))
        last.update(loss=float(loss.detach()), gradient_inf=max(float(p.grad.abs().max()) for p in model.parameters()))
        if actual["lbfgs_closure_calls"] == 1 or actual["lbfgs_closure_calls"] % 20 == 0:
            record(dict(phase="lbfgs", index=actual["lbfgs_closure_calls"],
                **last, **{name: float(value.detach()) for name, value in terms.items()}, **actual))
        if actual["lbfgs_closure_calls"] % 100 == 0:
            checkpoint("lbfgs_progress", actual, optimizer)
        return loss
    try:
        optimizer.step(closure)
    except BudgetHold as exc:
        model.load_state_dict(accepted_adam)
        exc.actual = actual.copy()
        exc.retained_endpoint = "adam10000"
        exc.trial_parameters_rolled_back = True
        raise
    state = optimizer.state[next(iter(model.parameters()))]
    actual["lbfgs_iterations"] = int(state.get("n_iter", 0))
    if int(state.get("func_evals", 0)) != actual["lbfgs_closure_calls"]:
        raise RuntimeError("LBFGS objective/gradient call counters disagree")
    # Evaluate gradient at the returned endpoint, not at a line-search trial.
    optimizer.zero_grad(set_to_none=True)
    loss, terms = all_start_closure_loss(model, train)
    loss.backward()
    if (not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all()
                                      for p in model.parameters())):
        raise FloatingPointError("Nonfinite final LBFGS objective/gradient")
    grad = max(float(p.grad.abs().max()) for p in model.parameters())
    reason = ("gradient_tolerance" if grad <= 1e-9 else
              "iteration_limit" if actual["lbfgs_iterations"] >= lbfgs_iterations else
              "evaluation_limit" if actual["lbfgs_closure_calls"] >= lbfgs_evaluations else
              "change_or_line_search_tolerance_not_gradient_converged")
    record(dict(phase="lbfgs_endpoint", index=actual["lbfgs_closure_calls"],
                loss=float(loss.detach()), gradient_inf=grad, **actual,
                **{name: float(value.detach()) for name, value in terms.items()}))
    checkpoint("lbfgs", actual, optimizer)
    return dict(**actual, gradient_inf=grad, gradient_converged=grad <= 1e-9,
                stopping_reason=reason, adam_parameter_steps=clocks,
                endpoint_gradient_calls=1, accepted_endpoint="returned_LBFGS_parameters",
                LBFGS_internal_func_evals=int(state.get("func_evals", 0)))


def render_plots(output, arrays, predictions, flows, history, font):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    if font is not None:
        font_path = Path(font)
    else:
        candidates = (Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
                      Path("/System/Library/Fonts/PingFang.ttc"))
        font_path = next((p for p in candidates if p.is_file()), None)
    if font_path is None or not font_path.is_file():
        raise ValueError("Chinese font required; supply --font")
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font_path)).get_name(),
                         "axes.unicode_minus": False, "font.size": 11})
    g, mask = arrays["direct_g"], arrays["splits"] == "train"
    grid = arrays["flow_grid"][:, 0]
    final = next(reversed(predictions))
    colors = {"source": "#777777", "adam10000": "#d28222", "lbfgs": "#0072b2", "budget_hold_partial": "#0072b2"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for name, mapped in flows.items():
        axes[0].plot(grid, mapped[:, 0], color=colors[name], label=DISPLAY[name])
        axes[1].plot(grid, mapped[:, 0] - grid, color=colors[name], label=DISPLAY[name])
    axes[0].plot(grid, grid, "k--", lw=1, label="恒等映射")
    axes[1].axhline(0, color="k", lw=.8)
    indices = np.linspace(0, len(grid) - 1, 25, dtype=int)
    drift = flows[final][:, 0] - grid
    axes[1].quiver(grid[indices], np.zeros(len(indices)), drift[indices], np.zeros(len(indices)),
        angles="xy", scale_units="xy", scale=1, color=colors[final], width=.004, alpha=.55)
    for ax, ylabel, title in zip(axes, ("R(g)", "R(g)−g"), ("原 R 与仅 R 精细拟合", "位移流场；箭头为实际 R(g)−g")):
        ax.set(xlabel="潜变量 g", ylabel=ylabel, title=title)
        ax.axvspan(g[mask, :2].min(), g[mask, :2].max(), color="#b0b0b0", alpha=.13, label="训练输入覆盖区")
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
    fig.savefig(output / "R_vs_g.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, (start, end) in zip(axes, ((0, 1), (1, 2))):
        for split, selected, marker in (("训练链", mask, "o"), ("留出链", ~mask, "x")):
            ax.scatter(g[selected, start, 0], g[selected, end, 0] - g[selected, start, 0],
                       marker=marker, s=27, label=f"直接编码：{split}")
        for name, mapped in flows.items():
            ax.plot(grid, mapped[:, 0] - grid, color=colors[name], label=DISPLAY[name])
        ax.axhline(0, color="k", lw=.8)
        ax.set(xlabel=f"输入 g_{start}", ylabel=f"直接编码 g_{end}−g_{start}",
               title=f"经验尺度分支 {start}→{end} 与 R 位移")
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
    fig.savefig(output / "empirical_branches.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, phase in zip(axes, ("adam", "lbfgs")):
        rows = [row for row in history if row["phase"] == phase]
        for key, label in (("loss", "加权闭合总损失"), ("e01", "单步 0→1"),
                           ("e12", "单步 1→2"), ("e02", "两步 0→2")):
            if rows:
                ax.semilogy([r["index"] for r in rows], [max(r[key], 1e-30) for r in rows], label=label)
        ax.set(xlabel="Adam 更新次数" if phase == "adam" else "LBFGS 目标与梯度调用次数",
               ylabel="原 g 单位的均方误差／加权损失", title="全训练集 Adam" if phase == "adam" else "全训练集 LBFGS")
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
    fig.savefig(output / "loss_curves.png", dpi=170)
    plt.close(fig)


def plan(options):
    return dict(schema_version=5, recipe=RECIPE, options=asdict(options),
        source_QCR_steps=3000, source_final_state_sha256=SOURCE_STATE,
        source_module_sha256=SOURCE_MODULES, R_initialization="source final raw residual R; function-preserving TRAIN-input normalization",
        model="1→96 tanh→96 tanh→1 residual; same capacity; float64; original g outputs",
        objective="5*e01 + 2.5*e12 + 2.5*e02", closure_objective="all_start_anchor",
        closure_scale_weights=None, recurrent_gradients_retained=True,
        optimizer=dict(adam_steps=ADAM_STEPS, adam_lr=.001, lbfgs_max_iter=LBFGS_ITERATIONS,
            lbfgs_max_eval=LBFGS_EVALUATIONS, lbfgs_history_size=20,
            lbfgs_tolerance_grad=1e-9, lbfgs_tolerance_change=1e-12, line_search="strong_wolfe"),
        endpoint_policy="source, fixed Adam10000, returned LBFGS endpoint; no TEST selection or TEST stopping",
        train_physical_chains=40, test_physical_chains=20, encoding="formal direct_g: mean Q(first64),Q(second64)",
        cache_qualification="One frozen C-noise evaluation per physical chain; not new physical or C-noise sampling",
        cache_vs_training_qualification="Cache:128 fields→two64mean; source training:64 fields→two32mean with fresh C noise",
        optimizer_qualification="Fresh Adam/LBFGS on function-preserving conditioned R; not exact source-optimizer continuation",
        CPU_threads=1, nice=10, GPU_used=False, cpu_budget_seconds=CPU_BUDGET_SECONDS,
        output_budget_bytes=OUTPUT_BUDGET_BYTES, report_reserve_cpu_seconds=45,
        labels_used_by_optimizer=False, TEST_used_by_optimizer=False,
        TEST_previously_inspected=True, Q_constructed=False, C_constructed=False,
        Q_calls=0, C_calls=0, Q_updates=0, C_updates=0, physical_sampling_calls=0,
        roots_searched=False, thermal_nu=None, original_run_mutated=False,
        Q_kappa_plot="source training/report/coupling_by_scale.png; fingerprint reference only, no Q invocation")


def run(options):
    root, output = Path(options.source_root).resolve(), Path(options.output).resolve()
    if output == root or root in output.parents or output in root.parents:
        raise ValueError("Output must be independent of immutable source")
    output.mkdir(parents=True, exist_ok=False)
    cpu_started, wall_started = time.process_time(), time.perf_counter()
    history, endpoints, predictions, flows, hashes, progress = [], {}, {}, {}, {}, {}
    endpoint_states = {}
    model, arrays, train, mask, stds, manifest = None, None, None, None, None, plan(options)
    phase = "verifying_source"
    def directory_bytes():
        return sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    def budget(reserve=45.):
        if time.process_time() - cpu_started >= CPU_BUDGET_SECONDS - reserve:
            raise BudgetHold("Independent 600 CPU-second budget reached (report reserve retained)")
        if directory_bytes() >= OUTPUT_BUDGET_BYTES:
            raise BudgetHold("Independent 64 MiB output budget reached")
    def status(state, **extra):
        write_json(output / "status.json", dict(recipe=RECIPE, status=state,
            workflow_complete=state == "complete", current=phase,
            actual=dict(progress), cpu_seconds=time.process_time() - cpu_started,
            wall_seconds=time.perf_counter() - wall_started, directory_bytes=directory_bytes(),
            returncode=0 if state == "complete" else 3 if state == "budget_hold" else 1 if state == "failed" else None,
            requires_user_action=state in ("failed", "budget_hold"), error=None,
            Q_calls=0, C_calls=0, Q_updates=0, C_updates=0, GPU_used=False, **extra))
    def record(row):
        progress.update({key: row[key] for key in ("adam_steps", "lbfgs_iterations", "lbfgs_closure_calls") if key in row})
        row["cpu_seconds"] = time.process_time() - cpu_started
        history.append(row)
        write_json(output / "history.json", history)
        status("running")
        print(json.dumps(row, allow_nan=False), flush=True)
    def endpoint(name):
        # Preserve fixed endpoints now. TEST metrics are deferred until all
        # optimization/stop decisions have finished, even for the source R.
        endpoint_states[name] = {key: value.detach().clone() for key, value in model.state_dict().items()}
    def checkpoint(name, actual, optimizer):
        progress.update(actual)
        torch.save(dict(recipe=RECIPE, state=model.state_dict(), optimizer=optimizer.state_dict(),
            actual=actual.copy(), source_pins=hashes,
            parameter_status=("line_search_trial_NOT_an_accepted_or_resumable_endpoint"
                              if name == "lbfgs_progress" else "accepted_endpoint"),
            automatic_resume_authorized=False), output / f"private_R_{name}.pt")
        if name in ("adam10000", "lbfgs"):
            endpoint(name)
        budget()
    def save_report():
        phase_before = phase
        write_json(output / "history.json", history)
        if arrays is not None and endpoint_states:
            retained = {key: value.detach().clone() for key, value in model.state_dict().items()}
            for name, state in endpoint_states.items():
                model.load_state_dict(state)
                predictions[name] = predict(model, arrays["direct_g"])
                with torch.no_grad():
                    flows[name] = model(torch.tensor(arrays["flow_grid"])).numpy()
                endpoints[name] = endpoint_metrics(arrays["direct_g"], predictions[name], mask, stds)
            model.load_state_dict(retained)
            np.savez(output / "evaluation_arrays.npz", direct_g=arrays["direct_g"],
                splits=arrays["splits"], cell_ids=arrays["cell_ids"], flow_grid=arrays["flow_grid"],
                **{f"{name}_predictions": value for name, value in predictions.items()},
                **{f"{name}_flow_next": value for name, value in flows.items()})
            render_plots(output, arrays, predictions, flows, history, options.font)
        manifest.update(input_sha256=hashes, source_unchanged={
            name: sha256_file(root / name) == digest for name, digest in hashes.items()})
        if not all(manifest["source_unchanged"].values()):
            raise RuntimeError("Immutable source fingerprint changed")
        if any(sha256_file(path) != digest for path, digest in manifest.get("code_sha256", {}).items()):
            raise RuntimeError("Pinned R finefit implementation changed")
        write_json(output / "manifest.json", manifest)
        write_json(output / "summary.json", dict(recipe=RECIPE, endpoint_metrics=endpoints,
            actual=progress, optimizer_result=manifest.get("optimizer_result"),
            cpu_seconds=time.process_time() - cpu_started, phase=phase_before,
            limitations=["仅固定一次 Q/C 编码缓存拟合，不代表物理系综总体误差。",
                "正式缓存每链 128 场取两半 64 的 Q 平均；原训练每批 64 场取两半 32，且更新 C 随机性。",
                "仅内部重参数化保留初始物理函数及导数；新 Adam/LBFGS 不恢复原优化器轨迹。",
                "留出集此前已检查，不是全新盲测；没有用于优化或停止选择。",
                "原物理采样质量告警保留；不能由低拟合误差认证平衡或物理不动点。"],
            source_sampling_warning_count=manifest.get("data_warning_count"), thermal_nu=None))
        public = {p.name: sha256_file(p) for p in output.iterdir()
                  if p.suffix in (".json", ".npz", ".png") and p.name not in ("public_files.json", "status.json")}
        write_json(output / "public_files.json", public)
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        if os.getpriority(os.PRIO_PROCESS, 0) < 10:
            os.nice(10 - os.getpriority(os.PRIO_PROCESS, 0))
        status("running")
        write_json(output / "plan.json", manifest)
        raw, train, mask, arrays, _, summary, hashes, loading = load_source(root)
        manifest["cache_vs_training_qualification"] = (
            "Source evaluation encodes mean Q(first64), Q(second64) from128 saved fields; "
            "joint training used random64-field batches and fresh C noise. This fits the fixed "
            "evaluation cache, not a reproduction of the complete stochastic training objective.")
        center, scale = fit_input_normalizer(train)
        model = conditioned_copy(raw, center, scale)
        manifest.update(normalizer_center=center.tolist(), normalizer_scale=scale.tolist(),
            normalizer_scope="only40 TRAIN chains, input g0/g1; ddof0; fixed",
            initialization_preservation=preservation_receipt(raw, model, train),
            R_parameters=sum(p.numel() for p in model.parameters()), source_loading=loading,
            source_module_sha256=summary["final_module_sha256"],
            data_warning_count=summary["data_warning_count"], actual_nice=os.getpriority(os.PRIO_PROCESS, 0),
            code_sha256={str(Path(__file__).resolve()): sha256_file(__file__),
                str(Path(__file__).parents[1] / "recurrent_rg/phi4_r_calibration.py"):
                    sha256_file(Path(__file__).parents[1] / "recurrent_rg/phi4_r_calibration.py")})
        stds = [float(train[:, k, :].std(unbiased=False)) for k in (1, 2)]
        manifest["TRAIN_target_stds_ddof0"] = stds
        endpoint("source")
        del raw
        write_json(output / "manifest.json", manifest)
        phase = "R_full_train_adam_then_lbfgs"
        result = train_full_batch(model, train, budget=budget, record=record, checkpoint=checkpoint)
        progress.update({key: result[key] for key in ("adam_steps", "lbfgs_iterations", "lbfgs_closure_calls")})
        manifest["optimizer_result"] = result
        phase = "report"
        save_report()
        budget(reserve=0)
        phase = None
        status("complete")
        return 0
    except BudgetHold as exc:
        if hasattr(exc, "actual"):
            progress.update(exc.actual)
        manifest["optimizer_result"] = dict(stopping_reason="budget_hold", gradient_converged=False,
            retained_endpoint=getattr(exc, "retained_endpoint", "latest_accepted_Adam_update"),
            LBFGS_trial_parameters_rolled_back=getattr(exc, "trial_parameters_rolled_back", False))
        if model is not None:
            torch.save(dict(recipe=RECIPE, state=model.state_dict(), actual=progress.copy(),
                source_pins=hashes, budget_hold=True), output / "private_R_budget_hold.pt")
            if arrays is not None:
                endpoint("budget_hold_partial")
        save_report()
        status("budget_hold", hold_reason=str(exc))
        return 3
    except Exception as exc:
        write_json(output / "error.json", dict(error=repr(exc), traceback=traceback.format_exc(), phase=phase))
        status("failed", error_detail=repr(exc))
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--font")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    options = Options(args.source_root, args.output, args.font)
    if not args.execute:
        print(json.dumps(dict(status="dry_plan", **plan(options)), indent=2, ensure_ascii=False))
        return 0
    return run(options)


if __name__ == "__main__":
    raise SystemExit(main())
