"""Frozen, paired C-symmetry counterfactual for a completed phi4 b=2 run.

No model updates, kernel selection, MI calculation or root search is performed.
Original fields remain local to the source host. Only small derived statistics
are written. The oddized mean is a diagnostic intervention, not a replacement
trained model or a claim of correct physical renormalization.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from experiments.phi4_b2_closure_comparison import (
    EVALUATION_SEED, RECIPE as SOURCE_RECIPE, load_dataset,
)
from experiments.phi4_qcr import (
    _device_receipt, module_fingerprints, sha256_file, state_fingerprint, tensor_fingerprint, write_json,
)
from recurrent_rg.phi4_b2_qcr import Phi4B2QCR


RECIPE = "phi4_b2_frozen_C_symmetry_diagnostic_v1"
RAW_STAT_NAMES = ("mean", "std_ddof0", "rms", "mean_abs", "raw_third_moment", "central_third_moment")
ODD_STAT_NAMES = ("mean_abs", "rms", "mean_abs_over_train_rms", "rms_over_train_rms")


class BudgetHold(RuntimeError):
    pass


@dataclass
class Options:
    source: str
    output: str
    device: str = "cuda:2"
    expected_gpu_uuid: str | None = None
    expected_checkpoint_sha256: str | None = None
    cpu_budget_seconds: float = 300.
    gpu_budget_seconds: float = 180.
    max_cuda_bytes: int = 8 * 1024**3

    def validate(self):
        for name in ("cpu_budget_seconds", "gpu_budget_seconds"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"Invalid {name}")
        if isinstance(self.max_cuda_bytes, bool) or self.max_cuda_bytes <= 0:
            raise ValueError("Invalid CUDA memory cap")
        if self.device not in ("cpu", "cuda:2"):
            raise ValueError("Only CPU or the explicitly fixed cuda:2 device is allowed")
        if self.device == "cuda:2" and not self.expected_gpu_uuid:
            raise ValueError("CUDA diagnostics require an expected physical GPU UUID")
        source, output = Path(self.source).resolve(), Path(self.output).resolve()
        if output.exists():
            raise FileExistsError("Never overwrite or resume a diagnostic")
        if source == output or source in output.parents or output in source.parents:
            raise ValueError("Diagnostic output must be independent of the source")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _training_root(source):
    root = Path(source).resolve()
    return root / "training" if (root / "training" / "manifest.json").is_file() else root


def load_source(source, expected_checkpoint_sha256=None):
    """Strict weights-only loading, endpoint tensor and public SHA validation."""
    root = _training_root(source)
    manifest, summary, status = [read_json(root / f"{name}.json")
                                 for name in ("manifest", "summary", "status")]
    if (manifest.get("recipe") != SOURCE_RECIPE or summary.get("recipe") != SOURCE_RECIPE
            or summary.get("status") != "complete" or status.get("status") != "complete"
            or status.get("workflow_complete") is not True or status.get("returncode") != 0
            or manifest.get("theory_dim") != 1 or manifest.get("lattice_sizes") != [32, 16, 8, 4]
            or summary.get("actual_steps") != 3000
            or summary.get("C_Q_pairing_verified") is not True):
        raise ValueError("Expected the completed paired scalar b=2 3000-step endpoint")
    pins = {}
    for key, name in (("manifest_sha256", "manifest.json"), ("summary_sha256", "summary.json"),
                      ("evaluation_arrays_sha256", "evaluation_arrays.npz")):
        path = root / name
        digest = sha256_file(path)
        if status.get(key) != digest:
            raise ValueError(f"Source receipt mismatch: {name}")
        pins[str(path)] = digest
    pins[str(root / "status.json")] = sha256_file(root / "status.json")
    import recurrent_rg.phi4_b2_qcr as model_module
    expected = [value for name, value in manifest.get("source_sha256", {}).items()
                if str(name).endswith("/recurrent_rg/phi4_b2_qcr.py")]
    if len(expected) != 1 or expected[0] != sha256_file(model_module.__file__):
        raise ValueError("Frozen model source differs from the source endpoint")
    checkpoint = root / "all_start" / "training_step_3000.pt"
    digest = sha256_file(checkpoint)
    if expected_checkpoint_sha256 is not None and digest != expected_checkpoint_sha256:
        raise ValueError("Checkpoint SHA256 differs from the pinned identity")
    pins[str(checkpoint)] = digest
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (saved.get("recipe") != SOURCE_RECIPE or saved.get("step") != 3000
            or saved.get("closure_objective") != "all_start_anchor"):
        raise ValueError("Wrong checkpoint recipe, step or closure objective")
    with torch.random.fork_rng(devices=[]):
        model = Phi4B2QCR(manifest["scale"], hidden=int(saved["model"]["C.net.0.weight"].shape[0]), theory_dim=1)
    model.load_state_dict(saved["model"], strict=True)
    model.eval().requires_grad_(False)
    modules = module_fingerprints(model)
    if modules != summary["arms"]["all_start"]["final_module_sha256"]:
        raise ValueError("Source endpoint tensor fingerprint mismatch")
    if any(modules[key] != summary["arms"]["legacy"]["final_module_sha256"][key] for key in ("Q", "C")):
        raise ValueError("Source Q/C pair identity is not verified")
    return model, manifest, summary, root, pins


@torch.no_grad()
def raw_field_statistics(field):
    """One physical chain pooled over saved fields/sites; no Z2 augmentation."""
    values = field.flatten()
    mean = values.mean()
    return torch.stack((mean, values.std(unbiased=False), values.square().mean().sqrt(),
                        values.abs().mean(), values.pow(3).mean(), (values - mean).pow(3).mean()))


@torch.no_grad()
def feature_statistics(model, field):
    half = field.shape[1] // 2
    if half < 1 or field.shape[1] % 2:
        raise ValueError("Even sample halves are needed for exact source evaluation replay")
    first, second = model.features(field[:, :half]), model.features(field[:, half:])
    target = .5 * (first + second)
    g = .5 * (model.Q(first) + model.Q(second))
    bins = target[0, :5] * 5.
    mirror = (bins - bins.flip(0)).abs().sum()
    endpoint_mass = bins[0] + bins[-1]
    return g[0], target[0], mirror, endpoint_mass


@torch.no_grad()
def paired_paths(model, field, original_rng, counterfactual_rng):
    """Paired Gaussian draws; only the deterministic mean is oddized."""
    if field.shape[2] % 8:
        raise ValueError("Three b=2 transitions require L divisible by eight")
    original, counterfactual, residual = [field], [field], []
    for _ in range(3):
        plus = model.C.mean_field(original[-1])
        minus = model.C.mean_field(-original[-1])
        even = .5 * (plus + minus)
        mean_abs, rms = even.abs().mean(), even.square().mean().sqrt()
        residual.append(torch.stack((mean_abs, rms, mean_abs / model.scale, rms / model.scale)))
        noise_original = torch.randn(plus.shape, dtype=plus.dtype, device=plus.device, generator=original_rng)
        cf_plus = model.C.mean_field(counterfactual[-1])
        cf_minus = model.C.mean_field(-counterfactual[-1])
        noise_cf = torch.randn(cf_plus.shape, dtype=cf_plus.dtype, device=cf_plus.device, generator=counterfactual_rng)
        if not torch.equal(noise_original, noise_cf):
            raise RuntimeError("Paired Gaussian-noise identity failed")
        original.append(plus + model.C.noise_std * noise_original)
        counterfactual.append(.5 * (cf_plus - cf_minus) + model.C.noise_std * noise_cf)
    if not torch.equal(original_rng.get_state(), counterfactual_rng.get_state()):
        raise RuntimeError("Counterfactual RNG state differs from original")
    return original, counterfactual, torch.stack(residual)


def compare_replay(arrays, dataset, source_arrays, atol=1e-10):
    """Every original chain, scale and feature is checked, not just means."""
    errors = {}
    for split in ("train", "test"):
        mask = dataset.splits == split
        for key in ("direct", "feature_target"):
            replay = arrays[f"original_{key}"][mask]
            expected = source_arrays[f"all_start_{split}_{key}"]
            if replay.shape != expected.shape or not np.isfinite(expected).all():
                raise ValueError(f"Source replay shape/finite mismatch: {split}/{key}")
            error = float(np.max(np.abs(replay - expected)))
            errors[f"{split}_{key}_max_abs"] = error
            if error > atol:
                raise ValueError(f"Original C replay differs from saved evaluation: {split}/{key}: {error}")
        for key, values in (("cell_ids", dataset.cell_ids), ("labels", dataset.labels)):
            if not np.array_equal(values[mask], source_arrays[f"all_start_{split}_{key}"]):
                raise ValueError(f"Source physical chain order mismatch: {split}/{key}")
    return dict(verified=True, absolute_tolerance=atol, all_saved_chains_and_scales=True, errors=errors)


def _plot(arrays, output):
    from scripts.report_phi4_qcr import _plotting
    plt = _plotting(None)
    cells = np.unique(arrays["cell_ids"])
    scales = np.arange(4)
    for filename, title, keys, ylabel in (
        ("amplitude_by_scale.png", "冻结 C 的幅度：原路径与奇对称反事实", ["rms", "mean_abs"], "场幅度 / TRAIN 微观 RMS"),
        ("symmetry_by_scale.png", "冻结 C 的镜像对称性：原路径与奇对称反事实", ["mirror"], "单点软分箱镜像 L1 差"),
        ("coupling_by_scale.png", "冻结 Q 的 r→g：原路径与奇对称反事实", ["g"], "g"),
    ):
        fig, ax = plt.subplots(figsize=(9, 5))
        for path, label, linestyle in (("original", "原路径", "-"), ("counterfactual", "奇对称反事实", "--")):
            if keys == ["g"]:
                for k in range(4):
                    x = [arrays["labels"][np.flatnonzero(arrays["cell_ids"] == c)[0], 0] for c in cells]
                    y = [arrays[f"{path}_direct"][arrays["cell_ids"] == c, k, 0].mean() for c in cells]
                    ax.plot(x, y, linestyle, marker="o", label=f"{label}，k={k}", color=f"C{k}")
                ax.set_xlabel("物理参数 r（仅事后评价）")
            else:
                for j, key in enumerate(keys):
                    if key == "mirror":
                        values = arrays[f"{path}_mirror_softbin_L1"]
                        name = "镜像差"
                    else:
                        values = arrays[f"{path}_raw_stats"][..., RAW_STAT_NAMES.index(key)] / float(arrays["train_rms"])
                        name = "均方根" if key == "rms" else "平均绝对值"
                    ax.plot(scales, values.mean(0), linestyle, marker="o", color=f"C{j}", label=f"{label}：{name}")
                ax.set_xlabel("粗粒化尺度 k：32→16→8→4")
                ax.set_xticks(scales)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / filename, dpi=150)
        plt.close(fig)


def run(options):
    options.validate()
    output = Path(options.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    device = torch.device(options.device)
    gpu_start, peak = None, 0

    def budget():
        nonlocal peak
        if time.process_time() - cpu_start > options.cpu_budget_seconds:
            raise BudgetHold("CPU diagnostic budget reached; no shorter rerun")
        if device.type == "cuda" and gpu_start is not None:
            torch.cuda.synchronize(device)
            peak = max(peak, int(torch.cuda.max_memory_allocated(device)), int(torch.cuda.max_memory_reserved(device)))
            if peak > options.max_cuda_bytes or time.perf_counter() - gpu_start > options.gpu_budget_seconds:
                raise BudgetHold("GPU diagnostic time or peak budget reached; no migration/rerun")

    def status(value, **kw):
        write_json(output / "status.json", dict(recipe=RECIPE, status=value,
            workflow_complete=value == "complete", returncode=0 if value == "complete" else None,
            error=None, requires_user_action=False, **kw))

    try:
        torch.set_num_threads(1)
        status("validating_source")
        model, manifest, summary, training, pins = load_source(options.source, options.expected_checkpoint_sha256)
        data_root = Path(manifest["data_root"]).resolve()
        dataset = load_dataset(data_root)
        if dataset.manifest_sha256 != manifest["data_manifest_sha256"] or dataset.receipts != manifest["data_shards"]:
            raise ValueError("Diagnostic physical data do not match source training")
        for row in dataset.receipts:
            for name, digest in (("path", "sha256"), ("sidecar", "sidecar_sha256")):
                pins[str(data_root / row[name])] = row[digest]
        pins[str(data_root / "manifest.json")] = dataset.manifest_sha256
        if float(model.scale) != dataset.scale:
            raise ValueError("TRAIN microscopic RMS differs from source normalization")
        with np.load(training / "evaluation_arrays.npz", allow_pickle=False) as archive:
            source_arrays = {key: archive[key] for key in archive.files if key.startswith("all_start_")}
        budget()
        gpu_start = time.perf_counter() if device.type == "cuda" else None
        receipt = _device_receipt(device, options.expected_gpu_uuid)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.to(device).eval().requires_grad_(False)
        initial_state = state_fingerprint(model)
        original_rng = torch.Generator(device=device).manual_seed(EVALUATION_SEED)
        counterfactual_rng = torch.Generator(device=device).manual_seed(EVALUATION_SEED)
        rows = []
        status("evaluating_frozen_counterfactual")
        with torch.no_grad():
            zero_tensor = model.C.mean_blocks(torch.zeros((1, 4), device=device, dtype=torch.float64))
            zero, zero_sha = zero_tensor.item(), tensor_fingerprint(zero_tensor)
            for chain in dataset.configurations:
                original, counterfactual, odd = paired_paths(model, chain.unsqueeze(0).to(device), original_rng, counterfactual_rng)
                row = {"original_input_odd_residual": odd.cpu().numpy()}
                for name, levels in (("original", original), ("counterfactual", counterfactual)):
                    raw, g, target, mirrors, endpoint = [], [], [], [], []
                    for field in levels:
                        a, b, c, d = feature_statistics(model, field)
                        raw.append(raw_field_statistics(field))
                        g.append(a); target.append(b); mirrors.append(c); endpoint.append(d)
                    for key, values in (("raw_stats", raw), ("direct", g), ("feature_target", target),
                                        ("mirror_softbin_L1", mirrors), ("softbin_endpoint_mass", endpoint)):
                        row[f"{name}_{key}"] = torch.stack(values).cpu().numpy()
                rows.append(row)
                budget()
        arrays = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
        if any(not np.isfinite(value).all() for value in arrays.values()):
            raise FloatingPointError("Nonfinite frozen counterfactual statistic")
        arrays.update(cell_ids=dataset.cell_ids, labels=dataset.labels, splits=dataset.splits,
                      train_rms=np.asarray(dataset.scale), lattice_sizes=np.asarray([32, 16, 8, 4]))
        replay = compare_replay(arrays, dataset, source_arrays)
        budget()
        gpu_seconds = time.perf_counter() - gpu_start if gpu_start is not None else 0.
        if initial_state != state_fingerprint(model) or any(p.grad is not None for p in model.parameters()):
            raise RuntimeError("Frozen diagnostic changed model tensors or created parameter gradients")
        for path, digest in pins.items():
            if sha256_file(path) != digest:
                raise RuntimeError(f"Read-only source changed during diagnostic: {path}")
        np.savez(output / "diagnostic_arrays.npz", **arrays)
        limitations = [
            "冻结原 Q/C/R，参数更新数为零；奇对称化仅是反事实诊断，未训练、未选核，也不改变默认模型。",
            "同一物理链起点及相同 Gaussian 噪声逐元素配对；反事实路径的中间场与原路径可以不同。",
            "冻结 Q 在改变后的 C 路径上可能分布外；g 的变化不直接证明闭合或物理正确性改善。",
            "镜像软分箱 L1、原始场矩与有限链统计均为描述量；不认证热化、相变、临界指数或真实 RG。",
            "保留源采样质量警告；不增加样本、不改 seed、不计算 MI、不搜索新不动点。",
        ]
        diagnostic = dict(recipe=RECIPE, status="complete", source=str(training), options=asdict(options),
            checkpoint_step=3000, model_updates=0, raw_stat_names=list(RAW_STAT_NAMES),
            odd_stat_names=list(ODD_STAT_NAMES), source_replay=replay, paired_noise_identical=True,
            C_zero_mean=zero, C_zero_mean_over_train_rms=zero / dataset.scale,
            C_zero_mean_sha256=zero_sha, fixed_noise_std=float(model.C.noise_std),
            counterfactual_mean_formula="0.5*(Cmean(x)-Cmean(-x)); same fixed Gaussian noise as original",
            source_data_warning_count=dataset.warning_count, evaluation_seed=EVALUATION_SEED,
            number_physical_chains=len(rows), sizes=[32, 16, 8, 4],
            source_module_sha256=module_fingerprints(model), all_parameter_gradients_none=True,
            source_and_model_unchanged=True, labels_used_only_for_post_evaluation=True,
            original_mean_raw_stats=arrays["original_raw_stats"].mean(0).tolist(),
            counterfactual_mean_raw_stats=arrays["counterfactual_raw_stats"].mean(0).tolist(),
            original_mean_mirror_softbin_L1=arrays["original_mirror_softbin_L1"].mean(0).tolist(),
            counterfactual_mean_mirror_softbin_L1=arrays["counterfactual_mirror_softbin_L1"].mean(0).tolist(),
            original_input_mean_odd_residual=arrays["original_input_odd_residual"].mean(0).tolist(),
            gpu_seconds=gpu_seconds, peak_cuda_bytes=peak, device=receipt,
            physical_critical_point_certified=False, thermal_nu=None, limitations=limitations)
        write_json(output / "diagnostic.json", diagnostic)
        _plot(arrays, output)
        budget()
        public = {path.name: sha256_file(path) for path in sorted(output.iterdir()) if path.is_file() and path.name != "status.json"}
        write_json(output / "receipt.json", dict(recipe=RECIPE, source_inputs_sha256=pins,
            public_files_sha256=public, source_replay_verified=True, source_unchanged=True,
            diagnostic_source_sha256=sha256_file(__file__),
            initial_model_state_sha256=initial_state, final_model_state_sha256=state_fingerprint(model),
            model_updates=0, no_raw_fields_written=True,
            cpu_seconds=time.process_time()-cpu_start, wall_seconds=time.perf_counter()-wall_start))
        status("complete", diagnostic_sha256=sha256_file(output / "diagnostic.json"),
               receipt_sha256=sha256_file(output / "receipt.json"), peak_cuda_bytes=peak)
        return diagnostic
    except Exception as error:
        hold = isinstance(error, BudgetHold)
        write_json(output / "status.json", dict(recipe=RECIPE, status="budget_hold" if hold else "failed",
            workflow_complete=False, returncode=3 if hold else 1, error=str(error),
            requires_user_action=True, traceback=traceback.format_exc()))
        if hold:
            return dict(status="budget_hold")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:2"), default="cuda:2")
    parser.add_argument("--expected-gpu-uuid")
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--cpu-budget-seconds", type=float, default=300.)
    parser.add_argument("--gpu-budget-seconds", type=float, default=180.)
    parser.add_argument("--max-cuda-bytes", type=int, default=8 * 1024**3)
    result = run(Options(**vars(parser.parse_args())))
    if result.get("status") == "budget_hold":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
