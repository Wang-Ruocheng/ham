"""TRAIN-only decoder attribution on immutable public b=2 endpoint arrays.

python -m scripts.diagnose_phi4_b2_decoders --training COMPLETE/training --output NEW_DIR

This is a descriptive public-statistic control, not QCR training. PCA controls
project observed target statistics and are geometric reconstruction oracles,
not learned physical coordinates or predictors from the previous RG scale.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
import json
from pathlib import Path
import time

import numpy as np
try:
    from threadpoolctl import threadpool_info, threadpool_limits
except ImportError:  # Optional runtime helper; never install for a diagnostic.
    threadpool_info = threadpool_limits = None

from scripts.report_phi4_b2_closure_comparison import RECIPE, SIZES, load_public_comparison
from scripts.report_phi4_qcr import INPUT_FILES, _plotting, sha256


METHOD = "phi4_b2_public_decoder_attribution_v1"
CONDITION_LIMIT = 1e8
LABELS = {
    "existing_shared_affine": "既有共享一维仿射解码",
    "pooled_pca_rank1": "共享一维 PCA 几何下界",
    "pooled_pca_rank2": "共享二维 PCA 几何下界",
    "per_scale_pca_rank1": "各尺度独立一维 PCA 下界",
    "fixed_g_affine": "固定旧 g：重拟合共享仿射",
    "fixed_g_polynomial": "固定旧 g：训练条件交叉验证多项式",
    "same_scale_train_constant": "本尺度训练均值基线",
}


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")


def _metric(prediction, target, reference):
    rmse = float(np.sqrt(np.mean(np.square(prediction - target))))
    std = float(np.sqrt(np.mean(np.var(reference, axis=0, ddof=0))))
    return dict(absolute_rmse=rmse, fixed_train_nrmse=rmse / std if std else None,
                train_target_population_std=std, physical_chains=len(target))


def _fit_pca(target, rank):
    center = target.mean(axis=0)
    _, singular, right = np.linalg.svd(target - center, full_matrices=False)
    if rank > len(right):
        raise ValueError("Insufficient TRAIN observations for the fixed PCA rank")
    basis = right[:rank].T
    return dict(center=center, basis=basis, singular_values=singular,
                rank=int(rank), training_observations=int(len(target)))


def _project_pca(model, target):
    return model["center"] + (target - model["center"]) @ model["basis"] @ model["basis"].T


def _fit_polynomial(g, target, degree):
    """Fit multi-output least squares; never alter degree/threshold on failure."""
    g = np.asarray(g, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64)
    center, scale = float(g.mean()), float(g.std(ddof=0))
    if not np.isfinite(scale) or scale <= 0:
        return dict(accepted=False, degree=degree, reason="zero TRAIN g standard deviation")
    design = np.vander((g - center) / scale, degree + 1, increasing=True)
    condition = float(np.linalg.cond(design))
    if not np.isfinite(condition) or condition > CONDITION_LIMIT:
        return dict(accepted=False, degree=degree, reason="design condition exceeds fixed 1e8 limit",
                    design_condition=condition if np.isfinite(condition) else None)
    coefficients, _, rank, singular = np.linalg.lstsq(design, target, rcond=None)
    if rank != degree + 1 or not np.isfinite(coefficients).all():
        return dict(accepted=False, degree=degree, reason="rank-deficient or nonfinite least squares",
                    design_condition=condition, design_rank=int(rank))
    return dict(accepted=True, degree=degree, g_center=center, g_std=scale,
                coefficients=coefficients, design_condition=condition, design_rank=int(rank),
                design_singular_values=singular, training_observations=len(g))


def _predict_polynomial(model, g):
    if model.get("accepted") is not True:
        raise ValueError("Cannot predict with a rejected polynomial fit")
    flat = np.asarray(g).reshape(-1)
    design = np.vander((flat - model["g_center"]) / model["g_std"],
                       model["degree"] + 1, increasing=True)
    prediction = design @ model["coefficients"]
    if not np.isfinite(prediction).all():
        raise ValueError("Polynomial prediction is nonfinite; no numerical repair is allowed")
    return prediction.reshape(np.asarray(g).shape[:-1] + (prediction.shape[-1],))


def _receipt(model):
    return {key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in model.items() if key not in ("coefficients", "basis", "center", "singular_values")}


def choose_polynomial_degree(train_g, train_target, train_cell_ids, check=lambda: None):
    """Grouped leave-one-TRAIN-condition-out selection, never inspect TEST."""
    ids = np.asarray(train_cell_ids)
    cells = np.unique(ids)
    if len(cells) < 2:
        raise ValueError("Polynomial selection requires at least two TRAIN conditions")
    candidates = []
    for degree in (1, 2, 3):
        folds, errors = [], []
        for cell in cells:
            check()
            hold = ids == cell
            fit = _fit_polynomial(train_g[~hold], train_target[~hold].reshape(-1, train_target.shape[-1]), degree)
            fold = dict(heldout_train_cell=cell.item() if hasattr(cell, "item") else cell,
                        **_receipt(fit))
            if fit["accepted"]:
                error = np.square(_predict_polynomial(fit, train_g[hold]) - train_target[hold])
                # Each held-out physical chain and each of four scales has equal weight.
                errors.append(error.reshape(len(error), -1).mean(axis=1))
                fold["validation_mse"] = float(error.mean())
            folds.append(fold)
        accepted = all(fold["accepted"] for fold in folds)
        candidates.append(dict(degree=degree, accepted=accepted, folds=folds,
            grouped_train_validation_mse=float(np.concatenate(errors).mean()) if accepted else None))
    valid = [row for row in candidates if row["accepted"]]
    degree = min(valid, key=lambda row: (row["grouped_train_validation_mse"], row["degree"]))["degree"] if valid else None
    return dict(selection="leave-one-whole-TRAIN-cell-out", candidates=candidates,
                selected_degree=degree, test_data_used=False, coupling_values_used=False,
                validation_weighting="equal physical chains, equal four scales, equal 105 features",
                condition_limit=CONDITION_LIMIT, adaptive_degree_expansion=False)


def analyze_arrays(arrays, *, check=lambda: None):
    """Numerical API receives no labels, only saved statistics, old g and IDs."""
    y = {split: np.asarray(arrays[f"{split}_feature_target"], dtype=np.float64)
         for split in ("train", "test")}
    g = {split: np.asarray(arrays[f"{split}_direct"], dtype=np.float64)
         for split in ("train", "test")}
    for split in y:
        if y[split].ndim != 3 or y[split].shape[1:] != (4, 105) or g[split].shape != (*y[split].shape[:2], 1):
            raise ValueError("Expected [physical chain, 4, 105] targets and [physical chain, 4, 1] old g")
        if not len(y[split]) or not np.isfinite(y[split]).all() or not np.isfinite(g[split]).all():
            raise ValueError("Decoder controls require nonempty finite public arrays")
    train_flat = y["train"].reshape(-1, 105)
    predictions, models, receipts = {}, {}, {}
    for rank in (1, 2):
        check()
        key = f"pooled_pca_rank{rank}"
        models[key] = _fit_pca(train_flat, rank)
        receipts[key] = dict(**_receipt(models[key]), fitted_on="TRAIN four scales, equal chains/scales",
                             target_projection_oracle=True, physical_dimension_inferred=False)
        predictions[key] = {split: _project_pca(models[key], value) for split, value in y.items()}
    models["per_scale_pca_rank1"] = [_fit_pca(y["train"][:, k], 1) for k in range(4)]
    receipts["per_scale_pca_rank1"] = dict(fitted_on="independent TRAIN statistics at each scale",
        target_projection_oracle=True, physical_dimension_inferred=False,
        scales=[_receipt(model) for model in models["per_scale_pca_rank1"]])
    predictions["per_scale_pca_rank1"] = {split: np.stack([
        _project_pca(models["per_scale_pca_rank1"][k], value[:, k]) for k in range(4)], axis=1)
        for split, value in y.items()}
    check()
    selection = choose_polynomial_degree(g["train"], y["train"], arrays["train_cell_ids"], check)
    for key, degree in (("fixed_g_affine", 1), ("fixed_g_polynomial", selection["selected_degree"])):
        if degree is None:
            receipts[key] = dict(accepted=False, reason="all preregistered TRAIN CV degrees rejected")
            continue
        model = _fit_polynomial(g["train"], train_flat, degree)
        models[key], receipts[key] = model, _receipt(model)
        if model["accepted"]:
            predictions[key] = {split: _predict_polynomial(model, value) for split, value in g.items()}
    predictions["existing_shared_affine"] = {split: np.asarray(arrays[f"{split}_feature_reconstruction"])
                                               for split in y}
    receipts["existing_shared_affine"] = dict(fitted_here=False, saved_Q_decoder=True)
    constant = y["train"].mean(axis=0)
    predictions["same_scale_train_constant"] = {split: np.broadcast_to(constant, value.shape).copy()
                                                 for split, value in y.items()}
    receipts["same_scale_train_constant"] = dict(fitted_on="TRAIN only, independently at each scale")
    rows = []
    for method, by_split in predictions.items():
        for split, prediction in by_split.items():
            for scale, size in enumerate(SIZES):
                rows.append(dict(method=method, split=split, scale=scale, lattice_size=size,
                                 **_metric(prediction[:, scale], y[split][:, scale], y["train"][:, scale])))
    serialized = {f"{method}_{split}_prediction": values for method, predictions_ in predictions.items()
                  for split, values in predictions_.items()}
    for key, model in models.items():
        if isinstance(model, list):
            for k, local in enumerate(model):
                serialized.update({f"{key}_k{k}_{name}": local[name] for name in ("center", "basis", "singular_values")})
        elif "basis" in model:
            serialized.update({f"{key}_{name}": model[name] for name in ("center", "basis", "singular_values")})
        elif model.get("accepted"):
            serialized[f"{key}_coefficients"] = model["coefficients"]
    return dict(method=METHOD, rows=rows, fits=receipts, polynomial_selection=selection,
                fit_weighting="equal physical chains and equal four scales; no resampling or label regression",
                normalization="sqrt(mean(featurewise TRAIN variance ddof=0)) separately at each scale",
                critical_r=None, thermal_nu=None, physical_dimension_inferred=False), serialized


def plot_errors(result, output, *, font=None):
    plt = _plotting(font)
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.4), constrained_layout=True)
    for row_index, split in enumerate(("train", "test")):
        for col, (metric, title) in enumerate((("absolute_rmse", "绝对 RMSE"), ("fixed_train_nrmse", "固定训练标准差 NRMSE"))):
            ax = axes[row_index, col]
            for method, label in LABELS.items():
                rows = sorted((row for row in result["rows"] if row["method"] == method and row["split"] == split), key=lambda row: row["scale"])
                if rows:
                    ax.plot([row["scale"] for row in rows], [row[metric] for row in rows], marker="o", label=label)
            ax.set(title=("训练条件" if split == "train" else "留出条件") + "：" + title,
                   xlabel="尺度 k 与格点边长", ylabel=title)
            ax.set_xticks(range(4), [f"{k}（L={size}）" for k, size in enumerate(SIZES)])
            ax.set_yscale("symlog", linthresh=1e-5)
            ax.grid(alpha=.22)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("固定公共统计的解码归因对照：PCA 是目标投影几何下界，不是新 Q 或物理维度认证")
    fig.savefig(Path(output) / "decoder_errors.png", dpi=170)
    plt.close(fig)


@contextmanager
def _single_thread_blas():
    """Limit BLAS and verify it, including old threadpoolctl/ILP64 on macOS."""
    try:
        if threadpool_limits is None:
            raise ImportError("Optional threadpoolctl is unavailable")
        limit = threadpool_limits(limits=1)
        pools = threadpool_info()
    except (ImportError, AttributeError):
        # threadpoolctl 2.2 mistakes NumPy's ILP64 library for an LP64 build.
        # Use only installed NumPy's known bundled OpenBLAS, never system paths.
        candidates = list((Path(np.__file__).parent / ".dylibs").glob("*openblas*"))
        candidates += list((Path(np.__file__).parent.parent / "numpy.libs").glob("*openblas*"))
        if not candidates:
            raise RuntimeError("Cannot verify single-thread NumPy BLAS with this runtime")
        restored, pools = [], []
        try:
            for path in candidates:
                library = ctypes.CDLL(str(path))
                access = next(((getattr(library, prefix + "get_num_threads" + suffix),
                                getattr(library, prefix + "set_num_threads" + suffix))
                               for prefix in ("openblas_", "scipy_openblas_", "scipy_openblas64_")
                               for suffix in ("64_", "_64_", "_", "")
                               if hasattr(library, prefix + "get_num_threads" + suffix)
                               and hasattr(library, prefix + "set_num_threads" + suffix)), None)
                if access is None:
                    raise RuntimeError("Bundled OpenBLAS has no supported thread-count controls")
                get_threads, set_threads = access
                get_threads.restype, set_threads.argtypes = ctypes.c_int, [ctypes.c_int]
                old = get_threads()
                restored.append((set_threads, old))
                set_threads(1)
                if get_threads() != 1:
                    raise RuntimeError("Bundled OpenBLAS single-thread verification failed")
                pools.append(dict(filepath=str(path), num_threads=get_threads(),
                                  internal_api="openblas", control="verified bundled ctypes fallback"))
            yield pools
        finally:
            for setter, old in restored:
                setter(old)
    else:
        try:
            if any(pool.get("num_threads") != 1 for pool in pools):
                raise RuntimeError("BLAS/OpenMP single-thread verification failed")
            yield pools
        finally:
            limit.restore_original_limits()


@contextmanager
def _single_thread_budget(seconds):
    if not 0 < seconds <= 120:
        raise ValueError("CPU budget must be in (0, 120] seconds")
    started = time.process_time()
    def check():
        if time.process_time() - started > seconds:
            raise RuntimeError("Decoder diagnostic CPU budget exceeded; no automatic retry or adjustment")
    with _single_thread_blas() as pools:
        yield check, started, pools
        check()


def run_decoder_controls(training, output, *, cpu_budget_seconds=120, font=None):
    output, training = Path(output).resolve(), Path(training).resolve()
    if output == training or output.is_relative_to(training) or training.is_relative_to(output):
        raise ValueError("Diagnostic output must be disjoint from the immutable training root")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic output: {output}")
    with _single_thread_budget(cpu_budget_seconds) as (check, started, pools):
        public = load_public_comparison(training)
        check()
        # Do not pass labels into fitting/selection, even though the public loader
        # independently validates that the original TRAIN/TEST split is disjoint.
        fit_arrays = {key: public.endpoints["legacy"][key] for key in (
            "train_feature_target", "test_feature_target", "train_direct", "test_direct",
            "train_feature_reconstruction", "test_feature_reconstruction", "train_cell_ids")}
        result, arrays = analyze_arrays(fit_arrays, check=check)
        check()
        if public.fingerprints != {name: sha256(training / name) for name in INPUT_FILES}:
            raise ValueError("Public sources changed during decoder diagnostic")
        output.mkdir(parents=True, exist_ok=False)
        _json(output / "comparison.json", result)
        np.savez_compressed(output / "decoder_arrays.npz", **arrays)
        plot_errors(result, output, font=font)
        check()
        if public.fingerprints != {name: sha256(training / name) for name in INPUT_FILES}:
            raise ValueError("Public sources changed during diagnostic report generation")
        outputs = {name: sha256(output / name) for name in ("comparison.json", "decoder_arrays.npz", "decoder_errors.png")}
        receipt = dict(method=METHOD, source_recipe=RECIPE, source_training=str(training),
            source_public_sha256=public.fingerprints, source_public_sha256_unchanged=True,
            source_analysis_arm="legacy; C/Q public arrays are exactly paired with all_start",
            source_code_sha256={str(Path(__file__).resolve()): sha256(__file__),
                str(Path(__file__).with_name("report_phi4_b2_closure_comparison.py").resolve()):
                    sha256(Path(__file__).with_name("report_phi4_b2_closure_comparison.py"))},
            outputs_sha256=outputs, cpu_seconds=time.process_time() - started,
            cpu_budget_seconds=cpu_budget_seconds, threadpools=pools, numpy_version=np.__version__,
            labels_used_for_fit=False, test_data_used_for_fit_or_selection=False,
            checkpoints_loaded=False, raw_physical_data_loaded=False, model_loaded=False,
            QCR_training_performed=False, inference_performed=False, sampling_performed=False,
            GPU_used=False, root_search_performed=False, MI_reestimated=False,
            numerical_fit="TRAIN-only NumPy least squares and SVD; fixed degrees 1/2/3; condition cap 1e8",
            limitations=["仅比较保存的局域统计重构；不是闭环训练，也不证明统计量在尺度变换下闭合。",
                "PCA 控制使用待重构目标自身投影，是几何下界而非新 Q、可部署解码器或物理维度发现。",
                "固定旧 g 的仿射与多项式拟合只用训练条件；仅用整条件交叉验证选择 1、2、3 阶。",
                "重构改善不证明恢复相变、不动点、临界面或普适指数；原采样警告与单种子局限保留。"],
            data_warnings=public.summary.get("data_warnings", []))
        _json(output / "receipt.json", receipt)
        _json(output / "status.json", dict(method=METHOD, status="complete", workflow_complete=True,
            returncode=0, error=None, requires_user_action=False,
            receipt_sha256=sha256(output / "receipt.json"), outputs_sha256=outputs))
        return receipt, result


def main(argv=None):
    parser = argparse.ArgumentParser(description="连续场二倍粗粒化公共统计解码归因对照")
    parser.add_argument("--training", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cpu-budget-seconds", type=float, default=120)
    parser.add_argument("--font")
    args = parser.parse_args(argv)
    receipt, _ = run_decoder_controls(args.training, args.output,
        cpu_budget_seconds=args.cpu_budget_seconds, font=args.font)
    print(json.dumps(dict(status="complete", output=str(args.output.resolve()),
                          cpu_seconds=receipt["cpu_seconds"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
