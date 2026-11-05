"""冻结公开统计的未来物理预测诊断；不导入模型或读取原始构型。"""
from __future__ import annotations

import os
for _thread_name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_name] = "1"

import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

PROTOCOL_PATH = Path(__file__).with_name("protocol.json")
METHODS = ("linear", "rbf", "rank_rbf", "rank_rbf_extra")
TRANSITIONS = ((0, 1), (1, 2), (0, 2))
RIDGES = (1e-6, .001, .1, 1., 10.)
BANDWIDTHS = (.25, .5, 1., 2.)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_json(path, value):
    Path(path).write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def midpoint_cdf(reference, values):
    """仅拟合折的经验CDF；相等值使用中点，不插值坐标间距。"""
    reference = np.asarray(reference)
    values = np.asarray(values)
    return (np.searchsorted(reference, values, side="left") +
            np.searchsorted(reference, values, side="right")) / (2. * len(reference))


def make_basis(fit, x, extra=None):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if fit["rank_reference"] is not None:
        x = midpoint_cdf(fit["rank_reference"], x)
    z = (x - fit["x_mean"]) / fit["x_std"]
    if fit["kind"] == "linear":
        basis = np.column_stack((np.ones(len(z)), z))
    else:
        basis = np.column_stack((np.ones(len(z)), np.exp(-.5 *
            ((z[:, None] - fit["centers"][None]) / fit["bandwidth"]) ** 2)))
    if fit["extra_ridge"] is not None:
        if extra is None:
            raise ValueError("该读出需要当前尺度额外物理特征")
        standardized = (np.asarray(extra) - fit["extra_mean"]) / fit["extra_std"]
        basis = np.column_stack((basis, standardized))
    return basis


def fit_probe(x, y, kind, ridge, bandwidth=None, extra=None, extra_ridge=None):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64)
    rank_reference = np.sort(x) if kind.startswith("rank_") else None
    transformed = midpoint_cdf(rank_reference, x) if rank_reference is not None else x
    x_mean = float(transformed.mean())
    x_std = max(float(transformed.std(ddof=0)), 1e-12)
    z = (transformed - x_mean) / x_std
    if kind not in METHODS or (extra_ridge is not None and kind != "rank_rbf_extra"):
        raise ValueError("未知读出类型或不支持的额外特征")
    fit = dict(kind=kind, ridge=float(ridge), bandwidth=bandwidth,
        rank_reference=rank_reference, x_mean=x_mean, x_std=x_std,
        centers=None if kind == "linear" else np.quantile(z, np.linspace(0, 1, 16)),
        target_mean=y.mean(axis=0), target_std=np.maximum(y.std(axis=0, ddof=0), 1e-12),
        extra_ridge=extra_ridge, extra_mean=None, extra_std=None)
    if extra_ridge is not None:
        if extra is None or len(extra) != len(x):
            raise ValueError("额外特征形状不符")
        fit.update(extra_mean=np.asarray(extra).mean(axis=0),
                   extra_std=np.maximum(np.asarray(extra).std(axis=0, ddof=0), 1e-12))
    basis = make_basis(fit, x, extra)
    penalty = np.full(basis.shape[1], ridge, dtype=np.float64)
    penalty[0] = 0.
    if extra_ridge is not None:
        penalty[-np.asarray(extra).shape[1]:] = extra_ridge
    gram = np.einsum("ni,nj->ij", basis, basis, optimize=False) / len(x)
    rhs = np.einsum("ni,nj->ij", basis, (y - fit["target_mean"]) / fit["target_std"], optimize=False) / len(x)
    fit["coefficient"] = np.linalg.solve(gram + np.diag(penalty), rhs)
    return fit


def predict_probe(fit, x, extra=None):
    basis = make_basis(fit, x, extra)
    return np.einsum("ni,ij->nj", basis, fit["coefficient"], optimize=False) * fit["target_std"] + fit["target_mean"]


def choose_probe(x, y, groups, kind, extra=None, *, ridges=RIDGES, bandwidths=BANDWIDTHS):
    """调用者只传TRAIN；同物理条件和链对永不跨折。"""
    x, y, groups = np.asarray(x), np.asarray(y), np.asarray(groups)
    extras = (None, *ridges) if kind == "rank_rbf_extra" else (None,)
    widths = (None,) if kind == "linear" else bandwidths
    candidates = [(ridge, width, er) for er in extras for width in widths for ridge in ridges]
    records = []
    for ridge, width, er in candidates:
        total, count, folds = 0., 0, []
        for fold in range(5):
            validation = groups % 5 == fold
            fitting = ~validation
            if not validation.any() or not fitting.any():
                raise ValueError("TRAIN条件不足以形成固定整组五折")
            fit = fit_probe(x[fitting], y[fitting], kind, ridge, width,
                None if extra is None else extra[fitting], er)
            prediction = predict_probe(fit, x[validation], None if extra is None else extra[validation])
            error = ((prediction - y[validation]) / fit["target_std"]) ** 2
            total += float(error.sum()); count += error.size
            folds.append(float(error.mean()))
        records.append(dict(ridge=ridge, bandwidth=width, extra_ridge=er,
                            validation_nmse=total / count, fold_nmse=folds))
    chosen_index = min(range(len(records)), key=lambda i: records[i]["validation_nmse"])
    chosen = records[chosen_index]
    return fit_probe(x, y, kind, chosen["ridge"], chosen["bandwidth"], extra, chosen["extra_ridge"]), dict(
        selected_index=chosen_index, selected=chosen, candidates=records)


def build_pairs(arrays):
    """返回链对与双向输入/目标；仅用公开标识，不使用耦合值。"""
    split, group, replica = (np.asarray(arrays[k]) for k in ("split", "group_index", "replica_index"))
    n = len(split)
    expected = {"g_full": (2, n, 5, 3), "observables": (n, 5, 3, 2, 5),
                "motif_means": (n, 5, 3, 2, 16)}
    for key, shape in expected.items():
        if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise ValueError(f"{key}形状或有限性不符")
    if set(np.unique(split)) != {0, 1}:
        raise ValueError("必须包含既有TRAIN和TEST")
    pairs, pair_split, pair_group = [], [], []
    for part in (0, 1):
        for gid in np.unique(group[split == part]):
            ids = np.flatnonzero((split == part) & (group == gid))
            if len(ids) != 12 or sorted(replica[ids]) != list(range(12)):
                raise ValueError("每条件必须恰有rep0..11的12条物理链")
            ordered = ids[np.argsort(replica[ids])]
            for offset in range(0, 12, 2):
                pairs.append(ordered[offset:offset + 2]); pair_split.append(part); pair_group.append(gid)
    source = np.asarray(pairs, dtype=np.int64)
    target = source[:, ::-1].copy()
    g = arrays["g_full"].mean(axis=2)
    obs = arrays["observables"].mean(axis=(1, 3))
    motif = arrays["motif_means"].mean(axis=(1, 3))
    current = np.concatenate((motif, obs), axis=-1)
    # axes: model,pair,direction,path / pair,direction,path,observable.
    x = np.stack([g[:, source, s] for s, _ in TRANSITIONS], axis=-1)
    y = np.stack([obs[target, t] for _, t in TRANSITIONS], axis=2)
    extra = np.stack([current[source, s] for s, _ in TRANSITIONS], axis=2)
    train = split == 0
    target_mean = np.stack([obs[train, t].mean(axis=0) for _, t in TRANSITIONS])
    target_std = np.stack([np.maximum(obs[train, t].std(axis=0, ddof=0), 1e-12) for _, t in TRANSITIONS])
    return dict(source_indices=source, target_indices=target, split=np.asarray(pair_split),
        group=np.asarray(pair_group), x=x, truth=y, extra=extra,
        target_mean=target_mean, target_std=target_std)


def bootstrap_weights(groups, repeats, rng):
    groups = np.asarray(groups)
    result = np.zeros((repeats, len(groups)), dtype=np.float64)
    for group in np.unique(groups):
        members = np.flatnonzero(groups == group)
        if len(members) != 6:
            raise ValueError("bootstrap每条件必须6个不相交链对")
        picks = rng.integers(0, len(members), size=(repeats, len(members)))
        counts = np.zeros((repeats, len(members)), dtype=np.float64)
        np.add.at(counts, (np.arange(repeats)[:, None], picks), 1.)
        result[:, members] = counts / len(groups)
    return result


def interval(values):
    return np.quantile(values, (.025, .975), axis=0)


def weighted_means(values, weights):
    shape = values.shape[1:]
    result = np.einsum("bn,nf->bf", weights, values.reshape(len(values), -1), optimize=False)
    return result.reshape((len(weights),) + shape)


def summaries(error_sq, target_std, weights):
    """输入model,pair,path,target，bootstrap先聚合MSE再开方。"""
    normalized = error_sq / target_std[None, None] ** 2
    boot = np.stack([weighted_means(row, weights) for row in normalized])
    raw_boot = np.stack([weighted_means(row, weights) for row in error_sq])
    point = normalized.mean(axis=1)
    raw_point = error_sq.mean(axis=1)
    models = []
    for m in range(len(error_sq)):
        models.append(dict(
            overall=dict(NMSE=float(point[m].mean()), NRMSE=float(np.sqrt(point[m].mean())),
                raw_RMSE=float(np.sqrt(raw_point[m].mean())),
                NMSE_ci95=interval(boot[m].mean(axis=(1, 2))),
                NRMSE_ci95=interval(np.sqrt(boot[m].mean(axis=(1, 2)))),
                raw_RMSE_ci95=interval(np.sqrt(raw_boot[m].mean(axis=(1, 2))))),
            by_path=dict(NMSE=point[m].mean(axis=-1), NRMSE=np.sqrt(point[m].mean(axis=-1)),
                raw_RMSE=np.sqrt(raw_point[m].mean(axis=-1)),
                NMSE_ci95=interval(boot[m].mean(axis=-1)),
                NRMSE_ci95=interval(np.sqrt(boot[m].mean(axis=-1))),
                raw_RMSE_ci95=interval(np.sqrt(raw_boot[m].mean(axis=-1)))),
            by_path_and_target=dict(NMSE=point[m], NRMSE=np.sqrt(point[m]), raw_RMSE=np.sqrt(raw_point[m]),
                NMSE_ci95=interval(boot[m]), NRMSE_ci95=interval(np.sqrt(boot[m])), raw_RMSE_ci95=interval(np.sqrt(raw_boot[m])))))
    result = dict(models=models)
    if len(error_sq) == 2:
        result["supervised_minus_unsupervised"] = dict(
            overall=dict(NMSE=float((point[1] - point[0]).mean()),
                NMSE_ci95=interval((boot[1] - boot[0]).mean(axis=(1, 2))),
                NRMSE=float(np.sqrt(point[1].mean()) - np.sqrt(point[0].mean())),
                NRMSE_ci95=interval(np.sqrt(boot[1].mean(axis=(1, 2))) - np.sqrt(boot[0].mean(axis=(1, 2)))),
                raw_RMSE=float(np.sqrt(raw_point[1].mean()) - np.sqrt(raw_point[0].mean())),
                raw_RMSE_ci95=interval(np.sqrt(raw_boot[1].mean(axis=(1, 2))) - np.sqrt(raw_boot[0].mean(axis=(1, 2))))),
            by_path=dict(NMSE=(point[1] - point[0]).mean(axis=-1),
                NMSE_ci95=interval((boot[1] - boot[0]).mean(axis=-1)),
                NRMSE=np.sqrt(point[1].mean(axis=-1)) - np.sqrt(point[0].mean(axis=-1)),
                NRMSE_ci95=interval(np.sqrt(boot[1].mean(axis=-1)) - np.sqrt(boot[0].mean(axis=-1))),
                raw_RMSE=np.sqrt(raw_point[1].mean(axis=-1)) - np.sqrt(raw_point[0].mean(axis=-1)),
                raw_RMSE_ci95=interval(np.sqrt(raw_boot[1].mean(axis=-1)) - np.sqrt(raw_boot[0].mean(axis=-1)))),
            by_path_and_target=dict(NMSE=point[1] - point[0], NMSE_ci95=interval(boot[1] - boot[0]),
                NRMSE=np.sqrt(point[1]) - np.sqrt(point[0]), NRMSE_ci95=interval(np.sqrt(boot[1]) - np.sqrt(boot[0])),
                raw_RMSE=np.sqrt(raw_point[1]) - np.sqrt(raw_point[0]),
                raw_RMSE_ci95=interval(np.sqrt(raw_boot[1]) - np.sqrt(raw_boot[0]))))
    return result


def scalar_difference_summary(values, weights):
    return dict(mean=float(values.mean()), ci95=interval(weighted_means(values, weights)))


def analyze(arrays, protocol, *, ridges=RIDGES, bandwidths=BANDWIDTHS):
    paired = build_pairs(arrays)
    train = paired["split"] == 0
    y, extra, x = paired["truth"], paired["extra"], paired["x"]
    rows_train = np.repeat(train, 2)
    groups_train = np.repeat(paired["group"][train], 2)
    predictions, errors, records = {}, {}, {}
    for method in METHODS:
        prediction = np.empty((2,) + y.shape)
        fits = []
        for model in range(2):
            for path in range(len(TRANSITIONS)):
                xx = x[model, :, :, path].reshape(-1)
                yy = y[:, :, path].reshape(-1, 5)
                ee = extra[:, :, path].reshape(-1, 21)
                fit, cv = choose_probe(xx[rows_train], yy[rows_train], groups_train, method,
                    ee[rows_train] if method == "rank_rbf_extra" else None, ridges=ridges, bandwidths=bandwidths)
                prediction[model, :, :, path] = predict_probe(fit, xx, ee if method == "rank_rbf_extra" else None).reshape(len(y), 2, 5)
                fits.append(dict(model=model, path=path, fit=fit, cross_validation=cv))
        predictions[method] = prediction
        errors[method] = ((prediction - y[None]) ** 2).mean(axis=2)
        records[method] = dict(fitted_models=fits, splits={})
    baseline_error = ((paired["target_mean"][None, None] - y) ** 2).mean(axis=1)
    rng = np.random.default_rng(protocol["bootstrap"]["seed"])
    baseline, gains, comparisons, saved_weights = {}, {}, {}, {}
    for split_name, mask in (("TRAIN", train), ("TEST", ~train)):
        weights = bootstrap_weights(paired["group"][mask], protocol["bootstrap"]["repeats"], rng)
        saved_weights[split_name] = weights
        for method in METHODS:
            records[method]["splits"][split_name] = summaries(errors[method][:, mask], paired["target_std"], weights)
        baseline[split_name] = summaries(baseline_error[None, mask], paired["target_std"], weights)
        gains_per_pair = ((errors["rank_rbf"][:, mask] - errors["rank_rbf_extra"][:, mask]) /
                          paired["target_std"][None, None] ** 2).mean(axis=(2, 3))
        gains[split_name] = dict(unsupervised=scalar_difference_summary(gains_per_pair[0], weights),
            supervised=scalar_difference_summary(gains_per_pair[1], weights),
            unsupervised_gain_minus_supervised_gain=scalar_difference_summary(gains_per_pair[0] - gains_per_pair[1], weights))
        comparisons[split_name] = {}
        for method in METHODS:
            diff = ((errors[method][1, mask] - baseline_error[mask]) / paired["target_std"][None] ** 2).mean(axis=(1, 2))
            comparisons[split_name][method] = scalar_difference_summary(diff, weights)
    relative = all(records[m]["splits"]["TEST"]["supervised_minus_unsupervised"]["overall"]["NMSE_ci95"][1] < 0
                   for m in ("rank_rbf", "rbf")) and comparisons["TEST"]["rank_rbf"]["ci95"][1] < 0
    mechanism = relative and gains["TEST"]["unsupervised"]["ci95"][0] > 0 and gains["TEST"]["unsupervised_gain_minus_supervised_gain"]["ci95"][0] > 0
    report = dict(schema_version=1, version=protocol["version"], protocol=protocol,
        source_actual_steps=[protocol["endpoint"], protocol["endpoint"]],
        model_order=["unsupervised", "supervised"], methods=records,
        constant_baseline=baseline, supervised_minus_constant_baseline=comparisons,
        conditional_extra_feature_NMSE_gains=gains,
        evidence=dict(relative_supervised_better=bool(relative), extra_feature_mechanism_supported=bool(mechanism),
            causal_claim_allowed=False, exponent_certified=False),
        target_mean_TRAIN=paired["target_mean"], target_std_TRAIN=paired["target_std"],
        counts=dict(train_pairs=int(train.sum()), test_pairs=int((~train).sum()), directions_per_pair=2,
                    C_draws=5, half_count=2),
        limitations=protocol["limitations"], QCR_updates=0, new_model_inference=0, remote_access=False,
        raw_RMSE_caution="不同观测量幅度不同，整体rawRMSE只作附录；主要比较共同TRAIN标准化NMSE。")
    output = dict(source_indices=paired["source_indices"], target_indices=paired["target_indices"],
        pair_split=paired["split"], pair_group=paired["group"], inputs_g_full_mean_C=x,
        current_extra_features=extra, future_truth=y, target_std_TRAIN=paired["target_std"],
        constant_baseline_error_sq=baseline_error, **{f"{k}_prediction": v for k, v in predictions.items()},
        **{f"{k}_pair_path_target_error_sq": v for k, v in errors.items()},
        **{f"bootstrap_{k}_weights": v for k, v in saved_weights.items()})
    return report, output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL_PATH.read_text())
    if protocol["methods"] != list(METHODS) or protocol["transitions"] != [list(t) for t in TRANSITIONS] or protocol["ridge_grid"] != list(RIDGES) or protocol["bandwidth_grid"] != list(BANDWIDTHS):
        raise ValueError("冻结协议和实现不一致")
    source = args.source.resolve()
    path = source / "collected.npz"
    actual_hash = sha256(path)
    if actual_hash != protocol["source_collected_sha256"]:
        raise ValueError("公开输入SHA不符")
    receipt = json.loads((source / "receipt.json").read_text())
    if receipt.get("collected_sha256") != actual_hash or receipt.get("labels_accessed") is not False:
        raise ValueError("公开采集回执不符")
    if [m["actual_epoch"] for m in receipt["models"]] != [protocol["endpoint"]] * 2:
        raise ValueError("历史模型实际训练步数不符")
    output = args.output.resolve()
    if output == source or source in output.parents:
        raise ValueError("独立诊断输出不能写入冻结源目录")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        save_json(output / "status.json", dict(status="running", QCR_updates=0))
        with np.load(path, allow_pickle=False) as stored:
            arrays = {k: stored[k] for k in ("g_full", "observables", "motif_means", "split", "group_index", "replica_index", "model_names")}
        if list(arrays["model_names"]) != protocol["models"]:
            raise ValueError("模型顺序不符")
        report, saved = analyze(arrays, protocol)
        if report["counts"]["train_pairs"] != 258 or report["counts"]["test_pairs"] != 126:
            raise ValueError("冻结TRAIN/TEST链对数量不符")
        report["fingerprints"] = dict(input_sha256=actual_hash, protocol_sha256=sha256(PROTOCOL_PATH),
            code_sha256=sha256(__file__), receipt_sha256=sha256(source / "receipt.json"))
        report["wall_seconds"] = time.monotonic() - started
        np.savez_compressed(output / "analysis_arrays.npz", **saved)
        save_json(output / "analysis.json", report)
        save_json(output / "status.json", dict(status="complete", workflow_complete=True, QCR_updates=0,
            new_model_inference=0, evidence=report["evidence"], wall_seconds=report["wall_seconds"]))
        print(json.dumps(jsonable(dict(status="complete", output=str(output), evidence=report["evidence"])), ensure_ascii=False))
    except BaseException as error:
        save_json(output / "status.json", dict(status="failed", error=f"{type(error).__name__}: {error}", QCR_updates=0))
        raise


if __name__ == "__main__":
    main()
