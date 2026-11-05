"""Prospective evaluation of fixed R endpoints on new phi4 chains and sizes.

The trained source Q/C, its amplitude scale and decoder stay frozen. New
physical parameters identify evaluation groups only. Four stochastic encodings
are measurements of each chain; uncertainty resamples physical chains.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from experiments import phi4_qcr as source
from experiments.phi4_ru_r_cache import _source_contract, _verify_unchanged, encode_chain
from experiments.phi4_ru_r_calibration import predict
from recurrent_rg.phi4_qcr import Phi4QCR
from recurrent_rg.phi4_r_calibration import FrozenResidualMap


SEEDS = (91, 92, 93)
MOMENTS = ("magnetization", "absolute_magnetization", "magnetization_squared", "magnetization_fourth")
PATHS = ((0, 1, "one_step", 0), (1, 2, "one_step", 1), (0, 2, "rollout", 2))


class BudgetHold(RuntimeError):
    pass


def evaluation_seed(identity, repeat, stream):
    if stream not in ("subset", "C") or repeat not in range(4):
        raise ValueError("Invalid evaluation stream")
    token = json.dumps(["phi4_ru_prospective_20261030", identity, repeat, stream], sort_keys=True)
    return int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big") & ((1 << 63) - 1)


def physical_metrics(moments, length):
    m, ma, m2, m4 = np.moveaxis(np.asarray(moments, dtype=float), -1, 0)
    if np.any(m2 <= 0):
        raise ValueError("Positive second moment required")
    return np.stack((ma, 1 - m4 / (3 * m2**2), length**2 * (m2 - m**2),
                     length**2 * (m2 - ma**2)), axis=-1)


def verify_inventory(root, hashes):
    root = Path(root).resolve()
    status = json.loads((root / "status.json").read_text())
    if not (status.get("status") == "complete" and status.get("workflow_complete")
            and status.get("returncode") == 0 and status.get("error") is None):
        raise ValueError("R validation must finish before prospective data evaluation")
    inventory = json.loads((root / "public_files.json").read_text())
    for name, expected in inventory.items():
        path = (root / name).resolve()
        if path.parent != root or path.suffix not in (".json", ".png", ".npz"):
            raise ValueError("Unexpected validation inventory member")
        if source.sha256_file(path) != expected:
            raise RuntimeError(f"Validation fingerprint mismatch: {path}")
        hashes[str(path)] = expected
    for path in (root / "status.json", root / "public_files.json"):
        hashes[str(path)] = source.sha256_file(path)


def restore_maps(validation):
    models = {}
    expected = json.loads((Path(validation) / "summary.json").read_text())["final_R_sha256"]
    with np.load(Path(validation) / "R_parameters.npz", allow_pickle=False) as archive:
        for arm in ("adam6000", "selected"):
            for seed in SEEDS:
                key = f"{arm}_seed{seed}"
                state = {name[len(key) + 1:]: torch.tensor(archive[name])
                         for name in archive.files if name.startswith(key + ".")}
                if set(state) != {"center", "scale", "net.0.weight", "net.0.bias", "net.2.weight",
                                  "net.2.bias", "net.4.weight", "net.4.bias"}:
                    raise ValueError(f"Missing fixed R state: {key}")
                with torch.random.fork_rng(devices=[]):
                    model = FrozenResidualMap(2, 96, center=state["center"], scale=state["scale"])
                model.load_state_dict(state, strict=True)
                model.eval().requires_grad_(False)
                if source.state_fingerprint(model) != expected[key]:
                    raise RuntimeError(f"Fixed R fingerprint mismatch: {key}")
                models[key] = model
    return models


def load_new_encodings(model, data_root, guard, hashes):
    manifest_path = Path(data_root) / "data_manifest.json"
    datasets = json.loads(manifest_path.read_text())["datasets"]
    hashes[str(manifest_path.resolve())] = source.sha256_file(manifest_path)
    direct, features, moments, labels, identities, warnings = [], [], [], [], [], []
    for dataset in datasets:
        guard()
        root = Path(dataset["root"]).resolve()
        if not root.is_relative_to(Path(data_root).resolve()):
            raise ValueError("New data shard must be within the follow-up output")
        manifest = root / "manifest.json"
        if source.sha256_file(manifest) != dataset["manifest_sha256"]:
            raise RuntimeError("New data manifest fingerprint mismatch")
        hashes[str(manifest)] = dataset["manifest_sha256"]
        meta = json.loads(manifest.read_text())
        for shard in meta["shards"]:
            guard()
            side_path, array_path = root / shard["sidecar"], root / shard["path"]
            for path, expected in ((side_path, shard["sidecar_sha256"]), (array_path, shard["sha256"])):
                if source.sha256_file(path) != expected:
                    raise RuntimeError(f"New data fingerprint mismatch: {path}")
                hashes[str(path)] = expected
            side = json.loads(side_path.read_text())
            if side["size"] != dataset["size"] or side["u"] != dataset["u"]:
                raise ValueError("Evaluation metadata disagrees with generated condition")
            with np.load(array_path, allow_pickle=False) as archive:
                fields = archive["configurations"].copy()
            if fields.shape != (4, 128, side["size"], side["size"], 1):
                raise ValueError("Expected four new 128-field chains per subdataset condition")
            for chain, row in enumerate(side["chains"]):
                if row["chain_index"] != chain:
                    raise ValueError("Physical chain order mismatch")
                identity = [dataset["name"], side["cell_index"], chain]
                gs, fs = [], []
                for repeat in range(4):
                    guard()
                    generator = torch.Generator().manual_seed(evaluation_seed(identity, repeat, "subset"))
                    indices = torch.randperm(128, generator=generator)[:64].numpy()
                    encoded = encode_chain(model, torch.from_numpy(fields[chain]), indices,
                                           c_seed=evaluation_seed(identity, repeat, "C"))
                    gs.append(encoded["direct"])
                    fs.append(encoded["feature_target"])
                direct.append(gs)
                features.append(fs)
                labels.append([side["size"], side["r"], side["u"]])
                identities.append(identity)
                moments.append([row["diagnostics"][name]["mean"] for name in MOMENTS])
                warnings.append(dict(identity=identity, warnings=row["warnings"],
                                     production_acceptance=row["production_acceptance"]))
            print(json.dumps(dict(event="frozen_collection_condition_complete", dataset=dataset["name"],
                                  cell=side["cell_index"], chains=len(direct))), flush=True)
    labels = np.asarray(labels)
    counts = {int(size): int(np.sum(labels[:, 0] == size)) for size in np.unique(labels[:, 0])}
    if counts != {36: 120, 108: 36} or len({tuple(x) for x in identities}) != 156:
        raise ValueError(f"Prospective physical-chain counts disagree with fixed design: {counts}")
    return dict(direct=np.moveaxis(np.asarray(direct), 1, 0),
                feature_target=np.moveaxis(np.asarray(features), 1, 0),
                moments=np.asarray(moments), labels=labels,
                identities=np.asarray([json.dumps(x) for x in identities]),
                warning_rows=warnings)


def path_metrics(direct, predictions, labels, target_stds, decoder, bias, features):
    result = []
    for length in sorted(set(labels[:, 0].astype(int))):
        mask = labels[:, 0] == length
        for start, end, kind, index in PATHS:
            g, target = direct[:, mask, start], direct[:, mask, end]
            prediction = predictions[kind][:, mask, index]
            residual = prediction - target
            rmse = float(np.sqrt(np.mean(residual**2)))
            chain_rmse = float(np.sqrt(np.mean(residual.mean(0)**2)))
            identity = float(np.sqrt(np.mean((g - target)**2)))
            chain_identity = float(np.sqrt(np.mean((g - target).mean(0)**2)))
            # NumPy size labels otherwise leave an np.int64 in the JSON summary.
            target_length = int(length // 3**end)
            std = target_stds[target_length]
            decoded = prediction @ decoder.T + bias
            feature_rmse = float(np.sqrt(np.mean((decoded - features[:, mask, end])**2)))
            truth_drift = (target - g).mean(0)
            predicted_drift = (prediction - g).mean(0)
            uvalues = np.unique(labels[mask, 1:3], axis=0)
            angles, opposite = [], 0
            for pair in uvalues:
                group = np.all(labels[mask, 1:3] == pair, axis=1)
                a, b = truth_drift[group].mean(0), predicted_drift[group].mean(0)
                norm = np.linalg.norm(a) * np.linalg.norm(b)
                angle = float(np.degrees(np.arccos(np.clip(a @ b / norm, -1, 1)))) if norm > 0 else None
                if angle is not None:
                    angles.append(angle)
                    opposite += int(angle > 90)
            result.append(dict(source_size=int(length), path=f"{length // 3**start}_to_{target_length}",
                start_scale=start, target_scale=end, rmse=rmse, chain_mean_rmse=chain_rmse,
                source_TRAIN_target_std=std, nrmse=rmse / std,
                target_reference_lattice=target_length,
                relative_displacement_error=rmse / identity if identity > 0 else None,
                raw_draw_relative_displacement_error=rmse / identity if identity > 0 else None,
                chain_mean_relative_displacement_error=chain_rmse / chain_identity if chain_identity > 0 else None,
                native_feature_rmse=feature_rmse, independent_physical_chains=int(mask.sum()),
                condition_opposite_count=opposite, conditions=len(uvalues),
                median_condition_angle_degrees=float(np.median(angles)) if angles else None))
    return result


def grouped_summary(arrays, predictions, *, bootstrap=1000):
    """Resample chain IDs, shared by all models and scales within each condition."""
    labels, direct, moments = arrays["labels"], arrays["direct"].mean(0), arrays["moments"]
    rng = np.random.default_rng(2026103004)
    result = []
    for condition in np.unique(labels, axis=0):
        indices = np.flatnonzero(np.all(labels == condition, axis=1))
        n, length = len(indices), int(condition[0])
        picks = rng.integers(n, size=(bootstrap, n))
        sampled_moments = moments[indices][picks].mean(1)
        physic = physical_metrics(moments[indices].mean(0), length)
        physic_ci = np.quantile(physical_metrics(sampled_moments, length), [.025, .975], axis=0)
        g = direct[indices]
        rows = {}
        for key, pred in predictions.items():
            residual = pred["one_step"].mean(0)[indices] - g[:, 1:]
            drift = pred["one_step"].mean(0)[indices] - g[:, :2]
            # One shared physical-chain resample across all scales and models.
            boot_residual = residual[picks].mean(1)
            rows[key] = dict(mean_R_minus_g=drift.mean(0).tolist(),
                mean_residual=residual.mean(0).tolist(),
                mean_residual_interval95=np.quantile(boot_residual, [.025, .975], axis=0).tolist())
        result.append(dict(size=length, r=float(condition[1]), u=float(condition[2]),
            physical_chains=n, physical_metrics=dict(zip(("absolute_magnetization", "binder",
                "susceptibility_connected", "susceptibility_absolute_subtraction"), physic.tolist())),
            physical_interval95=physic_ci.tolist(), g_mean=g.mean(0).tolist(),
            empirical_drift=np.diff(g.mean(0), axis=0).tolist(), models=rows))
    return result


def write_plots(output, groups, predictions, arrays):
    from scripts.report_phi4_qcr import _plotting
    plt = _plotting(None)
    colors = {36: "#397eaa", 108: "#b6754a"}
    fig, axes = plt.subplots(2, 5, figsize=(16, 6), constrained_layout=True)
    for col, u in enumerate(sorted({g["u"] for g in groups})):
        for length in (36, 108):
            rows = sorted([g for g in groups if g["u"] == u and g["size"] == length], key=lambda x: x["r"])
            if not rows:
                continue
            for ax, field, title in zip(axes[:, col], ("binder", "absolute_magnetization"),
                                         ("宾德累积量", "绝对磁化")):
                ax.plot([g["r"] for g in rows], [g["physical_metrics"][field] for g in rows],
                        "o-", color=colors[length], label=f"L={length}", markersize=3)
                ax.set(title=f"u={u:g}，{title}", xlabel="参数 r", ylabel=title)
        axes[0, col].legend(frameon=False, fontsize=8)
    fig.savefig(output / "physical_comparison.png", dpi=170)
    plt.close(fig)
    for seed in SEEDS:
        key = f"selected_seed{seed}"
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
        for row, length in enumerate((36, 108)):
            rows = [g for g in groups if g["size"] == length]
            for k, ax in enumerate(axes[row]):
                points = np.asarray([g["g_mean"][k] for g in rows])
                target = np.asarray([g["g_mean"][k+1] for g in rows])
                truth = target - points
                predicted = np.asarray([g["models"][key]["mean_R_minus_g"][k] for g in rows])
                ax.scatter(points[:, 0], points[:, 1], color="#777777", s=10, label="输入数据")
                for drift, color, name in ((truth, "#3c875d", "实际粗粒化位移"),
                                            (predicted, "#bb664e", "预测 R(g)−g")):
                    ax.quiver(points[:,0], points[:,1], drift[:,0], drift[:,1], angles="xy",
                              scale_units="xy", scale=1, color=color, width=.004, label=name)
                ax.set(title=f"L={length}，{length//3**k}→{length//3**(k+1)}，R 种子{seed}",
                       xlabel=r"潜变量 $g_1$", ylabel=r"潜变量 $g_2$")
                ax.legend(fontsize=7, frameon=False)
        fig.savefig(output / f"R_vs_g_seed{seed}.png", dpi=170)
        plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    direct, labels = arrays["direct"].mean(0), arrays["labels"]
    for row, length in enumerate((36, 108)):
        mask = labels[:, 0] == length
        for k, ax in enumerate(axes[row]):
            scatter = ax.scatter(labels[mask, 1], direct[mask, k, 1], c=labels[mask, 2],
                                 cmap="viridis", s=14)
            ax.set(title=f"L={length}，尺度 k={k}", xlabel="参数 r", ylabel=r"潜变量 $g_2$")
            fig.colorbar(scatter, ax=ax, label="参数 u")
    fig.savefig(output / "coupling_by_scale.png", dpi=170)
    plt.close(fig)


def run(source_root, data_root, validation, output):
    source_root, data_root, validation, output = map(lambda p: Path(p).resolve(),
                                                   (source_root, data_root, validation, output))
    if output.exists() or output == source_root or source_root in output.parents:
        raise FileExistsError("Prospective evaluation requires a separate fresh output")
    output.mkdir(parents=True)
    cpu, wall = time.process_time(), time.perf_counter()
    hashes = {}
    def guard():
        if time.process_time() - cpu > 600:
            raise BudgetHold("Fixed 600 CPU-second evaluation envelope reached")
    def status(state, **extra):
        source.write_json(output / "status.json", dict(status=state, workflow_complete=state == "complete",
            returncode=0 if state == "complete" else 3 if state == "budget_hold" else 1 if state == "failed" else None,
            cpu_seconds=time.process_time()-cpu, wall_seconds=time.perf_counter()-wall, **extra))
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        status("running", current="verify_fixed_models", error=None)
        verify_inventory(validation, hashes)
        paths, manifest, summary, source_hashes, code_hashes = _source_contract(source_root)
        hashes.update(source_hashes)
        hashes.update(code_hashes)
        hashes[str(Path(__file__).resolve())] = source.sha256_file(__file__)
        maps = restore_maps(validation)
        fixed_R = {key: source.state_fingerprint(model) for key, model in maps.items()}
        source.write_json(output / "manifest.json", dict(schema_version=5,
            recipe="phi4_ru_new_physical_chains_fixed_QC_selected_R_v1",
            source_root=str(source_root), data_root=str(data_root), validation=str(validation),
            source_state_sha256=summary["final_state_sha256"], fixed_R_sha256=fixed_R,
            evaluation_repeats=4, samples_per_encoding=64, physical_chains={"36":120,"108":36},
            frozen_amplitude_scale=summary["scale"], Q_updates=0, C_updates=0, R_updates=0,
            data_used_for_selection=False, input_sha256=dict(hashes)))
        checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
        if checkpoint.get("step") != 3000 or checkpoint.get("manifest") != manifest:
            raise RuntimeError("Source checkpoint contract differs")
        with torch.random.fork_rng(devices=[]):
            model = Phi4QCR(summary["scale"], hidden=96, theory_dim=2).cpu().double()
        model.load_state_dict(checkpoint["model"], strict=True)
        del checkpoint
        model.eval().requires_grad_(False)
        initial = source.state_fingerprint(model)
        if initial != summary["final_state_sha256"]:
            raise RuntimeError("Frozen Q/C source state differs")
        arrays = load_new_encodings(model, data_root, guard, hashes)
        guard()
        raw_eval = source_root / "run/evaluation_arrays.npz"
        hashes[str(raw_eval)] = source.sha256_file(raw_eval)
        with np.load(raw_eval, allow_pickle=False) as z:
            train_g = z["train_direct"].copy()
        # Match target physical lattice sizes to the original TRAIN reference.
        stds = {length:float(np.sqrt(np.var(train_g[:, k],axis=0,ddof=0).mean()))
                for k,length in enumerate((36,12,4))}
        decoder = model.Q.normalized_decoder_weight().detach().numpy()
        bias = model.Q.decoder.bias.detach().numpy()
        predictions, metrics = {}, {}
        reconstruction = arrays["direct"] @ decoder.T + bias
        q_reconstruction = []
        for length in (36,108):
            mask = arrays["labels"][:,0] == length
            for k in range(3):
                rmse = float(np.sqrt(np.mean((reconstruction[:,mask,k] - arrays["feature_target"][:,mask,k])**2)))
                q_reconstruction.append(dict(source_size=length, observed_size=length//3**k,
                    scale=k, native_feature_rmse=rmse,
                    qualification="固定Q/decoder对本次局部统计的重构，非物理信息丢失比例或不可约误差"))
        for key, rmap in maps.items():
            guard()
            predictions[key] = predict(rmap, arrays["direct"])
            metrics[key] = path_metrics(arrays["direct"], predictions[key], arrays["labels"],
                                       stds, decoder, bias, arrays["feature_target"])
        groups = grouped_summary(arrays, predictions)
        public_arrays = {key:value for key,value in arrays.items() if isinstance(value,np.ndarray)}
        # Keep R(mean g) and mean R(g) explicitly separate in the public arrays.
        for key, rmap in maps.items():
            public_arrays.update({f"{key}_{kind}":value for kind,value in predictions[key].items()})
            public_arrays[f"{key}_at_chain_mean"] = predict(rmap, arrays["direct"].mean(0))["one_step"]
        np.savez_compressed(output / "evaluation_arrays.npz", **public_arrays)
        source.write_json(output / "groups.json", groups)
        write_plots(output, groups, predictions, arrays)
        if source.state_fingerprint(model) != initial or any(p.grad is not None for p in model.parameters()):
            raise RuntimeError("Frozen Q/C/R changed during collection")
        if any(source.state_fingerprint(maps[key]) != value for key,value in fixed_R.items()):
            raise RuntimeError("Fixed validation-selected R changed during evaluation")
        _verify_unchanged(hashes)
        guard()
        result = dict(status="complete", workflow_complete=True, returncode=0,
            source_QCR_steps=3000, Q_updates=0, C_updates=0, R_updates=0,
            frozen_source_state_sha256=initial, fixed_R_sha256=fixed_R,
            physical_chains_by_size={"36":120,"108":36}, repeats_per_chain=4,
            conditional_chain_bootstrap=1000, all_models_and_scales_paired_within_size=True,
            cross_sizes_independently_sampled=True, physical_groups=groups,
            latent_and_native_feature_metrics=metrics, target_stds_by_physical_size=stds,
            Q_native_feature_reconstruction=q_reconstruction,
            labels_used_for_evaluation_only=True, new_data_used_for_selection=False,
            inherited_sampling_warnings=arrays["warning_rows"], equilibrium_certified=False,
            new_root_search=False, critical_r=None, thermal_nu=None,
            cpu_seconds=time.process_time()-cpu, wall_seconds=time.perf_counter()-wall,
            GPU_seconds=0, input_sha256=hashes,
            limitations=["变化带由上一轮发现、本轮预先固定；不构成盲测认证。",
                "不确定性条件于固定Q/C及已选定R，不覆盖选点、热化或有限尺寸偏差。",
                "新L108没有重新训练Q/C；局部统计重构误差及尺度依赖残差用于后续定位。",
                "闭合与箭头改善不认证物理相边界或不动点谱。"])
        source.write_json(output / "summary.json", result)
        status("complete", current=None, error=None, summary_sha256=source.sha256_file(output/"summary.json"))
        source.write_json(output / "public_files.json", {p.name:source.sha256_file(p)
            for p in output.iterdir() if p.is_file() and p.name != "public_files.json"})
    except BaseException as error:
        status("budget_hold" if isinstance(error, BudgetHold) else "failed",
               error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source_root", "data_root", "validation", "output"):
        parser.add_argument("--"+name.replace("_","-"), required=True)
    run(**vars(parser.parse_args()))
