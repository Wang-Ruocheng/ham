"""Read-only finite-chain diagnostics for the independent PT / Houdayer trial.

This module reads only the new trial's declared case arrays. It never loads a
model, generates a configuration, extends a chain, or certifies equilibrium.
Real-replica pairs share replicas: average their observables at each saved
time BEFORE autocorrelation or batch-means calculations.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recurrent_rg.data.spin_glass_pipeline import series_diagnostics


QUALIFICATION = (
    "有限相关马尔可夫链的描述性比较，不认证热化、独立性或算法加速。"
    "同一时刻先平均全部真实副本对，副本对不作为独立样本。"
    "随机／有序初态差异及分段漂移的 z 值仅为有限链诊断，不是严格检验或联合置信结论。"
    "保持玻尔兹曼分布的正确更新核不保证本次有限长度样本已经平衡。"
    "所选少量无序实现不是代表性无序平均；不得把温度或副本对计为独立无序样本。"
)
OBSERVABLES = ("energy_density", "q_squared", "q_fourth", "overlap_nearest_neighbor")
BATCH_SIZES = (16, 32, 64)
CHOSEN_BATCH_SIZE = 32


def _sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _finite_json(value):
    """Never serialize IEEE NaN/Infinity, including nested NumPy scalars."""
    if isinstance(value, np.ndarray):
        return _finite_json(value.tolist())
    if isinstance(value, np.generic):
        return _finite_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_finite_json(item) for item in value]
    return value


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(_finite_json(value), ensure_ascii=False,
        indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _contained(root, value):
    path = Path(value)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(root):
        raise ValueError("Declared input path escapes the trial root")
    return path


def _fingerprint(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"Invalid {name} fingerprint")
    return value


def _scalar_series(values):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise ValueError("Expected a nonempty finite scalar time series")
    return result


def batch_means(values, batch_size, *, minimum_blocks=8):
    """Nonoverlapping batch MCSE; short chains remain explicitly unresolved."""
    values = _scalar_series(values)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    blocks = len(values) // batch_size
    result = dict(batch_size=batch_size, blocks=blocks, minimum_blocks=minimum_blocks,
        discarded_tail_samples=len(values) % batch_size, available=blocks >= minimum_blocks,
        block_mean=None, mcse=None)
    if blocks >= minimum_blocks:
        means = values[:blocks * batch_size].reshape(blocks, batch_size).mean(axis=1)
        result.update(block_mean=float(means.mean()),
            mcse=float(np.sqrt(means.var(ddof=1) / blocks)))
    return result


def _difference(first_mean, first_mcse, second_mean, second_mcse):
    """second minus first; quadrature is descriptive, not a mixing guarantee."""
    delta = float(second_mean - first_mean)
    if first_mcse is None or second_mcse is None:
        return dict(difference=delta, combined_mcse=None, descriptive_z=None,
            uncertainty_resolved=False)
    error = float(np.hypot(first_mcse, second_mcse))
    return dict(difference=delta, combined_mcse=error,
        descriptive_z=delta / error if error > 0 else None,
        uncertainty_resolved=error > 0)


def describe_series(values, *, spacing, wall_seconds):
    values = _scalar_series(values)
    diagnostics = series_diagnostics(values)
    batches = {str(size): batch_means(values, size) for size in BATCH_SIZES}
    chosen = batches[str(CHOSEN_BATCH_SIZE)]
    half = len(values) // 2
    first, second = values[:half], values[half:]
    drift = dict(available=False, difference=None, combined_mcse=None, descriptive_z=None)
    if len(first) and len(second):
        first_batch = batch_means(first, CHOSEN_BATCH_SIZE)
        second_batch = batch_means(second, CHOSEN_BATCH_SIZE)
        drift = dict(available=first_batch["available"] and second_batch["available"],
            first_half_mean=float(first.mean()), second_half_mean=float(second.mean()),
            first_half_batch=first_batch, second_half_batch=second_batch,
            **_difference(first.mean(), first_batch["mcse"], second.mean(), second_batch["mcse"]))
    ess = diagnostics["ess"]
    iat = diagnostics["iat_saved_intervals"]
    return dict(**diagnostics, batch_means=batches,
        chosen_batch_size=CHOSEN_BATCH_SIZE, chosen_mcse=chosen["mcse"],
        first_second_half_drift=drift,
        iat_sweeps=None if iat is None else iat * spacing,
        descriptive_ess_per_sampling_wall_second=None if ess is None else ess / wall_seconds,
        rate_denominator="Full sampling wall time including burn-in; separate JIT execution check excluded",
        rate_is_not_an_acceleration_claim=True)


def aggregate_observables(configurations, energies):
    """Return [T,S] observables, never an artificial [T,pair*S] chain."""
    spins = np.asarray(configurations)
    energies = np.asarray(energies, dtype=np.float64)
    if spins.ndim != 5 or spins.shape[-2] != spins.shape[-1] or spins.shape[-1] < 3:
        raise ValueError("Expected square 2D configurations[T,R,S,L,L], L >= 3")
    nt, replicas, samples, side, _ = spins.shape
    if replicas < 2 or samples < 1 or not np.isin(spins, (-1, 1)).all():
        raise ValueError("Require at least two real replicas and binary +/-1 spins")
    if energies.shape != (nt, replicas, samples) or not np.isfinite(energies).all():
        raise ValueError("Finite energies must match configurations[T,R,S]")
    q2 = np.zeros((nt, samples), dtype=np.float64)
    q4, correlation = np.zeros_like(q2), np.zeros_like(q2)
    pair_count = replicas * (replicas - 1) // 2
    for t in range(nt):
        for a in range(replicas):
            for b in range(a + 1, replicas):
                overlap = spins[t, a].astype(np.float64) * spins[t, b]
                q = overlap.mean(axis=(1, 2))
                q2[t] += q ** 2 / pair_count
                q4[t] += q ** 4 / pair_count
                correlation[t] += .5 * (
                    (overlap * np.roll(overlap, 1, axis=1)).mean(axis=(1, 2))
                    + (overlap * np.roll(overlap, 1, axis=2)).mean(axis=(1, 2))) / pair_count
    return dict(energy_density=energies.mean(axis=1) / side ** 2,
        q_squared=q2, q_fourth=q4, overlap_nearest_neighbor=correlation)


def _validate_counts(values, shape, name):
    values = np.asarray(values)
    if values.shape != shape or values.dtype.kind not in "iu" or (values < 0).any():
        raise ValueError(f"Invalid {name} counts")
    return values


def load_case(root, declared, plan):
    key = declared.get("key")
    if not isinstance(key, str) or re.fullmatch(r"[A-Za-z0-9_-]+", key) is None:
        raise ValueError("Unsafe or missing case key")
    receipt_path = _contained(root, f"cases/{key}.json")
    receipt = json.loads(receipt_path.read_text())
    for name in ("key", "algorithm", "initial", "disorder_id"):
        if receipt.get(name) != declared.get(name):
            raise ValueError(f"Case identity differs from frozen plan: {name}")
    if receipt["algorithm"] not in ("pt", "houdayer_pt") or receipt["initial"] not in ("hot", "cold"):
        raise ValueError("Unknown sampling algorithm or initial condition")
    if type(receipt["disorder_id"]) is not int or receipt["disorder_id"] < 0:
        raise ValueError("Invalid disorder identity")
    if receipt.get("status") != "complete" or receipt.get("returncode") != 0:
        raise ValueError("Only complete cases with returncode zero can be reported")
    if receipt.get("source_unchanged") is not True or receipt.get("training_updates", 0) != 0:
        raise ValueError("Require unchanged-source receipt and no training updates")
    for name in ("samples", "replicas", "burn_in", "spacing"):
        if type(receipt.get(name)) is not int or receipt[name] < (0 if name == "burn_in" else 1):
            raise ValueError(f"Invalid case {name}")
        if name in plan and receipt[name] != plan[name]:
            raise ValueError(f"Case {name} differs from frozen plan")
    seconds = receipt.get("sampling_wall_seconds")
    if not isinstance(seconds, (int, float)) or not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("Require finite positive sampling wall time")
    bond_sha = _fingerprint(receipt.get("bond_sha256"), "bond")
    expected_bond = declared.get("expected_bond_sha256")
    if expected_bond is not None and expected_bond != bond_sha:
        raise ValueError("Historical bond fingerprint differs from plan")
    path = _contained(root, receipt["npz_path"])
    expected_sha = _fingerprint(receipt.get("npz_sha256"), "case NPZ")
    if path.stat().st_size != receipt.get("npz_bytes") or _sha(path) != expected_sha:
        raise ValueError("Case NPZ byte count or SHA256 mismatch")
    with np.load(path, allow_pickle=False) as archive:
        required = ("configurations", "energies", "temperatures", "round_trips", "swap_attempts",
            "swap_accepts", "trace_sweeps", "trace_energy_density", "trace_q2")
        if any(name not in archive for name in required):
            raise ValueError("Missing required sampling arrays")
        data = {name: archive[name] for name in required}
        optional = {name: archive[name] for name in archive.files if name.startswith("cluster_")}
    if data["configurations"].ndim != 5 or data["configurations"].shape[1] != receipt["replicas"]:
        raise ValueError("Saved real-replica count differs from receipt")
    if "shape" in plan and list(data["configurations"].shape[-2:]) != plan["shape"]:
        raise ValueError("Saved lattice shape differs from plan")
    sites = int(np.prod(data["configurations"].shape[-2:]))
    observables = aggregate_observables(data.pop("configurations"), data.pop("energies"))
    nt, samples = observables["q_squared"].shape
    replicas = receipt["replicas"]
    if samples != receipt["samples"]:
        raise ValueError("Saved sample count differs from receipt")
    temperatures = np.asarray(data["temperatures"], dtype=np.float64)
    if temperatures.shape != (nt,) or nt < 2 or not np.isfinite(temperatures).all() or (temperatures <= 0).any() or (np.diff(temperatures) <= 0).any():
        raise ValueError("Require a positive increasing temperature ladder")
    if "temperatures" in plan and not np.array_equal(temperatures, plan["temperatures"]):
        raise ValueError("Temperature ladder differs from plan")
    data["round_trips"] = _validate_counts(data["round_trips"], (replicas, nt), "round trip")
    attempts = _validate_counts(data["swap_attempts"], (replicas, nt - 1), "swap attempt")
    accepts = _validate_counts(data["swap_accepts"], (replicas, nt - 1), "swap acceptance")
    if (accepts > attempts).any():
        raise ValueError("Swap acceptances exceed attempts")
    sweeps = data["trace_sweeps"]
    if sweeps.ndim != 1 or not len(sweeps) or not np.isfinite(sweeps).all() or (np.diff(sweeps) <= 0).any() or (sweeps < 0).any():
        raise ValueError("Invalid trace sweep coordinates")
    for name in ("trace_energy_density", "trace_q2"):
        if data[name].shape != (nt, len(sweeps)) or not np.isfinite(data[name]).all():
            raise ValueError(f"Invalid aggregate {name}")
    public_arrays = {**observables, **data}
    # Cluster histograms are aggregate diagnostics, never raw configurations.
    cluster = {name: value.tolist() for name, value in optional.items()}
    if "cluster_size_histogram" in optional:
        histogram = np.asarray(optional["cluster_size_histogram"])
        if histogram.shape != (nt, sites + 1) or histogram.dtype.kind not in "iu" or (histogram < 0).any():
            raise ValueError("Invalid cluster-size histogram")
        cluster.update(empty_cluster_counts=histogram[:, 0].tolist(),
            full_lattice_cluster_counts=histogram[:, -1].tolist(),
            cluster_diagnostics_do_not_certify_mixing=True)
    return receipt, observables, public_arrays, cluster, dict(path=str(receipt_path.relative_to(root)),
        sha256=_sha(receipt_path), npz_path=str(path.relative_to(root)), npz_sha256=expected_sha)


def compare_cases(cases):
    """Compare independent initializations within J, never replica pairs as n."""
    index = {(case["disorder_id"], case["algorithm"], case["initial"]): case for case in cases}
    if len(index) != len(cases):
        raise ValueError("Duplicate (disorder, algorithm, initial) case")
    hot_cold, algorithms = [], []
    disorders = sorted({case["disorder_id"] for case in cases})
    for disorder in disorders:
        for algorithm in ("pt", "houdayer_pt"):
            hot, cold = index.get((disorder, algorithm, "hot")), index.get((disorder, algorithm, "cold"))
            if hot is None or cold is None:
                continue
            for first, second in zip(hot["temperatures"], cold["temperatures"]):
                row = dict(disorder_id=disorder, algorithm=algorithm, temperature=first["temperature"],
                    direction="ordered initial minus random initial", observables={})
                for name in OBSERVABLES:
                    a, b = first[name], second[name]
                    row["observables"][name] = _difference(a["mean"], a["chosen_mcse"], b["mean"], b["chosen_mcse"])
                hot_cold.append(row)
        groups = [[index.get((disorder, algorithm, initial)) for initial in ("hot", "cold")]
                  for algorithm in ("pt", "houdayer_pt")]
        if any(case is None for group in groups for case in group):
            continue
        for t in range(len(groups[0][0]["temperatures"])):
            row = dict(disorder_id=disorder, temperature=groups[0][0]["temperatures"][t]["temperature"],
                direction="Houdayer+PT minus PT; each algorithm averages both initial groups", observables={})
            for name in OBSERVABLES:
                means, errors = [], []
                for group in groups:
                    cells = [case["temperatures"][t][name] for case in group]
                    means.append(float(np.mean([cell["mean"] for cell in cells])))
                    errors.append(None if any(cell["chosen_mcse"] is None for cell in cells)
                        else float(np.hypot(*(cell["chosen_mcse"] for cell in cells)) / 2))
                row["observables"][name] = dict(pt_mean=means[0], houdayer_pt_mean=means[1],
                    **_difference(means[0], errors[0], means[1], errors[1]))
            algorithms.append(row)
    return dict(hot_cold=hot_cold, algorithm_comparison=algorithms,
        comparisons_are_descriptive=True, joint_confidence_claim=False,
        finite_data_equilibrium_certified=False, qualification=QUALIFICATION)


def make_figure(cases, comparison, arrays, output, *, font=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ft2font import FT2Font
    from matplotlib.text import Text
    from scripts.report_ising_zh import select_chinese_font

    font_path = select_chinese_font(font)
    properties = font_manager.FontProperties(fname=str(font_path))
    font_manager.fontManager.addfont(str(font_path))
    available = set(FT2Font(str(font_path)).get_charmap())
    plt.rcParams.update({"axes.unicode_minus": False, "font.family": properties.get_name()})
    labels = {"pt": "平行回火", "houdayer_pt": "簇更新＋平行回火"}
    colors = {"pt": "tab:blue", "houdayer_pt": "tab:orange"}
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for case in cases:
        prefix, algorithm = case["key"] + "__", case["algorithm"]
        style = "-" if case["initial"] == "hot" else "--"
        start = "随机初态" if case["initial"] == "hot" else "有序初态（全＋1）"
        label = f"J={case['disorder_id']}：{labels[algorithm]}，{start}"
        for ax, name in ((axes[0, 0], "trace_q2"), (axes[0, 1], "trace_energy_density")):
            ax.plot(arrays[prefix + "trace_sweeps"], arrays[prefix + name][0],
                style, color=colors[algorithm], alpha=.6, lw=.9, label=label)
        rate = [cell["q_squared"]["descriptive_ess_per_sampling_wall_second"] for cell in case["temperatures"]]
        axes[1, 0].plot([cell["temperature"] for cell in case["temperatures"]],
            [np.nan if value is None else value for value in rate], style,
            color=colors[algorithm], alpha=.5, marker=".", ms=3)
    burn = cases[0]["burn_in"]
    for ax in axes[0]:
        ax.axvline(burn, color="black", ls=":", lw=1, label="预定预热终点")
        ax.set_xlabel("累计完整格点访问轮次")
        ax.grid(alpha=.2)
    temperature = cases[0]["temperatures"][0]["temperature"]
    axes[0, 0].set(title=f"最低温 T={temperature:g}：副本对平均 q² 轨迹", ylabel="q²（先平均副本对）")
    axes[0, 1].set(title="最低温：真实副本平均能量密度轨迹", ylabel="每格点能量")
    axes[0, 0].legend(fontsize=6, ncol=2)
    axes[0, 1].legend(fontsize=6, ncol=2)
    axes[1, 0].set(title="q² 的描述性有效样本率；平衡未认证", xlabel="温度 T", ylabel="ESS／完整采样秒数（含预热）")
    for algorithm in ("pt", "houdayer_pt"):
        axes[1, 0].plot([], [], color=colors[algorithm], label=labels[algorithm])
        rows = [row for row in comparison["hot_cold"] if row["algorithm"] == algorithm]
        for disorder in sorted({row["disorder_id"] for row in rows}):
            subset = [row for row in rows if row["disorder_id"] == disorder]
            axes[1, 1].plot([row["temperature"] for row in subset],
                [row["observables"]["q_squared"]["difference"] for row in subset],
                ".-", color=colors[algorithm], alpha=.65,
                label=f"J={disorder}：{labels[algorithm]}")
    axes[1, 1].axhline(0, color="black", lw=.8)
    axes[1, 1].set(title="随机／有序初态组的 q² 差异（非平衡认证）", xlabel="温度 T", ylabel="有序初态均值－随机初态均值")
    for ax in axes[1]:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("二维自旋玻璃采样对照：固定无序、独立初态；不训练模型")
    for artist in fig.findobj(match=Text):
        missing = {char for char in artist.get_text() if "\u3400" <= char <= "\u9fff" and ord(char) not in available}
        if missing:
            raise ValueError(f"Chinese font lacks figure glyphs: {''.join(sorted(missing))}")
        copy = artist.get_fontproperties().copy()
        copy.set_file(str(font_path))
        artist.set_fontproperties(copy)
    fig.tight_layout()
    fig.savefig(Path(output) / "sampling_comparison.png", dpi=160)
    plt.close(fig)
    return dict(path=str(font_path), sha256=_sha(font_path), family=properties.get_name())


def report(root, *, output=None, font=None):
    root = Path(root).resolve()
    plan_path = _contained(root, "plan.json")
    plan = json.loads(plan_path.read_text())
    declared = plan.get("cases")
    if not isinstance(declared, list) or not declared:
        raise ValueError("Frozen plan must declare a nonempty case list")
    keys = [case.get("key") for case in declared]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate planned case keys")
    output = root / "report" if output is None else Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    _write_json(output / "status.json", dict(status="running", started_at=started,
        training_updates=0, model_inference_calls=0))
    try:
        cases, arrays, inputs, bonds, ladder = [], {}, [], {}, None
        for case in declared:
            receipt, observables, public, cluster, source = load_case(root, case, plan)
            disorder = receipt["disorder_id"]
            if disorder in bonds and bonds[disorder] != receipt["bond_sha256"]:
                raise ValueError("One disorder has conflicting bond fingerprints")
            bonds[disorder] = receipt["bond_sha256"]
            if ladder is not None and not np.array_equal(ladder, public["temperatures"]):
                raise ValueError("Cases have different temperature ladders")
            ladder = public["temperatures"]
            prefix = receipt["key"] + "__"
            arrays.update({prefix + name: value for name, value in public.items()})
            rows = []
            for t, temperature in enumerate(ladder):
                row = dict(temperature=float(temperature), temperature_index=t)
                for name in OBSERVABLES:
                    row[name] = describe_series(observables[name][t], spacing=receipt["spacing"],
                        wall_seconds=receipt["sampling_wall_seconds"])
                q2, q4 = row["q_squared"]["mean"], row["q_fourth"]["mean"]
                row["binder_1_minus_q4_over_3q2sq"] = 1 - q4 / (3 * q2 ** 2) if q2 > 0 else None
                rows.append(row)
            rates = np.divide(public["swap_accepts"], public["swap_attempts"],
                out=np.full_like(public["swap_accepts"], np.nan, dtype=float), where=public["swap_attempts"] > 0)
            flags = []
            if not np.asarray(public["round_trips"]).any():
                flags.append("No complete temperature-label round trip observed; not alone a proof of disequilibrium")
            if not np.isfinite(rates).all() or (rates < .05).any():
                flags.append("Some neighbor swap rates are missing or below 5%; rates alone do not certify mixing")
            for row in rows:
                q = row["q_squared"]
                if q["ess"] is None or q["ess"] < 32:
                    flags.append(f"T={row['temperature']:g}: limited aggregate q^2 ESS evidence")
                if q["chosen_mcse"] is None:
                    flags.append(f"T={row['temperature']:g}: fewer than 8 blocks for chosen batch size 32")
                z = q["first_second_half_drift"].get("descriptive_z")
                if z is not None and abs(z) >= 3:
                    flags.append(f"T={row['temperature']:g}: aggregate q^2 half-chain descriptive |z| >= 3")
            cases.append(dict(key=receipt["key"], disorder_id=disorder, algorithm=receipt["algorithm"],
                initial=receipt["initial"], initial_interpretation="Random spins" if receipt["initial"] == "hot" else "All +1 ordered spins, not a ground-state or low-energy claim",
                bond_sha256=receipt["bond_sha256"], samples=receipt["samples"],
                replicas=receipt["replicas"], pairs=receipt["replicas"] * (receipt["replicas"] - 1) // 2,
                pair_combinations_are_not_independent=True, scalar_time_series_length=receipt["samples"],
                aggregation="At each saved time average q^2 and q^4 across all unordered real-replica pairs; energy across real replicas",
                burn_in=receipt["burn_in"], spacing=receipt["spacing"],
                sampling_wall_seconds=receipt["sampling_wall_seconds"],
                jit_execution_check_wall_seconds=receipt.get("jit_execution_check_wall_seconds"),
                trace_aggregation=receipt.get("trace_aggregation", "Supplied group-level trace; real replicas/pairs already averaged at each trace time"),
                temperatures=rows, swap_acceptance=rates, round_trips=public["round_trips"],
                round_trips_note="Temperature-slot label transport only; cluster exchanges do not preserve an intact configuration ancestry",
                cluster=cluster, kernel_diagnostics=receipt.get("diagnostics", {}),
                quality_flags=flags, equilibrium_certified=False))
            inputs.append(source)
        comparison = compare_cases(cases)
        diagnostics = dict(schema_version=1, cases=cases, case_count=len(cases),
            independent_disorder_ids=sorted(bonds), independent_disorders=len(bonds),
            batch_sizes=list(BATCH_SIZES), chosen_batch_size=CHOSEN_BATCH_SIZE,
            minimum_blocks=8, equilibrium_certified=False, qualification=QUALIFICATION,
            source_unchanged=True, training_updates=0, model_inference_calls=0,
            selection=plan.get("selection"), execution_smoke=plan.get("execution_smoke", False))
        _write_json(output / "diagnostics.json", diagnostics)
        _write_json(output / "comparison.json", comparison)
        with (output / "series.npz").open("xb") as stream:
            np.savez(stream, **arrays)
        font_receipt = make_figure(cases, comparison, arrays, output, font=font)
        finished = datetime.now(timezone.utc).isoformat()
        _write_json(output / "status.json", dict(status="complete", returncode=0,
            started_at=started, finished_at=finished, training_updates=0,
            model_inference_calls=0, equilibrium_certified=False))
        files = [dict(path=name, sha256=_sha(output / name), bytes=(output / name).stat().st_size)
            for name in ("diagnostics.json", "comparison.json", "series.npz", "sampling_comparison.png", "status.json")]
        result = dict(status="complete", returncode=0, schema_version=1, started_at=started,
            finished_at=finished, case_count=len(cases), plan_sha256=_sha(plan_path), inputs=inputs,
            public_files=files, chinese_font=font_receipt, training_updates=0,
            model_inference_calls=0, equilibrium_certified=False,
            scientific_result=plan.get("scientific_result", not plan.get("execution_smoke", False)),
            qualification=QUALIFICATION)
        _write_json(output / "report.json", result)
        return result
    except Exception as exc:
        _write_json(output / "status.json", dict(status="failed", error=f"{type(exc).__name__}: {exc}",
            training_updates=0, model_inference_calls=0, requires_user_action=True))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--font", type=Path)
    args = parser.parse_args()
    result = report(args.root, output=args.output, font=args.font)
    print(json.dumps(dict(status=result["status"], case_count=result["case_count"],
        equilibrium_certified=False), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
