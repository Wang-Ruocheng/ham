"""Saved single-J EA observables, without sampling, models or phase-point fits.

Primary pair01 is retained. Six-pair summaries average the linear moments and
spectral powers at each saved time before nonlinear ratios, never treating
pairs as independent chains. Time quarters are descriptive, not confidence
intervals. Only J0's physical shard is read.
"""
from __future__ import annotations

import argparse
from itertools import combinations
import hashlib
import json
from pathlib import Path
import time

import numpy as np


SOURCE_MANIFEST_SHA = "05c3699d408e9e08fbda14fb5815fb3b2b2c61a481ed19ffd4a19e0e845eb48e"
SOURCE_SHARD_SHA = "0c60441a92f1252657c89110d9f1f729531f696da8814f729941eb2d212b34dc"
BOND_SHA = "e9eab00e409e9683c051f40a0ce3f25ebbdcef3ee5f6515b9d2f34be6479618a"
TC_REFERENCE = 1.1019
KEYS = ("q", "abs_q", "q2", "q4", "structure_factor")


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(json_safe(value), stream, indent=2, ensure_ascii=False, allow_nan=False)


def sample_observables(q_fields):
    fields = np.asarray(q_fields)
    if (fields.ndim != 4 or not len(fields) or len(set(fields.shape[1:])) != 1
            or fields.shape[1] < 2 or not np.isin(fields, (-1, 1)).all()):
        raise ValueError("Expected binary overlap fields [saved_time,L,L,L], L>=2")
    L, N = fields.shape[1], int(np.prod(fields.shape[1:]))
    q = fields.mean(axis=(1, 2, 3), dtype=np.float64)
    spectra = np.empty((len(fields), 4), dtype=np.float64)
    spectra[:, 0] = N * q ** 2
    phase = np.exp(2j * np.pi * np.arange(L) / L)
    for axis in range(3):
        summed = fields.sum(axis=tuple(1 + j for j in range(3) if j != axis), dtype=np.float64)
        mode = np.einsum("sx,x->s", summed, phase) / N
        spectra[:, axis + 1] = N * np.abs(mode) ** 2
    return dict(q=q, abs_q=np.abs(q), q2=q ** 2, q4=q ** 4, structure_factor=spectra)


def _xi(S0, Sk, L):
    if not np.isfinite(S0) or not np.isfinite(Sk) or Sk <= max(1., abs(S0)) * 1e-24:
        return np.nan, np.nan, "zero_or_invalid_minimum_wavevector_power", False
    ratio = S0 / Sk - 1.
    tolerance = 1e-12 * max(1., abs(S0 / Sk))
    if ratio < -tolerance:
        return np.nan, ratio, "negative_second_moment_ratio", False
    rounded = ratio < 0
    # Only machine-precision roundoff, never a finite negative estimator.
    nonnegative_ratio = 0. if rounded else ratio
    return np.sqrt(nonnegative_ratio) / (2 * L * np.sin(np.pi / L)), ratio, (
        "valid_roundoff_negative_within_1e-12_tolerance" if rounded else "valid"), True


def aggregate_observables(samples, L):
    if type(L) is not int or L < 2:
        raise ValueError("Require integer L>=2")
    lengths = {len(np.asarray(samples[k])) for k in KEYS}
    if len(lengths) != 1 or next(iter(lengths)) < 1:
        raise ValueError("Observable traces must share a nonempty time axis")
    if any(not np.isfinite(samples[k]).all() for k in KEYS):
        raise ValueError("Nonfinite observable traces")
    if np.asarray(samples["structure_factor"]).shape != (next(iter(lengths)), 4):
        raise ValueError("Four spectral powers per saved time required")
    m2, m4 = float(np.mean(samples["q2"])), float(np.mean(samples["q4"]))
    spectra = np.mean(samples["structure_factor"], axis=0)
    N = L ** 3
    np.testing.assert_allclose(spectra[0], N * m2, rtol=1e-12, atol=1e-12)
    U4 = m4 / m2 ** 2 if m2 > 0 else np.nan
    xi, ratio, reason, valid = _xi(spectra[0], spectra[1:].mean(), L)
    axes = [_xi(spectra[0], value, L) for value in spectra[1:]]
    return dict(q_mean=float(np.mean(samples["q"])), abs_q_mean=float(np.mean(samples["abs_q"])),
        q2_mean=m2, q4_mean=m4, chi_sg=N * m2, binder=.5 * (3 - U4), U4=U4,
        S0=float(spectra[0]), S_kmin_axes=spectra[1:].tolist(), S_kmin_mean=float(spectra[1:].mean()),
        second_moment_ratio=float(ratio), xi_over_L=float(xi), xi_over_L_axes=[float(a[0]) for a in axes],
        xi_valid=bool(valid), xi_reason=reason, xi_axis_reasons=[a[2] for a in axes])


def replica_observables(spins):
    spins = np.asarray(spins)
    if (spins.ndim != 5 or spins.shape[0] < 2 or spins.shape[1] < 1
            or len(set(spins.shape[2:])) != 1 or not np.isin(spins, (-1, 1)).all()):
        raise ValueError("Expected binary spins [replica,saved_time,L,L,L]")
    pairs = list(combinations(range(spins.shape[0]), 2))
    values = [sample_observables(spins[a] * spins[b]) for a, b in pairs]
    return dict(primary=values[0],
        pair_average={key: np.mean(np.stack([v[key] for v in values]), axis=0) for key in KEYS},
        pairs=pairs, pair_q=np.stack([v["q"] for v in values]))


def windows(samples, L, count):
    size = len(samples["q"])
    if size % count:
        raise ValueError("Time windows must divide the full trace without dropping samples")
    return [aggregate_observables({k: v[i * size // count:(i + 1) * size // count]
             for k, v in samples.items()}, L) for i in range(count)]


def compute(source, output):
    source, output = Path(source).resolve(), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    manifest = json.loads((source / "manifest.json").read_text())
    status = json.loads((source / "status.json").read_text())
    if (sha(source / "manifest.json") != SOURCE_MANIFEST_SHA or manifest["status"] != "complete"
            or status["workflow_complete"] is not True):
        raise ValueError("Unexpected or incomplete frozen physical source")
    row = next(r for r in manifest["shards"] if r["disorder_id"] == 0)
    if (row["path"] != "shards/disorder_000000.npz" or row["sha256"] != SOURCE_SHARD_SHA
            or row["bond_sha256"] != BOND_SHA):
        raise ValueError("Unexpected predeclared J0 shard or bond identity")
    shard = source / row["path"]
    protected = [source / n for n in ("manifest.json", "status.json", "plan.json", "sampling.json")] + [shard]
    before = {str(p): sha(p) for p in protected}
    if before[str(shard)] != SOURCE_SHARD_SHA or shard.stat().st_size != row["bytes"]:
        raise ValueError("Source shard fingerprint/size mismatch")
    sampling = json.loads((source / "sampling.json").read_text())
    original = next(r for r in sampling["disorders"] if r["disorder_id"] == 0)
    edges = np.linspace(-1, 1, 81)
    arrays, rows = {}, []
    started = time.monotonic()
    with np.load(shard, allow_pickle=False) as z:
        if int(z["disorder_id"]) != 0:
            raise ValueError("Wrong saved disorder")
        bonds = z["bonds"]
        if hashlib.sha256(np.ascontiguousarray(bonds, dtype="<f8").tobytes()).hexdigest() != BOND_SHA:
            raise ValueError("Saved bonds disagree with manifest")
        spins, temperatures = z["configurations"], z["temperatures"]
        if spins.shape != (20, 4, 512, 16, 16, 16) or spins.dtype != np.int8:
            raise ValueError("Unexpected full raw data geometry")
        arrays.update(temperatures=temperatures.copy(), temperature_indices=np.arange(20),
                      histogram_edges=edges, saved_sample_indices=np.arange(512))
        store = {prefix + "_" + key: [] for prefix in ("primary", "pair_average") for key in KEYS}
        hist_primary, hist_pairs, hist_quarters, pair_q, magnetization = [], [], [], [], []
        for t, T in enumerate(temperatures):
            value = replica_observables(spins[t])
            record = dict(temperature_index=t, temperature=float(T))
            for name in ("primary", "pair_average"):
                trace = value[name]
                record[name] = aggregate_observables(trace, 16)
                record[name + "_quarters"] = windows(trace, 16, 4)
                record[name + "_halves"] = windows(trace, 16, 2)
                for key in KEYS:
                    store[name + "_" + key].append(trace[key])
            # Independently match existing physical receipt moments; new spectral powers are not fitted.
            old = original["temperatures"][t]["overlap_pairs"][0]
            if old["real_replicas"] != [0, 1]:
                raise ValueError("Original primary pair changed")
            np.testing.assert_allclose([record["primary"]["q2_mean"], record["primary"]["q4_mean"],
                record["primary"]["chi_sg"]], [old["q2"], old["q4"], old["raw_overlap_susceptibility"]],
                rtol=1e-12, atol=1e-12)
            record["primary"]["original_receipt_moments_match"] = True
            q = value["primary"]["q"]
            hist_primary.append(np.histogram(q, edges, density=True)[0])
            hist_pairs.append(np.mean([np.histogram(qp, edges, density=True)[0] for qp in value["pair_q"]], axis=0))
            hist_quarters.append(np.stack([np.histogram(q[a:a + 128], edges, density=True)[0] for a in range(0, 512, 128)]))
            pair_q.append(value["pair_q"])
            magnetization.append(spins[t, 0].mean(axis=(1, 2, 3), dtype=np.float64))
            rows.append(record)
    arrays.update({key: np.stack(values) for key, values in store.items()})
    arrays.update(primary_Pq_density=np.stack(hist_primary), pair_average_Pq_density=np.stack(hist_pairs),
                  primary_quarter_Pq_density=np.stack(hist_quarters), pair_q=np.stack(pair_q),
                  magnetization_replica0=np.stack(magnetization))
    after = {str(p): sha(p) for p in protected}
    if before != after:
        raise RuntimeError("Read-only source fingerprint changed")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "observables.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    result = dict(status="complete", disorder_id=0, bond_sha256=BOND_SHA, L=16, N=4096,
        temperatures=20, saved_times_per_temperature=512, primary_replica_pair=[0, 1],
        secondary_pairs=[list(pair) for pair in combinations(range(4), 2)],
        source_files_sha256=before, source_files_unchanged=True,
        source_sampler_warning_count=len(original["warnings"]), source_sampler_warnings=original["warnings"],
        source_sampling_diagnostic="Existing warnings retained; no quality gate or extension introduced",
        script_sha256=sha(Path(__file__)), output_arrays_sha256=sha(output / "observables.npz"),
        all20_original_primary_moment_checks_passed=True,
        formulas=dict(q="mean_x s0(x)*s1(x)", S="N*mean_time(abs(mean_x qfield(x)*exp(i*k*x))**2)",
            chi_sg="N*mean(q**2), no beta factor and no empirical signed-q subtraction",
            binder="0.5*(3-mean(q**4)/mean(q**2)**2); not 1-U4/3",
            Smin="mean of the three minimum-wavevector powers, before any ratio",
            xi_over_L="sqrt(S0/Smin-1)/(2*L*sin(pi/L)); invalid estimates are null, not clipped",
            zero_denominator="Sk<=1e-24*max(1,S0) is numerically zero",
            roundoff_tolerance="Only negative ratio within relative1e-12 is interpreted as numerical zero"),
        grouping="Six related pairs: average moments and spectral powers within the same time first, then time; never six independent chains",
        time_windows=dict(quarters=[[0, 128], [128, 256], [256, 384], [384, 512]],
                          halves=[[0, 256], [256, 512]], stop_exclusive=True,
                          uncertainty="Descriptive time segments, not independent blocks or confidence intervals"),
        histogram="Raw signed q, 80 fixed equal-width bins on[-1,1]; no flip alignment, folding or symmetrization",
        Tc_reference=dict(value=TC_REFERENCE, source="https://arxiv.org/abs/1310.2910",
                          qualification="Bulk literature reference, not fitted to J0 L16"),
        ICM_temperature_cutoff=1.0, equilibrium_certified=False, independent_chain_CI=False,
        physical_sampling_updates=0, model_inference=False, training_updates=0, fitted_Tc=None, thermal_nu=None,
        qualification="One J and one original L16 only. No finite-size crossing, thermodynamic transition or thermal exponent is certified. RG-blocked sizes are not independent physical sizes.",
        elapsed_seconds=time.monotonic() - started, rows=rows)
    write_json(output / "observables.json", result)
    files = ("observables.json", "observables.npz")
    public = {name: dict(sha256=sha(output / name), bytes=(output / name).stat().st_size) for name in files}
    write_json(output / "public_sha256.json", public)
    return dict(output=str(output), public=public, public_manifest_sha256=sha(output / "public_sha256.json"),
                source_files_unchanged=True, elapsed_seconds=result["elapsed_seconds"])


def plot(directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import PowerNorm
    from matplotlib.ft2font import FT2Font
    from matplotlib.text import Text
    from scripts.report_ising_zh import select_chinese_font
    directory = Path(directory).resolve()
    report = json.loads((directory / "observables.json").read_text())
    if sha(directory / "observables.npz") != report["output_arrays_sha256"]:
        raise ValueError("Public array fingerprint mismatch")
    out = directory / "figures"
    out.mkdir(exist_ok=False)
    font = select_chinese_font(); font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font)).get_name(), "axes.unicode_minus": False})
    available = set(FT2Font(str(font)).get_charmap())
    T = np.array([r["temperature"] for r in report["rows"]])
    def values(key, mode="primary"):
        return np.array([np.nan if r[mode][key] is None else r[mode][key] for r in report["rows"]])
    outputs = []
    def save(fig, name):
        for text in fig.findobj(match=Text):
            missing = {c for c in text.get_text() if ord(c) > 127 and ord(c) not in available}
            if missing:
                raise ValueError(f"字体缺字：{missing}")
        fig.savefig(out / name, dpi=160); plt.close(fig); outputs.append(name)
    def references(ax, labels=False):
        ax.axvline(TC_REFERENCE, color="black", linestyle="--", alpha=.6,
                   label="体相文献参考 Tc≈1.102" if labels else None)
        ax.axvline(1., color="purple", linestyle=":", alpha=.55,
                   label="ICM 算法启用边界 T=1" if labels else None)
        ax.set_xlabel("温度 T"); ax.grid(alpha=.2)
    keys = ("abs_q_mean", "q2_mean", "chi_sg", "binder", "xi_over_L", "q_mean")
    titles = ("平均重叠幅度", "重叠二阶矩", "玻璃磁化率（无额外 β 因子）", "单 J 的 Binder 比率", "重叠二阶矩关联长度", "有符号重叠均值（检查符号采样）")
    labels = (r"$\langle |q|\rangle$", r"$\langle q^2\rangle$", r"$\chi_{SG}$", r"$B=(3-U_4)/2$", r"$\xi_L/L$", r"$\langle q\rangle$")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for i, (ax, key, title, label) in enumerate(zip(axes.flat, keys, titles, labels)):
        for mode, style, name in (("primary", "o-", "固定副本对 0/1（主结果）"),
                                   ("pair_average", "s--", "六对先组内平均（非独立）")):
            ax.plot(T, values(key, mode), style, markersize=3, linewidth=1.1, label=name)
        references(ax, labels=i == 0); ax.set(title=title, ylabel=label)
        if key in ("q2_mean", "chi_sg"):
            ax.set_yscale("log")
        if key == "q_mean": ax.axhline(0, color="gray", linewidth=.8)
        if i == 0: ax.legend(fontsize=7)
    fig.suptitle("固定 J₀、原始 L=16：全部 512 个保存时刻\n单尺寸描述性统计，无相变温度拟合、无独立样本置信区间", fontsize=14)
    save(fig, "physical_observables.png")

    with np.load(directory / "observables.npz", allow_pickle=False) as a:
        edges, hist = a["histogram_edges"], a["primary_Pq_density"]
        mid = (edges[:-1] + edges[1:]) / 2
        te = np.r_[T[0] ** 1.5 / np.sqrt(T[1]), np.sqrt(T[:-1] * T[1:]), T[-1] ** 1.5 / np.sqrt(T[-2])]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
        mesh = axes[0].pcolormesh(edges, te, hist, shading="flat", cmap="viridis", norm=PowerNorm(.5))
        fig.colorbar(mesh, ax=axes[0], label="重叠概率密度")
        axes[0].axhline(TC_REFERENCE, color="white", linestyle="--", linewidth=1)
        axes[0].set(xlabel="整体重叠 q", ylabel="温度 T", title="原始有符号 P(q)：全部温度")
        for index in (0, 7, 8, 19):
            axes[1].step(mid, np.where(hist[index] > 0, hist[index], np.nan), where="mid", label=f"T={T[index]:.3f}")
        axes[1].set(xlabel="整体重叠 q", ylabel="概率密度（对数轴）", title="预先固定的四个温度", yscale="log")
        axes[1].legend(); axes[1].grid(alpha=.2)
        fig.suptitle("同一 J₀ 已保存构型的重叠统计：固定副本对 0/1\n原始符号保留，未折叠、未对称化；数据尚未认证热化", fontsize=13)
        save(fig, "overlap_distributions.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, key, title in zip(axes.flat, ("q2_mean", "binder", "xi_over_L", "q_mean"),
            ("重叠二阶矩的时段变化", "Binder 比率的时段变化", "关联长度比的时段变化", "有符号重叠的时段变化")):
        for quarter in range(4):
            y = [r["primary_quarters"][quarter][key] for r in report["rows"]]
            ax.plot(T, y, ".-", markersize=3, linewidth=.9, label=f"保存序号 {quarter * 128 + 1}—{(quarter + 1) * 128}")
        ax.plot(T, values(key), "k-", linewidth=1.7, label="全部 512 时刻")
        references(ax); ax.set_title(title); ax.legend(fontsize=7)
        if key == "q2_mean": ax.set_yscale("log")
        if key == "q_mean": ax.axhline(0, color="gray", linewidth=.8)
    fig.suptitle("固定副本对 0/1：四个连续时间段，不是四条独立链\n保留全部数据；时段差异仅作稳定性诊断，不是置信区间", fontsize=14)
    save(fig, "time_segment_stability.png")
    result = dict(plots={n: sha(out / n) for n in outputs}, source_json_sha256=sha(directory / "observables.json"),
                  source_npz_sha256=sha(directory / "observables.npz"), no_model_or_sampling=True)
    write_json(out / "plot_receipt.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    if args.plot:
        result = plot(args.plot)
    elif args.source and args.output:
        result = compute(args.source, args.output)
    else:
        parser.error("Use --source/--output or --plot")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
