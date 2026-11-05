"""Read-only CPU enumeration of two frozen 2x2x2 binary C kernels.

No lattice data, MI estimator, stochastic draws, R roots, or training is used.
Pattern-averaged diagnostics weight all 256 inputs equally, not physically.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pattern_arrays():
    patterns = 2 * ((np.arange(256)[:, None] // (2 ** np.arange(8))) % 2) - 1
    offsets = np.array(list(itertools.product(range(2), repeat=3)))
    basis = np.ones((256, 256))
    orders = np.array([i.bit_count() for i in range(256)])
    for mask in range(1, 256):
        basis[:, mask] = patterns[:, [i for i in range(8) if mask & (1 << i)]].prod(1)
    return patterns, offsets, basis, orders


def symmetry_indices(patterns, offsets):
    """All 48 cube isometries (rotations and reflections), including identity."""
    permutations = []
    for axes in itertools.permutations(range(3)):
        for flips in itertools.product(range(2), repeat=3):
            transformed = offsets[:, axes].copy()
            for axis, flip in enumerate(flips):
                if flip:
                    transformed[:, axis] = 1 - transformed[:, axis]
            sites = [int(np.flatnonzero((offsets == point).all(1))[0]) for point in transformed]
            permutations.append((((patterns[:, sites] + 1) // 2) * (2 ** np.arange(8))).sum(1))
    assert len({tuple(p) for p in permutations}) == 48
    return np.array(permutations)


def entropy_bits(p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def summarize(p):
    patterns, offsets, basis, orders = pattern_arrays()
    nplus = (patterns > 0).sum(1)
    majority = np.where(nplus > 4, 1., np.where(nplus < 4, 0., .5))
    rules = {"majority_fair_ties": majority,
             "random_input_site": (nplus / 8),
             "independent_fair_coin": np.full(256, .5)}
    for site in range(8):
        rules[f"decimation_site_{site}"] = (patterns[:, site] + 1) / 2
        rules[f"majority_omit_site_{site}"] = (patterns[:, np.arange(8) != site].sum(1) > 0).astype(float)
    coefficients = basis.T @ (2 * p - 1) / 256
    power = np.array([np.square(coefficients[orders == k]).sum() for k in range(9)])
    isometries = symmetry_indices(patterns, offsets)
    cubic_error = p[isometries] - p[None]
    groups = []
    for n in range(9):
        values = p[nplus == n]
        groups.append(dict(nplus=n, patterns=len(values), mean=float(values.mean()),
                           std=float(values.std()), min=float(values.min()), max=float(values.max())))
    monotonicity = []
    for site in range(8):
        source = np.flatnonzero(patterns[:, site] < 0)
        delta = p[source + (1 << site)] - p[source]
        monotonicity.append(dict(site=site, offset=offsets[site].tolist(),
                                 negative_edges_beyond_001=int((delta < -.01).sum()),
                                 mean_probability_change=float(delta.mean()),
                                 min_probability_change=float(delta.min())))
    metrics = dict(
        averaging="uniform over 256 patterns; NOT physical data frequencies",
        p_plus_all_minus=float(p[0]), p_plus_all_plus=float(p[-1]),
        mean_p_plus=float(p.mean()), conditional_entropy_bits=float(entropy_bits(p).mean()),
        mixed_nonuniform_entropy_bits=float(entropy_bits(p[1:-1]).mean()),
        ambiguous_probability_01_to_09_count=int(((p > .1) & (p < .9)).sum()),
        nearly_deterministic_outside_001_to_099_count=int(((p <= .01) | (p >= .99)).sum()),
        flip_error_max=float(np.abs(p + p[::-1] - 1).max()),
        flip_error_mean=float(np.abs(p + p[::-1] - 1).mean()),
        cubic_isometries=48, cubic_probability_rmse=float(np.sqrt(np.square(cubic_error).mean())),
        cubic_probability_max_error=float(np.abs(cubic_error).max()),
        rule_probability_rmse={name: float(np.sqrt(np.square(p - rule).mean()))
                               for name, rule in rules.items()},
        majority_non_tie_threshold_agreement=float(((p[nplus != 4] > .5) == (nplus[nplus != 4] > 4)).mean()),
        nplus_groups=groups, walsh_power_by_order=power.tolist(),
        walsh_linear_site_coefficients=coefficients[2 ** np.arange(8)].tolist(),
        single_site_monotonicity=monotonicity,
        walsh_note="Exact uniform-input expansion of E[coarse spin | block], not Hamiltonian couplings.",
    )
    assert np.isclose(power.sum(), np.square(2 * p - 1).mean())
    return metrics, coefficients


def inspect_run(root):
    import torch
    from recurrent_rg.checkpoints import load_run
    from recurrent_rg.rsmi import kernel_probabilities

    root = Path(root).resolve()
    checkpoint = root / "run/checkpoint_last.pt"
    protected = [checkpoint, root / "job.json", root / "plan.json", root / "config.json",
                 root / "run/manifest.json", root / "run/summary.json", root / "report/report.json"]
    before = {str(path): sha256(path) for path in protected}
    job = json.loads((root / "job.json").read_text())
    if job.get("status") != "complete" or job.get("workflow_complete") is not True:
        raise ValueError("Only complete frozen runs may be inspected")
    loaded = load_run(root / "run", checkpoint="checkpoint_last.pt")
    model = loaded.model.cpu().eval().requires_grad_(False)
    spec = loaded.model_config
    if (spec.dimension, spec.block_factor, spec.kernel_layout, spec.kernel_sampling) != (
            3, 2, "nonoverlapping", "bernoulli"):
        raise ValueError("Expected the paired historical cubic Bernoulli block-2 runs")
    if loaded.payload.get("epoch") != 6000:
        raise ValueError("Expected actual final 6000-update checkpoint")

    def model_sha():
        digest = hashlib.sha256()
        for name, value in sorted(model.state_dict().items()):
            array = value.detach().cpu().contiguous().numpy()
            digest.update(json.dumps([name, str(array.dtype), list(array.shape)]).encode())
            digest.update(array.tobytes())
        return digest.hexdigest()

    state_before = model_sha()
    patterns = torch.tensor(pattern_arrays()[0], dtype=torch.float64)[None]
    with torch.no_grad():
        table = kernel_probabilities(model.rg_kernel, patterns)[0].numpy()
    assert np.isfinite(table).all() and np.allclose(table.sum(1), 1)
    metrics, coefficients = summarize(table[:, 1])
    state_after = model_sha()
    after = {str(path): sha256(path) for path in protected}
    gradients_none = all(parameter.grad is None for parameter in model.parameters())
    if before != after or state_before != state_after or not gradients_none:
        raise RuntimeError("Read-only diagnostic fingerprint mismatch")
    return dict(root=str(root), checkpoint=str(checkpoint), actual_updates=6000,
                protected_sha256=before, protected_files_unchanged=True,
                model_sha256_before=state_before, model_sha256_after=state_after,
                gradients_none=gradients_none, metrics=metrics), table[:, 1], coefficients


def plot(arrays, output, font):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    if not Path(font).is_file():
        raise FileNotFoundError(font)
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=font).get_name(),
                         "axes.unicode_minus": False})
    patterns = arrays["patterns"]
    nplus = (patterns > 0).sum(1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    labels = {"baseline": "含经验 RSMI", "no_mi": "无 RSMI：仅保色与方差下限"}
    for name, color, shift in (("baseline", "tab:blue", -.06), ("no_mi", "tab:orange", .06)):
        p = arrays[name + "_p_plus"]
        means = np.array([p[nplus == n].mean() for n in range(9)])
        axes[0].scatter(nplus + shift, p, s=12, alpha=.2, color=color)
        axes[0].plot(np.arange(9) + shift, means, "o-", color=color, label=labels[name])
        axes[2].bar(np.arange(9) + (shift * 3), np.array(summarize(p)[0]["walsh_power_by_order"]),
                    width=.35, color=color, label=labels[name])
    axes[0].plot(np.arange(9), np.where(np.arange(9) < 4, 0, np.where(np.arange(9) > 4, 1, .5)),
                 "k--", alpha=.7, label="多数表决（平票取正概率 1/2）")
    axes[0].set(title="全部输入块：输出正自旋的概率", xlabel="8 个输入中的正自旋数", ylabel="P(+ | 输入块)")
    axes[0].legend(fontsize=8)
    axes[1].scatter(arrays["baseline_p_plus"], arrays["no_mi_p_plus"], s=15, alpha=.6)
    axes[1].plot([0, 1], [0, 1], "k--", alpha=.5)
    axes[1].set(title="同一输入块的两核直接对照", xlabel="含经验 RSMI：P(+ | 输入块)",
                ylabel="无 RSMI：P(+ | 输入块)", xlim=(-.03, 1.03), ylim=(-.03, 1.03))
    axes[2].set(title="条件输出均值的局域规则分解", xlabel="输入自旋乘积的阶数", ylabel="系数平方和")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle("三维 Ising 最终 C 核：256 种输入等权诊断（非物理频率加权）")
    fig.savefig(output / "kernel_comparison.png", dpi=170)
    plt.close(fig)


def self_test():
    patterns = pattern_arrays()[0]
    decimation = (patterns[:, 0] + 1) / 2
    majority = np.where(patterns.sum(1) > 0, 1., np.where(patterns.sum(1) < 0, 0., .5))
    d, _ = summarize(decimation)
    m, _ = summarize(majority)
    coin, _ = summarize(np.full(256, .5))
    assert d["rule_probability_rmse"]["decimation_site_0"] == 0
    assert d["walsh_power_by_order"][1] == 1
    assert m["rule_probability_rmse"]["majority_fair_ties"] == 0
    assert m["flip_error_max"] == m["cubic_probability_max_error"] == 0
    assert coin["conditional_entropy_bits"] == 1
    assert sum(coin["walsh_power_by_order"]) == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--font", default="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--saved-tables", type=Path,
                        help="Derive additional exact rule details from saved public NPZ only")
    args = parser.parse_args()
    self_test()
    if args.self_test:
        print("Exact-rule diagnostic self-tests passed")
        return
    if args.saved_tables:
        with np.load(args.saved_tables, allow_pickle=False) as arrays:
            details = {name: summarize(arrays[name + "_p_plus"])[0]
                       for name in ("baseline", "no_mi")}
        details["input_sha256"] = sha256(args.saved_tables)
        details["scope"] = "Exact arithmetic on saved public tables; no models or data loaded"
        output = args.saved_tables.parent / "kernel_rule_details.json"
        with output.open("x") as stream:
            stream.write(json.dumps(details, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps(dict(output=str(output), sha256=sha256(output))))
        return
    if not all((args.baseline_root, args.candidate_root, args.output)):
        parser.error("Both run roots and a fresh output directory are required")
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    baseline, p0, w0 = inspect_run(args.baseline_root)
    candidate, p1, w1 = inspect_run(args.candidate_root)
    patterns, offsets, _, _ = pattern_arrays()
    arrays = dict(patterns=patterns, site_offsets=offsets, baseline_p_plus=p0, no_mi_p_plus=p1,
                  baseline_walsh_coefficients=w0, no_mi_walsh_coefficients=w1)
    difference = np.abs(p0 - p1)
    examples = [dict(pattern_code=int(i), spins=patterns[i].tolist(),
                     baseline_p_plus=float(p0[i]), no_mi_p_plus=float(p1[i]))
                for i in np.argsort(-difference)[:12]]
    report = dict(
        scope="CPU-only analytic 256-input kernel evaluation; no MI, training, data, sampling, or R root search",
        cpu_threads=1, gpu_used=False, site_order_xyz=offsets.tolist(),
        code_encoding="code=sum(2**i * (spin_i>0)); lexicographic cube sites, z fastest",
        baseline=baseline, no_mi=candidate,
        comparison=dict(probability_rmse=float(np.sqrt(np.square(p0 - p1).mean())),
                        probability_mae=float(difference.mean()),
                        threshold_disagreement_count=int(((p0 > .5) != (p1 > .5)).sum()),
                        largest_difference_examples=examples),
        caveats=["Uniform-pattern diagnostic is not physical-frequency averaged.",
                 "Both Q and R were cotrained too; kernel comparison alone does not establish causation.",
                 "Empirical RSMI may have finite-sample bias; deterministic C is not proof of physical information retention."])
    np.savez_compressed(args.output / "kernel_tables.npz", **arrays)
    (args.output / "kernel_comparison.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    plot(arrays, args.output, args.font)
    public = {name: sha256(args.output / name) for name in
              ("kernel_comparison.json", "kernel_tables.npz", "kernel_comparison.png")}
    (args.output / "public_sha256.json").write_text(json.dumps(public, indent=2) + "\n")
    print(json.dumps(dict(output=str(args.output), public_sha256=public,
                          baseline_metrics=baseline["metrics"], no_mi_metrics=candidate["metrics"],
                          comparison=report["comparison"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
