"""Bounded CPU-only rendering on the original host; export PNG/JSON only.

Run via SSH stdin from this saved source. Never downloads/saves fields or a
checkpoint, never constructs Q/R, and never trains or alters a frozen run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np


FORMAL_ROOT = Path("/mnt/nas-new/home/yange/ham/results/phi4_hopping_b3_g1_aligned_seed91_hmc004_budget4h_gpu3_vm225_20261031")
SNAPSHOT = Path("/mnt/nas-new/home/yange/ham/run_code/phi4_hopping_aligned_seed91_refined_hmc_20261031")
EXPECTED_C = "bfcece7c7f45249a616a5762700fec6d8aaba60e4a7ea3a58e466bb17b484332"
EXPECTED_DATA = "61892f0306ad0c2596f621dd4cb0fb94fd67ce54ee4372ebeb54ee8c2f82c683"
CELLS = (0, 6, 12)
CHAINS = (0, 2)
SAVED_INDEX = 64
NOISE_BASE_SEED = 20314231


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_fingerprint(module):
    result = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        digest = hashlib.sha256()
        for part in (str(value.dtype).encode(), str(value.shape).encode(), value.tobytes()):
            digest.update(part)
        result.update(name.encode())
        result.update(digest.hexdigest().encode())
    return result.hexdigest()


def block_mean(x):
    """Block origin (0,0), row-major, nonoverlapping stride-3 average."""
    length = x.shape[-1]
    if x.ndim != 2 or x.shape[0] != length or length % 3:
        raise ValueError("Expected a square array divisible by three")
    result = x.reshape(length // 3, 3, length // 3, 3).mean(axis=(1, 3))
    if not np.isclose(result.mean(), x.mean(), rtol=1e-12, atol=1e-12):
        raise AssertionError("Block mean must preserve spatial mean")
    return result


def moments(field, scale):
    value = np.asarray(field) / scale
    return dict(size=int(value.shape[0]), mean_over_s=float(value.mean()),
                rms_over_s=float(np.sqrt(np.mean(value ** 2))),
                std_over_s=float(value.std()), minimum_over_s=float(value.min()),
                maximum_over_s=float(value.max()))


def render_grid(plt, records, method, chain, scale, limit, output):
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 10.4))
    titles = {
        "block_mean": "常用标准核：非重叠三乘三块平均",
        "learned_sample": "最终训练 C：实际随机输出（3000 步）",
        "learned_mean_chain": "最终训练 C：逐层取条件均值（辅助诊断）",
    }
    for row, record in enumerate(records):
        fields = record[method]
        for col, field in enumerate(fields):
            ax = axes[row, col]
            picture = ax.imshow(field / scale, cmap="RdBu_r", vmin=-limit, vmax=limit,
                                interpolation="nearest", origin="upper")
            ax.set_xticks([])
            ax.set_yticks([])
            level = f"原始场：{field.shape[0]}乘{field.shape[0]}" if col == 0 else f"粗粒化 {col} 次：{field.shape[0]}乘{field.shape[0]}"
            ax.set_title(level + "\n" + rf"$m/s={np.mean(field)/scale:+.3f}$", fontsize=11)
            if col == 0:
                tag = "（扫描中点，非认证临界点）" if record["cell_index"] == 6 else ""
                ax.set_ylabel(rf"$\kappa={record['kappa']:.2f}$" + "\n" + tag, fontsize=11)
    fig.suptitle(titles[method] + "\n" + f"同一组留出构型；物理链 {chain}；保存序号 {SAVED_INDEX}（从零计数）", fontsize=15)
    fig.subplots_adjust(left=0.15, right=0.88, bottom=0.10, top=0.89, hspace=0.27, wspace=0.06)
    cax = fig.add_axes((0.90, 0.17, 0.025, 0.65))
    fig.colorbar(picture, cax=cax, label=r"场幅度 $\phi/s$（全部图共用色标）")
    if method == "block_mean":
        note = "每步取九点算术平均；无额外场幅度重标度、无噪声；不是唯一或精确的 RG 核。"
    elif method == "learned_sample":
        note = r"每步为 $\mu_\theta(\phi_B)+0.1s\,\xi$；各样本独立固定噪声种子；不翻转整体符号。"
    else:
        note = "逐层以条件均值替代采样；此图不是随机 C 两步输出的期望。"
    fig.text(0.5, 0.052, note, ha="center", fontsize=10)
    fig.text(0.5, 0.025, "预先固定首／中／末条件，不按图像选样；有限样本展示不认证热化、临界点或 RG 正确性。", ha="center", fontsize=10)
    fig.savefig(output, dpi=145)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.process_time()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite an existing preview")
    if FORMAL_ROOT in args.output.resolve().parents:
        raise ValueError("Do not write into the frozen formal experiment")
    job = json.loads((FORMAL_ROOT / "job.json").read_text())
    status = json.loads((FORMAL_ROOT / "training/status.json").read_text())
    if job.get("status") != "complete" or job.get("workflow_complete") is not True:
        raise ValueError("Formal workflow is not complete")
    if status.get("actual_steps") != 3000 or status.get("returncode") != 0:
        raise ValueError("Unexpected training endpoint")
    manifest = json.loads((FORMAL_ROOT / "training/manifest.json").read_text())
    summary = json.loads((FORMAL_ROOT / "training/summary.json").read_text())
    source_pins = manifest["source_sha256"]
    for name, expected in source_pins.items():
        if SNAPSHOT not in Path(name).parents or sha256_file(name) != expected:
            raise ValueError("Frozen source fingerprint mismatch")
    data_root = Path(manifest["data_root"])
    if data_root != FORMAL_ROOT / "data":
        raise ValueError("Unexpected current data root")
    if sha256_file(data_root / "manifest.json") != EXPECTED_DATA or manifest["data_manifest_sha256"] != EXPECTED_DATA:
        raise ValueError("Data manifest fingerprint mismatch")
    physical = json.loads((data_root / "manifest.json").read_text())
    if physical["options"]["sizes"] != [36] or physical["options"]["quartic_lambda"] != 1.0:
        raise ValueError("Unexpected physics geometry")

    sys.path.insert(0, str(SNAPSHOT))
    import torch
    from recurrent_rg.phi4_qcr import GaussianCoarseGrainer
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    checkpoint_path = FORMAL_ROOT / "training/training_step_3000.pt"
    checkpoint_sha = sha256_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["step"] != 3000:
        raise ValueError("Wrong checkpoint step")
    c_state = {name[2:]: value for name, value in checkpoint["model"].items() if name.startswith("C.")}
    scale = float(c_state["scale"])
    model = GaussianCoarseGrainer(scale, hidden=96, noise_ratio=.1).eval()
    model.load_state_dict(c_state, strict=True)
    model.requires_grad_(False)
    if state_fingerprint(model) != EXPECTED_C or summary["final_module_sha256"]["C"] != EXPECTED_C:
        raise ValueError("Final C fingerprint mismatch")
    del checkpoint, c_state

    by_chain = {chain: [] for chain in CHAINS}
    provenance = []
    for cell in CELLS:
        shard = next(row for row in physical["shards"] if row["cell_index"] == cell)
        path, sidecar_path = data_root / shard["path"], data_root / shard["sidecar"]
        if shard["split"] != "test" or sha256_file(path) != shard["sha256"] or sha256_file(sidecar_path) != shard["sidecar_sha256"]:
            raise ValueError("Selected TEST shard fingerprint mismatch")
        sidecar = json.loads(sidecar_path.read_text())
        with np.load(path, allow_pickle=False) as archive:
            values = archive["configurations"]
        if values.shape != (4, 128, 36, 36, 1) or values.dtype != np.float64 or not np.isfinite(values).all():
            raise ValueError("Unexpected physical field array")
        for chain in CHAINS:
            original = values[chain, SAVED_INDEX, :, :, 0].copy()
            standard = [original, block_mean(original)]
            standard.append(block_mean(standard[-1]))
            seed = NOISE_BASE_SEED + cell * 10 + chain
            generator = torch.Generator(device="cpu").manual_seed(seed)
            x = torch.from_numpy(original.copy())[None, None, :, :, None]
            with torch.inference_mode():
                sample1 = model(x, generator=generator)
                sample2 = model(sample1, generator=generator)
                mean1 = model.mean_field(x)
                mean2 = model.mean_field(mean1)
            sampled = [original, sample1[0, 0, :, :, 0].numpy().copy(), sample2[0, 0, :, :, 0].numpy().copy()]
            means = [original, mean1[0, 0, :, :, 0].numpy().copy(), mean2[0, 0, :, :, 0].numpy().copy()]
            record = dict(cell_index=cell, kappa=sidecar["kappa"], block_mean=standard,
                          learned_sample=sampled, learned_mean_chain=means)
            by_chain[chain].append(record)
            provenance.append(dict(cell_index=cell, kappa=sidecar["kappa"], split="test", chain=chain,
                chain_initial=sidecar["chains"][chain]["initial"], saved_index_zero_based=SAVED_INDEX,
                noise_seed=seed, source_path=shard["path"], source_sha256=shard["sha256"],
                moments={key: [moments(value, scale) for value in record[key]]
                         for key in ("block_mean", "learned_sample", "learned_mean_chain")}))
        del values
    if state_fingerprint(model) != EXPECTED_C or any(p.grad is not None for p in model.parameters()):
        raise AssertionError("C was modified during preview")
    if sha256_file(checkpoint_path) != checkpoint_sha:
        raise AssertionError("Checkpoint file changed during preview")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if not font.exists():
        raise FileNotFoundError("Chinese font unavailable")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font)).get_name(),
                         "axes.unicode_minus": False, "font.size": 11})
    # One symmetric scale for every original/coarse field and both chains.
    maximum = max(abs(value).max() / scale for records in by_chain.values()
                  for record in records for key in ("block_mean", "learned_sample", "learned_mean_chain")
                  for value in record[key])
    limit = math.ceil(maximum * 2) / 2
    args.output.mkdir(parents=True, exist_ok=False)
    files = []
    for chain in CHAINS:
        for method in ("block_mean", "learned_sample", "learned_mean_chain"):
            target = args.output / f"{method}_chain{chain}.png"
            render_grid(plt, by_chain[chain], method, chain, scale, limit, target)
            files.append(dict(path=target.name, sha256=sha256_file(target), bytes=target.stat().st_size))
    receipt = dict(diagnostic="current_final_C_configuration_preview_v1", checkpoint_step=3000,
        checkpoint_sha256=checkpoint_sha, final_C_sha256=EXPECTED_C, C_state_unchanged=True,
        all_parameter_gradients_none=True, device="cpu", cpu_threads=1, Q_or_R_inference=False,
        sampling_or_training=False, raw_fields_or_checkpoint_exported=False, physical_fields=6,
        selection="predeclared first/middle/last scan cells, chain0 and chain2, saved index64; no visual or quality selection",
        scale=scale, noise_std=float(model.noise_std), noise_seed_base=NOISE_BASE_SEED,
        standard_kernel="3x3 nonoverlapping arithmetic block average; stride3; origin0; no extra field renormalization; no noise",
        mean_chain_semantics="composition of conditional means, not expectation of two-step random C",
        color_scale_over_s=[-limit, limit], color_clipping=False, global_sign_alignment=False,
        source_pins_verified=source_pins, data_manifest_sha256=EXPECTED_DATA, fields=provenance,
        limitations=["Finite snapshots cannot certify equilibrium, phase boundaries or physical RG correctness",
                     "Middle cell kappa0.68 is scan midpoint, not a certified critical point",
                     "Learned C is noisy and nonlinear; stronger contrast is not proof of better phase separation",
                     "Physical sampling quality warnings from the formal run remain in force"],
        cpu_seconds=time.process_time() - started, public_files=files)
    (args.output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status="complete", output=str(args.output), cpu_seconds=receipt["cpu_seconds"],
                         C_state_unchanged=True, public_files=files,
                         receipt_sha256=sha256_file(args.output / "receipt.json")), ensure_ascii=False))


if __name__ == "__main__":
    main()
