"""Render saved high-kappa fields through the unchanged final C on CPU.

Reproduces the first frozen C-noise draw from the previous multi-field
diagnostic and checks scalar results against that diagnostic before drawing.
No field arrays or checkpoints are exported; no Q/R or physical sampling.
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


ROOT = Path("/mnt/nas-new/home/yange/ham/results/phi4_hopping_b3_g1_aligned_seed91_hmc004_budget4h_gpu3_vm225_20261031")
SNAPSHOT = Path("/mnt/nas-new/home/yange/ham/run_code/phi4_hopping_aligned_seed91_refined_hmc_20261031")
PREVIOUS = Path("/mnt/nas-new/home/yange/ham/results/phi4_hopping_final_C_ensemble_convergence_seed91_20261031_v2")
C_SHA = "bfcece7c7f45249a616a5762700fec6d8aaba60e4a7ea3a58e466bb17b484332"
DATA_SHA = "61892f0306ad0c2596f621dd4cb0fb94fd67ce54ee4372ebeb54ee8c2f82c683"
PRIOR_JSON_SHA = "8e6370520acd5e945cec1c2c71f199c8e7494b2e300fab89e979a0a07947ecf3"
PRIOR_ARRAY_SHA = "f3817ef73590193fd6e06616068dc7e7013ab33aa9c5331e3e71a502c82e1130"
CELLS = tuple(range(7, 13))
INDICES = tuple(range(0, 128, 8))
NOISE_BASE_SEED = 20314232


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_fingerprint(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        item = hashlib.sha256()
        for part in (str(value.dtype).encode(), str(value.shape).encode(), value.tobytes()):
            item.update(part)
        digest.update(name.encode())
        digest.update(item.hexdigest().encode())
    return digest.hexdigest()


def select_fields(value):
    if value.shape != (4, 128, 36, 36, 1):
        raise ValueError("Unexpected physical chain/sample/space/channel axes")
    selected = np.ascontiguousarray(np.take(value[..., 0], INDICES, axis=1))
    assert selected.shape == (4, 16, 36, 36)
    return selected


def scalar_summaries(field, scale):
    flat = field.reshape(4, 16, -1) / scale
    mean = flat.mean(axis=-1)
    rms = np.sqrt(np.mean(flat ** 2, axis=-1))
    return dict(m_over_s=mean, rms_over_s=rms, std_over_s=flat.std(axis=-1),
                q=np.abs(mean) / rms,
                majority_fraction=np.maximum((flat > 0).mean(axis=-1), (flat < 0).mean(axis=-1)))


def draw_single(plt, record, scale, limit, path):
    fig, axes = plt.subplots(3, 8, figsize=(19, 8))
    examples = [(chain, position) for chain in range(4) for position in (0, 8)]
    for col, (chain, position) in enumerate(examples):
        for level, field in enumerate(record["fields"]):
            ax = axes[level, col]
            picture = ax.imshow(field[chain, position] / scale, cmap="RdBu_r", vmin=-limit, vmax=limit,
                                interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if level == 0:
                ax.set_title(f"物理链 {chain}／保存序号 {INDICES[position]}", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{field.shape[-1]}乘{field.shape[-1]}", fontsize=12)
    fig.suptitle(rf"$\kappa={record['kappa']:.2f}$：最终 C 对多组构型的粗粒化" + "\n原始场 → 一次粗粒化 → 两次粗粒化", fontsize=16)
    fig.subplots_adjust(left=.055, right=.93, top=.86, bottom=.12, hspace=.10, wspace=.035)
    cax = fig.add_axes((.945, .17, .014, .60))
    fig.colorbar(picture, cax=cax, label=r"场幅度 $\phi/s$（六组共用色标）")
    fig.text(.5, .065, "四条物理链，各展示预先固定的保存序号 0、64；不挑图，不翻转整体正负。", ha="center", fontsize=11)
    fig.text(.5, .035, "最终 3000 步 C 的真实随机输出；只展示训练过的两次粗粒化，不作更深尺度外推。", ha="center", fontsize=11)
    fig.savefig(path, dpi=145)
    plt.close(fig)


def draw_overview(plt, records, scale, limit, path):
    fig = plt.figure(figsize=(13.6, 15.6))
    grid = fig.add_gridspec(6, 7, width_ratios=(1, 1, 1, .20, 1, 1, 1),
                           left=.075, right=.92, bottom=.095, top=.90, hspace=.11, wspace=.06)
    stages = ("原始：36乘36", "一次：12乘12", "两次：4乘4")
    for row, record in enumerate(records):
        for sample, chain in enumerate((0, 2)):
            for level, field in enumerate(record["fields"]):
                column = level + 4 * sample
                ax = fig.add_subplot(grid[row, column])
                picture = ax.imshow(field[chain, 8] / scale, cmap="RdBu_r", vmin=-limit, vmax=limit,
                                    interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(stages[level], fontsize=11)
                if column == 0:
                    ax.set_ylabel(rf"$\kappa={record['kappa']:.2f}$", fontsize=13)
    fig.suptitle("不同耦合系综在最终 C 下的构型变化", fontsize=19, y=.98)
    fig.text(.285, .935, "样本甲：物理链 0／保存序号 64", ha="center", fontsize=12)
    fig.text(.72, .935, "样本乙：物理链 2／保存序号 64", ha="center", fontsize=12)
    cax = fig.add_axes((.945, .18, .016, .64))
    fig.colorbar(picture, cax=cax, label=r"场幅度 $\phi/s$（全部面板共用色标）")
    fig.text(.5, .057, "六组均使用同样预先固定的样本位置；真实随机 C，不按磁化符号选样或作全局正负对齐。", ha="center", fontsize=11)
    fig.text(.5, .033, "每组另提供四条链共八张样本的完整图；构型展示不认证热化、临界点或完整 RG 正确性。", ha="center", fontsize=11)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.process_time()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.output.exists() or ROOT in args.output.resolve().parents or PREVIOUS in args.output.resolve().parents:
        raise FileExistsError("Refuse overwriting any previous artifact")
    for name in ("job.json", "training/status.json"):
        status = json.loads((ROOT / name).read_text())
        if status.get("status") != "complete" or status.get("workflow_complete") is not True:
            raise ValueError("Formal run is not complete")
    manifest = json.loads((ROOT / "training/manifest.json").read_text())
    for name, expected in manifest["source_sha256"].items():
        if SNAPSHOT not in Path(name).parents or sha(name) != expected:
            raise ValueError("Frozen source mismatch")
    data_root = Path(manifest["data_root"])
    if data_root != ROOT / "data" or sha(data_root / "manifest.json") != DATA_SHA:
        raise ValueError("Data manifest mismatch")
    if sha(PREVIOUS / "convergence.json") != PRIOR_JSON_SHA or sha(PREVIOUS / "per_field_metrics.npz") != PRIOR_ARRAY_SHA:
        raise ValueError("Previous diagnostic integrity mismatch")
    physical = json.loads((data_root / "manifest.json").read_text())
    sys.path.insert(0, str(SNAPSHOT))
    import torch
    from recurrent_rg.phi4_qcr import GaussianCoarseGrainer
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    checkpoint_path = ROOT / "training/training_step_3000.pt"
    checkpoint_sha = sha(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["step"] != 3000:
        raise ValueError("Wrong checkpoint step")
    state = {key[2:]: value for key, value in checkpoint["model"].items() if key.startswith("C.")}
    scale = float(state["scale"])
    model = GaussianCoarseGrainer(scale, 96, .1).eval().requires_grad_(False)
    model.load_state_dict(state, strict=True)
    if state_fingerprint(model) != C_SHA:
        raise ValueError("Final C mismatch")
    del state, checkpoint
    records, provenance = [], []
    with np.load(PREVIOUS / "per_field_metrics.npz", allow_pickle=False) as previous:
        for cell in CELLS:
            shard = next(item for item in physical["shards"] if item["cell_index"] == cell)
            source, sidecar_path = data_root / shard["path"], data_root / shard["sidecar"]
            if sha(source) != shard["sha256"] or sha(sidecar_path) != shard["sidecar_sha256"]:
                raise ValueError("Selected shard mismatch")
            sidecar = json.loads(sidecar_path.read_text())
            with np.load(source, allow_pickle=False) as data:
                original = select_fields(data["configurations"])
            if not np.isfinite(original).all():
                raise ValueError("Nonfinite saved field")
            generator = torch.Generator(device="cpu").manual_seed(NOISE_BASE_SEED + 100 * cell)
            x = torch.from_numpy(original[..., None].copy())
            with torch.inference_mode():
                first = model(x, generator=generator)
                second = model(first, generator=generator)
            fields = [original, first[..., 0].numpy().copy(), second[..., 0].numpy().copy()]
            for level, field in enumerate(fields):
                if field.shape != (4, 16, 36 // 3**level, 36 // 3**level):
                    raise AssertionError("Physical chain/sample axes changed")
                for metric, value in scalar_summaries(field, scale).items():
                    expected = previous[f"c{cell:04d}_learned_sample_level{level}_{metric}"][0]
                    np.testing.assert_allclose(value, expected, rtol=2e-12, atol=2e-12)
            records.append(dict(cell_index=cell, kappa=sidecar["kappa"], fields=fields))
            provenance.append(dict(cell_index=cell, kappa=sidecar["kappa"], split=shard["split"],
                                   path=shard["path"], sha256=shard["sha256"],
                                   noise_seed=NOISE_BASE_SEED + 100 * cell,
                                   scalar_summaries_match_previous_draw0=True))
    if state_fingerprint(model) != C_SHA or sha(checkpoint_path) != checkpoint_sha or any(p.grad is not None for p in model.parameters()):
        raise AssertionError("C or checkpoint changed")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if not font.exists():
        raise FileNotFoundError("Chinese font unavailable")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font)).get_name(), "axes.unicode_minus":False})
    limit = max(4.5, math.ceil(max(np.max(np.abs(field)) / scale for row in records for field in row["fields"]) * 2) / 2)
    args.output.mkdir(parents=True, exist_ok=False)
    public = []
    for record in records:
        target = args.output / f"C_kappa_{record['kappa']:.2f}.png"
        draw_single(plt, record, scale, limit, target)
        public.append(dict(path=target.name, sha256=sha(target), bytes=target.stat().st_size))
    target = args.output / "C_kappa_071_086_overview.png"
    draw_overview(plt, records, scale, limit, target)
    public.append(dict(path=target.name, sha256=sha(target), bytes=target.stat().st_size))
    receipt = dict(status="complete", workflow_complete=True, diagnostic="final_C_high_kappa_montage_v1",
        checkpoint_step=3000, C_sha256=C_SHA, C_unchanged=True, checkpoint_sha256=checkpoint_sha,
        data_manifest_sha256=DATA_SHA, previous_diagnostic_sha256=PRIOR_JSON_SHA,
        previous_scalar_array_sha256=PRIOR_ARRAY_SHA, all_scalar_summaries_match_previous_draw0=True,
        device="cpu", cpu_threads=1, physical_sampling_or_training=False, Q_or_R_inference=False,
        exported_raw_fields_or_checkpoint=False, physical_fields_replayed=384, displayed_physical_fields=48,
        full_figures_selection="all4chains x saved0,64; fixed before viewing", overview_selection="chains0,2 x saved64; fixed before viewing",
        scales=[36,12,4], scale=scale, noise_std=float(model.noise_std), noise_seed_base=NOISE_BASE_SEED,
        global_sign_alignment=False, color_scale_over_s=[-limit,limit], color_clipping=False,
        conditions=provenance, cpu_seconds=time.process_time()-started, public_files=public,
        qualifications=["Only the two trained coarse-graining steps; no deeper extrapolation",
                        "A snapshot display is not thermalization, critical point or physical RG certification",
                        "Formal sampling quality warnings remain; no data repairs or additional physical sampling"])
    target = args.output / "receipt.json"
    target.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(status="complete", output=str(args.output), public_files=public,
                         receipt_sha256=sha(target), cpu_seconds=receipt["cpu_seconds"])), flush=True)


if __name__ == "__main__":
    main()
