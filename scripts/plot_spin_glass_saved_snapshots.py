"""Plot a tiny, provenance-bound selection of saved fixed-J configurations.

No sampler, model, checkpoint, gauge alignment, or phase-point estimator is used.
Only the separately extracted public NPZ is read; existing figures are never replaced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plot(directory, output=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.ft2font import FT2Font
    from matplotlib.text import Text
    from scripts.report_ising_zh import select_chinese_font

    directory = Path(directory).resolve()
    output = directory if output is None else Path(output).resolve()
    if output != directory:
        output.mkdir(parents=True, exist_ok=False)
    receipt_path, source = directory / "selection.json", directory / "selected_snapshots.npz"
    before = {path.name: sha(path) for path in (receipt_path, source)}
    receipt = json.loads(receipt_path.read_text())
    if (receipt["disorder_id"] != 0 or receipt["source_files_unchanged"] is not True
            or receipt["gauge_alignment"] is not False or receipt["new_physical_sampling"] != 0
            or before[source.name] != receipt["output_sha256"]):
        raise ValueError("Invalid fixed-J raw-configuration provenance")
    with np.load(source, allow_pickle=False) as z:
        spins, overlap = z["spins"], z["overlap_pair01"]
        temperatures = z["temperatures"]
        saved = z["saved_sample_indices"]
        if (spins.shape != (4, 3, 2, 16, 16, 16) or spins.dtype != np.int8
                or overlap.shape != (4, 3, 16, 16, 16)
                or not np.isin(spins, (-1, 1)).all()
                or not np.array_equal(overlap, spins[:, :, 0] * spins[:, :, 1])
                or not np.array_equal(saved, [0, 255, 511])
                or not np.array_equal(z["temperature_indices"], [0, 7, 8, 19])
                or not np.array_equal(temperatures, receipt["temperatures"])):
            raise ValueError("Selected raw spins, overlap, or indices differ from receipt")

    font = select_chinese_font()
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font)).get_name(),
                         "axes.unicode_minus": False})
    available = set(FT2Font(str(font)).get_charmap())
    cmap = ListedColormap(["#3c6fa7", "#d65d5d"])
    norm = BoundaryNorm([-2, 0, 2], cmap.N)
    tc = receipt["Tc_reference"]["value"]
    slice_z, primary = 8, 2  # Fixed before viewing: last saved time, ninth z plane.
    outputs = []

    def save(fig, name):
        target = output / name
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
        for text in fig.findobj(match=Text):
            missing = {c for c in text.get_text() if ord(c) > 127 and ord(c) not in available}
            if missing:
                raise ValueError(f"中文字体缺少字形：{missing}")
        fig.savefig(target, dpi=170)
        plt.close(fig)
        outputs.append(target.name)

    def draw(ax, field):
        # Arrays use [x,y,z]; transpose for horizontal x, vertical y.
        image = ax.imshow(field[:, :, slice_z].T, cmap=cmap, norm=norm,
                          interpolation="nearest", origin="lower")
        ax.set_xticks([0, 7, 15]); ax.set_yticks([0, 7, 15])
        return image

    def column_title(t):
        side = "低于" if t < tc else "高于"
        return f"T={t:.3f}\n{side}体相参考温度"

    fig, axes = plt.subplots(3, 4, figsize=(12, 9.6), layout="constrained")
    for col, t in enumerate(temperatures):
        fields = (spins[col, primary, 0], spins[col, primary, 1], overlap[col, primary])
        for row, field in enumerate(fields):
            image = draw(axes[row, col], field)
            if row == 0:
                axes[row, col].set_title(column_title(t))
            if col == 0:
                axes[row, col].set_ylabel(("原始副本 0", "原始副本 1", "重叠场 q₀₁")[row] + "\n位置 y")
            if row == 2:
                axes[row, col].set_xlabel("位置 x\n" + rf"整格 $|\overline{{q}}|={abs(overlap[col, primary].mean()):.3f}$")
    colorbar = fig.colorbar(image, ax=axes, ticks=[-1, 1], fraction=.025, pad=.02)
    colorbar.ax.set_yticklabels(["−1（蓝）", "+1（红）"])
    fig.suptitle("同一 J₀：原始自旋与重叠场\n第 512 个保存构型，z=8 切片；体相参考 Tc≈1.10，不是本样本测定值", fontsize=14)
    save(fig, "spin_and_overlap_slices.png")

    fig, axes = plt.subplots(3, 4, figsize=(12, 9.6), layout="constrained")
    for row, index in enumerate(saved):
        for col, t in enumerate(temperatures):
            image = draw(axes[row, col], overlap[col, row])
            if row == 0:
                axes[row, col].set_title(column_title(t))
            axes[row, col].set_xlabel(rf"整格 $|\overline{{q}}|={abs(overlap[col, row].mean()):.3f}$")
            if col == 0:
                axes[row, col].set_ylabel(f"第 {index + 1} 个保存构型\n位置 y")
    colorbar = fig.colorbar(image, ax=axes, ticks=[-1, 1], fraction=.025, pad=.02)
    colorbar.ax.set_yticklabels(["反向：q=−1", "同向：q=+1"])
    fig.suptitle("同一 J₀ 的重叠场：三个固定保存时刻，均取 z=8 切片\n原始符号保留；同一组相关轨迹，不是三条独立链", fontsize=14)
    save(fig, "overlap_three_saved_times.png")

    if before != {path.name: sha(path) for path in (receipt_path, source)}:
        raise RuntimeError("Read-only selected-input fingerprint changed")
    record = dict(source_selection_sha256=before[receipt_path.name], source_npz_sha256=before[source.name],
                  selected_inputs_unchanged=True, slice_axis="z", slice_index_zero_based=slice_z,
                  primary_saved_sample_index_zero_based=int(saved[primary]),
                  displayed_global_abs_overlap=np.abs(overlap.mean(axis=(2, 3, 4))).tolist(),
                  field_pixels_are_raw_signed=True, sign_alignment=False, sampling_updates=0,
                  model_inference=False, training_updates=0, font=str(font),
                  physical_finite_sample_Tc_measured=False,
                  figures={name: sha(output / name) for name in outputs})
    with (output / "view_receipt.json").open("x") as stream:
        json.dump(record, stream, indent=2, ensure_ascii=False, allow_nan=False)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(plot(args.directory, args.output), ensure_ascii=False))
