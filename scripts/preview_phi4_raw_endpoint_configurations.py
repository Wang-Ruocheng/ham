"""Render existing physical fields only, on their original host; export PNG/JSON.

No torch, model, optimizer, new physical samples, coarse graining, or raw-array
export. Execute this saved source over SSH stdin, outside frozen output roots.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np


ROOT = Path("/mnt/nas-new/home/yange/ham/results/phi4_hopping_b2_L128_g1_seed91_gpu3_vm225_20261101_integrator_refined_v1")
DATA_SHA = "3d2023b401b15e20a6dca6c6b602959c237a030ff0e2f8f89676a7ca9ed30fa5"
CELLS = ((0, .32), (14, 1.04))
SAVED = (32, 96)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def select_fields(values):
    if values.shape != (4, 128, 128, 128, 1) or values.dtype != np.float64:
        raise ValueError("Unexpected physical chain/sample/spatial axes or dtype")
    chosen = np.take(values[..., 0], SAVED, axis=1).copy()
    if chosen.shape != (4, 2, 128, 128) or not np.isfinite(chosen).all():
        raise ValueError("Invalid selected physical fields")
    return chosen


def moments(field):
    x = np.asarray(field)
    mean, rms = float(x.mean()), float(np.sqrt(np.mean(x**2)))
    return dict(mean=mean, rms=rms, abs_mean_over_rms=abs(mean) / rms if rms else 0.,
                positive_site_fraction=float(np.mean(x > 0)),
                majority_sign_fraction=float(max(np.mean(x > 0), np.mean(x < 0)) + .5 * np.mean(x == 0)))


def render(plt, fields, target, limit, *, sign_only=False):
    fig, axes = plt.subplots(4, 4, figsize=(12.2, 12.0))
    for condition, (cell, kappa) in enumerate(CELLS):
        for index, saved in enumerate(SAVED):
            row = condition * 2 + index
            for chain in range(4):
                field = fields[cell][chain, index]
                value = np.sign(field) if sign_only else field
                ax = axes[row, chain]
                picture = ax.imshow(value, cmap="RdBu_r", vmin=-limit, vmax=limit,
                                    interpolation="nearest", origin="upper")
                metric = moments(field)
                ax.set_title(f"物理链 {chain}，保存序号 {saved}\n"
                             + rf"$m={metric['mean']:+.3f}$，$|m|/\mathrm{{RMS}}={metric['abs_mean_over_rms']:.2f}$", fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                if chain == 0:
                    ax.set_ylabel(rf"$\kappa={kappa:.2f}$", fontsize=13)
    kind = "仅显示原始场的正负号" if sign_only else "原始连续场幅度"
    fig.suptitle(f"低、高耦合的既有采样构型：{kind}\n"
                 + r"实际 $\kappa=0.32$ 与 $1.04$；均为 $128\times128$，未粗粒化", fontsize=16)
    fig.subplots_adjust(left=.07, right=.88, top=.89, bottom=.10, hspace=.30, wspace=.08)
    cax = fig.add_axes((.91, .18, .02, .62))
    bar = fig.colorbar(picture, cax=cax, label="原始场符号" if sign_only else r"物理场值 $\phi$（所有构型共用色标）")
    if sign_only:
        bar.set_ticks([-1, 0, 1])
        bar.set_ticklabels(["负", "零", "正"])
    fig.text(.5, .062, "预先固定四条物理链、每链保存序号 32 和 96（从零计数）；不按观感选样、不整体翻转符号。", ha="center", fontsize=10)
    note = "符号图仅辅助观察，不是 Ising 数据、学习 C 的输出或新增采样。" if sign_only else "每个面板是一张原始构型；不是时间／系综平均图。无新增采样、模型推理或训练。"
    fig.text(.5, .036, note, ha="center", fontsize=10)
    fig.text(.5, .013, "请求值 0.3／1 附近使用已有最近采样点；少量构型展示不认证热化或精确相变位置，原有警告保留。", ha="center", fontsize=9.5)
    fig.savefig(target, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.process_time()
    output = args.output.resolve()
    if output.exists() or output == ROOT or ROOT in output.parents:
        raise ValueError("Require a new independent preview directory")
    if os.uname().nodename != "thinklab-105-225":
        raise RuntimeError("Unexpected host; no fallback")
    resources = dict(host=os.uname().nodename, cpus=os.cpu_count(), load1=os.getloadavg()[0],
        mem_available_bytes=int(Path("/proc/meminfo").read_text().split("MemAvailable:")[1].split()[0]) * 1024,
        io_full_avg10=float(Path("/proc/pressure/io").read_text().split("full avg10=")[1].split()[0]))
    if (resources["mem_available_bytes"] < 36 * 1024**3 or resources["load1"] > 1.25 * resources["cpus"]
            or resources["io_full_avg10"] >= 20 or os.getpriority(os.PRIO_PROCESS, 0) < 10):
        raise RuntimeError("Read-only CPU resource gate failed; no fallback")
    script_sha = os.environ.get("DIAGNOSTIC_SCRIPT_SHA256", "")
    if len(script_sha) != 64 or any(c not in "0123456789abcdef" for c in script_sha):
        raise ValueError("Require saved script SHA256")
    job, plan = read(ROOT / "job.json"), read(ROOT / "plan.json")
    if (job.get("status") != "complete" or job.get("workflow_complete") is not True
            or job.get("returncode") != 0 or job.get("error") or job.get("requires_user_action") is not False
            or job.get("plan_sha256") != sha256(ROOT / "plan.json")):
        raise ValueError("Frozen experiment is not a clean completed workflow")
    data = ROOT / "data"
    physical = read(data / "manifest.json")
    if (sha256(data / "manifest.json") != DATA_SHA or read(data / "status.json").get("status") != "complete"
            or physical.get("field_representation") != "phi"
            or any(physical["options"].get(k) != v for k, v in
                   (("sizes", [128]), ("quartic_lambda", 1.), ("h", 0.), ("chains", 4), ("samples", 128)))):
        raise ValueError("Frozen physical data binding differs")
    protected = {ROOT / name: sha256(ROOT / name) for name in ("job.json", "plan.json", "data/status.json", "data/manifest.json", "data/quality_summary.json")}
    fields, records = {}, []
    for cell, kappa in CELLS:
        shard = next(r for r in physical["shards"] if r["cell_index"] == cell)
        for name, expected in ((shard["path"], shard["sha256"]), (shard["sidecar"], shard["sidecar_sha256"])):
            path = (data / name).resolve()
            if data not in path.parents or sha256(path) != expected:
                raise ValueError("Selected physical shard/sidecar SHA differs")
            protected[path] = expected
        sidecar = read(data / shard["sidecar"])
        if (not math.isclose(sidecar["kappa"], kappa, rel_tol=0, abs_tol=1e-12)
                or sidecar.get("field_representation") != "phi" or sidecar.get("status") != "complete"
                or sidecar.get("artificial_global_sign_flips") != 0
                or sidecar.get("symmetry_augmentation") is not False):
            raise ValueError("Unexpected physical condition or postprocessing")
        with np.load(data / shard["path"], allow_pickle=False) as archive:
            fields[cell] = select_fields(archive["configurations"])
        for chain in range(4):
            for index, saved in enumerate(SAVED):
                records.append(dict(cell=cell, kappa=kappa, chain=chain, saved_index=saved,
                    initial=sidecar["chains"][chain]["initial"],
                    metrics=moments(fields[cell][chain, index]),
                    chain_warnings=sidecar["chains"][chain]["warnings"]))
    for path, expected in protected.items():
        if sha256(path) != expected:
            raise RuntimeError("Frozen input changed during preview")
    if time.process_time() - started > 60:
        raise RuntimeError("Read-only preview CPU budget exhausted")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if not font.is_file():
        raise FileNotFoundError("Chinese font unavailable")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font)).get_name(),
                         "axes.unicode_minus": False})
    limit = math.ceil(max(float(np.abs(x).max()) for x in fields.values()) * 2) / 2
    output.mkdir(parents=True, exist_ok=False)
    public = []
    for filename, sign_only in (("raw_fields.png", False), ("raw_signs.png", True)):
        target = output / filename
        render(plt, fields, target, 1. if sign_only else limit, sign_only=sign_only)
        public.append(dict(path=filename, sha256=sha256(target), bytes=target.stat().st_size))
    if time.process_time() - started > 60:
        raise RuntimeError("Read-only preview CPU budget exhausted")
    receipt = dict(status="complete", source_root=str(ROOT), data_manifest_sha256=DATA_SHA,
        script_sha256=script_sha, host_resources=resources, fields_displayed=16,
        actual_kappas=[.32, 1.04], requested_approximate_kappas=[.3, 1.],
        selection=dict(chains=[0, 1, 2, 3], saved_indices=list(SAVED), policy="predeclared; no visual or sign selection"),
        model_loaded=False, new_model_inference=False, new_sampling=False, new_training=False,
        global_sign_flip=False, shared_physical_phi_color_limit=limit, color_clipping=False,
        raw_arrays_or_checkpoints_exported=False, scalar_metrics=records,
        input_files_unchanged={str(p): d for p, d in protected.items()},
        data_warning_count=read(data / "quality_summary.json")["warning_count"],
        equilibrium_certified=False, cpu_seconds=time.process_time() - started, public_files=public)
    target = output / "receipt.json"
    with target.open("x") as stream:
        json.dump(receipt, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    print(json.dumps(dict(status="complete", output=str(output), public_files=public,
                         receipt_sha256=sha256(target), cpu_seconds=receipt["cpu_seconds"])))


if __name__ == "__main__":
    main()
