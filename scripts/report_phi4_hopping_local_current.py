"""Chinese array-only report of the explicit current local-Q phi4 trial."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from scripts.report_phi4_hopping_qcr import _setup_plot, _plot_loss, _plot_all_start_closure



def crossings(x, y, target=0.):
    """All piecewise-linear branches; never select by a theoretical value."""
    x, y = np.asarray(x), np.asarray(y)-target
    values = []
    for i in range(len(x)-1):
        if y[i] == y[i+1] == 0:
            continue  # A flat interval does not identify an isolated crossing.
        if y[i] == 0:
            values.append(float(x[i]))
        if y[i]*y[i+1] < 0:
            values.append(float(x[i]-y[i]*(x[i+1]-x[i])/(y[i+1]-y[i])))
    if len(x) and y[-1] == 0 and (len(x)==1 or y[-2] != 0):
        values.append(float(x[-1]))
    return sorted(set(values))


def analyze_arrays(a, roots):
    from scipy.stats import spearmanr
    direct = a['direct_g'][..., 0]
    cells = np.unique(a['cell_ids'])
    rows = sorted([(float(a['labels'][a['cell_ids']==c, 0][0]),
                    direct[a['cell_ids']==c].mean(0)) for c in cells], key=lambda row:row[0])
    kappa = np.array([r[0] for r in rows]); mean = np.array([r[1] for r in rows])
    root_rows = []
    for root in roots['roots']:
        root_rows.append(dict(**root, kappa_interpolation_by_scale=[
            crossings(kappa, mean[:, k], root['g']) for k in range(3)]))
    correlations = {}
    for split in ('train','test','all'):
        mask = a['splits']==split if split!='all' else np.ones(len(direct),dtype=bool)
        correlations[split] = []
        for k in range(3):
            rho = float(spearmanr(a['labels'][mask,0],direct[mask,k]).statistic)
            correlations[split].append(rho if np.isfinite(rho) else None)
    return dict(roots=root_rows, kappa_g_spearman=correlations,
                coupling_grid=kappa.tolist(), physical_chain_mean_g_by_scale=mean.tolist(),
                direct_mean_drift_zeros=[crossings(kappa,mean[:,k+1]-mean[:,k]) for k in range(2)],
                Q_full_vs_half_mean_rmse_by_scale=np.sqrt(np.mean(
                    (a['direct_full_g']-a['direct_g'])**2,axis=(0,2))).tolist(),
                thermal_nu=None, physical_critical_point_certified=False,
                interpolation_qualification='Piecewise linear physical-chain means; all branches reported; labels used only after training')


def render(root):
    root = Path(root)
    manifest = json.loads((root/"manifest.json").read_text())
    summary = json.loads((root/"summary.json").read_text())
    history = json.loads((root/"history.json").read_text())
    roots = json.loads((root/"roots.json").read_text())
    ratio = manifest["options"].get("r_updates_per_batch", 1)
    recipes = {"phi4_hopping_b3_local_Q_free_gain_unsupervised1250_seed91_v1": 1}
    if (recipes.get(manifest["recipe"]) != ratio
            or summary["actual_steps"] != manifest["options"]["steps"]
            or history[-1]["step"] != summary["actual_steps"]):
        raise ValueError("Wrong recipe or incomplete endpoint")
    for name in ("Q", "C", "R"):
        expected = summary["actual_steps"] * (ratio if name == "R" else 1)
        if any(summary["optimizer_steps"][name][key] != expected for key in ("min", "max")):
            raise ValueError("Optimizer clock mismatch")
    with np.load(root/"evaluation_arrays.npz", allow_pickle=False) as z:
        a = {k:z[k].copy() for k in z.files}
    if a["direct_g"].shape != (60,3,1) or manifest["labels_used_in_training"]:
        raise ValueError("Unexpected geometry or label policy")
    plt = _setup_plot(None)
    font = plt.rcParams["font.family"]
    out = root/"report"
    out.mkdir(exist_ok=False)
    if ratio == 1:
        _plot_loss(history, out, plt)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        for ax, name in zip(axes, ("q_total", "c_total", "r_total")):
            ax.plot([r["step"] for r in history], [r[name] for r in history], '.-', label="批次更新前")
            if name == "r_total":
                ax.plot([r["step"] for r in history], [r["R_batch_after"] for r in history], '.-', label="同批次 R 更新后")
                ax.set_yscale("symlog", linthresh=1e-6)
            ax.set(xlabel="Q/C 外层更新次数", ylabel="加权目标", title={"q_total":"Q 表示目标", "c_total":"C 粗粒化目标", "r_total":"R 闭环目标"}[name])
            ax.grid(alpha=.2); ax.legend(fontsize=8)
        fig.suptitle("每批 Q/C 各更新一次，R 更新十次；同批改善不等于留出泛化")
        fig.savefig(out/"loss_curves.png", dpi=160); plt.close(fig)
    _plot_all_start_closure(summary["closure"], out, plt)
    direct = a["direct_g"][...,0]
    train = a["splits"] == "train"
    grid, nxt = a["flow_grid"][:,0], a["flow_next"][:,0]
    low, high = direct[train,:-1].min(), direct[train,:-1].max()
    fig, axes = plt.subplots(1,2,figsize=(13,4.7),constrained_layout=True)
    axes[0].plot(grid,nxt,color="#ba5a9c",label="训练后的 R(g)")
    axes[0].plot(grid,grid,"k--",lw=1,label="恒等映射")
    for k in range(2):
        for split,marker,text in ((train,"o","训练"),(~train,"x","留出")):
            axes[0].scatter(direct[split,k],direct[split,k+1],s=22,marker=marker,
                            color=f"C{k}",label=f"k={k}→{k+1}，{text}")
    indices = np.arange(0,len(grid),100)
    axes[1].quiver(grid[indices],np.zeros(len(indices)),(nxt-grid)[indices],np.zeros(len(indices)),
                   angles="xy",scale_units="xy",scale=1,color=".5",width=.004)
    for k in range(2):
        for split,marker in ((train,"|"),(~train,"x")):
            axes[1].scatter(direct[split,k],np.full(split.sum(),-.12*(k+1)),marker=marker,
                            s=35,color=f"C{k}",label=f"k={k} 数据" if marker=="|" else None)
    for index,r in enumerate(roots["roots"]):
        for ax,y in ((axes[0],r["g"]),(axes[1],0)):
            ax.scatter([r["g"]],[y],marker="D",s=65,edgecolors="k",
                       facecolors="k" if r["inside_train_input_envelope"] else "white",zorder=5)
            kind = {"repeller":"排斥点","attractor":"吸引点","neutral":"近中性点"}[r["numerical_stability"]]
            if not r["inside_train_input_envelope"]: kind += "（外推）"
            ax.annotate(f"{index+1}：{kind}",(r["g"],y),xytext=(5,12),textcoords="offset points",fontsize=8)
    for ax in axes:
        ax.axvspan(grid.min(),low,color=".8",alpha=.4)
        ax.axvspan(high,grid.max(),color=".8",alpha=.4)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
        ax.set_xlabel("局域 Q 坐标 g")
    axes[0].set(ylabel="输出 g",title="共享映射与不同尺度的真实目标")
    axes[1].set(title="位移箭头 R(g)−g 与数据覆盖",yticks=[],ylim=(-.36,.1),
                xlim=(min(grid.min(),nxt[indices].min()),max(grid.max(),nxt[indices].max())))
    fig.suptitle(f"b=3，36→12→4；Q/C 各 {summary['actual_steps']} 步，R {summary['actual_steps']*ratio} 步\n灰色为训练输入范围之外；数值交点不等于物理临界点")
    fig.savefig(out/"R_flow.png",dpi=160);plt.close(fig)
    fig,ax = plt.subplots(figsize=(8,5),constrained_layout=True)
    cells=np.unique(a["cell_ids"])
    for k in range(3):
        means=[];std=[];couplings=[]
        for c in cells:
            m=a["cell_ids"]==c
            couplings.append(a["labels"][m,0][0]);means.append(direct[m,k].mean());std.append(direct[m,k].std(ddof=1))
        ax.plot(couplings,means,color=f"C{k}",label=f"k={k}，L={36//3**k}")
        ax.fill_between(couplings,np.array(means)-std,np.array(means)+std,color=f"C{k}",alpha=.1)
        for split,marker,text in ((train,"o","训练"),(~train,"x","留出")):
            ax.scatter(a["labels"][split,0],direct[split,k],marker=marker,color=f"C{k}",s=15,
                       label=text if k==0 else None)
    for r in roots["roots"]:
        ax.axhline(r["g"], color=".5", linestyle="--", lw=.8)
    ax.set(xlabel="耦合 κ（仅事后评价）",ylabel="直接编码 g",title="局域 Q 的各尺度编码；阴影为四链标准差")
    ax.legend();ax.grid(alpha=.2)
    fig.savefig(out/"Q_g_kappa.png",dpi=160);plt.close(fig)
    diagnostics = analyze_arrays(a, roots)
    report=dict(diagnostics=diagnostics, status="complete",font=str(font),actual_steps=summary["actual_steps"],roots=roots,
                optimizer_steps=summary["optimizer_steps"], r_updates_per_batch=ratio,
                closure=summary["closure"],thermal_nu=None,critical_kappa=None,
                limitations=["单种子1250次联合更新；不等同于历史 CNN 联合加 R-only 的训练量。",
                             "保留原物理采样的有限链质量警告，不证明热化。",
                             "局域 Q、自由增益仿射重构、R 学习率 .003；无监督且无追加 R-only。",
                             "TRAIN 包络内不等于每个位置都有充分数据支撑；未认证物理临界点。",
                             "连续 Gaussian C 及平滑信息代理不等同于二值经验直方图 MI。"])
    (out/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    return report
