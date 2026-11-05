"""Array-only paired terminal-R report; no model or checkpoint loading."""
import json
from pathlib import Path
import numpy as np
from scripts.report_xy_qcr_g2 import render as render_geometry
from scripts.report_phi4_hopping_qcr import _setup_plot


def render(root):
    root=Path(root); report=render_geometry(root); out=root/'report'
    (out/'loss_curves.png').rename(out/'joint_loss_curves.png')
    summary=json.loads((root/'summary.json').read_text())
    history=json.loads((root/'r_history.json').read_text())
    plt=_setup_plot(None)
    fig,ax=plt.subplots(figsize=(7.5,4.5),layout='constrained')
    ax.semilogy([r['extra_R_updates'] for r in history],[r['train_loss'] for r in history],'o-',label='固定 TRAIN 缓存闭环损失')
    ax.set(xlabel='额外 R 更新次数（Q/C 零更新）',ylabel='全起点锚定闭环损失',title='固定 Q/C 后的 R-only 拟合')
    ax.legend();ax.grid(alpha=.2);fig.savefig(out/'loss_curves.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    paths=[(0,1),(1,2),(0,2)]
    for ax,metric,title in zip(axes,['rmse','nrmse'],['绝对 RMSE','同一 TRAIN 标准差归一化误差']):
        for offset,key,label in [(-.18,'before_closure','联合训练后'),(.18,'closure','R-only 后')]:
            rows=summary[key]['test']['all_start']; values=[next(r[metric] for r in rows if (r['source_scale'],r['target_scale'])==p) for p in paths]
            ax.bar(np.arange(3)+offset,values,width=.36,label=label)
        ax.set(xticks=np.arange(3),xticklabels=['54→18','18→6','54→6'],ylabel=title,title='留出温度：'+title)
        ax.legend();ax.grid(alpha=.15,axis='y')
    fig.savefig(out/'closure_comparison.png',dpi=170);plt.close(fig)
    report.update(joint_updates=1250,extra_R_updates=summary['extra_R_updates'],R_total_updates=summary['R_total_updates'],
        Q_C_unchanged=summary['Q_C_unchanged'],R_fit=summary['R_fit'],before_closure=summary['before_closure'],
        comparison='Same fixed Q/C and exact paired source evaluation encodings; only R changes',
        scalar_comparison_caveat='The scalar reference has only1250 R updates; it is not update-matched to this terminal fit',
        information_diagnostic_reused=True,loss_curves='R-only; original joint curve separately preserved')
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return report
