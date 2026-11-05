"""Model-free, paired Chinese report for the scalar Ising R-only continuation."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from scripts.report_phi4_hopping_qcr import _setup_plot
from scripts.report_ising_cnn_short_trial import crossings


def render(root, *, family='ising'):
    root=Path(root)
    summary=json.loads((root/'summary.json').read_text())
    manifest=json.loads((root/'manifest.json').read_text())
    history=json.loads((root/'history.json').read_text())
    roots=json.loads((root/'roots.json').read_text())
    if family not in ('ising', 'phi4'):
        raise ValueError('Unknown physical model')
    phi4 = family == 'phi4'
    assert manifest['recipe']==('phi4_hopping_b3_cnn_online_R_only_v1' if phi4 else 'ising2d_B0_cnn_b3_g1_online_R_only_v1')
    assert summary['frozen_before_sha256']==summary['frozen_after_sha256']
    assert summary['additional_Q_updates']==summary['additional_C_updates']==0
    with np.load(root/'evaluation_arrays.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
    g=a['direct_g'][...,0];K=a['labels'][:,0];train=a['splits']=='train'
    assert g.shape==((60,3) if phi4 else (192,3)) and train.sum()==(40 if phi4 else 129)
    knots=np.unique(K);means=np.array([g[K==v].mean(0) for v in knots]);stds=np.array([g[K==v].std(0,ddof=1) for v in knots])
    for entry in roots['roots']:
        entry['kappa_interpolation_by_scale' if phi4 else 'K_interpolation_by_scale']=[crossings(knots,means[:,k]-entry['g']) for k in range(3)]
    plt=_setup_plot(None);out=root/'report';out.mkdir(exist_ok=False)
    grid=a['flow_grid'][:,0];nxt=a['flow_next'][:,0];old=a['source_flow_next'][:,0]
    lo,hi=g[train,:-1].min(),g[train,:-1].max()
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),constrained_layout=True)
    axes[0].plot(grid,old,color='.55',ls='--',label='原联合训练 R')
    axes[0].plot(grid,nxt,color='C4',label='冻结 Q/C 后的 R-only')
    axes[0].plot(grid,grid,color='k',ls=':',lw=1,label='恒等映射')
    ids=np.arange(0,len(grid),100)
    for k in range(2):
        for mask,marker,text in [(train,'o','训练'),(~train,'x','留出')]:
            axes[0].scatter(g[mask,k],g[mask,k+1],s=13,marker=marker,color=f'C{k}',label=f'k={k}→{k+1}，{text}')
            axes[1].scatter(g[mask,k],np.full(mask.sum(),-.11*(k+1)),s=17,marker='|' if text=='训练' else 'x',color=f'C{k}')
    axes[1].quiver(grid[ids],np.zeros(len(ids)),(nxt-grid)[ids],np.zeros(len(ids)),angles='xy',scale_units='xy',scale=1,color='C4',width=.004)
    for i,v in enumerate(roots['roots']):
        name={'repeller':'排斥点','attractor':'吸引点','neutral':'近中性点'}[v['numerical_stability']]
        for ax,y in [(axes[0],v['g']),(axes[1],0)]:
            ax.scatter([v['g']],[y],marker='D',s=45,edgecolors='k',facecolors='k' if v['inside_train_input_envelope'] else 'white')
            ax.annotate(f'{i+1}：{name}',(v['g'],y),xytext=(3,10 if i%2==0 else -22),textcoords='offset points',fontsize=8)
    for ax in axes:
        ax.axvspan(grid.min(),lo,color='.8',alpha=.4);ax.axvspan(hi,grid.max(),color='.8',alpha=.4)
        ax.set_xlabel('固定 CNN-Q 坐标 g');ax.grid(alpha=.2)
    axes[0].set(ylabel='输出 g',title='相同 Q/C 目标，比较 R 拟合前后');axes[0].legend(fontsize=7)
    axes[1].set(title='R-only 位移箭头 R(g)−g；下方为数据覆盖',yticks=[],ylim=(-.31,.10),xlim=(min(grid.min(),nxt[ids].min()),max(grid.max(),nxt[ids].max())))
    description = ('二维 φ⁴，λ=1、h=0' if phi4 else '二维 Ising，B=0')
    source_steps = summary.get('source_joint_updates',850)
    fig.suptitle(f'{description}，36→12→4；源联合训练{source_steps}批＋R-only {summary["actual_extra_R_updates"]}次更新\n灰色为训练输入包络之外；不动点为数值诊断，未认证物理临界点',fontsize=11)
    fig.savefig(out/'R_comparison.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.8),constrained_layout=True)
    for k in range(3):
        ax.plot(knots,means[:,k],color=f'C{k}',label=f'k={k}，L={36//3**k}')
        ax.fill_between(knots,means[:,k]-stds[:,k],means[:,k]+stds[:,k],color=f'C{k}',alpha=.12)
        for mask,marker,text in [(train,'o','训练'),(~train,'x','留出')]:
            ax.scatter(K[mask],g[mask,k],s=10,marker=marker,color=f'C{k}',label=text if k==0 else None)
    coupling = 'κ' if phi4 else 'K'
    ax.set(xlabel=f'耦合 {coupling}（仅事后评价）',ylabel='直接编码 g',title=f'Q/C 完全冻结：各尺度 g({coupling}) 与原实验相同');ax.legend();ax.grid(alpha=.2)
    fig.savefig(out/('Q_g_kappa.png' if phi4 else 'Q_g_K.png'),dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
    steps=[r['extra_R_updates'] for r in history]
    axes[0].semilogy(steps,[r['train']['total'] for r in history],'o-',ms=3,label='TRAIN 固定面板闭合目标')
    axes[0].set(xlabel='额外 R 更新数',ylabel='全起点闭合目标',title='固定训练端点；不按留出误差选择' if phi4 else '只使用 TRAIN 判断平台');axes[0].legend(fontsize=8)
    labels=['0→1','0→2','1→2'];x=np.arange(3)
    for off,key,name in [(-.18,'before_closure','原联合训练'),(.18,'after_closure','R-only')]:
        axes[1].bar(x+off,[m['nrmse'] for m in summary[key]['test']['all_start']],width=.34,label=name)
    axes[1].set(xticks=x,xticklabels=labels,xlabel='起点尺度→目标尺度',ylabel='原 TRAIN 目标标准差归一化 RMSE',title='同一留出目标：拟合前后比较');axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(out/'loss_comparison.png',dpi=160);plt.close(fig)
    report=dict(status='complete',actual_extra_R_updates=summary['actual_extra_R_updates'],reason=summary['reason'],
        roots=roots,before_closure=summary['before_closure'],after_closure=summary['after_closure'],
        K_g_spearman=[float(spearmanr(K,g[:,k]).statistic) for k in range(3)],
        direct_scale_mean_drift_zeros=[crossings(knots,means[:,k+1]-means[:,k]) for k in range(2)],
        Q_C_unchanged=True,labels_used_in_training=False,thermal_nu=None,physical_critical_K_certified=False,
        limitations=['仅一个训练seed；数值不动点不是物理认证。',f'保留所有{coupling}反插值分支，不按接近理论值选择。',
        '固定6000次R更新，不做最优步数选择。' if phi4 else 'TRAIN平台不等于全局最优；留出集未用于停止。','固定C评价抽样、有限尺寸与链内相关误差仍存在。'])
    if phi4:
        report['kappa_g_spearman'] = report.pop('K_g_spearman')
        report['physical_critical_kappa_certified'] = report.pop('physical_critical_K_certified')
        report['source_R_updates'] = summary['source_R_updates']
        report['data_warning_count'] = summary['data_warning_count']
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return report
