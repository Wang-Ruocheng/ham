"""Model-free Chinese report of the scalar Ising CNN trial."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from scripts.report_phi4_hopping_qcr import _setup_plot,_plot_loss,_plot_all_start_closure


def crossings(x,y):
    result=[]
    for i in range(len(x)-1):
        if y[i]==0:result.append(float(x[i]))
        elif y[i]*y[i+1]<0:result.append(float(x[i]-y[i]*(x[i+1]-x[i])/(y[i+1]-y[i])))
    if y[-1]==0:result.append(float(x[-1]))
    return result


def render(root):
    root=Path(root)
    manifest=json.loads((root/'manifest.json').read_text());s=json.loads((root/'summary.json').read_text())
    h=json.loads((root/'history.json').read_text());r=json.loads((root/'roots.json').read_text())
    assert manifest['recipe'] in ('ising2d_B0_configuration_cnn_b3_g1_short_v1',
                                 'ising2d_B0_configuration_cnn_b3_g1_joint_refinedR_v1')
    assert s['actual_steps']==manifest['config']['training']['epochs']==h[-1]['step']
    ratio=manifest['config']['training'].get('r_updates_per_batch',1)
    assert all(c['min']==c['max']==s['actual_steps']*(ratio if name=='R' else 1)
               for name,c in s['optimizer_steps'].items())
    with np.load(root/'evaluation_arrays.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
    g=a['direct_g'][...,0];K=a['labels'][:,0];train=a['splits']=='train'
    assert g.shape==(192,3) and train.sum()==129
    knots=np.unique(K);means=np.array([g[K==v].mean(0) for v in knots]);stds=np.array([g[K==v].std(0,ddof=1) for v in knots])
    for entry in r['roots']:
        entry['K_interpolation_by_scale']=[crossings(knots,means[:,k]-entry['g']) for k in range(3)]
    plt=_setup_plot(None);out=root/'report';out.mkdir(exist_ok=False)
    if ratio==1:
        _plot_loss(h,out,plt)
    else:
        fig,axes=plt.subplots(1,3,figsize=(13,3.8),constrained_layout=True)
        for ax,name,label in zip(axes,['q','c','r'],['Q 表征目标','C 信息与保色目标','R 闭环目标']):
            ax.plot([v['step'] for v in h],[v[name+'_total'] for v in h],'.-',label='本轮更新前')
            if name=='r':
                ax.plot([v['step'] for v in h],[v['r_batch_after'] for v in h],'.-',label='R 内循环后（同批目标）')
            ax.set(xlabel='外层批次数（Q/C 更新数）',ylabel='目标值',title=label)
            ax.legend(fontsize=8);ax.grid(alpha=.2)
        fig.suptitle(f'每轮 Q/C 各 1 次、R 共 {ratio} 次；同批误差不是留出误差')
        fig.savefig(out/'loss_curves.png',dpi=160);plt.close(fig)
    _plot_all_start_closure(s['closure'],out,plt)
    grid=a['flow_grid'][:,0];nxt=a['flow_next'][:,0]
    lo,hi=g[train,:-1].min(),g[train,:-1].max()
    fig,axes=plt.subplots(1,2,figsize=(13,4.7),constrained_layout=True)
    axes[0].plot(grid,nxt,label='训练后的 R(g)',color='C4');axes[0].plot(grid,grid,'k--',lw=1,label='恒等映射')
    ids=np.arange(0,len(grid),100)
    axes[1].quiver(grid[ids],np.zeros(len(ids)),(nxt-grid)[ids],np.zeros(len(ids)),angles='xy',scale_units='xy',scale=1,color='C4',width=.004)
    for k in range(2):
        for mask,marker,text in [(train,'o','训练'),(~train,'x','留出')]:
            axes[0].scatter(g[mask,k],g[mask,k+1],s=15,marker=marker,color=f'C{k}',label=f'k={k}→{k+1}，{text}')
            axes[1].scatter(g[mask,k],np.full(mask.sum(),-.12*(k+1)),s=20,marker='|' if text=='训练' else 'x',color=f'C{k}')
    for i,v in enumerate(r['roots']):
        name={'repeller':'排斥点','attractor':'吸引点','neutral':'近中性点'}[v['numerical_stability']]
        for ax,y in [(axes[0],v['g']),(axes[1],0)]:
            ax.scatter([v['g']],[y],marker='D',s=50,edgecolors='k',facecolors='k' if v['inside_train_input_envelope'] else 'white')
            ax.annotate(f'{i+1}：{name}',(v['g'],y),xytext=(3,10),textcoords='offset points',fontsize=8)
    for ax in axes:
        ax.axvspan(grid.min(),lo,color='.8',alpha=.4);ax.axvspan(hi,grid.max(),color='.8',alpha=.4)
        ax.set_xlabel('CNN-Q 坐标 g');ax.grid(alpha=.2)
    axes[0].set(ylabel='输出 g',title='共享映射与直接编码目标');axes[0].legend(fontsize=7)
    axes[1].set(title='位移箭头 R(g)−g；下方为 k=0、1 数据覆盖',yticks=[],ylim=(-.35,.1),xlim=(min(grid.min(),nxt[ids].min()),max(grid.max(),nxt[ids].max())))
    clock_text=(f'{s["actual_steps"]} 步' if ratio==1 else
                f'Q/C 各 {s["actual_steps"]} 次，R {s["actual_steps"]*ratio} 次')
    fig.suptitle(f'二维 Ising，B=0，b=3，36→12→4；{clock_text}\n灰色为训练输入包络之外；数值交点不等于已认证物理临界点')
    fig.savefig(out/'R_flow.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)
    for k in range(3):
        ax.plot(knots,means[:,k],color=f'C{k}',label=f'k={k}，L={36//3**k}')
        ax.fill_between(knots,means[:,k]-stds[:,k],means[:,k]+stds[:,k],color=f'C{k}',alpha=.12)
        for mask,marker,text in [(train,'o','训练'),(~train,'x','留出')]:
            ax.scatter(K[mask],g[mask,k],s=12,marker=marker,color=f'C{k}',label=text if k==0 else None)
    ax.set(xlabel='耦合 K（仅事后评价）',ylabel='直接编码 g',title='逐构型 CNN 各尺度编码；阴影为三条物理链标准差');ax.legend();ax.grid(alpha=.2)
    fig.savefig(out/'Q_g_K.png',dpi=160);plt.close(fig)
    report=dict(status='complete',actual_steps=s['actual_steps'],roots=r,closure=s['closure'],
        optimizer_steps=s['optimizer_steps'],r_updates_per_batch=ratio,
        K_g_spearman=[float(spearmanr(K,g[:,k]).statistic) for k in range(3)],
        direct_scale_mean_drift_zeros=[crossings(knots,means[:,k+1]-means[:,k]) for k in range(2)],
        thermal_nu=None,physical_critical_K_certified=False,
        limitations=['单seed短预算；不是原历史长训练的等训练量复现。','三条物理链的离散性不包含训练seed、有限尺寸和采样偏差。',
                     '已保存构型仍有链内相关，不认证热化。','训练输入包络内不代表处处有充分数据支撑。',
                     '所有K反插值分支均保留，不按接近已知临界点挑选。'])
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return report
