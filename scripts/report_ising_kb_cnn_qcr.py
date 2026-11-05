"""Chinese KB report from frozen JSON/NPZ only; never loads a checkpoint."""
import json
from pathlib import Path

import numpy as np

from scripts.report_phi4_hopping_qcr import _setup_plot, _plot_all_start_closure

RECIPE = 'ising2d_KB_cnn_b3_g2_Rlr003_seed91_v1'


def plot_losses(rows, out, plt):
    fig, axes = plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    groups=[(('q_total','Q 总损失'),('q_reconstruction','局域图案重构'),
             ('q_subset_micro','微观子集一致性'),('q_subset_coarse','粗尺度子集一致性')),
            (('c_total','C 总损失'),('c_information','经验互信息'),
             ('c_variance','逐系综方差下限'),('c_uniform','均匀块保色负对数似然')),
            (('r_total','R 总损失'),('closure_one_step','单步闭合诊断'),
             ('closure_rollout','微观起点多步闭合诊断'))]
    for ax,group,title in zip(axes,groups,('表示目标 Q','粗粒化目标 C','闭环目标 R')):
        for key,label in group:
            points=[(row['step'],row[key]) for row in rows if key in row and np.isfinite(row[key])]
            if points: ax.plot(*np.asarray(points).T,label=label)
        ax.set(title=title,xlabel='更新次数',ylabel='损失或诊断量',yscale='symlog')
        ax.set_yscale('symlog',linthresh=1e-5);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(out/'loss_curves.png',dpi=150);plt.close(fig)


def render(root):
    root = Path(root)
    manifest, summary, history, root_data, probes = [json.loads((root/(n+'.json')).read_text())
        for n in ('manifest', 'summary', 'history', 'roots', 'train_probes')]
    if (manifest['recipe'] != RECIPE or summary['recipe'] != RECIPE
            or summary['status'] != 'complete' or summary['actual_steps'] != 6000
            or history[-1]['step'] != 6000 or manifest['contract']['steps'] != 6000
            or manifest['labels_used_in_training'] or manifest['test_used_in_training']
            or manifest['evaluation_label_columns'] != ['K', 'B']):
        raise ValueError('Wrong recipe, endpoint, or label policy')
    if set(summary['optimizer_steps']) != {'Q','C','R'}:
        raise ValueError('Missing Adam clocks')
    for clock in summary['optimizer_steps'].values():
        if clock['min'] != 6000 or clock['max'] != 6000:
            raise ValueError('Adam clock mismatch')
    with np.load(root/'evaluation_arrays.npz', allow_pickle=False) as z:
        a = {k:z[k] for k in z.files}
    cells = np.unique(a['cell_ids'])
    if (a['direct_g'].shape != (714,3,2) or a['labels'].shape != (714,2)
            or len(cells) != 357 or np.sum(a['splits']=='train') != 474
            or np.sum(a['splits']=='test') != 240):
        raise ValueError('Expected all KB cells, physical replicas and scales')
    for c in cells:
        selected = a['cell_ids']==c
        if (selected.sum()!=2 or len(np.unique(a['labels'][selected],axis=0))!=1
                or len(np.unique(a['splits'][selected]))!=1):
            raise ValueError('Cell aggregation crossed labels or split')
    plt = _setup_plot(None)
    out = root/'report'; out.mkdir(exist_ok=False)
    plot_losses(history,out,plt)
    _plot_all_start_closure(summary['closure'],out,plt)
    train = a['splits']=='train'
    grid, drift = a['flow_g'], a['flow_displacement']
    length = np.linalg.norm(drift,axis=1,keepdims=True)
    display = drift/np.maximum(length,1e-15)*.65*np.min(np.ptp(grid,axis=0))/32
    fig,ax = plt.subplots(figsize=(8,7),constrained_layout=True)
    ax.quiver(grid[:,0],grid[:,1],display[:,0],display[:,1],angles='xy',scale_units='xy',
              scale=1,width=.0022,color='#52606d',alpha=.65,zorder=1)
    for k in range(3):
        for mask,marker,text in ((train,'o','训练'),(~train,'x','留出')):
            p=a['direct_g'][mask,k]
            ax.scatter(p[:,0],p[:,1],s=12,marker=marker,color=f'C{k}',alpha=.55,
                       label=f'{text}数据 k={k}',zorder=3)
    names={'attracting':'吸引点','repelling':'排斥点','saddle':'鞍点','neutral_or_unresolved':'近中性或未定'}
    for i,r in enumerate(root_data['roots'],1):
        x,y=r['g']; inside=r['within_train_hull']
        ax.scatter([x],[y],marker='D',s=75,edgecolors='black',facecolors='black' if inside else 'white',zorder=6)
        ax.annotate(f"{i}：{names[r['classification']]}",(x,y),xytext=(5,8),textcoords='offset points',fontsize=9)
    bounds=a['flow_bounds']
    ax.set(xlabel='$g_1$',ylabel='$g_2$',xlim=bounds[0],ylim=bounds[1],aspect='equal',
           title='共享 R 流场及数据覆盖\n箭头为 R(g)−g 的方向（等长显示）；覆盖外为外推')
    ax.legend(fontsize=8,loc='best'); ax.grid(alpha=.15)
    fig.savefig(out/'R_flow.png',dpi=170); plt.close(fig)
    labels=np.array([a['labels'][a['cell_ids']==c][0] for c in cells])
    means=np.array([a['direct_g'][a['cell_ids']==c].mean(0) for c in cells])
    cell_test=np.array([a['splits'][a['cell_ids']==c][0]=='test' for c in cells])
    fig,axes=plt.subplots(2,3,figsize=(13,7.5),constrained_layout=True)
    for d in range(2):
        low,high=means[...,d].min(),means[...,d].max()
        for k in range(3):
            ax=axes[d,k]
            sc=ax.scatter(labels[:,0],labels[:,1],c=means[:,k,d],cmap='coolwarm',
                          vmin=low,vmax=high,s=35,marker='s')
            ax.scatter(labels[cell_test,0],labels[cell_test,1],marker='x',c='black',s=9,label='留出参数点')
            ax.set(xlabel='最近邻耦合 K',ylabel='外场 B',title=f'$g_{d+1}$：k={k}，L={36//3**k}')
            fig.colorbar(sc,ax=ax,label='两条物理链平均编码')
            if d==0 and k==0: ax.legend(fontsize=8)
    fig.suptitle('耦合到编码的映射；K、B 仅用于事后评价，不进入训练')
    fig.savefig(out/'Q_g_KB.png',dpi=160); plt.close(fig)
    ratio=np.asarray(root_data['flow_analysis']['pca_explained_variance_ratio'])
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    axes[0].bar(['主成分1','主成分2'],ratio)
    axes[0].set(ylabel='方差占比',title='训练流场输入 k=0,1 的主成分',ylim=(0,1.1))
    for i,value in enumerate(ratio): axes[0].text(i,value,f'{value:.2%}',ha='center',va='bottom')
    axes[1].plot([p['step'] for p in probes],[p['weighted_closure'] for p in probes],'.-')
    axes[1].set(xlabel='联合更新次数',ylabel='加权闭环损失',title='固定训练面板，固定 C 随机流；不用于选模',yscale='log')
    axes[1].grid(alpha=.2)
    fig.savefig(out/'PCA_and_train_probe.png',dpi=160); plt.close(fig)
    report=dict(status='complete',recipe=RECIPE,actual_steps=6000,closure=summary['closure'],
        roots=root_data['roots'],thermal_nu=None,pca_explained_variance_ratio=ratio.tolist(),
        final_MI=summary['final_MI'],equilibrium_certified=False,
        qualifications=['单 seed、有限尺寸和有限表示；不把数值不动点称为已认证物理临界点。',
            '复用历史采样：每参数点仅两条物理链、每链64构型，未新增热化认证。',
            'C 使用真实二值经验直方图互信息，不使用连续场高斯代理。',
            '未根据 K/B 标签、留出指标或不动点结果选择训练终点。',
            'PCA 方差占比不等于物理耦合空间维数；凸包内也不等于充分局部支撑。',
            '两半编码平均与完整系综编码分别保留；固定训练面板损失仍受 Q/C 坐标漂移影响。',
            '沿用当前 CNN 优化协议；与连续场运行的数据规模和观测模型不同，不是严格单因素对照。'])
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return report
