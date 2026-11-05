"""Model-free Chinese report; only read the saved JSON/NPZ, no new inference."""
from pathlib import Path
import json
import numpy as np

from scripts.report_phi4_hopping_qcr import _setup_plot, _plot_loss, _plot_all_start_closure

RECIPE='phi4_ru_b3_cnn_g2_Rlr003_v1'


def render(root):
    root=Path(root)
    manifest,summary,history,root_data,probes=[json.loads((root/(name+'.json')).read_text())
        for name in ('manifest','summary','history','roots','train_probes')]
    from experiments.phi4_ru_cnn_saddle import RECIPE as SADDLE_RECIPE
    from experiments.phi4_ru_cnn_nonlinear import RECIPE as NONLINEAR_RECIPE
    from experiments.phi4_ru_cnn_local_window import RECIPE as LOCAL_RECIPE
    from experiments.phi4_kappa_h_cnn import RECIPE as KH_RECIPE
    recipe = manifest['recipe']
    kappa_h = recipe == KH_RECIPE
    expected_labels=['kappa','h','quartic_lambda'] if kappa_h else ['r','u','h']
    if (recipe not in (RECIPE,SADDLE_RECIPE,NONLINEAR_RECIPE,LOCAL_RECIPE,KH_RECIPE) or summary['recipe']!=recipe or summary['status']!='complete'
            or summary['actual_steps']!=manifest['steps'] or history[-1]['step']!=manifest['steps']
            or manifest['labels_used_in_training'] or manifest['evaluation_label_columns']!=expected_labels):
        raise ValueError('Wrong recipe, endpoint, or label policy')
    for clock in summary['optimizer_steps'].values():
        if clock['min']!=manifest['steps'] or clock['max']!=manifest['steps']:
            raise ValueError('Adam clock mismatch')
    with np.load(root/'evaluation_arrays.npz',allow_pickle=False) as z:
        a={k:z[k] for k in z.files}
    if a['direct_g'].shape!=(180,3,2):
        raise ValueError('Expected the complete ru grid, all chains and scales')
    plt=_setup_plot(None)
    out=root/'report';out.mkdir(exist_ok=False)
    _plot_loss(history,out,plt)
    _plot_all_start_closure(summary['closure'],out,plt)
    train=a['splits']=='train'
    grid,drift=a['flow_g'],a['flow_displacement']
    # Scalar positive rescaling preserves directions in the original g axes.
    length=np.linalg.norm(drift,axis=1,keepdims=True)
    display=drift/np.maximum(length,1e-15)*.65*np.min(np.ptp(grid,axis=0))/32
    fig,ax=plt.subplots(figsize=(8,7),constrained_layout=True)
    ax.quiver(grid[:,0],grid[:,1],display[:,0],display[:,1],angles='xy',scale_units='xy',
              scale=1,width=.0022,color='#52606d',alpha=.65,zorder=1)
    for k in range(3):
        for mask,marker,text in ((train,'o','训练'),(~train,'x','留出')):
            p=a['direct_g'][mask,k]
            ax.scatter(p[:,0],p[:,1],s=17,marker=marker,color=f'C{k}',alpha=.7,
                       label=f'{text}数据 k={k}',zorder=3)
    names={'attracting':'吸引点','repelling':'排斥点','saddle':'鞍点','neutral_or_unresolved':'近中性或未定'}
    for i,r in enumerate(root_data['roots'],1):
        x,y=r['g'];inside=r['within_train_hull']
        ax.scatter([x],[y],marker='D',s=75,edgecolors='black',facecolors='black' if inside else 'white',zorder=6)
        ax.annotate(f"{i}：{names[r['classification']]}",(x,y),xytext=(5,8),textcoords='offset points',fontsize=9)
    bounds=a['flow_bounds']
    ax.set(xlabel='$g_1$',ylabel='$g_2$',xlim=bounds[0],ylim=bounds[1],aspect='equal',
           title='共享 R 流场及数据覆盖\n箭头为 R(g)−g 的方向（等长显示）；覆盖外为外推')
    ax.legend(fontsize=8,loc='best');ax.grid(alpha=.15)
    fig.savefig(out/'R_flow.png',dpi=170);plt.close(fig)
    cells=np.unique(a['cell_ids'])
    labels=np.array([a['labels'][a['cell_ids']==c][0,:2] for c in cells])
    means=np.array([a['direct_g'][a['cell_ids']==c].mean(0) for c in cells])
    cell_test=np.array([a['splits'][a['cell_ids']==c][0]=='test' for c in cells])
    fig,axes=plt.subplots(2,3,figsize=(13,7.5),constrained_layout=True)
    for d in range(2):
        low,high=means[...,d].min(),means[...,d].max()
        for k in range(3):
            ax=axes[d,k]
            sc=ax.scatter(labels[:,0],labels[:,1],c=means[:,k,d],cmap='coolwarm',
                          vmin=low,vmax=high,s=120,marker='s')
            ax.scatter(labels[cell_test,0],labels[cell_test,1],marker='x',c='black',s=25,label='留出参数点')
            ax.set(xlabel='近邻耦合 κ' if kappa_h else '质量参数 r',
                   ylabel='外场 h' if kappa_h else '四次耦合 u',title=f'$g_{d+1}$：k={k}，L={36//3**k}')
            fig.colorbar(sc,ax=ax,label='四条物理链平均编码')
            if d==0 and k==0:ax.legend(fontsize=8)
    fig.suptitle('耦合到编码的映射；κ、h 仅用于事后评价，不进入训练' if kappa_h else
                 '耦合到编码的映射；r、u 仅用于事后评价，不进入训练')
    fig.savefig(out/('Q_g_kappa_h.png' if kappa_h else 'Q_g_ru.png'),dpi=160);plt.close(fig)
    ratio=np.asarray(root_data['flow_analysis']['pca_explained_variance_ratio'])
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    axes[0].bar(['主成分1','主成分2'],ratio)
    axes[0].set(ylabel='方差占比',title='训练流场输入 k=0,1 的主成分')
    for i,value in enumerate(ratio):axes[0].text(i,value,f'{value:.2%}',ha='center',va='bottom')
    axes[0].set_ylim(0,1.1)
    axes[1].plot([p['step'] for p in probes],[p['weighted_closure'] for p in probes],'.-')
    axes[1].set(xlabel='联合更新次数',ylabel='加权闭环损失',title='固定训练面板，固定 C 随机流；不用于选模',yscale='log')
    axes[1].grid(alpha=.2)
    fig.savefig(out/'PCA_and_train_probe.png',dpi=160);plt.close(fig)
    report=dict(status='complete',recipe=recipe,actual_steps=summary['actual_steps'],
        closure=summary['closure'],roots=root_data['roots'],thermal_nu=None,
        data_warning_count=manifest['data_warning_count'],pca_explained_variance_ratio=ratio.tolist(),
        qualifications=['单 seed、有限尺寸和有限基；不把数值不动点称为已认证物理临界点。',
            '保留原采样质量警告，未证明热化；未根据耦合标签或留出指标选择模型。',
            '连续 C 使用高斯核和平滑信息代理，不等同于二值经验互信息。',
            'PCA 方差占比不等于物理耦合空间维数；凸包内也不意味着充分局部支撑。',
            '训练每系综64构型，终末评价用全部128；两半编码均值与完整系综编码分别保留。',
            ('本轮扫描 κ/h、固定 λ=1；非零外场簇提议含MH校正；沿用CNN/非线性重构和6000步联合训练，无额外loss或R-only；热磁两相关方向，不预设鞍点。'
             if kappa_h else
             '本轮只更换 Q 的共享非线性重构头；复用已冻结的同一批物理数据，Q/C/R 全部重新初始化、各6000步；无方向均衡。独立训练的潜坐标不可直接混比。'
             if recipe==NONLINEAR_RECIPE else
             '本轮仅扩大 r/u 扫描范围；CNN、6000步及学习率沿用上一轮，独立初始化；历史坐标不可直接混比。'
             if recipe==SADDLE_RECIPE else '新增 CNN 与训练长度使本轮不构成学习率单因素因果消融；历史坐标不可直接混比。')])
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return report
