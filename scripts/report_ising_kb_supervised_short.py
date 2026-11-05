"""Model-free Chinese report for the explicit supervised KB short experiment."""
import json
from pathlib import Path
import numpy as np
from scripts.report_phi4_hopping_qcr import _setup_plot, _plot_all_start_closure
from scripts.report_ising_kb_cnn_qcr import plot_losses
from experiments.ising_kb_supervised_short import RECIPE


def render(root):
    root=Path(root)
    m,s,h,r,p=[json.loads((root/(n+'.json')).read_text()) for n in
        ('manifest','summary','history','roots','train_probes')]
    steps=m['actual_planned_steps']
    if (m['recipe']!=RECIPE or s['recipe']!=RECIPE or s['status']!='complete'
            or s['actual_steps']!=steps or h[-1]['step']!=steps
            or not m['labels_used_in_training'] or m['test_used_in_training']
            or m['q_supervision']['scale']!=0 or m['q_supervision']['coarse_labels']):
        raise ValueError('Wrong supervised experiment/endpoint/label policy')
    if set(s['optimizer_steps'])!={'Q','C','R'}:raise ValueError('Missing optimizer clocks')
    if any(x['min']!=steps or x['max']!=steps for x in s['optimizer_steps'].values()):
        raise ValueError('Adam clock mismatch')
    with np.load(root/'evaluation_arrays.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
    if a['direct_g'].shape!=(714,3,2) or a['labels'].shape!=(714,2):
        raise ValueError('Expected all714 chains and both coupling labels')
    cells=np.unique(a['cell_ids']);train=a['splits']=='train'
    if len(cells)!=357 or train.sum()!=474:raise ValueError('Wrong cell/split count')
    for c in cells:
        mask=a['cell_ids']==c
        if mask.sum()!=2 or len(np.unique(a['labels'][mask],axis=0))!=1 or len(np.unique(a['splits'][mask]))!=1:
            raise ValueError('Cell aggregation crossed parameters or split')
    plt=_setup_plot(None);out=root/'report';out.mkdir(exist_ok=False)
    plot_losses(h,out,plt);_plot_all_start_closure(s['closure'],out,plt)
    grid=a['flow_g'];drift=a['flow_displacement']
    display=drift/np.maximum(np.linalg.norm(drift,axis=1,keepdims=True),1e-15)*.65*np.min(np.ptp(grid,axis=0))/32
    fig,ax=plt.subplots(figsize=(8,7),constrained_layout=True)
    ax.quiver(grid[:,0],grid[:,1],display[:,0],display[:,1],angles='xy',scale_units='xy',scale=1,width=.0022,color='#52606d',alpha=.65)
    for k in range(3):
        for mask,marker,name in ((train,'o','训练'),(~train,'x','留出')):
            points=a['direct_g'][mask,k]
            ax.scatter(*points.T,s=12,marker=marker,color=f'C{k}',alpha=.55,label=f'{name} k={k}')
    names=dict(attracting='吸引点',repelling='排斥点',saddle='鞍点',neutral_or_unresolved='近中性或未定')
    for i,v in enumerate(r['roots'],1):
        ax.scatter(*v['g'],marker='D',s=75,edgecolors='black',facecolors='black' if v['within_train_hull'] else 'white',zorder=5)
        ax.annotate(f"{i}：{names[v['classification']]}",v['g'],xytext=(5,8),textcoords='offset points')
    ax.set(xlabel='$g_1$',ylabel='$g_2$',xlim=a['flow_bounds'][0],ylim=a['flow_bounds'][1],aspect='equal',
        title='有监督 Q 的共享 R 流场与数据覆盖\n箭头为 R(g)−g 的方向（等长）；覆盖外为外推')
    ax.legend(fontsize=8);ax.grid(alpha=.15);fig.savefig(out/'R_flow.png',dpi=170);plt.close(fig)
    labels=np.array([a['labels'][a['cell_ids']==c][0] for c in cells])
    means=np.array([a['direct_g'][a['cell_ids']==c].mean(0) for c in cells])
    testcells=np.array([a['splits'][a['cell_ids']==c][0]=='test' for c in cells])
    fig,axes=plt.subplots(2,3,figsize=(13,7.5),constrained_layout=True)
    for d in range(2):
        for k in range(3):
            ax=axes[d,k];sc=ax.scatter(*labels.T,c=means[:,k,d],cmap='coolwarm',s=35,marker='s',vmin=means[...,d].min(),vmax=means[...,d].max())
            ax.scatter(*labels[testcells].T,marker='x',c='black',s=9)
            ax.set(xlabel='最近邻耦合 K',ylabel='外场 B',title=f'$g_{d+1}$：k={k}，L={36//3**k}')
            fig.colorbar(sc,ax=ax,label='两条物理链平均编码')
    fig.suptitle('仅 TRAIN 微观 Q 使用标准化 K/B 标签；粗尺度没有标签，叉号为留出参数点')
    fig.savefig(out/'Q_g_KB.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    mu=np.asarray(m['label_scaler']['mean']);std=np.asarray(m['label_scaler']['std'])
    predictions=a['direct_full_g'][:,0]*std+mu
    for d,name in enumerate(('K','B')):
        for mask,marker,text in ((train,'o','训练'),(~train,'x','留出')):
            axes[d].scatter(a['labels'][mask,d],predictions[mask,d],s=12,marker=marker,alpha=.5,label=text)
        limits=[a['labels'][:,d].min(),a['labels'][:,d].max()];axes[d].plot(limits,limits,'k--',lw=1)
        axes[d].set(xlabel=f'真实 {name}',ylabel=f'微观 Q 换算的 {name}',title=f'{name} 的监督拟合');axes[d].legend()
    axes[2].plot([v['step'] for v in h],[v['q_supervision'] for v in h],label='随机训练批次')
    axes[2].plot([v['step'] for v in p],[v['supervised_mse'] for v in p],'.-',label='固定训练面板')
    axes[2].set(xlabel='联合更新次数',ylabel='标准化监督均方误差',yscale='log',title='仅诊断，不用于选模');axes[2].legend()
    fig.savefig(out/'supervision_fit.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for k in range(3):
        g=a['direct_g'][train,k];v=np.linalg.svd(g-g.mean(0),compute_uv=False)**2;ratio=v/v.sum()
        axes[0].bar(np.arange(2)+.24*(k-1),ratio,width=.24,label=f'k={k}')
    axes[0].set(xticks=[0,1],xticklabels=['主成分1','主成分2'],ylabel='方差占比',title='逐尺度 TRAIN PCA，不混入尺度间偏移');axes[0].legend()
    axes[1].plot([v['step'] for v in p],[v['weighted_closure'] for v in p],'.-')
    axes[1].set(xlabel='联合更新次数',ylabel='加权闭环损失',yscale='log',title='固定训练面板；Q/C 仍会变化')
    fig.savefig(out/'PCA_and_train_probe.png',dpi=160);plt.close(fig)
    result=dict(status='complete',recipe=RECIPE,actual_steps=steps,roots=r['roots'],closure=s['closure'],supervision=s['supervision'],
        thermal_nu=None,qualifications=['这是有监督消融，不是无标签实验。',
        '监督只作用于 k=0 的 Q；原始耦合不能作为粗尺度的真实耦合标签。',
        '标准化只使用 TRAIN；TEST 不参与训练、终点选择或监督归一化。',
        '总时长受限，步数由训练前测速冻结；不同于历史6000步，不能直接归因监督优劣。',
        '单seed、有限尺寸、每点两链；未认证热化、不动点物理意义或临界指数。',
        '全系综编码用于监督评价，两半编码均值用于闭环，分别保留。'])
    (out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return result
