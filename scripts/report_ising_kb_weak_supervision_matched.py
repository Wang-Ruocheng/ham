"""Frozen-array report comparing supervision weights 0, 1 and 10."""
import json
from pathlib import Path
import numpy as np
from scripts.report_phi4_hopping_qcr import _setup_plot, _plot_all_start_closure
from scripts.report_ising_kb_cnn_qcr import plot_losses
from scripts.plot_saved_kb_parameter_flow import cell_means, sha256
from experiments.ising_kb_weak_supervision_matched import RECIPE, STEPS, BASELINE, BASELINE_PINS

BASELINE_ARRAY_SHA='8b9251a4fa6a7766c6bf206eaab94dd14535c4c6bdf50f355753c2637a6d4580'


def read_arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def linear_readout(a):
    """Post-hoc TRAIN-only affine readout, not part of QCR training."""
    train=a['splits']=='train';g=a['direct_full_g'][:,0];y=a['labels']
    design=np.column_stack([g,np.ones(len(g))])
    coefficients=np.linalg.lstsq(design[train],y[train],rcond=None)[0]
    predicted=design@coefficients;result={}
    for name,mask in [('train',train),('test',~train)]:
        mse=np.mean((predicted[mask]-y[mask])**2,axis=0)
        result[name]=dict(rmse=np.sqrt(mse).tolist(),r2=(1-mse/np.var(y[mask],axis=0)).tolist())
    return result


def pca(a):
    result=[]
    for k in range(3):
        g=a['direct_g'][a['splits']=='train',k];v=np.linalg.svd(g-g.mean(0),compute_uv=False)**2
        result.append((v/v.sum()).tolist() if v.sum()>0 else [0.,0.])
    return result


ZERO_BASELINE='/mnt/nas-new/home/yange/ham/results/ising2d_KB_cnn_unsupervised_matched1250_seed91_gpu3_vm225_20261104_v1'
ZERO_PINS=dict(manifest='310fb015c0780c39eeae2d467c5cb3835edb3410a296e251d30bfcf82c0559e6',
               summary='752f47c8ca649d23d08f7ee3feafae99fcffd99379f60b5882b5c724345f74ee')
ZERO_ARRAY_SHA='3bae59361fbe2ca6e9e9febf504ad8f7c15480b7b3fb0541ebbf2a1c36138dab'


def arm_metrics(a,s):
    return dict(closure=s['closure'],final_MI=s['final_MI'],roots=s['numerical_fixed_points'],
        TRAIN_affine_readout=linear_readout(a),PCA_by_scale=pca(a),
        feature_MSE_by_scale={split:np.mean((a['reconstructed_features'][a['splits']==split]-
            a['target_features'][a['splits']==split])**2,axis=(0,2)).tolist() for split in ('train','test')})


def comparison(a,summary,baselines=None):
    baselines=baselines or {'0':ZERO_BASELINE,'10':BASELINE}
    result=dict(actual_steps_all=STEPS,arms={'1':arm_metrics(a,summary)},identical_C_target_features={},
                readout_policy='Separate posthoc affine readouts fitted on TRAIN, evaluated on TEST; never update QCR.',
                caveat='Latent coordinates differ; raw latent losses/root coordinates are not directly comparable.')
    for weight,pins,array_sha in [('0',ZERO_PINS,ZERO_ARRAY_SHA),('10',BASELINE_PINS,BASELINE_ARRAY_SHA)]:
        root=Path(baselines[weight])
        for name,sha in pins.items():
            if sha256(root/(name+'.json'))!=sha:raise ValueError('Baseline JSON changed')
        if sha256(root/'evaluation_arrays.npz')!=array_sha:raise ValueError('Baseline arrays changed')
        old=read_arrays(root/'evaluation_arrays.npz');s=json.loads((root/'summary.json').read_text())
        if any(not np.array_equal(a[k],old[k]) for k in ('labels','splits','cell_ids')):
            raise ValueError('Paired data order/split differs')
        if s['actual_steps']!=STEPS:raise ValueError('Unmatched endpoint')
        result['arms'][weight]=arm_metrics(old,s)
        result['identical_C_target_features'][weight]=np.array_equal(a['target_features'],old['target_features'])
    return result


def plot_supervision_contributions(h,out,plt):
    fig,ax=plt.subplots(figsize=(7,4.5),constrained_layout=True)
    for key,label in [('q_representation','原表示目标（已加权）'),('q_supervision_weighted','监督项（权重 1）')]:
        ax.plot([x['step'] for x in h],[x[key] for x in h],label=label)
    ax.set(xlabel='联合更新次数',ylabel='加权损失贡献',yscale='log',title='Q 目标的实际贡献；不等同于梯度强度')
    ax.legend();ax.grid(alpha=.2);fig.savefig(out/'Q_loss_contributions.png',dpi=160);plt.close(fig)


def plot_flow(a,roots,out,plt):
    from matplotlib.colors import Normalize,TwoSlopeNorm
    means,labels,counts=cell_means(a)
    grid,drift=a['flow_g'],a['flow_displacement']
    if not np.allclose(a['flow_R']-grid,drift,rtol=1e-12,atol=1e-12):raise ValueError('Not R(g)-g')
    spacing=min(np.min(np.diff(np.unique(grid[:,d]))) for d in range(2))
    arrows=drift/np.maximum(np.linalg.norm(drift,axis=1,keepdims=True),1e-15)*.68*spacing
    norms=[Normalize(labels[:,0].min(),labels[:,0].max()),TwoSlopeNorm(vmin=labels[:,1].min(),vcenter=0,vmax=labels[:,1].max())]
    names=dict(attracting='吸引点',repelling='排斥点',saddle='鞍点',neutral_or_unresolved='近中性或未定')
    for k in range(3):
        fig,axes=plt.subplots(1,2,figsize=(12.8,7.6),layout='constrained')
        for d,ax in enumerate(axes):
            ax.quiver(*grid.T,*arrows.T,angles='xy',scale_units='xy',scale=1,color='#7c8792',alpha=.66,width=.0023,zorder=1)
            sc=ax.scatter(*means[:,k].T,c=labels[:,d],s=23,cmap='viridis' if d==0 else 'coolwarm',
                norm=norms[d],edgecolors='#30343b',linewidths=.28,zorder=3)
            fig.colorbar(sc,ax=ax,shrink=.82,label='原始最近邻耦合 K' if d==0 else '原始外场 B')
            for i,r in enumerate(roots,1):
                ax.scatter(*r['g'],marker='*',s=180,facecolors='white',edgecolors='black',zorder=5)
                ax.annotate(f"{names[r['classification']]}{i}"+('（外推）' if not r['within_train_hull'] else ''),
                    r['g'],xytext=(5,10),textcoords='offset points',fontsize=9,
                    bbox=dict(facecolor='white',edgecolor='none',alpha=.85))
            ax.set(xlabel='$g_1$',ylabel='$g_2$',aspect='equal',xlim=a['flow_bounds'][0],ylim=a['flow_bounds'][1],
                   title='按真实 K 着色' if d==0 else '按真实 B 着色');ax.grid(alpha=.12)
        fig.suptitle(f'二维 Ising 弱监督对照（权重 1）：1250 次更新，k={k}\n箭头为 R(g)−g 的方向（等长）')
        fig.supxlabel('散点为同参数两条物理链的平均编码，训练与留出均展示，无插值填色。\n'
            '覆盖外箭头为外推；星号未认证为物理临界点；颜色为原始参数，不是粗尺度有效耦合。',fontsize=9)
        fig.savefig(out/f'R_flow_by_KB_k{k}.png',dpi=190);plt.close(fig)


def render(root):
    root=Path(root)
    m,s,h,r,p=[json.loads((root/(n+'.json')).read_text()) for n in ('manifest','summary','history','roots','train_probes')]
    if (m['recipe']!=RECIPE or s['recipe']!=RECIPE or s['status']!='complete' or s['actual_steps']!=STEPS
            or h[-1]['step']!=STEPS or not m['labels_used_in_training'] or m['test_used_in_training']
            or m['q_supervision']['weight']!=1 or not m['q_supervision']['enabled']
            or m['q_supervision']['scale']!=0 or m['q_supervision']['coarse_labels']):raise ValueError('Wrong matched control')
    if set(s['optimizer_steps'])!={'Q','C','R'} or any(x['min']!=STEPS or x['max']!=STEPS for x in s['optimizer_steps'].values()):
        raise ValueError('Wrong Adam clocks')
    a=read_arrays(root/'evaluation_arrays.npz');means,labels,counts=cell_means(a)
    if a['direct_g'].shape!=(714,3,2) or len(means)!=357 or set(counts)!={2} or np.sum(a['splits']=='train')!=474:
        raise ValueError('Incomplete paired physical data')
    comp=comparison(a,s);plt=_setup_plot(None);out=root/'report';out.mkdir(exist_ok=False)
    plot_supervision_contributions(h,out,plt);plot_losses(h,out,plt);_plot_all_start_closure(s['closure'],out,plt);plot_flow(a,r['roots'],out,plt)
    fig,axes=plt.subplots(2,3,figsize=(13,7.5),constrained_layout=True)
    for d in range(2):
        for k in range(3):
            ax=axes[d,k];sc=ax.scatter(*labels.T,c=means[:,k,d],cmap='coolwarm',s=35,marker='s',vmin=means[...,d].min(),vmax=means[...,d].max())
            ax.set(xlabel='最近邻耦合 K',ylabel='外场 B',title=f'$g_{d+1}$：k={k}，L={36//3**k}')
            fig.colorbar(sc,ax=ax,label='同参数两链平均编码')
    fig.suptitle('弱监督 Q：仅微观 TRAIN 标签、权重 1；粗尺度仍按原始生成参数标记')
    fig.savefig(out/'Q_g_KB.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for k,ratio in enumerate(pca(a)):axes[0].bar(np.arange(2)+.24*(k-1),ratio,width=.24,label=f'k={k}')
    axes[0].set(xticks=[0,1],xticklabels=['主成分1','主成分2'],ylabel='方差占比',title='逐尺度 TRAIN PCA');axes[0].legend()
    axes[1].plot([v['step'] for v in p],[v['weighted_closure'] for v in p],'.-')
    axes[1].set(xlabel='联合更新次数',ylabel='加权闭环损失',yscale='log',title='固定训练面板；不用于选模')
    fig.savefig(out/'PCA_and_train_probe.png',dpi=160);plt.close(fig)
    result=dict(status='complete',recipe=RECIPE,actual_steps=STEPS,roots=r['roots'],closure=s['closure'],thermal_nu=None,
        qualifications=['同数据、种子、初始化、更新次数；仅将 Q 监督权重从 10 降为 1。',
        '这是有监督消融；微观 TRAIN 标签更新 Q，TEST 不参与训练。事后读出系数不参与梯度或选模。',
        '不同潜坐标不能直接比较绝对闭环误差或根坐标；NRMSE 也不能独立认证物理结构。',
        '单seed、有限尺寸、每点两链；未认证热化、不动点物理意义或临界指数。'])
    for name,value in [('comparison',comp),('report',result)]:
        (out/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return result
