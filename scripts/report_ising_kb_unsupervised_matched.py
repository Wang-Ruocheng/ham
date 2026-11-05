"""Frozen-array report: identical-data supervised versus label-free comparison."""
import json
from pathlib import Path
import numpy as np
from scripts.report_phi4_hopping_qcr import _setup_plot, _plot_all_start_closure
from scripts.report_ising_kb_cnn_qcr import plot_losses
from scripts.plot_saved_kb_parameter_flow import cell_means, sha256
from experiments.ising_kb_unsupervised_matched import RECIPE, STEPS, BASELINE, BASELINE_PINS

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


def comparison(a,summary,baseline=BASELINE):
    baseline=Path(baseline)
    for name,sha in BASELINE_PINS.items():
        if sha256(baseline/(name+'.json'))!=sha:raise ValueError('Baseline JSON changed')
    if sha256(baseline/'evaluation_arrays.npz')!=BASELINE_ARRAY_SHA:raise ValueError('Baseline arrays changed')
    old=read_arrays(baseline/'evaluation_arrays.npz');s=json.loads((baseline/'summary.json').read_text())
    if any(not np.array_equal(a[k],old[k]) for k in ('labels','splits','cell_ids')):
        raise ValueError('Paired data order/split differs')
    return dict(actual_steps_both=STEPS,labels_only_for_posthoc_readout=True,
        identical_C_target_features=np.array_equal(a['target_features'],old['target_features']),
        supervised=dict(closure=s['closure'],final_MI=s['final_MI'],roots=s['numerical_fixed_points'],
                        TRAIN_affine_readout=linear_readout(old),PCA_by_scale=pca(old)),
        unsupervised=dict(closure=summary['closure'],final_MI=summary['final_MI'],roots=summary['numerical_fixed_points'],
                          TRAIN_affine_readout=linear_readout(a),PCA_by_scale=pca(a)),
        caveat='Different latent coordinates: do not compare raw latent RMSE or root coordinates directly.')


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
        fig.suptitle(f'二维 Ising 无监督对照：1250 次更新，k={k}\n箭头为 R(g)−g 的方向（等长）')
        fig.supxlabel('散点为同参数两条物理链的平均编码，训练与留出均展示，无插值填色。\n'
            '覆盖外箭头为外推；星号未认证为物理临界点；颜色为原始参数，不是粗尺度有效耦合。',fontsize=9)
        fig.savefig(out/f'R_flow_by_KB_k{k}.png',dpi=190);plt.close(fig)


def render(root):
    root=Path(root)
    m,s,h,r,p=[json.loads((root/(n+'.json')).read_text()) for n in ('manifest','summary','history','roots','train_probes')]
    if (m['recipe']!=RECIPE or s['recipe']!=RECIPE or s['status']!='complete' or s['actual_steps']!=STEPS
            or h[-1]['step']!=STEPS or m['labels_used_in_training'] or m['test_used_in_training']
            or m['q_supervision']['weight']!=0 or m['q_supervision']['enabled']):raise ValueError('Wrong matched control')
    if set(s['optimizer_steps'])!={'Q','C','R'} or any(x['min']!=STEPS or x['max']!=STEPS for x in s['optimizer_steps'].values()):
        raise ValueError('Wrong Adam clocks')
    a=read_arrays(root/'evaluation_arrays.npz');means,labels,counts=cell_means(a)
    if a['direct_g'].shape!=(714,3,2) or len(means)!=357 or set(counts)!={2} or np.sum(a['splits']=='train')!=474:
        raise ValueError('Incomplete paired physical data')
    comp=comparison(a,s);plt=_setup_plot(None);out=root/'report';out.mkdir(exist_ok=False)
    plot_losses(h,out,plt);_plot_all_start_closure(s['closure'],out,plt);plot_flow(a,r['roots'],out,plt)
    fig,axes=plt.subplots(2,3,figsize=(13,7.5),constrained_layout=True)
    for d in range(2):
        for k in range(3):
            ax=axes[d,k];sc=ax.scatter(*labels.T,c=means[:,k,d],cmap='coolwarm',s=35,marker='s',vmin=means[...,d].min(),vmax=means[...,d].max())
            ax.set(xlabel='最近邻耦合 K',ylabel='外场 B',title=f'$g_{d+1}$：k={k}，L={36//3**k}')
            fig.colorbar(sc,ax=ax,label='同参数两链平均编码')
    fig.suptitle('无监督 Q：K/B 仅用于事后评价；粗尺度仍按原始生成参数标记')
    fig.savefig(out/'Q_g_KB.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for k,ratio in enumerate(pca(a)):axes[0].bar(np.arange(2)+.24*(k-1),ratio,width=.24,label=f'k={k}')
    axes[0].set(xticks=[0,1],xticklabels=['主成分1','主成分2'],ylabel='方差占比',title='逐尺度 TRAIN PCA');axes[0].legend()
    axes[1].plot([v['step'] for v in p],[v['weighted_closure'] for v in p],'.-')
    axes[1].set(xlabel='联合更新次数',ylabel='加权闭环损失',yscale='log',title='固定训练面板；不用于选模')
    fig.savefig(out/'PCA_and_train_probe.png',dpi=160);plt.close(fig)
    result=dict(status='complete',recipe=RECIPE,actual_steps=STEPS,roots=r['roots'],closure=s['closure'],thermal_nu=None,
        qualifications=['同数据、种子、初始化、更新次数；仅删除 Q 的监督项。',
        '事后线性读出使用 TRAIN 标签拟合，绝未用于 QCR 训练、终点选择或梯度。',
        '不同潜坐标不能直接比较绝对闭环误差或根坐标；NRMSE 也不能独立认证物理结构。',
        '单seed、有限尺寸、每点两链；未认证热化、不动点物理意义或临界指数。'])
    for name,value in [('comparison',comp),('report',result)]:
        (out/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return result
