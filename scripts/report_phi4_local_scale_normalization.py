"""Saved-array Chinese report; no model loading, root search or selection."""
from pathlib import Path
import json
import numpy as np
from experiments.phi4_local_scale_normalization import ARMS,READOUTS,Q_STEPS,R_STEPS

NAMES={'original':'源 Q＋原 R','source':'源 Q＋重拟合 R','A_original':'原目标追加 Q','B_scale_normalized':'尺度归一化追加 Q'}


def scores(a):
    p=a['target'];train=a['splits']=='train';test=~train
    groups=[np.flatnonzero(a['cell_ids']==c) for c in np.unique(a['cell_ids'])]
    tc=np.array([train[ix[0]] for ix in groups]);means=np.array([p[ix].mean(0) for ix in groups])
    centers=p[train].mean(0);std=np.sqrt(np.mean((p[train]-centers)**2,axis=(0,2)))
    axes=[np.linalg.svd(means[tc,k]-means[tc,k].mean(0),full_matrices=False)[2][:2].T for k in range(3)]
    results={}
    for arm in READOUTS:
        g=a[arm+'_g'];residual=a[arm+'_decoded']-p;reconstruction=[]
        for k in range(3):
            row=dict(k=k,train_rmse=float(np.sqrt((residual[train,k]**2).mean())),
                test_rmse=float(np.sqrt((residual[test,k]**2).mean())),original_TRAIN_target_std=float(std[k]),
                test_nrmse=float(np.sqrt((residual[test,k]**2).mean())/std[k]),
                test_pc_explained_fraction=(1-np.mean((residual[test,k]@axes[k])**2,axis=0)
                    /np.mean(((p[test,k]-centers[k])@axes[k])**2,axis=0)).tolist())
            reconstruction.append(row)
        results[arm]=dict(reconstruction=reconstruction,
            subset_rms=np.sqrt(np.mean((a[arm+'_first']-a[arm+'_second'])**2,axis=(0,2))).tolist(),
            half_full_rms=np.sqrt(np.mean((g-a[arm+'_g_full'])**2,axis=(0,2))).tolist())
    results['original']={'reconstruction':results['source']['reconstruction']}
    for arm in ('original',)+READOUTS:
        g=a[('source' if arm=='original' else arm)+'_g'];rows=[]
        for s,t,pred,decoded in [(0,1,a[arm+'_one_step'][:,0],a[arm+'_one_step_decoded'][:,0]),
                                (1,2,a[arm+'_one_step'][:,1],a[arm+'_one_step_decoded'][:,1]),
                                (0,2,a[arm+'_rollout'],a[arm+'_rollout_decoded'])]:
            scale=np.sqrt(np.mean((g[train,t]-g[train,t].mean(0))**2))
            g_rmse=float(np.sqrt(np.mean((pred[test]-g[test,t])**2)))
            p_rmse=float(np.sqrt(np.mean((decoded[test]-p[test,t])**2)))
            rows.append(dict(start=s,target=t,test_g_rmse=g_rmse,
                test_g_nrmse=g_rmse/max(float(scale),1e-30),
                test_decoded_statistics_nrmse=p_rmse/float(std[t])))
        results[arm]['closure']=rows
    # Paired resampling of whole physical chains within each fixed TEST cell.
    rng=np.random.default_rng(2026110308);testgroups=[ix for ix in groups if test[ix[0]]]
    draws=np.concatenate([ix[rng.integers(0,len(ix),(1000,len(ix)))] for ix in testgroups],axis=1)
    paired={}
    for k in range(3):
        ae=np.mean((a[ARMS[0]+'_decoded'][:,k]-p[:,k])**2,axis=-1)
        be=np.mean((a[ARMS[1]+'_decoded'][:,k]-p[:,k])**2,axis=-1)
        delta=(np.sqrt(be[draws].mean(1))-np.sqrt(ae[draws].mean(1)))/std[k]
        paired[str(k)]=dict(B_minus_A_nrmse_interval=np.quantile(delta,[.025,.5,.975]).tolist())
    return dict(arms=results,paired_Q_chain_bootstrap=paired,
                bootstrap_qualification='1000 paired whole-chain resamples within each fixed TEST cell; fixed C draw and training seed; not equilibrium or seed uncertainty')


def render(root):
    root=Path(root);summary=json.loads((root/'summary.json').read_text())
    assert summary['status']=='complete' and summary['source_unchanged'] and summary['C_updates']==0
    for arm in READOUTS:
        assert summary['arms'][arm]['Q_steps']==(0 if arm=='source' else Q_STEPS)
        assert summary['arms'][arm]['R_steps']==R_STEPS
    assert len({summary['arms'][arm]['R_initial_sha256'] for arm in READOUTS})==1
    assert summary['arms'][ARMS[0]]['Q_initial_sha256']==summary['arms'][ARMS[1]]['Q_initial_sha256']
    with np.load(root/'evaluation_arrays.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
    for v in a.values():
        if v.dtype.kind in 'fc':assert np.isfinite(v).all()
    result=scores(a);out=root/'report';out.mkdir(exist_ok=False)
    from scripts.report_phi4_hopping_qcr import _setup_plot
    plt=_setup_plot(None)
    fig,axes=plt.subplots(1,3,figsize=(14.5,4.2),layout='constrained')
    for arm in READOUTS:
        rows=result['arms'][arm]['reconstruction']
        axes[0].plot(range(3),[r['test_nrmse'] for r in rows],'-o',label=NAMES[arm])
        axes[1].plot(range(3),[r['test_pc_explained_fraction'][1] for r in rows],'-o',label=NAMES[arm])
    for arm in ('original',)+READOUTS:
        axes[2].plot(range(3),[r['test_decoded_statistics_nrmse'] for r in result['arms'][arm]['closure']],'-o',label=NAMES[arm])
    for ax in axes:ax.legend(fontsize=8);ax.set_xticks(range(3));ax.grid(alpha=.2)
    axes[0].set(xlabel='尺度 k',ylabel='留出重构 NRMSE',title='同一 TRAIN 描述量标准差归一化')
    axes[1].set(xlabel='尺度 k',ylabel='第二主成分解释率',title='主成分只用 TRAIN 拟合');axes[1].axhline(0,color='gray',lw=.7)
    axes[2].set_xticklabels(['36→12','12→4','36→4']);axes[2].set(ylabel='留出预测统计 NRMSE',title='共同统计空间：D(R(g)) 对比')
    fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for j,arm in enumerate(ARMS):
        h=json.loads((root/(arm+'_Q_history.json')).read_text())
        for k in range(3):axes[0,j].plot([x['step'] for x in h],[x['reconstruction_raw'][k] for x in h],label=f'原口径重构 k={k}')
        axes[0,j].set(title=NAMES[arm],xlabel='追加 Q 更新次数',ylabel='绝对平方误差',yscale='log');axes[0,j].legend(fontsize=8)
    for arm in READOUTS:
        h=json.loads((root/(arm+'_R_history.json')).read_text())
        axes[1,0].plot([x['step'] for x in h],[x['loss'] for x in h],label=NAMES[arm])
    axes[1,0].set(xlabel='R 拟合更新次数',ylabel='闭环训练损失（各臂坐标不同）',yscale='log');axes[1,0].legend(fontsize=8)
    metric=json.loads((root/'scale_metric.json').read_text())
    axes[1,1].bar(range(3),metric['weights']);axes[1,1].set(xticks=range(3),xlabel='尺度 k',ylabel='冻结的重构系数',title='仅重构项变权；子集项及 R 权重不变')
    fig.savefig(out/'loss_curves.png',dpi=160);plt.close(fig)
    train=a['splits']=='train'
    fig,axes=plt.subplots(1,4,figsize=(18,4.5),layout='constrained')
    for ax,arm in zip(axes,('original',)+READOUTS):
        grid=a[arm+'_flow_grid'];delta=a[arm+'_flow_displacement'];g=a[('source' if arm=='original' else arm)+'_g']
        arrows=delta/np.maximum(np.linalg.norm(delta,axis=1,keepdims=True),1e-15)*np.ptp(grid,axis=0).min()/30
        ax.quiver(*grid.T,*arrows.T,angles='xy',scale_units='xy',scale=1,width=.003,color='#777777',alpha=.6)
        for k in range(3):
            ax.scatter(*g[train,k].T,s=7,color=f'C{k}',label=f'训练 k={k}')
            ax.scatter(*g[~train,k].T,s=10,marker='x',color=f'C{k}')
        ax.set(title=NAMES[arm],xlabel='$g_1$',ylabel='$g_2$',aspect='equal');ax.legend(fontsize=7)
    fig.suptitle('R(g)−g 等长方向箭头；散点／叉号为训练／留出；数据外箭头仅为外推，不搜索不动点')
    fig.savefig(out/'R_flow.png',dpi=160);plt.close(fig)
    groups=[np.flatnonzero(a['cell_ids']==c) for c in np.unique(a['cell_ids'])]
    labels=np.array([a['labels'][ix[0],:2] for ix in groups])
    fig,axes=plt.subplots(3,3,figsize=(12,10),layout='constrained')
    for j,arm in enumerate(READOUTS):
        means=np.array([a[arm+'_g'][ix].mean(0) for ix in groups])
        for k in range(3):
            ax=axes[k,j]
            for u in np.unique(labels[:,1]):
                ix=np.flatnonzero(labels[:,1]==u);ax.plot(*means[ix,k].T,'-o',ms=3,label=f'u={u:.2f}')
            for r in np.unique(labels[:,0]):
                ix=np.flatnonzero(labels[:,0]==r);ax.plot(*means[ix,k].T,color='gray',lw=.6,alpha=.4)
            ax.set(title=NAMES[arm]+f'，k={k}',xlabel='$g_1$',ylabel='$g_2$',aspect='equal')
            if k==0:ax.legend(fontsize=7)
    fig.suptitle('参数网格经 Q 的映射：彩线固定 u，灰线固定 r；参数仅用于评价')
    fig.savefig(out/'Q_parameter_grid.png',dpi=160);plt.close(fig)
    result.update(status='complete',qualifications=['源 Q 热启动、固定同一个已训练 C；不是从头联合训练。',
        'A/B 完全相同批次和初始化，只有三项重构标量系数不同；全局匹配初始重构损失。',
        '每物理链使用原128个构型的偶数编号64个；一次固定 C 轨迹；未检验新的 C 随机轨迹。',
        '保留原55项采样警告；四条链不认证平衡或可靠物理维度。',
        '全部端点预先固定，无 TEST 选模；不因流场形状或不动点调参。',
        '各臂潜坐标可变化；优先比较同一描述量空间的重构与预测。'],thermal_nu=None)
    (out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return result
