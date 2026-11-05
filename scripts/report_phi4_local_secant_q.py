"""Read saved public arrays only; never select an endpoint or search a root."""
import json
from pathlib import Path
import numpy as np
from experiments.phi4_local_secant_q import ARMS

NAMES={'source':'源 Q','fresh_initial':'随机初始 Q','warm_base':'热启动／原目标','warm_local':'热启动／局部项',
       'fresh_base':'随机初始化／原目标','fresh_local':'随机初始化／局部项','original':'源 Q＋原 R'}


def scores(a):
    p=a['target'];train=a['inner_splits']=='train'
    center=p[train].mean(0);std=np.sqrt(((p[train]-center)**2).mean((0,2)))
    groups=[np.flatnonzero(a['cell_ids']==c) for c in np.unique(a['cell_ids'])]
    labels=np.array([a['labels'][ix[0],:2] for ix in groups]);lookup={tuple(np.round(x,8)):i for i,x in enumerate(labels)}
    pm=np.array([p[ix].mean(0) for ix in groups]);means=pm[[train[ix[0]] for ix in groups]]
    axes=[np.linalg.svd(means[:,k]-means[:,k].mean(0),full_matrices=False)[2][:2].T for k in range(3)]
    out={}
    for name in ('source','fresh_initial')+ARMS:
        d=a[name+'_decoded'];g=a[name+'_g'];res=d-p;row={}
        for split in ('train','validation','test'):
            ix=a['inner_splits']==split
            rmse=np.sqrt((res[ix]**2).mean((0,2)))
            row[split]=dict(rmse=rmse.tolist(),nrmse=(rmse/std).tolist(),
                pc2_explained=[float(1-np.mean((res[ix,k]@axes[k][:,1])**2)/max(np.mean(((p[ix,k]-center[k])@axes[k][:,1])**2),1e-30)) for k in range(3)])
        gm=np.array([g[ix].mean(0) for ix in groups]);dm=np.array([d[ix].mean(0) for ix in groups]);tangents=[]
        for k in range(3):
            ratios=[];errors=[];even_errors=[]
            for r,u in labels:
                keys=[(r-.15,u),(r+.15,u),(r,u-.25),(r,u+.25)]
                if not all(tuple(np.round(key,8)) in lookup for key in keys):continue
                ids=[lookup[tuple(np.round(key,8))] for key in keys]
                def jac(x):return np.stack(((x[ids[1],k]-x[ids[0],k])/2,(x[ids[3],k]-x[ids[2],k])/2),-1)
                sj=np.linalg.svd(jac(gm),compute_uv=False);ratios.append(float(sj[-1]/max(sj[0],1e-30)))
                for even,dest in ((False,errors),(True,even_errors)):
                    target=jac(pm);decoded=jac(dm)
                    if even:target=(target+target[::-1])/2;decoded=(decoded+decoded[::-1])/2
                    _,s,vh=np.linalg.svd(target,full_matrices=False)
                    dest.append(float(np.linalg.norm((decoded-target)@vh[-1])/max(s[-1],1e-30)))
            tangents.append(dict(k=k,grid_interiors=len(ratios),median_g_sigma2_over_sigma1=float(np.median(ratios)) if ratios else None,
                median_weak_secant_relative_error=float(np.median(errors)) if errors else None,
                median_even_weak_secant_relative_error=float(np.median(even_errors)) if even_errors else None))
        row['posthoc_all_grid_tangents']=tangents;out[name]=row
    for name in ('original',)+ARMS:
        if name+'_one_step_decoded' not in a:continue
        ix=a['inner_splits']=='test';closure=[]
        for s,t,pred in ((0,1,a[name+'_one_step_decoded'][:,0]),(1,2,a[name+'_one_step_decoded'][:,1]),(0,2,a[name+'_rollout_decoded'])):
            rmse=float(np.sqrt(((pred[ix]-p[ix,t])**2).mean()))
            closure.append(dict(start=s,target=t,rmse=rmse,nrmse=rmse/std[t]))
        out.setdefault(name,{})['test_closure']=closure
    return out


def render(root):
    root=Path(root);summary=json.loads((root/'summary.json').read_text());gate=json.loads((root/'gate.json').read_text())
    assert summary['status']=='complete' and summary['source_unchanged'] and summary['C_updates']==0
    assert all(summary['arms'][n]['Q_steps']==600 for n in ARMS)
    for init in ('warm','fresh'):assert summary['arms'][init+'_base']['Q_initial_sha256']==summary['arms'][init+'_local']['Q_initial_sha256']
    assert all(summary['arms'][n]['R_steps']==(1000 if summary['R_gate_passed'] else 0) for n in ARMS)
    if summary['R_gate_passed']:assert len({summary['arms'][n]['R_initial_sha256'] for n in ARMS})==1
    with np.load(root/'evaluation_arrays.npz',allow_pickle=False) as z:a={key:z[key] for key in z.files}
    assert all(np.isfinite(x).all() for x in a.values() if x.dtype.kind in 'fc')
    result=scores(a);out=root/'report';out.mkdir(exist_ok=False)
    from scripts.report_phi4_hopping_qcr import _setup_plot
    plt=_setup_plot(None)
    fig,axes=plt.subplots(1,3,figsize=(14,4.4),layout='constrained')
    for name in ('source',)+ARMS:
        axes[0].plot(range(3),result[name]['test']['nrmse'],'-o',label=NAMES[name])
        axes[1].plot(range(3),result[name]['test']['pc2_explained'],'-o',label=NAMES[name])
        axes[2].plot(range(3),[r['median_even_weak_secant_relative_error'] for r in result[name]['posthoc_all_grid_tangents']],'-o',label=NAMES[name])
    for ax in axes:ax.set_xticks(range(3));ax.set_xlabel('尺度 k');ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0].set(title='原 TEST：共同描述量空间',ylabel='重构 NRMSE')
    axes[1].set(title='全局主成分不代表局部物理维度',ylabel='第二主成分解释率');axes[1].axhline(0,color='gray',lw=.7)
    axes[2].set(title='全部网格中心：事后有限差分',ylabel='偶统计弱方向相对误差');axes[2].axhline(1,color='gray',lw=.7)
    fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(17,7),layout='constrained')
    for j,name in enumerate(ARMS):
        h=json.loads((root/(name+'_Q_history.json')).read_text());steps=[x['step'] for x in h]
        for k in range(3):axes[0,j].plot(steps,[x['reconstruction_raw'][k] for x in h],label=f'重构 k={k}')
        axes[0,j].set(title=NAMES[name],xlabel='Q 更新次数',ylabel='绝对平方误差',yscale='log');axes[0,j].legend(fontsize=8)
        axes[1,j].plot(steps,[x['base_total'] for x in h],label='原 Q 总目标')
        axes[1,j].plot(steps,[x['q_total'] for x in h],label='实际 Q 总目标')
        axes[1,j].plot(steps,[10*x['local'] for x in h],label='局部差异项×10（未乘系数）')
        axes[1,j].set(xlabel='Q 更新次数',ylabel='损失');axes[1,j].set_yscale('symlog',linthresh=1e-6);axes[1,j].legend(fontsize=8)
    fig.suptitle('相同批次与固定600步；不同初始化的初值不同；不按曲线选终点')
    fig.savefig(out/'loss_curves.png',dpi=160);plt.close(fig)
    groups=[np.flatnonzero(a['cell_ids']==c) for c in np.unique(a['cell_ids'])]
    labels=np.array([a['labels'][ix[0],:2] for ix in groups])
    fig,axes=plt.subplots(3,4,figsize=(16,10),layout='constrained')
    for j,name in enumerate(ARMS):
        means=np.array([a[name+'_g'][ix].mean(0) for ix in groups])
        for k in range(3):
            ax=axes[k,j]
            for u in np.unique(labels[:,1]):
                ix=np.flatnonzero(labels[:,1]==u);ix=ix[np.argsort(labels[ix,0])]
                ax.plot(*means[ix,k].T,'-o',ms=2.5,label=f'u={u:.2f}')
            for r in np.unique(labels[:,0]):
                ix=np.flatnonzero(labels[:,0]==r);ix=ix[np.argsort(labels[ix,1])];ax.plot(*means[ix,k].T,color='gray',lw=.6,alpha=.5)
            ax.set(title=NAMES[name]+f'，k={k}',xlabel='$g_1$',ylabel='$g_2$',aspect='equal');ax.tick_params(labelsize=8)
    handles,legend=axes[0,0].get_legend_handles_labels();fig.legend(handles,legend,loc='outside lower center',ncol=5)
    fig.suptitle('参数只用于事后标色：彩线固定 u，灰线固定 r；二维输出不等于二维信息')
    fig.savefig(out/'Q_parameter_grid.png',dpi=160);plt.close(fig)
    names=('original',)+ARMS if summary['R_gate_passed'] else ('original',)
    fig,axes=plt.subplots(1,len(names),figsize=(5*len(names),5),squeeze=False,layout='constrained')
    for ax,name in zip(axes[0],names):
        grid=a[name+'_flow_grid'];delta=a[name+'_flow_displacement'];g=a[('source' if name=='original' else name)+'_g']
        arrows=delta/np.maximum(np.linalg.norm(delta,axis=1,keepdims=True),1e-15)*np.ptp(grid,axis=0).max()/35
        ax.quiver(*grid.T,*arrows.T,angles='xy',scale_units='xy',scale=1,width=.003,color='#777777',alpha=.5)
        for k in range(3):ax.scatter(*g[:,k].T,s=8,label=f'数据 k={k}',color=f'C{k}')
        ax.set(title=NAMES[name],xlabel='$g_1$',ylabel='$g_2$',aspect='equal');ax.legend(fontsize=8)
    fig.suptitle('R(g)−g 方向箭头；散点为数据覆盖；不搜索不动点'+('' if summary['R_gate_passed'] else '\nQ 验证门未通过：仅展示原模型，未训练新的 R'))
    fig.savefig(out/'R_flow.png',dpi=160);plt.close(fig)
    if summary['R_gate_passed']:
        fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
        for name in ARMS:
            h=json.loads((root/(name+'_R_history.json')).read_text());ax.plot([x['step'] for x in h],[x['loss'] for x in h],label=NAMES[name])
        ax.set(xlabel='R 更新次数',ylabel='闭环损失（各臂潜坐标不同）',yscale='log');ax.legend();fig.savefig(out/'R_loss.png',dpi=160);plt.close(fig)
    result=dict(status='complete',arms=result,gate=gate,thermal_nu=None,
        qualifications=['固定同一个 C 缓存；零 C 更新、零新采样；每链64构型、一次固定 C 轨迹。',
            '原 TRAIN 内固定留出6条件作验证，28条件训练；原 TEST 的11条件不选参数、不决定 R 门控。',
            '内部验证条件仅从本轮更新排除；历史源 Q/C 训练曾使用这些条件，不是全流程独立验证。',
            '四臂固定600步，随机初始化不能据此认定已收敛；无多 seed 结论。',
            '数据保留55项历史采样警告；四条链的噪声检查不认证平衡或物理秩。',
            '全网格有限差分仅作事后描述，部分邻点属于 TRAIN；不是独立 TEST 导数检验。',
            '偶投影仅作评价，粗粒化 C 不保证 Z2 等变；不强制二维，不宣称鞍点或热指数。'])
    (out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return result
