"""Model-free Chinese report of the bounded decoder ablation; no root search."""
from pathlib import Path
import json
import numpy as np

ARMS=('A_affine','B_nonlinear','C_balanced')
NAMES=('线性重构','非线性重构','非线性＋方向均衡')


def scores(a, metrics):
    train=a['splits']=='train';test=~train;p=a['target']
    std=np.sqrt(np.mean((p[train]-p[train].mean(0))**2,axis=(0,2)))
    result={}
    for arm in ('source',)+ARMS:
        d=a[arm+'_decoded'];g=a[arm+'_g'];err=d-p
        reconstruction=[]
        for k in range(3):
            axes=metrics[f'{k}_axes'];var=metrics[f'{k}_variances'][metrics[f'{k}_reliable']]
            row=dict(k=k,train_rmse=float(np.sqrt(np.mean(err[train,k]**2))),
                     test_rmse=float(np.sqrt(np.mean(err[test,k]**2))),original_TRAIN_target_std=float(std[k]),
                     test_nrmse=float(np.sqrt(np.mean(err[test,k]**2))/std[k]),reliable_observable_directions=int(axes.shape[1]))
            if len(var):
                directional=np.mean((err[test,k]@axes)**2,axis=0)/np.maximum(var,1e-30)
                row.update(reliable_direction_mean_relative_mse=float(directional.mean()),
                           reliable_direction_relative_mse=directional.tolist())
            reconstruction.append(row)
        row=dict(reconstruction=reconstruction,
                 subset_rms=np.sqrt(np.mean((a[arm+'_first']-a[arm+'_second'])**2,axis=(0,2))).tolist(),
                 half_full_rms=np.sqrt(np.mean((g-a[arm+'_g_full'])**2,axis=(0,2))).tolist())
        cells=np.unique(a['cell_ids']);group=[a['cell_ids']==c for c in cells]
        means=np.array([g[ix].mean(0) for ix in group]);decoded=np.array([d[ix].mean(0) for ix in group])
        labels=np.array([a['labels'][ix][0,:2] for ix in group]);rs,us=np.unique(labels[:,0]),np.unique(labels[:,1])
        if len(rs)*len(us)!=len(cells):raise ValueError('Expected full evaluation grid')
        gj=np.stack(np.gradient(means.reshape(len(rs),len(us),3,2),rs/np.ptp(rs),us/np.ptp(us),axis=(0,1)),axis=-1)
        singular=np.linalg.svd(gj,compute_uv=False)
        row['posthoc_grid_g_Jacobian_sigma2_over_sigma1_median']=np.median(singular[...,1]/np.maximum(singular[...,0],1e-30),axis=(0,1)).tolist()
        # Descriptor-space derivative error does not improve just by stretching g axes.
        truth=np.array([p[ix].mean(0) for ix in group])
        tj=np.stack(np.gradient(truth.reshape(len(rs),len(us),3,-1),rs/np.ptp(rs),us/np.ptp(us),axis=(0,1)),axis=-1)
        dj=np.stack(np.gradient(decoded.reshape(len(rs),len(us),3,-1),rs/np.ptp(rs),us/np.ptp(us),axis=(0,1)),axis=-1)
        _,ss,vh=np.linalg.svd(tj,full_matrices=False)
        weak=vh[...,-1,:]
        weakerror=np.linalg.norm(np.einsum('...ij,...j->...i',dj-tj,weak),axis=-1)/np.maximum(ss[...,-1],1e-30)
        row['posthoc_descriptor_weak_tangent_relative_error_median']=np.median(weakerror,axis=(0,1)).tolist()
        if arm!='source':
            closure=[]
            for start,target,pred,predp in ((0,1,a[arm+'_one_step'][:,0],a[arm+'_one_step_decoded'][:,0]),
                                          (1,2,a[arm+'_one_step'][:,1],a[arm+'_one_step_decoded'][:,1]),
                                          (0,2,a[arm+'_rollout'],a[arm+'_rollout_decoded'])):
                error=pred-g[:,target];gstd=np.sqrt(np.mean((g[train,target]-g[train,target].mean(0))**2))
                delta=g[:,target]-g[:,start]
                closure.append(dict(start=start,target=target,test_g_rmse=float(np.sqrt(np.mean(error[test]**2))),
                    test_g_nrmse=float(np.sqrt(np.mean(error[test]**2))/max(gstd,1e-30)),
                    test_error_over_displacement_rms=float(np.linalg.norm(error[test])/max(np.linalg.norm(delta[test]),1e-30)),
                    test_decoded_statistics_nrmse=float(np.sqrt(np.mean((predp[test]-p[test,target])**2))/std[target])))
            row['R_readout']=closure
        result[arm]=row
    return result


def render(root):
    root=Path(root)
    manifest=json.loads((root/'manifest.json').read_text());summary=json.loads((root/'summary.json').read_text())
    if summary['status']!='complete' or not summary['source_unchanged'] or summary['C_updates']:
        raise ValueError('Incomplete or mutated source')
    for arm in ARMS:
        if summary['arms'][arm]['Q_steps']!=600 or summary['arms'][arm]['R_steps']!=1000:
            raise ValueError('Incomplete fixed endpoint')
    with np.load(root/'evaluation_arrays.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
    with np.load(root/'metric_arrays.npz',allow_pickle=False) as z:m={k:z[k] for k in z.files}
    result=scores(a,m)
    out=root/'report';out.mkdir(exist_ok=False)
    from scripts.report_phi4_hopping_qcr import _setup_plot
    plt=_setup_plot(None)
    train=a['splits']=='train'
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for i,(arm,name) in enumerate(zip(ARMS,NAMES)):
        axes[0].plot(range(3),[r['test_nrmse'] for r in result[arm]['reconstruction']],'.-',label=name)
        axes[1].plot(range(3),[r.get('reliable_direction_mean_relative_mse',np.nan) for r in result[arm]['reconstruction']],'.-',label=name)
        axes[2].plot(range(3),[r['test_decoded_statistics_nrmse'] for r in result[arm]['R_readout']],'.-',label=name)
    axes[0].plot(range(3),[r['test_nrmse'] for r in result['source']['reconstruction']],'k--',label='源 Q（未追加训练）')
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2);ax.set_xticks(range(3))
    axes[0].set(xlabel='尺度 k',ylabel='留出重构 NRMSE',title='原 TRAIN 统计标准差归一化')
    axes[1].set(xlabel='尺度 k',ylabel='可靠方向平均相对平方误差',title='同一统计空间度量；不是物理维数认证')
    axes[2].set_xticklabels(['36→12','12→4','36→4']);axes[2].set(ylabel='预测统计 NRMSE',title='D(R(g)) 的留出预测')
    fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for j,(arm,name) in enumerate(zip(ARMS,NAMES)):
        h=json.loads((root/(arm+'_history.json')).read_text());rh=json.loads((root/(arm+'_R_history.json')).read_text())
        for k in range(3):axes[0,j].plot([x['step'] for x in h],[x['reconstruction_raw'][k] for x in h],label=f'重构 k={k}')
        axes[0,j].set(title=name,xlabel='追加 Q 更新次数',ylabel='原口径重构平方误差',yscale='log');axes[0,j].legend(fontsize=8)
        axes[1,j].plot([x['step'] for x in rh],[x['loss'] for x in rh]);axes[1,j].set(xlabel='独立 R 读出更新次数',ylabel='均匀目标尺度闭环损失',yscale='log')
    fig.suptitle('固定终点；训练批次曲线不用于选模')
    fig.savefig(out/'loss_curves.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
    for ax,arm,name in zip(axes,ARMS,NAMES):
        grid=a[arm+'_flow_grid'];delta=a[arm+'_flow_displacement'];g=a[arm+'_g']
        length=np.linalg.norm(delta,axis=-1,keepdims=True)
        arrow=delta/np.maximum(length,1e-15)*np.ptp(grid,axis=0).min()/32
        ax.quiver(grid[:,0],grid[:,1],arrow[:,0],arrow[:,1],angles='xy',scale_units='xy',scale=1,width=.003,color='#667788',alpha=.7)
        for k in range(3):
            ax.scatter(g[train,k,0],g[train,k,1],s=10,color=f'C{k}',alpha=.6,label=f'训练 k={k}')
            ax.scatter(g[~train,k,0],g[~train,k,1],s=13,marker='x',color=f'C{k}',alpha=.8)
        ax.set(title=name,xlabel='$g_1$',ylabel='$g_2$',aspect='equal');ax.legend(fontsize=7)
    fig.suptitle('短 R 读出：R(g)−g 等长方向箭头；散点为训练、叉号为留出；覆盖外为外推\n各臂坐标不同；不搜索或认证不动点')
    fig.savefig(out/'R_flow.png',dpi=150);plt.close(fig)
    cells=np.unique(a['cell_ids']);groups=[a['cell_ids']==c for c in cells]
    labels=np.array([a['labels'][ix][0,:2] for ix in groups]);testcell=np.array([not train[ix][0] for ix in groups])
    for k in range(3):
        fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
        for j,(arm,name) in enumerate(zip(ARMS,NAMES)):
            means=np.array([a[arm+'_g'][ix,k].mean(0) for ix in groups])
            for d in range(2):
                ax=axes[d,j];sc=ax.scatter(labels[:,0],labels[:,1],c=means[:,d],s=130,marker='s',cmap='coolwarm')
                ax.scatter(labels[testcell,0],labels[testcell,1],c='black',marker='x',s=20)
                ax.set(title=name+f'：$g_{d+1}$',xlabel='质量参数 r',ylabel='四次耦合 u');fig.colorbar(sc,ax=ax)
        fig.suptitle(f'尺度 k={k} 的耦合—编码映射；四链平均，叉号为留出参数点\nr、u 仅用于事后评价，不进入训练')
        fig.savefig(out/f'Q_g_ru_k{k}.png',dpi=140);plt.close(fig)
    report=dict(status='complete',metrics=result,source=manifest['source'],data_warning_count=manifest['data_warning_count'],
        qualifications=['固定历史 C、源 Q 热启动、小批量短适配；不是从头联合 QCR 的结论。',
            '每物理链64个已保存构型，一次固定 C 随机轨迹；未检验新 C 重复的泛化。',
            '重复物理链用于有限样本噪声估计；保留原热化警告，不认证平衡。',
            '弱方向和局部参数导数仅为描述性诊断，不认证物理维数。',
            'R 只作相同预算的新读出；各臂 g 坐标不同，优先比较共同统计空间的预测。',
            '无耦合监督，无 TEST 选模，无不动点搜索；thermal_nu=null。'],thermal_nu=None)
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return report
