"""Chinese, arrays-only report for the dg=1 residual decoder ablation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts import report_phi4_hopping_qonly_balance as common

MODELS = ('source', 'A_affine', 'B_nonlinear')
DISPLAY = ('源 Q', 'A：共享仿射解码器', 'B：共享非线性解码器')
COLORS = ('#777777', '#0072B2', '#D55E00')
INPUTS = ('plan.json', 'history.json', 'metrics.json', 'summary.json', 'evaluation_arrays.npz')


def load_run(root):
    pins = {p:common._sha(root/p) for p in INPUTS}
    plan, history, metrics, summary = [json.loads((root/p).read_text()) for p in INPUTS[:4]]
    if (summary.get('status') != 'complete' or summary.get('theory_dim') != 1
            or summary.get('C_updates') != 0 or summary.get('R_updates') != 0
            or summary.get('labels_used_in_training') is not False
            or summary.get('old_R_valid_for_new_Q') is not False):
        raise ValueError('Incompatible or incomplete Q-only dg1 experiment')
    steps = [row['step'] for row in history]
    if steps != sorted(set(steps)) or steps[0] != 0 or steps[-1] != summary['actual_steps']:
        raise ValueError('Incomplete/non-monotone training history')
    with np.load(root/'evaluation_arrays.npz', allow_pickle=False) as archive:
        arrays={k:archive[k].copy() for k in archive.files}
    for key,value in arrays.items():
        if value.dtype.kind in 'fc' and not np.isfinite(value).all():
            raise ValueError(f'Nonfinite statistics: {key}')
    common._numeric(arrays['target_features'],'target',(60,4,625))
    common._numeric(arrays['labels'],'labels',(60,3))
    train=arrays['splits']=='train'
    if train.sum()!=40 or (arrays['splits']=='test').sum()!=20:
        raise ValueError('Unexpected TRAIN/TEST split')
    for cell in np.unique(arrays['cell_ids']):
        mask=arrays['cell_ids']==cell
        if mask.sum()!=4 or len(np.unique(arrays['splits'][mask]))!=1:
            raise ValueError('Physical condition split overlap')
    if not np.allclose(arrays['train_target_mean'],arrays['target_features'][train].mean(0),rtol=1e-9,atol=1e-12):
        raise ValueError('TRAIN normalization changed')
    if not np.allclose(arrays['train_target_std'],np.sqrt(arrays['target_features'][train].var(0).mean(-1)),rtol=1e-9,atol=1e-12):
        raise ValueError('TRAIN standard deviation changed')
    for name in MODELS:
        for suffix,shape in (('direct_g',(60,4,1)),('direct_full_g',(60,4,1)),('decoded',(60,4,625)),('decoder_jacobian',(60,4,625))):
            common._numeric(arrays[name+'_'+suffix],name+'_'+suffix,shape)
        if not np.allclose(arrays[name+'_direct_g'],.5*(arrays[name+'_g_first']+arrays[name+'_g_second']),rtol=1e-9,atol=1e-12):
            raise ValueError('Half-ensemble estimator changed')
        for split in ('train','test'):
            mask=arrays['splits']==split
            rmse=np.sqrt(((arrays[name+'_decoded'][mask]-arrays['target_features'][mask])**2).mean((0,2)))
            if not np.allclose(rmse,metrics[split][name]['rmse'],rtol=1e-9,atol=1e-12):
                raise ValueError('Saved metrics disagree with arrays')
    return plan,history,metrics,summary,arrays,pins


def calculate(arrays):
    train=arrays['splits']=='train'
    pcs=[common._first_pc(arrays['target_features'][train,k])[0] for k in range(4)]
    result={}
    for name in MODELS:
        g=arrays[name+'_direct_g'][:,:,0]
        groups=common._groups(g,arrays)
        jac=arrays[name+'_decoder_jacobian']
        norms=np.linalg.norm(jac,axis=-1)
        record=dict(condition_statistics=[dict(kappa=row['kappa'],split=row['split'],mean=row['mean'].tolist(),chain_sd=row['chain_sd'].tolist()) for row in groups])
        for split in ('train','test'):
            mask=arrays['splits']==split
            selected=[r for r in groups if r['split']==split]
            record[split]=dict(g_std=g[mask].std(0).tolist(),
                kappa_spearman=[common._spearman([r['kappa'] for r in selected],[r['mean'][k] for r in selected]) for k in range(4)],
                decoder_jacobian_norm_mean=norms[mask].mean(0).tolist(),
                decoder_jacobian_norm_min=norms[mask].min(0).tolist(),
                decoder_jacobian_norm_max=norms[mask].max(0).tolist(),
                mean_abs_cos_local_train_pc1=[None if pcs[k] is None else float(np.mean(np.abs(jac[mask,k]@pcs[k])/np.maximum(norms[mask,k],1e-15))) for k in range(4)],
                reconstruction_sum625=float(np.square(arrays[name+'_decoded'][mask]-arrays['target_features'][mask]).sum(-1).mean()))
        result[name]=record
    return result


def render_report(output):
    root=Path(output).resolve()
    plan,history,metrics,summary,arrays,pins=load_run(root)
    dest=root/'report'
    if dest.exists():raise FileExistsError('Refusing to overwrite report')
    dest.mkdir()
    plt=common._plot_setup()
    computed=calculate(arrays)
    plots=[]
    def save(fig,name):
        common._save(fig,dest/name,plt);plots.append(name)

    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    for k,ax in enumerate(axes.flat):
        for name,label,color in zip(MODELS,DISPLAY,COLORS):
            rows=computed[name]['condition_statistics']
            x=np.array([r['kappa'] for r in rows]);y=np.array([r['mean'][k] for r in rows]);err=np.array([r['chain_sd'][k] for r in rows])
            ax.plot(x,y,color=color,label=label)
            ax.fill_between(x,y-err,y+err,color=color,alpha=.12)
            for split,marker in (('train','o'),('test','x')):
                mask=np.array([r['split']==split for r in rows]);ax.scatter(x[mask],y[mask],color=color,marker=marker,s=24)
        ax.set(title=f'k={k}，边长 {128//2**k}；独立纵轴',xlabel='初始耦合 κ（仅事后评价）',ylabel='Q 识别出的 g')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('固定一维表示的解码器结构对照\n圆点：训练条件；叉号：留出条件；四链均值±链间标准差（非置信区间）')
    save(fig,'Q_g_kappa_by_scale.png')

    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for ax,key,title in zip(axes,('rmse','nrmse'),('留出集绝对 RMSE','留出集 NRMSE：固定原训练目标标准差')):
        for i,(name,label,color) in enumerate(zip(MODELS,DISPLAY,COLORS)):
            ax.bar(np.arange(4)+(i-1.5)*.19,metrics['test'][name][key],width=.18,color=color,label=label)
        ax.bar(np.arange(4)+1.5*.19,metrics['test']['source']['constant_train_mean_'+key],width=.18,color='white',edgecolor='black',hatch='//',label='训练均值常数预测')
        ax.set(xticks=np.arange(4),xticklabels=[f'k={k}' for k in range(4)],ylabel=key.upper(),title=title)
        ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('相同的 625 维局部统计目标；数值越低越好')
    save(fig,'reconstruction_comparison.png')

    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    steps=[r['step'] for r in history]
    for ax,key,title in zip(list(axes.flat)[:3],('q_total','reconstruction_raw','subset_micro'),('共同定义的 Q 总损失','等权重构损失','微观子集一致性损失')):
        for name,label,color in zip(MODELS[1:],DISPLAY[1:],COLORS[1:]):
            ax.plot(steps,[max(r['arms'][name][key],1e-20) for r in history],label=label,color=color)
        ax.set(title=title,xlabel='额外 Q 更新次数',ylabel='固定训练探针损失',yscale='log');ax.legend(fontsize=8);ax.grid(alpha=.2)
    ax=axes.flat[3]
    for k,color in enumerate(('#0072B2','#009E73','#E69F00','#CC79A7')):
        for name,style in zip(MODELS[1:],('--','-')):
            ax.plot(steps,[r['arms'][name]['reconstruction_by_scale'][k] for r in history],ls=style,color=color,label=f'k={k}，{name[0]}')
    ax.set(title='各尺度重构：虚线 A，实线 B',xlabel='额外 Q 更新次数',ylabel='平方误差之和',yscale='log');ax.legend(ncol=2,fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('全部曲线仅来自训练集；固定 300 步终点，不根据留出表现选步数')
    save(fig,'loss_curves.png')

    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for name,label,color in zip(MODELS,DISPLAY,COLORS):
        grid=arrays[name+'_decoder_grid'][:,0]
        jac=arrays[name+'_decoder_grid_jacobian']
        axes[0].plot(grid,np.linalg.norm(jac,axis=-1),label=label,color=color)
        axes[1].plot(range(4),computed[name]['train']['mean_abs_cos_local_train_pc1'],marker='o',label=label,color=color)
    axes[0].set(xlabel='各自训练输入范围内的 g',ylabel='解码器导数范数',title='检查是否依赖过大的解码放大')
    axes[1].set(xlabel='尺度 k',ylabel='导数方向与该尺度主成分的平均绝对余弦',title='局部切线方向：主成分只用训练目标估计',xticks=range(4),ylim=(0,1.05))
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('非线性残差各层谱范数≤2；总解码器全局 Lipschitz 上界≤9\n上界约束不保证可逆性，也不认证物理坐标')
    save(fig,'decoder_geometry.png')
    common._frozen_r(arrays,dest/'R_frozen_reference.png',plt);plots.append('R_frozen_reference.png')
    if pins!={p:common._sha(root/p) for p in INPUTS}:raise RuntimeError('Report inputs changed')
    report=dict(status='complete',source_sha256=pins,comparison=computed,metrics=metrics,
        theory_dim=1,source_Q_steps=3000,additional_Q_steps_per_arm=300,C_updates=0,R_updates=0,
        thermal_nu=None,critical_kappa=None,old_R_valid_for_new_Q=False,plots=plots,
        limitations=[
            '仅 Q 与解码器更新；固定 C 缓存；没有新增物理采样或重新拟合 R。',
            '两臂等权、相同初始编码器和初始解码函数、相同批次；新增残差参数来自固定随机种子。',
            '源 QCR 3000 步与额外 Q 300 步分开；不是完整 QCR 3300 步。',
            '各臂坐标可变化；不能把旧 R 拼接到新 Q 后解释不动点。',
            'g 的分散度、κ 相关性和较低重构误差都不单独证明 RG 正确。',
            '既有 75 项采样警告保留；有限尺寸、单数据、单种子，未认证热化或临界指数。'])
    (dest/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return ['report/'+p for p in [*plots,'comparison.json']]
