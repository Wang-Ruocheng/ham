"""Model-free Chinese report for the fixed-dg1 fresh encoder comparison."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from scripts import report_phi4_hopping_qonly_balance as common

MODELS=('source','A_patch','B_statistics','C_hierarchical')
DISPLAY=('历史源 Q（非等预算）','A：局部块编码','B：统计量直接编码','C：逐构型卷积编码')
COLORS=('#888888','#0072B2','#D55E00','#009E73')
INPUTS=('plan.json','history.json','metrics.json','summary.json','evaluation_arrays.npz')


def load_run(root):
    root=Path(root)
    pins={p:common._sha(root/p) for p in INPUTS}
    plan,history,metrics,summary=[json.loads((root/p).read_text()) for p in INPUTS[:4]]
    if (summary.get('status')!='complete' or summary.get('theory_dim')!=1
        or summary.get('C_updates')!=0 or summary.get('R_updates')!=0
        or summary.get('labels_used_in_training') is not False
        or summary.get('test_used_in_training') is not False
        or summary.get('all_tested_encoders_fresh') is not True
        or summary.get('old_R_valid_for_new_Q') is not False
        or summary.get('thermal_nu') is not None):
        raise ValueError('Incompatible scalar Q-only experiment')
    if [r['step'] for r in history]!=list(range(0,301,50)) or summary['actual_steps']!=300:
        raise ValueError('Incomplete fixed-endpoint history')
    for row in history:
        if set(row['arms'])!=set(MODELS[1:]):raise ValueError('Missing history arm')
        for data in row['arms'].values():
            if len(data['reconstruction_by_scale'])!=4:raise ValueError('Incomplete scale history')
    with np.load(root/'evaluation_arrays.npz',allow_pickle=False) as z:
        arrays={k:z[k].copy() for k in z.files}
    for key,value in arrays.items():
        if value.dtype.kind in 'fc' and not np.isfinite(value).all():raise ValueError('Nonfinite '+key)
    common._numeric(arrays['target_features'],'target',(60,4,625))
    common._numeric(arrays['labels'],'labels',(60,3))
    train=arrays['splits']=='train'
    if train.sum()!=40 or (arrays['splits']=='test').sum()!=20:raise ValueError('Wrong partition')
    for cell in np.unique(arrays['cell_ids']):
        mask=arrays['cell_ids']==cell
        if mask.sum()!=4 or len(np.unique(arrays['splits'][mask]))!=1:raise ValueError('Condition split overlap')
    for key,expected in [('train_target_mean',arrays['target_features'][train].mean(0)),
                         ('train_target_std',np.sqrt(arrays['target_features'][train].var(0).mean(-1)))]:
        if not np.allclose(arrays[key],expected,rtol=1e-9,atol=1e-12):raise ValueError('TRAIN normalization changed')
    for name in MODELS:
        for suffix,shape in [('direct_g',(60,4,1)),('direct_full_g',(60,4,1)),('decoded',(60,4,625))]:
            common._numeric(arrays[name+'_'+suffix],name+'_'+suffix,shape)
        if name!='source':
            common._numeric(arrays[name+'_decoder_jacobian'],name+' jacobian',(60,4,625))
            if not np.allclose(arrays[name+'_direct_g'],.5*(arrays[name+'_g_first']+arrays[name+'_g_second']),rtol=1e-9,atol=1e-12):
                raise ValueError('Half-ensemble estimate changed')
        for split in ('train','test'):
            mask=arrays['splits']==split
            rmse=np.sqrt(((arrays[name+'_decoded'][mask]-arrays['target_features'][mask])**2).mean((0,2)))
            constant=np.sqrt(((arrays['train_target_mean']-arrays['target_features'][mask])**2).mean((0,2)))
            for key,expected in [('rmse',rmse),('nrmse',rmse/arrays['train_target_std']),
                                 ('constant_train_mean_rmse',constant),('constant_train_mean_nrmse',constant/arrays['train_target_std'])]:
                if not np.allclose(expected,metrics[split][name][key],rtol=1e-9,atol=1e-12):raise ValueError('Saved metrics disagree')
    return plan,history,metrics,summary,arrays,pins


def calculate(arrays):
    train=arrays['splits']=='train'
    pcs=[common._first_pc(arrays['target_features'][train,k])[0] for k in range(4)]
    result={}
    for name in MODELS:
        g=arrays[name+'_direct_g'][:,:,0];groups=common._groups(g,arrays)
        record=dict(condition_statistics=[dict(kappa=r['kappa'],split=r['split'],mean=r['mean'].tolist(),chain_sd=r['chain_sd'].tolist()) for r in groups])
        for split in ('train','test'):
            mask=arrays['splits']==split;selected=[r for r in groups if r['split']==split]
            record[split]=dict(g_std=g[mask].std(0).tolist(),
                kappa_spearman=[common._spearman([r['kappa'] for r in selected],[r['mean'][k] for r in selected]) for k in range(4)],
                half_mean_vs_full_rmse=np.sqrt(((arrays[name+'_direct_g'][mask]-arrays[name+'_direct_full_g'][mask])**2).mean((0,2))).tolist(),
                reconstruction_sum625=float(np.square(arrays[name+'_decoded'][mask]-arrays['target_features'][mask]).sum(-1).mean()))
            if name!='source':
                jac=arrays[name+'_decoder_jacobian'][mask];norm=np.linalg.norm(jac,axis=-1)
                record[split].update(decoder_norm_min=norm.min(0).tolist(),decoder_norm_max=norm.max(0).tolist(),
                    mean_abs_cos_local_train_pc1=[None if pcs[k] is None else float(np.mean(np.abs(jac[:,k]@pcs[k])/np.maximum(norm[:,k],1e-15))) for k in range(4)])
        result[name]=record
    return result


def render_report(output):
    root=Path(output).resolve();plan,history,metrics,summary,arrays,pins=load_run(root)
    dest=root/'report'
    if dest.exists():raise FileExistsError('Refusing to overwrite report')
    dest.mkdir();plt=common._plot_setup();computed=calculate(arrays);plots=[]
    def save(fig,name):common._save(fig,dest/name,plt);plots.append(name)
    fig,axes=plt.subplots(4,4,figsize=(16,12),constrained_layout=True)
    for k in range(4):
        for j,(name,label,color) in enumerate(zip(MODELS,DISPLAY,COLORS)):
            ax=axes[k,j]
            rows=computed[name]['condition_statistics'];x=np.array([r['kappa'] for r in rows]);y=np.array([r['mean'][k] for r in rows]);sd=np.array([r['chain_sd'][k] for r in rows])
            ax.plot(x,y,label=label,color=color);ax.fill_between(x,y-sd,y+sd,color=color,alpha=.1)
            for split,marker in [('train','o'),('test','x')]:
                mask=np.array([r['split']==split for r in rows]);ax.scatter(x[mask],y[mask],marker=marker,color=color,s=22)
            ax.set(title=f'{label}\nk={k}，边长 {128//2**k}',xlabel='初始耦合 κ（仅评价）',ylabel='各自坐标 g');ax.grid(alpha=.2)
    fig.suptitle('一维表示的编码器对照：三组均从随机初始化训练 300 步\n每格独立纵轴；圆点：训练；叉号：留出；带宽：四链标准差，非置信区间；不同模型坐标不要求一致')
    save(fig,'Q_g_kappa_by_scale.png')
    fig,axes=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
    for ax,key,title in zip(axes,['rmse','nrmse'],['留出集绝对 RMSE','留出集 NRMSE：原训练目标标准差']):
        for i,(name,label,color) in enumerate(zip(MODELS,DISPLAY,COLORS)):
            ax.bar(np.arange(4)+(i-2)*.16,metrics['test'][name][key],width=.15,label=label,color=color)
        ax.bar(np.arange(4)+2*.16,metrics['test']['source']['constant_train_mean_'+key],width=.15,color='white',edgecolor='black',hatch='//',label='训练均值常数预测')
        ax.set(xticks=range(4),xticklabels=[f'k={k}' for k in range(4)],title=title,ylabel=key.upper());ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.suptitle('相同 625 维局部统计目标；越低越好；历史源 Q 仅作参考')
    save(fig,'reconstruction_comparison.png')
    fig,axes=plt.subplots(2,2,figsize=(14,9),constrained_layout=True);steps=[r['step'] for r in history]
    for ax,key,title in zip(list(axes.flat)[:3],['q_total','reconstruction_raw','subset_micro'],['共同定义的 Q 总损失','等尺度重构损失','微观子集一致性损失']):
        for name,label,color in zip(MODELS[1:],DISPLAY[1:],COLORS[1:]):ax.plot(steps,[max(r['arms'][name][key],1e-20) for r in history],label=label,color=color)
        ax.set(title=title,xlabel='从随机初始化的 Q 更新次数',ylabel='固定训练探针损失',yscale='log');ax.legend(fontsize=8);ax.grid(alpha=.2)
    ax=axes.flat[3]
    for k,color in enumerate(['#0072B2','#D55E00','#009E73','#CC79A7']):
        for name,style in zip(MODELS[1:],['--','-',':']):ax.plot(steps,[r['arms'][name]['reconstruction_by_scale'][k] for r in history],ls=style,color=color,label=f'k={k}，{name[0]}')
    ax.set(title='各尺度重构：虚线 A，实线 B，点线 C',xlabel='Q 更新次数',ylabel='625 项平方误差之和',yscale='log');ax.legend(ncol=3,fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('只用训练集；固定 300 步终点；无留出集选模或早停')
    save(fig,'loss_curves.png')
    fig,axes=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
    for name,label,color in zip(MODELS[1:],DISPLAY[1:],COLORS[1:]):
        v=computed[name]['train'];lo=np.array(v['decoder_norm_min']);hi=np.array(v['decoder_norm_max'])
        axes[0].plot(range(4),(hi+lo)/2,marker='o',color=color,label=label);axes[0].fill_between(range(4),lo,hi,color=color,alpha=.12)
        axes[1].plot(range(4),v['mean_abs_cos_local_train_pc1'],marker='o',color=color,label=label)
    axes[0].set(title='训练数据处解码器导数范数范围',ylabel='导数范数（范围中点与最小—最大带）')
    axes[1].set(title='解码切线与同尺度训练主成分的方向对齐',ylabel='平均绝对余弦',ylim=(0,1.05))
    for ax in axes:ax.set(xlabel='尺度 k',xticks=range(4));ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('共同解码器结构；全局 Lipschitz 上界≤9；不保证可逆或物理正确')
    save(fig,'decoder_geometry.png')
    common._frozen_r(arrays,dest/'R_frozen_reference.png',plt);plots.append('R_frozen_reference.png')
    if pins!={p:common._sha(root/p) for p in INPUTS}:raise RuntimeError('Report inputs changed')
    report=dict(status='complete',source_sha256=pins,comparison=computed,metrics=metrics,plots=plots,theory_dim=1,
        fresh_Q_steps_per_arm=300,source_reference_QCR_steps=3000,C_updates=0,R_updates=0,old_R_valid_for_new_Q=False,
        thermal_nu=None,critical_kappa=None,fixed_point_search_performed=False,
        limitations=['三组随机编码器，共同随机解码器，相同批次与目标；结构容量和计算量不相等。',
        '历史 QCR 3000 步仅作参考；不是这些随机编码器的等预算对照。',
        '局部统计直接编码仅用各自子集的统计量，不输入 κ 或尺度标签。',
        '固定 C 数据、固定一维输出；没有训练 R，不将旧 R 与新 Q 拼接解释不动点。',
        '统计重构或参数相关性改善不等于正确 RG；局部目标仍不能保证长程信息。',
        '保留 75 项采样警告；单数据、单种子、有限尺寸、未认证热化。'])
    (dest/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return ['report/'+p for p in [*plots,'comparison.json']]
