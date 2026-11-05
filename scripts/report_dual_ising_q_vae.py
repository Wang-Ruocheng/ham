"""Post-training, parameter-labelled diagnostics for the explicit Q-only pilot."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from experiments.dual_ising_q_vae import write


def polynomial_features(x):
    if x.shape[1]==1:
        z=x[:,0];return np.column_stack([np.ones(len(z)),z,z*z,z*z*z])
    a,b=x.T
    return np.column_stack([np.ones(len(x)),a,b,a*a,a*b,b*b,a**3,a*a*b,a*b*b,b**3])


def readout(train,test,ytrain,ytest):
    center=train.mean(0);scale=np.maximum(train.std(0),1e-8)
    a=polynomial_features((train-center)/scale);b=polynomial_features((test-center)/scale)
    penalty=np.eye(a.shape[1])*1e-3;penalty[0,0]=0
    prediction=b@np.linalg.solve(a.T@a+penalty,a.T@ytrain)
    return (1-((prediction-ytest)**2).sum(0)/((ytest-ytest.mean(0))**2).sum(0)).tolist()


def factor_diagnostics(g,labels):
    cells,inverse=np.unique(labels,axis=0,return_inverse=True)
    rows=[g[inverse==i] for i in range(len(cells))]
    if any(len(r)!=2 for r in rows):raise ValueError('Exactly two independent chains per parameter cell required')
    means=np.stack([r.mean(0) for r in rows]);k1=np.unique(cells[:,0]);k2=np.unique(cells[:,1])
    if len(cells)!=len(k1)*len(k2):raise ValueError('Full evaluation grid required for empirical main effects')
    grid=np.empty((len(k1),len(k2),2))
    for cell,m in zip(cells,means):grid[np.where(k1==cell[0])[0][0],np.where(k2==cell[1])[0][0]]=m
    variance=grid.var(axis=(0,1));denom=np.maximum(variance,1e-30)
    s1=grid.mean(1).var(0)/denom;s2=grid.mean(0).var(0)/denom
    effects=np.column_stack([s1,s2]);interaction=np.maximum(0,1-s1-s2)
    order=[0,1] if effects[0,0]+effects[1,1]>=effects[0,1]+effects[1,0] else [1,0]
    centered=means-means.mean(0);cov=centered.T@centered/len(means)
    eig,axes=np.linalg.eigh(cov)
    differences=np.stack([r[0]-r[1] for r in rows]);noise=((differences@axes)**2).mean(0)/4
    metrics=dict(parameter_main_effects=effects.tolist(),interaction_fraction=interaction.tolist(),
        evaluation_axis_assignment=order,axis_separation_score=float(np.mean([effects[j,order[j]] for j in range(2)])),
        cell_mean_covariance_eigenvalues=eig.tolist(),cell_mean_PCA_fractions=(eig[::-1]/max(eig.sum(),1e-30)).tolist(),
        cell_mean_variance_over_estimated_replica_noise=(eig/np.maximum(noise,1e-30)).tolist(),
        notes=['Main effects are a post-hoc full-grid ANOVA using physical parameter labels, never a training loss.',
               'Only coordinate permutation is used for scoring; no supervised rotation or nonlinear axis correction.',
               'Full covariance rank may come from a curved one-dimensional code; it is not identifiability proof.',
               'Replica-noise ratio assumes independent physical chains; only two chains per cell.'])
    return metrics,cells,means


def report(root,train_labels,test_labels):
    from scripts.report_phi4_hopping_qcr import _setup_plot
    root=Path(root);out=root/'report';out.mkdir(exist_ok=False);plan=json.loads((root/'plan.json').read_text())
    plt=_setup_plot(None);labels=np.concatenate([train_labels,test_labels]);summaries={};all_arrays={};arm_arrays={}
    for arm in plan['arms']:
        name=arm['name'];a=dict(np.load(root/name/'evaluation_arrays.npz'));g=np.concatenate([a['train_g'],a['test_g']])
        metrics=json.loads((root/name/'metrics.json').read_text());factors,cells,means=factor_diagnostics(g,labels)
        metrics.update(factors);metrics['heldout_two_coordinate_cubic_K_R2']=readout(a['train_g'],a['test_g'],train_labels,test_labels)
        metrics['heldout_single_coordinate_cubic_K_R2']=[readout(a['train_g'][:,[j]],a['test_g'][:,[j]],train_labels,test_labels) for j in range(2)]
        metrics['heldout_Spearman_g_K']=[[float(spearmanr(a['test_g'][:,j],test_labels[:,i]).statistic) if np.std(a['test_g'][:,j])>0 else 0. for i in range(2)] for j in range(2)]
        center=a['train_target_histogram'].mean(0)
        constant=float(((a['test_target_histogram']-center)**2).sum(-1).mean())
        metrics['test_constant_TRAIN_histogram_mse']=constant
        metrics['test_reconstruction_improvement_over_constant']=1-metrics['test']['sum_bin_mse']/constant
        metrics['TRAIN_mean_signal_to_posterior_variance']=(np.var(a['train_g'],axis=0)/np.exp(a['train_logvar']).mean(0)).tolist()
        summaries[name]=metrics;arm_arrays[name]=a
        all_arrays[name+'_cell_means']=means;all_arrays[name+'_g']=g
    all_arrays.update(labels=labels,cells=cells,split=np.array(['train']*len(train_labels)+['test']*len(test_labels)))
    fig,axes=plt.subplots(3,2,figsize=(11,14),layout='constrained')
    for row,arm in enumerate(plan['arms']):
        a=arm_arrays[arm['name']]
        for j in range(2):
            ax=axes[row,j]
            ax.scatter(*a['train_g'].T,c=train_labels[:,j],cmap='viridis',vmin=0,vmax=1.3,s=24,alpha=.75,label='训练参数：物理链')
            sc=ax.scatter(*a['test_g'].T,c=test_labels[:,j],cmap='viridis',vmin=0,vmax=1.3,s=32,marker='^',edgecolors='black',linewidths=.25,label='留出参数：物理链')
            ax.set(title=arm['label']+f'，按原始 $K_{j+1}$ 着色',xlabel=r'$g_1=\mu_1$',ylabel=r'$g_2=\mu_2$')
            ax.ticklabel_format(useOffset=False,style='plain');ax.legend(fontsize=8)
            fig.colorbar(sc,ax=ax,label=f'生成数据的原始 $K_{j+1}$')
    fig.suptitle('双 Ising，微观 Q-only：后验均值编码；未训练 R，不绘制 RG 流')
    fig.savefig(out/'latent_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for col,arm in enumerate(plan['arms']):
        for j in range(2):
            ax=axes[j,col];means=all_arrays[arm['name']+'_cell_means']
            sc=ax.scatter(*cells.T,c=means[:,j],s=60,marker='s',cmap='viridis')
            ax.set(title=arm['label']+f'，$g_{j+1}$',xlabel='$K_1$',ylabel='$K_2$');fig.colorbar(sc,ax=ax,label=f'$g_{j+1}$')
    fig.suptitle('参数到编码：每格为两条独立物理链均值；不填补未采样位置')
    fig.savefig(out/'coupling_to_g.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,arm in zip(axes,plan['arms']):
        v=np.array(summaries[arm['name']]['parameter_main_effects'])
        im=ax.imshow(v,vmin=0,vmax=1,cmap='Blues')
        for j in range(2):
            for k in range(2):ax.text(k,j,f'{v[j,k]:.3f}',ha='center',va='center',color='white' if v[j,k]>.6 else 'black')
        ax.set(title=arm['label'],xticks=[0,1],xticklabels=['$K_1$','$K_2$'],yticks=[0,1],yticklabels=['$g_1$','$g_2$'])
    fig.colorbar(im,ax=axes,label='该参数的独立主效应 / 编码总方差')
    fig.suptitle('事后参数主效应：理想轴分离接近对角或反对角；交互作用另报')
    fig.savefig(out/'factor_main_effects.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for col,arm in enumerate(plan['arms']):
        h=json.loads((root/arm['name']/'history.json').read_text());steps=[r['step'] for r in h]
        for split,style,label in [('train','-','训练'),('test','--','留出')]:
            axes[0,col].plot(steps,[r[split]['sum_bin_mse'] for r in h],style,label=label+'重构 MSE')
            axes[1,col].plot(steps,[r[split]['KL'] for r in h],style,label=label+' KL')
        axes[0,col].set(title=arm['label'],xlabel='Q 更新次数',ylabel='概率重构 MSE',yscale='log')
        axes[1,col].set(xlabel='Q 更新次数',ylabel='平均 KL（nats）')
        for row in range(2):axes[row,col].legend(fontsize=8);axes[row,col].grid(alpha=.15)
    fig.suptitle('固定训练终点；未按测试标签、测试损失或解耦评分挑选模型')
    fig.savefig(out/'loss_curves.png',dpi=160);plt.close(fig)
    np.savez_compressed(out/'comparison_arrays.npz',**all_arrays)
    write(out/'comparison.json',dict(status='complete',arms=summaries,
        scope='Q-only, microscopic L36, one training seed, cached full joint 2x2 histograms',
        label_boundary='K labels used only AFTER all three fits, for plots, fixed post-hoc readouts, and main-effect diagnostics',
        no_R_flow_or_fixed_point_claim=True,
        limitations=['本轮改变了 Q 输入与目标，是独立表示能力诊断，不是原 QCR 的严格复现。',
                     '后验采样噪声不计作有效潜变量；实际编码采用后验均值。',
                     '总相关性小不保证保留参数信息，也不保证逐坐标物理解耦。',
                     '完整网格只用于事后诊断；实际 TRAIN 子集不保证参数恰好统计独立。',
                     '高耦合区域的构型统计可能趋同，不能要求从有限样本精确反推所有 K。',
                     '单 seed、有限链与有限尺寸；无重整化流、相变或临界指数认证。']))
