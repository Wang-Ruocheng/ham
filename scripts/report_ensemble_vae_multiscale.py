"""Post-training scale-wise parameter diagnostics; no Q/R fitting or root search."""
import json
import numpy as np
from scipy.stats import spearmanr
from scripts.report_dual_ising_q_vae import factor_diagnostics, readout
from scripts.run_dual_ising_current import write_json


def report_multiscale(arrays, reconstruction, roots, history, out, plt, *, parameter_names=('K_1','K_2')):
    g, labels = arrays['direct_g'], arrays['labels']
    train = arrays['splits']=='train'; test = ~train
    diagnostics = []
    for k in range(g.shape[1]):
        metrics, cells, means = factor_diagnostics(g[:,k],labels)
        metrics.update(scale=k,sites=[36,12,4][k],
            heldout_two_coordinate_cubic_K_R2=readout(g[train,k],g[test,k],labels[train],labels[test]),
            heldout_Spearman_g_K=[[float(spearmanr(g[test,k,j],labels[test,i]).statistic)
                                  if g[test,k,j].std()>0 else None for i in range(2)] for j in range(2)])
        pred, target = reconstruction['decoded'][:,:,k], reconstruction['targets'][:,:,k]
        metrics['probability_reconstruction']={}
        center=target[:,train].mean((0,1))
        for name,selector in [('train',train),('test',test)]:
            mse=float(((pred[:,selector]-target[:,selector])**2).sum(-1).mean())
            constant=float(((center-target[:,selector])**2).sum(-1).mean())
            metrics['probability_reconstruction'][name]=dict(sum_bin_mse=mse,
                constant_TRAIN_histogram_mse=constant,improvement=1-mse/constant if constant else None)
        metrics['nearest_sampled_parameters_to_roots']=[]
        width=np.maximum(g[train,k].std(0),1e-8)
        for number,root in enumerate(roots,1):
            distance=np.linalg.norm((means-np.asarray(root['g']))/width,axis=1)
            nearest=np.argsort(distance)[:3]
            metrics['nearest_sampled_parameters_to_roots'].append(dict(root_id=number,
                sampled_parameters=cells[nearest].tolist(),standardized_distances=distance[nearest].tolist(),
                note='Nearest actual two-chain mean codes only; NOT inferred root couplings or critical points'))
        diagnostics.append(metrics)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for k,ax in enumerate(axes):
        v=np.array(diagnostics[k]['parameter_main_effects']);im=ax.imshow(v,vmin=0,vmax=1,cmap='Blues')
        for j in range(2):
            for i in range(2):ax.text(i,j,f'{v[j,i]:.3f}',ha='center',va='center',color='white' if v[j,i]>.6 else 'black')
        ax.set(title=f'k={k}，格点边长 {[36,12,4][k]}',xticks=[0,1],xticklabels=[f'${n}$' for n in parameter_names],yticks=[0,1],yticklabels=['$g_1$','$g_2$'])
    fig.colorbar(im,ax=axes,label='参数主效应 / 编码总方差')
    fig.suptitle('同一 Q 的多尺度分离：原始坐标，未按尺度旋转／重新排序')
    fig.savefig(out/'q_multiscale_separation.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for k in range(3):
        for row,(key,label) in enumerate([('vae_cross_entropy','跨子集交叉熵'),('vae_KL','后验 KL')]):
            for split,ls in [('train','-'),('test','--')]:
                points=[(h['epoch'],h[f'{split}_{key}_level_{k}']) for h in history]
                axes[row,k].plot(*np.array(points).T,ls,label='训练' if split=='train' else '留出')
            axes[row,k].set(title=f'k={k}',xlabel='联合训练更新次数',ylabel=label+'（nats）');axes[row,k].legend();axes[row,k].grid(alpha=.15)
    fig.suptitle('Q 的实际目标：各尺度等权，交叉熵＋逐渐增至 0.01 的 KL；未使用参数监督')
    fig.savefig(out/'q_vae_losses.png',dpi=170);plt.close(fig)
    write_json(out/'q_multiscale.json',dict(scales=diagnostics,
        labels_used_only_post_training=True,parameter_names=parameter_names,shared_Q_and_decoder=True,
        coordinate_transform_applied=False,encoding='Mean of two half-ensemble posterior means, then four fixed C/subset draws',
        Q_only_comparison_caveat='Joint Q is newly initialized, minibatched and shared over three evolving scales; not a continuation of the full-batch k=0 Q-only run',
        limitations=['参数主效应使用完整网格，包含训练与留出；留出相关／读出另报。',
                     '粗尺度颜色和评价标签始终是原始扫描参数，不是有效耦合。',
                     '同一 Q 的编码无逐尺度坐标校正；分别报告配对评分不等于坐标对应保持不变。',
                     '有限尺寸、两条物理链、单训练种子；四次 C／子集抽样不是独立物理链。']))
    return diagnostics
