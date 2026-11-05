"""Paired post-hoc observable/geometry comparison; labels never enter training."""
import json
from pathlib import Path
import numpy as np
import torch
from recurrent_rg.checkpoints import load_run
from recurrent_rg.terminal_r_fit import fingerprint
from scripts.run_dual_ising_current import write_json


def summary(draws, stats, labels, ntrain):
    errors = ((stats['decoded']-stats['targets'])**2).sum(-1)
    reference = stats['targets'][:, :ntrain]
    center = reference.mean(axis=(0,1))
    variance = ((reference-center)**2).sum(-1).mean(axis=(0,1))
    output = dict(reconstruction={}, geometry=[])
    for split, selection in [('train',slice(0,ntrain)),('test',slice(ntrain,None))]:
        mse = errors[:,selection].mean(axis=(0,1))
        output['reconstruction'][split] = [dict(scale=k, sum_bin_mse=float(mse[k]),
            train_target_sum_variance=float(variance[k]), nrmse=float(np.sqrt(mse[k]/variance[k]))) for k in range(3)]
    direct = draws.mean(0)
    cells, inverse = np.unique(labels,axis=0,return_inverse=True)
    means = np.stack([direct[inverse==i].mean(0) for i in range(len(cells))])
    for k in range(3):
        x=direct[:ntrain,k]; sv=np.linalg.svd(x-x.mean(0),compute_uv=False)
        fractions=sv**2/(sv**2).sum()
        output['geometry'].append(dict(scale=k, train_pca_fractions=fractions.tolist()))
    return output, errors, cells, means


def compare(model, draws, stats, train, test, baseline_root, out, config, closure):
    from scripts.report_dual_ising_current import collect, predictions, closure_metrics
    from scripts.report_phi4_hopping_qcr import _setup_plot
    baseline=load_run(baseline_root); baseline.model.to(next(model.parameters()).device).eval()
    if config['data'] != baseline.config['data'] or config['training'] != baseline.config['training'] or config['loss'] != baseline.config['loss']:
        raise ValueError('Decoder comparison changed data/training/loss configuration')
    allowed={'affine_reconstruction','reconstruction_hidden','reconstruction_hidden_layers','reconstruction_spectral_bound','reconstruction_probability_loss'}
    changed={k for k in set(config['model'])|set(baseline.config['model']) if config['model'].get(k)!=baseline.config['model'].get(k)}
    if changed-allowed: raise ValueError('Unrelated model change: '+str(changed-allowed))
    old_draws,old_stats=collect(baseline.model,train,test,samples=config['training']['sample_batch'],reconstruction=True)
    labels=torch.cat([train.couplings,test.couplings]).numpy(); ntrain=len(train.configurations)
    old,old_errors,cells,old_means=summary(old_draws,old_stats,labels,ntrain)
    new,new_errors,_,new_means=summary(draws,stats,labels,ntrain)
    same_c=fingerprint(model.rg_kernel)==fingerprint(baseline.model.rg_kernel)
    target_max=float(np.max(np.abs(stats['targets']-old_stats['targets'])))
    if same_c and target_max>1e-12: raise AssertionError('Paired C evaluation target mismatch')
    old['closure']=closure_metrics(old_draws,predictions(baseline.model,old_draws),ntrain)
    new['closure']=closure
    norms=[float(torch.linalg.matrix_norm(layer.weight,2)) for layer in model.theory_encoder.distribution_decoder if isinstance(layer,torch.nn.Linear)]
    result=dict(status='complete',baseline=str(baseline_root),affine=old,nonlinear=new,
        changed_model_fields=sorted(changed),identical_final_C=same_c,paired_target_max_difference=target_max,
        spectral_norms=norms,actual_probability_lipschitz_upper_bound=float(.5*np.prod(norms)),
        evaluation_seed=91951,labels_used_for_training=False,
        caveats=['仅一个训练种子；四次 C 重复先在同一物理链内平均。',
                 '两次训练的潜坐标不同，原始潜空间 RMSE 不直接比较；NRMSE 各用自身 TRAIN 目标尺度。',
                 '局域统计 MSE 使用相同概率坐标；只有 C 相同且配对目标一致时才是严格相同目标。',
                 '较小的第二主成分不能单独证明信息消失；较大的第二主成分也不认证两个物理方向。',
                 '谱上界限制解码器放大能力，不保证满秩、不动点或相结构正确。'])
    write_json(out/'decoder_comparison.json',result)
    np.savez_compressed(out/'decoder_comparison_arrays.npz',labels=labels,cells=cells,
        affine_cell_means=old_means,nonlinear_cell_means=new_means,
        affine_reconstruction_errors=old_errors,nonlinear_reconstruction_errors=new_errors)
    plt=_setup_plot(None)
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for title,data,color in [('仿射解码',old,'C0'),('非线性概率解码',new,'C1')]:
        axes[0].plot(range(3),[v['sum_bin_mse'] for v in data['reconstruction']['test']],'-o',color=color,label=title)
        axes[1].plot(range(3),[v['train_pca_fractions'][1] for v in data['geometry']],'-o',color=color,label=title)
        axes[2].plot(range(3),[v['nrmse'] for v in data['closure']['test']],'-o',color=color,label=title)
    axes[0].set(title='留出局域统计重构',xlabel='尺度 k',ylabel='概率重构 MSE（所有格求和）',xticks=range(3))
    axes[1].set(title='TRAIN 第二主成分占比',xlabel='尺度 k',ylabel='方差占比',xticks=range(3),yscale='log')
    axes[2].set(title='末尾 R-only 后留出闭合',xlabel='尺度转移',ylabel='NRMSE（各自 TRAIN 标准差）',xticks=range(3),xticklabels=['0→1','1→2','0→2'])
    for ax in axes: ax.legend(fontsize=8); ax.grid(alpha=.15)
    fig.suptitle('同数据、种子与训练日程；仅更换 Q 解码器及其尺度约束')
    fig.savefig(out/'decoder_comparison.png',dpi=170); plt.close(fig)
    write_json(out/'decoder_comparison_status.json',dict(status='complete'))
