"""Paired scalar QCR assessment; labels diagnose, never select numerical roots."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import brentq
from scipy.stats import spearmanr
from recurrent_rg.checkpoints import load_run
from recurrent_rg.data.ensembles import load_ensembles
from recurrent_rg.terminal_r_fit import fingerprint
from scripts.report_dual_ising_current import collect, predictions, closure_metrics
from scripts.report_ising_b0_local_motif_supervised import crossings
from scripts.report_phi4_hopping_qcr import _setup_plot
from scripts.run_dual_ising_current import write_json


def scalar_flow(model, direct, labels, ntrain):
    source = direct[:ntrain, :2, 0]
    lo, hi = float(source.min()), float(source.max())
    pad = .1*max(hi-lo, 1e-6)
    grid = np.linspace(lo-pad, hi+pad, 2049)
    def mapping(x):
        with torch.no_grad():
            return model.advance_theory(torch.as_tensor(x, dtype=torch.double).reshape(-1,1)).numpy()[:,0]
    delta = mapping(grid)-grid
    roots = []
    if np.max(np.abs(delta)) <= 1e-10:
        return grid, mapping(grid), roots, dict(identity_unresolved=True)
    candidates = []
    for i in range(len(grid)-1):
        if delta[i] == 0: candidates.append(float(grid[i]))
        elif delta[i]*delta[i+1] < 0:
            candidates.append(brentq(lambda v: float(mapping([v])[0]-v), grid[i], grid[i+1], xtol=1e-12))
    knots = np.unique(labels)
    means = np.stack([direct[labels == k,:,0].mean(0) for k in knots])
    for value in candidates:
        if roots and abs(value-roots[-1]['g']) < 1e-7: continue
        z = torch.tensor([[value]], dtype=torch.double, requires_grad=True)
        mapped = model.advance_theory(z)
        slope = float(torch.autograd.grad(mapped.sum(), z)[0].item())
        roots.append(dict(g=value, residual=abs(float(mapped)-value), derivative=slope,
            stability='attracting' if abs(slope)<1-1e-3 else 'repelling' if abs(slope)>1+1e-3 else 'neutral',
            inside_train_input_envelope=lo<=value<=hi,
            K_interpolation_by_scale=[crossings(knots,means[:,k]-value) for k in range(3)]))
    return grid, mapping(grid), roots, dict(train_input_interval=[lo,hi],search_interval=[float(grid[0]),float(grid[-1])],
        method='2049-point sign-change brackets + Brent; tangent roots may be missed; all branches retained',
        physical_critical_point_certified=False, thermal_nu=None)


def assess(root, name):
    run = root/name; final = load_run(run); joint = load_run(run/'checkpoint_last.pt')
    for obj in [final,joint]: obj.model.cpu().eval()
    assert final.model_config.theory_dim == 1
    for part in ['theory_encoder','rg_kernel']:
        assert fingerprint(getattr(final.model,part)) == fingerprint(getattr(joint.model,part))
    train,test = load_ensembles(final.config['data']['cache'],mmap=True)
    draws,recon = collect(final.model,train,test,reconstruction=True)
    ntrain = len(train.configurations); mask = np.arange(draws.shape[1]) < ntrain
    labels = torch.cat([train.couplings,test.couplings]).numpy().reshape(-1)
    direct = draws.mean(0)
    a = dict(direct_draws=draws,direct_g=direct,labels=labels,train_mask=mask,
             target_histograms=recon['targets'],decoded_histograms=recon['decoded'])
    metrics = dict(closure={}, scales=[])
    for tag,model in [('joint',joint.model),('final',final.model)]:
        pred = predictions(model,draws);a[tag+'_predictions']=pred
        metrics['closure'][tag]=closure_metrics(draws,pred,ntrain)
    for k in range(3):
        g=direct[:,k,0];target=recon['targets'][:,:,k];decoded=recon['decoded'][:,:,k]
        center=target[:,mask].mean((0,1));rows={}
        for split,m in [('train',mask),('test',~mask)]:
            error=float(((target[:,m]-decoded[:,m])**2).sum(-1).mean())
            constant=float(((target[:,m]-center)**2).sum(-1).mean())
            rows[split]=dict(sum_bin_mse=error,constant_TRAIN_prediction_mse=constant,
                            improvement_over_constant=1-error/constant if constant else None)
        # Same post-hoc TRAIN-only affine readout for both coordinate systems.
        coef=np.linalg.lstsq(np.column_stack([g[mask],np.ones(mask.sum())]),labels[mask],rcond=None)[0]
        readout=np.column_stack([g[~mask],np.ones((~mask).sum())])@coef
        metrics['scales'].append(dict(scale=k,probability_reconstruction=rows,
            heldout_Spearman=float(spearmanr(g[~mask],labels[~mask]).statistic) if g[~mask].std()>0 else None,
            heldout_affine_K_readout_RMSE=float(np.sqrt(np.mean((readout-labels[~mask])**2))),
            heldout_raw_g_minus_original_K_RMSE=float(np.sqrt(np.mean((g[~mask]-labels[~mask])**2))),
            coarse_labels_note='Original K is a diagnostic, NOT an effective coarse coupling target'))
    grid,mapped,roots,search=scalar_flow(final.model,direct,labels,ntrain)
    a.update(flow_g=grid,flow_R=mapped)
    metrics.update(roots=roots,root_search=search,
        terminal_R=json.loads((run/'r_only/summary.json').read_text()),
        training_summary=json.loads((run/'summary.json').read_text()),
        manifest=json.loads((run/'manifest.json').read_text()))
    return a,metrics


def render(root, arrays, reports, out):
    plt=_setup_plot(None)
    titles={'unsupervised':'无监督','supervised':'微观 K 监督（权重 10）'}
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    for ax,(name,a) in zip(axes,arrays.items()):
        g=a['direct_g'][...,0];K=a['labels'];knots=np.unique(K)
        for k in range(3):
            means=np.array([g[K==v,k].mean() for v in knots]);sd=np.array([g[K==v,k].std(ddof=1) if (K==v).sum()>1 else 0. for v in knots])
            ax.plot(knots,means,label=f'k={k}，L={36//3**k}')
            ax.fill_between(knots,means-sd,means+sd,alpha=.15)
        if name=='supervised':ax.plot(knots,knots,'k:',label='微观监督目标 g=K')
        ax.set(title=titles[name],xlabel='原始耦合 K',ylabel='编码 g');ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.suptitle('同一物理数据／种子；各点为三条物理链均值，阴影为链间标准差')
    fig.savefig(out/'coupling_to_g.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    classes=dict(attracting='吸引点',repelling='排斥点',neutral='近中性点')
    for ax,(name,a) in zip(axes,arrays.items()):
        g=a['direct_g'][...,0];grid=a['flow_g'];ax.plot(grid,a['flow_R'],color='C4',label='R(g)');ax.plot(grid,grid,'k--',label='恒等映射')
        for k,marker in [(0,'o'),(1,'^')]:
            sc=ax.scatter(g[:,k],g[:,k+1],c=a['labels'],s=14,cmap='viridis',marker=marker,label=f'k={k}→{k+1}')
        for i,r in enumerate(reports[name]['roots'],1):
            ax.scatter(r['g'],r['g'],marker='*',s=90,color='red')
            ax.annotate(f'{i} {classes[r["stability"]]}'+('（外推）' if not r['inside_train_input_envelope'] else ''),
                        (r['g'],r['g']),xytext=(3,8),textcoords='offset points',fontsize=8)
        ax.set(title=titles[name],xlabel='输入 g',ylabel='R(g)／直接编码的下一尺度 g');ax.legend(fontsize=8);ax.grid(alpha=.15)
        fig.colorbar(sc,ax=ax,label='生成时的原始 K')
    fig.suptitle('末尾 R-only 后的共享映射；数值不动点不等于已认证物理临界点')
    fig.savefig(out/'R_flow.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    groups=[('Q 表示与监督',[('separation','交叉熵＋KL'),('weighted_coupling_supervision','加权微观监督')]),
            ('C 粗粒化',[('rsmi','负互信息'),('kernel_noncollapse','方差下限'),('kernel_sign','保色')]),
            ('R 闭环',[('all_start_anchor','全起点锚定'),('commutator','一步闭合'),('rollout','多步闭合')])]
    for row,name in enumerate(arrays):
        history=json.loads((root/name/'history_progress.json').read_text())
        for ax,(title,terms) in zip(axes[row],groups):
            for split,ls in [('train','-'),('test','--')]:
                for term,label in terms:
                    points=[(h['epoch'],h[split+'_'+term]) for h in history if split+'_'+term in h]
                    if points:ax.plot(*np.array(points).T,ls,label=label+('（训练）' if split=='train' else '（留出）'))
            ax.set(title=titles[name]+'：'+title,xlabel='联合更新次数',ylabel='诊断量',yscale='symlog');ax.legend(fontsize=6);ax.grid(alpha=.15)
    fig.savefig(out/'loss_curves.png',dpi=170);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);root=p.parse_args().root
    torch.set_num_threads(1);out=root/'report';out.mkdir(exist_ok=False)
    arrays={};reports={}
    for name in ['unsupervised','supervised']:
        arrays[name],reports[name]=assess(root,name)
        np.savez_compressed(out/(name+'_arrays.npz'),**arrays[name])
    left,right=[reports[n]['manifest'] for n in reports]
    assert left['data_sha256']==right['data_sha256']
    assert left['config']==right['config']
    assert not left['labels_used_in_training'] and right['q_supervision']['weight']==10
    assert left['training_start_module_sha256']==right['training_start_module_sha256']
    # CUDA parallel reductions can differ at roundoff despite paired RNG and
    # independent gradients. Record actual differences, never claim bitwise identity.
    kernels=[load_run(root/n).model.rg_kernel for n in arrays]
    max_difference=max(float((v-kernels[1].state_dict()[key]).abs().max())
                       for key,v in kernels[0].state_dict().items() if v.is_floating_point())
    c_comparison=dict(bitwise_equal=fingerprint(kernels[0])==fingerprint(kernels[1]),
        max_absolute_parameter_difference=max_difference,
        same_initialization_and_objective=True,labels_do_not_update_C=True)
    render(root,arrays,reports,out)
    write_json(out/'report.json',dict(status='complete',arms=reports,paired_initialization=True,C_comparison=c_comparison,
        limitations=['单个训练种子，固定监督权重；不宣称监督一般优于无监督。',
        '潜坐标尺度不同，绝对闭环 MSE 不能直接比较；同时提供各自 TRAIN 目标标准差归一化误差与共同物理统计重构误差。',
        '粗尺度的 K 只是原始制备参数；并未监督有效耦合，不把 g=K 强加到粗尺度。',
        '四次 C／子集抽样不是独立物理链；链内相关、有限尺寸与训练随机性未被充分量化。',
        '根搜索有界且可能漏掉切触根；保留所有反插值分支，不依据已知 Kc 筛选。'],thermal_nu=None))


if __name__=='__main__':main()
