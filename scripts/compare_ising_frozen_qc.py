"""Matched cached-ensemble audit of joint800, joint2400 and frozen-QC2400."""
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recurrent_rg.checkpoints import load_run
from recurrent_rg.runner import prepared_data, evaluate_run
from recurrent_rg.config import RunConfig
from recurrent_rg.evaluation import collect_trajectories, closure_metrics, scalar_flow
from recurrent_rg.kernel_regularizers import uniform_block_probabilities
from archive.qcr_ablations.kernel_distinct import kernel_distinct_loss


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--separate',action='store_true',help='Include simultaneous separated-gradient training')
    parser.add_argument('--distinct',action='store_true',help='Read-only comparison of historical distinct-loss runs; no training')
    parser.add_argument('--rsmi',action='store_true',help='Compare joint, separate and RSMI separate')
    parser.add_argument('--fixed',action='store_true',help='1D comparison: joint, learned-RSMI C, fixed positive decimation')
    args=parser.parse_args()
    if args.distinct:args.separate=True
    if args.rsmi:args.separate=True
    if args.fixed:args.separate=True
    if args.rsmi and args.distinct:parser.error('Choose a single intervention comparison')
    torch.set_num_threads(2)
    out=Path('results/ising_separateQCR_comparison_seed91' if args.separate else 'results/ising_frozenQC800_comparison_seed91')
    if args.distinct:out=Path('results/ising_distinct_comparison_seed91')
    if args.rsmi:out=Path('results/ising_rsmiSeparate_comparison_seed91')
    if args.fixed:out=Path('results/ising1d_fixedPositive_comparison_seed91')
    out.mkdir(parents=True,exist_ok=True)
    reports={}
    for dim,gdim in ([(1,1)] if args.fixed else [(1,1),(2,2)]):
        source=Path(f'results/neutral_ising{dim}d_g{gdim}_w0_seed91/qr')
        frozen=Path(f'results/ising{dim}d_g{gdim}_freezeQC800_R2400_seed91/qr')
        cfg=RunConfig.from_dict(load_run(source,800).config)
        tx,vx=prepared_data(cfg)
        opts=dict(samples=cfg.evaluation.samples,max_rg_steps=cfg.training.max_rg_steps,
                  spin_flip_augment=cfg.data.spin_flip_augment)
        variants=[('joint800',source,800),('joint2400',source,2400),('frozen2400',frozen,2400)]
        separate=Path(f'results/ising{dim}d_g{gdim}_separateQCR2400_seed91/qr')
        if args.separate:variants.append(('separate2400',separate,2400))
        distinct=Path(f'results/ising{dim}d_g{gdim}_separateDistinct1m0p8_2400_seed91/qr')
        if args.distinct:variants=[('joint2400',source,2400),('separate2400',separate,2400),('distinct2400',distinct,2400)]
        rsmi=Path(f'results/ising{dim}d_g{gdim}_rsmiSeparate_w2_2400_seed91/qr')
        if args.rsmi:variants=[('joint2400',source,2400),('separate2400',separate,2400),('rsmi2400',rsmi,2400)]
        fixed=Path('results/ising1d_g1_fixedPositiveSeparate2400_seed91/qr')
        if args.fixed:variants=[('joint2400',source,2400),('rsmi2400',rsmi,2400),('fixed2400',fixed,2400)]
        fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
        paths=[('joint',source),('frozen QC',frozen)]
        if args.separate:paths.append(('separate QC/R',separate))
        if args.distinct:paths=[('joint',source),('separate QC/R',separate),('separate + distinct',distinct)]
        if args.rsmi:paths=[('joint',source),('separate QC/R',separate),('separate + RSMI',rsmi)]
        if args.fixed:paths=[('joint',source),('RSMI separate',rsmi),('fixed C separate',fixed)]
        for label,path in paths:
            h=json.loads((path/'history_progress.json').read_text())
            h=[r for r in h if r['epoch']>=(1 if args.separate else 800)]
            for split,style in [('train','-'),('test','--')]:
                axes[0].plot([r['epoch'] for r in h],[r.get(f'{split}_qcr_total',r[f'{split}_total']) if args.fixed else r[f'{split}_total'] for r in h],style,label=f'{label} {split}')
                axes[1].plot([r['epoch'] for r in h],[5*(r[f'{split}_commutator']+r[f'{split}_rollout']) for r in h],style,label=f'{label} {split}')
        for ax,title in zip(axes,['Full objective','R loss: 5 one-step + 5 rollout']):
            ax.set(title=title,xlabel='Training update',ylabel='Loss');ax.legend(fontsize=8);ax.grid(alpha=.2)
        if args.fixed:axes[0].set_title('QCR loss excluding RSMI')
        fig.savefig(out/f'ising{dim}d_loss.png',dpi=170);plt.close(fig)
        if args.rsmi:
            h=json.loads((rsmi/'history_progress.json').read_text())
            fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
            for split in ('train','test'):
                ax.plot([r['epoch'] for r in h],[r[f'{split}_rsmi_bits'] for r in h],label=f'{split} InfoNCE')
                ax.plot([r['epoch'] for r in h],[r[f'{split}_shuffled_rsmi_bits'] for r in h],'--',label=f'{split} shuffled')
            ax.set(xlabel='Training update',ylabel='Empirical InfoNCE (bits)',title='RSMI diagnostic, not exact MI')
            ax.legend();ax.grid(alpha=.2);fig.savefig(out/f'ising{dim}d_rsmi.png',dpi=170);plt.close(fig)
        if dim==1:fig,axes=plt.subplots(2,len(variants),figsize=(5*len(variants),8),layout='constrained')
        results={}
        for col,(name,path,step) in enumerate(variants):
            run=load_run(path,step);m=run.model
            tr=collect_trajectories(m,tx.configurations,**opts)
            te=collect_trajectories(m,vx.configurations,**opts)
            if name=='joint800':
                initial={k:v.clone() for k,v in m.state_dict().items() if k.startswith(('theory_encoder.','rg_kernel.'))}
                original=te['direct'].copy()
            if name=='frozen2400':
                for key,value in initial.items():assert torch.equal(value,m.state_dict()[key]),key
                np.testing.assert_array_equal(original,te['direct'])
            d=np.concatenate([tr['direct'],te['direct']]);labels=np.concatenate([tx.couplings.numpy(),vx.couplings.numpy()]).reshape(len(d),-1)
            results[name]=dict(closure=closure_metrics(te,reference_direct=tr['direct']),
                covariance_eigenvalues=[np.linalg.eigvalsh(np.atleast_2d(np.cov(d[:,k].T,bias=True))).tolist() for k in range(d.shape[1])])
            with torch.no_grad():
                if run.config['model']['fixed_kernel']:
                    results[name]['uniform_block_probabilities']=[[1.,0.],[0.,1.]]
                    results[name]['distinct_loss_margin_0p8']=0.
                else:
                    results[name]['uniform_block_probabilities']=uniform_block_probabilities(m.rg_kernel).tolist()
                    results[name]['distinct_loss_margin_0p8']=float(kernel_distinct_loss(m.rg_kernel))
            logged=next(row for row in json.loads((path/'history_progress.json').read_text()) if row['epoch']==step)
            results[name]['checkpoint_logged_loss']={k:v for k,v in logged.items()
                if k in ['epoch','train_total','test_total','test_anchor','test_coarse_consistency','test_kernel_noncollapse',
                         'test_separation','test_commutator','test_rollout','test_kernel_distinct',
                         'test_qcr_total','test_rsmi','test_rsmi_bits','test_shuffled_rsmi_bits']}
            np.savez_compressed(out/f'ising{dim}d_{name}.npz',**te,labels=vx.couplings.numpy())
            if dim==1:
                lo,hi=d[:,:-1,0].min(),d[:,:-1,0].max();pad=max(.4,.4*(hi-lo))
                flow=scalar_flow(m,float(lo-pad),float(hi+pad),points=6001)
                for root in flow['roots']:root['inside_observed_inputs']=bool(lo<=root['g'][0]<=hi)
                results[name].update(roots=flow['roots'],search_range=flow['search_range'],observed_inputs=[float(lo),float(hi)])
                ax=axes[0,col];ax.plot(flow['grid'],flow['mapped'],label='R(g)');ax.plot(flow['grid'],flow['grid'],'k--',label='identity')
                ax.axvspan(lo,hi,color='green',alpha=.1,label='Observed inputs')
                for r in flow['roots']:ax.scatter(r['g'][0],r['g'][0],c='green' if r['stability']=='stable' else 'red')
                ax.set(title=name,xlabel='g',ylabel='R(g)');ax.legend(fontsize=8)
                js=np.unique(labels[:,0]);means=np.stack([d[labels[:,0]==j].mean(0) for j in js])
                for k in range(d.shape[1]):axes[1,col].plot(js,means[:,k,0],label=f'k={k}')
                axes[1,col].set(xlabel='J',ylabel='Q(C^k D)');axes[1,col].legend(fontsize=8)
            else:
                dest=out/f'ising2d_{name}'
                evaluate_run(path,checkpoint=step,output=dest)
                results[name]['fixed_points']=json.loads((dest/'fixed_points.json').read_text())
        if dim==1:fig.savefig(out/'ising1d_R_and_J.png',dpi=170);plt.close(fig)
        results['contract']='Cached held-out ensembles, identical sampling seeds; frozen QC tensors and direct targets verified exactly equal to update800. Coupling labels used only for diagnostics.'
        if args.distinct:results['contract']='Cached held-out ensembles, paired evaluation RNG; final2400, one seed. Coupling labels used only for diagnostics. New distinct term changes the full objective; R loss remains the same.'
        if args.rsmi:results['contract']='Cached held-out ensembles, paired evaluation RNG; final2400, one seed. RSMI replaces C penalty and adds an auxiliary critic, so full objectives are not directly comparable. R loss unchanged. Couplings for diagnostics only.'
        if args.fixed:results['contract']='1D fixed positive decimation; separate Q/R gradients, no critic or C auxiliaries. Same seed/data/Q-R initial weights and2400 updates. Plotted QCR loss excludes RSMI for comparability. Couplings for diagnostics only.'
        reports[f'ising{dim}d']=results
    (out/'comparison.json').write_text(json.dumps(reports,indent=2)+'\n')
    print(out,flush=True)


if __name__=='__main__':main()
