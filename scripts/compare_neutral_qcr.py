"""Audit matched joint-QCR neutral ablations, including physical-flow figures."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recurrent_rg.checkpoints import load_run
from recurrent_rg.kernel_regularizers import uniform_reference_marginal, kernel_neutral_loss
from recurrent_rg.rsmi import kernel_probabilities
from recurrent_rg.evaluation import scalar_flow


def kernel_audit(model):
    c=model.rg_kernel
    width=c.local_encoder[0].in_features
    patterns=(2*((torch.arange(2**width)[:,None]//(2**torch.arange(width)))%2)-1).double()[None]
    with torch.no_grad():
        p=kernel_probabilities(c,patterns)[0,:,1].numpy()
        marginal=uniform_reference_marginal(c).numpy()
        loss=float(kernel_neutral_loss(c))
    positive_count=(patterns[0].numpy()>0).sum(-1)
    iid_p=[.5]
    for _ in range(6):
        value=iid_p[-1]
        iid_p.append(float(np.sum(p*value**positive_count*(1-value)**(width-positive_count))))
    audit = dict(patterns=patterns[0].tolist(),p_plus=p.tolist(),neutral_loss=loss,
                uniform_reference_marginal=marginal.tolist(),iid_uniform_mean_by_scale=(2*np.array(iid_p)-1).tolist(),
                conditional_entropy_bits=float(np.mean(-p*np.log2(np.maximum(p,1e-300))-(1-p)*np.log2(np.maximum(1-p,1e-300)))),
                probability_std_uniform_patterns=float(np.std(p)),
                sign_equivariance_rms=float(np.sqrt(np.mean((p+p[::-1]-1)**2))),
                uniform_input_p_minus_to_minus=float(1-p[0]),uniform_input_p_plus_to_plus=float(p[-1]),
                uniform_input_output_separation=float(abs(p[-1]-p[0])))
    if width==2:
        v=patterns[0].numpy();response=2*p-1
        audit['conditional_mean_walsh_coefficients']=dict(constant=float(response.mean()),
            s1=float(np.mean(response*v[:,0])),s2=float(np.mean(response*v[:,1])),
            s1s2=float(np.mean(response*v.prod(1))))
    return audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=Path('results'))
    parser.add_argument('--output',type=Path,default=Path('results/neutral_comparison_seed91'))
    parser.add_argument('--dimensions',type=int,choices=(1,2),nargs='+',default=[1,2])
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    summary={}
    for dim in dict.fromkeys(args.dimensions):
        runs=[];summary[str(dim)]={}
        for weight in (0,10):
            parent=args.results/f'neutral_ising{dim}d_g{dim}_w{weight}_seed91'
            run=load_run(parent/'qr');folder=parent/'qr'/'report'
            hist=json.loads((parent/'qr'/'history_progress.json').read_text())
            physical=json.loads((folder/'rsmi_physical_audit.json').read_text())
            roots=json.loads((folder/'fixed_points.json').read_text())
            entry=dict(config=run.config,kernel=kernel_audit(run.model),final_losses=hist[-1],
                       physical=physical,roots=roots,
                       timing=json.loads((parent/'timing.json').read_text()),
                       data_sha256=json.loads((parent/'qr'/'manifest.json').read_text())['data_sha256'])
            summary[str(dim)][str(weight)]=entry
            z=np.load(folder/'encodings.npz')
            runs.append(dict(model=run.model,history=hist,entry=entry,z=z,weight=weight))
        # This check fails if anything but the requested weight differs.
        a=json.loads(json.dumps(runs[0]['entry']['config']));b=json.loads(json.dumps(runs[1]['entry']['config']))
        a['loss']['kernel_neutral_weight']=b['loss']['kernel_neutral_weight']
        assert a==b and runs[0]['entry']['data_sha256']==runs[1]['entry']['data_sha256']
        summary[str(dim)]['matched_settings_except_neutral_weight']=True
        fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
        for run in runs:
            h=run['history'];w=run['weight']
            for split,ls in [('train','-'),('test','--')]:
                axes[0].plot([v['epoch'] for v in h],[v[split+'_total'] for v in h],ls,label=f'w={w} {split}')
                axes[1].plot([v['epoch'] for v in h],
                    [v[split+'_total']-w*v.get(split+'_kernel_neutral',0.) for v in h],ls,label=f'w={w} {split}')
            axes[2].plot(run['entry']['kernel']['iid_uniform_mean_by_scale'],'-o',label=f'w={w}')
        axes[0].set(title='Total loss (different objectives)',xlabel='updates',ylabel='loss',yscale='log')
        axes[1].set(title='Common QCR loss, excluding neutral',xlabel='updates',ylabel='loss',yscale='log')
        axes[2].set(title='Exact IID reference drift under C',xlabel='RG step k',ylabel='mean spin',ylim=(-1.05,1.05))
        for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
        fig.savefig(args.output/f'ising{dim}d_loss_and_bias.png',dpi=160);plt.close(fig)
        if dim==1:
            fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
            for col,run in enumerate(runs):
                model,z,w=run['model'],run['z'],run['weight']
                lower,upper=run['entry']['roots']['search_box'];f=scalar_flow(model,lower[0],upper[0])
                ax=axes[0,col];ax.plot(f['grid'],f['mapped'],label='R(g)');ax.plot(f['grid'],f['grid'],'k--',label='identity')
                tr=z['train'][:,:-1,0];ax.axvspan(tr.min(),tr.max(),color='green',alpha=.1,label='training R inputs')
                for item in f['roots']:
                    v=item['g'][0];ax.scatter(v,v,c='green' if item['stability']=='stable' else 'red',s=40)
                ax.set(xlabel='g',ylabel='R(g)',title=f'1D Ising, neutral weight {w}');ax.legend(fontsize=8)
                d=np.concatenate([z['train'],z['test']])[:,:,0];j=np.concatenate([z['train_labels'],z['test_labels']]).reshape(-1);knots=np.unique(j)
                for k in range(d.shape[1]):axes[1,col].plot(knots,[d[j==x,k].mean() for x in knots],label=f'k={k}')
                axes[1,col].set(xlabel='J',ylabel='Q(C^k D)',title='Direct encodings, not R rollouts');axes[1,col].legend(fontsize=8)
            fig.savefig(args.output/'ising1d_R_and_J.png',dpi=170);plt.close(fig)
        else:
            fig,axes=plt.subplots(1,2,figsize=(13,6),layout='constrained')
            for ax,run in zip(axes,runs):
                model,z,w=run['model'],run['z'],run['weight'];roots=run['entry']['roots']
                lower,upper=np.array(roots['search_box']);xx,yy=np.meshgrid(np.linspace(lower[0],upper[0],27),np.linspace(lower[1],upper[1],27));grid=np.column_stack([xx.ravel(),yy.ravel()])
                with torch.no_grad():delta=model.advance_theory(torch.from_numpy(grid)).numpy()-grid
                delta/=np.maximum(np.linalg.norm(delta,axis=1,keepdims=True),1e-12)
                ax.quiver(*grid.T,*delta.T,color='.55',alpha=.65)
                g=np.concatenate([z['train'],z['test']])[:,0];kb=np.concatenate([z['train_labels'],z['test_labels']]);ks=np.unique(kb[:,0]);bs=np.unique(kb[:,1]);b0=bs[np.argmin(abs(bs))]
                sc=ax.scatter(*g.T,c=kb[:,1],cmap='coolwarm',s=8,vmin=-.01,vmax=.01)
                curve=np.array([g[(kb[:,0]==k)&(kb[:,1]==b0)].mean(0) for k in ks]);ax.plot(*curve.T,'k-',lw=1,label='B=0 microscopic curve')
                gc=run['entry']['physical']['critical_microscopic_g'];ax.scatter(*gc,c='magenta',marker='D',s=60,label='Q(Kc,0)')
                for i,item in enumerate(roots['roots'],1):
                    ax.scatter(*item['g'],marker='*',s=100,c='gold',edgecolors='k');ax.annotate(f'F{i}: {item["stability"]}',item['g'],fontsize=8)
                ax.set(xlabel='g1',ylabel='g2',xlim=(lower[0],upper[0]),ylim=(lower[1],upper[1]),title=f'Neutral weight {w}: R(g)-g')
                ax.legend(fontsize=8);fig.colorbar(sc,ax=ax,label='Microscopic B')
            fig.suptitle('Joint QCR, seed 91. Normalized arrows; off-data field is extrapolation.')
            fig.savefig(args.output/'ising2d_R_comparison.png',dpi=170);plt.close(fig)
        # Per-term loss curves, using the actual external weights.
        fig,axes=plt.subplots(1,2,figsize=(13,4),layout='constrained')
        for ax,run in zip(axes,runs):
            cfg=run['entry']['config'];weights=dict(cfg['loss']['weights']);weights['rollout']=cfg['loss']['rollout_weight'];weights['kernel_neutral']=run['weight']
            for name,weight in weights.items():
                if weight:ax.plot([v['epoch'] for v in run['history']], [weight*v.get('test_'+name,0.) for v in run['history']],label=name)
            ax.set(xlabel='updates',ylabel='weighted test loss',yscale='symlog',title=f'neutral weight {run["weight"]}');ax.legend(fontsize=7);ax.grid(alpha=.2)
        fig.savefig(args.output/f'ising{dim}d_test_loss_terms.png',dpi=160);plt.close(fig)
    (args.output/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(args.output)


if __name__=='__main__':main()
