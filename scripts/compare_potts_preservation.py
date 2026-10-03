"""Paired Potts preservation/neutral ablation, using completed scalar reports."""
import argparse
import json
from itertools import permutations
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recurrent_rg.checkpoints import load_run
from recurrent_rg.evaluation import scalar_flow


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--ablation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--term',choices=('kernel_sign_weight','kernel_neutral_weight'),default='kernel_sign_weight')
    args=p.parse_args();torch.set_num_threads(2)
    label='Neutral' if args.term=='kernel_neutral_weight' else 'Preservation'
    term='kernel_neutral' if args.term=='kernel_neutral_weight' else 'kernel_sign'
    manifests=[json.loads((path/'manifest.json').read_text()) for path in (args.baseline,args.ablation)]
    configs=[json.loads(json.dumps(m['config'])) for m in manifests]
    weights=[c['loss'].pop(args.term,0.) for c in configs]
    if configs[0]!=configs[1] or manifests[0]['data_sha256']!=manifests[1]['data_sha256']:
        raise ValueError(f'Paired comparison requires identical configuration/data except {args.term}')
    args.output.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    lossfig,lossaxes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    results=[]
    for col,(path,weight) in enumerate(zip((args.baseline,args.ablation),weights)):
        run=load_run(path);out=path/'full_report'
        a=np.load(out/'encodings.npz');audit=json.loads((out/'potts_audit.json').read_text())
        metrics=json.loads((out/'metrics.json').read_text());history=json.loads((path/'history_progress.json').read_text())
        g=np.concatenate([a['train'],a['test']])[:,:,0]
        K=np.concatenate([a['train_labels'],a['test_labels']]).ravel();knots=np.unique(K)
        means=np.stack([g[K==v].mean(0) for v in knots])
        lo,hi=audit['flow']['search_range'];flow=scalar_flow(run.model,lo,hi,points=3001)
        ax=axes[0,col];ax.plot(flow['grid'],flow['mapped'],label='R(g)')
        ax.plot(flow['grid'],flow['grid'],'k--',label='Identity')
        ax.axvspan(*audit['observed_R_input_range'],alpha=.10,color='green',label='Observed R-input range')
        ax.axvline(audit['microscopic_g_at_Kc'],color='purple',ls=':',label='Q(D) at Kc')
        for i,r in enumerate(flow['roots']):
            z=r['g'][0];ev=r['eigenvalues'][0]['real']
            ax.scatter(z,z,color='green' if r['stability']=='stable' else 'red')
            ax.annotate(f"F{i+1}: {r['stability']}\nR'={ev:.3f}",(z,z),xytext=(5,8),textcoords='offset points',fontsize=8)
        ax.set(xlabel='g (run-specific coordinate)',ylabel='R(g)',title=f'{label} weight = {weight:g}')
        ax.legend(fontsize=8);ax.grid(alpha=.2)
        ax=axes[1,col]
        for k in range(g.shape[1]):ax.plot(knots,means[:,k],label=f'Q(C^{k} D)')
        ax.axvline(audit['theoretical_Kc'],color='purple',ls=':',label='Theoretical Kc')
        ax.set(xlabel='K=J/T',ylabel='g',title='Direct coarse-scale encodings');ax.legend(fontsize=8);ax.grid(alpha=.2)
        for ax,split in zip(lossaxes,('train','test')):
            # Compare the objective terms present in BOTH experiments.
            values=[h[f'{split}_total']-weight*h.get(f'{split}_{term}',0.) for h in history]
            ax.plot([h['epoch'] for h in history],values,label=f'{label} weight={weight:g}')
            ax.set(xlabel='updates',ylabel=f'common loss ({label.lower()} excluded)',title=split)
            ax.set_yscale('log');ax.grid(alpha=.2);ax.legend(fontsize=8)
        last=history[-1]
        patterns=np.asarray(metrics['kernel']['patterns']);probs=np.asarray(metrics['kernel']['probabilities'])
        uniform=np.stack([probs[(patterns==a).all(1)][0] for a in range(3)])
        choices=list(permutations(range(3)))
        scores=[float(uniform[np.arange(3),list(p)].mean()) for p in choices]
        best=int(np.argmax(scores))
        mean=uniform.mean(0)
        information=float((uniform*np.log(np.maximum(uniform,1e-300)/np.maximum(mean[None,:],1e-300))).sum()/3)
        iid=[np.full(3,1/3)]
        for _ in range(6):
            pattern_weights=np.prod(iid[-1][patterns.astype(int)],axis=1)
            iid.append(pattern_weights@probs)
        conditional_entropy=float(-(probs*np.log(np.maximum(probs,1e-300))).sum(1).mean())
        marginal=probs.mean(0)
        reference_mi=float(-(marginal*np.log(np.maximum(marginal,1e-300))).sum()-conditional_entropy)
        from recurrent_rg.kernel_regularizers import kernel_neutral_loss
        results.append(dict(run=str(path),**{args.term:weight},neutral_loss=float(kernel_neutral_loss(run.model.rg_kernel).detach()),
            iid_reference_color_probabilities_by_scale=np.array(iid).tolist(),
            uniform_pattern_mean_conditional_entropy_nats=conditional_entropy,
            uniform_reference_input_output_MI_nats=reference_mi,
            final_loss={s:last[f'{s}_total'] for s in ('train','test')},
            common_loss={s:last[f'{s}_total']-weight*last.get(f'{s}_{term}',0.) for s in ('train','test')},
            closure=metrics['closure'],uniform_color_transition_matrix=uniform.tolist(),
            best_uniform_color_permutation=list(choices[best]),best_permutation_probability=scores[best],
            uniform_color_information_nats=information,
            audit=audit))
    fig.savefig(args.output/'R_and_K_comparison.png',dpi=170);plt.close(fig)
    lossfig.savefig(args.output/'common_loss_comparison.png',dpi=170);plt.close(lossfig)
    (args.output/'comparison.json').write_text(json.dumps(dict(results=results,
        note=f'Same seed/data; g coordinates are run-specific. Common loss excludes {label.lower()} for both. Root searches are bounded.'),indent=2)+'\n')
    print(args.output)


if __name__=='__main__':main()
