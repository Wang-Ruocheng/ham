"""Additional scalar Potts plots; all coupling labels are diagnostic only."""
import argparse
import json
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
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--checkpoint',type=int)
    args=p.parse_args();torch.set_num_threads(2)
    loaded=load_run(args.run,args.checkpoint);m=loaded.model;out=args.run/('full_report' if args.checkpoint is None else f'full_report_{args.checkpoint}')
    a=np.load(out/'encodings.npz')
    g=np.concatenate([a['train'],a['test']])[:,:,0]
    K=np.concatenate([a['train_labels'],a['test_labels']]).ravel()
    knots=np.unique(K);means=np.stack([g[K==v].mean(0) for v in knots])
    q=loaded.config['model']['categorical_states']
    kc=float(np.log1p(np.sqrt(q)));gc=float(np.interp(kc,knots,means[:,0]))
    inputs=g[:,:-1];lo,hi=inputs.min(),inputs.max();pad=max(.25,.35*(hi-lo))
    flow=scalar_flow(m,float(lo-pad),float(hi+pad),points=6001)
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    ax=axs[0];ax.plot(flow['grid'],flow['mapped'],label='R(g)');ax.plot(flow['grid'],flow['grid'],'k--',label='Identity')
    ax.axvspan(lo,hi,color='green',alpha=.10,label='Observed R-input range')
    ax.axvline(gc,color='purple',ls=':',label='Q(D) at theoretical Kc')
    for i,r in enumerate(flow['roots']):
        z=r['g'][0];ev=r['eigenvalues'][0]['real']
        r['inside_observed_input_range']=bool(lo<=z<=hi)
        hits=[]
        for x,y,ga,gb in zip(knots[:-1],knots[1:],means[:-1,0],means[1:,0]):
            if min(ga,gb)<=z<=max(ga,gb) and ga!=gb:hits.append(float(x+(y-x)*(z-ga)/(gb-ga)))
        r['microscopic_inverse_K']=hits
        ax.scatter(z,z,color='green' if r['stability']=='stable' else 'red',zorder=5)
        ax.annotate(f'F{i+1}: {r["stability"]}\nR\'={ev:.3f}',(z,z),xytext=(5,10),textcoords='offset points',fontsize=8)
    ax.set(xlabel='g',ylabel='R(g)',title=f'q={q} Potts: learned scalar RG map');ax.grid(alpha=.2);ax.legend(fontsize=8)
    ax=axs[1]
    ax.scatter(K,g[:,0],s=9,alpha=.22,color='black',label='Microscopic ensemble encodings')
    for k in range(g.shape[1]):ax.plot(knots,means[:,k],label=f'Q(C^{k} D)')
    ax.axvline(kc,color='purple',ls=':',label=f'Theory Kc={kc:.5f}')
    for i,r in enumerate(flow['roots']):ax.axhline(r['g'][0],ls='--',lw=.8,alpha=.45,label=f'F{i+1}')
    ax.set(xlabel='K=J/T',ylabel='g',title='Coupling map and direct coarse-graining');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(out/'R_and_K_mapping.png',dpi=180);plt.close(fig)
    # A second wider search is an extrapolation audit, not evidence of phases.
    wide=scalar_flow(m,float(lo-2*(hi-lo+1)),float(hi+2*(hi-lo+1)),points=12001)
    report=dict(theoretical_Kc=kc,microscopic_g_at_Kc=gc,observed_R_input_range=[float(lo),float(hi)],
                flow= {k:v for k,v in flow.items() if k not in ('grid','mapped')},
                wide_flow={k:v for k,v in wide.items() if k not in ('grid','mapped')})
    # IID color bias; all permutations for small q, a labeled sample otherwise.
    from itertools import permutations
    volume=4
    rng=np.random.default_rng(91)
    perms=list(permutations(range(q))) if q<=4 else [tuple(range(q))]+[tuple(rng.permutation(q)) for _ in range(31)]
    report['permutation_audit']=dict(count=len(perms),exhaustive=q<=4,seed=91)
    states=(torch.arange(q**volume)[:,None]//(q**torch.arange(volume)))%q
    patterns=torch.nn.functional.one_hot(states,q).double().flatten(1)
    with torch.no_grad():
        probs=torch.softmax(m.rg_kernel.field_head.logits(m.rg_kernel.local_encoder(patterns)),-1)
        errors=[]
        for perm in perms:
            p=torch.tensor(perm);x=torch.nn.functional.one_hot(p[states],q).double().flatten(1)
            px=torch.softmax(m.rg_kernel.field_head.logits(m.rg_kernel.local_encoder(x)),-1)
            errors.append(float((px[:,p]-probs).abs().mean()))
    report['C']=dict(iid_uniform_output_probabilities=probs.mean(0).tolist(),
        mean_permutation_error=float(np.mean(errors)),max_mean_permutation_error=max(errors))
    # Read-only invariance audit: same microscopic samples with relabeled colors.
    from recurrent_rg.data.ensembles import load_ensembles
    _,test=load_ensembles(loaded.config['data']['cache'])
    selected=test.configurations[::3,:64].double()
    with torch.no_grad():
        base=m.theory_encoder(selected)
        color_diffs=[]
        for perm in perms:
            alt=m.theory_encoder(selected[...,list(perm)])
            color_diffs.append(float((alt-base).square().mean().sqrt()))
    report['Q_color_relabel']=dict(rmse_by_permutation=color_diffs,
        base_std=float(base.std(unbiased=False)),note='Same test configurations, global color relabeling; no retraining')
    (out/'potts_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
