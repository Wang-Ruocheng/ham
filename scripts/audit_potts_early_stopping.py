"""Label-free checkpoint selection on fresh chains, with independent audit chains.

Retrospective, one training seed. Rule constants are fixed before evaluation.
Couplings are used to reproduce held-out cells, never as loss/selection targets.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recurrent_rg.checkpoints import load_run
from recurrent_rg.config import RunConfig
from recurrent_rg.data.ensembles import load_ensembles
from recurrent_rg.data.potts import sample_potts
from recurrent_rg.data.statistics import categorical_plaquette_distribution
from recurrent_rg.evaluation import collect_trajectories, closure_metrics, scalar_flow
from recurrent_rg.losses import qcr_objective


def select_checkpoints(steps, means, sems, patience=2, min_relative_improvement=.01):
    steps,means,sems=map(np.asarray,(steps,means,sems))
    if len(steps)==0 or not(len(steps)==len(means)==len(sems)):
        raise ValueError('Need matching nonempty checkpoint statistics')
    if np.any(~np.isfinite(means)) or np.any(~np.isfinite(sems)) or np.any(sems<0):
        raise ValueError('Checkpoint statistics must be finite with nonnegative errors')
    best=int(np.argmin(means))
    earliest=int(np.flatnonzero(means<=means[best]+sems[best])[0])
    incumbent=0;stale=0;stop=len(steps)-1
    for i in range(1,len(steps)):
        if means[i]<means[incumbent]*(1-min_relative_improvement):
            incumbent=i;stale=0
        else:
            stale+=1
        if stale>=patience:
            stop=i;break
    return dict(best_validation=int(steps[best]),one_standard_error=int(steps[earliest]),
                patience_selected=int(steps[incumbent]),patience_stop=int(steps[stop]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=Path('results/potts2d_g1_neutral0_2400_seed91'))
    p.add_argument('--output',type=Path,default=Path('results/potts2d_early_stopping_seed91'))
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    cfg=RunConfig.from_dict(json.loads((args.run/'manifest.json').read_text())['config'])
    _,oldtest=load_ensembles(cfg.data.cache)
    ks=torch.unique(oldtest.couplings).numpy();blocks={}
    protocol=dict(training_run=str(args.run),checkpoints=list(range(400,2401,400)),
        selection_rules=['minimum validation mean','patience=2, relative min_delta=.01',
                         'earliest within best mean + best SEM'],
        samples_per_ensemble=64,replicas=3,burn=1024,spacing=8,
        seed_formula='base + replica*1000 + coupling_index',
        seed_bases=dict(validation=640091,audit=740091),
        note='Fresh chains at existing held-out K cells; no new training seeds. Audit never selects checkpoints. SEM reflects three chain/kernel realizations, not training-seed uncertainty.')
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for split,base in protocol['seed_bases'].items():
        blocks[split]=[]
        for rep in range(3):
            xs=[]
            for i,k in enumerate(ks):
                a=sample_potts(float(k),16,64,base+1000*rep+i,burn=1024)
                xs.append(torch.nn.functional.one_hot(torch.from_numpy(a.astype(np.int64)),3).to(torch.uint8))
            blocks[split].append(torch.stack(xs))
        print(f'Sampled fresh {split}: {len(ks)} cells x3 chains x64 configurations',flush=True)
    torch.save(dict(K=torch.from_numpy(ks),**blocks),args.output/'fresh_ensembles.pt')
    rows=[];flows=[];curves=[]
    for step in protocol['checkpoints']:
        m=load_run(args.run/f'training_step_{step}.pt').model;m.eval();row=dict(step=step)
        for split,xs in blocks.items():
            records=[];direct=[]
            for rep,x in enumerate(xs):
                seed=12345+rep
                with torch.random.fork_rng(devices=[]),torch.no_grad():
                    torch.manual_seed(seed)
                    total,terms=qcr_objective(m,x.double(),pattern_fn=categorical_plaquette_distribution,
                                             **cfg.loss.objective_kwargs())
                trajectories=collect_trajectories(m,x,samples=64,max_rg_steps=2,sampling_seed=seed)
                records.append(dict(total=float(total),terms={k:float(v) for k,v in terms.items()},
                                    closure=closure_metrics(trajectories)))
                direct.append(trajectories['direct'][:,:,0])
            totals=np.array([r['total'] for r in records])
            row[split]=dict(mean=float(totals.mean()),sem=float(totals.std(ddof=1)/np.sqrt(3)),records=records)
            if split=='audit':
                g=np.stack(direct);lo,hi=g[:,:,:-1].min(),g[:,:,:-1].max();width=hi-lo+1
                f=scalar_flow(m,float(lo-2*width),float(hi+2*width),points=6001)
                # Below is physical posthoc audit, never accessed by the selector.
                mean_g=g.mean(0);kc=float(np.log1p(np.sqrt(3)))
                for root in f['roots']:
                    z=root['g'][0];hits=[]
                    for ka,kb,ga,gb in zip(ks[:-1],ks[1:],mean_g[:-1,0],mean_g[1:,0]):
                        if min(ga,gb)<=z<=max(ga,gb) and ga!=gb:
                            hits.append(float(ka+(kb-ka)*(z-ga)/(gb-ga)))
                    root['microscopic_inverse_K']=hits;root['inside_observed_inputs']=bool(lo<=z<=hi)
                row['physical_audit']=dict(roots=f['roots'],search_range=f['search_range'],
                    g_at_Kc=float(np.interp(kc,ks,mean_g[:,0])),theoretical_Kc=kc,
                    spread_by_scale=g.reshape(-1,3).std(0).tolist())
                flows.append(f);curves.append(mean_g)
        rows.append(row);print(f'Audited {step}: validation={row["validation"]["mean"]:.6g}, audit={row["audit"]["mean"]:.6g}',flush=True)
    selection=select_checkpoints([r['step'] for r in rows],[r['validation']['mean'] for r in rows],
                                [r['validation']['sem'] for r in rows])
    result=dict(protocol=protocol,selection=selection,checkpoints=rows)
    (args.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    steps=[r['step'] for r in rows]
    for split in ('validation','audit'):
        axs[0].errorbar(steps,[r[split]['mean'] for r in rows],yerr=[r[split]['sem'] for r in rows],marker='o',label=split)
        for k in (1,2):
            values=[np.mean([v['closure']['rollout']['nrmse'][k] for v in r[split]['records']]) for r in rows]
            axs[1].plot(steps,values,'-o',label=f'{split}, k={k}')
    axs[0].set(xlabel='Training updates',ylabel='QCR objective (mean +/- SEM)',title='Fresh independent chains')
    axs[1].set(xlabel='Training updates',ylabel='Rollout NRMSE',title='Normalization: each evaluated target SD')
    for ax in axs:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(args.output/'validation_curves.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    selected=set(selection[k] for k in ('best_validation','one_standard_error','patience_selected'))
    for ax,row,f in zip(axs.flat,rows,flows):
        ax.plot(f['grid'],f['mapped'],label='R(g)');ax.plot(f['grid'],f['grid'],'k--')
        for r in f['roots']:ax.scatter(r['g'][0],r['g'][0],c='green' if r['stability']=='stable' else 'red')
        ax.axvline(row['physical_audit']['g_at_Kc'],color='purple',ls=':',label='Q(D) at Kc (posthoc)')
        ax.set(title=f'{row["step"]} updates'+(' [selected]' if row['step'] in selected else ''),xlabel='g',ylabel='R(g)')
        ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.savefig(args.output/'checkpoint_flows.png',dpi=150);plt.close(fig)
    print(json.dumps(selection),flush=True)


if __name__=='__main__':main()
