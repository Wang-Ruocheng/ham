"""All-ensemble report for a learned-C, covariance-regularized QCR run."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from recurrent_rg.checkpoints import load_run
from recurrent_rg.data.ensembles import load_ensembles
from recurrent_rg.evaluation import collect_trajectories,closure_metrics
from recurrent_rg.plotting.reports import plot_diagnostics
from experiments.ising2d_q_only import readout


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--checkpoint',type=int);args=p.parse_args()
    torch.set_num_threads(2);loaded=load_run(args.run,args.checkpoint);model=loaded.model
    train,test=load_ensembles(loaded.config['data']['cache'])
    out=args.run/(f'step{args.checkpoint}_report' if args.checkpoint else 'full_report');out.mkdir(exist_ok=True)
    tr=collect_trajectories(model,train.configurations,samples=64,spin_flip_augment=False)
    te=collect_trajectories(model,test.configurations,samples=64,spin_flip_augment=False)
    metrics=[]
    for k in range(3):
        eig=np.linalg.eigvalsh(np.cov(te['direct'][:,k].T,bias=True))
        r2,_=readout(tr['direct'][:,k],te['direct'][:,k],train.couplings.numpy(),test.couplings.numpy())
        metrics.append(dict(k=k,eigenvalues=eig.tolist(),weak_fraction=float(eig[0]/eig.sum()),K_B_R2=r2))
    factors=model.rg_kernel.geometry.factors
    volume=model.rg_kernel.geometry.block_volume
    if volume > 16:
        raise ValueError('Exhaustive binary kernel audit limited to 16 input spins')
    patterns=(2*((torch.arange(2**volume)[:,None]//(2**torch.arange(volume)))%2)-1).double()
    with torch.no_grad():truth=model.rg_kernel(patterns.reshape(2**volume,1,*factors,1)).flatten().tolist()
    report=dict(representation=metrics,closure=closure_metrics(te,reference_direct=tr['direct']),kernel=dict(patterns=patterns.tolist(),outputs=truth))
    stochastic = loaded.config['model'].get('kernel_sampling') == 'bernoulli'
    if stochastic:
        with torch.no_grad():
            logits=model.rg_kernel.field_head.logits(model.rg_kernel.local_encoder(patterns))
        report['kernel']['p_plus']=torch.sigmoid(2*logits).flatten().tolist()
        report['kernel']['outputs_note']='One random draw per pattern, NOT a deterministic truth table'
        repeats=[]
        for seed in range(5):
            draw=collect_trajectories(model,test.configurations,samples=64,
                                      spin_flip_augment=False,sampling_seed=seed)
            repeats.append(closure_metrics(draw,reference_direct=tr['direct']))
        report['sampling_repeats']=dict(seeds=list(range(5)),closure=repeats,
            note='Coarse-sampling noise for one trained model, not independent training seeds')
    # Test actual recurrent fields, not only individual block truth tables.
    spatial=train.configurations.shape[2:-1]
    eta=(1-2*(torch.stack(torch.meshgrid(*[torch.arange(n) for n in spatial], indexing='ij')).sum(0)%2)).double()
    probes=torch.stack([torch.ones_like(eta),-torch.ones_like(eta),eta,-eta])[:,None,...,None]
    if stochastic:
        with torch.random.fork_rng(devices=[]),torch.no_grad():
            torch.manual_seed(12345)
            fields=model.coarse_grain(probes)
    else:
        with torch.no_grad(): fields=model.coarse_grain(probes)
    probe_rows=[]
    for k,field in enumerate(fields):
        coarse_eta=(1-2*(torch.stack(torch.meshgrid(*[torch.arange(n) for n in field.shape[2:-1]], indexing='ij')).sum(0)%2)).double()
        spins=field[:,0,...,0]
        probe_rows.append(dict(k=k,shape=list(spins.shape[1:]),
            magnetization=spins.flatten(1).mean(1).tolist(),
            staggered_magnetization=(spins*coarse_eta).flatten(1).mean(1).tolist(),
            opposite_checkerboards_identical=bool(torch.equal(spins[2],spins[3]))))
    report['ordered_probes']=dict(order=['ferro_plus','ferro_minus','checker_plus','checker_minus'],scales=probe_rows)
    if stochastic:
        report['ordered_probes']['note']='One sampled realization per initial field; not ensemble expectation'
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez(out/'encodings.npz',train=tr['direct'],test=te['direct'],test_rollout=te['rollout'],train_labels=train.couplings.numpy(),test_labels=test.couplings.numpy())
    plot_diagnostics(model,te,test.couplings.numpy(),('K','B'),out,training_inputs=tr['direct'][:,:-1])
    fig,axs=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    cloud=te['direct'].reshape(-1,2);lo=cloud.min(0)-.2;hi=cloud.max(0)+.2
    for k in range(3):
        for j in range(2):
            ax=axs[j,k];g=te['direct'][:,k]
            sc=ax.scatter(*g.T,c=test.couplings[:,j],cmap='viridis' if j==0 else 'coolwarm',s=12,norm=None if j==0 else SymLogNorm(linthresh=.001,vmin=-.5,vmax=.5))
            ax.set(xlabel='g1',ylabel='g2',title=f'k={k}, weak PC {100*metrics[k]["weak_fraction"]:.2f}%',aspect='equal',xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]));fig.colorbar(sc,ax=ax,label='K' if j==0 else 'B (symlog)')
    fig.savefig(out/'latent_KB.png',dpi=170);plt.close(fig)
    h=json.loads((args.run/'history_progress.json').read_text())
    loss_config=loaded.config['loss']
    weights=dict(loss_config['weights'])
    weights.update(rollout=loss_config['rollout_weight'],isotropy=loss_config['isotropy_weight'])
    weights['tangent'] = loss_config.get('tangent_weight',0.)
    weights['orientation'] = loss_config.get('orientation_weight',0.)
    weights['kernel_sign'] = loss_config.get('kernel_sign_weight',0.)
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,split in zip(axs,('train','test')):
        for term,w in weights.items():
            if w:ax.plot([r['epoch'] for r in h],[r[split+'_'+term]*w for r in h],label=term)
        ax.plot([r['epoch'] for r in h],[r[split+'_total'] for r in h],'k-',label='total')
        ax.set(xlabel='updates',ylabel='weighted loss',title=split+' (fixed 64 ensembles)');ax.set_yscale('symlog',linthresh=.001);ax.legend(fontsize=7)
    fig.savefig(out/'loss.png',dpi=170);plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
