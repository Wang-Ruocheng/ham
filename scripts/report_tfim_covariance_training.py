"""Compare saved TFIM covariance-regularized checkpoints, without training."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recurrent_rg.checkpoints import load_run
from recurrent_rg.data.ensembles import load_ensembles
from recurrent_rg.evaluation import collect_trajectories, closure_metrics
from recurrent_rg.plotting.reports import plot_diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--steps', nargs='+', type=int, default=[2000, 4000, 6000])
    args = parser.parse_args()
    torch.set_num_threads(2)
    reports = []
    clouds = []
    fig, axes = plt.subplots(1, len(args.steps), figsize=(6*len(args.steps), 5), squeeze=False)
    for ax, step in zip(axes[0], args.steps):
        loaded = load_run(args.run, step)
        cfg = loaded.config
        train, test = load_ensembles(cfg['data']['cache'])
        kwargs = dict(samples=cfg['evaluation']['samples'],
                      max_rg_steps=cfg['training']['max_rg_steps'],
                      spin_flip_augment=cfg['data']['spin_flip_augment'])
        tr = collect_trajectories(loaded.model, train.configurations, **kwargs)
        te = collect_trajectories(loaded.model, test.configurations, **kwargs)
        out = args.run / f'step{step}'
        plot_diagnostics(loaded.model, te, test.couplings.numpy(), ('h/J',), out,
                         training_inputs=tr['direct'][:, :-1])
        np.savez_compressed(out/'trajectories.npz', **te, train_direct=tr['direct'],
                            train_couplings=train.couplings.numpy(), couplings=test.couplings.numpy())
        eigenvalues = [np.linalg.eigvalsh(np.cov(te['direct'][:, k].T, bias=True)).tolist()
                       for k in range(te['direct'].shape[1])]
        patterns = torch.tensor([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]], dtype=torch.float64)
        with torch.no_grad():
            outputs = loaded.model.rg_kernel(patterns.reshape(4, 1, 2, 1)).flatten().tolist()
        report = dict(step=step, closure=closure_metrics(te, reference_direct=tr['direct']),
                      covariance_eigenvalues=eigenvalues,
                      weak_fraction=[v[0]/sum(v) if sum(v)>0 else None for v in eigenvalues],
                      kernel=dict(patterns=patterns.tolist(), outputs=outputs),
                      fixed_points=json.loads((out/'fixed_points.json').read_text()),
                      parameters={name:sum(p.numel() for p in module.parameters()) for name,module in
                                  [('Q',loaded.model.theory_encoder),('C',loaded.model.rg_kernel),('R',loaded.model.theory_rg)]})
        displacement = te['direct'][:,1:]-te['direct'][:,:-1]
        step_error = te['one_step']-te['direct'][:,1:]
        denominator = np.sqrt(np.mean(displacement**2, axis=(0,2)))
        report['step_error_over_true_displacement'] = [float(a/b) if b>1e-12 else None for a,b in
            zip(np.sqrt(np.mean(step_error**2,axis=(0,2))),denominator)]
        (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        reports.append(report)
        clouds.append((step, te, test.couplings.numpy()))
        g = te['direct'].reshape(-1,2)
        low = g.min(0)-.3; high = g.max(0)+.3
        xx,yy=np.meshgrid(np.linspace(low[0],high[0],25),np.linspace(low[1],high[1],25))
        grid=np.stack([xx.ravel(),yy.ravel()],axis=1)
        with torch.no_grad(): delta=loaded.model.advance_theory(torch.from_numpy(grid)).numpy()-grid
        norm=np.linalg.norm(delta,axis=1)
        unit=delta/np.maximum(norm[:,None],1e-12)
        ax.quiver(*grid.T,*unit.T,color='gray',alpha=.5)
        colors=np.broadcast_to(test.couplings.numpy().reshape(-1,1),te['direct'].shape[:2]).ravel()
        sc=ax.scatter(*g.T,c=colors,cmap='coolwarm',vmin=0,vmax=2,s=8)
        for item in report['fixed_points']['roots']:
            ax.scatter(*item['g'],marker='*',s=110,c={'stable':'green','saddle':'orange','unstable':'red','nonhyperbolic':'purple'}[item['stability']],edgecolors='black')
        ax.set(xlabel='g1',ylabel='g2',title=f'{step} updates; normalized R(g)-g',xlim=(low[0],high[0]),ylim=(low[1],high[1]))
        fig.colorbar(sc,ax=ax,label='h/J (original ensemble)')
    fig.tight_layout();fig.savefig(args.run/'flow_comparison.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,len(clouds),figsize=(5*len(clouds),7),squeeze=False,layout='constrained')
    for col,(step,te,labels) in enumerate(clouds):
        for j in range(2):
            for k in range(te['direct'].shape[1]):
                values=np.unique(labels); means=np.array([te['direct'][labels.reshape(-1)==v,k,j].mean() for v in values])
                axes[j,col].plot(values,means,'-',label=f'k={k}')
            axes[j,col].axvline(1,color='black',ls=':',lw=1)
            axes[j,col].set(xlabel='h/J',ylabel=f'g{j+1}',title=f'{step} updates');axes[j,col].legend(fontsize=7)
    fig.savefig(args.run/'coupling_by_scale.png',dpi=170);plt.close(fig)
    h=json.loads((args.run/'history_progress.json').read_text())
    cfg=load_run(args.run,args.steps[-1]).config
    weights=dict(cfg['loss']['weights'])
    weights.update(rollout=cfg['loss']['rollout_weight'],isotropy=cfg['loss']['isotropy_weight'])
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    for ax,split in zip(axes,['train','test']):
        for key,w in weights.items():
            if not w:
                continue
            ax.plot([r['epoch'] for r in h],[w*r[f'{split}_{key}'] for r in h],label=key)
        ax.plot([r['epoch'] for r in h],[r[f'{split}_total'] for r in h],'k-',label='total')
        ax.set(xlabel='optimizer updates',ylabel='weighted loss',title=f'{split}: fixed evaluation subset',yscale='symlog');ax.legend(fontsize=8)
    fig.savefig(args.run/'loss_components.png',dpi=170);plt.close(fig)
    (args.run/'checkpoint_comparison.json').write_text(json.dumps(reports,indent=2)+'\n')
    print(json.dumps(reports,indent=2))


if __name__ == '__main__':
    main()
