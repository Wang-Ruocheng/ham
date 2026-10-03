"""Post-hoc temperature-curve projections and direct versus recurrent BC flows."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from recurrent_rg.checkpoints import load_run


def project_curve(point, curve, temperatures):
    """Nearest polyline projection, not an inverse when off the curve."""
    start=curve[:-1];delta=np.diff(curve,axis=0)
    fraction=np.clip(np.sum((point-start)*delta,axis=1)/np.maximum(np.sum(delta*delta,axis=1),1e-30),0,1)
    projections=start+fraction[:,None]*delta
    distances=np.linalg.norm(projections-point,axis=1);i=int(np.argmin(distances))
    return dict(nearest_projected_temperature=float(temperatures[i]+fraction[i]*(temperatures[i+1]-temperatures[i])),
                distance=float(distances[i]),projected_g=projections[i].tolist(),
                note='Nearest polyline projection, NOT a physical inverse if the residual is appreciable')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(2)
    loaded=load_run(args.run);m=loaded.model;folder=args.run/'full_report'
    if loaded.config['data']['coupling_names']!=['T/J'] or loaded.config['model']['theory_dim']!=2:
        raise ValueError('Requires temperature scan and dim g=2')
    data=np.load(folder/'encodings.npz')
    temperatures=np.concatenate([data['train_labels'],data['test_labels']]).ravel()
    direct=np.concatenate([data['train'],data['test']]);knots=np.unique(temperatures)
    means=np.stack([direct[temperatures==v].mean(0) for v in knots])
    metrics=json.loads((folder/'metrics.json').read_text());report=dict(reference_Tc=1.398,roots={})
    report['reference_g_by_scale']=[[float(np.interp(1.398,knots,means[:,k,j])) for j in range(2)] for k in range(means.shape[1])]
    for key in ['fixed_points_R','fixed_points_R2']:
        report['roots'][key]=[dict(g=r['g'],stability=r['stability'],eigenvalues=r['eigenvalues'],
            projections_by_scale=[project_curve(np.array(r['g']),means[:,k],knots) for k in range(means.shape[1])]) for r in metrics[key]]
    selected=sorted(set(int(np.argmin(abs(knots-v))) for v in [1.,1.3,1.4,1.5,1.8,2.4]))
    paths=[]
    with torch.no_grad():
        # Average predictions of individual ensembles, not R of their mean.
        z=torch.from_numpy(direct[:,0]);rollout=[z.numpy()]
        for _ in range(12):
            z=m.advance_theory(z);rollout.append(z.numpy())
    rollout=np.stack(rollout,1)
    rmeans=np.stack([rollout[temperatures==v].mean(0) for v in knots])
    fig,axes=plt.subplots(1,3,figsize=(16,4.7),layout='constrained')
    colors=plt.cm.viridis(np.linspace(.05,.95,len(selected)))
    for i,color in zip(selected,colors):
        label=f'T/J={knots[i]:.3f}'
        axes[0].plot(*means[i].T,'o-',c=color,label=label)
        axes[0].plot(*rmeans[i,:means.shape[1]].T,'x--',c=color)
        for j in range(2):
            axes[j+1].plot(np.arange(13),rmeans[i,:,j],'--',c=color,label=label)
            axes[j+1].plot(np.arange(means.shape[1]),means[i,:,j],'o-',c=color)
        paths.append(dict(T=float(knots[i]),direct=means[i].tolist(),rollout_mean=rmeans[i].tolist()))
    axes[0].set(xlabel='g1',ylabel='g2',title='Observed scales: direct (solid) vs R (dashed)')
    for j in range(2):
        axes[j+1].axvspan(2,12,color='gray',alpha=.12,label='Beyond trained depth')
        axes[j+1].set(xlabel='k',ylabel=f'g{j+1}',title='R-only extension beyond k=2 is unvalidated')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(folder/'critical_trajectories.png',dpi=170);plt.close(fig)
    report['selected_trajectories']=paths
    eig=np.linalg.eigvalsh(np.cov(direct.reshape(-1,2).T,bias=True))
    report['all_scale_covariance_eigenvalues']=eig.tolist()
    report['all_scale_weak_fraction']=float(eig[0]/eig.sum())
    report['note']='A small global PCA eigenvalue is not proof of intrinsic rank one; off-curve roots and long rollouts are not physically validated.'
    (folder/'temperature_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
