"""Paired covariance ablation: saved trajectories, roots, and long rollout."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recurrent_rg.checkpoints import load_run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--regularized',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(2)
    roots=[args.baseline,args.regularized]
    manifests=[json.loads((r/'manifest.json').read_text()) for r in roots]
    configs=[json.loads(json.dumps(m['config'])) for m in manifests]
    weights=[c['loss'].pop('isotropy_weight') for c in configs]
    assert weights==[0,20] and configs[0]==configs[1]
    assert manifests[0]['data_sha256']==manifests[1]['data_sha256']
    loaded=[load_run(r,6000) for r in roots]
    assert torch.equal(loaded[0].payload['batch_generator_state'],loaded[1].payload['batch_generator_state'])
    output=args.baseline/'paired_comparison';output.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(13,5.5),layout='constrained')
    report={'config_diff_only':'isotropy_weight: 0 vs 20','same_data_sha256':True,
            'same_final_batch_rng_state':True,'runs':[]}
    pairs=[]
    for ax,path,run,weight in zip(axes,roots,loaded,weights):
        data=np.load(path/'step6000/trajectories.npz');g=data['direct'];labels=data['couplings'].reshape(-1)
        metrics=json.loads((path/'step6000/report.json').read_text())
        all_labels=np.concatenate([data['train_couplings'].reshape(-1),labels])
        all_g=np.concatenate([data['train_direct'][:,0],g[:,0]])
        knots=np.unique(all_labels)
        curve=np.stack([all_g[all_labels==v].mean(0) for v in knots])
        segments=curve[1:]-curve[:-1]
        for root in metrics['fixed_points']['roots']:
            point=np.array(root['g'])
            fraction=np.clip(np.sum((point-curve[:-1])*segments,axis=1)/np.maximum(np.sum(segments**2,axis=1),1e-30),0,1)
            projections=curve[:-1]+fraction[:,None]*segments
            residual=np.linalg.norm(projections-point,axis=1);i=int(residual.argmin())
            root['nearest_microscopic_curve_h_over_J']=float(knots[i]+fraction[i]*(knots[i+1]-knots[i]))
            root['microscopic_curve_residual']=float(residual[i])
            root['projection_note']='Nearest polyline projection, not a unique physical inverse or critical-point validation.'
        history=json.loads((path/'history_progress.json').read_text());last=history[-1]
        common=last['test_total']-weight*last.get('test_isotropy',0)
        cloud=g.reshape(-1,2);lo=cloud.min(0)-.3;hi=cloud.max(0)+.3
        if metrics['fixed_points']['roots']:
            fixed=np.array([item['g'] for item in metrics['fixed_points']['roots']])
            lo=np.minimum(lo,fixed.min(0)-.12);hi=np.maximum(hi,fixed.max(0)+.12)
        xx,yy=np.meshgrid(np.linspace(lo[0],hi[0],29),np.linspace(lo[1],hi[1],29))
        grid=np.stack([xx.ravel(),yy.ravel()],1)
        with torch.no_grad():delta=run.model.advance_theory(torch.from_numpy(grid)).numpy()-grid
        unit=delta/np.maximum(np.linalg.norm(delta,axis=1)[:,None],1e-12)
        ax.quiver(*grid.T,*unit.T,color='gray',alpha=.5)
        sc=ax.scatter(*cloud.T,c=np.repeat(labels,g.shape[1]),cmap='coolwarm',vmin=0,vmax=2,s=10)
        for root in metrics['fixed_points']['roots']:
            ax.scatter(*root['g'],marker='*',s=150,edgecolors='black',
                       c={'stable':'green','saddle':'orange','unstable':'red','nonhyperbolic':'purple'}[root['stability']],label=root['stability'])
        ax.set(xlabel='g1',ylabel='g2',xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),title=f'Covariance weight {weight}; 6000 updates')
        if metrics['fixed_points']['roots']:
            handles,names=ax.get_legend_handles_labels();unique=dict(zip(names,handles))
            ax.legend(unique.values(),unique.keys())
        fig.colorbar(sc,ax=ax,label='original h/J; all scales')
        unique=np.unique(labels);couplings=[float(unique[unique<1].max()),float(unique[unique>1].min())]
        idx=[np.flatnonzero(labels==v) for v in couplings]
        pred=torch.from_numpy(g[:,0]).clone();trajectory=[pred.numpy().copy()]
        with torch.no_grad():
            for _ in range(50):
                pred=run.model.advance_theory(pred);trajectory.append(pred.numpy().copy())
        trajectory=np.stack(trajectory,axis=1)
        direct_distance=np.linalg.norm(g[idx[0]].mean(0)-g[idx[1]].mean(0),axis=1)
        roll_distance=np.linalg.norm(trajectory[idx[0]].mean(0)-trajectory[idx[1]].mean(0),axis=1)
        pairs.append((weight,direct_distance,roll_distance))
        record=dict(weight=weight,test_common_objective=common,test_total=last['test_total'],
                    weak_fraction=metrics['weak_fraction'][0],closure=metrics['closure'],
                    kernel=metrics['kernel'],fixed_points=metrics['fixed_points'],pair_couplings=couplings,
                    pair_direct_distance=direct_distance.tolist(),pair_rollout_distance=roll_distance.tolist())
        report['runs'].append(record)
    fig.savefig(output/'R_comparison.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,(weight,d,r) in zip(axes,pairs):
        ax.plot(np.arange(len(r)),r/r[0],label='R rollout')
        ax.plot(np.arange(len(d)),d/d[0],'o-',label='direct Q(C^k D)')
        ax.axvline(5,color='black',ls=':',label='training horizon')
        ax.set(xlabel='k',ylabel='pair distance / initial distance',title=f'Covariance weight {weight}')
        ax.legend(fontsize=8)
    fig.savefig(output/'critical_pair.png',dpi=170);plt.close(fig)
    (output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(output)


if __name__=='__main__':main()
