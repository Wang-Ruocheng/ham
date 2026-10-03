"""Noise-aware descriptive rank checks; physical labels enter diagnostics only."""
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
from recurrent_rg.data.views import ensemble_views


def encode_split(run,x):
    samples=run.config['evaluation']['samples'];flip=run.config['data']['spin_flip_augment']
    full=[];halves=[]
    with torch.no_grad():
        for chunk in x.split(16):
            fields=run.model.coarse_grain(ensemble_views(chunk,samples,spin_flip_augment=flip))
            full.append(torch.stack([run.model.theory_encoder(z) for z in fields],1).numpy())
            halves.append(torch.stack([torch.stack([run.model.theory_encoder(h) for h in z.chunk(2,dim=1)],1) for z in fields],2).numpy())
    return np.concatenate(full),np.concatenate(halves)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--checkpoint',type=int,default=6000);args=p.parse_args()
    torch.set_num_threads(2);run=load_run(args.run,args.checkpoint)
    train,test=load_ensembles(run.config['data']['cache'])
    tg,th=encode_split(run,train.configurations);vg,vh=encode_split(run,test.configurations)
    report={'checkpoint':args.checkpoint,'test_scales':[],
        'caveat':'Global covariance rank is not intrinsic dimension; split-noise ratios and finite-grid tangent tests are descriptive, not formal hypothesis tests.'}
    for k in range(vg.shape[1]):
        g=vg[:,k];eig,basis=np.linalg.eigh(np.cov(g.T,bias=True))
        split_error=(vh[:,0,k]-vh[:,1,k])/2
        noise=((split_error@basis)**2).mean(0)
        report['test_scales'].append(dict(k=k,eigenvalues=eig.tolist(),weak_fraction=float(eig[0]/eig.sum()),
            pc_signal_to_split_noise=[float(a/b) if b>1e-20 else None for a,b in zip(eig,noise)]))
    out=args.run/f'dimension_audit_step{args.checkpoint}';out.mkdir(exist_ok=True)
    labels=np.concatenate([train.couplings.numpy(),test.couplings.numpy()]).reshape(len(tg)+len(vg),-1)
    g=np.concatenate([tg,vg]);halves=np.concatenate([th,vh])
    if labels.shape[1]==2:
        ks,bs=np.unique(labels[:,0]),np.unique(labels[:,1])
        means=np.empty((len(ks),len(bs),2));noise=np.empty((len(ks),len(bs)))
        for i,k in enumerate(ks):
            for j,b in enumerate(bs):
                take=(labels[:,0]==k)&(labels[:,1]==b)
                means[i,j]=g[take,0].mean(0)
                err=(halves[take,0,0]-halves[take,1,0])/2
                noise[i,j]=(err**2).sum(1).mean()/take.sum()
        sinangles=[];ratios=[];resolvable=[]
        for i in range(1,len(ks)-1):
            for j in range(1,len(bs)-1):
                # Half central secants: spans of adjacent grid cells, not derivatives
                # in arbitrarily chosen K/B units. Nonuniform B spacing is retained.
                a=(means[i+1,j]-means[i-1,j])/2
                b=(means[i,j+1]-means[i,j-1])/2
                sv=np.linalg.svd(np.stack([a,b],1),compute_uv=False)
                angle=abs(a[0]*b[1]-a[1]*b[0])/max(np.linalg.norm(a)*np.linalg.norm(b),1e-20)
                noise_a=(noise[i+1,j]+noise[i-1,j])/4
                noise_b=(noise[i,j+1]+noise[i,j-1])/4
                ratio=sv[-1]/np.sqrt(max(noise_a+noise_b,1e-20))
                sinangles.append(angle);ratios.append(ratio)
                resolvable.append(np.linalg.norm(a)>2*np.sqrt(noise_a) and np.linalg.norm(b)>2*np.sqrt(noise_b))
        report['local_KB_grid']=dict(interior_cells=len(ratios),
            median_sine_of_tangent_angle=float(np.median(sinangles)),
            median_smallest_secant_singular_over_noise=float(np.median(ratios)),
            fraction_smallest_secant_over_2noise=float(np.mean(np.array(ratios)>2)),
            fraction_both_secants_resolved=float(np.mean(resolvable)),
            note='Nonuniform finite-grid secants; curvature and finite sampling can affect this statistic. Not a proof of manifold dimension.')
        fig,ax=plt.subplots(figsize=(7,6),layout='constrained')
        for i in range(len(ks)):ax.plot(*means[i].T,color='steelblue',alpha=.65,lw=.8)
        for j in range(len(bs)):ax.plot(*means[:,j].T,color='darkorange',alpha=.65,lw=.8)
        ax.plot([],[],color='steelblue',label='vary B at fixed K')
        ax.plot([],[],color='darkorange',label='vary K at fixed B')
        ax.set(xlabel='g1',ylabel='g2',title='Microscopic coupling-grid means; labels only for audit');ax.legend()
        fig.savefig(out/'KB_grid.png',dpi=180);plt.close(fig)
        np.savez_compressed(out/'grid.npz',K=ks,B=bs,g=means,noise=noise)
    else:
        trainlabels=train.couplings.numpy().reshape(-1);testlabels=test.couplings.numpy().reshape(-1)
        values=np.unique(trainlabels);means=np.array([tg[trainlabels==v,0].mean(0) for v in values])
        pred=np.column_stack([np.interp(testlabels,values,means[:,j]) for j in range(2)])
        report['heldout_micro_curve_R2']=float(1-((vg[:,0]-pred)**2).sum()/((vg[:,0]-vg[:,0].mean(0))**2).sum())
    np.savez_compressed(out/'encodings.npz',train=tg,test=vg,train_labels=train.couplings.numpy(),test_labels=test.couplings.numpy())
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
