"""Direct C/Q versus R audit for the middle-dense scalar AT run; no training."""
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recurrent_rg.checkpoints import load_run
from recurrent_rg.data.ensembles import load_ensembles
from recurrent_rg.data.ashkin_teller import observables
from recurrent_rg.evaluation import scalar_flow


def main():
    torch.set_num_threads(2)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('results/ashkin_teller2d_g1_middle_dense_mi_variance2ratio1_color1_1200_seed91'))
    folder=parser.parse_args().run
    run=load_run(folder);m=run.model;m.eval();tr,te=load_ensembles(run.config['data']['cache'])
    x=torch.cat([tr.configurations,te.configurations]);K=torch.cat([tr.couplings,te.couplings])[:,0].numpy()
    knots=np.unique(K);draws=[];obsdraws=[]
    with torch.no_grad():
        fixed=run.config['model']['fixed_kernel']
        seeds=(12345,) if fixed else (12345,12346,12347)
        for seed in seeds:
            torch.manual_seed(seed);ds=[];ob=[]
            for chunk in x.split(8):
                levels=m.coarse_grain(chunk.double(),max_rg_steps=3)
                ds.append(torch.stack([m.theory_encoder(z) for z in levels],1).numpy()[...,0])
                rows=[]
                for z in levels:
                    c=z.argmax(-1).numpy();stats=observables(c.reshape(-1,*c.shape[2:]))
                    rows.append(np.stack([stats[n].reshape(c.shape[:2]).mean(1) for n in ('m_sigma','m_tau','m_product')],-1))
                ob.append(np.stack(rows,1))
            draws.append(np.concatenate(ds));obsdraws.append(np.concatenate(ob))
        d=np.stack(draws);obs=np.stack(obsdraws)
        one=m.advance_theory(torch.from_numpy(d[:,:,:-1,None])).numpy()[...,0]
        z=torch.from_numpy(d[0,:,0,None]);roll=[z.numpy()[:,0]]
        for _ in range(20):z=m.advance_theory(z);roll.append(z.numpy()[:,0])
        roll=np.stack(roll,1)
        codes=(torch.arange(256)[:,None]//(4**torch.arange(4)))%4
        pat=torch.nn.functional.one_hot(codes,4).double().flatten(1)
        table=(torch.nn.functional.one_hot(codes[:,0],4).double().numpy() if fixed else
               torch.softmax(m.rg_kernel.field_head.logits(m.rg_kernel.local_encoder(pat)),-1).numpy())
    means=np.stack([d[:,K==v].mean((0,1)) for v in knots]);om=np.stack([obs[:,K==v].mean((0,1)) for v in knots])
    rmeans=np.stack([roll[K==v].mean(0) for v in knots]);op=np.stack([one[:,K==v].mean((0,1)) for v in knots])
    roots=scalar_flow(m,-5,10,points=18001)['roots']
    for r in roots:
        z=r['g'][0];r['inverse_K']=[float(a+(b-a)*(z-u)/(v-u)) for a,b,u,v in zip(knots[:-1],knots[1:],means[:-1,0],means[1:,0]) if u!=v and min(u,v)<=z<=max(u,v)]
    report=dict(roots=roots,probes=[],closure={},C_reference={},sampling_seeds=list(seeds),fixed_C=fixed,
                note='k=0,1,2 trained; k=3 has 2x2 lattice, repeated periodic Q patches, untrained. No long-scale convergence claim. Fixed C is deterministic; stochastic C uses three sampling seeds, not training seeds.')
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    selected=[np.argmin(abs(knots-v)) for v in (.15,.20,.21,.28)]
    colors=plt.cm.tab10(np.arange(4))
    for i,color in zip(selected,colors):
        sd=d[:,K==knots[i]].reshape(-1,4).std(0,ddof=1)
        axes[0,0].errorbar(np.arange(4),means[i],yerr=sd,fmt='o-',c=color,label=f'K={knots[i]:.3f} direct')
        axes[0,0].plot(np.arange(4),rmeans[i,:4],'x--',c=color)
        axes[0,1].plot(np.arange(21),rmeans[i],c=color,label=f'K={knots[i]:.3f}')
        report['probes'].append(dict(K=float(knots[i]),direct_g=means[i].tolist(),direct_sd=sd.tolist(),R_rollout=rmeans[i].tolist(),physical_order=om[i].tolist()))
    for k in range(3):
        axes[1,0].plot(knots,means[:,k],label=f'Q(C^{k}D)')
    for k in range(2):
        axes[1,1].plot(knots,means[:,k+1]-means[:,k],label=f'direct step {k}')
        axes[1,1].plot(knots,op[:,k]-means[:,k],'--',label=f'R step {k}')
    axes[1,1].axhline(0,c='k',lw=.7)
    for ax in axes[0]:ax.axvspan(2,3 if ax is axes[0,0] else 20,color='.8',alpha=.3)
    for ax in axes[1]:ax.set_xlim(.15,.26);ax.set_xlabel('K')
    for ax,title in zip(axes.ravel(),['Direct Q (solid) vs R (dashed); k=3 is tiny-lattice extrapolation','R-only long rollout; beyond k=2 unvalidated','Microscopic and coarse encodings','Direct drift vs fitted drift']):
        ax.set_title(title,fontsize=10);ax.legend(fontsize=7);ax.grid(alpha=.2)
    axes[0,0].set(xlabel='k',ylabel='g');axes[0,1].set(xlabel='k',ylabel='g')
    for name,mask in [('test',np.arange(len(K))>=len(tr.configurations)),('middle',(K>=.195)&(K<=.215)),('first',(K>=.17)&(K<=.20)),('second',(K>=.21)&(K<=.245))]:
        error=one[:,mask,:2]-d[:,mask,1:3];motion=d[:,mask,1:3]-d[:,mask,:2]
        rmse=np.sqrt((error**2).mean((0,1)));size=np.sqrt((motion**2).mean((0,1)))
        report['closure'][name]=dict(rmse=rmse.tolist(),motion_rms=size.tolist(),ratio=(rmse/size).tolist())
    signs=np.array([[-1,-1,1],[-1,1,-1],[1,-1,-1],[1,1,1]])
    for name,p in [('disordered',np.ones(4)/4),('product_plus',np.array([.5,0,0,.5])),('product_minus',np.array([0,.5,.5,0]))]:
        rows=[]
        for k in range(6):
            rows.append(dict(k=k,p=p.tolist(),signed_order=(p@signs).tolist()))
            p=np.prod(p[codes.numpy()],axis=1)@table
        report['C_reference'][name]=rows
    out=folder/'direct_audit';out.mkdir(exist_ok=True)
    fig.savefig(out/'Q_vs_R.png',dpi=160)
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(out/'arrays.npz',K=K,direct=d,one_step=one,rollout=roll,observables=obs)
    print(json.dumps({k:v for k,v in report.items() if k!='C_reference'},indent=2))


if __name__=='__main__':main()
