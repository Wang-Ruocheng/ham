"""Post-hoc fixed-decimation AT curve projection and local closure diagnostics."""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recurrent_rg.checkpoints import load_run
from recurrent_rg.data.ensembles import load_ensembles
from recurrent_rg.evaluation import collect_trajectories
from recurrent_rg.data.ashkin_teller import observables


def main():
    torch.set_num_threads(2)
    folder=Path('results/ashkin_teller2d_g2_fixed_decimation_1200_seed91')
    run=load_run(folder);m=run.model;m.eval()
    tr,te=load_ensembles(run.config['data']['cache'])
    x=torch.cat([tr.configurations,te.configurations]);ks=torch.cat([tr.couplings,te.couplings])[:,0].numpy()
    t=collect_trajectories(m,x,samples=128,max_rg_steps=2)
    d=t['direct'];one=t['one_step'];knots=np.unique(ks)
    means=np.stack([d[ks==v].mean(0) for v in knots]);pred=np.stack([one[ks==v].mean(0) for v in knots])
    obs=[]
    for k in range(3):
        z=x[:,:,::2**k,::2**k].argmax(-1).numpy();stats=observables(z.reshape(-1,*z.shape[2:]))
        obs.append(np.stack([stats[name].reshape(z.shape[:2]).mean(1) for name in ('m_sigma','m_tau','m_product')],-1))
    obs=np.stack(obs,1);obsmean=np.stack([obs[ks==v].mean(0) for v in knots])
    roots=json.loads((folder/'report/fixed_points.json').read_text())['roots']
    for r in roots:
        r['projections']=[]
        for k in range(3):
            a=means[:-1,k];delta=np.diff(means[:,k],axis=0)
            f=np.clip(((np.array(r['g'])-a)*delta).sum(-1)/np.maximum((delta*delta).sum(-1),1e-20),0,1)
            residual=np.linalg.norm(a+f[:,None]*delta-r['g'],axis=-1);i=residual.argmin()
            r['projections'].append(dict(k=k,K=float(knots[i]+f[i]*(knots[i+1]-knots[i])),distance=float(residual[i])))
    out=folder/'fixed_audit';out.mkdir(exist_ok=True)
    report=dict(roots=roots,covariance_eigenvalues=[np.linalg.eigvalsh(np.cov(d[:,k].T,bias=True)).tolist() for k in range(3)],windows={},drift=[])
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    for k in range(3):
        for j in range(2):axes[0,0].plot(knots,means[:,k,j],ls=('-','--',':')[k],label=f'g{j+1}, k={k}')
        for j,name in enumerate(('sigma','tau','product')):
            axes[0,1].plot(knots,obsmean[:,k,j],ls=('-','--',':')[k],label=f'{name}, k={k}')
    for k in range(2):
        tangent=np.gradient(means[:,k],knots,axis=0);denom=np.maximum((tangent*tangent).sum(-1),1e-20)
        direct=((means[:,k+1]-means[:,k])*tangent).sum(-1)/denom
        recurrent=((pred[:,k]-means[:,k])*tangent).sum(-1)/denom
        report['drift'].append(dict(k=k,K=knots.tolist(),direct=direct.tolist(),R=recurrent.tolist()))
        axes[1,0].plot(knots,direct,label=f'direct, k={k}');axes[1,0].plot(knots,recurrent,'--',label=f'R, k={k}')
    for name,lo,hi in [('first',.165,.207),('second',.207,.245)]:
        mask=(ks>=lo)&(ks<=hi);err=np.sqrt(((one[mask]-d[mask,1:])**2).mean((0,2)));motion=np.sqrt(((d[mask,1:]-d[mask,:-1])**2).mean((0,2)))
        report['windows'][name]=dict(rmse=err.tolist(),direct_motion=motion.tolist(),error_over_motion=(err/motion).tolist())
    for i in [np.argmin(abs(knots-v)) for v in (.165,.189,.201,.225,.25)]:
        z=torch.from_numpy(means[i,0]);path=[z.numpy()]
        with torch.no_grad():
            for k in range(10):z=m.advance_theory(z);path.append(z.numpy())
        path=np.stack(path)
        axes[1,1].plot(path[:,0],path[:,1],'--',label=f'K={knots[i]:.3f}')
        axes[1,1].plot(means[i,:,0],means[i,:,1],'o-',ms=4)
    for ax,title in zip(axes.ravel(),['Q(C^k D)','Configuration order (finite-size)','K-tangent drift, NOT exact coupling RG','Direct (solid) vs R rollout (dashed)']):
        ax.set_title(title);ax.grid(alpha=.2);ax.legend(fontsize=7)
    for ax in (axes[0,0],axes[0,1],axes[1,0]):ax.set_xlim(.13,.28);ax.set_xlabel('K')
    axes[1,0].axhline(0,c='k',lw=.5);axes[1,1].set(xlabel='g1',ylabel='g2')
    fig.savefig(out/'diagnostic.png',dpi=160)
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='drift'},indent=2))
    for k in range(2):
        print('drift',k,[(round(float(K),4),round(a,5),round(b,5)) for K,a,b in zip(knots,report['drift'][k]['direct'],report['drift'][k]['R']) if .16<K<.25])


if __name__=='__main__':main()
