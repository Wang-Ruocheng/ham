"""Read-only model diagnostics for the two AT latent dimensions; no fitting."""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recurrent_rg.checkpoints import load_run
from recurrent_rg.data.ensembles import load_ensembles
from recurrent_rg.data.ashkin_teller import observables
from recurrent_rg.evaluation import scalar_flow, discrete_fixed_point
from recurrent_rg.empirical_mi import lattice_mi_statistics
from scipy.optimize import root


def main():
    torch.set_num_threads(2)
    out=Path('results/ashkin_teller_comparison_audit');out.mkdir(exist_ok=True)
    report={};tables=[]
    fig,axes=plt.subplots(2,4,figsize=(19,9),layout='constrained')
    for row,dim in enumerate((1,2)):
        folder=Path(f'results/ashkin_teller2d_g{dim}_random_mi_variance2ratio1_color1_1200_seed91')
        loaded=load_run(folder);m=loaded.model;m.eval()
        train,test=load_ensembles(loaded.config['data']['cache'])
        xs=torch.cat([train.configurations,test.configurations]);ks=torch.cat([train.couplings,test.couplings])[:,0].numpy()
        knots=np.unique(ks);ds=[];os=[]
        torch.manual_seed(12345)
        with torch.no_grad():
            for chunk in xs.split(8):
                levels=m.coarse_grain(chunk.double())
                ds.append(torch.stack([m.theory_encoder(x) for x in levels],1).numpy())
                obs=[]
                for x in levels:
                    colors=x.argmax(-1).numpy();shape=colors.shape
                    stats=observables(colors.reshape(-1,*shape[2:]))
                    obs.append(np.stack([stats[k].reshape(shape[:2]).mean(1) for k in ('m_sigma','m_tau','m_product')],-1))
                os.append(np.stack(obs,1))
            d=np.concatenate(ds);obs=np.concatenate(os)
            one=m.advance_theory(torch.from_numpy(d[:,:-1])).numpy()
            z=torch.from_numpy(d[:,0]);roll=[z.numpy()]
            for _ in range(12):z=m.advance_theory(z);roll.append(z.numpy())
            roll=np.stack(roll,1)
            codes=(torch.arange(256)[:,None]//(4**torch.arange(4)))%4
            patterns=torch.nn.functional.one_hot(codes,4).double().flatten(1)
            table=torch.softmax(m.rg_kernel.field_head.logits(m.rg_kernel.local_encoder(patterns)),-1)
        tables.append(table.numpy())
        means=np.stack([d[ks==v].mean(0) for v in knots]);om=np.stack([obs[ks==v].mean(0) for v in knots])
        rm=np.stack([roll[ks==v].mean(0) for v in knots]);op=np.stack([one[ks==v].mean(0) for v in knots])
        drift=means[:,1]-means[:,0];pred=op[:,0]-means[:,0]
        tangent=np.gradient(means[:,0],knots,axis=0)
        denom=np.maximum((tangent*tangent).sum(-1),1e-15)
        dk=(drift*tangent).sum(-1)/denom;pk=(pred*tangent).sum(-1)/denom
        rec=dict(K=knots.tolist(),mean_g=means.tolist(),observables=om.tolist(),
                 direct_projected_dK=dk.tolist(),R_projected_dK=pk.tolist(),
                 projection_note='Local tangent projection, not a closed effective coupling map',
                 covariance_eigenvalues=[np.linalg.eigvalsh(np.atleast_2d(np.cov(d[:,k].T,bias=True))).tolist() for k in range(3)],
                 per_scale_rmse=np.sqrt(((one-d[:,1:])**2).mean((0,2))).tolist())
        rec['windows']={}
        for name,lo,hi in [('first',.165,.207),('second',.207,.245),('highT',.08,.15),('lowT',.3,.65)]:
            mask=(ks>=lo)&(ks<=hi)
            err=np.sqrt(((one[mask]-d[mask,1:])**2).mean())
            movement=np.sqrt(((d[mask,1:]-d[mask,:-1])**2).mean())
            rec['windows'][name]=dict(rmse=float(err),direct_step_rms=float(movement),error_over_motion=float(err/movement))
        roots=[]
        if dim==1:
            roots=scalar_flow(m,-3,6,points=12001)['roots']
        else:
            lo=d.reshape(-1,dim).min(0)-.5;hi=d.reshape(-1,dim).max(0)+.5
            def fun(x):
                with torch.no_grad():return m.advance_theory(torch.tensor(x,dtype=torch.double)).numpy()-x
            seeds=np.array(np.meshgrid(*[np.linspace(a,b,17) for a,b in zip(lo,hi)])).reshape(2,-1).T
            for seed in np.concatenate([seeds,d.reshape(-1,2)]):
                fit=root(fun,seed);p=fit.x
                if np.linalg.norm(fun(p))>1e-7 or np.any(p<lo) or np.any(p>hi):continue
                if any(np.linalg.norm(p-np.array(r['g']))<1e-5 for r in roots):continue
                roots.append(discrete_fixed_point(m,p))
        for r in roots:
            p=np.array(r['g']);a=means[:-1,0];delta=np.diff(means[:,0],axis=0)
            frac=np.clip(((p-a)*delta).sum(-1)/np.maximum((delta*delta).sum(-1),1e-30),0,1)
            residual=np.linalg.norm(a+frac[:,None]*delta-p,axis=-1);i=int(residual.argmin())
            r['nearest_curve_K']=float(knots[i]+frac[i]*(knots[i+1]-knots[i]));r['curve_distance']=float(residual[i])
            if dim==1:
                r['all_inverse_K']=[float(x+(y-x)*(p[0]-ga)/(gb-ga)) for x,y,ga,gb in zip(knots[:-1],knots[1:],means[:-1,0,0],means[1:,0,0]) if ga!=gb and min(ga,gb)<=p[0]<=max(ga,gb)]
        rec['roots']=roots
        rec['C']={}
        for name,bits in [('sigma',2*(codes//2)-1),('tau',2*(codes%2)-1),('product',(2*(codes//2)-1)*(2*(codes%2)-1))]:
            outbits={'sigma':[-1,-1,1,1],'tau':[-1,1,-1,1],'product':[1,-1,-1,1]}[name]
            expected=table@torch.tensor(outbits,dtype=torch.double)
            majority=bits.sum(-1);mask=majority!=0
            rec['C'][name]=dict(majority_agreement=float(((1+expected[mask]*majority[mask].sign())/2).mean()))
            unanimous=bits.sum(-1).abs()==4
            rec['C'][name]['unanimous_retention']=float(((1+expected[unanimous]*bits[unanimous,0])/2).mean())
        # Test the actual AT Z2 x Z2 symmetries; all four-state permutations are not AT symmetries.
        equiv=[]
        with torch.no_grad():
            for xor in (1,2,3):
                p=torch.arange(4)^xor
                v=torch.nn.functional.one_hot(codes^xor,4).double().flatten(1)
                alt=torch.softmax(m.rg_kernel.field_head.logits(m.rg_kernel.local_encoder(v)),-1)
                equiv.append(float((alt[:,p]-table).abs().mean()))
        rec['C']['Z2xZ2_mean_probability_error']=equiv
        # Exact pushforward of spatially IID reference distributions: disjoint
        # blocks remain independent. These are structural probes, not new training data.
        iid={};code_np=codes.numpy();tab=table.numpy()
        signs=np.array([[-1,-1,1],[-1,1,-1],[1,-1,-1],[1,1,1]])
        for name,p0 in [('disordered',np.ones(4)/4),('product_plus',np.array([.5,0,0,.5])),
                        ('product_minus',np.array([0,.5,.5,0]))]:
            p=p0.copy();steps=[]
            for k in range(9):
                steps.append(dict(k=k,probabilities=p.tolist(),signed_order=(p@signs).tolist()))
                p=np.prod(p[code_np],axis=1)@tab
            iid[name]=steps
        rec['C']['iid_reference_pushforward']=iid
        for j,name in enumerate(('sigma','tau','product')):
            for k in range(3):axes[row,0].plot(knots,om[:,k,j],label=f'{name}, k={k}',ls=('-','--',':')[k])
        for j in range(dim):
            for k in range(3):axes[row,1].plot(knots,means[:,k,j],ls=('-','--',':')[k],label=f'g{j+1}, k={k}')
        axes[row,2].plot(knots,dk,label='Q(CD)-Q(D)');axes[row,2].plot(knots,pk,'--',label='R(Q(D))-Q(D)');axes[row,2].axhline(0,color='k',lw=.6)
        for i in [int(np.argmin(abs(knots-v))) for v in (.15,.189,.201,.225,.27)]:
            for j in range(dim):
                axes[row,3].plot(np.arange(13),rm[i,:,j],label=f'K={knots[i]:.3f}, g{j+1}')
                axes[row,3].plot(np.arange(3),means[i,:,j],'o',ms=3)
        for col,title in enumerate(('Direct configuration order','Q encoding by scale','Local K-tangent drift (not exact RG)','R rollout; dots = direct Q')):
            ax=axes[row,col];ax.set(title=f'dim_g={dim}: {title}',xlabel='K' if col<3 else 'k');ax.legend(fontsize=6);ax.grid(alpha=.2)
            if col<3:ax.set_xlim(.12,.29)
        # Actual empirical shuffle, same ensembles; probes only, no training.
        ids=[int(np.argmin(abs(ks-v))) for v in (.12,.189,.201,.225,.28)]
        info={}
        for name,seed in [('actual',None),('shuffled',551)]:
            stats=lattice_mi_statistics(xs[ids],states=4,shuffle_visible_seed=seed,sparse_conditionals=True)
            info[name]=dict(bits=(stats.information(table)/np.log(2)).tolist(),diagnostics=stats.diagnostics)
        rec['MI_probe']=dict(K=ks[ids].tolist(),**info)
        report[str(dim)]=rec
        np.savez_compressed(out/f'g{dim}_arrays.npz',K=ks,direct=d,one_step=one,rollout=roll,observables=obs,C_table=table.numpy())
        print(f'Finished dim {dim}',flush=True)
    report['C_max_difference_between_runs']=float(np.max(abs(tables[0]-tables[1])))
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    fig.savefig(out/'comparison.png',dpi=160)
    print(out,flush=True)


if __name__=='__main__':main()
