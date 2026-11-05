"""Compare literal full-ensemble Q(CD) with R(Q(D)), using frozen-C repeats."""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from recurrent_rg.checkpoints import load_run


def main():
    torch.set_num_threads(2)
    root=Path('results/spin1_geometry_ablation_seed91')
    out=root/'direct_vs_R';out.mkdir(exist_ok=True)
    runs=[('baseline','Baseline',Path('results/spin1_signedK_g1_block3_locked_recipe_h96_3000_seed91')),
          ('weight30','Geometry weight 30',root/'weight30'),
          ('tail50_top10','Tail mix 0.5',root/'tail50_top10')]
    summary={};norm=Normalize(-.9,.9);cmap=plt.get_cmap('coolwarm')
    figk,ak=plt.subplots(1,3,figsize=(16,4.4),layout='constrained')
    for col,(key,title,path) in enumerate(runs):
        loaded=load_run(path)
        z=np.load(path/'closure_conflicts/repeated_encodings.npz')
        means=z['fixed_cache_full'].mean(0);sd=z['fixed_cache_full'].std(0,ddof=1)
        K=z['labels'];g0=means[:,0];direct=means[:,1];noise=sd[:,1]
        with torch.no_grad():pred=loaded.model.advance_theory(torch.from_numpy(g0[:,None])).numpy().ravel()
        lo=min(g0.min(),direct.min(),pred.min());hi=max(g0.max(),direct.max(),pred.max());pad=.08*(hi-lo)
        fig,axes=plt.subplots(1,3,figsize=(16,4.4),layout='constrained',sharex=True,sharey=True)
        for ax,y,subtitle in zip(axes[:2],[direct,pred],['Direct: Q(CD)','Predicted: R(Q(D))']):
            if ax is axes[0]:ax.errorbar(g0,y,yerr=noise,fmt='none',ecolor='grey',alpha=.4,lw=.8)
            sc=ax.scatter(g0,y,c=K,cmap=cmap,norm=norm,s=22,edgecolors='none')
            ax.set_title(subtitle)
        ax=axes[2]
        ax.scatter(g0,direct,c=K,cmap=cmap,norm=norm,s=22,label='Q(CD)')
        order=np.argsort(g0);ax.plot(g0[order],pred[order],'k-',lw=1.6,label='R(Q(D))')
        ax.vlines(g0,np.minimum(direct,pred),np.maximum(direct,pred),color='grey',alpha=.35,lw=.7)
        ax.set_title('Overlay: vertical segments are residuals');ax.legend(fontsize=9)
        for ax in axes:
            ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],':',color='grey',alpha=.6)
            ax.set(xlabel='Q(D)',ylabel='Next-scale g',xlim=(lo-pad,hi+pad),ylim=(lo-pad,hi+pad));ax.grid(alpha=.2)
        fig.colorbar(sc,ax=axes,label='Initial K=J/T',shrink=.9)
        fig.suptitle(f'{title}, epoch 3000; full-ensemble Q; direct target averaged over 12 C draws')
        fig.savefig(out/f'{key}.png',dpi=170);plt.close(fig)
        knots=np.unique(K);qmean=np.array([direct[K==v].mean() for v in knots]);rmean=np.array([pred[K==v].mean() for v in knots])
        ax=ak[col];ax.plot(knots,qmean,'o-',ms=3,label='Q(CD)');ax.plot(knots,rmean,'s--',ms=3,label='R(Q(D))')
        ax.fill_between(knots,np.minimum(qmean,rmean),np.maximum(qmean,rmean),color='grey',alpha=.15)
        ax.set(title=title,xlabel='Initial K=J/T',ylabel='Next-scale g');ax.legend();ax.grid(alpha=.2)
        summary[key]=dict(full_ensemble=True,C_draws=len(z['fixed_cache_full']),
            all_ensembles_rmse=float(np.sqrt(np.mean((direct-pred)**2))),
            test_rmse=float(np.sqrt(np.mean((direct[z['splits']=='test']-pred[z['splits']=='test'])**2))),
            mean_C_draw_sd=float(noise.mean()),max_abs_residual=float(abs(direct-pred).max()),
            note='Same 64 stored configurations per ensemble. Q(CD) is mean over repeated C draws. Full Q differs slightly from the training half-split mean. No extrapolation of model outside input range in overlay.')
    figk.savefig(out/'by_K.png',dpi=170);plt.close(figk)
    (out/'metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
