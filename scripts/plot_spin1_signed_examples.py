"""Show unmodified cached spin-1 configurations at matched signed couplings."""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from recurrent_rg.data.ensembles import load_ensembles


def main():
    torch.set_num_threads(2)
    cache=Path('data_cache/spin1_signedK_delta0_L36_n64_seed91/ensembles.pt')
    out=Path('results/spin1_signedK_configuration_examples');out.mkdir(exist_ok=True)
    train,test=load_ensembles(cache)
    datasets={'train':train,'test':test}
    cmap=ListedColormap(['#245da8','#eeeeee','#d94835'])
    norm=BoundaryNorm([-1.5,-.5,.5,1.5],3)
    legend=[Patch(facecolor=cmap(i),label=f's = {s:+d}') for i,s in enumerate([-1,0,1])]
    targets=[-.9,.9,-.59,.59,-.31,.31]
    sample_ids=[0,16,32,48]
    records=[];examples=[]
    fig,axs=plt.subplots(6,4,figsize=(11,15),layout='constrained')
    for row,target in enumerate(targets):
        candidates=[(abs(float(k)-target),split,i,float(k)) for split,ds in datasets.items()
                    for i,k in enumerate(ds.couplings.flatten())]
        _,split,idx,K=min(candidates)
        ds=datasets[split]
        ids=torch.where((ds.couplings.flatten()-K).abs()<1e-10)[0]
        spins=ds.configurations[ids].argmax(-1).numpy().astype(float)-1
        pooled=spins.reshape(-1,*spins.shape[-2:])
        eta=1-2*(np.indices(pooled.shape[-2:]).sum(0)%2)
        corr=.5*(pooled*np.roll(pooled,1,-1)+pooled*np.roll(pooled,1,-2)).mean()
        record=dict(K=K,split=split,ensemble_index=idx,sample_indices=sample_ids,
                    statistics_samples=len(pooled),abs_m=float(abs(pooled.mean((1,2))).mean()),
                    abs_staggered_m=float(abs((pooled*eta).mean((1,2))).mean()),
                    zero_fraction=float((pooled==0).mean()),nearest_neighbor=float(corr))
        records.append(record)
        chosen=ds.configurations[idx].argmax(-1).numpy()-1
        examples.append(chosen[0])
        for col,sample in enumerate(sample_ids):
            ax=axs[row,col];ax.imshow(chosen[sample],cmap=cmap,norm=norm,interpolation='nearest')
            ax.set_xticks([]);ax.set_yticks([])
            ax.set_title(f'sample {sample}',fontsize=9)
            if col==0:ax.set_ylabel(f'K = {K:+.4f} ({split})\n<|m|>={record["abs_m"]:.3f}, <|m_stag|>={record["abs_staggered_m"]:.3f}',fontsize=10)
    fig.suptitle('Actual cached spin-1 ensembles: 36 x 36, no coarse graining\nMatched |K|; fixed sample indices, no visual selection or sign alignment',fontsize=13)
    fig.legend(handles=legend,loc='outside lower center',ncol=3)
    fig.savefig(out/'signed_configurations.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(9,4.5),layout='constrained')
    for ax,s,K in zip(axs,examples[:2],[-.9,.9]):
        ax.imshow(s[:16,:16],cmap=cmap,norm=norm,interpolation='nearest')
        ax.set_xticks(np.arange(-.5,16,1),minor=True);ax.set_yticks(np.arange(-.5,16,1),minor=True)
        ax.grid(which='minor',color='#aaaaaa',linewidth=.3)
        ax.tick_params(which='both',bottom=False,left=False,labelbottom=False,labelleft=False)
        ax.set_title(f'K = {K:+.1f}: top-left 16 x 16 of sample 0')
    fig.legend(handles=legend,loc='outside lower center',ncol=3)
    fig.savefig(out/'ordered_zoom.png',dpi=170);plt.close(fig)
    (out/'metrics.json').write_text(json.dumps(dict(cache=str(cache.resolve()),records=records,
        note='Unmodified cache; sample indices predetermined. Statistics use both replicas (128 cached configurations); MCMC draws are not guaranteed independent.'),indent=2)+'\n')
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
