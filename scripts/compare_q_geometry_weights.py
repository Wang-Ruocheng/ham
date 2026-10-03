"""Compare a paired loss-weight change, without refitting or whitening."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--term',choices=('geometry','isotropy'),default='geometry');args=p.parse_args()
    term=args.term
    paths=[args.baseline,args.run];records=[json.loads((p/'report.json').read_text()) for p in paths]
    arrays=[np.load(p/'iso_both'/'encodings.npz') for p in paths]
    cloud=np.concatenate([z['test'][:,0] for z in arrays]);lo=cloud.min(0);hi=cloud.max(0);pad=.08*np.maximum(hi-lo,.1)
    fig,axs=plt.subplots(2,2,figsize=(11,9),layout='constrained')
    summary=[]
    for i,(record,z) in enumerate(zip(records,arrays)):
        weight=record.get(term+'_weight',10);arm=record['arms']['iso_both'];last=arm['history'][-1]['test'];g=z['test'][:,0];labels=z['test_labels']
        for j,ax in enumerate(axs[i]):
            sc=ax.scatter(*g.T,c=labels[:,j],s=16,cmap='viridis' if j==0 else 'coolwarm',norm=None if j==0 else SymLogNorm(linthresh=.001,vmin=-.5,vmax=.5),**({'vmin':.2,'vmax':.68} if j==0 else {}))
            ax.set(xlim=(lo[0]-pad[0],hi[0]+pad[0]),ylim=(lo[1]-pad[1],hi[1]+pad[1]),aspect='equal',xlabel='g1',ylabel='g2',title=f'{term.capitalize()} weight {weight}; weak PC {100*last["weak_fraction"][0]:.4g}%')
            fig.colorbar(sc,ax=ax,label='K' if j==0 else 'B (symmetric log)')
        terms=last['components'];summary.append(dict(varied_term=term,weight=weight,weak_fraction=last['weak_fraction'][0],eigenvalues=last['eigenvalues'][0],K_B_R2=arm['heldout_quadratic_K_B_R2'][0],raw_geometry=terms['geometry']/record.get('geometry_weight',10),raw_isotropy=terms['isotropy']/record.get('isotropy_weight',10),weighted_components=terms,common_weight10_total=last['loss']-terms[term]+terms[term]*10/weight))
    fig.suptitle('Held-out k=0; same initialization, batches and 2000 updates')
    fig.savefig(args.run/f'{term}_weight_comparison.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for r in records:
        h=r['arms']['iso_both']['history'];w=r.get(term+'_weight',10);x=[a['step'] for a in h]
        axs[0].plot(x,[a['test']['weak_fraction'][0]*100 for a in h],label=f'{term} {w}')
        for split,style in [('train','-'),('test','--')]:
            vals=[a[split]['loss']-a[split]['components'][term]+a[split]['components'][term]*10/w for a in h]
            axs[1].plot(x,vals,style,label=f'{w}, {split}')
    axs[0].set(xlabel='updates',ylabel='test weak variance fraction (%)',yscale='log');axs[0].legend()
    axs[1].set(xlabel='updates',ylabel='common weight-10 objective',yscale='log');axs[1].legend()
    fig.savefig(args.run/f'{term}_weight_metrics.png',dpi=180);plt.close(fig)
    (args.run/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
