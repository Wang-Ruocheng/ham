"""Critical-region direct/rollout audit in fixed train-only PCA coordinates."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root=Path(__file__).resolve().parents[1]
    out=root/'results/joint_qcr_ising2d_uniform_field_g2_h60_L32_fixedC_2000_seed91_cached/step2000'
    data=np.load(out/'field_flow_data.npz')
    direct,rollout=data['direct'],data['rollout']
    train=direct[~data['is_test'].astype(bool),0]
    center=train.mean(0)
    _,s,v=np.linalg.svd(train-center,full_matrices=False)
    scale=s/np.sqrt(len(train)-1)
    # One coordinate system for all k and both paths; no per-scale re-whitening.
    d=(direct-center)@v.T/scale
    r=(rollout-center)@v.T/scale
    groups=[('Temperature direction: B=0',[(.425,0),(.44,0),(.455,0)]),
            ('Field direction: K=0.44',[(.44,-.05),(.44,0),(.44,.05)])]
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
    gapfig,gapaxes=plt.subplots(1,2,figsize=(11,4))
    steps=np.arange(d.shape[1]); records=[]
    for col,(title,cases) in enumerate(groups):
        dm=[];rm=[]
        for i,(K,B) in enumerate(cases):
            mask=np.isclose(data['KB'][:,0],K)&np.isclose(data['KB'][:,1],B,atol=1e-12)
            assert mask.sum()==3
            a,b=d[mask],r[mask]
            am,bm=a.mean(0),b.mean(0)
            dm.append(am);rm.append(bm)
            label=f'K={K:.3f}, B={B:+.2f}'
            for axis in range(2):
                ax=axes[axis,col]
                ax.errorbar(steps,am[:,axis],yerr=a.std(0,ddof=1)[:,axis],fmt='o-',color=f'C{i}',capsize=3,label=label)
                ax.plot(steps,bm[:,axis],'x--',color=f'C{i}')
                ax.set(ylabel=f'PC{axis+1} / microscopic train SD',xticks=steps)
                ax.grid(alpha=.2)
            records.append({'K':K,'B':B,'split':'test' if data['is_test'][mask].all() else 'train',
                            'direct':am.tolist(),'rollout':bm.tolist(),
                            'direct_replica_sd':a.std(0,ddof=1).tolist(),
                            'raw_direct':direct[mask].mean(0).tolist(),
                            'raw_rollout':rollout[mask].mean(0).tolist()})
        axes[0,col].set_title(title);axes[0,col].legend(fontsize=8)
        axes[1,col].set_xlabel('RG step k (L=32,16,8,4)')
        # Separation between the two outer parameter cells, NOT between adjacent k.
        for paths,ls,name in [(dm,'o-','Direct'),(rm,'x--','R rollout')]:
            delta=paths[2]-paths[0]
            for axis in range(2):
                gapaxes[col].plot(steps,np.abs(delta[:,axis]),ls,color=f'C{axis}',label=f'{name} PC{axis+1}')
        gapaxes[col].set(title=title,xlabel='RG step k',ylabel='Outer-cell separation / train SD',xticks=steps)
        gapaxes[col].grid(alpha=.2);gapaxes[col].legend(fontsize=8)
    fig.suptitle('Fixed original QCR checkpoint: direct Q(C^k D) (solid circles) vs R^k(Q(D)) (dashed crosses)')
    fig.text(.5,.015,'PC1: strong direction; PC2: weak direction. Fixed PCA fitted only to microscopic training ensembles.\n'
             'Error bars: SD of 3 independent ensembles. K=0.44 is near, not exactly at, criticality; |B|=0.05 is not infinitesimal.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.95));fig.savefig(out/'critical_closure_pca.png',dpi=170);plt.close(fig)
    gapfig.tight_layout();gapfig.savefig(out/'critical_pair_separation.png',dpi=170);plt.close(gapfig)
    result={'coordinate_center':center.tolist(),'pca_rows':v.tolist(),'coordinate_std':scale.tolist(),
            'weak_variance_fraction':float(scale[1]**2/(scale**2).sum()),'cases':records}
    (out/'critical_closure_pca.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
