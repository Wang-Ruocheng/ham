"""Display actual frozen stochastic-C trajectories used by recent Q-only runs."""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.patches import Patch
from recurrent_rg.data.ensembles import load_ensembles
from recurrent_rg.runner import file_fingerprint
from experiments.q_only_candidates import validate_frozen_levels


def main():
    torch.set_num_threads(2)
    cache=Path('results/spin1_Qonly_cache_exact_seed91')
    metadata=json.loads((cache/'manifest.json').read_text())
    path=cache/'frozen_levels.pt'
    assert file_fingerprint(path)==metadata['cached_levels_sha256']
    levels=torch.load(path,weights_only=True)
    train,test=load_ensembles('data_cache/spin1_signedK_delta0_L36_n64_seed91/ensembles.pt');datasets=dict(train=train,test=test)
    for split,ls in levels.items():
        validate_frozen_levels(ls);torch.testing.assert_close(ls[0],datasets[split].configurations)
    out=Path('results/spin1_current_C_scale_examples');out.mkdir(exist_ok=True)
    cmap=ListedColormap(['#245da8','#eeeeee','#d94835']);norm=BoundaryNorm([-1.5,-.5,.5,1.5],3)
    targets=[-.9,.9,-.59,.59,-.31,.31];sample_ids=[0,16]
    fig,axes=plt.subplots(6,6,figsize=(14,14),layout='constrained')
    records=[]
    for row,target in enumerate(targets):
        choices=[(abs(float(k)-target),split,i,float(k)) for split,d in datasets.items() for i,k in enumerate(d.couplings.flatten())]
        _,split,idx,K=min(choices);ids=torch.where((datasets[split].couplings.flatten()-K).abs()<1e-10)[0]
        record=dict(K=K,split=split,display_ensemble=idx,samples=sample_ids,scales=[])
        for scale,v in enumerate(levels[split]):
            pooled=v[ids].argmax(-1).numpy().astype(float)-1;pooled=pooled.reshape(-1,*pooled.shape[-2:]);L=pooled.shape[-1]
            eta=1-2*(np.indices((L,L)).sum(0)%2)
            record['scales'].append(dict(k=scale,L=L,samples=len(pooled),zero_fraction=float((pooled==0).mean()),abs_m=float(abs(pooled.mean((1,2))).mean()),abs_staggered_m=float(abs((pooled*eta).mean((1,2))).mean()),nearest_neighbor=float((pooled*(np.roll(pooled,1,-1)+np.roll(pooled,1,-2))/2).mean())))
            for j,sample in enumerate(sample_ids):
                ax=axes[row,scale*2+j];s=v[idx,sample].argmax(-1).numpy()-1
                ax.imshow(s,cmap=cmap,norm=norm,interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
                if row==0:ax.set_title(f'k={scale}, {L}x{L}\nsample {sample}',fontsize=11)
                if scale==0 and j==0:ax.set_ylabel(f'K={K:+.4f}',fontsize=12)
        records.append(record)
    fig.suptitle('Actual configurations under the frozen learned probabilistic C\nSame samples across scales; no sign alignment or sample selection; 3x3 blocks, stride 3',fontsize=14)
    fig.legend(handles=[Patch(facecolor=cmap(i),label=f's={s:+d}') for i,s in enumerate([-1,0,1])],loc='outside lower center',ncol=3)
    fig.savefig(out/'configurations_scales.png',dpi=160);plt.close(fig)
    (out/'metrics.json').write_text(json.dumps(dict(source_C=metadata.get('source_checkpoint',metadata.get('source')),cache_sha256=metadata['cached_levels_sha256'],coarse_draw_seed=9191,records=records,notes='Statistics use all cached configurations from matching K replicas; one stochastic C draw; no regeneration. Coarse colors are categorical output labels, not automatically physical microscopic spins.'),indent=2))
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
