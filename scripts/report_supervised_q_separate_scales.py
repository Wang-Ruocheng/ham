"""Reevaluate saved independent-scale original Qs and plot their performance."""
import copy
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.joint_qcr_ising2d_field import build_model
from experiments.supervised_q_ising2d_multiscale import patterns_for, predict, metrics


def main():
    root=Path(__file__).resolve().parents[1]
    out=root/'results/supervised_Q_ising2d_uniform_KB_multiscale_h60_seed91'
    torch.set_num_threads(2)
    checkpoint=torch.load(out/'checkpoint_10000.pt',weights_only=True)
    base=build_model(False).theory_encoder
    qs=torch.nn.ModuleList([copy.deepcopy(base) for _ in range(4)])
    qs.load_state_dict(checkpoint['separate'],strict=True)
    assert all(q.head[0].in_features==12 for q in qs)
    assert all(sum(p.numel() for p in q.parameters())==2390 for q in qs)
    h=torch.load(out/'exact_patch_histograms.pt',weights_only=True)
    saved=np.load(out/'predictions.npz')
    history=json.loads((out/'history.json').read_text())
    patterns=patterns_for(base)
    results={}
    for split in ('train','test'):
        with torch.no_grad():
            p=torch.stack([predict(q,h[split][:,k],patterns) for k,q in enumerate(qs)],1)
        physical=(p*checkpoint['std']+checkpoint['mean']).numpy()
        np.testing.assert_allclose(physical,saved[f'separate_{split}'],rtol=1e-12,atol=1e-12)
        results[split]=metrics(p,torch.from_numpy(saved[f'{split}_KB']),checkpoint['mean'],checkpoint['std'])
        np.testing.assert_allclose(results[split]['normalized_mse'],history[-1][f'separate_{split}']['normalized_mse'],rtol=1e-12)
    fig,axes=plt.subplots(1,4,figsize=(15,3.8),sharey=True)
    for k,ax in enumerate(axes):
        for split,color in [('train','tab:blue'),('test','tab:orange')]:
            ax.semilogy([r['step'] for r in history],[r[f'separate_{split}']['normalized_mse'][k] for r in history],color=color,label=f'Independent {split}')
        ax.semilogy([r['step'] for r in history],[r['shared_test']['normalized_mse'][k] for r in history],'k--',alpha=.6,label='Shared Q test')
        ax.set(title=f'Independent Q{k}: L={32//2**k}',xlabel='Training step',ylabel='Normalized KB MSE')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'independent_scale_loss.png',dpi=160);plt.close(fig)
    y=saved['test_KB']
    fig,axes=plt.subplots(3,4,figsize=(14,10),sharex=True,sharey=True)
    for row,K in enumerate((.2,.44,.68)):
        mask=np.isclose(y[:,0],K)
        fields=np.unique(y[mask,1])
        for k in range(4):
            ax=axes[row,k]
            ax.plot([-.5,.5],[-.5,.5],'k--',lw=1,label='Ideal')
            for name,color in [('shared','tab:blue'),('separate','tab:orange')]:
                av=[saved[name+'_test'][mask & np.isclose(y[:,1],b),k,1].mean() for b in fields]
                ax.plot(fields,av,'o-',color=color,ms=3,label=name+' Q')
            ax.set(title=f'K={K:.2f}, k={k}',xlabel='True original B',ylabel='Predicted B')
            ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Original Q without magnetization feature; held-out cells, three-ensemble means')
    fig.tight_layout();fig.savefig(out/'independent_fixed_K_B_slices.png',dpi=160);plt.close(fig)
    (out/'independent_scale_verified.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
