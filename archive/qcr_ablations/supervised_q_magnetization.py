"""Isolated supervised control: original Q plus signed current-scale mean spin."""
import copy
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.joint_qcr_ising2d_field import build_model
from experiments.supervised_q_ising2d_multiscale import patterns_for, predict, metrics


class MagnetizationQ(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.local_mlp = copy.deepcopy(base.local_mlp)
        self.head = copy.deepcopy(base.head)
        first = base.head[0]
        expanded = nn.Linear(first.in_features + 1, first.out_features).to(first.weight)
        with torch.no_grad():
            expanded.weight[:, :-1].copy_(first.weight)
            expanded.weight[:, -1].zero_()
            expanded.bias.copy_(first.bias)
        self.head[0] = expanded
        offsets = list(base.geometry.patch_offsets())
        self.center = offsets.index((0, 0))

    def forward(self, h, patterns):
        pooled = h @ self.local_mlp(patterns)
        magnetization = (h @ patterns[:, self.center]).unsqueeze(-1)
        return self.head(torch.cat((pooled, magnetization), dim=-1))


def main():
    root = Path(__file__).resolve().parents[2]
    baseline = root / 'results/supervised_Q_ising2d_uniform_KB_multiscale_h60_seed91'
    out = root / 'results/supervised_Q_magnetization_uniform_KB_seed91'
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(91)
    base = build_model(False).theory_encoder
    q = MagnetizationQ(base)
    patterns = patterns_for(base)
    cache = torch.load(baseline/'exact_patch_histograms.pt', weights_only=True)
    tx, ty, vx, vy = torch.load(cache['source'], weights_only=True)
    ty, vy = ty.double(), vy.double()
    th, vh = cache['train'], cache['test']
    checks = []
    for k in range(4):
        actual = vx[:3, :, ::2**k, ::2**k].double().mean((1,2,3,4))
        cached = vh[:3,k] @ patterns[:,q.center]
        torch.testing.assert_close(actual, cached, rtol=0, atol=1e-14)
        torch.testing.assert_close(q(vh[:3,k], patterns), predict(base,vh[:3,k],patterns))
        checks.append(float((actual-cached).abs().max()))
    del tx, vx
    mean, std = ty.mean(0), ty.std(0, unbiased=False)
    target = (ty-mean)/std
    config = {'seed':91, 'steps':10000, 'lr':.001, 'system_batch':32,
              'samples_per_ensemble':256, 'train_ensembles':len(ty), 'test_ensembles':len(vy),
              'Q_parameters':sum(p.numel() for p in q.parameters()),
              'source':cache['source'], 'baseline':str(baseline),
              'scales':[32,16,8,4], 'feature':'Signed mean spin of CURRENT scale ensemble',
              'initialization':'Same original Q weights; added magnetization column zero; initial outputs identical',
              'target':'Original K,B at every scale, train-only standardization; equal-scale MSE',
              'normalization_mean':mean.tolist(), 'normalization_std':std.tolist(),
              'magnetization_cache_max_errors':checks}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    optimizer = torch.optim.Adam(q.parameters(), lr=.001)
    generator = torch.Generator().manual_seed(91)
    history = []
    start = time.monotonic()
    def evaluate(step):
        with torch.no_grad():
            row = {'step':step, 'seconds':time.monotonic()-start,
                   'train':metrics(q(th,patterns),ty,mean,std),
                   'test':metrics(q(vh,patterns),vy,mean,std)}
        history.append(row)
        if step%1000 == 0:
            print(step, 'test MSE:', row['test']['normalized_mse'], flush=True)
    evaluate(0)
    for step in range(1,10001):
        ids = torch.randperm(len(ty),generator=generator)[:32]
        optimizer.zero_grad(set_to_none=True)
        loss = (q(th[ids],patterns)-target[ids,None,:]).square().mean()
        loss.backward()
        optimizer.step()
        if step%100 == 0:
            evaluate(step)
        if step in (2000,10000):
            torch.save({'Q':q.state_dict(), 'optimizer':optimizer.state_dict(),
                        'mean':mean, 'std':std, 'step':step,
                        'generator_state':generator.get_state()},out/f'checkpoint_{step}.pt')
    bh = json.loads((baseline/'history.json').read_text())
    bp = np.load(baseline/'predictions.npz')
    np.testing.assert_array_equal(bp['test_KB'],vy.numpy())
    np.testing.assert_array_equal(bp['train_KB'],ty.numpy())
    with torch.no_grad():
        train_pred = (q(th,patterns)*std+mean).numpy()
        test_pred = (q(vh,patterns)*std+mean).numpy()
    np.savez_compressed(out/'predictions.npz',train=train_pred,test=test_pred,
                        baseline_test=bp['shared_test'],train_KB=ty.numpy(),test_KB=vy.numpy())
    summary = {'config':config, 'final':history[-1], 'baseline_final':bh[-1],
               'magnetization_weight_norm':q.head[0].weight[:,-1].norm().item()}
    (out/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig, axes = plt.subplots(1,4,figsize=(15,3.7),sharey=True)
    for k,ax in enumerate(axes):
        for records,prefix,color,name in [(bh,'shared_','tab:blue','Original'),(history,'','tab:orange','+ mean spin')]:
            for split,ls in [('train','--'),('test','-')]:
                ax.semilogy([r['step'] for r in records], [r[prefix+split]['normalized_mse'][k] for r in records],
                            color=color,ls=ls,label=f'{name} {split}')
        ax.set(title=f'k={k}, L={32//2**k}',xlabel='Training step',ylabel='Normalized MSE')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'loss_comparison.png',dpi=160);plt.close(fig)
    fig,axes = plt.subplots(3,4,figsize=(14,10),sharex=True,sharey=True)
    truth = vy.numpy()
    for row,K in enumerate((.2,.44,.68)):
        mask = np.isclose(truth[:,0],K)
        B = np.unique(truth[mask,1])
        for k in range(4):
            ax=axes[row,k]
            ax.plot([-.5,.5],[-.5,.5],'k--',lw=1,label='Ideal')
            for pred,color,label in [(bp['shared_test'],'tab:blue','Original'),(test_pred,'tab:orange','+ mean spin')]:
                av=[pred[mask & np.isclose(truth[:,1],b),k,1].mean() for b in B]
                ax.plot(B,av,'o-',color=color,ms=3,label=label)
            ax.set(title=f'K={K:.2f}, k={k}',xlabel='True original B',ylabel='Predicted B')
            ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Held-out cells only; points average three independent ensembles')
    fig.tight_layout();fig.savefig(out/'fixed_K_B_slices.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(14,7))
    for d,label in enumerate(('K','B')):
        for k in range(4):
            ax=axes[d,k]
            ax.scatter(truth[:,d],test_pred[:,k,d],c=truth[:,1-d],cmap='coolwarm',s=8,alpha=.5)
            ax.plot([truth[:,d].min(),truth[:,d].max()],[truth[:,d].min(),truth[:,d].max()],'k--',lw=1)
            ax.set(xlabel=f'True original {label}',ylabel=f'Predicted {label}',
                   title=f'k={k}; R2={history[-1]["test"]["r2"][k][d]:.3f}')
    fig.tight_layout();fig.savefig(out/'coupling_predictions.png',dpi=160);plt.close(fig)
    print(json.dumps(summary['final'],indent=2),flush=True)


if __name__ == '__main__':
    main()
