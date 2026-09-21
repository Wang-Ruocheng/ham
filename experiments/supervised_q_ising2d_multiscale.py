"""Supervised Q-only diagnostic: infer ORIGINAL (K, B), not effective couplings.

Fixed binary decimation allows exact caching of 3x3 pattern frequencies.
head(histogram @ local_mlp(patterns)) is the existing Q, not a new encoder.
"""
import argparse
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.joint_qcr_ising2d_field import build_model


def patterns_for(q):
    width = q.geometry.patch_volume
    powers = 2 ** torch.arange(width)
    return (2*((torch.arange(2**width)[:, None]//powers) % 2)-1).double()


def frequencies(x, geometry):
    """Per-ensemble exact periodic patch frequencies; no sampling or binning approximation."""
    indices = torch.zeros(x.shape[:-1], dtype=torch.int64)
    for bit, offset in enumerate(geometry.patch_offsets()):
        spin = torch.roll(x[..., 0], tuple(-v for v in offset), (2, 3))
        indices += (spin > 0).long() * (2**bit)
    flat = indices.flatten(1)
    offsets = torch.arange(len(x))[:, None] * 512
    return torch.bincount((flat+offsets).flatten(), minlength=len(x)*512).reshape(len(x),512).double()/flat.shape[1]


def predict(q, histogram, patterns):
    return q.head(histogram @ q.local_mlp(patterns))


def encode(x, geometry):
    rows = []
    for start in range(0,len(x),4):
        batch = x[start:start+4]
        rows.append(torch.stack([frequencies(batch[:,:,::2**k,::2**k],geometry) for k in range(4)],1))
        if start % 200 == 0:
            print(f'Histogram cache: {min(start+4,len(x))}/{len(x)}',flush=True)
    return torch.cat(rows)


def metrics(pred, target, mean, scale):
    physical = pred*scale+mean
    err = physical-target[:,None,:]
    mse = err.square().mean(0)
    variance = target.var(0,unbiased=False)
    return {'normalized_mse': (err/scale).square().mean((0,2)).tolist(),
            'rmse': mse.sqrt().tolist(), 'r2': (1-mse/variance).tolist()}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=10000)
    parser.add_argument('--seed',type=int,default=91)
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    train_x,train_y,test_x,test_y=torch.load(args.cache,weights_only=True)
    train_y,test_y=train_y.double(),test_y.double()
    assert not set(map(tuple,train_y.tolist())) & set(map(tuple,test_y.tolist()))
    template=build_model(False).theory_encoder
    patterns=patterns_for(template)
    histpath=out/'exact_patch_histograms.pt'
    if histpath.exists():
        cached=torch.load(histpath,weights_only=True)
        assert cached['source']==str(args.cache.resolve())
        train_h,test_h=cached['train'],cached['test']
    else:
        train_h,test_h=encode(train_x,template.geometry),encode(test_x,template.geometry)
        torch.save({'source':str(args.cache.resolve()),'train':train_h,'test':test_h},histpath)
    # Verify the acceleration against actual Q outputs AND gradients at every scale.
    checks=[]
    for k in range(4):
        x=test_x[:2,:,::2**k,::2**k].double()
        direct=template(x)
        fast=predict(template,test_h[:2,k],patterns)
        torch.testing.assert_close(direct,fast,rtol=1e-10,atol=1e-12)
        a=torch.autograd.grad(direct.square().sum(),template.parameters())
        b=torch.autograd.grad(fast.square().sum(),template.parameters())
        maximum=max((u-v).abs().max().item() for u,v in zip(a,b))
        assert maximum<1e-10
        checks.append({'k':k,'output_max_error':(direct-fast).abs().max().item(),'gradient_max_error':maximum})
    del train_x,test_x
    mean,scale=train_y.mean(0),train_y.std(0,unbiased=False)
    target=(train_y-mean)/scale
    shared=copy.deepcopy(template)
    separate=torch.nn.ModuleList([copy.deepcopy(template) for _ in range(4)])
    optimizers=[torch.optim.Adam(shared.parameters(),lr=.001),torch.optim.Adam(separate.parameters(),lr=.001)]
    generator=torch.Generator().manual_seed(args.seed)
    config={'seed':args.seed,'steps':args.steps,'learning_rate':.001,'system_batch':32,
            'samples_per_ensemble':256,'train_ensembles':len(train_y),'test_ensembles':len(test_y),
            'grid_sizes':[32,16,8,4],'source':str(args.cache.resolve()),'Q_parameters':sum(p.numel() for p in template.parameters()),
            'target':'Original microscopic K=beta*J, B=beta*h at EVERY scale, not effective RG couplings',
            'normalization_mean':mean.tolist(),'normalization_std':scale.tolist(),
            'objective':'mean_{system,k,coordinate} ((Q(C^k D)-normalized_original_KB)^2)',
            'shared':'One unchanged Q, no k input, same paired-seed initialization as QCR',
            'separate':'Four independent copies of identical Q, each only sees one scale; control, not one 9560-parameter Q',
            'data':'Original disjoint physical-cell train/test split; no augmentation, all 256 samples; no R, A or unsupervised losses',
            'verification':checks}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    history=[];start=time.monotonic()
    def predictions(h):
        return {'shared':predict(shared,h,patterns),
                'separate':torch.stack([predict(q,h[:,k],patterns) for k,q in enumerate(separate)],1)}
    def evaluate(step):
        row={'step':step,'elapsed_seconds':time.monotonic()-start}
        with torch.no_grad():
            for split,h,y in [('train',train_h,train_y),('test',test_h,test_y)]:
                for name,pred in predictions(h).items():
                    row[f'{name}_{split}']=metrics(pred,y,mean,scale)
        history.append(row)
        (out/'history.json').write_text(json.dumps(history,indent=2)+'\n')
        if step % 1000 == 0:
            print('step',step,'shared test',row['shared_test']['normalized_mse'],
                  'separate test',row['separate_test']['normalized_mse'],flush=True)
    evaluate(0)
    for step in range(1,args.steps+1):
        ids=torch.randperm(len(train_y),generator=generator)[:32]
        for opt in optimizers: opt.zero_grad(set_to_none=True)
        ps=predict(shared,train_h[ids],patterns)
        loss=(ps-target[ids,None,:]).square().mean()
        loss.backward();optimizers[0].step()
        # Sum, rather than average, per-model objectives to match each one's learning scale.
        loss=sum((predict(q,train_h[ids,k],patterns)-target[ids]).square().mean() for k,q in enumerate(separate))
        loss.backward();optimizers[1].step()
        if step%100==0 or step==args.steps: evaluate(step)
        if step in {2000,args.steps}:
            torch.save({'shared':shared.state_dict(),'separate':separate.state_dict(),'mean':mean,'std':scale,
                        'optimizers':[o.state_dict() for o in optimizers], 'step':step,
                        'generator_state':generator.get_state()},out/f'checkpoint_{step}.pt')
    with torch.no_grad():
        final={f'{name}_{split}':(p*scale+mean).numpy()
               for split,h in [('train',train_h),('test',test_h)] for name,p in predictions(h).items()}
    np.savez_compressed(out/'predictions.npz',**final,train_KB=train_y.numpy(),test_KB=test_y.numpy())
    (out/'summary.json').write_text(json.dumps({'config':config,'final':history[-1],
           'at_2000':next((r for r in history if r['step']==2000),None)},indent=2)+'\n')
    fig,axes=plt.subplots(2,4,figsize=(15,7),sharex=True,sharey=True)
    for row,name in enumerate(('shared','separate')):
        for k in range(4):
            ax=axes[row,k]
            for split in ('train','test'):
                ax.semilogy([r['step'] for r in history],[r[f'{name}_{split}']['normalized_mse'][k] for r in history],label=split)
            ax.axvline(2000,color='gray',ls=':',lw=1)
            ax.set(title=f'{name} Q: k={k}, L={32//2**k}',xlabel='Training steps',ylabel='Normalized supervised MSE')
            ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Predict ORIGINAL (K, B) from decimated ensembles; seed 91; no R or geometry loss')
    fig.tight_layout();fig.savefig(out/'loss_by_scale.png',dpi=160);plt.close(fig)
    for name in ('shared','separate'):
        fig,axes=plt.subplots(2,4,figsize=(15,7))
        for d,label in enumerate(('K','B')):
            for k in range(4):
                ax=axes[d,k];truth=test_y[:,d].numpy();est=final[f'{name}_test'][:,k,d]
                ax.scatter(truth,est,c=test_y[:,1-d],cmap='coolwarm',s=10,alpha=.6)
                lo=min(truth.min(),est.min());hi=max(truth.max(),est.max())
                ax.plot([lo,hi],[lo,hi],'k--',lw=1)
                ax.set(xlabel=f'True original {label}',ylabel=f'Predicted {label}',title=f'k={k}; R2={history[-1][name+"_test"]["r2"][k][d]:.3f}')
                ax.grid(alpha=.2)
        fig.suptitle(f'{name.capitalize()} Q: full held-out parameter recovery at step {args.steps}')
        fig.tight_layout();fig.savefig(out/f'{name}_coupling_predictions.png',dpi=160);plt.close(fig)
    print(json.dumps(history[-1],indent=2),flush=True)


if __name__=='__main__':
    main()
