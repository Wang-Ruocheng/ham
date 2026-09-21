"""Paired, equal-parameter Q-only architecture controls on uniform (K,B) data."""
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
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.joint_qcr_ising2d_field import build_model
from experiments.supervised_q_ising2d_multiscale import patterns_for, metrics


class QVariant(nn.Module):
    def __init__(self, base, variant):
        super().__init__()
        self.local_mlp=copy.deepcopy(base.local_mlp)
        self.head=copy.deepcopy(base.head)
        self.variant=variant
        self.norm=nn.BatchNorm1d(12,eps=1e-5,momentum=.01,affine=False).double() if variant=='normalized' else nn.Identity()

    def features(self,h,patterns):
        if self.variant=='hierarchical':
            # h: [systems, scales, samples, 512], exact per-sample frequencies.
            u=h @ self.local_mlp[:4](patterns)
            return self.local_mlp[4:](u).mean(-2)
        return h @ self.local_mlp(patterns)

    def forward(self,h,patterns):
        z=self.features(h,patterns)
        shape=z.shape
        return self.head(self.norm(z.reshape(-1,12)).reshape(shape))

    @torch.no_grad()
    def calibrate(self,train_h,patterns):
        if self.variant=='normalized':
            z=self.features(train_h,patterns).reshape(-1,12)
            self.norm.running_mean.copy_(z.mean(0))
            self.norm.running_var.copy_(z.var(0,unbiased=False))


def cache_counts(x,geometry,path):
    shape=(len(x),4,256,512)
    if path.exists():
        arr=np.load(path,mmap_mode='r')
        assert arr.shape==shape and arr.dtype==np.int16
        return arr
    arr=np.lib.format.open_memmap(path,mode='w+',dtype=np.int16,shape=shape)
    for start in range(0,len(x),4):
        batch=x[start:start+4]
        for k in range(4):
            v=batch[:,:,::2**k,::2**k,0]
            index=torch.zeros(v.shape,dtype=torch.long)
            for bit,offset in enumerate(geometry.patch_offsets()):
                index+=(torch.roll(v,tuple(-o for o in offset),(2,3))>0).long()*2**bit
            flat=index.reshape(-1,v.shape[-1]**2)
            offsets=torch.arange(len(flat))[:,None]*512
            counts=torch.bincount((flat+offsets).flatten(),minlength=len(flat)*512).reshape(len(batch),256,512)
            arr[start:start+len(batch),k]=counts.numpy().astype(np.int16)
        if start%200==0: print('sample histograms',path.name,start,'/',len(x),flush=True)
    arr.flush()
    return arr


def sample_hist(counts,ids):
    # Integer counts fit exactly into int16 (maximum 1024); float64 matches baseline.
    h=torch.from_numpy(np.array(counts[ids],copy=True)).double()
    return h/torch.tensor([1024,256,64,16],dtype=torch.float64)[None,:,None,None]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--steps',type=int,default=10000)
    p.add_argument('--seed',type=int,default=91)
    args=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(args.seed)
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    train_x,train_y,test_x,test_y=torch.load(args.cache,weights_only=True)
    train_y,test_y=train_y.double(),test_y.double()
    base=build_model(False).theory_encoder
    patterns=patterns_for(base)
    previous=json.loads((args.baseline/'summary.json').read_text())
    assert previous['config']['source']==str(args.cache.resolve())
    assert previous['config']['seed']==args.seed and previous['config']['steps']==args.steps
    cached=torch.load(args.baseline/'exact_patch_histograms.pt',weights_only=True)
    train_h,test_h=cached['train'],cached['test']
    train_c=cache_counts(train_x,base.geometry,out/'train_sample_counts.npy')
    test_c=cache_counts(test_x,base.geometry,out/'test_sample_counts.npy')
    # Check every cached ensemble against the independently computed aggregate histogram.
    for counts,h in [(train_c,train_h),(test_c,test_h)]:
        for start in range(0,len(h),16):
            torch.testing.assert_close(sample_hist(counts,slice(start,start+16)).mean(2),h[start:start+16],rtol=0,atol=0)
    del train_x,test_x
    mean=train_y.mean(0);scale=train_y.std(0,unbiased=False);target=(train_y-mean)/scale
    config={'seed':args.seed,'steps':args.steps,'lr':.001,'system_batch':32,'samples':256,
            'data':str(args.cache.resolve()),'baseline':str(args.baseline.resolve()),'parameters_each':2390,
            'target':'standardized ORIGINAL K,B at every scale; not effective couplings',
            'normalization':'BatchNorm over system-scale pooled features; affine=False, eps=1e-5, momentum=.01; eval statistics recalibrated from full TRAIN only',
            'hierarchy':'9->60->12 Tanh at patch level; mean positions; existing 12->12 Tanh per configuration; mean configurations; unchanged 12->60->2 head',
            'evaluation':'full train/test sets every 500 steps, fixed endpoint; baseline reused unchanged',
            'sample_cache':'Exact per-configuration pattern counts, validated against every aggregate histogram'}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    all_hist={};final_predictions={};models={}
    for name in ['normalized','hierarchical']:
        q=QVariant(base,name).double()
        assert sum(v.numel() for v in q.parameters())==2390
        optimizer=torch.optim.Adam(q.parameters(),lr=.001)
        gen=torch.Generator().manual_seed(args.seed)
        history=[];start=time.monotonic()
        def predict_all(h,counts):
            if name!='hierarchical': return q(h,patterns)
            return torch.cat([q(sample_hist(counts,slice(i,i+16)),patterns) for i in range(0,len(h),16)])
        def evaluate(step):
            q.calibrate(train_h,patterns);q.eval()
            row={'step':step,'seconds':time.monotonic()-start}
            with torch.no_grad():
                for split,h,counts,y in [('train',train_h,train_c,train_y),('test',test_h,test_c,test_y)]:
                    pred=predict_all(h,counts)
                    row[split]=metrics(pred,y,mean,scale)
                    if step==args.steps: final_predictions[f'{name}_{split}']=(pred*scale+mean).numpy()
            history.append(row)
            (out/f'{name}_history.json').write_text(json.dumps(history,indent=2)+'\n')
            print(name,step,'seconds',round(row['seconds'],1),'test',row['test']['normalized_mse'],flush=True)
            q.train()
        evaluate(0)
        for step in range(1,args.steps+1):
            ids=torch.randperm(len(train_y),generator=gen)[:32]
            h=sample_hist(train_c,ids.numpy()) if name=='hierarchical' else train_h[ids]
            optimizer.zero_grad(set_to_none=True)
            loss=(q(h,patterns)-target[ids,None,:]).square().mean()
            loss.backward();optimizer.step()
            if step%500==0 or step==args.steps: evaluate(step)
            if step in {2000,args.steps}:
                torch.save({'Q':q.state_dict(),'optimizer':optimizer.state_dict(),'mean':mean,'std':scale,
                            'step':step,'variant':name,'generator_state':gen.get_state()},out/f'{name}_step{step}.pt')
        all_hist[name]=history;models[name]=q.eval()
    baseline_hist=json.loads((args.baseline/'history.json').read_text())
    all_hist['baseline']=[{'step':r['step'],'train':r['shared_train'],'test':r['shared_test']} for r in baseline_hist]
    baseline_pred=np.load(args.baseline/'predictions.npz')
    for split in ['train','test']: final_predictions[f'baseline_{split}']=baseline_pred[f'shared_{split}']
    np.savez_compressed(out/'predictions.npz',**final_predictions,train_KB=train_y.numpy(),test_KB=test_y.numpy())
    summary={'config':config,'final':{name:h[-1] for name,h in all_hist.items()}}
    # Fixed-K held-out RMSE, so strong-sign discrimination cannot conceal magnitude errors.
    regions={}
    for name in all_hist:
        regions[name]={}
        for k in np.unique(test_y[:,0].numpy()):
            mask=np.isclose(test_y[:,0].numpy(),k)
            error=final_predictions[name+'_test'][mask]-test_y.numpy()[mask,None,:]
            regions[name][str(k)]=np.sqrt((error**2).mean(0)).tolist()
    summary['test_rmse_by_K']=regions
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    colors={'baseline':'#777777','normalized':'#d55e00','hierarchical':'#0072b2'}
    fig,axes=plt.subplots(1,4,figsize=(16,4.5),sharey=True)
    for k,ax in enumerate(axes):
        for name,h in all_hist.items():
            for split,ls in [('train','--'),('test','-')]:
                ax.semilogy([r['step'] for r in h],[r[split]['normalized_mse'][k] for r in h],
                            color=colors[name],ls=ls,label=name+' '+split)
        ax.set(title=f'k={k}, L={32//2**k}',xlabel='Steps',ylabel='Normalized label MSE');ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle('Shared Q architecture controls: same 2390 parameters, seed 91, uniform original (K,B) labels')
    fig.tight_layout();fig.savefig(out/'loss_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(15,7))
    for d,label in enumerate(['K','B']):
        for k,ax in enumerate(axes[d]):
            for name in all_hist:
                ax.plot([r['step'] for r in all_hist[name]],[r['test']['rmse'][k][d] for r in all_hist[name]],color=colors[name],label=name)
            ax.set(title=f'{label}, k={k}',xlabel='Steps',ylabel='Test RMSE');ax.grid(alpha=.2)
    axes[0,0].legend();fig.tight_layout();fig.savefig(out/'rmse_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(3,4,figsize=(15,10),sharex=True,sharey=True)
    y=test_y.numpy()
    for row,name in enumerate(['baseline','normalized','hierarchical']):
        for k,ax in enumerate(axes[row]):
            ax.scatter(y[:,1],final_predictions[name+'_test'][:,k,1],c=y[:,0],cmap='viridis',vmin=.2,vmax=.68,s=9,alpha=.6)
            ax.plot([-.5,.5],[-.5,.5],'k--',lw=1)
            ax.set(title=f'{name}, k={k}: R2={summary["final"][name]["test"]["r2"][k][1]:.3f}',xlabel='True B',ylabel='Predicted B');ax.grid(alpha=.2)
    fig.suptitle('B recovery: points colored by true K (purple=0.20, yellow=0.68)')
    fig.tight_layout();fig.savefig(out/'B_predictions.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(3,4,figsize=(15,10),sharex=True,sharey=True)
    for row,K in enumerate([.2,.44,.68]):
        mask=np.isclose(y[:,0],K);bs=np.unique(y[mask,1])
        for k,ax in enumerate(axes[row]):
            for name in all_hist:
                pred=final_predictions[name+'_test'][:,k,1]
                means=[pred[mask&np.isclose(y[:,1],b)].mean() for b in bs]
                ax.plot(bs,means,'o-',ms=3,color=colors[name],label=name)
            ax.plot([-.5,.5],[-.5,.5],'k--',lw=1)
            ax.set(title=f'K={K:.2f}, k={k}',xlabel='True B (held-out cells)',ylabel='Mean predicted B');ax.grid(alpha=.2)
    axes[0,0].legend();fig.tight_layout();fig.savefig(out/'fixed_K_B_slices.png',dpi=180);plt.close(fig)
    print(json.dumps(summary['final'],indent=2),flush=True)


if __name__=='__main__':
    main()
