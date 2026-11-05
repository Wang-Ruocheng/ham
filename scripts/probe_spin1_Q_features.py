"""Read-only frozen-Q layer probes; labels train diagnostic readouts only."""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recurrent_rg.checkpoints import load_run, build_model
from recurrent_rg.data.ensembles import load_ensembles
from experiments.q_only_cached_sweep import sample_counts


def probe(a, b, y, z):
    mu=a.mean(0); sd=np.maximum(a.std(0),1e-10)
    a=(a-mu)/sd; b=(b-mu)/sd
    a=np.column_stack([a,np.ones(len(a))]); b=np.column_stack([b,np.ones(len(b))])
    reg=np.eye(a.shape[1])*1e-3; reg[-1,-1]=0
    w=np.linalg.solve(a.T@a+reg,a.T@y)
    pred=b@w
    return dict(rmse=float(np.sqrt(np.mean((pred-z)**2))),
                r2=float(1-np.sum((pred-z)**2)/np.sum((z-z.mean())**2))),pred


def main():
    torch.set_num_threads(2)
    root=Path('results/spin1_Qonly_loss_candidates_exact_2400_seed91')
    out=root/'feature_audit';out.mkdir(exist_ok=True)
    base=load_run(root/'baseline.pt')
    tr,te=load_ensembles(base.config['data']['cache'])
    datasets=[tr,te]; hs=[]; correlations=[]
    for ds in datasets:
        h=[]; corr=[]
        for x in ds.configurations:
            h.append(np.asarray(sample_counts(x[None],base.model.geometry,3).mean(0)).ravel())
            s=x.argmax(-1).double()-1
            corr.append(float((s*(s.roll(1,-1)+s.roll(1,-2))/2).mean()))
        hs.append(np.stack(h));correlations.append(np.array(corr))
    ks=[d.couplings.numpy().reshape(len(d.configurations),-1)[:,0] for d in datasets]
    codes=(torch.arange(3**9)[:,None]//(3**torch.arange(9)))%3
    patterns=torch.nn.functional.one_hot(codes,3).flatten(1).double()
    report={}; fig,axes=plt.subplots(2,3,figsize=(14,8))
    for arm,ax in zip(['baseline','collision','ranking','future','manifold','random'],axes.flat):
        if arm=='random':
            torch.manual_seed(91);q=build_model(base.model_config,2).theory_encoder
        else:q=load_run(root/f'{arm}.pt').model.theory_encoder
        vals={};v=patterns
        with torch.no_grad():
            for j,layer in enumerate(q.local_mlp):
                v=layer(v)
                if j in (1,3,5):vals[f'pooled_local_{j}']=[h@v.numpy() for h in hs]
            pooled=vals['pooled_local_5']
            vals['head_hidden']=[q.head[:2](torch.from_numpy(h)).numpy() for h in pooled]
            vals['g']=[q.head(torch.from_numpy(h)).numpy() for h in pooled]
            direct=q(te.configurations[:1].double()).numpy()
            np.testing.assert_allclose(direct,vals['g'][1][:1],atol=1e-10,rtol=1e-10)
        report[arm]={}
        for name,(a,b) in vals.items():
            m,pred=probe(a,b,*ks)
            cm,_=probe(a,b,*correlations)
            keep=[abs(k)>.2 for k in ks]
            _,sp=probe(a[keep[0]],b[keep[1]],np.sign(ks[0][keep[0]]),np.sign(ks[1][keep[1]]))
            m.update(correlation_r2=cm['r2'],sign_accuracy_absK_above_point2=float(np.mean(np.sign(sp)==np.sign(ks[1][keep[1]]))))
            report[arm][name]=m
            if name=='pooled_local_5':ax.scatter(ks[1],pred,label='linear probe from pooled features',s=20)
        ax.scatter(ks[1],vals['g'][1][:,0],label='actual g',s=16)
        ax.plot([-.9,.9],[-.9,.9],'k--',lw=1);ax.set(title=arm,xlabel='K (held-out)',ylabel='probe K / actual g');ax.legend(fontsize=7)
        np.savez(out/f'{arm}_features.npz',**{f'{n}_{s}':v[i] for n,v in vals.items() for i,s in enumerate(['train','test'])})
    report['protocol']=dict(train_ensembles=len(ks[0]),test_ensembles=len(ks[1]),ridge=1e-3,standardization='train only',scope='microscopic k=0',labels='diagnostic readouts only; frozen Q/C/R',sign_test_n=int(np.sum(abs(ks[1])>.2)),limitations='One seed; linear decodability is not a proof of statistical sufficiency; correlated replicas are not independent coupling points')
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    fig.tight_layout();fig.savefig(out/'feature_vs_output.png',dpi=160);plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
