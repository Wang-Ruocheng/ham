"""Independent-data sample-size audit of frozen spin-1 Q/C; no optimization."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from recurrent_rg.checkpoints import load_run
from recurrent_rg.data.blume_capel import sample_signed_spin1, observable_series
from recurrent_rg.runner import file_fingerprint


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);started=time.perf_counter()
    loaded=load_run(a.run);model=loaded.model.eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    before=file_fingerprint(loaded.checkpoint)
    coupling=np.array([-.69,-.675,-.6652173913043479,-.655,-.64,.82,.8608695652173913,.9])
    sizes=[64,256,1024];replicas=4;draws=3;length=36
    starts=['random','plus','minus','zero']
    metadata=dict(checkpoint=str(loaded.checkpoint),checkpoint_sha256=before,K=coupling.tolist(),
        sizes=sizes,replicas=replicas,C_draws=draws,length=length,burn=4096,spacing=16,
        seed_formula='1700000 + 1000*K_index + replica',starts_in_ferromagnetic_gauge=starts,
        no_retraining=True,protocol='Independent MCMC chains, unrelated to training seeds; nested N prefixes within each chain. Three C draws; full-ensemble and half-split encodings both computed.',
        limitation='Fixed L=36; four independent chains per K, not a thermodynamic-limit study. Error bars are between-chain SD, not iid-sample confidence bounds.')
    (a.output/'protocol.json').write_text(json.dumps(metadata,indent=2)+'\n')
    enc=np.zeros((len(coupling),replicas,draws,len(sizes),3,2))
    obs=[]
    for i,K in enumerate(coupling):
        for rep in range(replicas):
            seed=1700000+1000*i+rep
            raw=sample_signed_spin1(float(K),length,1024,seed,burn=4096,spacing=16,start=starts[rep])
            np.savez_compressed(a.output/f'raw_K{i:02d}_rep{rep}.npz',spins=raw,K=K,seed=seed)
            series=observable_series(raw,delta=0.,exchange=K)
            eta=1-2*(np.indices((length,length)).sum(0)%2)
            series['abs_staggered_magnetization']=abs((raw*eta).mean((1,2)))
            obs.append(dict(K=float(K),replica=rep,start=starts[rep],observables={key:dict(
                mean=float(v.mean()),first_half=float(v[:512].mean()),second_half=float(v[512:].mean()),
                lag1=float(np.corrcoef(v[:-1],v[1:])[0,1]) if v.std()>0 else None) for key,v in series.items()}))
            x=torch.nn.functional.one_hot(torch.from_numpy(raw.astype(np.int64)+1),3)[None].double()
            with torch.random.fork_rng(devices=[]),torch.no_grad():
                for draw in range(draws):
                    torch.manual_seed(2700000+1000*i+10*rep+draw)
                    levels=model.coarse_grain(x)
                    for ni,n in enumerate(sizes):
                        for k,level in enumerate(levels):
                            z=level[:,:n]
                            enc[i,rep,draw,ni,k,0]=model.theory_encoder(z).item()
                            enc[i,rep,draw,ni,k,1]=(.5*(model.theory_encoder(z[:,:n//2])+model.theory_encoder(z[:,n//2:]))).item()
            np.savez_compressed(a.output/'encodings_progress.npz',encodings=enc[:i+1],K=coupling[:i+1],completed_K=i,completed_replica=rep,sizes=sizes)
            print(f'K {i+1}/{len(coupling)}, chain {rep+1}/{replicas}, elapsed {time.perf_counter()-started:.1f}s',flush=True)
    np.savez_compressed(a.output/'encodings.npz',encodings=enc,K=coupling,sizes=sizes)
    (a.output/'sampling_audit.json').write_text(json.dumps(obs,indent=2)+'\n')
    # Match the training definition: average Q of two half ensembles.
    chain=enc[...,1].mean(2);mean=chain.mean(1);sd=chain.std(1,ddof=1)
    summary=[]
    for ni,n in enumerate(sizes):
        i,j=2,7
        pair=[]
        for idx in [i,j]:
            pair.append(dict(K=float(coupling[idx]),g=mean[idx,ni].tolist(),between_chain_sd=sd[idx,ni].tolist(),
                C_draw_sd=np.sqrt(enc[idx,:,:,ni,:,1].var(axis=1,ddof=1).mean(0)).tolist()))
        summary.append(dict(N=n,predeclared_pair=pair,input_gap=float(abs(mean[i,ni,0]-mean[j,ni,0])),
            target_gap=float(abs(mean[i,ni,1]-mean[j,ni,1])),
            input_difference_SE=float(np.sqrt(sd[i,ni,0]**2+sd[j,ni,0]**2)/np.sqrt(replicas)),
            target_difference_SE=float(np.sqrt(sd[i,ni,1]**2+sd[j,ni,1]**2)/np.sqrt(replicas))))
    # Interpolate only inside each sampled branch's input range; no extrapolation.
    branch_tests=[]
    for ni,n in enumerate(sizes):
        branch=[]
        for mask in [coupling<0,coupling>0]:
            ids=np.flatnonzero(mask);ids=ids[np.argsort(mean[ids,ni,0])]
            branch.append((mean[ids,ni,0],mean[ids,ni,1]))
        lo=max(b[0].min() for b in branch);hi=min(b[0].max() for b in branch)
        if lo<=hi:
            grid=np.linspace(lo,hi,200)
            gap=np.abs(np.interp(grid,*branch[0])-np.interp(grid,*branch[1]))
            branch_tests.append(dict(N=n,overlapping_input_interval=[float(lo),float(hi)],
                interpolated_target_gap_range=[float(gap.min()),float(gap.max())],
                note='Piecewise-linear interpolation diagnostic, not proof of exact collisions.'))
    after=file_fingerprint(loaded.checkpoint)
    assert before==after,'Checkpoint changed during diagnostic'
    report=dict(protocol=metadata,results=summary,branch_interpolation=branch_tests,
        full_vs_split_rmse_by_N=np.sqrt(((enc[...,0]-enc[...,1])**2).mean(axis=(0,1,2,4))).tolist(),
        checkpoint_unchanged=True,elapsed_seconds=time.perf_counter()-started)
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(16,4.8),layout='constrained')
    for ni,(n,ax) in enumerate(zip(sizes,axes)):
        for mask,color,label in [(coupling<0,'tab:blue','Negative K'),(coupling>0,'tab:red','Positive K')]:
            ids=np.flatnonzero(mask);ids=ids[np.argsort(mean[ids,ni,0])]
            ax.errorbar(mean[ids,ni,0],mean[ids,ni,1],xerr=sd[ids,ni,0],yerr=sd[ids,ni,1],fmt='o-',color=color,label=label,capsize=3)
        for idx in [2,7]:ax.annotate(f'K={coupling[idx]:.4f}',mean[idx,ni,:2],xytext=(4,6),textcoords='offset points',fontsize=8)
        gx=np.linspace(mean[:,ni,0].min()-.03,mean[:,ni,0].max()+.03,200)
        with torch.no_grad():gy=model.advance_theory(torch.from_numpy(gx[:,None])).numpy().ravel()
        ax.plot(gx,gy,'k--',label='Frozen R');ax.set(xlabel='g0',ylabel='g1',title=f'N={n} per ensemble');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Independent diagnostic data; frozen Q/C/R; bars: SD across 4 chains (3 C draws averaged)')
    fig.savefig(a.output/'sample_size_branches.png',dpi=170);plt.close(fig)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
