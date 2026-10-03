"""Can independently sampled small systems replace long QCR scale chains?

Three fresh, paired-initialization QCR fits, same loss and update budget.
No long-system training configurations/derived coarse levels enter short fits.
Labels are used for splitting and diagnostics only. Finite beta=16, dtau=.5.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
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
from recurrent_rg.tfim1d_cluster import sample_tfim1d_chain
from recurrent_rg.model_nd import RecurrentRGNetwork1D
from recurrent_rg.data.views import balanced_views
from recurrent_rg.training import train
from recurrent_rg.data.statistics import two_spin_distribution
from recurrent_rg.evaluation import find_fixed_points_1d


def sample_task(task):
    size, index, replica, seed = task
    h = np.linspace(.01, 2., 64)[index]
    x = sample_tfim1d_chain(float(h), sites=size, seed=seed,
                           initial=(0,1,-1)[replica % 3])
    return task, x


def make_data(out, seed, workers):
    # Disjoint seed ranges per size; test couplings withheld in their entirety.
    tasks = []
    for size, offset in [(128, 0), (16, 100000)]:
        for i in range(64):
            replicas = 24 if size == 16 and i % 3 != 1 else 3
            tasks.extend((size,i,r,seed*1000000+offset+i*100+r) for r in range(replicas))
    arrays = {16: [], 128: []}; labels = {16: [],128: []}; metadata = {16: [],128: []}
    started = time.perf_counter()
    # Compile before thread dispatch. Numba has per-thread RNG states.
    sample_tfim1d_chain(1., draws=1, equil=1, gap=1)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n, (task, x) in enumerate(pool.map(sample_task, tasks), 1):
            size,i,r,chain_seed = task
            arrays[size].append(torch.from_numpy(x).unsqueeze(-1))
            labels[size].append(float(np.linspace(.01,2.,64)[i]))
            metadata[size].append(dict(index=i, replica=r, seed=chain_seed, test=i % 3 == 1))
            if n % 100 == 0 or n == len(tasks):
                print(f'Sampling {n}/{len(tasks)}, {time.perf_counter()-started:.1f}s', flush=True)
    result = {}
    for size in (128,16):
        x = torch.stack(arrays[size]); h = torch.tensor(labels[size],dtype=torch.float64)
        test = torch.tensor([row['test'] for row in metadata[size]])
        result[size] = (x[~test], h[~test], x[test], h[test])
        torch.save(result[size],out/f'ensembles_L{size}.pt')
    (out/'sampling_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return result


def trajectories(model, x):
    # Historical experiment terminates at four sites; no mutable model depth.
    steps = int(np.log2(x.shape[2] // 4))
    if x.shape[2] != 4 * 2**steps:
        raise ValueError('This comparison requires a power-of-two chain ending at L4')
    ds,rs,ss = [],[],[]
    samples=x.shape[1]
    with torch.no_grad():
        for chunk in x.split(16):
            levels = model.coarse_grain(balanced_views(chunk,samples).double(), max_rg_steps=steps)
            direct = torch.cat([model.theory_encoder(z) for z in levels],dim=1)
            seq = [direct[:, :1]]
            for _ in range(steps):
                seq.append(model.advance_theory(seq[-1]))
            ds.append(direct.numpy()); rs.append(torch.cat(seq,dim=1).numpy())
            ss.append(torch.cat([model.theory_encoder(z[:,:samples])-model.theory_encoder(z[:,samples:])
                                for z in levels],dim=1).numpy())
    return np.concatenate(ds),np.concatenate(rs),np.concatenate(ss)


def diagnostic(model, tx, vx, vh, long_vx, long_vh):
    td,_,_ = trajectories(model,tx)
    own_d,own_r,_ = trajectories(model,vx)
    d,r,split = trajectories(model,long_vx)
    scale = d.std(0)
    rmse = np.sqrt(((d-r)**2).mean(0))
    support = (float(td.min()),float(td.max()))
    lo,hi = min(support[0],float(d.min())),max(support[1],float(d.max()))
    pad = .15*(hi-lo)
    roots = find_fixed_points_1d(model,lo-pad,hi+pad)
    h = long_vh.numpy(); hs = np.unique(h)
    mean_g = np.array([d[h == value,0].mean() for value in hs])
    for root in roots:
        delta = mean_g-root['g']
        roots_crossing = np.flatnonzero(delta[:-1]*delta[1:] <= 0)
        root['bare_h_over_J_crossings'] = [float(hs[i]-delta[i]*(hs[i+1]-hs[i])/(delta[i+1]-delta[i]))
                                         for i in roots_crossing if delta[i+1] != delta[i]]
        root['inside_training_latent_range'] = support[0] <= root['g'] <= support[1]
    pairs = torch.tensor([[-1,-1],[-1,1],[1,-1],[1,1]],dtype=torch.float64).reshape(4,1,2,1)
    with torch.no_grad():
        truth = model.rg_kernel(pairs).flatten().tolist()
    record = dict(common_L128_test_rmse=rmse.tolist(),common_L128_test_std=scale.tolist(),
                  common_L128_test_nrmse=(rmse/np.maximum(scale,1e-12)).tolist(),
                  common_L128_test_split_rmse=np.sqrt((split**2).mean(0)).tolist(),
                  own_size_test_nrmse=(np.sqrt(((own_d-own_r)**2).mean(0))/np.maximum(own_d.std(0),1e-12)).tolist(),
                  training_latent_range=support,
                  common_test_out_of_training_range_fraction=((d<support[0])|(d>support[1])).mean(0).tolist(),
                  roots=roots, C_truth_table=truth)
    return record,dict(direct=d,rollout=r,h=h,training_direct=td)


def plots(out, models, histories, arrays, records):
    colors = ['C0','C1','C2','C3']; names = list(models)
    fig,axes = plt.subplots(1,2,figsize=(11,4.3))
    for name,c in zip(names,colors):
        axes[0].plot(range(1,6),np.array(records[name]['common_L128_test_nrmse'])[1:]*100,'o-',color=c,label=name)
        hist = histories[name]
        axes[1].plot([a['epoch'] for a in hist],[a['test_total'] for a in hist],color=c,label=name+' test')
        axes[1].plot([a['epoch'] for a in hist],[a['train_total'] for a in hist],color=c,ls='--',alpha=.6)
    axes[0].axvline(2.5,color='gray',ls=':',label='Short-chain training horizon')
    axes[0].set(xlabel='k on independent L128 test ensembles',ylabel='Rollout RMSE / direct Q std (%)',title='Common five-step test (each model uses its own Q/C)')
    axes[1].set(xlabel='Updates',ylabel='Original QCR objective',yscale='log',title='Own-size loss: solid=test, dashed=train')
    for ax in axes: ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'closure_and_loss.png',dpi=170);plt.close(fig)
    fig,axes = plt.subplots(2,len(names),figsize=(5*len(names),8))
    for j,name in enumerate(names):
        model = models[name]; a = arrays[name];d,r,h = a['direct'],a['rollout'],a['h']
        lo,hi = min(d.min(),a['training_direct'].min()),max(d.max(),a['training_direct'].max());pad=.15*(hi-lo)
        z = torch.linspace(lo-pad,hi+pad,600,dtype=torch.float64)[:,None]
        with torch.no_grad(): rg = model.advance_theory(z).numpy().ravel()
        ax = axes[0,j];ax.plot(z.numpy().ravel(),rg,label='R(g)');ax.plot(z.numpy().ravel(),z.numpy().ravel(),'k--',label='Identity')
        for root in records[name]['roots']:
            ax.scatter(root['g'],root['g'],marker='o' if root['stability']=='stable' else 'x',s=65,c='C3')
        ax.set(title=name,xlabel='g',ylabel='R(g)');ax.legend(fontsize=8);ax.grid(alpha=.2)
        ax=axes[1,j]
        hs=np.unique(h)
        for k in (0,1,2,5):
            ax.plot(hs,[d[h==v,k].mean() for v in hs],label=f'Q(C^{k} D)')
        if 'own_direct' in a and name.startswith('short'):
            oh=a['own_h'];od=a['own_direct']
            ax.plot(np.unique(oh),[od[oh==v,0].mean() for v in np.unique(oh)],'k--',alpha=.6,label='Q(D), native L16')
        ax.axvline(1.,color='gray',ls=':');ax.set(xlabel='h/J (held-out L128 ensembles)',ylabel='g');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Paired seed, independently sampled small systems; latent coordinates need not align')
    fig.tight_layout();fig.savefig(out/'R_and_coupling.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,len(names),figsize=(5*len(names),4.4))
    for ax,name in zip(axes,names):
        a=arrays[name];h=a['h'];d=a['direct'];r=a['rollout']
        for target,c in zip((.8,.96,1.05,1.2),('C0','C1','C2','C3')):
            value=np.unique(h)[np.argmin(abs(np.unique(h)-target))];mask=h==value
            ax.plot(range(6),d[mask].mean(0),'o-',color=c,label=f'{value:.3f} Q')
            ax.plot(range(6),r[mask].mean(0),'x--',color=c,label=f'{value:.3f} R')
        ax.axvline(2.5,color='gray',ls=':');ax.set(title=name,xlabel='k (L128 test)',ylabel='g');ax.legend(fontsize=7,ncol=2);ax.grid(alpha=.2)
    fig.suptitle('Direct Q(C^k D) versus autonomous R^k(Q(D)); not an exact physical RG reference')
    fig.tight_layout();fig.savefig(out/'long_test_trajectories.png',dpi=170);plt.close(fig)


def collect_results(out):
    report=json.loads((out/'report.json').read_text())
    models={};histories={};arrays={}
    datasets={size:torch.load(out/f'ensembles_L{size}.pt',weights_only=True) for size in (16,128)}
    for dest in sorted(out.glob('*/metrics.json')):
        n=dest.parent.name;rec=json.loads(dest.read_text());report['results'][n]=rec
        m=RecurrentRGNetwork1D(levels=rec['levels'],hidden=60,theory_dim=1).double()
        missing=m.load_state_dict(torch.load(out/n/'qcr_model.pt',weights_only=True),strict=False)
        assert not missing.unexpected_keys and all(k.startswith('action_network.') for k in missing.missing_keys)
        m.theory_encoder.binary_lookup=m.rg_kernel.binary_lookup=True
        models[n]=m;histories[n]=json.loads((out/n/'history_progress.json').read_text())
        with np.load(out/n/'trajectories.npz') as a: arrays[n]=dict(a)
        size=128 if n.startswith('long') else 16
        tx,_,vx,vh=datasets[size]
        if n=='short_L16_pooled8':
            tx=tx.reshape(-1,1024,16,1)
            # Recompute using all pooled samples, including diagnostic support.
            fresh,array=diagnostic(m,tx,vx,vh,datasets[128][2],datasets[128][3])
            rec.update(fresh);arrays[n]=array
            dest.write_text(json.dumps(rec,indent=2)+'\n')
            np.savez_compressed(out/n/'trajectories.npz',**array)
        own_d,_,_=trajectories(m,vx)
        arrays[n]['own_direct']=own_d;arrays[n]['own_h']=vh.numpy()
        if n=='short_L16_rep3': tx=tx[torch.arange(len(tx)) % 24 < 3]
        rec['pattern_coverage']=pattern_coverage(m,tx,datasets[128][2])
        dest.write_text(json.dumps(rec,indent=2)+'\n')
    plots(out,models,histories,arrays,report['results'])
    fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    report['input_observables']={}
    for size in (128,16):
        tx,th,vx,vh=datasets[size]
        x=torch.cat((tx,vx)).double().squeeze(-1);h=torch.cat((th,vh))
        hs=h.unique(sorted=True)
        observables=[(x*torch.roll(x,1,-1)).mean((1,2)),
                     (x*torch.roll(x,8,-1)).mean((1,2)),
                     x.mean(-1).square().mean(-1)]
        values=[]
        for ax,z in zip(axes,observables):
            means=[float(z[h==v].mean()) for v in hs]
            values.append(means);ax.plot(hs.numpy(),means,label=f'L={size}')
        report['input_observables'][str(size)]=dict(h_over_J=hs.tolist(),C1=values[0],C8=values[1],m2=values[2])
    for ax,label in zip(axes,('Nearest-neighbor correlation','Correlation at separation 8','Whole-chain magnetization squared')):
        ax.set(xlabel='h/J',ylabel=label);ax.axvline(1.,color='gray',ls=':');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Independent finite-size TFIM data: local agreement does not imply long-range agreement')
    fig.tight_layout();fig.savefig(out/'input_size_comparison.png',dpi=160);plt.close(fig)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


def pattern_coverage(model, train_x, test_x):
    """Nearest observed training-input distribution; descriptive, not a test.

    Q sees averaged three-spin patch features. Its input statistics are
    determined by these eight frequencies, but R may still lack closure.
    """
    def distributions(x, levels):
        rows=[]
        with torch.no_grad():
            for chunk in x.split(8):
                zs=model.coarse_grain(balanced_views(chunk,x.shape[1]).double(), max_rg_steps=levels-1)
                fs=[]
                for z in zs:
                    p=model.geometry.extract_patches(z)
                    idx=((p>0).long()*torch.tensor([1,2,4])).sum(-1)
                    fs.append(torch.nn.functional.one_hot(idx,8).double().mean(tuple(range(1,idx.ndim))))
                rows.append(torch.stack(fs,1))
        return torch.cat(rows)
    train=distributions(train_x,model.levels)
    test=distributions(test_x,6)
    inputs=train[:,:-1].reshape(-1,8)
    distances=torch.cdist(test[:,:-1].reshape(-1,8),inputs).min(1).values.reshape(len(test),5)
    return dict(input_scale_k=list(range(5)),mean_nearest_distance=distances.mean(0).tolist(),
                p95_nearest_distance=torch.quantile(distances,.95,dim=0).tolist(),
                note='Euclidean distance in empirical three-spin frequencies, to all training one-step input scales. Descriptive support diagnostic, not calibrated uncertainty.')


def pooled_control(out, fit_only=False):
    """Same small-system data, pooled to match spins per encoder call/update.

    Run after the primary experiment. This separates more replicas from
    improved precision of each individual ensemble representation.
    """
    protocol=json.loads((out/'protocol.json').read_text())
    seed=protocol['seed'];epochs=protocol['epochs'];torch.set_num_threads(2)
    sx,sh,sv,svh=torch.load(out/'ensembles_L16.pt',weights_only=True)
    _,_,lv,lvh=torch.load(out/'ensembles_L128.pt',weights_only=True)
    assert len(sx) % 24 == 0
    tx=sx.reshape(-1,8*128,16,1)
    grouped_h=sh.reshape(-1,8)
    assert torch.all(grouped_h==grouped_h[:,:1])
    name='short_L16_pooled8';dest=out/name
    if dest.exists(): raise FileExistsError(dest)
    torch.manual_seed(seed)
    model=RecurrentRGNetwork1D(levels=3,hidden=60,theory_dim=1).double()
    model.theory_encoder.binary_lookup=model.rg_kernel.binary_lookup=True
    started=time.perf_counter()
    hist=train(model,tx,sv,pattern_fn=two_spin_distribution,epochs=epochs,
               system_batch=32,sample_batch=1024,eval_samples=48,eval_systems=24,
               learning_rate=.001,seed=seed,progress_dir=dest,
               objective_kwargs=dict(coordinate_reduction='sum',rollout_weight=5.))
    torch.save({k:v for k,v in model.state_dict().items() if not k.startswith('action_network.')},dest/'qcr_model.pt')
    record,array=diagnostic(model,tx,sv,svh,lv,lvh)
    record.update(training_seconds=time.perf_counter()-started,train_shape=list(tx.shape),test_shape=list(sv.shape),
                  raw_training_spins=tx.numel(),levels=3,final_train_loss=hist[-1]['train_total'],
                  final_test_loss=hist[-1]['test_total'],sample_batch=1024,
                  note='Pools eight independent L16 replicas per ensemble. Same raw data as rep24, same raw spins per training batch as L128. Loss curves still use 48 configurations; common L128 test uses128.',
                  optimized_parameters=sum(p.numel() for m in (model.theory_encoder,model.rg_kernel,model.theory_rg) for p in m.parameters()))
    (dest/'metrics.json').write_text(json.dumps(record,indent=2)+'\n')
    np.savez_compressed(dest/'trajectories.npz',**array)
    if not fit_only:
        collect_results(out)
    print('POOLED DONE',json.dumps(record),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=2000)
    parser.add_argument('--seed',type=int,default=91)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--pooled-control',action='store_true')
    parser.add_argument('--fit-pooled-only',action='store_true')
    parser.add_argument('--collect',action='store_true')
    args=parser.parse_args();out=args.output_dir
    if args.collect:
        torch.set_num_threads(2)
        collect_results(out)
        return
    if args.fit_pooled_only:
        pooled_control(out,fit_only=True)
        return
    if args.pooled_control:
        pooled_control(out)
        return
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);started=time.perf_counter()
    data=make_data(out,args.seed,args.workers)
    data_seconds=time.perf_counter()-started
    lx,lh,lv,lvh=data[128];sx,sh,sv,svh=data[16]
    # First three independently simulated replicas at every training coupling.
    subset=torch.arange(len(sx)) % 24 < 3
    arms={'long_L128_rep3':(lx,lh,lv,lvh,6),
          'short_L16_rep3':(sx[subset],sh[subset],sv,svh,3),
          'short_L16_rep24':(sx,sh,sv,svh,3)}
    assert lx.numel()==sx.numel(), 'Matched raw-spin training budget'
    models={};histories={};arrays={};records={};initial=None
    protocol=dict(seed=args.seed,epochs=args.epochs,hidden=60,theory_dim=1,
                  beta=16.,dtau=.5,slices=32,sampler='Swendsen-Wang; 512 full equilibration sweeps, gap16',
                  original_configurations_per_ensemble=128,system_batch=32,sample_batch=128,
                  learning_rate=.001,geometry_scale=4.,noise_multiplier=2.,rollout_weight=5.,
                  spin_flip_data_augmentation=True,labels_used_in_loss=False,
                  short_data_independent_of_long_data=True,
                  split='Coupling indices modulo3 ==1 held out at every size; all replicas grouped',
                  primary_endpoint='2000 updates, not test selected',
                  caveat='Different finite sizes and number of observed RG scales; no claim of continuum/ground-state convergence. Equal raw-spin data does not equal independent information or compute.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for name,(tx,th,vx,vh,levels) in arms.items():
        dest=out/name;torch.manual_seed(args.seed)
        model=RecurrentRGNetwork1D(levels=levels,hidden=60,theory_dim=1).double()
        model.theory_encoder.binary_lookup=model.rg_kernel.binary_lookup=True
        state={k:v.clone() for k,v in model.state_dict().items()}
        if initial is None: initial=state
        else: assert all(torch.equal(v,initial[k]) for k,v in state.items())
        print('TRAIN',name,tuple(tx.shape),flush=True);t=time.perf_counter()
        hist=train(model,tx,vx,pattern_fn=two_spin_distribution,epochs=args.epochs,
                   system_batch=32,sample_batch=128,eval_samples=48,eval_systems=24,
                   learning_rate=.001,seed=args.seed,progress_dir=dest,
                   objective_kwargs=dict(coordinate_reduction='sum',rollout_weight=5.))
        torch.save({k:v for k,v in model.state_dict().items() if not k.startswith('action_network.')},dest/'qcr_model.pt')
        record,array=diagnostic(model,tx,vx,vh,lv,lvh)
        record.update(training_seconds=time.perf_counter()-t,train_shape=list(tx.shape),test_shape=list(vx.shape),
                      raw_training_spins=tx.numel(),levels=levels,final_train_loss=hist[-1]['train_total'],
                      final_test_loss=hist[-1]['test_total'],
                      optimized_parameters=sum(p.numel() for m in (model.theory_encoder,model.rg_kernel,model.theory_rg) for p in m.parameters()))
        (dest/'metrics.json').write_text(json.dumps(record,indent=2)+'\n')
        np.savez_compressed(dest/'trajectories.npz',**array)
        models[name]=model;histories[name]=hist;arrays[name]=array;records[name]=record
        print('RESULT',name,json.dumps(record),flush=True)
    plots(out,models,histories,arrays,records)
    report=dict(protocol=protocol,results=records,data_seconds=data_seconds,total_seconds=time.perf_counter()-started)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('DONE',out,'seconds',report['total_seconds'],flush=True)


if __name__=='__main__':main()
