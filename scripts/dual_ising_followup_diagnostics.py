"""Finite-size audit and data-only predictive-sufficiency probes for frozen Q."""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recurrent_rg.checkpoints import load_run
from recurrent_rg.data.ensembles import load_ensembles
from recurrent_rg.evaluation import collect_trajectories
from dual_ising_followup_data import SOURCE,OUT


def size_report():
    data={L:np.load(OUT/f'size_L{L}.npz') for L in [16,32,64]}
    report={};fig,axs=plt.subplots(1,3,figsize=(16,5),layout='constrained')
    for L,v in data.items():
        d=v['direct'];mean=d.mean(1);se=d.std(1,ddof=1)/2
        critical=mean[1];ls=np.array([L//2**k for k in range(mean.shape[1])])
        report[str(L)]=dict(remaining_lengths=ls.tolist(),critical_mean=critical.tolist(),critical_se=se[1].tolist(),
            critical_two_step_displacement=float(np.linalg.norm(critical[2]-critical[0])),
            critical_first_step_displacement=float(np.linalg.norm(critical[1]-critical[0])),
            critical_final_displacement=float(np.linalg.norm(critical[-1]-critical[0])),
            hot_cold_difference=(d[1,[1,3]].mean(0)-d[1,[0,2]].mean(0)).tolist(),
            separation_both_sides=np.linalg.norm(mean[2]-mean[0],axis=1).tolist())
        axs[0].errorbar(*critical.T,xerr=se[1,:,0],yerr=se[1,:,1],fmt='o-',capsize=3,label=f'initial L={L}')
        for k,z in enumerate(critical):axs[0].annotate(f'{ls[k]}',z,xytext=(3,4),textcoords='offset points',fontsize=7)
        for j,ax in enumerate(axs[1:]):ax.errorbar(ls,critical[:,j],yerr=se[1,:,j],fmt='o-',capsize=3,label=f'initial L={L}')
    axs[0].set(xlabel='g1',ylabel='g2',title='Direct Q at (Kc,Kc); numbers = remaining L')
    for j,ax in enumerate(axs[1:]):ax.set(xlabel='Remaining lattice length',ylabel=f'g{j+1}',xscale='log',title='Compare identical remaining sizes')
    for ax in axs:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(OUT/'finite_size_critical.png',dpi=180)
    report['caveat']='Fixed Q trained only at L16,8,4; larger sizes are out-of-training-size inference. Four independent chains per point, 128 samples each. This is a finite-size diagnostic, not an infinite-volume extrapolation.'
    (OUT/'size_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('SIZE',json.dumps(report),flush=True)


def pair_features(states):
    """Generic categorical joint histograms, no Ising spin or coupling features."""
    rows=[]
    for x in states:
        levels=[]
        for k in range(2):
            z=x[:,::2**k,::2**k].astype(np.int64)
            channels=[]
            for distance in [1,2,4]:
                hist=np.zeros(16)
                for axis in [1,2]:hist+=np.bincount((4*z+np.roll(z,distance,axis)).ravel(),minlength=16)
                channels.extend(hist/(2*z.size))
            levels.append(channels)
        rows.append(levels)
    return np.array(rows)


def probe_report():
    torch.set_num_threads(2)
    run=load_run(SOURCE);tr,te=load_ensembles(run.config['data']['cache'])
    saved=torch.load(SOURCE/'R_only_refit_6000'/'fixed_targets.pt',weights_only=True)
    cache=OUT/'predictive_features.npz'
    if not cache.exists():
        values={}
        for name,data in [('train',tr),('test',te)]:values[name]=pair_features(data.configurations.argmax(-1).numpy())
        raw=np.load(SOURCE/'critical_sampling_audit'/'sampled_states.npz')['states'].reshape(-1,128,16,16)
        values['audit']=pair_features(raw)
        x=torch.nn.functional.one_hot(torch.from_numpy(raw.astype(np.int64)),4).to(torch.uint8)
        values['audit_g']=collect_trajectories(run.model,x,samples=128,max_rg_steps=2,chunk_size=2)['direct']
        np.savez_compressed(cache,**values)
    f=np.load(cache);targets={name:saved[name] for name in ['train','test']};targets['audit']=torch.from_numpy(f['audit_g'])
    report=dict(purpose='Predictive sufficiency diagnostic, NOT a replacement 2D Q or RG model. Predict next frozen-Q encoding from current Q alone, or Q plus generic pair histograms. Same original train/test coupling-cell split, no physical label inputs. Local-control repeats distance-1 histograms to match the multi-distance input/parameter count. Four independent audit chains each split into two 128-sample ensembles.',models={})
    for mode in ['Q_only','Q_local_control','Q_multidistance']:
        xs={}
        for name,g in targets.items():
            parts=[g[:,:-1]]
            if mode!='Q_only':
                h=f[name] if mode=='Q_multidistance' else np.tile(f[name][...,:16],(1,1,3))
                parts.append(torch.from_numpy(h))
            xs[name]=torch.cat(parts,-1).reshape(-1,sum(p.shape[-1] for p in parts))
        origin=xs['train'].mean(0);scale=xs['train'].std(0).clamp_min(.005)
        xs={name:(x-origin)/scale for name,x in xs.items()}
        ys={name:(g[:,1:]-g[:,:-1]).reshape(-1,2) for name,g in targets.items()}
        torch.manual_seed(91)
        net=torch.nn.Sequential(torch.nn.Linear(len(origin),60),torch.nn.Tanh(),torch.nn.Linear(60,60),torch.nn.Tanh(),torch.nn.Linear(60,2)).double()
        opt=torch.optim.Adam(net.parameters(),lr=.001);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,3000,eta_min=1e-5)
        for step in range(3000):
            opt.zero_grad();loss=(net(xs['train'])-ys['train']).square().sum(-1).mean();loss.backward();opt.step();sch.step()
        result=dict(parameters=sum(p.numel() for p in net.parameters()))
        for name,x in xs.items():
            with torch.no_grad():e=(net(x)-ys[name]).reshape(-1,2,2)
            sd=saved['train'][:,1:].var(0,unbiased=False).mean(-1).sqrt()
            result[name]=dict(rmse=e.square().mean((0,2)).sqrt().tolist(),nrmse=(e.square().mean((0,2)).sqrt()/sd).tolist())
        report['models'][mode]=result
        torch.save(dict(model=net.state_dict(),origin=origin,scale=scale,description=report['purpose']),OUT/f'probe_{mode}.pt')
        print('PROBE',mode,result,flush=True)
    (OUT/'probe_report.json').write_text(json.dumps(report,indent=2)+'\n')


def nearest_report():
    saved=torch.load(SOURCE/'R_only_refit_6000'/'fixed_targets.pt',weights_only=True)
    y=torch.cat([saved['train_couplings'],saved['test_couplings']]).numpy()
    g=torch.cat([saved['train'],saved['test']]).numpy();cells=np.unique(y,axis=0)
    groups=[g[(y==p).all(1)] for p in cells]
    labels=cells.tolist()
    for filename in [OUT/'dense_encodings.npz',SOURCE/'critical_sampling_audit'/'trajectories.npz']:
        a=np.load(filename);groups.extend(list(a['direct']));labels.extend(a['couplings'].tolist())
    means=np.stack([d.mean(0) for d in groups]);ses=np.stack([d.std(0,ddof=1)/np.sqrt(len(d)) for d in groups])
    x=means[:,:-1].reshape(-1,2);target=means[:,1:].reshape(-1,2)
    sx=ses[:,:-1].reshape(-1,2);sy=ses[:,1:].reshape(-1,2)
    distance=np.linalg.norm(x[:,None]-x[None,:],axis=-1)
    scale=np.tile([0,1],len(groups));distance[scale[:,None]==scale[None,:]]=np.inf
    js=distance.argmin(1);ii=np.arange(len(x));pairs=sorted(set(tuple(sorted((int(i),int(j)))) for i,j in zip(ii,js)))
    rows=[]
    for i,j in pairs:
        source_gap=float(np.linalg.norm(x[i]-x[j]));target_gap=float(np.linalg.norm(target[i]-target[j]))
        source_se=float(np.sqrt((sx[i]**2+sx[j]**2).sum()));target_se=float(np.sqrt((sy[i]**2+sy[j]**2).sum()))
        rows.append(dict(i=i,j=j,source_couplings=labels[i//2],other_couplings=labels[j//2],scales=[int(scale[i]),int(scale[j])],source_gap=source_gap,target_gap=target_gap,source_se=source_se,target_se=target_se,
            noise_overlap=bool(source_gap<=2*source_se),successors_resolved=bool(target_gap>3*target_se),required_secant_gain=target_gap/max(source_gap,1e-12)))
    suspects=[r for r in rows if r['noise_overlap'] and r['successors_resolved']]
    suspects.sort(key=lambda r:r['target_gap']/max(r['target_se'],1e-12),reverse=True)
    report=dict(cross_scale_nearest_pairs=len(rows),noise_overlapping_sources=sum(r['noise_overlap'] for r in rows),overlap_but_resolved_successors=len(suspects),top_candidates=suspects[:12],
        caveat='Suggestive only: noise-overlap is not equality, and a steep but single-valued R can amplify small input differences. This does not prove nonclosure. Multiple comparisons uncorrected; finite-size and representation effects are confounded.')
    (OUT/'nearest_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('NEAREST',json.dumps(report),flush=True)


if __name__=='__main__':
    OUT.mkdir(exist_ok=True);size_report();nearest_report();probe_report()
