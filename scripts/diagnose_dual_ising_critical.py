"""Direct critical-line sampling audit, without retraining or coupling supervision."""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recurrent_rg.checkpoints import load_run
from recurrent_rg.data.dual_ising import sample_dual_ising_couplings
from recurrent_rg.evaluation import collect_trajectories


def main():
    torch.set_num_threads(2)
    folder = Path('results/dual_ising2d_g2_uniformK1K2_fixed_1200_seed91')
    out = folder / 'critical_sampling_audit'
    out.mkdir(exist_ok=True)
    model = load_run(folder).model.eval()
    kc = np.log(1 + np.sqrt(2)) / 2
    eps = .03
    probes, cells = [], []
    def index(p):
        p = tuple(p)
        if p not in cells:
            cells.append(p)
        return cells.index(p)
    for axis in range(2):
        for other in [.2, kc, .7]:
            center = np.array([kc, kc]); center[1-axis] = other
            normal, tangent = np.eye(2)[axis], np.eye(2)[1-axis]
            probes.append(dict(axis=axis, other=other,
                triplet=[index(center+s*eps*normal) for s in [-1, 0, 1]],
                tangent=[index(center+s*eps*tangent) for s in [-1, 1]]))
    cache = out / 'sampled_states.npz'
    if cache.exists():
        saved = np.load(cache)
        assert np.array_equal(saved['couplings'], cells)
        states = saved['states']
    else:
        rows = []
        for i, (a,b) in enumerate(cells):
            rows.append(np.stack([sample_dual_ising_couplings(a,b,16,256,
                740000+100*i+r,burn=2048,spacing=16,ordered=bool(r%2)) for r in range(4)]))
            print(f'sampled {i+1}/{len(cells)}', flush=True)
        states = np.stack(rows)
        np.savez_compressed(cache, states=states, couplings=cells)
    # Each independently simulated chain supplies two 128-configuration ensembles.
    x = torch.nn.functional.one_hot(torch.from_numpy(states.astype(np.int64)).reshape(-1,128,16,16),4).to(torch.uint8)
    t = collect_trajectories(model,x,samples=128,max_rg_steps=2)
    arrays = {key: value.reshape(len(cells),4,2,*value.shape[1:]).mean(2) for key,value in t.items()}
    np.savez_compressed(out/'trajectories.npz', couplings=cells, **arrays)
    d, r = arrays['direct'], arrays['rollout']
    mean, rm = d.mean(1), r.mean(1)
    se = d.std(1,ddof=1)/2
    fig, axs = plt.subplots(2,3,figsize=(15,9),layout='constrained')
    metrics = []
    for ax,p in zip(axs.flat,probes):
        ids = p['triplet']; center = ids[1]
        for idx,label,color in zip(ids,['Kc - 0.03','Kc','Kc + 0.03'],['#0072b2','black','#d55e00']):
            ax.errorbar(*mean[idx].T,xerr=se[idx,:,0],yerr=se[idx,:,1],fmt='o-',c=color,capsize=3,label=label+' direct Q')
            ax.plot(*rm[idx].T,'x--',c=color,label=label+' R rollout')
            for k,z in enumerate(mean[idx]): ax.annotate(str(k),z,xytext=(4,5),textcoords='offset points',fontsize=8,color=color)
        ax.set(title=f'K{p["axis"]+1} near Kc; K{2-p["axis"]}={p["other"]:.4f}',xlabel='g1',ylabel='g2')
        ax.grid(alpha=.2)
        delta = mean[ids[2]]-mean[ids[0]]
        tangent = mean[p['tangent'][1]]-mean[p['tangent'][0]]
        normals = np.column_stack([-tangent[:,1],tangent[:,0]])
        normals /= np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
        gap = np.linalg.norm(delta,axis=1)
        ngap = np.abs(np.sum(delta*normals,axis=1))
        drift = mean[center]-mean[center,0]
        metrics.append(dict(axis=p['axis']+1,other_K=p['other'],
            direct_gap=gap.tolist(),direct_gap_gain=(gap/gap[0]).tolist(),
            rollout_gap=np.linalg.norm(rm[ids[2]]-rm[ids[0]],axis=1).tolist(),
            direct_normal_gap=ngap.tolist(),
            critical_direct=mean[center].tolist(),critical_se=se[center].tolist(),
            critical_drift_norm=np.linalg.norm(drift,axis=1).tolist(),
            critical_drift_on_initial_normal=(drift@normals[0]).tolist(),
            critical_rollout=rm[center].tolist(),
            closure_error_norm=np.linalg.norm(rm[center]-mean[center],axis=1).tolist(),
            hot_cold_direct_difference=(d[center,[1,3]].mean(0)-d[center,[0,2]].mean(0)).tolist()))
    axs[0,0].legend(fontsize=7)
    fig.suptitle('Fresh sampling at exact Kc and both sides; labels = RG step\nSolid: Q(C^k D), dashed: R^k(Q(D)); error bars: SE across 4 independent chains')
    fig.savefig(out/'critical_trajectories.png',dpi=180)
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for p,m in zip(probes,metrics):
        label=f'vary K{p["axis"]+1}, other={p["other"]:.3f}'
        axs[0].plot(range(3),m['direct_gap_gain'],'o-',label=label)
        axs[1].plot(range(3),np.array(m['rollout_gap'])/m['direct_gap'][0],'o-',label=label)
    for ax,title in zip(axs,['Direct Q: separation relative to k=0','R rollout: separation relative to k=0']):
        ax.axhline(1,c='gray',ls=':');ax.set(title=title,xlabel='RG step k',ylabel='separation gain',xticks=[0,1,2]);ax.grid(alpha=.2)
    axs[0].legend(fontsize=7);fig.savefig(out/'separation_gain.png',dpi=180)
    report=dict(kc=kc,epsilon=eps,L=16,scales=[16,8,4],unique_cells=len(cells),
        configurations=int(states.shape[0]*states.shape[1]*states.shape[2]),
        replicas=4,samples_per_replica=256,burn=2048,spacing=16,
        caveat='Normal drift uses initial local tangent, not a proven global separatrix. Finite-size Q gaps need not grow monotonically under exact decimation. SE measures sampling variation, not finite-size or model bias.',probes=metrics)
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
