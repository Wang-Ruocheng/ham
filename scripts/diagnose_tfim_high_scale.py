"""Inspect actual high-k C outputs, not just autonomous R iteration."""
import json
import argparse
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recurrent_rg.data.views import balanced_views
from recurrent_rg.data.statistics import two_spin_distribution
from recurrent_rg.model_nd import RecurrentRGNetwork1D


def inspect(model, raw, x, name, original_sites):
    levels = [balanced_views(raw, raw.shape[1])]
    gs, rg, records = [], [], []
    exact_decimation = True
    with torch.no_grad():
        while levels[-1].shape[2] > 1:
            levels.append(model.rg_kernel(levels[-1]))
        for k, level in enumerate(levels):
            expected = ((-1)**k) * levels[0][:, :, ::2**k]
            exact_decimation = exact_decimation and torch.allclose(level, expected, atol=1e-12, rtol=0)
            g = model.theory_encoder(level)[:, 0]
            gs.append(g.numpy())
            rg.append(g if k == 0 else model.advance_theory(rg[-1][:, None])[:, 0])
            spins = level[..., 0]
            nearest = (spins * spins.roll(-1, dims=2)).mean(dim=(1,2)).numpy()
            absm = spins.mean(dim=2).abs().mean(dim=1).numpy()
            patterns = two_spin_distribution(level).numpy()
            records.append({'k':k, 'sites':level.shape[2], 'g':g.numpy().tolist(),
                            'R_rollout':rg[-1].numpy().tolist(), 'G1':nearest.tolist(),
                            'mean_abs_m':absm.tolist(), 'patterns':patterns.tolist()})
    return {'name':name, 'x':x.tolist(), 'original_sites':original_sites,
            'exact_decimation_check':exact_decimation, 'records':records}, levels


def main():
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=root/'results/joint_qcr_tfim_1d_g1_v2_h60_sb128_block2_data_geometry_seed91')
    run = parser.parse_args().run
    model = RecurrentRGNetwork1D(levels=4,hidden=60,theory_dim=1).double().eval()
    inc = model.load_state_dict(torch.load(run/'qcr_model.pt',weights_only=True),strict=False)
    assert not inc.unexpected_keys and all(k.startswith('action_network.') for k in inc.missing_keys)
    model.theory_encoder.binary_lookup = True
    model.rg_kernel.binary_lookup = True
    cache=root/'data_cache/tfim1d_wolff_v2_sites128_M32_dtau0p5_eq2048_skip16_replicas3_ensemble128_seed91_nJ64.pt'
    tx,th,vx,vh=torch.load(cache,weights_only=True)
    raw,h=torch.cat((tx,vx)),torch.cat((th,vh)).numpy()
    indices=[]
    for target in [.5,.95,1.05,1.5]:
        value=h[np.argmin(abs(h-target))]
        indices.extend(np.flatnonzero(h==value).tolist())
    original, levels=inspect(model,raw[indices],h[indices],'original finite-temperature L128 cache',128)
    selected_x=h[indices][::3]
    fig,axes=plt.subplots(4,4,figsize=(14,10))
    for row,x in enumerate(selected_x):
        # First saved replica, original 64 samples in the first half of
        # balanced_views. Do not show artificially added spin flips.
        for col,k in enumerate([0,3,5,6]):
            spins=levels[k][row*3,:64,:,0].numpy()
            ax=axes[row,col]
            ax.imshow(spins,cmap='gray',vmin=-1,vmax=1,interpolation='nearest',aspect='auto',
                      extent=(0,spins.shape[1],64,0))
            rec=original['records'][k]
            same=(1+np.mean(rec['G1'][row*3:row*3+3]))/2
            ax.set_title(f'k={k}, n={spins.shape[1]}\nP(adjacent same)={same:.3f}',fontsize=10)
            if col==0: ax.set_ylabel(f'h/J={x:.4f}\nSample index')
            else: ax.set_yticklabels([])
            ax.set_xlabel('Retained spatial site')
    fig.suptitle('Latest trained C: actual original-data configurations at high k',fontsize=14)
    fig.text(.5,.015,f"C equals (-1)^k times first-site decimation: {original['exact_decimation_check']}. Statistics: all three replicas.\n"
             'k>3 is beyond training depth; 4/2-site periodic Q patches repeat or strongly overlap sites.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.96))
    fig.savefig(run/'high_k_original_configurations.png',dpi=200)
    plt.close(fig)
    xx=np.array([.95,1.,1.05])
    longraw=torch.tensor(np.stack([np.load(root/f'results/tfim_phase_transition_fss/samples_L1024_x{x:.6f}.npz')['samples'] for x in xx]),dtype=torch.float64)[...,None]
    long, _=inspect(model,longraw,xx,'exact T0 L1024 diagnostic; distribution shift from training',1024)
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for row,info in enumerate([original,long]):
        plot_records=[r for r in info['records'] if r['sites'] >= 4]
        steps=np.array([r['k'] for r in plot_records])
        xs=np.array(info['x'])
        for j,value in enumerate(np.unique(xs)):
            ids=np.flatnonzero(xs==value)
            for col,key in enumerate(['G1','mean_abs_m','g']):
                series=[np.mean(np.array(r[key])[ids]) for r in plot_records]
                axes[row,col].plot(steps,series,'o-',label=f'h/J={value:.4f}',color=f'C{j}')
                if key=='g':
                    pred=[np.mean(np.array(r['R_rollout'])[ids]) for r in plot_records]
                    axes[row,col].plot(steps,pred,'--',color=f'C{j}',alpha=.8)
        for col,title in enumerate(['Nearest-neighbor correlation','Mean absolute magnetization','Solid: actual Q(C^kD); dashed: R rollout']):
            ax=axes[row,col]
            ax.axvspan(3.05,steps[-1]+.1,color='gray',alpha=.08)
            ax.set_title(title,fontsize=11)
            ax.set_xlabel('k / remaining sites')
            ax.set_xticks(steps)
            ax.set_xticklabels([f"{r['k']}\n{r['sites']}" for r in plot_records],fontsize=8)
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
        axes[row,0].set_ylabel('Original L128 cache' if row==0 else 'Exact L1024 (different input distribution)')
    fig.suptitle('Separate C/data changes from learned R extrapolation',fontsize=14)
    fig.text(.5,.01,'Plots stop at n=4: at n=2 Q repeats sites in its 3-spin patch; at n=1 all flip-balanced ensembles are identical.\n'
             'Shaded region: beyond training depth. L1024 is an independent diagnostic, not a matched finite-temperature size ablation.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.065,1,.96))
    fig.savefig(run/'high_k_diagnosis.png',dpi=200)
    plt.close(fig)
    result={'original':original,'long_chain':long}
    (run/'high_k_diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')
    for info in [original,long]:
        print(info['name'])
        xs=np.array(info['x'])
        for x in np.unique(xs):
            ids=np.flatnonzero(xs==x)
            print(float(x),[{key: (r[key] if key in ['k','sites'] else round(float(np.mean(np.array(r[key])[ids])),5)) for key in ['k','sites','G1','g','R_rollout']} for r in info['records']])


if __name__=='__main__':
    main()
