"""Plot finite-size Potts energy distributions and initial-state agreement."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,required=True);a=p.parse_args()
    meta=json.loads((a.cache/'sampling.json').read_text());data=np.load(a.cache/'probe_energies.npz')
    ks=data['K'];kc=float(np.log1p(np.sqrt(meta['q'])));ic=int(np.argmin(abs(ks-kc)))
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained');checks=[]
    for j in [max(0,ic-1),ic,min(len(ks)-1,ic+1)]:
        samples=np.concatenate([data[f'energy_{j}_{s}'] for s in [0,1]])
        axes[0].hist(samples,bins=50,density=True,histtype='step',label=f'K={ks[j]:.5f}')
    for s in [0,1]:
        samples=data[f'energy_{ic}_{s}'];label='ordered start' if s else 'random start'
        axes[1].hist(samples,bins=np.linspace(-2,-.5,61),density=True,histtype='step',label=label)
        axes[2].plot(np.arange(len(samples)),samples,lw=.6,alpha=.7,label=label)
    for i,k in enumerate(ks):
        pair=[r for r in meta['probes'] if r['K']==k]
        se=np.hypot(pair[0]['energy_batch_se'],pair[1]['energy_batch_se'])
        checks.append(dict(K=float(k),initial_state_mean_difference=float(pair[1]['energy_mean']-pair[0]['energy_mean']),
            difference_over_combined_batch_SE=float(abs(pair[1]['energy_mean']-pair[0]['energy_mean'])/se) if se else None,
            lag1=[r['bond_lag1'] for r in pair]))
    axes[0].set(xlabel='Energy / (J site)',ylabel='Density',title=f'q={meta["q"]}, L={meta["L"]}: finite-size energy histograms')
    axes[1].set(xlabel='Energy / (J site)',ylabel='Density',title='At exact infinite-volume Kc: start comparison')
    axes[2].set(xlabel='Recorded sample',ylabel='Energy / (J site)',title='Kc time series; correlated MCMC samples')
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(a.cache/'sampling.png',dpi=160);plt.close(fig)
    result=dict(q=meta['q'],Kc=kc,checks=checks,note='Batch SE uses 32 recorded samples per block; agreement is not a proof of equilibrium. Finite-size phase weights need not be equal at infinite-volume Kc.')
    (a.cache/'sampling_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
