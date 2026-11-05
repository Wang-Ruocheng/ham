"""Clock-specific scalar diagnostics; BKT references are approximate, not exact roots."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recurrent_rg.checkpoints import load_run
from recurrent_rg.evaluation import scalar_flow
from recurrent_rg.data.ensembles import load_ensembles
from recurrent_rg.empirical_mi import lattice_mi_statistics
from recurrent_rg.rsmi import categorical_kernel_probabilities


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(2);run=load_run(a.run);m=run.model.eval();out=a.run/'full_report'
    data=np.load(out/'encodings.npz')
    g=np.concatenate([data['train'],data['test']])[:,:,0]
    K=np.concatenate([data['train_labels'],data['test_labels']]).ravel()
    knots=np.unique(K);means=np.stack([g[K==v].mean(0) for v in knots])
    # Rounded literature temperatures T/J~0.90 and ~0.70. These are only guides.
    refs=np.array([1/.90,1/.70]);gc=np.interp(refs,knots,means[:,0])
    low,high=float(g[:,:-1].min()),float(g[:,:-1].max());pad=max(.25,.35*(high-low))
    flow=scalar_flow(m,low-pad,high+pad,points=6001)
    for r in flow['roots']:
        z=r['g'][0]
        r['inside_observed_input_range']=bool(low<=z<=high)
        r['microscopic_inverse_K']=[float(x+(y-x)*(z-u)/(v-u)) for x,y,u,v in zip(knots[:-1],knots[1:],means[:-1,0],means[1:,0]) if u!=v and min(u,v)<=z<=max(u,v)]
    fig,ax=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    ax[0].plot(flow['grid'],flow['mapped'],label='R(g)');ax[0].plot(flow['grid'],flow['grid'],'k--',label='identity')
    ax[0].axvspan(low,high,color='green',alpha=.1,label='observed R inputs')
    for i,r in enumerate(flow['roots']):
        z=r['g'][0];ax[0].scatter(z,z,marker='*',s=100,label=f'F{i+1}: {r["stability"]}')
    for k in range(means.shape[1]):ax[1].plot(knots,means[:,k],label=f'Q(C^{k} D)')
    for Kc,z,color in zip(refs,gc,['purple','brown']):
        ax[0].axvline(z,color=color,ls=':',label=f'Q(D) at K~{Kc:.2f}')
        ax[1].axvline(Kc,color=color,ls=':',label=f'BKT guide K~{Kc:.2f}')
    ax[0].set(xlabel='g',ylabel='R(g)',title='Six-state clock: scalar map')
    ax[1].set(xlabel='K=J/T',ylabel='g',title='Direct encodings; not a five-root target')
    for z in ax:z.legend(fontsize=7);z.grid(alpha=.2)
    fig.savefig(out/'R_and_K_mapping.png',dpi=180)
    _,test=load_ensembles(run.config['data']['cache']);q=6
    # Clock symmetry is dihedral, NOT all permutations of the six labels.
    x=test.configurations[::6,:64].double();labels=x.argmax(-1)
    with torch.no_grad():
        original=m.theory_encoder(x);errors=[]
        for sign in (1,-1):
            for shift in range(q):
                z=torch.nn.functional.one_hot((sign*labels+shift)%q,q).double()
                errors.append(float(torch.sqrt(((m.theory_encoder(z)-original)**2).mean())))
    ks=test.couplings[:,0].numpy();ids=[int(np.argmin(abs(ks-v))) for v in [.6,refs[0],1.27,refs[1],2.]]
    actual=lattice_mi_statistics(test.configurations[ids],states=q)
    shuffled=lattice_mi_statistics(test.configurations[ids],states=q,shuffle_visible_seed=551)
    codes=(torch.arange(q**4)[:,None]//(q**torch.arange(4)))%q
    patterns=torch.nn.functional.one_hot(codes,q).double().flatten(1)[None]
    with torch.no_grad():
        table=categorical_kernel_probabilities(m.rg_kernel,patterns)[0]
        real=actual.information(table)/math.log(2);null=shuffled.information(table)/math.log(2)
    result=dict(BKT_reference_K=refs.tolist(),reference_note='Rounded numerical T/J~.90,.70, not exact critical values; intermediate phase is critical, not an isolated sink.',
                roots=flow['roots'],observed_input_range=[low,high],
                dihedral_Q_rmse=errors,dihedral_Q_max_rmse=max(errors),
                probe_K=ks[ids].tolist(),MI_bits=real.tolist(),shuffled_MI_bits=null.tolist(),
                MI_statistics=actual.diagnostics,
                caveat='L16 pilot, only two RG steps, bounded root search, finite-sample MI bias. One trained seed; no BKT identification from roots alone.')
    (out/'clock_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
