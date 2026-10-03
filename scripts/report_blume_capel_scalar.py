"""Post-training scalar Blume-Capel flow; physical temperature is audit-only."""
import argparse
import json
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recurrent_rg.checkpoints import load_run
from recurrent_rg.evaluation import scalar_flow
from recurrent_rg.data.ensembles import load_ensembles


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--checkpoint',type=int)
    p.add_argument('--reference-temperature',type=float,default=1.398)
    a=p.parse_args();torch.set_num_threads(2)
    r=load_run(a.run,a.checkpoint);m=r.model
    if r.config['data']['coupling_names']!=['T/J'] or r.config['model']['categorical_states']!=3:raise ValueError('Requires the three-state temperature scan')
    out=a.run/('full_report' if a.checkpoint is None else f'full_report_{a.checkpoint}')
    data=np.load(out/'encodings.npz');g=np.concatenate([data['train'],data['test']])[:,:,0]
    t=np.concatenate([data['train_labels'],data['test_labels']]).ravel()
    knots=np.unique(t);means=np.stack([g[t==v].mean(0) for v in knots])
    tc=a.reference_temperature
    if not knots.min()<tc<knots.max():raise ValueError('Reference outside temperature grid')
    gc=float(np.interp(tc,knots,means[:,0]));lo,hi=g[:,:-1].min(),g[:,:-1].max();pad=max(.25,.4*(hi-lo))
    flow=scalar_flow(m,float(lo-pad),float(hi+pad),points=6001)
    for root in flow['roots']:
        z=root['g'][0];hits=[]
        for u,v,ga,gb in zip(knots[:-1],knots[1:],means[:-1,0],means[1:,0]):
            if min(ga,gb)<=z<=max(ga,gb) and ga!=gb:hits.append(float(u+(v-u)*(z-ga)/(gb-ga)))
        root.update(microscopic_inverse_temperature=hits,inside_observed_input_range=bool(lo<=z<=hi))
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    ax=axs[0];ax.plot(flow['grid'],flow['mapped'],label='R(g)');ax.plot(flow['grid'],flow['grid'],'k--',label='Identity')
    ax.axvspan(lo,hi,color='green',alpha=.1,label='Observed R-input range');ax.axvline(gc,color='purple',ls=':',label='Q(D) at reference Tc')
    for i,root in enumerate(flow['roots'],1):
        z=root['g'][0];ax.scatter(z,z,c='green' if root['stability']=='stable' else 'red')
        ax.annotate(f'F{i}: {root["stability"]}\nR\'={root["eigenvalues"][0]["real"]:.3f}',(z,z),xytext=(5,8),textcoords='offset points',fontsize=8)
    ax.set(xlabel='g',ylabel='R(g)',title='Blume-Capel: Delta/J=1');ax.margins(y=.12)
    ax=axs[1];ax.scatter(t,g[:,0],s=8,alpha=.2,c='black',label='Microscopic ensembles')
    for k in range(g.shape[1]):ax.plot(knots,means[:,k],label=f'Q(C^{k} D)')
    ax.axvline(tc,c='purple',ls=':',label=f'Literature Tc/J ~ {tc}')
    for i,root in enumerate(flow['roots'],1):ax.axhline(root['g'][0],ls='--',alpha=.4,label=f'F{i}')
    ax.set(xlabel='T/J',ylabel='g',title='Temperature mapping and coarse-graining')
    for ax in axs:ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(out/'R_and_T_mapping.png',dpi=180);plt.close(fig)
    # Only +/- exchange is a physical label symmetry: swapping the zero state
    # with a magnetic state is NOT a symmetry of Blume-Capel.
    _,test=load_ensembles(r.config['data']['cache']);x=test.configurations[:32,:64].double()
    uniform=torch.nn.functional.one_hot(torch.arange(3)[:,None].expand(3,4),3).double().flatten(1)
    with torch.no_grad():
        q=m.theory_encoder(x);qflip=m.theory_encoder(x[...,[2,1,0]])
        probs=m.rg_kernel.field_head.logits(m.rg_kernel.local_encoder(uniform)).softmax(-1)
    wide=scalar_flow(m,float(lo-2*(hi-lo+1)),float(hi+2*(hi-lo+1)),points=12001)
    twice=SimpleNamespace(advance_theory=lambda x:m.advance_theory(m.advance_theory(x)))
    two=scalar_flow(twice,float(lo-pad),float(hi+pad),points=6001)
    for root in two['roots']:
        z=torch.tensor(root['g'],dtype=torch.double)
        with torch.no_grad():partner=m.advance_theory(z)
        root['one_step_residual']=float((partner-z).norm())
        root['is_R_fixed_point']=root['one_step_residual']<1e-6
        root['partner_under_R']=partner.tolist()
        value=root['g'][0]
        root['microscopic_inverse_temperature']=[float(u+(v-u)*(value-ga)/(gb-ga))
            for u,v,ga,gb in zip(knots[:-1],knots[1:],means[:-1,0],means[1:,0])
            if ga!=gb and min(ga,gb)<=value<=max(ga,gb)]
    fig,ax=plt.subplots(figsize=(7,5),layout='constrained')
    ax.plot(two['grid'],two['mapped'],label='R(R(g))');ax.plot(two['grid'],two['grid'],'k--',label='Identity')
    ax.axvspan(lo,hi,color='green',alpha=.1,label='Observed R-input range');ax.axvline(gc,color='purple',ls=':',label='Q(D) at reference Tc')
    for i,root in enumerate(two['roots'],1):
        z=root['g'][0];ax.scatter(z,z,c='green' if root['stability']=='stable' else 'red')
        label='R root' if root['is_R_fixed_point'] else '2-cycle point'
        ax.annotate(f'{i}: {root["stability"]}, {label}',(z,z),xytext=(5,8),textcoords='offset points',fontsize=7)
    ax.set(xlabel='g',ylabel='R squared(g)',title='Two-step audit; not all roots are R fixed points');ax.margins(y=.12);ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(out/'R2_audit.png',dpi=170);plt.close(fig)
    report=dict(reference_Tc=tc,reference_source='https://arxiv.org/abs/1612.02138',reference_note='Numerical literature estimate at Delta/J=1, not an exact critical temperature',
        microscopic_g_at_reference_Tc=gc,observed_R_input_range=[float(lo),float(hi)],
        flow={k:v for k,v in flow.items() if k not in ('grid','mapped')},
        wide_flow={k:v for k,v in wide.items() if k not in ('grid','mapped')},
        two_step_flow={k:v for k,v in two.items() if k not in ('grid','mapped')},
        uniform_block_probabilities=probs.tolist(),state_order=[-1,0,1],
        Q_spin_flip_rmse=float((q-qflip).square().mean().sqrt()),Q_test_std=float(q.std(unbiased=False)),
        note='Temperature inverse is microscopic piecewise linear interpolation, not effective running coupling. No labels used in training; fixed-Delta path is not guaranteed RG-closed.')
    (out/'blume_capel_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
