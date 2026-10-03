"""Validate 1D zero-field Ising endpoints, drift, and population closure."""
import argparse,itertools,json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recurrent_rg.checkpoints import load_run
from recurrent_rg.rsmi import kernel_probabilities
from recurrent_rg.evaluation import scalar_flow
from scripts.audit_ising1d_rsmi_loss import exact_patterns


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--checkpoint',type=int);args=p.parse_args()
    torch.set_num_threads(2);run=load_run(args.run,args.checkpoint);m=run.model
    patterns=torch.tensor(list(itertools.product((-1.,1.),repeat=3)),dtype=torch.double)
    binary=(2*((torch.arange(4)[:,None]//(2**torch.arange(2)))%2)-1).double()[None]
    with torch.no_grad():
        table=kernel_probabilities(m.rg_kernel,binary)[0].numpy()
        features=m.theory_encoder.local_mlp(patterns)
    def encode(js):
        prob=np.stack([exact_patterns(j,table)[0] for j in js])
        with torch.no_grad():return m.theory_encoder.head(torch.from_numpy(prob)@features).squeeze(-1).numpy()
    js=np.r_[0.,np.linspace(.01,2,300),np.linspace(2.01,8,100),np.inf]
    g=encode(js);g0,ginf=g[0,0],g[-1,0];span=abs(ginf-g0)
    lo,hi=g[:,:-1].min(),g[:,:-1].max();pad=.4*(hi-lo)+.2
    flow=scalar_flow(m,float(lo-pad),float(hi+pad),points=6001)
    with torch.no_grad():r=m.advance_theory(torch.from_numpy(g[:,:,None])).squeeze(-1).numpy()
    interior=(js>=.01)&(js<=2)
    expected=encode(.5*(np.logaddexp(2*js[interior],-2*js[interior])-np.log(2)))[:,0]
    roots=flow['roots']
    for root in roots:
        v=root['g'][0]
        root['distance_to_J0_over_endpoint_span']=float(abs(v-g0)/max(span,1e-12))
        root['distance_to_Jinfinity_over_endpoint_span']=float(abs(v-ginf)/max(span,1e-12))
    result=dict(kernel_P_plus=table[:,1].tolist(),population_g_J0=float(g0),population_g_Jinfinity=float(ginf),
        endpoint_residual_over_span=(abs(r[[0,-1],0]-g[[0,-1],0])/max(span,1e-12)).tolist(),roots=roots,
        physical_interval_direction_fraction=float(((r[interior,0]-g[interior,0])*(g0-g[interior,0])>0).mean()),
        population_closure_rmse_by_scale=np.sqrt(((r[interior,:-1]-g[interior,1:])**2).mean(0)).tolist(),
        decimation_flow_reference_rmse_over_span=float(np.sqrt(((r[interior,0]-expected)**2).mean())/max(span,1e-12)),
        note='Population pooled Q on finite L128 distributions. J used only for held-out physical diagnostics, not selection/training. Decimation reference is a comparison, not assumed correct for arbitrary learned C. No finite-temperature critical point in this model.')
    out=args.run/('endpoint_report' if args.checkpoint is None else f'endpoint_report_{args.checkpoint}');out.mkdir(exist_ok=True)
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axs=plt.subplots(1,3,figsize=(15,4.3),layout='constrained')
    axs[0].plot(flow['grid'],flow['mapped'],label='Learned R(g)');axs[0].plot(flow['grid'],flow['grid'],'k--',label='identity')
    for value,label in [(g0,'Q(J=0)'),(ginf,'Q(J=infinity)')]:axs[0].axvline(value,ls=':',label=label)
    axs[0].plot(g[interior,0],expected,label='Exact decimation via Q(J)')
    for root in roots:axs[0].scatter(root['g'][0],root['g'][0],c='green' if root['stability']=='stable' else 'red')
    axs[0].set(xlabel='g',ylabel='R(g)',title='Endpoint and flow check')
    for k in range(6):axs[1].plot(js[interior],g[interior,k],label=f'k={k}')
    axs[1].set(xlabel='J',ylabel='Q(C^k D)',title='Population coupling mapping')
    chosen=[.1,.5,1.,2.,4.,np.inf];direct=encode(chosen)
    with torch.no_grad():
        roll=[direct[:,0]]
        for _ in range(5):roll.append(m.advance_theory(torch.from_numpy(roll[-1][:,None])).squeeze(-1).numpy())
    roll=np.stack(roll,1)
    for i,j in enumerate(chosen):
        line,=axs[2].plot(range(6),direct[i],'-o',label=f'J={j}: Q(C^k D)')
        axs[2].plot(range(6),roll[i],'--x',color=line.get_color())
    axs[2].set(xlabel='RG step k',ylabel='g',title='Solid Q(C^k D), dashed R^k(Q(D))')
    for ax in axs:ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.savefig(out/'physical_validation.png',dpi=160)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
