"""Read-only model audit: exact finite-ring probabilities through learned C."""
import itertools
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recurrent_rg.checkpoints import load_run
from recurrent_rg.rsmi import kernel_probabilities


def exact_patterns(j, table, length=128, steps=5):
    """Sum all spins and independent C noise via two-state transfer matrices.

    M_h(k) weights an original block conditional on its coarse output h.
    M_h(k+1)=sum_ab C(h|a,b) M_a(k) M_b(k). Normalizing T by
    its largest eigenvalue prevents overflow without changing ring weights.
    """
    p=.5*(1+np.tanh(j))
    transition=np.array([[p,1-p],[1-p,p]])
    z=np.trace(np.linalg.matrix_power(transition,length))
    mats=np.stack([np.diag([1,0])@transition,np.diag([0,1])@transition])
    patterns=[];magnetization=[]
    for k in range(steps+1):
        n=2**k
        tail=np.linalg.matrix_power(transition,length-3*n)
        probs=np.array([np.trace(mats[a]@mats[b]@mats[c]@tail)/z for a,b,c in itertools.product(range(2),repeat=3)])
        assert np.min(probs)>-1e-12 and abs(probs.sum()-1)<1e-10
        patterns.append(probs)
        magnetization.append(2*np.trace(mats[1]@np.linalg.matrix_power(transition,length-n))/z-1)
        mats=np.stack([sum(table[a+2*b,h]*(mats[a]@mats[b]) for a,b in itertools.product(range(2),repeat=2)) for h in range(2)])
    return np.array(patterns),np.array(magnetization)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=Path('results/ising1d_g1_rsmiSeparate_w2_2400_seed91/qr'))
    args=p.parse_args()
    torch.set_num_threads(2)
    run=args.run
    out=run/'loss_audit';out.mkdir(exist_ok=True)
    m=load_run(run).model
    binary=(2*((torch.arange(4)[:,None]//(2**torch.arange(2)))%2)-1).double()[None]
    from recurrent_rg.model_nd import FixedDecimationCoarseGrainer
    with torch.no_grad():
        table=(np.array([[1.,0.],[0.,1.],[1.,0.],[0.,1.]]) if isinstance(m.rg_kernel,FixedDecimationCoarseGrainer)
               else kernel_probabilities(m.rg_kernel,binary)[0].numpy())
    js=np.linspace(.01,2,1001)
    data=[exact_patterns(j,table) for j in js]
    probs=np.stack([x[0] for x in data]);mag=np.stack([x[1] for x in data])
    patterns=torch.tensor(list(itertools.product((-1.,1.),repeat=3)),dtype=torch.double)
    with torch.no_grad():
        features=m.theory_encoder.local_mlp(patterns)
        g=m.theory_encoder.head(torch.from_numpy(probs)@features).squeeze(-1).numpy()
        one=m.advance_theory(torch.from_numpy(g[:,:-1,None])).squeeze(-1).numpy()
        roll=[g[:,0]]
        for _ in range(5):roll.append(m.advance_theory(torch.from_numpy(roll[-1][:,None])).squeeze(-1).numpy())
        roll=np.stack(roll,1)
    collisions=[]
    for a,b in itertools.combinations(range(5),2):
        x,y=g[:,a],g[:,b]
        mono=lambda z:bool(np.all(np.diff(z)>=0) or np.all(np.diff(z)<=0))
        if not mono(x) or not mono(y):continue
        lo,hi=max(x.min(),y.min()),min(x.max(),y.max())
        if hi<=lo:continue
        grid=np.linspace(lo,hi,1001);ia,ib=np.argsort(x),np.argsort(y)
        ya=np.interp(grid,x[ia],g[ia,a+1]);yb=np.interp(grid,y[ib],g[ib,b+1])
        index=np.argmax(abs(ya-yb));at=grid[index]
        collisions.append(dict(k_a=a,k_b=b,g=float(at),next_a=float(ya[index]),next_b=float(yb[index]),
            difference=float(abs(ya[index]-yb[index])),J_a=float(np.interp(at,x[ia],js[ia])),J_b=float(np.interp(at,y[ib],js[ib]))))
    h=json.loads((run/'history_progress.json').read_text())[-1]
    weights=dict(anchor=2,coarse_consistency=5,separation=10,commutator=5,rollout=5)
    weighted={key:weight*h['test_'+key] for key,weight in weights.items()}
    examples={str(j):exact_patterns(j,table)[1].tolist() for j in [0,.5,1,2]}
    report=dict(kernel_patterns=binary[0].tolist(),kernel_probabilities=table.tolist(),weighted_test_loss=weighted,
        logged_rollout_by_scale=[h[f'test_rollout_level_{k}'] for k in range(1,6)],
        exact_population_uniform_J_one_step_MSE=((one-g[:,1:])**2).mean(0).tolist(),
        exact_population_uniform_J_rollout_MSE=((roll-g)**2).mean(0).tolist(),
        mean_magnetization=examples,cross_scale_collisions=collisions,
        note='Exact finite L128 distribution through the specified learned stochastic or fixed C; Q applied to population pooled features. No retraining. Uniform dense J averaging differs from logged finite-sample test loss.')
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axs=plt.subplots(1,3,figsize=(16,4.5),layout='constrained')
    for k in range(6):
        axs[0].plot(js,g[:,k],label=f'k={k}');axs[2].plot(js,mag[:,k],label=f'k={k}')
        if k<5:axs[1].plot(g[:,k],g[:,k+1],label=f'{k} to {k+1}')
    grid=np.linspace(g[:,:-1].min(),g[:,:-1].max(),1001)
    with torch.no_grad():mapped=m.advance_theory(torch.from_numpy(grid[:,None])).squeeze(-1).numpy()
    axs[1].plot(grid,mapped,'k--',label='Learned R')
    axs[0].set(xlabel='J',ylabel='g_k',title='Exact population Q(C^k D)')
    axs[1].set(xlabel='g_k',ylabel='g_(k+1)',title='Can one shared R fit all scales?')
    axs[2].set(xlabel='J',ylabel='Mean spin',title='Bias generated by C',ylim=(-1.05,1.05))
    for ax in axs:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(out/'exact_scale_audit.png',dpi=170);plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
