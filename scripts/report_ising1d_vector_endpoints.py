"""Read-only population endpoint audit for classical Ising with dim_g=2."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recurrent_rg.checkpoints import load_run
from recurrent_rg.rsmi import kernel_probabilities
from scripts.audit_ising1d_rsmi_loss import exact_patterns


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(2)
    loaded=load_run(args.run);m=loaded.model
    assert loaded.config['model']['dimension']==1 and loaded.config['model']['theory_dim']==2
    assert loaded.config['data']['coupling_names']==['J']
    assert loaded.config['training']['max_rg_steps']==5
    patterns=torch.tensor(list(itertools.product((-1.,1.),repeat=3)),dtype=torch.double)
    blocks=(2*((torch.arange(4)[:,None]//(2**torch.arange(2)))%2)-1).double()[None]
    with torch.no_grad():
        table=kernel_probabilities(m.rg_kernel,blocks)[0].numpy()
        features=m.theory_encoder.local_mlp(patterns)
    def encode(js):
        prob=np.stack([exact_patterns(j,table,length=128,steps=5)[0] for j in js])
        with torch.no_grad():return m.theory_encoder.head(torch.from_numpy(prob)@features).numpy()
    js=np.linspace(.01,2,300);g=encode(js)
    endpoints=encode([0.,np.inf]);span=np.linalg.norm(endpoints[0,0]-endpoints[1,0])
    with torch.no_grad():
        predicted=m.advance_theory(torch.from_numpy(g[:,:-1])).numpy()
        endpoint_r=m.advance_theory(torch.from_numpy(endpoints[:,0])).numpy()
    expected=encode(.5*(np.logaddexp(2*js,-2*js)-np.log(2)))[:,0]
    roots=json.loads((args.run/'full_report/metrics.json').read_text())['fixed_points_R']
    for r in roots:
        r['endpoint_distance_over_span']=(np.linalg.norm(endpoints[:,0]-r['g'],axis=-1)/span).tolist()
        jac=torch.autograd.functional.jacobian(m.advance_theory,torch.tensor(r['g'],dtype=torch.double)).detach().numpy()
        values,vectors=np.linalg.eig(jac)
        axis=(endpoints[0,0]-endpoints[1,0])/span
        r['endpoint_axis_eigenvector_alignment']=[dict(eigenvalue_real=float(v.real),
            eigenvalue_imag=float(v.imag),absolute_cosine=float(abs(np.vdot(vectors[:,i],axis))/np.linalg.norm(vectors[:,i]))) for i,v in enumerate(values)]
    eig=np.linalg.eigvalsh(np.cov(g[:,0].T,bias=True))
    report=dict(kernel_p_plus=table[:,1].tolist(),population_g_J0=endpoints[0,0].tolist(),
        population_g_Jinfinity=endpoints[1,0].tolist(),roots=roots,
        endpoint_residual_over_span=(np.linalg.norm(endpoint_r-endpoints[:,0],axis=-1)/span).tolist(),
        population_covariance_eigenvalues=eig.tolist(),weak_fraction=float(eig[0]/eig.sum()),
        decimation_reference_vector_rmse_over_span=float(np.sqrt(np.mean(np.sum((predicted[:,0]-expected)**2,axis=-1)))/span),
        population_closure_vector_rmse_by_scale=np.sqrt(np.mean(np.sum((predicted-g[:,1:])**2,axis=-1),axis=0)).tolist(),
        positive_alignment_with_decimation_fraction=float(np.mean(np.sum((predicted[:,0]-g[:,0])*(expected-g[:,0]),axis=-1)>0)),
        note='Finite L128 population pooled Q, not expectation of finite-sample Q. Couplings only diagnostic. Covariance rank is linear, not intrinsic dimension. Endpoint/extrapolation errors remain meaningful; no finite-temperature critical point.')
    out=args.run/'endpoint_report';out.mkdir(exist_ok=True)
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axs=plt.subplots(1,3,figsize=(16,4.8),layout='constrained')
    ax=axs[0]
    for k in range(6):ax.plot(*g[:,k].T,label=f'k={k}')
    for i,label in enumerate(['Q(J=0)','Q(J=infinity)']):ax.scatter(*endpoints[i,0],marker='D',s=55,label=label)
    for i,r in enumerate(roots):
        ax.scatter(*r['g'],marker='*',s=100,c='black');ax.annotate(f'F{i+1}: {r["stability"]}',r['g'],fontsize=8)
    ax.set(xlabel='g1',ylabel='g2',title='Population curves and R fixed points')
    ax=axs[1]
    for j in range(2):
        ax.plot(js,g[:,0,j],label=f'g{j+1}: microscopic')
        ax.plot(js,predicted[:,0,j],'--',label=f'g{j+1}: R(Q(J))')
        ax.plot(js,expected[:,j],':',label=f'g{j+1}: Q(exact decimation J)')
    ax.set(xlabel='J',ylabel='g coordinate',title='Coupling map and one-step comparison')
    ax=axs[2];chosen=[.1,.5,1.,2.,np.inf];direct=encode(chosen);roll=[direct[:,0]]
    with torch.no_grad():
        for _ in range(5):roll.append(m.advance_theory(torch.from_numpy(roll[-1])).numpy())
    roll=np.stack(roll,axis=1)
    direction=(endpoints[0,0]-endpoints[1,0])/span
    # A fixed endpoint-axis projection shows scale evolution without fitting
    # a separate axis or reparameterization at each scale.
    for i,j in enumerate(chosen):
        line,=ax.plot(range(6),(direct[i]-endpoints[1,0])@direction/span,'-o',label=f'J={j}')
        ax.plot(range(6),(roll[i]-endpoints[1,0])@direction/span,'--x',color=line.get_color())
    ax.set(xlabel='k',ylabel='Endpoint-axis coordinate',title='Solid Q(C^k D); dashed R^k(Q(D))')
    for ax in axs:ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.savefig(out/'physical_validation.png',dpi=170);plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
