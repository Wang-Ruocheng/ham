"""AT diamond hierarchical RG: exact on that graph, MK-type on square lattice.

Two length-2 branches in parallel, b=2, dimension log(4)/log(2)=2.
Series first: S=T@T; parallel: T'=S elementwise squared.
No square-lattice exact critical couplings are claimed.
"""
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def step(k, u):
    # T has three classes: A=exp(2k+u), B=exp(-u), C=exp(-2k+u).
    a,b,c=2*k+u,-u,-2*k+u
    sa=logsumexp([2*a,2*b+np.log(2),2*c])
    sb=np.log(2)+b+logsumexp([a,c])
    sc=np.log(2)+logsumexp([a+c,2*b])
    return np.array([.5*(sa-sc),.5*(sa+sc-2*sb)])


def trajectory(k, n=18):
    rows=[np.array([k,2*k])]
    for _ in range(n):rows.append(step(*rows[-1]))
    return np.array(rows)


def phase(k):
    a,b=trajectory(k,24)[-1]
    return 2 if a>1 else (1 if b>1 else 0)


def boundary(lo,hi,target):
    for _ in range(45):
        mid=(lo+hi)/2
        if phase(mid)>=target:hi=mid
        else:lo=mid
    return (lo+hi)/2


def verify():
    # Independent explicit sum over four internal states verifies extraction.
    spins=np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
    for k,u in [(0.,0.),(.2,.4),(.4,.8),(.3,0.)]:
        bonds=spins[:,None,:]*spins[None,:,:]
        t=np.exp(k*bonds.sum(-1)+u*bonds.prod(-1))
        transformed=(t@t)**2
        a,b,c=transformed[0,0],transformed[0,1],transformed[0,3]
        direct=np.array([np.log(a/c)/4,np.log(a*c/b**2)/4])
        np.testing.assert_allclose(step(k,u),direct,atol=1e-12)
    # Decoupled Ising and product-only Ising limits.
    np.testing.assert_allclose(step(.3,0),[np.log(np.cosh(.6)),0],atol=1e-12)
    np.testing.assert_allclose(step(0,.3),[0,np.log(np.cosh(.6))],atol=1e-12)


def main():
    verify()
    out=Path('results/ashkin_teller_theoretical_MK');out.mkdir(exist_ok=True)
    critical=[boundary(.15,.28,1),boundary(.28,.4,2)]
    starts=[.15,.28,.4];names=['Disordered','Product ordered','Fully ordered']
    colors=['#3274a1','#d58b20','#bc4455'];fig,axs=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
    rows={}
    for k,name,color in zip(starts,names,colors):
        flow=trajectory(k);z=np.tanh(flow);rows[name]={'K2_initial':k,'K4_initial':2*k,'couplings':flow.tolist()}
        for j in range(2):axs[j].plot(np.arange(len(flow)),z[:,j],'.-',color=color,label=f'{name}: K0={k:g}')
        axs[2].plot(*z.T,'.-',color=color,label=name)
        for i in range(1,8):
            if np.linalg.norm(z[i+1]-z[i])>.015:axs[2].annotate('',xy=z[i+1],xytext=z[i],arrowprops={'arrowstyle':'->','color':color})
    for j in range(2):
        axs[j].set(xlabel='RG step k',ylabel=['tanh(K2,k)','tanh(K4,k)'][j],ylim=(-.04,1.07),title=['Two-spin coupling K2 = J/T','Four-spin coupling K4 = U/T'][j])
        axs[j].set_xticks(np.arange(0,19,3));axs[j].grid(alpha=.2)
        axs[j].text(.03,.96,'1 represents infinite coupling',transform=axs[j].transAxes,va='top',fontsize=9,color='.35')
    axs[0].legend(loc='center right',fontsize=8)
    raw=np.linspace(0,2.5,200);axs[2].plot(np.tanh(raw),np.tanh(2*raw),'--',color='.65',lw=1,label='Initial sampling line K4=2 K2')
    axs[2].scatter([0,0,1],[0,1,1],c=colors,s=80,marker='s',zorder=4)
    for xy,txt,offset in [((0,0),'D: (0,0)',(8,9)),((0,1),'P: (0,infinity)',(8,-18)),((1,1),'F: ordered sink',(-95,-20))]:
        axs[2].annotate(txt,xy,xytext=offset,textcoords='offset points',fontsize=9)
    axs[2].set(xlabel='tanh(K2)',ylabel='tanh(K4)',xlim=(-.05,1.05),ylim=(-.05,1.05),title='Three distinct phase sinks; not three K2 values')
    axs[2].legend(loc='lower right',fontsize=7);axs[2].grid(alpha=.2)
    fig.suptitle('Ashkin-Teller RG: diamond hierarchy / MK-type approximation\nExact on the hierarchical graph, NOT exact square-lattice RG',fontsize=13)
    fig.savefig(out/'three_phase_flow.png',dpi=180)
    report={'scheme':'T_next=(T@T)**2, elementwise square; two length-2 branches in parallel; b=2',
            'Hamiltonian':'-beta H = sum[K2*(sigma_i*sigma_j+tau_i*tau_j)+K4*sigma_i*sigma_j*tau_i*tau_j]',
            'initial_line':'K4=2*K2 only at step zero',
            'approximate_boundary_intersections':critical,'trajectories':rows,
            'caveats':['Not exact square-lattice RG or fixed-site decimation.','Boundary values belong only to this approximation.','Compactification tanh shows infinity as 1; a phase sink is not a finite critical coupling.'],
            'sources':['https://pmc.ncbi.nlm.nih.gov/articles/PMC10858004/','https://www.sciencedirect.com/science/article/abs/pii/S0378437123008038']}
    (out/'theory.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'approximate_boundaries':critical,'final_couplings':{k:v['couplings'][-1] for k,v in rows.items()},'verification':'Explicit transfer matrix and Ising limits passed'},indent=2))


if __name__=='__main__':main()
