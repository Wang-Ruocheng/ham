"""Magnetic-field departure from the 2D Ising critical point, local theory."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    out=Path('results/theory_ising2d_local_critical_scaling')
    out.mkdir(parents=True,exist_ok=True)
    kc=.5*np.log(1+np.sqrt(2));b=2.;yh=15/8
    steps=np.arange(6);factor=b**yh
    initial=[-2e-5,-1e-5,0.,1e-5,2e-5]
    colors=['#2166ac','#67a9cf','black','#ef8a62','#b2182b']
    fig,axs=plt.subplots(1,2,figsize=(11.5,4.7),layout='constrained')
    rows=[]
    for v,c in zip(initial,colors):
        flow=v*factor**steps
        axs[0].plot(steps,flow,'o-',color=c,lw=1.8,label=f'$B_0={v:+.0e}$' if v else '$B_0=0$')
        rows.append(dict(B0=v,B_linearized=flow.tolist()))
    axs[0].set(xlabel='RG step k (b = 2)',ylabel='Magnetic scaling coordinate (B units)',
               title='Positive / negative fields move away from zero',xticks=steps)
    axs[0].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    axs[0].legend(fontsize=8);axs[0].grid(alpha=.2)
    for v,c in [(1e-5,'#ef8a62'),(2e-5,'#b2182b')]:
        axs[1].plot(steps,2*v*factor**steps,'o-',color=c,label=f'$B_0=\pm {v:.0e}$')
    axs[1].set(xlabel='RG step k',ylabel=r'Pair separation $|B_k^{(+)}-B_k^{(-)}|$',
               title=f'Separation grows by {factor:.3f} per step',xticks=steps,yscale='log')
    axs[1].legend(fontsize=9);axs[1].grid(alpha=.2,which='both')
    fig.suptitle(r'2D Ising near $(K_c,0)$: $u_t=0$, $u_h\simeq B$, $u_h(k)=2^{15k/8}u_h(0)$')
    fig.savefig(out/'Kc_magnetic_separation.png',dpi=180);plt.close(fig)
    report=dict(Kc=kc,b=b,yh=yh,magnetic_eigenvalue=factor,steps=steps.tolist(),rows=rows,
        approximation='Universal local linear scaling with ut=0 and normalization uh=B to leading order. Not a global closed nearest-neighbor K/B recursion or a learned-kernel prediction.',
        caveat='Bare K=Kc at nonzero B can include nonlinear thermal scaling-field corrections of order B^2; holding thermal scaling field zero is a leading-order convention here. Spin-sign convention fixed across scales.',
        source='https://www.damtp.cam.ac.uk/user/tong/sft/sfthtml/S3.html')
    (out/'magnetic_theory.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
