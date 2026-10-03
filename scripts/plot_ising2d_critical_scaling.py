"""Local RG scaling illustration, not an exact closed nearest-neighbor RG map."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    out=Path('results/theory_ising2d_local_critical_scaling');out.mkdir(parents=True,exist_ok=True)
    kc=.5*np.log(1+np.sqrt(2));steps=np.arange(5)
    deltas=[-.001,-.0005,0,.0005,.001];colors=['#2166ac','#67a9cf','black','#ef8a62','#b2182b']
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    rows=[]
    for delta,c in zip(deltas,colors):
        trajectory=kc+delta*2.**steps
        axs[0].plot(steps,trajectory,'-o',color=c,label=f'K0=Kc{delta:+.4f}' if delta else 'K0=Kc')
        rows.append(dict(K0=kc+delta,B0=0,K_linearized=trajectory.tolist(),B=[0]*len(steps)))
    axs[0].axhline(kc,color='black',ls=':',alpha=.5)
    axs[0].set(xlabel='RG steps k (length rescaling b=2)',ylabel='K coordinate (local linearized convention)',title='B=0: departures from Kc grow on opposite sides',xticks=steps);axs[0].legend(fontsize=8);axs[0].grid(alpha=.2)
    for delta,c in [( .0005,'#ef8a62'),(.001,'#b2182b')]:
        axs[1].plot(steps,2*delta*2.**steps,'-o',color=c,label=f'K0=Kc +/- {delta:g}')
    axs[1].set(xlabel='RG steps k',ylabel='Pair separation |K+(k)-K-(k)|',title='Two-sided separation doubles per step',xticks=steps);axs[1].set_yscale('log',base=2);axs[1].grid(alpha=.2);axs[1].legend(fontsize=8)
    fig.suptitle('2D Ising critical scaling: thermal exponent yt=1, magnetic exponent yh=15/8\nUniversal local linearization; NOT the analytic flow of plain decimation')
    fig.savefig(out/'B0_separation.png',dpi=180);plt.close(fig)
    (out/'theory.json').write_text(json.dumps(dict(Kc=kc,b=2,yt=1,yh=15/8,rows=rows,
        convention='Scaling fields ut~K-Kc and uh~B. Plot fixes linear units ut=K-Kc, uh=B; K plotted is a local coordinate, not inferred effective nearest-neighbor coupling after exact decimation.',
        scope='Only leading local critical scaling. Higher-order terms and irrelevant couplings omitted; no finite-size or Monte Carlo data used.',source='https://www.damtp.cam.ac.uk/user/tong/sft/sfthtml/S3.html'),indent=2)+'\n')
    print(out.resolve())


if __name__=='__main__':main()
