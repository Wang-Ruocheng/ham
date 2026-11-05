"""Exact thermodynamic phase boundaries of two uncoupled square Ising layers."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def main():
    kc=np.log(1+np.sqrt(2))/2
    out=Path('results/dual_ising_theory');out.mkdir(exist_ok=True)
    fig,(ax,bx)=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    grid=np.linspace(0,1.3,401);a,b=np.meshgrid(grid,grid)
    colors=['#d5e8f6','#ffdfb5','#d7ead3','#f1cdd7']
    ax.pcolormesh(grid,grid,(a>kc).astype(int)+2*(b>kc),cmap=ListedColormap(colors),shading='auto',vmin=0,vmax=3)
    ax.axvline(kc,c='k',ls='--');ax.axhline(kc,c='k',ls='--')
    ax.plot(grid,grid/2,c='#673ab7',lw=2.5,label='Sampling: K2 = K1 / 2')
    ax.scatter([kc,2*kc],[kc/2,kc],c='#673ab7',s=65,zorder=5)
    ax.annotate('transition 1',(kc,kc/2),xytext=(.48,.08),arrowprops={'arrowstyle':'->'},fontsize=9)
    ax.annotate('transition 2',(2*kc,kc),xytext=(.93,.32),arrowprops={'arrowstyle':'->'},fontsize=9)
    for x,y,txt in [(.21,.11,'Both disordered'),(.88,.16,'Only layer 1 ordered'),(.21,.87,'Only layer 2\nordered'),(.9,.95,'Both ordered')]:
        ax.text(x,y,txt,ha='center',fontsize=10)
    ax.set(xlabel='K1 = J1/T',ylabel='K2 = J2/T',xlim=(0,1.3),ylim=(0,1.3),title='Exact phase diagram (zero fields, no inter-layer coupling)')
    ax.legend(loc='upper right',fontsize=9)
    # Exact spontaneous magnetization magnitudes, thermodynamic limit.
    for ratio,name in [(1,'layer 1'),(.5,'layer 2')]:
        coupling=ratio*grid;mag=np.zeros_like(grid);mask=coupling>kc
        mag[mask]=(1-np.sinh(2*coupling[mask])**(-4))**.125
        bx.plot(grid,mag,lw=2,label=name)
    for k in [kc,2*kc]:bx.axvline(k,color='k',ls='--',lw=1)
    bx.set(xlabel='K = J1/T along sampling line',ylabel='Spontaneous magnetization magnitude',ylim=(-.04,1.08),title='Two exact transitions: K = 0.440687 and 0.881374')
    bx.legend();bx.grid(alpha=.2)
    fig.savefig(out/'phase_diagram.png',dpi=180)


if __name__=='__main__':main()
