"""Square-lattice AT phase topology; split branches are SCHEMATIC, not fits."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def main():
    out=Path('results/ashkin_teller_theoretical_MK');out.mkdir(exist_ok=True)
    potts=np.log(3)/4
    ising=np.log(1+np.sqrt(2))/2
    j=np.linspace(0,.67,750);u=np.linspace(0,1.35,850)
    J,U=np.meshgrid(j,u)
    # Illustrative monotone branches connect known endpoints/asymptote.
    # These expressions are NOT theoretical critical-line equations.
    lower=ising-(ising-potts)*(np.minimum(J,potts)/potts)**2
    upper_j=ising/2+(potts-ising/2)*np.exp(-np.maximum(U-potts,0)/.15)
    selfdual=np.arcsinh(np.exp(-2*U))/2
    regions=np.where(U<=potts,np.where(J>selfdual,2,0),
                     np.where(J>upper_j,2,np.where(U>lower,1,0)))
    fig,ax=plt.subplots(figsize=(8.4,7.2),layout='constrained')
    ax.pcolormesh(j,u,regions,cmap=ListedColormap(['#d7e8f6','#ffebc5','#f4d7da']),vmin=0,vmax=2,shading='auto',rasterized=True)
    ue=np.linspace(0,potts,300);ax.plot(np.arcsinh(np.exp(-2*ue))/2,ue,'k-',lw=2,label='Exact direct-transition boundary')
    js=np.linspace(0,potts,300);ax.plot(js,ising-(ising-potts)*(js/potts)**2,'k--',lw=1.7,label='Split boundaries: schematic only')
    us=np.linspace(potts,1.35,300);ax.plot(ising/2+(potts-ising/2)*np.exp(-(us-potts)/.15),us,'k--',lw=1.7)
    ax.plot([0,.67],[0,.67],':',color='.4',lw=1.3,label='U/J = 1 (Potts line)')
    ks=np.linspace(.08,.65,200);ax.plot(ks,2*ks,color='#673ab7',lw=2.5,label='Our data: U/J = 2')
    ax.annotate('',xy=(.53,1.06),xytext=(.45,.9),arrowprops={'arrowstyle':'->','color':'#673ab7','lw':2.5})
    ax.text(.47,1.12,'Increasing K = J/T\n(cooling at fixed U/J)',color='#673ab7',fontsize=10)
    ax.scatter([0,potts,ising],[ising,potts,0],c='k',s=28,zorder=6)
    ax.annotate('4-state Potts critical point\n(J/T, U/T) = (ln 3 / 4, ln 3 / 4)',(potts,potts),xytext=(.34,.42),fontsize=9,arrowprops={'arrowstyle':'-','color':'.25'})
    ax.text(.16,.12,'DISORDERED\nNeither spin nor product ordered',fontsize=10,ha='center')
    # Region labels in axes coordinates, clear of boundaries and sampling ray.
    ax.text(.105,.79,'PRODUCT ORDERED\nOnly sigma*tau ordered',ha='center',fontsize=10)
    ax.text(.48,.76,'FULLY ORDERED\nsigma, tau and sigma*tau ordered',ha='center',fontsize=10)
    ax.set(xlim=(0,.67),ylim=(0,1.35),xlabel='J / T  (two-spin coupling)',ylabel='U / T  (four-spin coupling)',
           title='Square-lattice Ashkin-Teller: three phase regions\nFerromagnetic quadrant; split boundaries are qualitative')
    ax.legend(loc='upper right',fontsize=8,framealpha=.95)
    ax.grid(alpha=.14)
    fig.savefig(out/'square_phase_regions.png',dpi=180)


if __name__=='__main__':main()
