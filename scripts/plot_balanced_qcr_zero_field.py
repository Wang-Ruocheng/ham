"""Overlay microscopic zero-field encoding on the learned RG vector field."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.joint_qcr_ising2d_field import build_model


def main():
    repo=Path(__file__).resolve().parents[1]
    out=repo/'results/joint_qcr_ising2d_uniform_balanced_g2_seed91'
    f=np.load(out/'field_flow_data.npz');ks=f['K_grid'];bs=f['B_grid']
    means=f['means'][:,:,0];j=int(np.argmin(abs(bs)));line=means[:,j]
    roots=json.loads((out/'diagnostics.json').read_text())['roots']
    zero=np.isclose(f['KB'][:,1],0,atol=1e-12)
    replicas=f['direct'][zero,0]
    torch.set_num_threads(2);m=build_model(False).eval()
    m.load_state_dict(torch.load(out/'qcr_model.pt',weights_only=True),strict=False)
    stable=np.array([r['g'] for r in roots if r['type']=='stable'])
    # Local window contains all cell means and stable roots, not every noisy replica.
    local=np.concatenate([line,stable]);mid=.5*(local.min(0)+local.max(0))
    span=max(np.ptp(local,axis=0))*1.4
    lower=mid-span/2;upper=mid+span/2
    fig,axs=plt.subplots(1,2,figsize=(14,6))
    for col,ax in enumerate(axs):
        if col==0:
            allg=f['direct'].reshape(-1,2);lo=allg.min(0);hi=allg.max(0);pad=.1*(hi-lo);lo-=pad;hi+=pad
        else:lo,hi=lower,upper
        xx,yy=np.meshgrid(np.linspace(lo[0],hi[0],25),np.linspace(lo[1],hi[1],25))
        points=torch.tensor(np.stack([xx.ravel(),yy.ravel()],1),dtype=torch.float64)
        with torch.no_grad():v=(m.advance_theory(points)-points).numpy().reshape(*xx.shape,2)
        # Arrows have true one-step displacement; only clip display to panel extent.
        ax.quiver(xx,yy,v[...,0],v[...,1],angles='xy',scale_units='xy',scale=1,color='.65',width=.0025,zorder=1)
        ax.scatter(means[...,0],means[...,1],c=np.broadcast_to(bs[None],means.shape[:2]),cmap='coolwarm',vmin=-.5,vmax=.5,s=8,alpha=.3,zorder=2)
        ax.plot(line[:,0],line[:,1],'-',color='#00a08a',lw=2.3,zorder=4,label='B=0: mean Q(D), increasing K')
        im=ax.scatter(line[:,0],line[:,1],c=ks,cmap='viridis',vmin=ks.min(),vmax=ks.max(),s=32,edgecolors='white',linewidths=.4,zorder=5)
        if col==1:
            ax.scatter(replicas[:,0],replicas[:,1],s=13,marker='+',color='#00a08a',alpha=.3,zorder=3,label='Individual B=0 ensembles')
            for i in [0,5,10,20,27]:
                ax.annotate('',xy=line[i+1],xytext=line[i],arrowprops={'arrowstyle':'->','color':'#007568','lw':1.5},zorder=6)
            for i,offset in [(0,(8,-18)),(len(ks)-1,(8,8))]:
                ax.annotate(f'K={ks[i]:.2f}',line[i],xytext=offset,textcoords='offset points',fontsize=8)
        near=int(np.argmin(abs(ks-.5*np.log1p(np.sqrt(2)))))
        ax.scatter(*line[near],s=170,marker='*',facecolor='gold',edgecolor='black',zorder=7,label='K=0.44, B=0 (near critical K)')
        for n,r in enumerate(roots,1):
            x=np.array(r['g'])
            if np.all(x>=lo)&np.all(x<=hi):
                ax.scatter(*x,s=70,marker='X',color='black',zorder=8)
                ax.annotate(f'F{n}: {r["type"]}',x,xytext=(7,-12),textcoords='offset points',fontsize=9,zorder=9)
        ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),xlabel='g1',ylabel='g2',title='Full learned flow' if col==0 else 'Zero-field region (same g coordinates)')
        ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.15)
        if col==0:ax.add_patch(Rectangle(lower,span,span,fill=False,ec='#00a08a',ls='--',lw=1.4))
    axs[1].legend(fontsize=8,loc='upper left')
    fig.colorbar(im,ax=axs[1],label='Original K along B=0',fraction=.045,pad=.03)
    fig.suptitle('Balanced QCR, seed91, step2000: B=0 microscopic encoding over R(g)-g')
    fig.text(.5,.015,'Connecting line follows K, NOT RG time. Cell means use 3 ensembles; no smoothing. Gray arrows: R(g)-g; off-data flow is extrapolation.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.95));fig.savefig(out/'R_vector_field_Bzero.png',dpi=180);plt.close(fig)
    print('near critical g:',line[near].tolist(),'stable:',stable.tolist())


if __name__=='__main__':main()
