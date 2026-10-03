"""Mark finite-size AT transition estimates on the fixed-C latent flow."""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recurrent_rg.checkpoints import load_run
from recurrent_rg.data.ensembles import load_ensembles


def main():
    torch.set_num_threads(2)
    folder=Path('results/ashkin_teller2d_g2_fixed_decimation_1200_seed91')
    run=load_run(folder);m=run.model;m.eval()
    tr,te=load_ensembles(run.config['data']['cache'])
    x=torch.cat([tr.configurations,te.configurations]);K=torch.cat([tr.couplings,te.couplings])[:,0].numpy()
    with torch.no_grad():g=torch.cat([m.theory_encoder(z.double()) for z in x.split(8)]).numpy()
    knots=np.unique(K);means=np.stack([g[K==v].mean(0) for v in knots])
    def interp(k):return np.array([np.interp(k,knots,means[:,j]) for j in range(2)])
    estimates=[(.19,(.18,.20),'tab:purple','Product ordering'),(.23,(.22,.24),'tab:red','Single-spin ordering')]
    roots=json.loads((folder/'report/fixed_points.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(14,5.8),layout='constrained')
    lo,hi=np.array(roots['search_box']);centers=np.stack([interp(v[0]) for v in estimates])
    zoomlo=centers.min(0)-[.22,.07];zoomhi=centers.max(0)+[.22,.07]
    records=[]
    for ax,(lower,upper),title in zip(axes,[(lo,hi),(zoomlo,zoomhi)],['Full flow','Transition-region zoom']):
        xx,yy=np.meshgrid(np.linspace(lower[0],upper[0],29),np.linspace(lower[1],upper[1],23))
        grid=np.column_stack([xx.ravel(),yy.ravel()])
        with torch.no_grad():delta=m.advance_theory(torch.from_numpy(grid)).numpy()-grid
        delta/=np.maximum(np.linalg.norm(delta,axis=1,keepdims=True),1e-15)
        ax.quiver(*grid.T,*delta.T,color='.65',alpha=.65,pivot='mid')
        ax.plot(*means.T,color='black',lw=1,label='Microscopic mean Q(D), increasing K along curve')
        scatter=ax.scatter(*g.T,c=K,cmap='viridis',s=10,alpha=.7)
        for value,interval,color,name in estimates:
            band=np.stack([interp(v) for v in np.linspace(*interval,70)])
            center=interp(value)
            ax.plot(*band.T,color=color,lw=7,alpha=.35,label=f'{name}: K in [{interval[0]:.2f}, {interval[1]:.2f}]')
            ax.scatter(*center,color=color,marker='D',s=80,edgecolor='white',zorder=5)
            ax.annotate(f'K ~ {value:.2f}',center,xytext=(7,12),textcoords='offset points',color=color)
        for r in roots['roots']:
            point=np.array(r['g'])
            if np.all(point>=lower) and np.all(point<=upper):
                ax.scatter(*point,c='orange',edgecolor='black',marker='*',s=150,label='Learned saddle',zorder=6)
        ax.set(xlim=(lower[0],upper[0]),ylim=(lower[1],upper[1]),xlabel='g1',ylabel='g2',title=title)
        ax.legend(fontsize=7,loc='upper left');ax.grid(alpha=.15)
        fig.colorbar(scatter,ax=ax,label='Microscopic K = J/T')
    fig.suptitle('Fixed decimation, dim_g=2: normalized R(g)-g\nMarkers/bands: finite-L transition estimates, NOT exact theoretical critical couplings',fontsize=11)
    for value,interval,color,name in estimates:
        records.append(dict(name=name,K_estimate=value,K_interval=interval,g=interp(value).tolist(),
                            g_interval_endpoints=[interp(v).tolist() for v in interval]))
    out=folder/'fixed_audit';out.mkdir(exist_ok=True)
    fig.savefig(out/'flow_transition_locations.png',dpi=180)
    (out/'transition_locations.json').write_text(json.dumps(dict(markers=records,
        method='Piecewise linear interpolation of 128-sample microscopic Q means over three ensembles per K.',
        caveat='K intervals are heuristic finite-L=16 transition neighborhoods, not confidence intervals or exact theoretical critical values.'),indent=2)+'\n')
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
