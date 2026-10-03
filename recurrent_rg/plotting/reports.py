"""Generic scalar/vector QCR reports, without physical-model guesses."""
from pathlib import Path
import json
import numpy as np
import torch
from ..evaluation import scalar_flow, discrete_fixed_point


def plot_diagnostics(model, trajectories, couplings, coupling_names, output, *, training_inputs=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.optimize import root
    from scipy.spatial import cKDTree
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    d, r = trajectories['direct'], trajectories['rollout']
    dim = d.shape[-1]
    inputs = None if training_inputs is None else np.asarray(training_inputs).reshape(-1,dim)
    points = d.reshape(-1,dim)
    bounds_points = points if inputs is None else np.concatenate((points,inputs))
    pad = np.maximum(.2,.2*np.ptp(bounds_points,axis=0))
    lower,upper = bounds_points.min(0)-pad,bounds_points.max(0)+pad
    roots=[];search_note='Bounded numerical search is not exhaustive.'
    if dim == 1:
        flow=scalar_flow(model,float(lower[0]),float(upper[0]))
        roots=flow['roots']
        search_note=flow['search_note']
        fig,ax=plt.subplots(figsize=(7,6))
        ax.plot(flow['grid'],flow['mapped'],label='R(g)')
        ax.plot(flow['grid'],flow['grid'],'k--',label='Identity')
        if inputs is not None:
            ax.axvspan(inputs.min(),inputs.max(),alpha=.12,color='green',label='Training input range')
        for item in roots:
            g=item['g'][0];stable=item['stability']=='stable'
            ax.scatter(g,g,c='green' if stable else 'red',marker='o' if stable else 'x')
            item['inside_training_input_range'] = None if inputs is None else bool(inputs.min()<=g<=inputs.max())
        ax.set(xlabel='g',ylabel='R(g)',title='Discrete scalar RG map');ax.legend();ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(out/'R_vs_g.png',dpi=160);plt.close(fig)
    elif dim == 2:
        gx,gy=np.meshgrid(np.linspace(lower[0],upper[0],27),np.linspace(lower[1],upper[1],27))
        grid=np.stack((gx.ravel(),gy.ravel()),1)
        with torch.no_grad(): delta=model.advance_theory(torch.from_numpy(grid)).numpy()-grid
        norm=np.linalg.norm(delta,axis=1); direction=delta/np.maximum(norm[:,None],1e-12)
        fig,ax=plt.subplots(figsize=(8,6))
        q=ax.quiver(grid[:,0],grid[:,1],direction[:,0],direction[:,1],norm,cmap='viridis',alpha=.65)
        fig.colorbar(q,ax=ax,label='|R(g)-g| (arrows normalized)')
        ax.scatter(points[:,0],points[:,1],s=4,c='black',alpha=.25,label='Evaluated encodings')
        if inputs is not None: ax.scatter(inputs[:,0],inputs[:,1],s=5,c='orange',alpha=.3,label='Training R inputs')
        def f(z):
            with torch.no_grad(): return model.advance_theory(torch.tensor(z,dtype=torch.float64)).numpy()-z
        for x in np.linspace(lower[0],upper[0],9):
            for y in np.linspace(lower[1],upper[1],9):
                fit=root(f,[x,y]);z=fit.x
                if not fit.success or np.linalg.norm(f(z))>1e-7 or np.any(z<lower) or np.any(z>upper): continue
                if any(np.linalg.norm(z-np.array(a['g']))<1e-5 for a in roots): continue
                item=discrete_fixed_point(model,z)
                item['nearest_training_input_distance']=None if inputs is None else float(cKDTree(inputs).query(z)[0])
                roots.append(item);ax.scatter(*z,marker='*',s=100,label=item['stability'])
        ax.set(xlabel='g1',ylabel='g2',title='Discrete displacement R(g)-g; off-data arrows are extrapolation')
        ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'R_vector_field.png',dpi=160);plt.close(fig)
    else:
        raise ValueError('Flow figures currently support dim(g)=1 or2')
    (out/'fixed_points.json').write_text(json.dumps(dict(roots=roots,search_box=[lower.tolist(),upper.tolist()],
        convention='Jacobian of full discrete R',search_note=search_note),indent=2)+'\n')
    if couplings is not None and coupling_names:
        labels=np.asarray(couplings).reshape(len(d),-1)
        if labels.shape[1]!=len(coupling_names): raise ValueError('Coupling metadata mismatch')
        fig,axes=plt.subplots(dim,labels.shape[1],squeeze=False,figsize=(6*labels.shape[1],3.5*dim))
        for j in range(dim):
            for a,name in enumerate(coupling_names):
                axes[j,a].scatter(labels[:,a],d[:,0,j],s=10,alpha=.65)
                axes[j,a].set(xlabel=name,ylabel=f'g{j+1}',title='Microscopic Q(D)');axes[j,a].grid(alpha=.2)
        fig.tight_layout();fig.savefig(out/'coupling_vs_g.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for i in np.linspace(0,len(d)-1,min(4,len(d))).astype(int):
        for j in range(dim):
            line=ax.plot(d[i,:,j],'-o',label=f'ensemble {i}, g{j+1}: Q')[0]
            ax.plot(r[i,:,j],'--x',color=line.get_color())
    ax.set(xlabel='k',ylabel='g',title='Direct Q: solid; autonomous R rollout: dashed');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'trajectories.png',dpi=160);plt.close(fig)
