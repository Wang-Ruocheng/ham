"""Three-dimensional surface of the full discrete map norm, NOT displacement."""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recurrent_rg.checkpoints import load_run


def main():
    root=Path('results/dual_ising2d_g2_dense_joint_fast_2400_seed91')
    out=root/'report';torch.set_num_threads(2)
    model=load_run(root).model.eval()
    bounds=np.array(json.loads((out/'fixed_points.json').read_text())['search_box'])
    x=np.linspace(bounds[0,0],bounds[1,0],161)
    y=np.linspace(bounds[0,1],bounds[1,1],161)
    xx,yy=np.meshgrid(x,y);points=np.column_stack([xx.ravel(),yy.ravel()])
    with torch.no_grad():mapped=model.advance_theory(torch.from_numpy(points)).numpy()
    z=np.linalg.norm(mapped,axis=1).reshape(xx.shape)
    data=np.load(out/'trajectories.npz')
    inputs=data['train_direct'][:,:-1].reshape(-1,2)
    with torch.no_grad():sample_z=model.advance_theory(torch.from_numpy(inputs)).norm(dim=-1).numpy()
    fig=plt.figure(figsize=(10,8),layout='constrained');ax=fig.add_subplot(111,projection='3d')
    surface=ax.plot_surface(xx,yy,z,cmap='viridis',rasterized=True,linewidth=0,antialiased=True,alpha=.92,rcount=121,ccount=121)
    ax.scatter(inputs[:,0],inputs[:,1],np.zeros(len(inputs)),s=3,c='.35',alpha=.35,label='Training R inputs (floor projection)',depthshade=False)
    ax.contour(xx,yy,z,zdir='z',offset=0,levels=10,cmap='viridis',linewidths=.7,alpha=.7)
    ax.set(xlabel=r'$g_1$',ylabel=r'$g_2$',zlabel=r'$\|R(g)\|_2$',
           xlim=tuple(bounds[:,0]),ylim=tuple(bounds[:,1]),zlim=(0,float(z.max())*1.07),
           title=r'Full discrete map norm: $\|R(g)\|_2 = \sqrt{R_1(g)^2+R_2(g)^2}$'+'\nNot a potential; not the displacement norm')
    ax.view_init(elev=28,azim=-125);ax.set_box_aspect((1.3,1,.9));ax.legend(loc='upper left',fontsize=8)
    fig.colorbar(surface,ax=ax,shrink=.65,pad=.07,label=r'$\|R(g)\|_2$')
    fig.savefig(out/'R_norm_surface_3d.png',dpi=190)
    ax.view_init(elev=30,azim=-50)
    fig.savefig(out/'R_norm_surface_3d_alternate.png',dpi=190)
    np.savez_compressed(out/'R_norm_surface_3d.npz',g1=xx,g2=yy,map_norm=z,mapped=mapped.reshape(*xx.shape,2),training_inputs=inputs,training_map_norm=sample_z)
    report=dict(checkpoint=str(root/'qcr_model.pt'),quantity='Euclidean norm of full discrete R(g), not R(g)-g',bounds=bounds.tolist(),grid_shape=list(z.shape),range=[float(z.min()),float(z.max())],note='The rectangular surface includes unsupported regions; gray floor dots show training input coverage. Height is coordinate-origin dependent and is not a potential or fixed-point residual.')
    (out/'R_norm_surface_3d.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
