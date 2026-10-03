"""Compare one- and two-step discrete displacements, with a local root zoom."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from recurrent_rg.checkpoints import load_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--root-index', type=int, help='One-based index of a known R fixed point to zoom')
    args = parser.parse_args()
    torch.set_num_threads(2)
    loaded = load_run(args.run)
    model = loaded.model.eval()
    folder = args.run/'full_report'
    roots = json.loads((folder/'fixed_points.json').read_text())
    if not roots['roots']:
        raise ValueError('No known R fixed point to zoom')
    if args.root_index is not None:
        if not 1 <= args.root_index <= len(roots['roots']):raise ValueError('Invalid root index')
        root = roots['roots'][args.root_index-1]
    else:
        root = next((r for r in roots['roots'] if r['stability']=='unstable'), roots['roots'][0])
    center = np.array(root['g'])
    x = torch.tensor(center, dtype=torch.double, requires_grad=True)
    one = model.advance_theory
    two = lambda a: one(one(a))
    j1 = torch.autograd.functional.jacobian(one, x).detach().numpy()
    j2 = torch.autograd.functional.jacobian(two, x).detach().numpy()
    np.testing.assert_allclose(j2, j1@j1, rtol=1e-7, atol=1e-8)
    eig1, vecs = np.linalg.eig(j1)
    eig2 = np.linalg.eigvals(j2)
    z = np.load(folder/'encodings.npz')
    g = np.concatenate([z['train'],z['test']])[:,0]
    kb = np.concatenate([z['train_labels'],z['test_labels']])
    norm = Normalize(kb[:,1].min(),kb[:,1].max())
    ks = np.unique(kb[:,0]); bs = np.unique(kb[:,1]); b0 = bs[np.argmin(abs(bs))]
    zero = np.array([g[(kb[:,0]==k)&(kb[:,1]==b0)].mean(0) for k in ks])
    lo,hi = np.array(roots['search_box'])
    zoom = .12*min(hi-lo)
    fig, axes = plt.subplots(2,2,figsize=(11,10),layout='constrained')
    for row in range(2):
        lower,upper = (lo,hi) if row==0 else (center-zoom,center+zoom)
        xx,yy = np.meshgrid(np.linspace(lower[0],upper[0],25),np.linspace(lower[1],upper[1],25))
        points = np.column_stack([xx.ravel(),yy.ravel()])
        for col, fn in enumerate((one,two)):
            ax = axes[row,col]
            with torch.no_grad():
                delta = fn(torch.from_numpy(points)).numpy()-points
            arrows = delta/np.maximum(np.linalg.norm(delta,axis=1,keepdims=True),1e-12)*.026*min(upper-lower)
            sc = ax.scatter(*g.T,c=kb[:,1],norm=norm,cmap='coolwarm',s=8,alpha=.45)
            ax.plot(*zero.T,'k-',lw=1,alpha=.7,label='B=0 microscopic curve')
            ax.quiver(*points.T,*arrows.T,color='.25',angles='xy',scale_units='xy',scale=1,width=.003)
            if row==0:
                for r in roots['roots']:
                    ax.scatter(*r['g'],marker='*',c='white',edgecolors='black',s=120,zorder=6)
            else:
                for n in range(2):
                    if abs(eig1[n].imag)>1e-8:continue
                    v = np.real(vecs[:,n]); v /= np.linalg.norm(v)
                    ends = center[None]+np.array([-.85,.85])[:,None]*zoom*v
                    ax.plot(*ends.T,ls='--',lw=1.3,color=['#009e73','#cc79a7'][n],label=f'eigen-direction {n+1}')
            ax.scatter(*center,marker='*',c='gold',edgecolors='black',s=150,zorder=7)
            ax.set(xlim=(lower[0],upper[0]),ylim=(lower[1],upper[1]),aspect='equal',xlabel='g1',ylabel='g2',
                   title=(r'$R(g)-g$' if col==0 else r'$R^2(g)-g$')+(' | full view' if row==0 else ' | center zoom'))
            if row==1:
                ax.legend(fontsize=7,loc='upper right')
    fig.colorbar(sc,ax=axes,label='Microscopic B (sampled ensembles)',shrink=.65)
    name = 'Direction-loss trial' if 'orientation0p1' in str(args.run) else 'Random probabilistic C baseline'
    if loaded.config['loss'].get('kernel_sign_weight',0):
        name = f'Random probabilistic C | sign weight={loaded.config["loss"]["kernel_sign_weight"]:g}'
    fig.suptitle(name+f'\nDR eigenvalues: {eig1[0]:.4f}, {eig1[1]:.4f}; '
                 +f'D(R^2): {eig2[0]:.4f}, {eig2[1]:.4f}\n'
                 +'Normalized displacement arrows, not continuous-time trajectories')
    fig.savefig(folder/'R_one_vs_two_step.png',dpi=170)
    plt.close(fig)
    report = dict(center=center.tolist(),jacobian_R=j1.tolist(),jacobian_R2=j2.tolist(),
                  eigenvalues_R=eig1.real.tolist(),eigenvalues_R2=eig2.real.tolist(),
                  eigenvalues_R_imag=eig1.imag.tolist(),eigenvalues_R2_imag=eig2.imag.tolist(),
                  root_residual_R2=float(torch.linalg.vector_norm(two(x)-x).detach()),
                  note='R^2 means composition R(R(g)). Only known R roots are marked; period-two cycles were not searched. Off-data arrows are model extrapolation.')
    (folder/'two_step_flow.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
