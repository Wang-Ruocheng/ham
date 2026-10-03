"""Two nested close-ups of a saved dim-g=2 discrete map, without retraining."""
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


def plot_interpolated_kb(out, center, critical, values, vectors, micro, labels, saved):
    """Forward interpolation on physical-neighbor triangles, not latent inversion."""
    ks, bs = np.unique(labels[:,0]), np.unique(labels[:,1])
    means = np.array([[micro[np.all(labels == [k,b], axis=1)].mean(0)
                      for b in bs] for k in ks])
    if not np.isfinite(means).all():
        raise ValueError('Interpolation requires a complete K/B grid')
    bary = np.array([[a/30,b/30,1-(a+b)/30] for a in range(31) for b in range(31-a)])
    dense, couplings = [], []
    for i in range(len(ks)-1):
        for j in range(len(bs)-1):
            for ids in ([(i,j),(i+1,j),(i+1,j+1)], [(i,j),(i+1,j+1),(i,j+1)]):
                pts = np.array([means[a,b] for a,b in ids])
                if np.any(pts.max(0)<center-.5) or np.any(pts.min(0)>center+.5):
                    continue
                dense.append(bary@pts)
                couplings.append(bary@np.array([[ks[a],bs[b]] for a,b in ids]))
    dense, couplings = np.concatenate(dense), np.concatenate(couplings)
    flat = means.reshape(-1,2)
    physical = np.array([[k,b] for k in ks for b in bs])
    fig, axes = plt.subplots(2,2,figsize=(12,11),layout='constrained')
    for row,radius in enumerate([.5,.06]):
        visible = np.all(abs(dense-center)<=radius,axis=1)
        # Shared physical K scale; B scale follows the displayed interpolated samples.
        bmax = max(float(np.max(abs(couplings[visible,1]))) if visible.any() else 0., 1e-6)
        for col,(name,cmap,norm) in enumerate([
            ('K','viridis',Normalize(ks.min(),ks.max())),
            ('B','coolwarm',Normalize(-bmax,bmax))]):
            ax = axes[row,col]
            points,delta = saved[f'points_{row}'],saved[f'displacement_{row}']
            arrows = delta/np.maximum(np.linalg.norm(delta,axis=1,keepdims=True),1e-12)*(.052*radius)
            ax.quiver(*points.T,*arrows.T,color='.6',angles='xy',scale_units='xy',scale=1,width=.0028,zorder=1)
            sc = ax.scatter(*dense[visible].T,c=couplings[visible,col],cmap=cmap,norm=norm,s=8,linewidths=0,zorder=3,rasterized=True)
            mask=np.all(abs(flat-center)<=radius,axis=1)
            ax.scatter(*flat[mask].T,c=physical[mask,col],cmap=cmap,norm=norm,s=23,edgecolors='black',linewidths=.4,zorder=4)
            ax.plot(*means[:,np.argmin(abs(bs))].T,color='black',alpha=.5,lw=.7,zorder=2,label='Microscopic B=0 curve')
            for n,val in enumerate(values):
                if abs(val.imag)>1e-9: continue
                segment=center+np.array([-.9,.9])[:,None]*radius*vectors[:,n].real
                ax.plot(*segment.T,'--',color='#009e73' if abs(val)<1 else '#d55e00',alpha=.7,lw=1.1,zorder=2)
            ax.scatter(*critical,marker='D',c='magenta',edgecolors='black',s=60,zorder=5,label='Critical input reference')
            ax.scatter(*center,marker='*',c='white',edgecolors='black',s=160,zorder=6,label='Learned fixed point')
            ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),aspect='equal',xlabel='g1',ylabel='g2',title=f'Interpolated {name} | center +/- {radius:g}')
            fig.colorbar(sc,ax=ax,shrink=.8,label=f'{name} (linear scale)')
    axes[0,0].legend(fontsize=7,loc='upper right')
    fig.suptitle('Local R(g)-g with forward-interpolated microscopic K/B\nOnly physical-neighbor triangles colored; overlaps may be nonunique; blank regions have no assigned K/B',fontsize=11)
    fig.savefig(out/'R_center_zoom_KB.png',dpi=190)
    plt.close(fig)
    (out/'R_center_zoom_KB_notes.json').write_text(json.dumps(dict(
        method='Barycentric forward interpolation of neighboring K/B grid triangles, 30 subdivisions per edge, using cell mean microscopic encodings',
        caveat='Colors are not a unique inverse map. Mapped triangles can overlap, especially for nearly collinear encodings. No off-support K/B extrapolation. B color range differs by zoom level.',
        center=center.tolist(),radii=[.5,.06]),indent=2)+'\n')
    print(out/'R_center_zoom_KB.png')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--root-index', type=int, default=0)
    p.add_argument('--interpolate-KB', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(2)
    model = load_run(args.run).model.eval()
    out = args.run/'full_report'
    item = json.loads((out/'fixed_points.json').read_text())['roots'][args.root_index]
    center = np.array(item['g'])
    jac = torch.autograd.functional.jacobian(
        lambda g: model.advance_theory(g[None])[0], torch.tensor(center)).detach().numpy()
    values, vectors = np.linalg.eig(jac)
    z = np.load(out/'encodings.npz')
    micro = np.concatenate([z['train'], z['test']])[:, 0]
    labels = np.concatenate([z['train_labels'], z['test_labels']])
    ks = np.unique(labels[:, 0])
    b0 = np.array([micro[np.isclose(labels[:, 0], k)&np.isclose(labels[:, 1], 0)].mean(0) for k in ks])
    kc = .5*np.log(1+np.sqrt(2))
    critical = np.array([np.interp(kc,ks,b0[:,d]) for d in range(2)])
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout='constrained')
    saved = {}
    for panel, (ax, radius) in enumerate(zip(axes, [.5, .06])):
        lo, hi = center-radius, center+radius
        xx, yy = np.meshgrid(np.linspace(lo[0], hi[0], 25), np.linspace(lo[1], hi[1], 25))
        points = np.column_stack([xx.ravel(), yy.ravel()])
        with torch.no_grad(): delta = model.advance_theory(torch.from_numpy(points)).numpy()-points
        speed = np.linalg.norm(delta, axis=1)
        arrows = delta/np.maximum(speed[:, None], 1e-12)*(.052*radius)
        ax.quiver(*points.T, *arrows.T, color='.45', angles='xy', scale_units='xy', scale=1, width=.003)
        sc = ax.scatter(*micro.T, c=labels[:,1], cmap='coolwarm', norm=Normalize(-1,1), s=19, edgecolors='white', linewidths=.25, zorder=4, label='Microscopic Q(D), colored by B')
        ax.plot(*b0.T, color='black', lw=1, alpha=.7, label='Microscopic B=0 curve', zorder=3)
        for n, val in enumerate(values):
            if abs(val.imag)>1e-9: continue
            direction = vectors[:,n].real
            segment = center+np.array([-.9,.9])[:,None]*radius*direction
            ax.plot(*segment.T, '--', color='#009e73' if abs(val)<1 else '#d55e00', lw=1.5,
                    label=f'Local eigenvector: lambda={val.real:.4f}', zorder=2)
        ax.scatter(*critical, marker='D', c='magenta', edgecolors='black', s=70, zorder=6, label='Q(D at Kc, B=0) reference')
        ax.scatter(*center, marker='*', c='white', edgecolors='black', s=170, zorder=7, label=f'F{args.root_index+1}: {item["stability"]}')
        ax.set(xlim=(lo[0],hi[0]), ylim=(lo[1],hi[1]), xlabel='g1', ylabel='g2', aspect='equal', title=f'Center +/- {radius:g}')
        ax.grid(alpha=.12)
        if panel == 0: ax.legend(fontsize=7, loc='upper right')
        saved[f'points_{panel}']=points; saved[f'displacement_{panel}']=delta
    fig.colorbar(sc, ax=axes, shrink=.8, label='Microscopic B (linear color scale)')
    fig.suptitle(f'Local discrete displacement R(g)-g around ({center[0]:.4f}, {center[1]:.4f})\nArrows normalized; off-data arrows are extrapolation; dashed lines are local eigenvectors, not trajectories', fontsize=11)
    fig.savefig(out/'R_center_zoom.png', dpi=180)
    plt.close(fig)
    np.savez_compressed(out/'R_center_zoom.npz', center=center, jacobian=jac, **saved)
    print(out/'R_center_zoom.png')
    if args.interpolate_KB:
        plot_interpolated_kb(out,center,critical,values,vectors,micro,labels,saved)


if __name__ == '__main__':
    main()
