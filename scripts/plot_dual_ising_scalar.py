"""Annotate a scalar dual-Ising report with the two known input transitions."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recurrent_rg.checkpoints import load_run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--checkpoint',type=int)
    p.add_argument('--output',type=Path,help='Directory containing diagnostic encodings and fixed points')
    kind=p.add_mutually_exclusive_group()
    kind.add_argument('--signed-ising',action='store_true',help='Annotate zero-field single Ising with critical K=+/-Kc instead of dual layers')
    kind.add_argument('--positive-ising',action='store_true',help='Annotate zero-field Ising with K>=0 and one critical point')
    kind.add_argument('--signed-spin1',action='store_true',help='Zero-crystal-field spin-1 Ising, numerical reference |Kc|~1/1.693')
    kind.add_argument('--positive-spin1',action='store_true',help='Positive K spin-1, numerical critical reference 1/1.693')
    args=p.parse_args();torch.set_num_threads(1)
    run=load_run(args.run,args.checkpoint)
    if run.config['model']['theory_dim']!=1:raise ValueError('Requires scalar g')
    out=args.output or args.run/('full_report' if args.checkpoint is None else f'full_report_{args.checkpoint}')
    z=np.load(out/'encodings.npz')
    g=np.concatenate([z['train'],z['test']])[:,:,0]
    K=np.concatenate([z['train_labels'],z['test_labels']]).reshape(-1)
    knots=np.unique(K);means=np.stack([g[K==v].mean(0) for v in knots])
    kc=.5*np.log(1+np.sqrt(2));critical=np.array([-kc,kc] if args.signed_ising else [kc,2*kc])
    sampling_file=Path(run.config['data']['cache']).parent/'sampling.json'
    dual_metadata=None
    if not any([args.signed_ising,args.positive_ising,args.signed_spin1,args.positive_spin1]) and sampling_file.exists():
        candidate=json.loads(sampling_file.read_text())
        if candidate.get('model')=='two uncoupled square Ising layers' and candidate.get('grid_mode')=='ratio_line':
            dual_metadata=candidate
            critical=np.asarray(candidate['critical_K'],dtype=float)
    if args.positive_ising:critical=np.array([kc])
    if args.signed_spin1:critical=np.array([-1/1.693,1/1.693])
    if args.positive_spin1:critical=np.array([1/1.693])
    markers=np.interp(critical,knots,means[:,0])
    records=json.loads((out/'fixed_points.json').read_text())
    lo,hi=np.array(records['search_box']).reshape(2)
    x=np.linspace(lo,hi,2001)
    with torch.no_grad():y=run.model.advance_theory(torch.from_numpy(x[:,None])).numpy()[:,0]
    fig,axes=plt.subplots(1,3,figsize=(16,4.7),layout='constrained')
    axes[0].plot(x,y,label='R(g)');axes[0].plot(x,x,'k--',label='Identity')
    axes[1].plot(x,y-x,label='R(g)-g');axes[1].axhline(0,c='k',ls='--')
    inputs=z['train'][:,:-1,0]
    for ax in axes[:2]:
        ax.axvspan(inputs.min(),inputs.max(),alpha=.08,color='green',label='Training input range')
    for k in range(means.shape[1]):axes[2].plot(knots,means[:,k],label=f'Q(C^{k}D)')
    for i,(k,v,color) in enumerate(zip(critical,markers,['crimson','deepskyblue'])):
        for ax in axes[:2]:ax.axvline(v,c=color,ls=':',label=f'Q(Kc{i+1})')
        axes[2].axvline(k,c=color,ls=':',label=f'Kc{i+1}={k:.4f}')
    for i,r in enumerate(records['roots']):
        v=r['g'][0];color='green' if r['stability']=='stable' else 'darkorange'
        axes[0].scatter(v,v,c=color,s=45,zorder=5)
        axes[1].scatter(v,0,c=color,s=45,zorder=5)
        axes[1].annotate(f'F{i+1}: {r["stability"]}',(v,0),xytext=(3,9),textcoords='offset points',fontsize=8)
    for ax,title,xlabel,ylabel in zip(axes,['Discrete map','Scalar drift','Coupling-to-g by scale'],['g','g','Initial K=J1/T'],['R(g)','R(g)-g','g']):
        ax.set(title=title,xlabel=xlabel,ylabel=ylabel);ax.legend(fontsize=7);ax.grid(alpha=.2)
    step=f'checkpoint {args.checkpoint}' if args.checkpoint is not None else 'final checkpoint'
    system='Signed-K square Ising, B=0' if args.signed_ising else 'Dual Ising K2=K1/2'
    if dual_metadata:
        scan=dual_metadata.get('scan_axis','K1');other='K2' if scan=='K1' else 'K1'
        system=f'Dual Ising {other}={dual_metadata["ratio"]:g} {scan}'
        axes[2].set_xlabel(f'Initial {scan}')
    if args.positive_ising:system='Square Ising, K>=0, B=0'
    if args.signed_spin1:system='Signed-K spin-1 Ising, Delta=B=0 (numerical critical reference)'
    if args.positive_spin1:system='Positive-K spin-1 Ising, Delta=B=0 (numerical critical reference)'
    if args.signed_ising or args.positive_ising:axes[2].set_xlabel('Initial K=J/T')
    if args.signed_spin1:axes[2].set_xlabel('Initial K=J/T')
    if args.positive_spin1:axes[2].set_xlabel('Initial K=J/T')
    fig.suptitle(f'{system}; {step}. Critical markers annotate input data, not fitted fixed points.')
    fig.savefig(out/'R_and_two_transitions.png',dpi=160);plt.close(fig)
    (out/'critical_markers.json').write_text(json.dumps(dict(critical_K=critical.tolist(),microscopic_g=markers.tolist(),checkpoint=args.checkpoint),indent=2)+'\n')


if __name__=='__main__':main()
