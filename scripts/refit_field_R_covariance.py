"""Paired R-only regression with fixed, train-only covariance normalization."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recurrent_rg.model_nd import RecurrentRGNetwork2D


def errors(model, g):
    one = model.advance_theory(g[:, :-1])-g[:, 1:]
    z = g[:, 0]
    multi = []
    for k in range(1, g.shape[1]):
        z = model.advance_theory(z)
        multi.append(z-g[:, k])
    return one, torch.stack(multi, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=2000)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(91)
    start = time.perf_counter()
    summary = json.loads((args.source/'summary.json').read_text())
    cfg = summary['config']
    cached = np.load(args.source/f'step{cfg["epochs"]}'/'field_flow_data.npz')
    g = torch.from_numpy(cached['direct'])
    mask = cached['is_test']
    tr, te = g[~mask], g[mask]
    base = RecurrentRGNetwork2D(levels=cfg['levels'], hidden=cfg['hidden'], theory_dim=2).double().eval()
    mismatch = base.load_state_dict(torch.load(args.source/'qcr_model.pt',weights_only=True),strict=False)
    assert not mismatch.unexpected_keys and all(k.startswith('action_network.') for k in mismatch.missing_keys)
    for param in base.parameters():
        param.requires_grad_(False)
    base.theory_encoder.binary_lookup = base.rg_kernel.binary_lookup = True
    # Recompute several held-out/training encodings to verify the cached target/checkpoint pairing.
    tx, _, vx, _ = torch.load(cfg['data_cache'], weights_only=True)
    with torch.no_grad():
        for data, target in ((tx[:2], tr[:2]), (vx[:2], te[:2])):
            actual = torch.stack([base.theory_encoder(x) for x in base.coarse_grain(data.double())],1)
            torch.testing.assert_close(actual, target, rtol=1e-10, atol=1e-12)
    cov = torch.stack([torch.cov(tr[:,k].T, correction=0) for k in range(1,g.shape[1])])
    eig, axes = torch.linalg.eigh(cov)
    eps = 1e-6*cov.diagonal(dim1=-2,dim2=-1).mean(-1)
    adjusted = eig+eps[:,None]
    adjusted = torch.maximum(adjusted,adjusted[:,-1:]/1e6)
    weight = (axes@torch.diag_embed(adjusted.reciprocal())@axes.transpose(-1,-2)).detach()
    assert not weight.requires_grad
    torch.save(dict(train=tr,test=te,train_covariance=cov,weight=weight),args.output/'fixed_targets.pt')

    def objective(model, target, mode):
        es = errors(model,target)
        if mode == 'euclidean':
            return 5*sum(e.square().sum(-1).mean() for e in es)
        return 5*sum(torch.einsum('nki,kij,nkj->nk',e,weight,e).mean() for e in es)

    def metrics(model, target):
        with torch.no_grad():
            result = {}
            for name,e in zip(('one_step','rollout'),errors(model,target)):
                pc = torch.einsum('nki,kij->nkj',e,axes)
                result[name] = dict(rmse=e.square().mean((0,2)).sqrt().tolist(),
                    nrmse=(e.square().mean((0,2))/cov.diagonal(dim1=-2,dim2=-1).mean(-1)).sqrt().tolist(),
                    pc_nrmse_weak_strong=(pc.square().mean(0)/eig).sqrt().tolist())
            result['euclidean_loss'] = float(objective(model,target,'euclidean'))
            result['covariance_loss'] = float(objective(model,target,'covariance'))
            return result

    report = dict(source=str(args.source.resolve()),steps=args.steps,
        contract='Frozen Q/C, fixed full-ensemble targets; full-batch R-only fitting; identical pretrained R initialization, fresh Adam lr .001, cosine to .00001, seed91; no coupling labels; no test-based checkpoint selection.',
        normalization='Per-target-scale covariance from training ensembles only, epsilon=1e-6*trace(cov)/2, inverse condition cap 1e6; fixed/detached; equal scale weights; 5*one-step+5*rollout. This is not joint QCR loss.',
        covariance_eigenvalues_weak_strong=eig.tolist(),regularization_epsilon=eps.tolist(),
        effective_condition_number=(adjusted[:,-1]/adjusted[:,0]).tolist(),
        baseline=dict(train=metrics(base,tr),test=metrics(base,te)))
    models = {'baseline':base}
    for mode in ('euclidean','covariance'):
        model = copy.deepcopy(base)
        for param in model.theory_rg.parameters():
            param.requires_grad_(True)
        opt = torch.optim.Adam(model.theory_rg.parameters(),lr=.001)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.steps,eta_min=1e-5)
        history = []
        for step in range(1,args.steps+1):
            opt.zero_grad(set_to_none=True)
            loss = objective(model,tr,mode)
            assert torch.isfinite(loss), 'Nonfinite training loss'
            loss.backward()
            opt.step()
            sched.step()
            if step == 1 or step%100 == 0 or step==args.steps:
                history.append(dict(step=step,train=metrics(model,tr),test=metrics(model,te)))
                print(mode,step,'test weak-PC',history[-1]['test']['rollout']['pc_nrmse_weak_strong'],flush=True)
        for name,value in model.state_dict().items():
            if not name.startswith('theory_rg.'):
                assert torch.equal(value,base.state_dict()[name]),name
        dest = args.output/mode
        dest.mkdir()
        torch.save({k:v for k,v in model.state_dict().items() if not k.startswith('action_network.')},dest/'qcr_model.pt')
        report[mode] = dict(train=metrics(model,tr),test=metrics(model,te),history=history)
        models[mode] = model
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')

    fig,axs = plt.subplots(1,3,figsize=(15,4.4))
    kvals = np.arange(1,g.shape[1])
    for name,model in models.items():
        m = report[name]['test']['rollout']
        axs[0].plot(kvals,100*np.array(m['nrmse']),'o-',label=name)
        axs[1].plot(kvals,100*np.array(m['pc_nrmse_weak_strong'])[:,0],'o-',label=name)
        if name!='baseline':
            h=report[name]['history']
            axs[2].plot([r['step'] for r in h],[r['test']['covariance_loss'] for r in h],label=name)
    axs[0].set(title='Held-out pooled rollout error',ylabel='NRMSE (%)',xlabel='k',xticks=kvals)
    axs[1].set(title='Held-out weak-direction error',ylabel='PC NRMSE (%)',xlabel='k',xticks=kvals)
    axs[2].set(title='Common covariance metric (test)',ylabel='5 step + 5 rollout',xlabel='R-only updates',yscale='log')
    for ax in axs:
        ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(args.output/'error_comparison.png',dpi=180);plt.close(fig)

    # Identical physical encoding in every panel: only R differs.
    cloud = g.numpy().reshape(-1,2)
    lo,hi=cloud.min(0),cloud.max(0)
    padding=.12*np.maximum(hi-lo,.1);lo-=padding;hi+=padding
    xx,yy=np.meshgrid(np.linspace(lo[0],hi[0],27),np.linspace(lo[1],hi[1],27))
    points=np.column_stack((xx.ravel(),yy.ravel()))
    means=cached['means'][:,:,0];ks=cached['K_grid'];bs=cached['B_grid']
    fig,axs=plt.subplots(1,3,figsize=(15,6))
    for ax,(name,model) in zip(axs,models.items()):
        def residual(x):
            with torch.no_grad():
                return model.theory_rg(torch.as_tensor(x,dtype=torch.float64)).numpy()
        v=residual(points);speed=np.linalg.norm(v,axis=1)
        v=v/np.maximum(speed[:,None],1e-12)*.028*min(hi-lo)
        ax.quiver(*points.T,*v.T,color='.65',angles='xy',scale_units='xy',scale=1,width=.003)
        ax.scatter(*means.reshape(-1,2).T,c=np.tile(bs,len(ks)),cmap='coolwarm',s=8,vmin=-.5,vmax=.5)
        ax.plot(*means[:,np.argmin(abs(bs))].T,'k-',lw=1,label='B=0 input curve')
        roots=[]
        for x in np.linspace(lo[0],hi[0],9):
            for y in np.linspace(lo[1],hi[1],9):
                sol=root(residual,[x,y]);z=sol.x
                if np.linalg.norm(residual(z))>1e-8 or not np.isfinite(z).all() or (z<lo).any() or (z>hi).any() or any(np.linalg.norm(z-r['g'])<1e-4 for r in roots):continue
                jac=torch.autograd.functional.jacobian(model.advance_theory,torch.tensor(z,dtype=torch.float64)).detach().numpy()
                ev=np.linalg.eigvals(jac);mod=abs(ev)
                kind='stable' if (mod<1).all() else 'unstable' if (mod>1).all() else 'saddle'
                roots.append(dict(g=z.tolist(),eigenvalues=[dict(real=float(e.real),imag=float(e.imag)) for e in ev],kind=kind))
                ax.scatter(*z,c='black',marker='*',s=70)
        report[name]['roots_in_plot_box']=roots
        ax.set(title=name,xlabel='g1',ylabel='g2',xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]));ax.set_aspect('equal');ax.grid(alpha=.15)
    fig.suptitle('Fixed Q/C: R(g)-g; normalized arrows; off-data fields are extrapolation')
    fig.tight_layout();fig.savefig(args.output/'R_field_comparison.png',dpi=180);plt.close(fig)
    report['elapsed_seconds']=time.perf_counter()-start
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('DONE',report['elapsed_seconds'],flush=True)


if __name__=='__main__':main()
