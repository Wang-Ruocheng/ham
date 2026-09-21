"""Saved balanced-QCR diagnostics; cached exact full-ensemble predictions."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
from scipy.optimize import root
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.joint_qcr_ising2d_field import build_model


def main():
    repo=Path(__file__).resolve().parents[1]
    out=repo/'results/joint_qcr_ising2d_uniform_balanced_g2_seed91'
    old=repo/'results/joint_qcr_ising2d_uniform_field_g2_h60_L32_fixedC_2000_seed91_cached/step2000'
    torch.set_num_threads(2);model=build_model(False).eval()
    model.load_state_dict(torch.load(out/'qcr_model.pt',weights_only=True),strict=False)
    data=np.load(out/'field_flow_data.npz');baseline=np.load(old/'field_flow_data.npz')
    test=data['is_test'].astype(bool);report={}
    for name,f in [('baseline',baseline),('balanced',data)]:
        d,r=f['direct'],f['rollout'];training=d[~test,0]
        _,s,v=np.linalg.svd(training-training.mean(0),full_matrices=False)
        scales=s/np.sqrt(len(training)-1)
        e=(r[test]-d[test])@v.T/scales
        rmse=np.sqrt(((r[test]-d[test])**2).mean((0,2)))
        report[name]={'weak_fraction':float(scales[1]**2/(scales**2).sum()),
           'test_rollout_nrmse':(rmse/np.sqrt(d[test].var(0).mean(-1))).tolist(),
           'fixed_train_pc_nrmse_strong_weak':np.sqrt((e**2).mean(0)).tolist()}
        if name=='balanced':center=training.mean(0);vectors=v;sd=scales
    means=data['means'][:,:,0];ks=data['K_grid'];bs=data['B_grid'];y=data['KB']
    fig,axs=plt.subplots(1,2,figsize=(11,4.5))
    for ax,values,label in zip(axs,[np.broadcast_to(ks[:,None],means.shape[:2]),np.broadcast_to(bs[None],means.shape[:2])],['K','B']):
        for row in means:ax.plot(row[:,0],row[:,1],color='.8',lw=.4)
        for row in means.transpose(1,0,2):ax.plot(row[:,0],row[:,1],color='.8',lw=.4)
        im=ax.scatter(means[...,0],means[...,1],c=values,cmap='coolwarm',s=10)
        fig.colorbar(im,ax=ax,label=label);ax.set(xlabel='g1',ylabel='g2',title=f'Encoded microscopic grid colored by {label}')
        ax.set_aspect('equal',adjustable='datalim')
    fig.tight_layout();fig.savefig(out/'latent_KB_mapping.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for i,ax in enumerate(axs):
        im=ax.pcolormesh(bs,ks,means[...,i],shading='nearest',cmap='coolwarm');fig.colorbar(im,ax=ax)
        ax.set(xlabel='B',ylabel='K',title=f'g{i+1}(K,B)');ax.plot([0,0],[.4406868,.68],'k--',lw=1)
    fig.tight_layout();fig.savefig(out/'g_KB_maps.png',dpi=170);plt.close(fig)
    d=data['direct'];r=data['rollout'];low=d.reshape(-1,2).min(0);high=d.reshape(-1,2).max(0);pad=.08*(high-low)
    gx,gy=np.meshgrid(np.linspace(low[0]-pad[0],high[0]+pad[0],27),np.linspace(low[1]-pad[1],high[1]+pad[1],27))
    points=np.stack((gx.ravel(),gy.ravel()),1)
    def delta(x):
        with torch.no_grad():return (model.advance_theory(torch.as_tensor(x,dtype=torch.float64))-torch.as_tensor(x,dtype=torch.float64)).numpy()
    roots=[]
    for x in np.concatenate((means.reshape(-1,2)[::12],points[::20])):
        sol=root(delta,x)
        if not sol.success or np.linalg.norm(delta(sol.x))>1e-7 or any(np.linalg.norm(sol.x-a)<1e-4 for a in roots):continue
        if np.any(sol.x<low-pad) or np.any(sol.x>high+pad):continue
        roots.append(sol.x)
    records=[]
    for x in roots:
        jac=torch.autograd.functional.jacobian(model.advance_theory,torch.tensor(x,dtype=torch.float64)).detach().numpy()
        eig=np.linalg.eigvals(jac);typ='stable' if max(abs(eig))<1 else ('saddle' if min(abs(eig))<1 else 'unstable')
        distance=np.linalg.norm(d.reshape(-1,2)-x,axis=1).min()
        records.append({'g':x.tolist(),'eigenvalues':[[float(t.real),float(t.imag)] for t in eig],
                        'type':typ,'nearest_direct_point_distance':float(distance)})
    report['roots']=records
    flow=delta(points).reshape(*gx.shape,2)
    fig,ax=plt.subplots(figsize=(8,6))
    ax.quiver(gx,gy,flow[...,0],flow[...,1],angles='xy',scale_units='xy',scale=1,color='.5',width=.002)
    im=ax.scatter(means[...,0],means[...,1],c=np.broadcast_to(bs[None],means.shape[:2]),cmap='coolwarm',s=9)
    fig.colorbar(im,ax=ax,label='Original B')
    for i,rec in enumerate(records):
        ax.scatter(*rec['g'],marker='X',s=80,color='black');ax.annotate(f'F{i+1}: {rec["type"]}',rec['g'],xytext=(5,7),textcoords='offset points',fontsize=8)
    ax.set(xlabel='g1',ylabel='g2',title='R(g)-g: balanced loss, seed91, step2000');ax.set_aspect('equal',adjustable='box')
    fig.tight_layout();fig.savefig(out/'R_vector_field.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(11,7),sharex=True)
    for col,cases in enumerate([[(.425,0),(.44,0),(.455,0)],[(.44,-.05),(.44,0),(.44,.05)]]):
        for i,(K,B) in enumerate(cases):
            mask=np.isclose(y[:,0],K)&np.isclose(y[:,1],B,atol=1e-12)
            a=(d[mask]-center)@vectors.T/sd;b=(r[mask]-center)@vectors.T/sd
            for j in range(2):
                ax=axs[j,col];ax.errorbar(range(4),a.mean(0)[:,j],yerr=a.std(0,ddof=1)[:,j],fmt='o-',color=f'C{i}',label=f'K={K}, B={B}',capsize=2)
                ax.plot(range(4),b.mean(0)[:,j],'x--',color=f'C{i}');ax.set(ylabel=f'PC{j+1} / train SD',xticks=range(4),xlabel='k');ax.grid(alpha=.2)
        axs[0,col].legend(fontsize=8)
    fig.suptitle('Direct Q(C^k D): solid; R rollout: dashed. Fixed train-only PCA; replica SD bars.')
    fig.tight_layout();fig.savefig(out/'critical_closure_pca.png',dpi=170);plt.close(fig)
    history=json.loads((out/'history_progress.json').read_text())
    fig,axs=plt.subplots(2,3,figsize=(12,7))
    for ax,key in zip(axs.ravel(),['total','geometry','use','same','step','rollout']):
        for split in ('train','test'):ax.semilogy([a['epoch'] for a in history],[a[split+'_'+key] for a in history],label=split)
        ax.set(title=key,xlabel='Training step');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'loss_components.png',dpi=170);plt.close(fig)
    (out/'diagnostics.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
