"""Read-only diagnostic of the fixed-C QCR distance objective, not a new loss."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.joint_qcr_ising2d_field import build_model
from experiments.joint_qcr_fixed_cache import halves
from experiments.supervised_q_ising2d_multiscale import patterns_for


def main():
    root=Path(__file__).resolve().parents[1]
    run=root/'results/joint_qcr_ising2d_uniform_field_g2_h60_L32_fixedC_2000_seed91_cached'
    out=run/'geometry_pair_audit';out.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    model=build_model(False)
    state=torch.load(run/'training_step_2000.pt',weights_only=True)['model']
    missing=model.load_state_dict(state,strict=False)
    assert not missing.unexpected_keys and all(s.startswith('action_network.') for s in missing.missing_keys)
    flow=np.load(run/'step2000/field_flow_data.npz')
    pca=json.loads((run/'step2000/critical_closure_pca.json').read_text())
    axes=np.array(pca['pca_rows'])
    patterns=patterns_for(model.theory_encoder)
    bits=(patterns>0).long();ids=8*bits[:,4]+4*bits[:,7]+2*bits[:,5]+bits[:,8]
    marginal=torch.nn.functional.one_hot(ids,16).double()
    all_results={}
    for split in ('train','test'):
        counts=np.load(root/f'results/supervised_Q_structure_uniform_KB_seed91/{split}_sample_counts.npy',mmap_mode='r')
        y=flow['KB'][flow['is_test'].astype(bool)==(split=='test')]
        latent=[];pa=[];pb=[]
        for start in range(0,len(counts),16):
            ha,hb=halves(counts,np.arange(start,min(start+16,len(counts))),np.arange(256))
            with torch.no_grad():
                values=model.theory_encoder.local_mlp(patterns)
                latent.append((.5*(model.theory_encoder.head(ha@values)+model.theory_encoder.head(hb@values))).numpy())
                pa.append((ha@marginal).numpy());pb.append((hb@marginal).numpy())
        g=np.concatenate(latent);a=np.concatenate(pa);b=np.concatenate(pb)
        i,j=np.triu_indices(len(y),1)
        sameK=np.isclose(y[i,0],y[j,0]);sameB=np.isclose(y[i,1],y[j,1],atol=1e-12)
        signs=np.sign(np.where(abs(y[:,1])<1e-12,0,y[:,1]))
        masks={'all':np.ones(len(i),bool),'fixed_B_change_K':sameB&~sameK,
               'fixed_K_same_sign_B':sameK&~sameB&(signs[i]==signs[j])&(signs[i]!=0),
               'opposite_sign_B':signs[i]*signs[j]<0,
               'B_zero_change_K':sameB&~sameK&(signs[i]==0),
               'same_KB_replicas':sameK&sameB}
        rows=[]
        for k in range(4):
            mean=.5*(a[:,k]+b[:,k]);noise=.25*((a[:,k]-b[:,k])**2).sum(-1)
            distance=np.linalg.norm(mean[i]-mean[j],axis=-1)
            allowance=2*np.sqrt(noise[i]+noise[j])
            lo=4*np.maximum(distance-allowance,0);hi=4*(distance+allowance)
            delta=g[i,k]-g[j,k];gd=np.linalg.norm(delta,axis=-1)
            residual=np.minimum(gd-lo,0)+np.maximum(gd-hi,0)
            loss=residual**2
            # Gradient of ONE pair penalty wrt its first latent endpoint.
            force=2*residual[:,None]*delta/np.maximum(gd[:,None],1e-30)
            fpc=force@axes.T
            spectrum=np.linalg.svd(mean-mean.mean(0),compute_uv=False)**2
            row={'k':k,'target_variance_first_two':(spectrum[:2]/spectrum.sum()).tolist(),'groups':{}}
            for name,mask in masks.items():
                sq=(fpc[mask]**2).sum(0)
                row['groups'][name]={'pairs':int(mask.sum()),'pair_fraction':float(mask.mean()),
                   'target_distance':float((4*distance[mask]).mean()),'latent_distance':float(gd[mask].mean()),
                   'noise_allowance':float((4*allowance[mask]).mean()),'lower_zero_fraction':float((lo[mask]==0).mean()),
                   'under_distance_fraction':float((gd[mask]<lo[mask]).mean()),
                   'over_distance_fraction':float((gd[mask]>hi[mask]).mean()),
                   'mean_loss':float(loss[mask].mean()),'total_loss_share':float(loss[mask].sum()/loss.sum()),
                   'pair_force_PC2_fraction':float(sq[1]/max(sq.sum(),1e-30))}
            rows.append(row)
        all_results[split]=rows
        print(split,json.dumps(rows,indent=2),flush=True)
    result={'settings':'Full pair population; fixed first128/last128 split of all256 samples; checkpoint2000. Not replayed training minibatches.',
            'gradient_caveat':'Pair endpoint gradients before aggregation or Q chain rule, NOT parameter gradients. Categories overlap.',
            'results':all_results}
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    groups=['fixed_B_change_K','fixed_K_same_sign_B','opposite_sign_B','B_zero_change_K']
    labels=['Fixed B, vary K','Fixed K, same-sign B','Opposite-sign B','B=0, vary K']
    fig,axs=plt.subplots(1,3,figsize=(15,4))
    for name,label in zip(groups,labels):
        for ax,key in zip(axs,['mean_loss','lower_zero_fraction','pair_force_PC2_fraction']):
            ax.plot(range(4),[r['groups'][name][key] for r in all_results['train']],'o-',label=label)
    for ax,title in zip(axs,['Mean pair loss','Fraction with zero lower bound','PC2 fraction of pair force squared']):
        ax.set(title=title,xlabel='Scale k',xticks=range(4));ax.grid(alpha=.2)
    axs[2].set_yscale('log');axs[0].legend(fontsize=8)
    fig.suptitle('Training population diagnostic; categories overlap; fixed sample split')
    fig.tight_layout();fig.savefig(out/'pair_audit.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    main()
