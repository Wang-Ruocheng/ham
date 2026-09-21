"""Compare original and extended joint QCR checkpoints, without refitting."""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recurrent_rg.model_nd import RecurrentRGNetwork1D
from fit_frozen_qcr_R import metrics
from experiments.joint_qcr_ising import evaluate, two_spin_distribution


def main():
    torch.set_num_threads(1)
    out=Path('results/joint_qcr_tfim_1d_g2_coordsum_continue6000_seed91')
    runs={2000:Path('results/joint_qcr_tfim_1d_g2_coordsum_levels6_2000_seed91'),
          4000:out/'step4000',6000:out/'step6000'}
    results={};arrays={}
    tx,_,vx,_=torch.load('data_cache/tfim1d_wolff_v2_sites128_M32_dtau0p5_eq2048_skip16_replicas3_ensemble128_seed91_nJ64.pt',weights_only=True)
    for step,run in runs.items():
        data=np.load(run/'vector_flow_data.npz');g=torch.from_numpy(data['direct'])
        h=data['h_over_J'];is_test=data['is_test'].astype(bool)
        model=RecurrentRGNetwork1D(levels=6,hidden=60,theory_dim=2,kernel_layout='nonoverlapping').double().eval()
        model.load_state_dict(torch.load(run/'qcr_model.pt',weights_only=True),strict=False)
        model.theory_encoder.binary_lookup=model.rg_kernel.binary_lookup=True
        with torch.no_grad():
            z=g[:,0];rs=[z.numpy()]
            for _ in range(50):z=model.advance_theory(z);rs.append(z.numpy())
        r=np.stack(rs,1);item={'train':metrics(model,g[~is_test]),'test':metrics(model,g[is_test]),'pairs':{}}
        for key,a,b in [('near',.95,1.05),('far',.5,1.5)]:
            ha=h[np.argmin(abs(h-a))];hb=h[np.argmin(abs(h-b))];ia=h==ha;ib=h==hb
            direct=np.linalg.norm(data['direct'][ia].mean(0)-data['direct'][ib].mean(0),axis=-1)
            predicted=np.linalg.norm(r[ia].mean(0)-r[ib].mean(0),axis=-1)
            item['pairs'][key]={'h_over_J':[float(ha),float(hb)],'Q_distance':direct.tolist(),
                                'R_distance':predicted.tolist(),'R50_over_Q5':float(predicted[50]/max(direct[5],1e-12))}
        summary=json.loads((run/'summary.json').read_text());cfg=summary['config']
        exact={'epoch':step}
        for split,x in [('train',tx),('test',vx)]:
            ids=torch.linspace(0,len(x)-1,min(cfg['eval_systems'],len(x))).long()
            vals=evaluate(model,x[ids],two_spin_distribution,cfg['eval_samples'],objective_kwargs={
                k:cfg[k] for k in ['objective_mode','coordinate_reduction','geometry_scale','noise_multiplier','rollout_weight']})
            exact.update({split+'_'+k:v for k,v in vals.items()})
        item['logged_loss']=exact
        if not any(row['epoch']==step for row in summary['history']):
            summary['history'].append(exact)
            (run/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        results[step]=item;arrays[str(step)+'_rollout']=r
        fig,axs=plt.subplots(1,2,figsize=(10,4))
        knots=np.unique(h)
        for d,ax in enumerate(axs):
            for k in range(6):
                ys=np.array([data['direct'][h==v,k,d].mean() for v in knots])
                ax.plot(knots,ys,label=f'k={k}')
            ax.axvline(1,color='k',ls=':');ax.set(xlabel='h/J',ylabel=f'g{d+1}',title=f'{step} joint updates')
            ax.legend(fontsize=7);ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(out/f'coupling_mapping_step{step}.png',dpi=180);plt.close(fig)
    (out/'continuation_comparison.json').write_text(json.dumps(results,indent=2)+'\n')
    np.savez(out/'long_rollouts.npz',**arrays)
    fig,axs=plt.subplots(1,3,figsize=(15,4))
    history=json.loads((runs[6000]/'summary.json').read_text())['history']
    for split in ['train','test']:axs[0].plot([r['epoch'] for r in history],[r[split+'_total'] for r in history],label=split)
    axs[0].axvline(2000,color='gray',ls=':');axs[0].set(xlabel='Total joint updates',ylabel='Original total QCR loss',yscale='log')
    for step,item in results.items():
        axs[1].plot(range(1,6),item['test']['rollout_nrmse'],'o-',label=str(step))
        pair=item['pairs']['near'];axs[2].plot(range(51),np.array(pair['R_distance'])/pair['Q_distance'][5],label=str(step))
    axs[1].set(xlabel='k',ylabel='Full-test rollout NRMSE')
    axs[2].axvspan(5,50,color='gray',alpha=.12);axs[2].set(xlabel='k',ylabel='Near-pair R distance / actual Q distance at k=5')
    for ax in axs:ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'continuation_comparison.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(14,4))
    for ax,(step,item) in zip(axs,results.items()):
        for key,style in [('near','-'),('far','--')]:
            pair=item['pairs'][key]
            ax.plot(range(6),pair['Q_distance'],'o'+style,label=key+' Q')
            ax.plot(range(6),pair['R_distance'][:6],'x'+style,label=key+' R')
        ax.set(title=f'{step} joint updates',xlabel='k',ylabel='Pair distance in this checkpoint coordinates');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'pair_closure_comparison.png',dpi=180);plt.close(fig)
    print(json.dumps({s:{'test_closure':r['test']['closure_loss'],'test_k5_nrmse':r['test']['rollout_nrmse'][-1],
                       'near_R50_over_Q5':r['pairs']['near']['R50_over_Q5']} for s,r in results.items()},indent=2))


if __name__=='__main__':main()
