"""Evaluate saved 500/1000/2000-step checkpoints on identical ensembles."""
import json
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recurrent_rg.data.views import balanced_views
from recurrent_rg.training import evaluate
from recurrent_rg.data.statistics import two_spin_distribution
from recurrent_rg.evaluation import find_fixed_points_1d
from recurrent_rg.model_nd import RecurrentRGNetwork1D


def main():
    torch.set_num_threads(2)
    root=Path(__file__).resolve().parents[1]
    base=root/'results/joint_qcr_tfim_1d_g1_v2_h60_sb128_block2_data_geometry_rollout5_seed91'
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,default=root/'results/joint_qcr_tfim_1d_g1_v2_h60_sb128_block2_data_geometry_rollout5_2000_seed91')
    run=parser.parse_args().run
    summary=json.loads((run/'summary.json').read_text())
    depth=summary['config']['levels']
    tx,th,vx,vh=torch.load(root/'data_cache/tfim1d_wolff_v2_sites128_M32_dtau0p5_eq2048_skip16_replicas3_ensemble128_seed91_nJ64.pt',weights_only=True)
    raw,h=torch.cat([tx,vx]),torch.cat([th,vh])
    values=[h[(h-target).abs().argmin()] for target in [.95,1.05]]
    ids=torch.cat([torch.where(h==v)[0] for v in values])
    fig,axes=plt.subplots(2,3,figsize=(16,9))
    hist=summary['history']
    for split in ['train','test']:
        axes[0,0].semilogy([v['epoch'] for v in hist],[v[f'{split}_total'] for v in hist],label=split)
    axes[0,0].set(title='Same objective: loss through 2000 updates',xlabel='Update',ylabel='Total loss')
    out={}
    for column,step in enumerate([500,1000,2000]):
        ckpt=torch.load(run/f'training_step_{step}.pt',weights_only=True)
        model=RecurrentRGNetwork1D(levels=depth,hidden=60,theory_dim=1).double().eval()
        model.load_state_dict(ckpt['model'],strict=False)
        model.theory_encoder.binary_lookup=True
        model.rg_kernel.binary_lookup=True
        row={}
        if step==500 and depth==4:
            old=torch.load(base/'qcr_model.pt',weights_only=True)
            row['max_weight_difference_from_original_500']=max(float((old[k]-ckpt['model'][k]).abs().max()) for k in old)
        for split,x in [('train',tx),('test',vx)]:
            indices=torch.linspace(0,len(x)-1,24).long()
            row[split+'_logged_protocol']=evaluate(model,x[indices],two_spin_distribution,48,
                objective_kwargs=dict(objective_mode='data_geometry',geometry_scale=4.,noise_multiplier=2.,rollout_weight=5.))
        direct,rollout=[],[]
        with torch.no_grad():
            for chunk in vx.split(8):
                levels=model.coarse_grain(balanced_views(chunk,128))
                gs=torch.stack([model.theory_encoder(z)[:,0] for z in levels],1)
                rs=[gs[:,0,None]]
                for _ in range(depth-1): rs.append(model.advance_theory(rs[-1]))
                direct.append(gs.numpy())
                rollout.append(torch.cat(rs,1).numpy())
            d,r=np.concatenate(direct),np.concatenate(rollout)
            row['full_test_rmse']=np.sqrt(np.mean((d-r)**2,axis=0)).tolist()
            row['full_test_std']=d.std(axis=0).tolist()
            row['full_test_nrmse']=(np.array(row['full_test_rmse'])/d.std(axis=0)).tolist()
            axes[0,1].plot(list(range(1,depth)),row['full_test_nrmse'][1:],'o-',label=f'{step} steps')
            grid=torch.linspace(-1.5,2,500,dtype=torch.float64)[:,None]
            axes[0,2].plot(grid[:,0],model.advance_theory(grid)[:,0],label=f'{step} steps')
            level=balanced_views(raw[ids],128)
            records=[]
            for k in range(6):
                g=model.theory_encoder(level)
                z=g if k==0 else model.advance_theory(z)
                records.append(dict(k=k,sites=level.shape[2],Q=g[:,0].tolist(),R=z[:,0].tolist()))
                if k<5: level=model.rg_kernel(level)
            row['nearby_ensembles']=dict(h_over_J=h[ids].tolist(),records=records)
            for j,value in enumerate(values):
                for key,style in [('Q','o-'),('R','x--')]:
                    series=[np.mean(rec[key][3*j:3*j+3]) for rec in records]
                    axes[1,column].plot(range(6),series,style,color=f'C{j}',label=f'{float(value):.3f}, {key}')
            if depth<6: axes[1,column].axvspan(depth-1+.05,5.1,color='gray',alpha=.12)
            axes[1,column].set(title=f'{step} updates: actual Q vs autonomous R',xlabel='Coarse-graining step k',ylabel='g')
        out[str(step)]=row
    axes[0,1].set(title='All test ensembles: rollout error',xlabel='Horizon k',ylabel='RMSE / direct-Q std',xticks=list(range(1,depth)))
    axes[0,2].plot([-1.5,2],[-1.5,2],'k:',label='identity')
    axes[0,2].set(title='R(g), coordinates unaligned across checkpoints',xlabel='g',ylabel='R(g)')
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle(f'TFIM dim(g)=1, seed 91: supervised k=0..{depth-1}, lr=0.001, fixed data and parameters')
    fig.text(.5,.01,f'Supervised through k={depth-1}. Lower panels average three replicas per h/J. Coordinates may shift between checkpoints.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.97))
    fig.savefig(run/'training_duration_comparison.png',dpi=170)
    (run/'duration_comparison.json').write_text(json.dumps(out,indent=2)+'\n')
    for step,row in out.items():
        print(step,json.dumps({k:v for k,v in row.items() if k not in ['nearby_ensembles','train_logged_protocol','test_logged_protocol']}))
        print('total losses',row['train_logged_protocol']['total'],row['test_logged_protocol']['total'])


if __name__=='__main__': main()
