"""Compare supervision depths on the same six actual data scales."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recurrent_rg.data.views import balanced_views
from recurrent_rg.model_nd import RecurrentRGNetwork1D


def main():
    torch.set_num_threads(2)
    root=Path(__file__).resolve().parents[1]
    old=root/'results/joint_qcr_tfim_1d_g1_v2_h60_sb128_block2_data_geometry_rollout5_2000_seed91'
    new=root/'results/joint_qcr_tfim_1d_g1_levels6_rollout5_2000_seed91'
    tx,th,vx,vh=torch.load(root/'data_cache/tfim1d_wolff_v2_sites128_M32_dtau0p5_eq2048_skip16_replicas3_ensemble128_seed91_nJ64.pt',weights_only=True)
    raw,h=torch.cat([tx,vx]),torch.cat([th,vh]).numpy()
    values=[h[np.argmin(abs(h-v))] for v in [.95,1.05]]
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    report={}
    for col,step in enumerate([1000,2000]):
        for directory,depth,style in [(old,3,'--'),(new,5,'-')]:
            model=RecurrentRGNetwork1D(levels=6,hidden=60,theory_dim=1).double().eval()
            state=torch.load(directory/f'training_step_{step}.pt',weights_only=True)['model']
            inc=model.load_state_dict(state,strict=False)
            assert not inc.unexpected_keys and all(k.startswith('action_network.') for k in inc.missing_keys)
            model.theory_encoder.binary_lookup=True
            model.rg_kernel.binary_lookup=True
            ds,rs=[],[]
            with torch.no_grad():
                for chunk in raw.split(8):
                    levels=model.coarse_grain(balanced_views(chunk,128))
                    d=torch.cat([model.theory_encoder(v) for v in levels],1)
                    seq=[d[:,:1]]
                    for _ in range(5): seq.append(model.advance_theory(seq[-1]))
                    ds.append(d.numpy()); rs.append(torch.cat(seq,1).numpy())
                pairs=torch.tensor([[-1,-1],[-1,1],[1,-1],[1,1]],dtype=torch.float64).reshape(4,1,2,1)
                truth=model.rg_kernel(pairs).flatten().tolist()
            d,r=np.concatenate(ds),np.concatenate(rs)
            test=d[len(tx):]; pred=r[len(tx):]
            rmse=np.sqrt(((test-pred)**2).mean(0)); std=test.std(0)
            item=dict(test_rmse=rmse.tolist(),test_std=std.tolist(),test_nrmse=(rmse/std).tolist(),C_truth=truth)
            item['nearby']={}
            for j,v in enumerate(values):
                mask=h==v
                item['nearby'][str(v)]=dict(Q=d[mask].mean(0).tolist(),R=r[mask].mean(0).tolist())
                if depth==5:
                    axes[1,col].plot(range(6),d[mask].mean(0),'o-',color=f'C{j}',label=f'h/J={v:.3f}, Q')
                    axes[1,col].plot(range(6),r[mask].mean(0),'x--',color=f'C{j}',label=f'h/J={v:.3f}, R')
                with torch.no_grad():
                    z=torch.from_numpy(d[mask,:1])
                    for _ in range(20): z=model.advance_theory(z)
                item['nearby'][str(v)]['R20_mean']=float(z.mean())
            report[f'depth{depth}_step{step}']=item
            axes[0,col].plot(range(1,6),(rmse/std)[1:],'o'+style,label=f'supervised k<= {depth}')
        axes[0,col].set(title=f'{step} updates: matched six-scale test',xlabel='k',ylabel='Rollout RMSE / direct-Q std',xticks=range(1,6))
        axes[1,col].set(title=f'New depth-5 model, {step} updates',xlabel='k (all shown scales supervised)',ylabel='g',xticks=range(6))
    for ax in axes.flat: ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Same seed, data and parameters; expanded supervision from k<=3 to k<=5')
    fig.tight_layout(rect=(0,0,1,.96))
    fig.savefig(new/'depth_comparison.png',dpi=180)
    (new/'depth_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
