"""Compare the same completed checkpoint, not unequal final training lengths."""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recurrent_rg.checkpoints import load_run

root=Path('results/spin1_head_ablation_150_seed91')
torch.set_num_threads(1)
fig,axs=plt.subplots(2,3,figsize=(14,7),layout='constrained')
models={};summary={}
for row,kind in enumerate(['nonlinear','linear']):
    loaded=load_run(root/kind,50);models[kind]=loaded.model
    p=root/kind/'full_report_50';m=json.loads((p/'metrics.json').read_text())
    lo,hi=np.array(m['fixed_points']['search_box']).ravel();g=np.linspace(lo,hi,500)
    with torch.no_grad():r=loaded.model.advance_theory(torch.tensor(g[:,None])).numpy().ravel()
    axs[row,0].plot(g,r);axs[row,0].plot(g,g,'k--');axs[row,0].set(xlabel='g',ylabel='R(g)',title=kind+' head, epoch 50')
    axs[row,1].plot(g,r-g);axs[row,1].axhline(0,color='k',ls='--');axs[row,1].set(xlabel='g',ylabel='R(g)-g')
    z=np.load(p/'encodings.npz');K=np.r_[z['train_labels'].ravel(),z['test_labels'].ravel()]
    direct=np.concatenate([z['train'],z['test']])[:,:,0];kn=np.unique(K)
    for k in range(3):axs[row,2].plot(kn,[direct[K==v,k].mean() for v in kn],label=f'k={k}')
    axs[row,2].set(xlabel='K',ylabel='Q(C^k D)');axs[row,2].legend()
    summary[kind]=dict(closure=m['closure'],fixed_points=m['fixed_points'],parameters=m['parameters'])
for ax in axs.flat:ax.grid(alpha=.2)
fig.savefig(root/'paired50.png',dpi=160);plt.close(fig)
a=models['nonlinear'].rg_kernel.state_dict();b=models['linear'].rg_kernel.state_dict()
summary['C_max_parameter_difference']=max(float((a[k]-b[k]).abs().max()) for k in a)
summary['note']='Matched epoch 50; nonlinear final 150, linear timed out after logged epoch 60. Defaults unchanged.'
(root/'paired50_metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained')
for kind in ['nonlinear','linear']:
    h=[r for r in json.loads((root/kind/'history_progress.json').read_text()) if r['epoch']<=50]
    for split,ls in [('train','-'),('test','--')]:
        for ax,key,w in zip(axs,['separation','commutator','rollout'],[10,5,5]):
            ax.plot([r['epoch'] for r in h],[w*r[split+'_'+key] for r in h],ls,marker='o',label=kind+' '+split)
            ax.set(xlabel='epoch',ylabel='weighted loss',title=key)
for ax in axs:ax.legend(fontsize=7);ax.grid(alpha=.2)
fig.suptitle('Matched early logs only (epochs 1 and 30); no invented epoch-50 evaluation')
fig.savefig(root/'paired_loss.png',dpi=150);plt.close(fig)
print(json.dumps(summary))
