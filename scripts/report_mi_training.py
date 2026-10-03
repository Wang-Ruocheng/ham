"""Loss comparison for finite-cache MI experiments in either dimension."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recurrent_rg.checkpoints import load_run
from recurrent_rg.rsmi import kernel_probabilities,categorical_kernel_probabilities


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--baseline',type=Path);a=p.parse_args()
    out=a.run/'report';out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    records={}
    run_meta=json.loads((a.run/'manifest.json').read_text())
    run_label=('Independent C + MI init' if run_meta.get('data_driven_initialization')
               else 'Random C + independent gradients' if run_meta.get('independent_c')
               else 'Random C + QC/R gradients')
    paths=[(run_label,a.run)]
    if a.baseline:paths.insert(0,('Baseline',a.baseline))
    for label,path in paths:
        meta=json.loads((path/'manifest.json').read_text());cfg=meta['config'];lc=cfg['loss'];w=lc['weights']
        history=json.loads((path/'history_progress.json').read_text());x=[r['epoch'] for r in history]
        for split,style in [('train','-'),('test','--')]:
            def vals(key):return np.array([r[f'{split}_{key}'] for r in history])
            qcr=np.array([r.get(f'{split}_qcr_total',r[f'{split}_total']) for r in history])
            color_weight=lc.get('kernel_sign_weight',0.)
            color=color_weight*vals('kernel_sign') if color_weight else np.zeros(len(history))
            axes[0,0].plot(x,qcr-w['kernel_noncollapse']*vals('kernel_noncollapse')-color,style,label=f'{label}: {split}')
            axes[0,1].plot(x,w['commutator']*vals('commutator')+lc['rollout_weight']*vals('rollout'),style,label=f'{label}: {split}')
            if path==a.run:
                axes[1,0].plot(x,w['kernel_noncollapse']*vals('kernel_noncollapse'),style,label=split)
                if color_weight:axes[1,0].plot(x,color,style,label=f'{split}: color NLL')
                axes[1,1].plot(x,vals('rsmi_bits'),style,label=split)
        last=history[-1]
        records[label]={'test_QCR':last.get('test_qcr_total',last['test_total']),
            'weighted_test_terms':{k:weight*last['test_'+k] for k,weight in dict(anchor=w['anchor'],coarse_consistency=w['coarse_consistency'],separation=w['separation'],kernel_noncollapse=w['kernel_noncollapse'],commutator=w['commutator'],rollout=lc['rollout_weight']).items()},
            'estimator':meta.get('estimator','not logged'),'last_MI_bits':last.get('test_rsmi_bits')}
        if lc.get('kernel_sign_weight',0.):records[label]['weighted_test_terms']['kernel_sign']=lc['kernel_sign_weight']*last['test_kernel_sign']
    titles=['Common QCR loss (C and MI excluded)','Weighted one-step + rollout','Weighted C regularizers','New run: exact empirical MI, bits']
    for ax,title in zip(axes.flat,titles):ax.set(title=title,xlabel='Training update');ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0,0].set_yscale('log');axes[0,1].set_yscale('log')
    fig.savefig(out/'loss_comparison.png',dpi=160)
    loaded=load_run(a.run);model=loaded.model;width=2**loaded.config['model']['dimension']
    categorical=loaded.config['model'].get('categorical_states',0);q=categorical or 2
    codes=(torch.arange(q**width)[:,None]//(q**torch.arange(width)))%q
    patterns=(torch.nn.functional.one_hot(codes,q).flatten(1).double() if categorical else (2*codes-1).double())[None]
    with torch.no_grad():table=(categorical_kernel_probabilities if categorical else kernel_probabilities)(model.rg_kernel,patterns)[0]
    records['kernel']={'patterns':codes.tolist(),'probabilities':table.tolist()}
    if not categorical:records['kernel']['flip_equivariance_max_error']=float((table[:,1]+table.flip(0)[:,1]-1).abs().max())
    (out/'training_comparison.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
