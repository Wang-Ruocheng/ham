"""Paired from-scratch geometry ablations; preserve baseline and fixed C recipe."""
import concurrent.futures
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from recurrent_rg.config import RunConfig
from recurrent_rg.runner import file_fingerprint


def main():
    root=Path('results/spin1_geometry_ablation_seed91').resolve()
    root.mkdir(parents=True,exist_ok=False)
    baseline=Path('results/spin1_signedK_g1_block3_locked_recipe_h96_3000_seed91').resolve()
    source=RunConfig.load('configs/spin1_signedK_g1_block3_locked_recipe_h96_seed91.json').to_dict()
    variants={}
    for name in ['weight30','tail50_top10']:
        cfg=copy.deepcopy(source)
        if name=='weight30':cfg['loss']['weights']['separation']=30
        else:
            cfg['loss'].update(geometry_tail_mix=.5,geometry_tail_fraction=.1,geometry_tail_min_reliability=.5)
        variants[name]=cfg
        (root/f'{name}_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    old=json.loads((baseline/'manifest.json').read_text())
    assert file_fingerprint(source['data']['cache'])==old['data_sha256']
    assert source['model']==old['config']['model']
    assert json.loads(json.dumps(source['training']))==old['config']['training']
    protocol=dict(baseline=str(baseline),data_sha256=old['data_sha256'],
        seed=91,epochs=3000,from_scratch=True,parallel_runs=2,threads_per_run=2,
        changes=dict(weight30='Only geometry coefficient 10 -> 30',
                     tail50_top10='Geometry coefficient 10; 0.5 old loss + 0.5 reliability-weighted top 10% normalized residual; eligible iff lower>0 and reliability>=0.5; fallback to old loss if empty'),
        unchanged='Data, initialization seed, architecture, optimizer, C objective, independent gradients, R loss, update count',
        criterion='Final checkpoint, fixed-point structure, coupling interpolation, Q(C^k D) vs R rollout and near-input target conflicts; compare geometry using original base metric, not altered totals')
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONPATH='.')

    def run_one(name):
        run=root/name;started=time.time()
        commands=[
            [sys.executable,'-u','-m','recurrent_rg','train','--config',str(root/f'{name}_config.json'),'--output',str(run)],
            [sys.executable,'scripts/report_scalar_joint.py','--run',str(run)],
            [sys.executable,'scripts/report_mi_training.py','--run',str(run)],
            [sys.executable,'scripts/plot_dual_ising_scalar.py','--run',str(run),'--signed-spin1'],
            [sys.executable,'-u','scripts/audit_scalar_closure_conflicts.py','--run',str(run)]]
        state=dict(name=name,status='running',started=started)
        status=root/f'{name}_status.json'
        try:
            for stage,cmd in enumerate(commands):
                state.update(stage=stage,command=cmd);status.write_text(json.dumps(state,indent=2)+'\n')
                with (root/f'{name}_stage{stage}.log').open('w') as log:
                    subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            state.update(status='complete',elapsed_seconds=time.time()-started)
        except Exception as exc:
            state.update(status='failed',error=str(exc))
        status.write_text(json.dumps(state,indent=2)+'\n')
        return state

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        outcomes=list(pool.map(run_one,variants))
    (root/'outcomes.json').write_text(json.dumps(outcomes,indent=2)+'\n')
    if any(x['status']!='complete' for x in outcomes):raise RuntimeError('Ablation failed; inspect stage logs')
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from recurrent_rg.checkpoints import load_run
    torch.set_num_threads(2)
    fig,axes=plt.subplots(3,3,figsize=(17,13),layout='constrained');summaries={}
    for col,(label,path) in enumerate([('Baseline',baseline),('Geometry weight 30',root/'weight30'),('Tail mix 0.5',root/'tail50_top10')]):
        loaded=load_run(path);out=path/'full_report'
        metrics=json.loads((out/'metrics.json').read_text());summaries[label]={key:metrics[key] for key in ['fixed_points','closure']}
        lo,hi=np.array(metrics['fixed_points']['search_box']).ravel();grid=np.linspace(lo,hi,2000)
        with torch.no_grad():y=loaded.model.advance_theory(torch.from_numpy(grid[:,None])).numpy().ravel()
        ax=axes[0,col];ax.plot(grid,y-grid);ax.axhline(0,c='k',ls=':')
        for point in metrics['fixed_points']['roots']:ax.scatter(point['g'][0],0,c='green' if point['stability']=='stable' else 'orange')
        ax.set(title=label,xlabel='g',ylabel='R(g)-g')
        z=np.load(out/'encodings.npz');g=np.concatenate([z['train'],z['test']])[:,:,0];K=np.concatenate([z['train_labels'],z['test_labels']]).ravel()
        knots=np.unique(K);means=np.stack([g[K==k].mean(0) for k in knots])
        for k in range(3):axes[1,col].plot(knots,means[:,k],label=f'Q(C^{k}D)')
        for kc in [-1/1.693,1/1.693]:axes[1,col].axvline(kc,ls=':',c='grey')
        axes[1,col].set(xlabel='K',ylabel='g');axes[1,col].legend()
        h=json.loads((path/'history_progress.json').read_text())
        for split in ['train','test']:
            vals=[sum(r[f'{split}_geometry_base_level_{k}'] for k in range(3))/3 if f'{split}_geometry_base_level_0' in r else r[f'{split}_separation'] for r in h]
            axes[2,col].plot([r['epoch'] for r in h],vals,label=split)
        axes[2,col].set(xlabel='Epoch',ylabel='Original unweighted geometry metric',yscale='log');axes[2,col].legend()
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.savefig(root/'comparison.png',dpi=160);plt.close(fig)
    # Independent C gradients and matched RNG should keep C paired across runs.
    base_model=load_run(baseline).model
    for name in variants:
        other=load_run(root/name).model
        summaries[name+'_C_max_parameter_difference']=max(float((a-b).abs().max()) for a,b in zip(base_model.rg_kernel.parameters(),other.rg_kernel.parameters()))
    (root/'comparison.json').write_text(json.dumps(summaries,indent=2)+'\n')
    print(root/'comparison.png',flush=True)


if __name__=='__main__':main()
