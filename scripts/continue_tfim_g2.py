"""Continue the original joint dim-2 TFIM run without changing its objective."""
import json
from pathlib import Path
import sys
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recurrent_rg.model_nd import RecurrentRGNetwork1D
from experiments.joint_qcr_ising import train, two_spin_distribution, evaluate


def main():
    source=Path('results/joint_qcr_tfim_1d_g2_coordsum_levels6_2000_seed91')
    out=Path('results/joint_qcr_tfim_1d_g2_coordsum_continue6000_seed91')
    out.mkdir(parents=True,exist_ok=True)
    summary=json.loads((source/'summary.json').read_text());cfg=summary['config']
    torch.set_num_threads(cfg['threads']);torch.manual_seed(cfg['seed'])
    model=RecurrentRGNetwork1D(levels=cfg['levels'],hidden=cfg['hidden'],theory_dim=2,
                               kernel_layout=cfg['kernel_layout']).double()
    model.theory_encoder.binary_lookup=model.rg_kernel.binary_lookup=True
    state=torch.load(source/'training_step_2000.pt',weights_only=True)
    model.load_state_dict(state['model'],strict=False)
    tx,_,vx,_=torch.load('data_cache/tfim1d_wolff_v2_sites128_M32_dtau0p5_eq2048_skip16_replicas3_ensemble128_seed91_nJ64.pt',weights_only=True)
    kwargs={k:cfg[k] for k in ['objective_mode','coordinate_reduction','geometry_scale','noise_multiplier','rollout_weight']}
    # Confirm resumed checkpoint reproduces the original evaluation before training.
    audit={}
    for name,x in [('train',tx),('test',vx)]:
        ids=torch.linspace(0,len(x)-1,min(cfg['eval_systems'],len(x))).long()
        audit[name]=evaluate(model,x[ids],two_spin_distribution,cfg['eval_samples'],objective_kwargs=kwargs)
        assert abs(audit[name]['total']-summary['history'][-1][name+'_total'])<1e-8
    (out/'resume_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print('Checkpoint evaluation verified; resuming Adam and RNG at step 2000',flush=True)
    history=train(model,tx,vx,pattern_fn=two_spin_distribution,epochs=6000,
        system_batch=cfg['system_batch'],sample_batch=cfg['sample_batch'],
        eval_samples=cfg['eval_samples'],eval_systems=cfg['eval_systems'],
        learning_rate=cfg['learning_rate'],seed=cfg['seed'],progress_dir=out,
        objective_kwargs=kwargs,resume_state=state,initial_history=summary['history'])
    for step in [4000,6000]:
        dest=out/f'step{step}';dest.mkdir(exist_ok=True)
        checkpoint=torch.load(out/f'training_step_{step}.pt',weights_only=True)
        torch.save(checkpoint['model'],dest/'qcr_model.pt')
        s=dict(summary);s['config']=dict(cfg,epochs=step,output_dir=str(dest))
        s['history']=[r for r in history if r['epoch']<=step]
        s['continuation']={'source':str(source),'restored_optimizer_and_rng':True,'start_step':2000,
                           'unchanged_loss_and_learning_rate':True}
        # Do not propagate stale original endpoint diagnostics.
        s={k:v for k,v in s.items() if k in ['config','history','training_contract','data','continuation']}
        (dest/'summary.json').write_text(json.dumps(s,indent=2)+'\n')
    print('DONE',out,flush=True)


if __name__=='__main__':main()
