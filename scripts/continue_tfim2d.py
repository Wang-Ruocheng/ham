"""Resume unchanged scalar QCR training on the existing 2D quantum TFIM data."""
import json
from pathlib import Path
import sys
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recurrent_rg.model_nd import RecurrentRGNetwork2D
from experiments.joint_qcr_ising import train, evaluate, plaquette_distribution
from experiments.qcr_scalar_report import scalar_report


def main():
    source=Path('results/joint_qcr_tfim2d_g1_L16_M640_seed91')
    out=Path('results/joint_qcr_tfim2d_g1_L16_M640_continue6000_seed91')
    out.mkdir(parents=True,exist_ok=True)
    summary=json.loads((source/'summary.json').read_text());cfg=summary['config']
    torch.set_num_threads(cfg['threads']);torch.manual_seed(cfg['seed'])
    model=RecurrentRGNetwork2D(levels=cfg['levels'],hidden=cfg['hidden'],theory_dim=1,
                               kernel_layout=cfg['kernel_layout']).double()
    model.theory_encoder.binary_lookup=model.rg_kernel.binary_lookup=True
    state=torch.load(source/'training_step_2000.pt',weights_only=True)
    model.load_state_dict(state['model'],strict=False)
    tx,_,vx,_=torch.load(cfg['data_cache'],weights_only=True)
    kwargs={k:cfg[k] for k in ['objective_mode','geometry_scale','noise_multiplier','rollout_weight']}
    audit={}
    for name,x in [('train',tx),('test',vx)]:
        ids=torch.linspace(0,len(x)-1,min(cfg['eval_systems'],len(x))).long()
        audit[name]=evaluate(model,x[ids],plaquette_distribution,cfg['eval_samples'],objective_kwargs=kwargs)
        assert abs(audit[name]['total']-summary['history'][-1][name+'_total'])<1e-8
    (out/'resume_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    torch.save(state,out/'training_step_2000.pt')
    print('Original 2D TFIM evaluation verified; resume optimizer and RNG at 2000',flush=True)
    history=train(model,tx,vx,pattern_fn=plaquette_distribution,epochs=6000,
        system_batch=cfg['system_batch'],sample_batch=cfg['sample_batch'],
        eval_samples=cfg['eval_samples'],eval_systems=cfg['eval_systems'],
        learning_rate=cfg['learning_rate'],seed=cfg['seed'],progress_dir=out,
        objective_kwargs=kwargs,resume_state=state,initial_history=summary['history'])
    torch.save({k:v for k,v in model.state_dict().items() if not k.startswith('action_network.')},out/'qcr_model.pt')
    # Record exact milestone evaluations in addition to regular progress logs.
    for step in [4000,6000]:
        checkpoint=torch.load(out/f'training_step_{step}.pt',weights_only=True)
        model.load_state_dict(checkpoint['model'],strict=False)
        row={'epoch':step}
        for name,x in [('train',tx),('test',vx)]:
            ids=torch.linspace(0,len(x)-1,min(cfg['eval_systems'],len(x))).long()
            vals=evaluate(model,x[ids],plaquette_distribution,cfg['eval_samples'],objective_kwargs=kwargs)
            row.update({name+'_'+k:v for k,v in vals.items()})
        history=[r for r in history if r['epoch']!=step]+[row]
    history.sort(key=lambda r:r['epoch'])
    result={k:summary[k] for k in ['training_contract','optimized_parameters','data']}
    result.update(config=dict(cfg,epochs=6000,output_dir=str(out)),history=history,
                  continuation=dict(source=str(source),start_step=2000,restored_optimizer_and_rng=True,
                                    unchanged_loss_structure_data_and_learning_rate=True))
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    scalar_report(out,cfg['data_cache'],dimension=2,checkpoint_steps=[2000,4000,6000])
    print('DONE',out,flush=True)


if __name__=='__main__':main()
