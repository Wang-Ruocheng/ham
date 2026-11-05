"""Freeze Q/C at update800; continue R with original Adam and RNG state."""
import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import torch
from recurrent_rg.checkpoints import load_run
from recurrent_rg.config import RunConfig
from recurrent_rg.runner import prepared_data, file_fingerprint
from recurrent_rg.training import train
from recurrent_rg.data.statistics import categorical_plaquette_distribution, two_spin_distribution, plaquette_distribution


def pattern_function(config):
    """Use the same configuration statistics as the standard training runner."""
    if config.model.categorical_states:
        return categorical_plaquette_distribution
    if config.model.dimension == 3:
        from recurrent_rg.data.statistics import cube_distribution
        return cube_distribution
    return two_spin_distribution if config.model.dimension == 1 else plaquette_distribution


def restrict_adam_state(payload, model):
    """Retain R moments/step from shared Q,C,R Adam, in the trainer's order."""
    result=copy.deepcopy(payload)
    old=result['optimizer']
    if len(old['param_groups'])!=1:raise ValueError('Expected single Adam parameter group')
    modules=(model.theory_encoder,model.rg_kernel,model.theory_rg)
    counts=[len(list(m.parameters())) for m in modules]
    ids=old['param_groups'][0]['params']
    if len(ids)!=sum(counts):raise ValueError('Source optimizer must contain Q,C,R in trainer order')
    retained=ids[sum(counts[:2]):]
    group=copy.deepcopy(old['param_groups'][0]);group['params']=list(range(len(retained)))
    result['optimizer']=dict(state={i:copy.deepcopy(old['state'][key]) for i,key in enumerate(retained) if key in old['state']},
                             param_groups=[group])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('results/potts2d_g1_neutral0_2400_seed91'))
    p.add_argument('--output',type=Path,default=Path('results/potts2d_g1_freezeQC800_R2400_seed91'))
    args=p.parse_args();loaded=load_run(args.source,800);cfg=RunConfig.from_dict(loaded.config)
    if tuple(cfg.training.train_modules)!=('Q','C','R'):raise ValueError('Requires joint source')
    cfg=replace(cfg,training=replace(cfg.training,epochs=2400,train_modules=('R',)))
    torch.set_num_threads(cfg.training.threads)
    tx,vx=prepared_data(cfg);state=restrict_adam_state(loaded.payload,loaded.model)
    args.output.mkdir(parents=True,exist_ok=False)
    metadata=dict(config=cfg.to_dict(),data_sha256=file_fingerprint(cfg.data.cache))
    contract=dict(source_checkpoint=str(loaded.checkpoint),freeze_after=800,final_update=2400,
        frozen_modules=['Q','C'],optimized_modules=['R'],optimizer='Retain R Adam moments and step; unchanged lr',
        sampling='Restore original batch and categorical RNG states; C remains stochastic with fixed weights',
        objective='Unchanged QCR objective and weights; only R-dependent terms contribute parameter gradients',
        labels_used_in_training=False)
    (args.output/'manifest.json').write_text(json.dumps(dict(**metadata,intervention=contract),indent=2)+'\n')
    state['metadata']=metadata
    torch.save(state,args.output/'training_step_800.pt')
    frozen={k:v.clone() for k,v in loaded.model.state_dict().items() if k.startswith(('theory_encoder.','rg_kernel.'))}
    history=train(loaded.model,tx.configurations,vx.configurations,pattern_fn=pattern_function(cfg),
        epochs=2400,system_batch=cfg.training.system_batch,sample_batch=cfg.training.sample_batch,
        eval_samples=cfg.evaluation.samples,eval_systems=cfg.evaluation.systems,
        learning_rate=cfg.training.learning_rate,seed=cfg.training.seed,
        progress_dir=args.output,spin_flip_augment=cfg.data.spin_flip_augment,objective_kwargs=cfg.loss.objective_kwargs(),
        resume_state=state,initial_history=state['history'],train_modules=('R',),
        checkpoint_every=400,checkpoint_metadata=metadata,record_every=100)
    for key,value in frozen.items():assert torch.equal(value,loaded.model.state_dict()[key]),key
    (args.output/'summary.json').write_text(json.dumps(dict(config=cfg.to_dict(),intervention=contract,
        frozen_tensors_bitwise_unchanged=True,history=history),indent=2)+'\n')
    print('Completed; Q/C bitwise unchanged',flush=True)


if __name__=='__main__':main()
