"""Controlled warm-checkpoint continuation: EMA prediction updates only Q."""
import argparse
import copy
import json
import time
from pathlib import Path
import torch
from torch.func import functional_call
from recurrent_rg.checkpoints import load_run, checkpoint_payload, qcr_state_dict
from recurrent_rg.config import RunConfig
from recurrent_rg.runner import prepared_data, file_fingerprint, evaluate_run
from recurrent_rg.training import evaluate, separate_qc_r_gradients
from recurrent_rg.losses import qcr_objective, distribution_geometry_loss
from recurrent_rg.data.statistics import categorical_plaquette_distribution


def prediction_loss(model, teacher, batch):
    with torch.no_grad():levels=[z.detach() for z in model.coarse_grain(batch)]
    current=torch.stack([model.theory_encoder(z) for z in levels[:-1]],1)
    with torch.no_grad():target=torch.stack([teacher(z) for z in levels[1:]],1)
    state={k:v.detach() for k,v in model.theory_rg.named_parameters()}
    predicted=current+functional_call(model.theory_rg,state,(current,))
    return (predicted-target).square().sum(-1).mean()


def future_geometry_loss(model, batch, loss_config):
    """Cross-scale, data-targeted geometry of current and next distributions.

    Only levels with a successor are compared; no fabricated terminal target.
    Half splits estimate sampling noise. Neither R nor a moving Q teacher defines
    the target. Concatenation divided by sqrt(2) keeps mean squared distance units.
    """
    with torch.no_grad():
        levels = [z.detach() for z in model.coarse_grain(batch)]
        mid = batch.shape[1] // 2
        if mid < 2:
            raise ValueError('Need at least four configurations')
        pa = [categorical_plaquette_distribution(z[:, :mid]) for z in levels]
        pb = [categorical_plaquette_distribution(z[:, mid:]) for z in levels]
        a = torch.cat([torch.cat((pa[k], pa[k+1]), -1) for k in range(len(levels)-1)]) / 2**.5
        b = torch.cat([torch.cat((pb[k], pb[k+1]), -1) for k in range(len(levels)-1)]) / 2**.5
    g = torch.cat([.5*(model.theory_encoder(z[:, :mid]) + model.theory_encoder(z[:, mid:]))
                   for z in levels[:-1]])
    value, _ = distribution_geometry_loss(g, a, b, scale=loss_config.geometry_scale,
        noise_multiplier=loss_config.noise_multiplier, mode=loss_config.geometry_mode,
        relative_floor=loss_config.geometry_relative_floor,
        max_inverse_weight=loss_config.geometry_max_inverse_weight,
        neighbors=loss_config.geometry_neighbors, local_fraction=loss_config.geometry_local_fraction)
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--weight',type=float,default=1.);p.add_argument('--start',type=int,default=400)
    p.add_argument('--ramp',type=int,default=200);p.add_argument('--ema',type=float,default=.99)
    p.add_argument('--objective',choices=['prediction','future_geometry'],default='prediction')
    a=p.parse_args()
    if a.weight<0 or a.ramp<1 or not 0<=a.ema<1:raise ValueError('Invalid predictive objective settings')
    run=load_run(a.source,a.start);cfg=RunConfig.from_dict(run.config);m=run.model
    if not cfg.model.fixed_kernel:raise ValueError('This ablation requires fixed C')
    torch.set_num_threads(cfg.training.threads)
    tr,te=prepared_data(cfg);t=cfg.training;e=cfg.evaluation
    teacher=copy.deepcopy(m.theory_encoder).eval().requires_grad_(False)
    q=list(m.theory_encoder.parameters());r=list(m.theory_rg.parameters())
    opt=torch.optim.Adam(q+r,lr=t.learning_rate);opt.load_state_dict(run.payload['optimizer'])
    gen=torch.Generator();gen.set_state(run.payload['batch_generator_state']);torch.set_rng_state(run.payload['torch_rng_state'])
    contract=dict(training_driver='at_predictive_q',config=cfg.to_dict(),data_sha256=file_fingerprint(cfg.data.cache),
                  source=str(a.source),warm_checkpoint=a.start,prediction_weight=a.weight,ema=a.ema,ramp=a.ramp,
                  extra_objective=a.objective,
                  gradient='Q: representation + ramped extra objective; R: original closure; C fixed')
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'manifest.json').write_text(json.dumps(contract,indent=2)+'\n')
    history=list(run.payload['history']);started=time.perf_counter()
    train_eval=tr.configurations[torch.linspace(0,len(tr.configurations)-1,min(e.systems,len(tr.configurations))).long()]
    test_eval=te.configurations[torch.linspace(0,len(te.configurations)-1,min(e.systems,len(te.configurations))).long()]
    for step in range(a.start+1,t.epochs+1):
        m.train();ids=torch.randperm(len(tr.configurations),generator=gen)[:t.system_batch]
        samples=torch.randperm(tr.configurations.shape[1],generator=gen)[:t.sample_batch]
        batch=tr.configurations[ids][:,samples].double();opt.zero_grad(set_to_none=True)
        base,terms=qcr_objective(m,batch,pattern_fn=categorical_plaquette_distribution,**cfg.loss.objective_kwargs())
        closure=terms.get('closure_total',cfg.loss.weights.commutator*terms['commutator']+cfg.loss.rollout_weight*terms['rollout'])
        pred=(prediction_loss(m,teacher,batch) if a.objective=='prediction'
              else future_geometry_loss(m,batch,cfg.loss))
        weight=a.weight*min(1.,(step-a.start)/a.ramp)
        if not torch.isfinite(base+weight*pred):raise FloatingPointError('Nonfinite loss')
        separate_qc_r_gradients(base+weight*pred,closure,q,r);opt.step()
        with torch.no_grad():
            for target,source in zip(teacher.parameters(),m.theory_encoder.parameters()):target.lerp_(source,1-a.ema)
        if step%t.log_every==0 or step==t.epochs:
            row=dict(epoch=step,prediction_weight=weight)
            for name,x in [('train',train_eval),('test',test_eval)]:
                metrics=evaluate(m,x,categorical_plaquette_distribution,e.samples,spin_flip_augment=False,objective_kwargs=cfg.loss.objective_kwargs())
                with torch.no_grad():extra=float(prediction_loss(m,teacher,x[:,:e.samples].double()) if a.objective=='prediction'
                                                else future_geometry_loss(m,x[:,:e.samples].double(),cfg.loss))
                metrics.update({f'q_{a.objective}':extra,'optimized_total':metrics['total']+weight*extra})
                row.update({name+'_'+k:v for k,v in metrics.items()})
            history.append(row);(a.output/'history_progress.json').write_text(json.dumps(history,indent=2)+'\n')
            print(f'epoch {step}: test common={row["test_total"]:.6f}, Q {a.objective}={row[f"test_q_{a.objective}"]:.6f}',flush=True)
        if step%t.checkpoint_every==0:
            payload=checkpoint_payload(m,opt,gen,step,metadata=contract,history=history)
            payload['ema_Q']=teacher.state_dict();torch.save(payload,a.output/f'training_step_{step}.pt')
    torch.save(qcr_state_dict(m),a.output/'qcr_model.pt')
    (a.output/'summary.json').write_text(json.dumps(dict(config=cfg.to_dict(),history=history,elapsed_seconds=time.perf_counter()-started),indent=2)+'\n')
    evaluate_run(a.output,output=a.output/'report')


if __name__=='__main__':main()
