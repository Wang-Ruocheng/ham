"""Controlled cosine QC freeze: unchanged first400, fade400..1200, R-only thereafter."""
import argparse
import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import torch
from recurrent_rg.checkpoints import load_run,checkpoint_payload,qcr_state_dict
from recurrent_rg.config import RunConfig
from recurrent_rg.runner import prepared_data,file_fingerprint
from recurrent_rg.training import evaluate
from recurrent_rg.losses import qcr_objective
from recurrent_rg.data.statistics import categorical_plaquette_distribution


def qc_multiplier(step,start=400,end=1200):
    if end<=start:raise ValueError('End must exceed start')
    t=max(0.,min(1.,(step-start)/(end-start)))
    return .5*(1+math.cos(math.pi*t))


def split_adam(payload,n_qc):
    state=copy.deepcopy(payload['optimizer'])
    if len(state['param_groups'])!=1:raise ValueError('Expected shared single-group Adam')
    group=state['param_groups'][0];ids=group['params']
    if not 0<n_qc<len(ids):raise ValueError('Invalid parameter partition')
    state['param_groups']=[dict(copy.deepcopy(group),params=ids[:n_qc]),dict(copy.deepcopy(group),params=ids[n_qc:])]
    return state


def assign_gradients(total,closure,qc,r,gamma):
    """QC gets grad(L_QC + gamma L_R); R gets grad(L_R), with no resampling."""
    if qc and gamma>0:
        qg=torch.autograd.grad(total-(1-gamma)*closure,qc,retain_graph=True,allow_unused=True)
        for p,g in zip(qc,qg):p.grad=g
    rg=torch.autograd.grad(closure,r)
    for p,g in zip(r,rg):p.grad=g


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('results/potts2d_g1_neutral0_2400_seed91'))
    p.add_argument('--output',type=Path,default=Path('results/potts2d_g1_softQC400_1200_R2400_seed91'))
    args=p.parse_args();loaded=load_run(args.source,400);m=loaded.model
    cfg=RunConfig.from_dict(loaded.config);torch.set_num_threads(cfg.training.threads)
    if tuple(cfg.training.train_modules)!=('Q','C','R'):raise ValueError('Expected joint source')
    tx,vx=prepared_data(cfg);t=cfg.training;e=cfg.evaluation
    qc=list(m.theory_encoder.parameters())+list(m.rg_kernel.parameters());r=list(m.theory_rg.parameters())
    for p in qc+r:p.requires_grad_(True)
    opt=torch.optim.Adam([dict(params=qc),dict(params=r)],lr=t.learning_rate)
    opt.load_state_dict(split_adam(loaded.payload,len(qc)))
    generator=torch.Generator();generator.set_state(loaded.payload['batch_generator_state'])
    torch.set_rng_state(loaded.payload['torch_rng_state'])
    args.output.mkdir(parents=True,exist_ok=False)
    contract=dict(source_checkpoint=str(loaded.checkpoint),schedule='s(t)=0.5*(1+cos(pi*clip((t-400)/800,0,1)))',
        QC_learning_rate='0.001*s(t)',QC_gradient='grad(L_QC+s(t)*L_R)',
        R_learning_rate=.001,R_gradient='grad(L_R)',hard_frozen_after=1200,
        initial_updates='Exact original joint prefix1..400',optimizer='Retain original Adam moments/steps in two groups',
        labels_used_in_training=False,selection='Final2400; schedule fixed before this run',
        caveat='Controlled soft-freeze ablation, not a from-scratch QC-only warmup or R-weight ramp')
    metadata=dict(config=cfg.to_dict(),data_sha256=file_fingerprint(cfg.data.cache),intervention=contract)
    (args.output/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    history=copy.deepcopy(loaded.payload['history']);frozen=None
    torch.save(checkpoint_payload(m,opt,generator,400,metadata=metadata,history=history),args.output/'training_step_400.pt')
    tr=tx.configurations[torch.linspace(0,len(tx.configurations)-1,min(e.systems,len(tx.configurations))).long()]
    te=vx.configurations[torch.linspace(0,len(vx.configurations)-1,min(e.systems,len(vx.configurations))).long()]
    for step in range(401,2401):
        mult=qc_multiplier(step);opt.param_groups[0]['lr']=t.learning_rate*mult
        if mult==0 and frozen is None:
            for param in qc:param.requires_grad_(False)
            frozen={k:v.clone() for k,v in m.state_dict().items() if k.startswith(('theory_encoder.','rg_kernel.'))}
        m.train();systems=torch.randperm(len(tx.configurations),generator=generator)[:t.system_batch]
        samples=torch.randperm(tx.configurations.shape[1],generator=generator)[:t.sample_batch]
        batch=tx.configurations[systems][:,samples].double()
        opt.zero_grad(set_to_none=True)
        total,terms=qcr_objective(m,batch,pattern_fn=categorical_plaquette_distribution,**cfg.loss.objective_kwargs())
        closure=terms.get('closure_total',cfg.loss.weights.commutator*terms['commutator']+cfg.loss.rollout_weight*terms['rollout'])
        if not torch.isfinite(total):raise FloatingPointError(f'Nonfinite at {step}')
        assign_gradients(total,closure,qc,r,mult);opt.step()
        if step%100==0:
            metrics={s:evaluate(m,x,categorical_plaquette_distribution,e.samples,spin_flip_augment=False,
                               objective_kwargs=cfg.loss.objective_kwargs()) for s,x in [('train',tr),('test',te)]}
            row=dict(epoch=step,qc_multiplier=mult,qc_lr=opt.param_groups[0]['lr'],r_lr=opt.param_groups[1]['lr'])
            for split,values in metrics.items():row.update({f'{split}_{k}':v for k,v in values.items()})
            history.append(row)
            (args.output/'history_progress.json').write_text(json.dumps(history,indent=2)+'\n')
            print(f'epoch {step}/2400: QC multiplier={mult:.3f}, train={row["train_total"]:.6g} test={row["test_total"]:.6g}',flush=True)
        if step%400==0:
            torch.save(checkpoint_payload(m,opt,generator,step,metadata=metadata,history=history),args.output/f'training_step_{step}.pt')
    for key,value in frozen.items():assert torch.equal(value,m.state_dict()[key]),key
    torch.save(qcr_state_dict(m),args.output/'qcr_model.pt')
    torch.save(checkpoint_payload(m,opt,generator,2400,metadata=metadata,history=history),args.output/'checkpoint_last.pt')
    (args.output/'summary.json').write_text(json.dumps(dict(config=cfg.to_dict(),intervention=contract,
        QC_bitwise_frozen_after1200=True,history=history),indent=2)+'\n')
    print('Completed soft freeze; Q/C unchanged after1200',flush=True)


if __name__=='__main__':main()
