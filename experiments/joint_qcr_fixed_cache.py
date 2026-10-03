"""Exact cached execution of the existing fixed-C field QCR experiment.

No parameter labels enter the objective. Random minibatch/sample order and
optimizer/checkpoint/evaluation conventions match joint_qcr_ising.train.
"""
import json
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments import joint_qcr_ising2d_field as entry
from recurrent_rg.losses import data_geometry_statistics_objective, qcr_objective
from recurrent_rg.data.statistics import plaquette_distribution
from recurrent_rg.config import QCRWeights
from experiments.supervised_q_ising2d_multiscale import patterns_for


def cached_objective(model,ha,hb,**kwargs):
    if any(True for _ in model.rg_kernel.parameters()):
        raise ValueError('Exact cached path requires a fixed decimation kernel')
    patterns=patterns_for(model.theory_encoder)
    values=model.theory_encoder.local_mlp(patterns)
    first=model.theory_encoder.head(ha@values)
    second=model.theory_encoder.head(hb@values)
    bits=(patterns>0).long()
    ids=8*bits[:,4]+4*bits[:,7]+2*bits[:,5]+bits[:,8]
    marginal=F.one_hot(ids,16).double()
    pa,pb=ha@marginal,hb@marginal
    m=(.5*(ha+hb)@patterns[:,4]).mean(0)
    ratio=kwargs.get('kernel_variance_ratio',.25)
    bound=(1-ratio*(1-m[:-1].square()).clamp(0,1)).sqrt()
    kernel=F.relu(m[1:].abs()-bound).square().mean()
    return data_geometry_statistics_objective(model,first,second,pa.unbind(1),pb.unbind(1),kernel,
             QCRWeights(anchor=2.,separation=10.),kwargs['geometry_scale'],kwargs['noise_multiplier'],kwargs['rollout_weight'],
             kwargs.get('coordinate_reduction','sum'))


def halves(counts,systems,samples):
    selected=np.asarray(counts[systems])
    n=len(samples)//2
    size=np.array([1024,256,64,16])[None,:,None]
    # Sum int16 counts in int64 before division; exact dyadic histogram fractions.
    return tuple(torch.from_numpy(selected[:,:,ids].sum(axis=2)/(n*size)) for ids in (samples[:n],samples[n:]))


def train_cached(model,train_x,test_x,*,pattern_fn,epochs,system_batch,sample_batch,
                 eval_samples,eval_systems,learning_rate,seed,progress_dir,
                 spin_flip_augment,objective_kwargs):
    assert not spin_flip_augment and objective_kwargs['kernel_noncollapse_mode']=='variance_floor'
    cache_dir=Path('results/supervised_Q_structure_uniform_KB_seed91')
    manifest=json.loads((cache_dir/'config.json').read_text())
    run=json.loads((progress_dir/'run_config.json').read_text())
    assert manifest['data']==run['config']['data_cache']
    train_c=np.load(cache_dir/'train_sample_counts.npy',mmap_mode='r')
    test_c=np.load(cache_dir/'test_sample_counts.npy',mmap_mode='r')
    assert len(train_c)==len(train_x) and len(test_c)==len(test_x)
    params=[p for mod in (model.theory_encoder,model.theory_rg) for p in mod.parameters()]
    # Independent direct-path checks at initialization, with the actual first random batch.
    check_gen=torch.Generator().manual_seed(seed)
    ids=torch.randperm(len(train_x),generator=check_gen)[:system_batch]
    samples=torch.randperm(train_x.shape[1],generator=check_gen)[:sample_batch]
    ha,hb=halves(train_c,ids.numpy(),samples.numpy())
    a,at=qcr_objective(model,train_x[ids][:,samples].double(),pattern_fn=pattern_fn,**objective_kwargs)
    b,bt=cached_objective(model,ha,hb,**objective_kwargs)
    torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-11)
    for key in at:torch.testing.assert_close(at[key],bt[key],rtol=1e-9,atol=1e-10)
    ga=torch.autograd.grad(a,params);gb=torch.autograd.grad(b,params)
    for u,v in zip(ga,gb):torch.testing.assert_close(u,v,rtol=1e-8,atol=1e-9)
    audit={'loss_error':abs(a.item()-b.item()),'max_gradient_error':max((u-v).abs().max().item() for u,v in zip(ga,gb)),
           'cache_directory':str(cache_dir.resolve()),'labels_used':False}
    (progress_dir/'cache_equivalence.json').write_text(json.dumps(audit,indent=2)+'\n')
    print('EXACT CACHE VERIFIED',audit,flush=True)
    del a,b,at,bt,ga,gb
    optimizer=torch.optim.Adam(params,lr=learning_rate)
    generator=torch.Generator().manual_seed(seed)
    train_ids=torch.linspace(0,len(train_x)-1,min(eval_systems,len(train_x))).long().numpy()
    test_ids=torch.linspace(0,len(test_x)-1,min(eval_systems,len(test_x))).long().numpy()
    eval_h=[halves(c,ids,np.arange(eval_samples)) for c,ids in [(train_c,train_ids),(test_c,test_ids)]]
    history=[]
    for epoch in range(epochs):
        model.train()
        ids=torch.randperm(len(train_x),generator=generator)[:system_batch]
        samples=torch.randperm(train_x.shape[1],generator=generator)[:sample_batch]
        ha,hb=halves(train_c,ids.numpy(),samples.numpy())
        optimizer.zero_grad(set_to_none=True)
        loss,_=cached_objective(model,ha,hb,**objective_kwargs)
        loss.backward();optimizer.step()
        state={k:v for k,v in model.state_dict().items() if not k.startswith('action_network.')}
        if (epoch+1)%500==0:
            torch.save({'epoch':epoch+1,'model':state,'optimizer':optimizer.state_dict(),
                        'batch_generator_state':generator.get_state(),'torch_rng_state':torch.get_rng_state()},
                       progress_dir/f'training_step_{epoch+1}.pt')
        if epoch==0 or (epoch+1)%max(1,epochs//50)==0 or epoch+1==epochs:
            model.eval();row={'epoch':epoch+1}
            with torch.no_grad():
                for split,(ha,hb) in zip(['train','test'],eval_h):
                    total,terms=cached_objective(model,ha,hb,**objective_kwargs)
                    row.update({split+'_total':float(total)}|{split+'_'+k:float(v) for k,v in terms.items()})
            history.append(row)
            (progress_dir/'history_progress.json').write_text(json.dumps(history,indent=2)+'\n')
            torch.save(state,progress_dir/'qcr_latest.pt')
            print(f"epoch {epoch+1}/{epochs}: train={row['train_total']:.6g} test={row['test_total']:.6g}",flush=True)
    return history


if __name__=='__main__':
    entry.train=train_cached
    entry.main()
