"""Fixed-dg1 fresh-encoder comparison: local patches, direct p, hierarchical CNN.

Explicit Q-only diagnostic; same frozen C trajectory, loss, fresh optimizer and
paired batches. No physical parameter/scale labels enter an encoder. Source QCR
3000 is a historical reference, not a compute-matched freshly trained baseline.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from scripts import run_phi4_hopping_qonly_balance as base
from scripts import run_phi4_hopping_qonly_nonlinear as previous

ARMS=('A_patch','B_statistics','C_hierarchical')
STEPS=300
GPU_SECONDS=1800.
CPU_SECONDS=7200.
MAX_CUDA_BYTES=8*1024**3
MAX_BYTES=1024**3


class EncoderQ(nn.Module):
    def __init__(self,decoder):
        super().__init__()
        self.decoder=copy.deepcopy(decoder)

    def decode(self,g):return self.decoder(g)

    @torch.no_grad()
    def normalize_decoder_(self):self.decoder.normalize_()


class PatchQ(EncoderQ):
    def __init__(self,encoder,decoder):
        super().__init__(decoder)
        self.encoder=encoder
        self.encoder.decoder=nn.Identity()

    def encode(self,fields,per_field_p):return self.encoder(fields)


class StatisticsQ(EncoderQ):
    def __init__(self,decoder):
        super().__init__(decoder)
        self.encoder=nn.Sequential(nn.Linear(625,64),nn.SiLU(),nn.Linear(64,16),nn.SiLU(),nn.Linear(16,1)).double()

    def encode(self,fields,per_field_p):
        # Only statistics from the supplied subset are used, never full-ensemble
        # target statistics or condition labels from another subset.
        return self.encoder(per_field_p.detach().mean(dim=1))


class HierarchicalQ(EncoderQ):
    def __init__(self,scale,decoder,frame_chunk=16):
        super().__init__(decoder)
        self.register_buffer('scale',torch.tensor(float(scale),dtype=torch.float64))
        self.frame_chunk=frame_chunk
        # Translation equivariance, 7x7 receptive field, no batch/instance norm.
        self.spatial=nn.Sequential(
            nn.Conv2d(1,16,3,padding=1,padding_mode='circular'),nn.SiLU(),
            nn.Conv2d(16,16,3,padding=2,dilation=2,groups=16,padding_mode='circular'),nn.SiLU(),
        ).double()
        self.frame_head=nn.Sequential(nn.Linear(32,32),nn.SiLU(),nn.Linear(32,16),nn.SiLU()).double()
        self.head=nn.Sequential(nn.Linear(32,64),nn.SiLU(),nn.Linear(64,1)).double()

    def frame_features(self,frames):
        features=self.spatial(frames/self.scale)
        pooled=torch.cat((features.mean((-2,-1)),features.square().mean((-2,-1))),dim=-1)
        return self.frame_head(pooled)

    def encode(self,fields,per_field_p):
        e,n,l,_,_=fields.shape
        frames=fields.detach().reshape(e*n,l,l,1).permute(0,3,1,2).contiguous()
        chunks=[]
        for part in frames.split(self.frame_chunk):
            chunks.append(checkpoint(self.frame_features,part,use_reentrant=False)
                if torch.is_grad_enabled() else self.frame_features(part))
        h=torch.cat(chunks).reshape(e,n,16)
        return self.head(torch.cat((h.mean(1),h.var(1,unbiased=False)),dim=-1))


def make_models(encoder_class,scale,target_train_mean):
    """All encoders fresh; one identical initial nonlinear decoder for all arms.

    Decoder intercept uses only the pooled TRAIN target mean. Direction is
    random unit norm, not PCA. Output heads have small random nonzero weights
    and zero bias; all parameter tensors receive gradients from the first step.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(91)
        affine=nn.Linear(1,625).double()
        with torch.no_grad():
            affine.weight.normal_();affine.weight.div_(affine.weight.norm())
            affine.bias.copy_(torch.as_tensor(target_train_mean,dtype=torch.float64).mean(0))
        decoder=previous.ResidualDecoder(affine)
        result={}
        for arm in ARMS:
            torch.manual_seed(91)
            if arm=='A_patch':
                q=PatchQ(encoder_class(scale,hidden=96,theory_dim=1,local_features=12,patch_chunk_size=262144),decoder)
                last=q.encoder.head[-1]
            elif arm=='B_statistics':q=StatisticsQ(decoder);last=q.encoder[-1]
            else:q=HierarchicalQ(scale,decoder);last=q.head[-1]
            with torch.no_grad():last.weight.normal_(0,.01);last.bias.zero_()
            result[arm]=q
    return result


def q_objective(q,levels,targets):
    rec=[];subset=[]
    if len(levels)!=4 or targets.shape[1]!=4:raise ValueError('Four scales required')
    for k,fields in enumerate(levels):
        half=fields.shape[1]//2
        if fields.shape[1]!=2*half or half<1:raise ValueError('Equal nonempty subsets required')
        p=targets[:,k].detach()
        first=q.encode(fields[:,:half].detach(),p[:,:half])
        second=q.encode(fields[:,half:].detach(),p[:,half:])
        g=.5*(first+second)
        rec.append((q.decode(g)-p.mean(1)).square().sum(-1).mean())
        subset.append((first-second).square().sum(-1).mean())
    rec=torch.stack(rec);subset=torch.stack(subset)
    return 10*rec.mean()+20*subset[0]+5*subset[1:].mean(),rec,subset


def update(q,optimizer,levels,targets):
    optimizer.zero_grad(set_to_none=True)
    loss,_,_=q_objective(q,levels,targets)
    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite Q loss')
    loss.backward()
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in q.parameters()):
        raise FloatingPointError('Missing/nonfinite Q gradient')
    optimizer.step();q.normalize_decoder_()
    if any(not bool(torch.isfinite(p).all()) for p in q.parameters()):raise FloatingPointError('Nonfinite Q parameter')
    optimizer.zero_grad(set_to_none=True)
    return float(loss.detach())


@torch.no_grad()
def probe(q,fields,targets,check):
    rows=[];sub=[];device=next(q.parameters()).device
    for i in range(len(targets)):
        _,r,s=q_objective(q,[x[i:i+1,:32].to(device) for x in fields],targets[i:i+1,:,:32].to(device))
        rows.append(r.cpu().numpy());sub.append(s.cpu().numpy());check()
    r=np.mean(rows,0);s=np.mean(sub,0)
    return dict(q_total=float(10*r.mean()+20*s[0]+5*s[1:].mean()),reconstruction_raw=float(r.mean()),
        reconstruction_by_scale=r.tolist(),subset_micro=float(s[0]),subset_coarse=float(s[1:].mean()),subset_by_scale=s.tolist())


@torch.no_grad()
def evaluate(q,fields,targets,check):
    records={key:[] for key in ('direct_g','direct_full_g','g_first','g_second','decoded')}
    device=next(q.parameters()).device
    for i in range(len(targets)):
        first=[];second=[];full=[]
        for k,x in enumerate(fields):
            f=x[i:i+1].to(device);p=targets[i:i+1,k].to(device)
            first.append(q.encode(f[:,:64],p[:,:64]).cpu().numpy()[0])
            second.append(q.encode(f[:,64:],p[:,64:]).cpu().numpy()[0])
            full.append(q.encode(f,p).cpu().numpy()[0]);check()
        a,b=np.asarray(first),np.asarray(second);g=.5*(a+b)
        for key,value in (('direct_g',g),('direct_full_g',np.asarray(full)),('g_first',a),('g_second',b),
                          ('decoded',q.decode(torch.as_tensor(g,device=device)).cpu().numpy())):
            records[key].append(value)
    return {key:np.asarray(value) for key,value in records.items()}


def decoder_diagnostics(q,values):
    device=next(q.parameters()).device
    x=torch.as_tensor(values,dtype=torch.float64,device=device).reshape(-1,1)
    with torch.no_grad():
        _,jac=torch.func.jvp(q.decode,(x,),(torch.ones_like(x),))
    norm=np.linalg.norm(jac.cpu().numpy(),axis=-1)
    bounds=q.decoder.bounds()
    if norm.max()>bounds['global_lipschitz_upper_bound']+1e-8:raise RuntimeError('Decoder gain bound failed')
    return jac.cpu().numpy(),dict(**bounds,observed_norm_min=float(norm.min()),observed_norm_max=float(norm.max()))


def run(output,gpu_index,gpu_uuid):
    wall=time.monotonic();output=Path(output).resolve()
    if output.exists() or any(r==output or r in output.parents for r in (base.SOURCE,base.SNAPSHOT,previous.CACHE_ROOT)):
        raise ValueError('Fresh independent output required')
    if os.uname().nodename!='thinklab-105-225' or os.getpriority(os.PRIO_PROCESS,0)<10:raise RuntimeError('Wrong host/priority')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=gpu_uuid or os.environ.get('CUDA_DEVICE_ORDER')!='PCI_BUS_ID' or torch.cuda.is_initialized():
        raise RuntimeError('Require pinned GPU mask before CUDA initialization')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    output.mkdir(parents=True)
    job=dict(status='running',workflow_complete=False,current='resource_admission',actual_steps=0,
        pid=os.getpid(),returncode=None,error=None,requires_user_action=False,C_updates=0,R_updates=0)
    def status(**kw):
        job.update(kw,elapsed_seconds=time.monotonic()-wall,cpu_seconds=time.process_time())
        for n in ('job.json','status.json'):base.write(output/n,job)
        print(json.dumps(job),flush=True)
    def check(reserve=0.,directory=False):
        if time.monotonic()-wall+reserve>GPU_SECONDS or time.process_time()+reserve>CPU_SECONDS:
            raise base.BudgetHold('Budget exhausted; no shortened/restarted experiment')
        if directory and sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>MAX_BYTES:raise base.BudgetHold('Directory budget')
        if torch.cuda.is_initialized() and torch.cuda.max_memory_allocated()>MAX_CUDA_BYTES:raise base.BudgetHold('CUDA memory budget')
    status()
    try:
        admission=[]
        while len(admission)<2:
            try:admission.append(base.gpu_snapshot(gpu_index,gpu_uuid))
            except base.BudgetHold:
                admission=[];status(current='waiting_resources');time.sleep(30);check();continue
            if len(admission)==1:time.sleep(10);check()
        status(current='source_cache_verification')
        state,formal,protected,manifest,source_summary=base.bind_sources()
        for name in ('run_phi4_hopping_qonly_encoders.py','report_phi4_hopping_qonly_encoders.py',
                     'run_phi4_hopping_qonly_nonlinear.py','run_phi4_hopping_qonly_balance.py','report_phi4_hopping_qonly_balance.py'):
            path=Path(__file__).resolve().with_name(name);protected[path]=base.sha(path)
        fields,targets,cache_receipt=previous.bind_cache(formal,protected)
        Model,_,_=base.snapshot_imports()
        source_model=Model(float(state['scale']),hidden=96,theory_dim=1,local_features=12,patch_chunk_size=262144,block_size=2,steps=3)
        source_model.load_state_dict(state,strict=True);source_model.requires_grad_(False)
        del state
        target=targets.mean(2).numpy();train=np.flatnonzero(formal['splits']=='train')
        mean=target[train].mean(0);std=np.sqrt(target[train].var(0).mean(-1))
        if len(train)!=40 or not (std>0).all():raise ValueError('Invalid TRAIN partition/statistics')
        initial=make_models(type(source_model.Q),float(source_model.scale),mean)
        decoder_pins={n:base.state_sha(q.decoder.state_dict()) for n,q in initial.items()}
        if len(set(decoder_pins.values()))!=1:raise RuntimeError('Initial decoder weights differ')
        device=torch.device('cuda:0')
        if torch.cuda.device_count()!=1:raise RuntimeError('Wrong CUDA mask')
        torch.cuda.set_per_process_memory_fraction(MAX_CUDA_BYTES/torch.cuda.get_device_properties(device).total_memory,device)
        torch.cuda.reset_peak_memory_stats(device)
        plan=dict(schema_version=5,experiment='dg1_fresh_encoder_comparison_v1',arms=list(ARMS),theory_dim=1,
            steps_per_arm=STEPS,source_reference_QCR_steps=3000,all_tested_encoders_fresh=True,
            architecture=dict(A='raw3x3 patch MLP hidden96/features12 + pooled scalar head',
                B='same-subset p mean ->625-64-16-1 SiLU',
                C='periodic Conv1-16 k3 + depthwise16 k3 dilation2; spatial mean/second moment; frame MLP32-32-16; ensemble mean/variance; head32-64-1'),
            decoder='identical random unit affine direction, pooled TRAIN p mean intercept, zero-final residual1-32-32-625 tanh, per-layer spectral cap2',
            output_initialization='small random final weight std.01 and zero bias, no source encoder transfer/PCA/pretraining',
            scale_normalization='one original frozen TRAIN field RMS, no per-configuration normalization',
            loss='10 equal mean scale SUM625 reconstruction +20 micro subset +5 mean coarse subset',
            batch=dict(physical_chains=8,fields_per_chain=32,disjoint_halves=16,paired_seed=192),
            optimizer=dict(name='fresh Adam',lr=.001),C_updates=0,R_updates=0,new_sampling=0,new_C_trajectories=0,
            historical_source_is_not_compute_matched=True,test_used_in_training=False,labels_used_in_training=False,
            trainable_parameters={n:sum(p.numel() for p in q.parameters()) for n,q in initial.items()},
            encoder_parameters={n:sum(p.numel() for p in q.parameters())-sum(p.numel() for p in q.decoder.parameters()) for n,q in initial.items()},
            source=str(base.SOURCE),cache=str(previous.CACHE_ROOT/'frozen_fields.pt'),cache_sha256=previous.CACHE_SHA,
            source_protected_sha256={str(p):s for p,s in protected.items()},initial_decoder_sha256=decoder_pins,
            physical_gpu_index=gpu_index,expected_gpu_uuid=gpu_uuid,admission=admission,
            budget=dict(gpu_reservation_seconds=GPU_SECONDS,cpu_seconds=CPU_SECONDS,directory_bytes=MAX_BYTES,own_cuda_peak_bytes=MAX_CUDA_BYTES),
            warnings=manifest['data_warning_count'],equilibrium_certified=False,thermal_nu=None,training_started=False)
        base.write(output/'plan.json',plan)
        base.write(output/'cache_receipt.json',dict(cache_sha256=previous.CACHE_SHA,reused_read_only=True,
            target_maxabs_vs_source=float(np.max(np.abs(target-formal['target_features']))),new_C_trajectories=0))
        train_fields=[x[train] for x in fields];train_targets=targets[train]
        def batch(indices,samples):return ([x[indices][:,samples].to(device) for x in train_fields],train_targets[indices][:,:,samples].to(device))
        status(current='bounded_benchmark')
        test_models={n:copy.deepcopy(q).to(device) for n,q in initial.items()}
        optim={n:torch.optim.Adam(q.parameters(),lr=.001) for n,q in test_models.items()}
        rng=torch.Generator().manual_seed(192);times=[];arm_times={n:[] for n in ARMS}
        for i in range(4):
            torch.cuda.synchronize();start=time.monotonic();levels,p=batch(*base.batch_schedule(40,128,generator=rng))
            for n,q in test_models.items():
                torch.cuda.synchronize();begin=time.monotonic();update(q,optim[n],levels,p);torch.cuda.synchronize()
                if i:arm_times[n].append(time.monotonic()-begin)
            if i:times.append(time.monotonic()-start)
            check()
        del test_models,optim,levels,p;torch.cuda.empty_cache()
        projection=1.5*max(times)*(STEPS+100)+150.
        base.write(output/'benchmark.json',dict(warmup_triples=1,timed_triples=3,triple_update_seconds=times,
            per_arm_update_seconds=arm_times,remaining_projection_seconds=projection,elapsed_seconds=time.monotonic()-wall,formal_updates=0))
        check(projection)
        models={n:q.to(device) for n,q in initial.items()}
        optim={n:torch.optim.Adam(q.parameters(),lr=.001) for n,q in models.items()}
        initial_pins={n:base.state_sha(q.state_dict()) for n,q in models.items()}
        history=[]
        def log(step):
            row=dict(step=step,arms={n:probe(q,train_fields,train_targets,check) for n,q in models.items()})
            history.append(row);base.write(output/'history.json',history);check(directory=True)
        log(0);plan['training_started']=True;base.write(output/'plan.json',plan)
        status(current='paired_Q_training',training_started=True)
        rng=torch.Generator().manual_seed(192);schedule=hashlib.sha256()
        for step in range(1,STEPS+1):
            idx,samples=base.batch_schedule(40,128,generator=rng)
            schedule.update(idx.numpy().tobytes());schedule.update(samples.numpy().tobytes())
            levels,p=batch(idx,samples)
            for n,q in models.items():update(q,optim[n],levels,p)
            del levels,p;check(150.)
            if step%50==0:log(step);status(actual_steps=step)
        status(current='fixed_endpoint_evaluation',actual_steps=STEPS)
        arrays=dict(target_features=target,train_target_mean=mean,train_target_std=std,
            source_direct_g=formal['direct_g'],source_direct_full_g=formal['direct_full_g'],source_decoded=formal['reconstructed_features'],
            source_flow_grid=formal['flow_grid'],source_flow_next=formal['flow_next'])
        info={}
        for n,q in models.items():
            values=evaluate(q,fields,targets,check)
            arrays.update({n+'_'+k:v for k,v in values.items()})
            jac,info[n]=decoder_diagnostics(q,values['direct_g']);arrays[n+'_decoder_jacobian']=jac.reshape(60,4,625)
            torch.save(dict(Q=q.state_dict(),architecture=plan['architecture'][n[0]],theory_dim=1,steps=STEPS),output/(n+'_Q_final.pt'))
        arrays.update({n:formal[n] for n in ('labels','splits','cell_ids')})
        if any(v.dtype.kind in 'fc' and not np.isfinite(v).all() for v in arrays.values()):raise FloatingPointError('Nonfinite evaluation')
        np.savez_compressed(output/'evaluation_arrays.npz',**arrays)
        metrics={s:{n:base.calculate_metrics(target[formal['splits']==s],arrays[n+'_decoded'][formal['splits']==s],std,mean)
            for n in ('source',*ARMS)} for s in ('train','test')}
        base.write(output/'metrics.json',metrics)
        counts={n:base.optimizer_counts(optim[n],q) for n,q in models.items()}
        if any(c['minimum']!=STEPS or c['maximum']!=STEPS for c in counts.values()):raise RuntimeError('Incorrect optimizer counters')
        def verify():
            if any(base.sha(p)!=s for p,s in protected.items()):raise RuntimeError('Protected source/cache/code changed')
            if any(base.state_sha(getattr(source_model,n).state_dict())!=source_summary['final_module_sha256'][n]
                   or any(p.grad is not None for p in getattr(source_model,n).parameters()) for n in ('Q','C','R')):
                raise RuntimeError('Source Q/C/R changed')
            if any(p.grad is not None for q in models.values() for p in q.parameters()):raise RuntimeError('Uncleared gradients')
        verify()
        summary=dict(status='complete',actual_steps=STEPS,theory_dim=1,all_tested_encoders_fresh=True,source_reference_QCR_steps=3000,
            optimizer_parameter_steps=counts,initial_Q_sha256=initial_pins,final_Q_sha256={n:base.state_sha(q.state_dict()) for n,q in models.items()},
            source_Q_C_R_unchanged=True,C_updates=0,R_updates=0,old_R_valid_for_new_Q=False,labels_used_in_training=False,
            test_used_in_training=False,model_selection=False,early_stopping=False,paired_batch_schedule_sha256=schedule.hexdigest(),
            decoder_diagnostics=info,warnings=manifest['data_warning_count'],equilibrium_certified=False,thermal_nu=None,
            elapsed_seconds_before_report=time.monotonic()-wall,own_peak_cuda_bytes=torch.cuda.max_memory_allocated())
        base.write(output/'summary.json',summary);status(current='array_only_report')
        report=importlib.import_module('scripts.report_phi4_hopping_qonly_encoders')
        public_report=report.render_report(output)
        verify();check(directory=True)
        base.write(output/'execution.json',dict(status='complete',returncode=0,source_cache_code_unchanged=True,
            elapsed_reservation_seconds=time.monotonic()-wall,cpu_seconds=time.process_time(),own_peak_cuda_bytes=torch.cuda.max_memory_allocated()))
        status(status='complete',workflow_complete=True,current='complete',returncode=0,error=None,requires_user_action=False)
        public=['plan.json','cache_receipt.json','benchmark.json','history.json','metrics.json','summary.json','execution.json',
                'job.json','status.json','evaluation_arrays.npz',*public_report]
        if any(Path(p).is_absolute() or '..' in Path(p).parts or Path(p).suffix not in ('.json','.png','.npz') for p in public):
            raise ValueError('Invalid public artifact path')
        base.write(output/'public_files.json',dict(files=[dict(path=p,bytes=(output/p).stat().st_size,sha256=base.sha(output/p)) for p in public]))
        return 0
    except Exception as exc:
        traceback.print_exc();status(status='budget_hold' if isinstance(exc,base.BudgetHold) else 'failed',workflow_complete=False,
            returncode=2,error=str(exc),requires_user_action=True)
        return 2


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='固定一维输出的三种 Q 编码器对照')
    parser.add_argument('--execute',action='store_true');parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu-index',type=int,default=3);parser.add_argument('--gpu-uuid',default='GPU-a167decd-ce49-6c8f-6bcf-136fde596683')
    args=parser.parse_args()
    if args.execute:raise SystemExit(run(args.output,args.gpu_index,args.gpu_uuid))
    print(json.dumps(dict(execution_requested=False,arms=ARMS,theory_dim=1,steps=STEPS,gpu_seconds=GPU_SECONDS,output=str(args.output))))
