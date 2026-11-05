"""单因素 b2 连续场检验；原数据、独立梯度、固定3000步，不改变默认配方。"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from experiments.phi4_b2_closure_comparison import (
    load_dataset, initialize_decoder, independent_step, evaluate,
)
from experiments.phi4_qcr import (
    _device_receipt, sha256_file, state_fingerprint, module_fingerprints,
    optimizer_step_counts, write_json,
)
from recurrent_rg.phi4_b2_qcr import sample_regions_b2
from recurrent_rg.phi4_b2_validation import ValidationModel
from recurrent_rg.tensor_io import load_cpu_tensors
from scripts.report_phi4_b2_closure_comparison import load_public_comparison


RECIPE = "phi4_b2_single_factor_validation_v1"
JOINT_ARMS = ("cubic_decoder", "odd_C")
FORMAL_STEPS = 3000


class BudgetHold(RuntimeError):
    pass


@dataclass
class Options:
    source: str
    output: str
    stage: str = "joint"
    device: str = "cuda:2"
    expected_gpu_uuid: str | None = None
    cpu_budget_seconds: float = 7200.
    gpu_budget_seconds: float = 3600.
    max_cuda_bytes: int = 8*1024**3
    steps: int = FORMAL_STEPS
    seed: int = 91
    system_batch: int = 8
    sample_batch: int = 64
    information_pairs: int = 128

    def validate(self):
        if self.stage not in ("profile", "joint", "r_tail"):
            raise ValueError("Unknown validation stage")
        if self.stage=="profile" and (self.steps!=20 or self.seed!=191):
            raise ValueError("Execution-only profile pins twenty updates and independent seed191")
        if self.steps<1 or self.system_batch<1 or self.sample_batch<4 or self.sample_batch%2:
            raise ValueError("Invalid training dimensions")
        if self.cpu_budget_seconds<=0 or self.gpu_budget_seconds<=0 or self.max_cuda_bytes<=0:
            raise ValueError("Positive fixed budgets required")
        if self.device not in ("cpu","cuda:2"):
            raise ValueError("Explicit CPU testing or fixed physical cuda:2 only")
        if self.device=="cuda:2" and not self.expected_gpu_uuid:
            raise ValueError("Actual physical GPU UUID must be pinned")
        if Path(self.output).exists():
            raise FileExistsError("Never overwrite, resume or retry this output")


def common_fingerprints(model):
    """Ignore only the two new zero-initialized cubic coefficients."""
    class CommonQ:
        def state_dict(self):
            return {k:v for k,v in model.Q.state_dict().items()
                    if k not in ("quadratic_weight","cubic_weight")}
    result=module_fingerprints(model)
    result["Q"]=state_fingerprint(CommonQ())
    return result


def _sha_files(paths):
    return {str(p):sha256_file(p) for p in paths}


def source_context(source):
    source=Path(source).resolve()
    training=source/"training" if (source/"training/manifest.json").exists() else source
    public=load_public_comparison(training)
    if public.summary["actual_steps"]!=3000 or public.manifest["lattice_sizes"]!=[32,16,8,4]:
        raise ValueError("Expected the completed 3000-step source experiment")
    checkpoint=training/"all_start/training_step_3000.pt"
    pins=_sha_files([training/n for n in public.fingerprints]+[checkpoint])
    data_root=Path(public.manifest["data_root"]).resolve()
    dataset=load_dataset(data_root)
    if (dataset.manifest_sha256!=public.manifest["data_manifest_sha256"]
            or dataset.receipts!=public.manifest["data_shards"]):
        raise ValueError("Physical dataset differs from original source")
    pins[str(data_root/"manifest.json")]=dataset.manifest_sha256
    for row in dataset.receipts:
        for path,digest in (("path","sha256"),("sidecar","sidecar_sha256")):
            pins[str(data_root/row[path])]=row[digest]
    return training,public,dataset,pins,checkpoint


def fresh_joint(dataset,device,seed,public,budget,verify_source=True):
    """One shared baseline TRAIN-bias initialization; only the requested factor changes."""
    torch.manual_seed(seed)
    base=ValidationModel(dataset.scale,variant="baseline").to(device)
    constructor=common_fingerprints(base)
    init=initialize_decoder(base,dataset,device,seed,budget)
    initial=common_fingerprints(base)
    if verify_source and (constructor!=public.manifest["random_construction_module_sha256"]
            or initial!=public.manifest["initial_module_sha256"]):
        raise RuntimeError("Fresh common initialization does not reproduce the pinned baseline")
    models={}
    for variant in JOINT_ARMS:
        with torch.random.fork_rng(devices=[]):
            model=ValidationModel(dataset.scale,variant=variant).to(device)
        incompatible=model.load_state_dict(base.state_dict(),strict=False)
        missing=set(incompatible.missing_keys)
        expected={"Q.quadratic_weight","Q.cubic_weight"} if variant=="cubic_decoder" else set()
        if missing!=expected or incompatible.unexpected_keys:
            raise RuntimeError("Single-factor model cannot load common random state")
        if common_fingerprints(model)!=initial:
            raise RuntimeError("Initial common parameter tensors differ between factors")
        models[variant]=model
    return models,init,initial


def frozen_tail(dataset,device,checkpoint,public):
    payload=load_cpu_tensors(checkpoint)
    if payload["step"]!=3000 or payload["closure_objective"]!="all_start_anchor":
        raise ValueError("Wrong source checkpoint")
    with torch.random.fork_rng(devices=[]):
        model=ValidationModel(dataset.scale,variant="baseline").to(device)
    model.load_state_dict(payload["model"],strict=True)
    if module_fingerprints(model)!=public.summary["arms"]["all_start"]["final_module_sha256"]:
        raise RuntimeError("Source endpoint tensor fingerprints differ")
    model.Q.requires_grad_(False);model.C.requires_grad_(False)
    model.Q.eval();model.C.eval()
    return model


def r_tail_objective(model,batch,generator,information_pairs=128):
    """Same stochastic physical estimator; no Q/C objective or MI evaluation."""
    with torch.no_grad():
        levels=model.coarse_grain(batch,generator)
        half=batch.shape[1]//2
        direct=[.5*(model.Q(model.features(level[:,:half]))+model.Q(model.features(level[:,half:])))
                for level in levels]
        # Consume exactly the original region-index RNG draws, without MI.
        sample_regions_b2(batch,information_pairs,generator)
    errors={}
    for s in range(3):
        prediction=direct[s]
        for t in range(s+1,4):
            prediction=model.advance(prediction)
            errors[s,t]=(prediction-direct[t]).square().sum(-1).mean()
    one=torch.stack([errors[k,k+1] for k in range(3)]).mean()
    rollout=torch.stack([errors[0,t] for t in range(1,4)]).mean()
    loss=10*torch.stack([torch.stack([errors[s,t] for s in range(t)]).mean() for t in range(1,4)]).mean()
    terms=dict(r_loss=float(loss.detach()),closure_one_step=float(one.detach()),
        closure_rollout=float(rollout.detach()),closure_all_start_anchor=float(loss.detach()))
    for (s,t),error in errors.items():terms[f"closure_error_{s}_{t}"]=float(error.detach())
    return loss,terms


def tail_step(model,batch,optimizer,generator,information_pairs):
    loss,metrics=r_tail_objective(model,batch,generator,information_pairs)
    if not torch.isfinite(loss):raise FloatingPointError("Nonfinite R-only loss")
    optimizer.zero_grad(set_to_none=True);loss.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.R.parameters()):
        raise FloatingPointError("Nonfinite/missing R-only gradient")
    if any(p.grad is not None for module in (model.Q,model.C) for p in module.parameters()):
        raise RuntimeError("R-only gradients leaked into frozen Q/C")
    optimizer.step()
    return metrics


def parameter_counts(model,optimizers,steps):
    active=optimizer_step_counts(optimizers,steps)
    return {key:active[key] if key in active else dict(min=0,max=0,parameters=len(list(getattr(model,key).parameters())))
            for key in ("Q","C","R")}


def optimize(models,dataset,device,options,output,budget,status,profile=False):
    initial={name:module_fingerprints(m) for name,m in models.items()}
    optimizers={name:({"R":torch.optim.Adam(m.R.parameters(),lr=.001)} if name=="r_tail" else
        {key:torch.optim.Adam(getattr(m,key).parameters(),lr=.001) for key in ("Q","C","R")})
        for name,m in models.items()}
    rng={name:torch.Generator(device=device).manual_seed(options.seed+202) for name in models}
    batch_rng=torch.Generator(device="cpu").manual_seed(options.seed+101)
    train_indices=torch.as_tensor(dataset.train_indices)
    history=[];schedule=hashlib.sha256();start=time.perf_counter();latest=0
    for step in range(1,options.steps+1):
        budget()
        systems=train_indices[torch.randperm(len(train_indices),generator=batch_rng)[:options.system_batch]]
        samples=torch.randperm(dataset.configurations.shape[1],generator=batch_rng)[:options.sample_batch]
        schedule.update(systems.numpy().tobytes());schedule.update(samples.numpy().tobytes())
        batch=dataset.configurations[systems][:,samples].to(device)
        metrics={}
        for name,model in models.items():
            model.train()
            if name=="r_tail":
                model.Q.eval();model.C.eval()
                metrics[name]=tail_step(model,batch,optimizers[name]["R"],rng[name],options.information_pairs)
            else:
                metrics[name]=independent_step(model,batch,optimizers[name],rng[name],"all_start_anchor",options.information_pairs)
        if step==1 or step%100==0 or step==options.steps:
            row=dict(step=step,arms=metrics,elapsed_seconds=time.perf_counter()-start)
            history.append(row)
            write_json(output/"history_progress.json",dict(recipe=RECIPE,history=history,
                latest_step=step,latest_checkpoint_step=latest))
            print(__import__("json").dumps(dict(event="single_factor_progress",**row)),flush=True)
            status("training",step=step,latest=latest)
        if not profile and (step%1000==0 or step==options.steps):
            for name,model in models.items():
                arm=output/name;arm.mkdir(exist_ok=True)
                torch.save(dict(recipe=RECIPE,stage=options.stage,step=step,variant=name,
                    model=model.state_dict(),optimizers={k:o.state_dict() for k,o in optimizers[name].items()}),
                    arm/f"training_step_{step}.pt")
            latest=step
        status("training",step=step,latest=latest,persist=False)
    budget();elapsed=time.perf_counter()-start
    counts={name:parameter_counts(models[name],optimizers[name],options.steps) for name in models}
    for name,model in models.items():
        if name=="r_tail" and any(module_fingerprints(model)[k]!=initial[name][k] for k in ("Q","C")):
            raise RuntimeError("Frozen Q/C tensors changed during tail")
        for p in model.parameters():p.grad=None
    write_json(output/"history_progress.json",dict(recipe=RECIPE,history=history,
        latest_step=options.steps,latest_checkpoint_step=latest))
    return history,counts,initial,schedule.hexdigest(),elapsed,latest


@torch.no_grad()
def evaluate_validation(model,dataset,device,budget):
    """Augment the existing evaluation using only fields it already produces.

    A temporary C hook observes its three calls per physical chain; it never
    makes another C/Q/R call or consumes a random draw. Microscopic fields
    come directly from that same dataset chain. Each diagnostic is retained
    separately for every chain and all four scales. Mirror L1 is
    ``sum(abs(p-p.flip(0)))`` (not half the L1): p is the chain-pooled five-bin
    soft distribution with centers [-2,-1,0,1,2], bandwidth .5, expressed in
    the single original TRAIN RMS. No amplitude calibration is introduced.
    """
    scale=float(model.scale.detach().cpu())
    observed=[]

    def statistics(field):
        values=field.detach().reshape(-1)
        centers=torch.arange(-2,3,dtype=values.dtype,device=values.device)
        probability=torch.softmax(-.5*((values[:,None]/scale-centers)/.5).square(),dim=-1).mean(0)
        return torch.stack((values.mean(),values.square().mean().sqrt(),
                            (probability-probability.flip(0)).abs().sum())).cpu().numpy()

    def observe(module,args,result):
        if result.ndim!=5 or result.shape[0]!=1 or result.shape[-1]!=1:
            raise RuntimeError("Expected one physical chain per existing C evaluation call")
        observed.append(statistics(result))

    handle=model.C.register_forward_hook(observe)
    try:
        arrays=evaluate(model,dataset,device,budget)
    finally:
        handle.remove()
    expected=3*len(dataset.configurations)
    if len(observed)!=expected:
        raise RuntimeError(f"Expected exactly {expected} existing C calls, observed {len(observed)}")
    coarse=np.stack(observed).reshape(len(dataset.configurations),3,3)
    microscopic=np.stack([statistics(chain) for chain in dataset.configurations])[:,None,:]
    diagnostics=np.concatenate((microscopic,coarse),axis=1)
    for split in ("train","test"):
        mask=dataset.splits==split
        for column,name in enumerate(("field_mean","field_rms","field_mirror_l1")):
            arrays[f"{split}_{name}"]=diagnostics[mask,:,column].copy()
    return arrays


def save_flow(model,arrays,device):
    values=np.concatenate([arrays[f"{s}_direct"][:,:-1].ravel() for s in ("train","test")])
    low,high=float(values.min()),float(values.max());pad=.1*max(high-low,1e-6)
    grid=np.linspace(low-pad,high+pad,256)
    with torch.no_grad():mapped=model.advance(torch.as_tensor(grid[:,None],dtype=torch.float64,device=device)).cpu().numpy()
    arrays.update(flow_g=grid[:,None],flow_R=mapped,flow_displacement=mapped-grid[:,None])


def run(options):
    options.validate();output=Path(options.output).resolve();source=Path(options.source).resolve()
    if output==source or source in output.parents or output in source.parents:
        raise ValueError("Output must be independent of source")
    output.mkdir(parents=True,exist_ok=False)
    wall_start=time.perf_counter();cpu_start=time.process_time();gpu_start=None;peak=0;actual=0;latest=0
    device=torch.device(options.device)
    def budget():
        nonlocal peak
        if time.process_time()-cpu_start>options.cpu_budget_seconds:raise BudgetHold("CPU budget reached; no retry")
        if device.type=="cuda" and gpu_start is not None:
            torch.cuda.synchronize(device)
            peak=max(peak,int(torch.cuda.max_memory_allocated(device)),int(torch.cuda.max_memory_reserved(device)))
            if peak>options.max_cuda_bytes or time.perf_counter()-gpu_start>options.gpu_budget_seconds:
                raise BudgetHold("GPU time/memory budget reached; no migration or shortening")
    def status(value,step=None,checkpoint_step=None,persist=True,**extra):
        nonlocal actual,latest
        if "latest" in extra:
            checkpoint_step=extra.pop("latest")
        if step is not None:actual=step
        if checkpoint_step is not None:latest=checkpoint_step
        if not persist:return
        write_json(output/"status.json",dict(recipe=RECIPE,stage=options.stage,status=value,
            workflow_complete=value=="complete",returncode=0 if value=="complete" else None,
            actual_steps=actual,latest_checkpoint_step=latest,
            error=None,requires_user_action=False,**extra))
    try:
        torch.set_num_threads(1);status("validating_source")
        training,public,dataset,pins,checkpoint=source_context(source)
        if options.system_batch>len(dataset.train_indices) or options.sample_batch>dataset.configurations.shape[1]:
            raise ValueError("Requested batch exceeds available TRAIN data")
        budget();gpu_start=time.perf_counter() if device.type=="cuda" else None
        receipt=_device_receipt(device,options.expected_gpu_uuid)
        if device.type=="cuda":torch.cuda.reset_peak_memory_stats(device)
        setup=time.perf_counter();q_init=None;common_initial={}
        if options.stage in ("profile","joint"):
            models,q_init,shared=fresh_joint(dataset,device,options.seed,public,budget,options.stage!="profile")
            common_initial={name:common_fingerprints(m) for name,m in models.items()}
        else:
            models={"r_tail":frozen_tail(dataset,device,checkpoint,public)}
            common_initial={"r_tail":common_fingerprints(models["r_tail"])}
        initialization=time.perf_counter()-setup
        if options.stage=="profile":
            _,_,_,_,joint_time,_=optimize(models,dataset,device,options,output,budget,status,profile=True)
            del models
            if device.type=="cuda":torch.cuda.empty_cache()
            setup=time.perf_counter();tail=frozen_tail(dataset,device,checkpoint,public)
            initialization+=time.perf_counter()-setup
            _,_,_,_,tail_time,_=optimize({"r_tail":tail},dataset,device,options,output,budget,status,profile=True)
            budget()
            if _sha_files([Path(p) for p in pins])!=pins:raise RuntimeError("Source changed during execution-only profile")
            profile=dict(recipe=RECIPE,status="complete",workflow_complete=True,returncode=0,
                profile_only=True,steps_per_arm=20,seed=191,initialization_seconds=initialization,
                joint_update_seconds=joint_time,r_tail_update_seconds=tail_time,
                gpu_uuid=receipt.get("uuid"),device=receipt,peak_cuda_bytes=peak,
                formal_checkpoint_reused=False,profile_endpoint_reused_for_formal=False,
                source_endpoint_loaded_for_r_tail_profile=True,
                checkpoint_reuse_scope="No profile-trained endpoint is reused for formal training; tail profile loads the immutable source endpoint",
                scientific_result=False,source_unchanged=True,
                cpu_seconds=time.process_time()-cpu_start,wall_seconds=time.perf_counter()-wall_start)
            write_json(output/"profile.json",profile);status("complete",profile_only=True,profile_sha256=sha256_file(output/"profile.json"))
            return profile
        manifest=dict(recipe=RECIPE,schema_version=5,stage=options.stage,options=asdict(options),
            source_root=str(source),source_public_sha256=public.fingerprints,source_checkpoint_sha256=sha256_file(checkpoint),
            data_manifest_sha256=dataset.manifest_sha256,source_inputs_sha256=pins,
            lattice_sizes=[32,16,8,4],theory_dim=1,block_size=2,stride=2,dtype="float64",
            initial_common_module_sha256=common_initial,initial_module_sha256={n:module_fingerprints(m) for n,m in models.items()},
            arms={n:dict(variant=n,initial_common_module_sha256=common_initial[n],
                initial_module_sha256=module_fingerprints(m),architecture={**m.validation_receipt(),
                    **(dict(source_checkpoint_reused=True,same_seed_common_initial_parameters=False,
                        frozen_source_Q_C=True) if n=="r_tail" else {})}) for n,m in models.items()},
            q_initialization_receipt=q_init,shared_bias_calibration="same baseline random C TRAIN mean bias for both joint arms",
            labels_used_in_training=False,gradient_routing="Q representation only; C own objective; R all-start closure only",
            closure_objective="all_start_anchor",closure_scale_weights=None,closure_weights=dict(commutator=5,rollout_weight=5,total_all_start=10),
            learning_rate=.001,optimizer="fresh Adam; source optimizer state never reused",source_steps=3000,
            requested_additional_steps=options.steps,default_recipe_modified=False,
            Q_C_frozen=options.stage=="r_tail",stochastic_raw_physical_batches=True,fixed_encoding_cache_used=False,
            tail_MI_computed=False,schedule_seed=options.seed+101,conditional_generator_seed=options.seed+202,
            tail_region_indices_consumed_for_common_RNG_schedule=True,data_warning_count=dataset.warning_count,
            sampler_qualification=dataset.qualification,root_search_performed=False,critical_r=None,thermal_nu=None,
            limitations=["单因素历史 b2 消融，不改变默认 b3 或共享仿射配方",
                "cubic/odd_C 是新的完整QCR训练；R-tail仅更新R，三者均非监督学习",
                "旧baseline复用前次正式端点，必须精确核对共同初始化、物理批次和C随机流",
                "odd_C改变实际粗粒化输出分布，不可把其统计目标误差与原C的目标混称同一物理量",
                "不同Q的g坐标不作跨臂原始误差或数值根位置排名；不认证临界指数",
                "原采样警告保留，不认证平衡；不增加采样或TEST选模"])
        write_json(output/"manifest.json",manifest)
        history,counts,initial,schedule,update_time,latest=optimize(models,dataset,device,options,output,budget,status)
        actual=options.steps
        if options.steps==3000 and schedule!=public.manifest["paired_schedule_sha256"]:
            raise RuntimeError("Physical minibatch schedule differs from matched baseline")
        source_final=public.summary["arms"]["all_start"]["final_module_sha256"]
        if options.stage=="joint" and module_fingerprints(models["cubic_decoder"])["C"]!=source_final["C"]:
            raise RuntimeError("Cubic decoder changed independently trained C relative to baseline")
        if options.stage=="r_tail" and any(module_fingerprints(models["r_tail"])[k]!=source_final[k] for k in ("Q","C")):
            raise RuntimeError("R-only tail changed source Q/C")
        evaluation_start=time.perf_counter();status("evaluating",step=actual,latest=latest)
        arrays={};arms={}
        for name,model in models.items():
            model.eval().requires_grad_(False)
            before=state_fingerprint(model)
            local=evaluate_validation(model,dataset,device,budget);save_flow(model,local,device)
            if state_fingerprint(model)!=before or any(p.grad is not None for p in model.parameters()):
                raise RuntimeError("Frozen evaluation changed model state or gradients")
            if name in ("cubic_decoder","r_tail"):
                keys=("feature_target",) if name=="cubic_decoder" else ("direct","direct_full","feature_target","feature_reconstruction")
                for split in ("train","test"):
                    for key in keys:
                        if not np.array_equal(local[f"{split}_{key}"],public.endpoints["all_start"][f"{split}_{key}"]):
                            raise RuntimeError(f"Matched evaluation target changed: {name}/{split}/{key}")
            arrays.update({f"{name}_{key}":value for key,value in local.items()})
            arms[name]=dict(actual_steps=actual,optimizer_steps=counts[name],source_QCR_steps=3000,
                initial_module_sha256=initial[name],initial_common_module_sha256=common_initial[name],
                final_module_sha256=module_fingerprints(model),closure_objective="all_start_anchor",
                Q_updates=0 if name=="r_tail" else actual,C_updates=0 if name=="r_tail" else actual,R_updates=actual,
                numerical_fixed_points=[],root_search_performed=False,critical_r=None,thermal_nu=None)
            budget()
        evaluation_time=time.perf_counter()-evaluation_start
        if _sha_files([Path(p) for p in pins])!=pins:raise RuntimeError("Immutable source/data changed during training")
        manifest["paired_schedule_sha256"]=schedule;write_json(output/"manifest.json",manifest)
        np.savez_compressed(output/"evaluation_arrays.npz",**arrays)
        summary=dict(recipe=RECIPE,schema_version=5,stage=options.stage,status="complete",workflow_complete=True,returncode=0,
            actual_steps=actual,actual_steps_by_arm={n:actual for n in models},arms=arms,source_steps=3000,
            source_root=str(source),source_public_sha256=public.fingerprints,data_manifest_sha256=dataset.manifest_sha256,
            source_inputs_unchanged=True,initial_common_module_sha256=common_initial,initial_module_sha256=initial,
            paired_schedule_sha256=schedule,labels_used_in_training=False,gradient_routing_verified=True,
            cubic_C_target_source_equal=options.stage=="joint",Q_C_frozen_verified=options.stage=="r_tail",
            r_tail_QC_source_equal=options.stage=="r_tail",stochastic_raw_physical_batches=True,fixed_encoding_cache_used=False,
            peak_cuda_bytes=peak,gpu_uuid=receipt.get("uuid"),device=receipt,
            timing=dict(initialization_seconds=initialization,update_seconds=update_time,evaluation_seconds=evaluation_time,
                cpu_seconds=time.process_time()-cpu_start,wall_seconds=time.perf_counter()-wall_start),
            data_warning_count=dataset.warning_count,equilibrium_certified=False,root_search_performed=False,critical_r=None,thermal_nu=None)
        write_json(output/"summary.json",summary);budget()
        status("complete",step=actual,latest=latest,manifest_sha256=sha256_file(output/"manifest.json"),
            summary_sha256=sha256_file(output/"summary.json"),history_progress_sha256=sha256_file(output/"history_progress.json"),
            evaluation_arrays_sha256=sha256_file(output/"evaluation_arrays.npz"),peak_cuda_bytes=peak)
        return summary
    except Exception as error:
        held=isinstance(error,BudgetHold)
        write_json(output/"status.json",dict(recipe=RECIPE,stage=options.stage,status="budget_hold" if held else "failed",
            workflow_complete=False,returncode=3 if held else 1,error=str(error),requires_user_action=True,
            actual_steps=actual,latest_checkpoint_step=latest,traceback=traceback.format_exc()))
        if held:return dict(status="budget_hold")
        raise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("source","output"):p.add_argument("--"+name,required=True)
    p.add_argument("--stage",choices=("profile","joint","r_tail"),required=True)
    p.add_argument("--device",default="cuda:2");p.add_argument("--expected-gpu-uuid")
    p.add_argument("--cpu-budget-seconds",type=float,default=7200.)
    p.add_argument("--gpu-budget-seconds",type=float,default=3600.)
    p.add_argument("--max-cuda-bytes",type=int,default=8*1024**3)
    args=vars(p.parse_args(argv));profile=args["stage"]=="profile"
    result=run(Options(**args,steps=20 if profile else 3000,seed=191 if profile else 91))
    return 3 if result["status"]=="budget_hold" else 0


if __name__=="__main__":
    raise SystemExit(main())
