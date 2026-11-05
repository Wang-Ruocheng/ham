"""Fresh paired b=2 scalar phi4 QCR: legacy versus all-start closure.

Only R's objective differs. Q and C receive their original independent
objectives, with identical initial tensors, physical batches and model RNGs.
No older checkpoint, Q-closure feedback, physical coupling supervision, PCA
initialization, or nonuniform scale weighting is used.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from experiments.phi4_qcr import (
    FieldDataset, _device_receipt, _inside, cell_grid_evaluation,
    module_fingerprints, optimizer_step_counts, sha256_file,
    state_fingerprint, tensor_fingerprint, write_json,
)
from recurrent_rg.phi4_b2_qcr import Phi4B2QCR, continuous_objectives_b2


RECIPE = "phi4_b2_closure_objective_comparison_v1"
MODES = dict(legacy="one_step_and_rollout", all_start="all_start_anchor")
EVALUATION_SEED = 20261027


class BudgetHold(RuntimeError):
    pass


@dataclass
class Options:
    data: str
    output: str
    device: str = "cuda:2"
    expected_gpu_uuid: str | None = None
    steps: int = 3000
    seed: int = 91
    system_batch: int = 8
    sample_batch: int = 64
    information_pairs: int = 128
    log_every: int = 100
    checkpoint_every: int = 1000
    cpu_budget_seconds: float = 7200.
    gpu_budget_seconds: float = 1800.
    max_cuda_bytes: int = 8 * 1024**3
    profile_only: bool = False

    def validate(self):
        for key in ("steps", "system_batch", "sample_batch", "information_pairs", "log_every", "checkpoint_every", "max_cuda_bytes"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"Invalid positive integer {key}")
        if self.sample_batch < 4 or self.sample_batch % 2 or self.information_pairs < 2:
            raise ValueError("Even sample batch >=4 and at least two information pairs required")
        for key in ("cpu_budget_seconds", "gpu_budget_seconds"):
            if not math.isfinite(getattr(self,key)) or getattr(self,key) <= 0:
                raise ValueError(f"Invalid {key}")
        if self.profile_only and (self.steps != 20 or self.seed != 191):
            raise ValueError("Execution-only profile must use fresh seed191 and 20 updates per arm")
        if Path(self.output).exists():
            raise FileExistsError("Never overwrite or resume a closure comparison")


def load_dataset(root):
    """Geometry-specific loader; only configurations enter model optimization."""
    root = Path(root).resolve()
    manifest = json.loads((root/"manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError("Physical data are not complete")
    opts = manifest["options"]
    if opts["dimension"] != 2 or len(opts["sizes"]) != 1:
        raise ValueError("One two-dimensional microscopic size required")
    side = int(opts["sizes"][0])
    if side < 8 or side % 8 or opts["samples"] < 4 or opts["samples"] % 2:
        raise ValueError("Three b=2 transitions and equal saved-sample halves required")
    values, splits, cells, labels, receipts = [], [], [], [], []
    seen = set()
    n_cells = len(opts["r_values"])*len(opts["u_values"])
    for shard in manifest["shards"]:
        cell = int(shard["cell_index"])
        if cell in seen or not 0 <= cell < n_cells or shard["size"] != side:
            raise ValueError("Duplicate or unexpected physical shard")
        seen.add(cell)
        data, sidecar = _inside(root,shard["path"]), _inside(root,shard["sidecar"])
        if sha256_file(data) != shard["sha256"] or sha256_file(sidecar) != shard["sidecar_sha256"]:
            raise ValueError("Physical shard/sidecar SHA256 mismatch")
        meta = json.loads(sidecar.read_text())
        split = "test" if cell % opts["test_every"] == opts["test_offset"] else "train"
        expected = [opts["r_values"][cell//len(opts["u_values"])],opts["u_values"][cell%len(opts["u_values"])],opts["h"]]
        if (meta.get("status") != "complete" or meta["size"] != side or meta["dimension"] != 2
                or meta["cell_index"] != cell or meta["split"] != split or shard["split"] != split
                or meta["data_sha256"] != shard["sha256"] or [meta[k] for k in ("r","u","h")] != expected):
            raise ValueError("Whole-parameter-cell split or provenance mismatch")
        with np.load(data,allow_pickle=False) as archive:
            x = archive["configurations"]
        if (x.shape != (opts["chains"],opts["samples"],side,side,1)
                or x.dtype != np.float64 or not np.isfinite(x).all()):
            raise ValueError("Invalid physical field array")
        values.append(torch.from_numpy(np.ascontiguousarray(x)))
        splits.extend([split]*len(x));cells.extend([cell]*len(x));labels.extend([expected]*len(x))
        receipts.append({key:shard[key] for key in ("path","sha256","sidecar","sidecar_sha256")})
    if seen != set(range(n_cells)) or set(splits) != {"train","test"}:
        raise ValueError("Incomplete physical grid or missing holdout")
    result = FieldDataset(torch.cat(values),np.asarray(splits),np.asarray(cells,dtype=np.int64),
        np.asarray(labels,dtype=np.float64),sha256_file(root/"manifest.json"),receipts,
        int(manifest.get("warning_count",0)),manifest.get("qualification","Uncertified finite chains"),
        manifest.get("schema","unspecified"),manifest.get("sample_semantics","correlated_markov_chains"))
    if not math.isfinite(result.scale) or result.scale <= 0:
        raise ValueError("TRAIN microscopic RMS must be finite and positive")
    return result


@torch.no_grad()
def initialize_decoder(model,dataset,device,seed,budget_check):
    """TRAIN-only mean bias under the random, untrained C, at all four scales."""
    before=module_fingerprints(model)
    protected=dict(encoder=state_fingerprint(model.Q.net),decoder_weight=tensor_fingerprint(model.Q.decoder.weight))
    cpu_rng=torch.get_rng_state().clone()
    cuda_rng=torch.cuda.get_rng_state(device).clone() if device.type=="cuda" else None
    generator=torch.Generator(device=device).manual_seed(seed+9000)
    mean=torch.zeros(105,device=device,dtype=torch.float64)
    for index in dataset.train_indices:
        levels=model.coarse_grain(dataset.configurations[index].unsqueeze(0).to(device),generator)
        mean.add_(sum(model.features(x)[0] for x in levels)/len(levels))
        budget_check()
    mean.div_(len(dataset.train_indices))
    if not torch.isfinite(mean).all():
        raise FloatingPointError("Nonfinite TRAIN-only decoder calibration")
    model.Q.decoder.bias.copy_(mean)
    after=module_fingerprints(model)
    if (before["C"]!=after["C"] or before["R"]!=after["R"]
            or protected["encoder"]!=state_fingerprint(model.Q.net)
            or protected["decoder_weight"]!=tensor_fingerprint(model.Q.decoder.weight)
            or not torch.equal(cpu_rng,torch.get_rng_state())
            or (cuda_rng is not None and not torch.equal(cuda_rng,torch.cuda.get_rng_state(device)))
            or any(p.grad is not None for p in model.parameters())):
        raise RuntimeError("Initialization changed protected state or RNG")
    return dict(mode="train_mean_bias",scales=[0,1,2,3],scale_weighting="uniform",
        physical_ensembles=len(dataset.train_indices),fields_per_ensemble=int(dataset.configurations.shape[1]),
        seed=seed+9000,labels_used=False,test_data_used=False,pca_initialization=False,
        warm_start=False,C_updates=0,constructor_module_sha256=before,
        training_start_module_sha256=after,mean_sha256=tensor_fingerprint(mean),
        all_parameter_gradients_none=True,global_RNG_unchanged=True)


def independent_step(model,batch,optimizers,generator,mode,information_pairs):
    q,c,r,terms=continuous_objectives_b2(model,batch,generator=generator,
        information_pairs=information_pairs,closure_mode=mode)
    params={key:list(getattr(model,key).parameters()) for key in ("Q","C","R")}
    grads={key:torch.autograd.grad(loss,params[key]) for key,loss in zip(("Q","C","R"),(q,c,r))}
    for key in params:
        optimizers[key].zero_grad(set_to_none=True)
        for parameter,gradient in zip(params[key],grads[key]):
            if not torch.isfinite(gradient).all():
                raise FloatingPointError(f"Nonfinite {mode}/{key} gradient")
            parameter.grad=gradient
    for optimizer in optimizers.values():
        optimizer.step()
    model.Q.normalize_decoder_()
    return terms


@torch.no_grad()
def evaluate(model,dataset,device,budget_check):
    model.eval()
    rng=torch.Generator(device=device).manual_seed(EVALUATION_SEED)
    rows=[]
    for chain in dataset.configurations:
        levels=model.coarse_grain(chain.unsqueeze(0).to(device),rng)
        direct=[];full=[];target=[]
        for level in levels:
            half=level.shape[1]//2
            a,b=model.features(level[:,:half]),model.features(level[:,half:])
            direct.append(.5*(model.Q(a)+model.Q(b)));target.append(.5*(a+b))
            full.append(model.Q(target[-1]))
        g=torch.stack(direct,dim=1)
        one=torch.stack([model.advance(g[:,k]) for k in range(3)],dim=1)
        rollout=[g[:,0]]
        for _ in range(3):rollout.append(model.advance(rollout[-1]))
        rollout=torch.stack(rollout,dim=1)
        values=dict(direct=g,direct_full=torch.stack(full,dim=1),one_step=one,rollout=rollout,
            feature_target=torch.stack(target,dim=1),feature_reconstruction=model.Q.decode(g),
            one_step_feature_prediction=model.Q.decode(one),rollout_feature_prediction=model.Q.decode(rollout))
        rows.append({key:value.cpu().numpy()[0] for key,value in values.items()})
        budget_check()
    arrays={}
    for split in ("train","test"):
        mask=dataset.splits==split
        for key in rows[0]:arrays[f"{split}_{key}"]=np.stack([row[key] for row in rows])[mask]
        arrays[f"{split}_labels"]=dataset.labels[mask]
        arrays[f"{split}_cell_ids"]=dataset.cell_ids[mask]
    # Generalized equivalent of the old grid helper, now with three inputs.
    direct=np.concatenate([arrays[f"{s}_direct"] for s in ("train","test")])
    one=np.concatenate([arrays[f"{s}_one_step"] for s in ("train","test")])
    ids=np.concatenate([arrays[f"{s}_cell_ids"] for s in ("train","test")])
    labels=np.concatenate([arrays[f"{s}_labels"] for s in ("train","test")])
    unique=np.unique(ids)
    means=np.stack([direct[ids==cell].mean(0) for cell in unique])
    mean_R=np.stack([one[ids==cell].mean(0) for cell in unique])
    R_mean=model.advance(torch.as_tensor(means[:,:-1],device=device)).cpu().numpy()
    arrays.update(grid_cell_ids=unique,grid_labels=np.stack([labels[np.flatnonzero(ids==c)[0]] for c in unique]),
        grid_mean_direct=means,grid_mean_R=mean_R,grid_R_of_mean=R_mean,
        grid_jensen_difference=mean_R-R_mean,grid_displacement=R_mean-means[:,:-1])
    return arrays


def flow_and_roots(model,arrays,device):
    """Bounded endpoint R evaluation, not physical critical-point inference."""
    from scipy.optimize import brentq
    data=np.concatenate([arrays[f"{s}_direct"][:,:-1].ravel() for s in ("train","test")])
    low,high=float(data.min()),float(data.max())
    padding=.1*max(high-low,1e-6)
    grid=np.linspace(low-padding,high+padding,256)
    with torch.no_grad():mapped=model.advance(torch.as_tensor(grid[:,None],dtype=torch.float64,device=device)).cpu().numpy()
    displacement=mapped[:,0]-grid
    def drift(g):
        with torch.no_grad():
            x=torch.tensor([[g]],dtype=torch.float64,device=device)
            return float((model.advance(x)-x).item())
    candidates=[float(grid[i]) for i in range(len(grid)) if displacement[i]==0.]
    for i in range(len(grid)-1):
        if displacement[i]*displacement[i+1]<0:
            candidates.append(float(brentq(drift,grid[i],grid[i+1],xtol=1e-13)))
    roots=[]
    for g in sorted(candidates):
        if roots and abs(g-roots[-1]["g"])<1e-8:continue
        x=torch.tensor([[g]],device=device,dtype=torch.float64,requires_grad=True)
        derivative=float(torch.autograd.grad(model.advance(x).sum(),x)[0].item())
        kind="attracting" if abs(derivative)<1-1e-6 else "repelling" if abs(derivative)>1+1e-6 else "neutral"
        support=[]
        for split in ("train","test"):
            for k in range(4):
                values=arrays[f"{split}_direct"][:,k,0]
                support.append(dict(split=split,scale=k,within_range=bool(values.min()<=g<=values.max()),nearest_distance=float(np.min(abs(values-g)))))
        roots.append(dict(g=g,residual=abs(drift(g)),R_derivative=derivative,classification=kind,
            data_nearby=support,physical_critical_point_certified=False,thermal_nu=None))
    arrays.update(flow_g=grid[:,None],flow_R=mapped,flow_displacement=displacement[:,None])
    return roots


def train(options):
    options.validate()
    output=Path(options.output).resolve();data_root=Path(options.data).resolve()
    if output==data_root or output in data_root.parents or data_root in output.parents:
        raise ValueError("Training output must be separate from physical data")
    output.mkdir(parents=True,exist_ok=False)
    wall_start,cpu_start=time.perf_counter(),time.process_time()
    actual=0;checkpoint_step=0;compute_start=None;device=torch.device(options.device)
    peak=0
    def budget():
        nonlocal peak
        if time.process_time()-cpu_start>options.cpu_budget_seconds:
            raise BudgetHold("CPU budget reached; no restart or shorter training")
        if device.type=="cuda" and compute_start is not None:
            torch.cuda.synchronize(device)
            peak=max(peak,int(torch.cuda.max_memory_allocated(device)),int(torch.cuda.max_memory_reserved(device)))
            if peak>options.max_cuda_bytes or time.perf_counter()-compute_start>options.gpu_budget_seconds:
                raise BudgetHold("GPU time/peak budget reached; no migration or shorter training")
    def status(value,**kw):
        write_json(output/"status.json",dict(recipe=RECIPE,status=value,workflow_complete=value=="complete",
            actual_steps_by_arm={name:actual for name in MODES},latest_checkpoint_step=checkpoint_step,
            returncode=0 if value=="complete" else None,error=None,requires_user_action=False,**kw))
    try:
        torch.set_num_threads(1);torch.manual_seed(options.seed)
        status("initializing")
        dataset=load_dataset(data_root)
        if options.system_batch>len(dataset.train_indices) or options.sample_batch>dataset.configurations.shape[1]:
            raise ValueError("Batch exceeds existing TRAIN ensembles/samples")
        budget()
        compute_start=time.perf_counter()
        receipt=_device_receipt(device,options.expected_gpu_uuid)
        if device.type=="cuda":torch.cuda.reset_peak_memory_stats(device)
        base=Phi4B2QCR(dataset.scale,hidden=96,theory_dim=1).to(device)
        constructor=module_fingerprints(base)
        init_receipt=initialize_decoder(base,dataset,device,options.seed,budget)
        initial=module_fingerprints(base)
        models={name:copy.deepcopy(base) for name in MODES}
        del base
        optimizers={name:{key:torch.optim.Adam(getattr(model,key).parameters(),lr=.001)
                          for key in ("Q","C","R")} for name,model in models.items()}
        field_rng={name:torch.Generator(device=device).manual_seed(options.seed+202) for name in MODES}
        batch_rng=torch.Generator(device="cpu").manual_seed(options.seed+101)
        import recurrent_rg.phi4_b2_qcr as model_module
        sources={str(Path(__file__).resolve()):sha256_file(__file__),str(Path(model_module.__file__).resolve()):sha256_file(model_module.__file__)}
        manifest=dict(schema_version=5,recipe=RECIPE,options=asdict(options),theory_dim=1,
            lattice_sizes=[int(dataset.configurations.shape[2]//2**k) for k in range(4)],
            data_root=str(data_root),data_manifest_sha256=dataset.manifest_sha256,data_shards=dataset.receipts,
            block_size=2,stride=2,dtype="float64",scale=dataset.scale,
            initialization="fresh matched random Q/C/R; TRAIN mean decoder bias only",
            random_construction_module_sha256=constructor,initial_module_sha256=initial,
            q_initialization_receipt=init_receipt,source_sha256=sources,closure_modes=MODES,
            closure_scale_weights=None,closure_weights=dict(commutator=5,rollout_weight=5,total_all_start=10),
            optimizer="three independent fresh Adam per arm",learning_rate=.001,learning_rate_policy="constant",
            Q_closure_feedback=False,labels_used_in_training=False,model_input_members=["configurations"],
            warm_start=False,pca_initialization=False,C_pretraining=False,kernel_selection=False,
            gradient_routing="Q representation only; C objective only; R own closure only; simultaneous independent gradients",
            data_warning_count=dataset.warning_count,sampler_schema=dataset.sampler_schema,
            sample_semantics=dataset.sample_semantics,sampler_qualification=dataset.qualification,
            continuous_recipe="Gaussian C; 105 soft features; kernel-smoothed full-shell information proxy, not exact Shannon MI",
            default_recipe_modified=False,profile_only=options.profile_only,thermal_nu=None,critical_r=None,
            equilibrium_certified=False,device=receipt)
        write_json(output/"manifest.json",manifest)
        history=[];schedule_hash=hashlib.sha256();train_indices=torch.as_tensor(dataset.train_indices)
        update_start=time.perf_counter();initialization_seconds=update_start-compute_start
        status("training")
        for step in range(1,options.steps+1):
            budget()
            systems=train_indices[torch.randperm(len(train_indices),generator=batch_rng)[:options.system_batch]]
            samples=torch.randperm(dataset.configurations.shape[1],generator=batch_rng)[:options.sample_batch]
            schedule_hash.update(systems.numpy().tobytes());schedule_hash.update(samples.numpy().tobytes())
            batch=dataset.configurations[systems][:,samples].to(device)
            metrics={}
            for name,model in models.items():
                model.train()
                metrics[name]=independent_step(model,batch,optimizers[name],field_rng[name],MODES[name],options.information_pairs)
            if metrics["legacy"]["q_loss"]!=metrics["all_start"]["q_loss"] or metrics["legacy"]["c_loss"]!=metrics["all_start"]["c_loss"]:
                raise RuntimeError("Paired Q/C objectives diverged between R-only arms")
            actual=step
            if step==1 or step%options.log_every==0 or step==options.steps:
                modules={name:module_fingerprints(model) for name,model in models.items()}
                if any(modules["legacy"][key]!=modules["all_start"][key] for key in ("Q","C")):
                    raise RuntimeError("Q/C tensor pairing failed")
                row=dict(step=step,arms=metrics,elapsed_seconds=time.perf_counter()-update_start,C_Q_pairing_verified=True)
                history.append(row)
                write_json(output/"history_progress.json",dict(recipe=RECIPE,history=history,latest_step=step,latest_checkpoint_step=checkpoint_step))
                print(json.dumps(dict(event="phi4_b2_paired_progress",**row),allow_nan=False),flush=True)
                status("training",peak_cuda_bytes=peak)
            if not options.profile_only and (step%options.checkpoint_every==0 or step==options.steps):
                for name,model in models.items():
                    arm=output/name;arm.mkdir(exist_ok=True)
                    torch.save(dict(recipe=RECIPE,step=step,model=model.state_dict(),closure_objective=MODES[name],
                        optimizers={key:op.state_dict() for key,op in optimizers[name].items()},initial_module_sha256=initial),arm/f"training_step_{step}.pt")
                checkpoint_step=step
        budget();update_seconds=time.perf_counter()-update_start
        counts={name:optimizer_step_counts(optimizers[name],options.steps) for name in MODES}
        for model in models.values():
            for parameter in model.parameters():parameter.grad=None
        manifest["paired_schedule_sha256"]=schedule_hash.hexdigest()
        write_json(output/"manifest.json",manifest)
        if options.profile_only:
            budget()
            profile=dict(status="complete",workflow_complete=True,returncode=0,profile_only=True,
                steps_per_arm=options.steps,arms=2,seed=options.seed,device=options.device,
                gpu_uuid=receipt.get("uuid"),initialization_seconds=initialization_seconds,
                update_seconds=update_seconds,evaluation_seconds=0.,peak_cuda_bytes=peak,
                cpu_seconds=time.process_time()-cpu_start,
                gpu_used_seconds=time.perf_counter()-compute_start if device.type=="cuda" else 0.,
                timing_scope="Both arms; initialization/calibration/optimizer setup and all 20 updates each included; no scientific endpoint evaluation",
                C_Q_pairing_verified=True,gradient_routing_verified=True,optimizer_steps=counts,
                scientific_result=False,warm_start_reused_for_formal=False)
            write_json(output/"profile.json",profile)
            status("complete",profile_only=True,profile_sha256=sha256_file(output/"profile.json"))
            return profile
        status("evaluating")
        evaluation_start=time.perf_counter();all_arrays={};summaries={}
        final_before={name:state_fingerprint(model) for name,model in models.items()}
        for name,model in models.items():
            arrays=evaluate(model,dataset,device,budget)
            roots=flow_and_roots(model,arrays,device)
            summaries[name]=dict(actual_steps=options.steps,optimizer_steps=counts[name],closure_objective=MODES[name],
                initial_module_sha256=initial,final_module_sha256=module_fingerprints(model),
                numerical_fixed_points=roots,thermal_nu=None,critical_r=None)
            all_arrays.update({f"{name}_{key}":value for key,value in arrays.items()})
            if state_fingerprint(model)!=final_before[name] or any(p.grad is not None for p in model.parameters()):
                raise RuntimeError("Frozen evaluation changed model state/gradients")
            budget()
        for key in ("Q","C"):
            if summaries["legacy"]["final_module_sha256"][key]!=summaries["all_start"]["final_module_sha256"][key]:
                raise RuntimeError("Final shared Q/C tensor pairing failed")
        for split in ("train","test"):
            for key in ("direct","direct_full","feature_target","feature_reconstruction","labels","cell_ids"):
                if not np.array_equal(all_arrays[f"legacy_{split}_{key}"],all_arrays[f"all_start_{split}_{key}"]):
                    raise RuntimeError(f"Final paired evaluation differs: {split}/{key}")
        if sha256_file(data_root/"manifest.json")!=dataset.manifest_sha256:
            raise RuntimeError("Physical manifest changed during training")
        for row in dataset.receipts:
            if sha256_file(data_root/row["path"])!=row["sha256"] or sha256_file(data_root/row["sidecar"])!=row["sidecar_sha256"]:
                raise RuntimeError("New physical shard/sidecar changed during training")
        for name,digest in sources.items():
            if sha256_file(name)!=digest:raise RuntimeError("Frozen training source changed")
        budget();evaluation_seconds=time.perf_counter()-evaluation_start
        np.savez(output/"evaluation_arrays.npz",**all_arrays)
        write_json(output/"history_progress.json",dict(recipe=RECIPE,history=history,latest_step=actual,latest_checkpoint_step=checkpoint_step))
        summary=dict(recipe=RECIPE,schema_version=5,status="complete",actual_steps=actual,
            actual_steps_by_arm={name:actual for name in MODES},arms=summaries,closure_modes=MODES,
            initial_module_sha256=initial,C_Q_pairing_verified=True,pairing_verified=True,
            gradient_routing_verified=True,frozen_evaluation_verified=True,paired_schedule_sha256=schedule_hash.hexdigest(),
            data_manifest_sha256=dataset.manifest_sha256,data_warning_count=dataset.warning_count,
            sampler_schema=dataset.sampler_schema,sample_semantics=dataset.sample_semantics,
            sampler_qualification=dataset.qualification,labels_used_in_training=False,
            theory_dim=1,lattice_sizes=manifest["lattice_sizes"],block_size=2,peak_cuda_bytes=peak,
            default_recipe_modified=False,warm_start=False,Q_closure_feedback=False,
            timing=dict(initialization_seconds=initialization_seconds,update_seconds=update_seconds,
                evaluation_seconds=evaluation_seconds,compute_seconds=time.perf_counter()-compute_start,
                gpu_used_seconds=time.perf_counter()-compute_start if device.type=="cuda" else 0.,
                cpu_seconds=time.process_time()-cpu_start,wall_seconds=time.perf_counter()-wall_start),
            critical_r=None,thermal_nu=None,equilibrium_certified=False,device=receipt)
        write_json(output/"summary.json",summary);budget()
        status("complete",actual_steps=actual,summary_sha256=sha256_file(output/"summary.json"),
            manifest_sha256=sha256_file(output/"manifest.json"),
            history_progress_sha256=sha256_file(output/"history_progress.json"),
            evaluation_arrays_sha256=sha256_file(output/"evaluation_arrays.npz"),peak_cuda_bytes=peak)
        return summary
    except Exception as error:
        hold=isinstance(error,BudgetHold)
        write_json(output/"status.json",dict(recipe=RECIPE,status="budget_hold" if hold else "failed",
            workflow_complete=False,actual_steps_by_arm={name:actual for name in MODES},
            latest_checkpoint_step=checkpoint_step,returncode=3 if hold else 1,error=str(error),
            requires_user_action=True,traceback=traceback.format_exc()))
        if hold:return dict(status="budget_hold")
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("data","output"):parser.add_argument(f"--{name}",required=True)
    parser.add_argument("--device",default="cuda:2")
    parser.add_argument("--expected-gpu-uuid")
    for name,default in (("steps",3000),("seed",91),("system-batch",8),("sample-batch",64),
                         ("information-pairs",128),("log-every",100),("checkpoint-every",1000),
                         ("max-cuda-bytes",8*1024**3)):
        parser.add_argument(f"--{name}",type=int,default=default)
    parser.add_argument("--cpu-budget-seconds",type=float,default=7200.)
    parser.add_argument("--gpu-budget-seconds",type=float,default=1800.)
    parser.add_argument("--profile-only",action="store_true")
    result=train(Options(**vars(parser.parse_args())))
    if result.get("status")=="budget_hold":raise SystemExit(3)


if __name__=="__main__":main()
