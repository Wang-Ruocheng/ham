"""Authorized frozen-C, scalar-Q closure-compatibility ablation.

This separate CPU-only entry point does not change any default recipe.  It
reuses immutable per-field statistics, not physical configurations.  A and B
start from the same completed Q/R endpoint, with fresh optimizers.  Only B
lets a parameter-frozen copy of R guide Q through its input derivatives.
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
    Phi4QCR, cell_grid_evaluation, closure_metrics, module_fingerprints,
    scalar_flow, sha256_file, state_fingerprint, write_json,
)
from experiments.phi4_q_only_diagnostic import (
    ARRAY_KEYS, EVALUATION_METADATA, load_cache, schedule, statistic_halves,
)


RECIPE = "phi4_scalar_frozenC_Q_closure_compatibility_v1"
ARM_NAMES = ("A", "B")


class BudgetHold(RuntimeError):
    pass


@dataclass
class Options:
    source_run: str
    cache: str
    output: str
    steps: int = 3000
    seed: int = 91
    system_batch: int = 8
    sample_batch: int = 64
    log_every: int = 100
    cpu_budget_seconds: float = 800.

    def validate(self):
        for key in ("steps", "system_batch", "sample_batch", "log_every"):
            if not isinstance(getattr(self, key), int) or getattr(self, key) < 1:
                raise ValueError(f"Invalid positive integer: {key}")
        if self.sample_batch % 2:
            raise ValueError("The sample batch must have equal halves")
        if not math.isfinite(self.cpu_budget_seconds) or self.cpu_budget_seconds <= 0:
            raise ValueError("Invalid CPU budget")


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def closure_objective(model, g, frozen_R=False):
    """S=2 all-start anchor; frozen parameters still differentiate inputs."""
    if g.ndim != 3 or g.shape[1:] != (3, 1) or g.shape[0] < 1:
        raise ValueError("Scalar encodings must have shape [ensembles,3,1]")

    def advance(x):
        if not frozen_R:
            return model.advance(x)
        state = {name: value.detach() for name, value in model.R.named_parameters()}
        state.update({name: value.detach() for name, value in model.R.named_buffers()})
        return x + torch.func.functional_call(model.R, state, (x,))

    first = advance(g[:, 0])
    second = advance(g[:, 1])
    twice = advance(first)
    e01 = (first - g[:, 1]).square().sum(-1).mean()
    e12 = (second - g[:, 2]).square().sum(-1).mean()
    e02 = (twice - g[:, 2]).square().sum(-1).mean()
    return 5. * e01 + 2.5 * e12 + 2.5 * e02


def paired_objectives(model, first, second, reference_std, guard_scale,
                      closure_alpha, closure_enabled, guard_features=None):
    """No C calls; Q-compatible closure has no R parameter gradients.

    The common guard acts on full TRAIN ensembles (both cached C draws), not
    the noisy eight-ensemble minibatch. It is a floor, not a scale-wise mean
    alignment, and does not require g_s == g_t.
    """
    first, second = first.detach(), second.detach()
    if (first.shape != second.shape or first.ndim != 3
            or first.shape[1:] != (3, 105) or first.shape[0] < 1):
        raise ValueError("Half statistics must be [ensembles,3,105]")
    if not torch.isfinite(first).all() or not torch.isfinite(second).all():
        raise FloatingPointError("Nonfinite frozen statistics")
    ga, gb = model.Q(first), model.Q(second)
    g = .5 * (ga + gb)
    target = .5 * (first + second)
    rec = (model.Q.decode(g) - target).square().sum(-1).mean(0)
    subset = (ga - gb).square().sum(-1).mean(0)
    base = 10. * rec.mean() + 20. * subset[0] + 5. * subset[1:].mean()
    guard_input = target if guard_features is None else guard_features.detach()
    if guard_input.shape[-2:] != (3, 105):
        raise ValueError("Guard statistics must end in [3,105]")
    if guard_input.numel() == 0 or not torch.isfinite(guard_input).all():
        raise FloatingPointError("Guard statistics must be finite and nonempty")
    guard_g = model.Q(guard_input).reshape(-1, 3, 1)
    std = guard_g.var(0, unbiased=False)[:, 0].clamp_min(1e-30).sqrt()
    ref = torch.as_tensor(reference_std, dtype=std.dtype, device=std.device).detach()
    if ref.shape != (3,) or not torch.isfinite(ref).all() or torch.any(ref <= 1e-12):
        raise ValueError("Finite positive, noncollapsed TRAIN reference standard deviations required")
    if any(not math.isfinite(float(x)) or float(x) < 0 for x in (guard_scale, closure_alpha)):
        raise ValueError("Frozen objective coefficients must be finite and nonnegative")
    guard = (1. - std / (.5 * ref)).clamp_min(0.).square().mean()
    compatible = closure_objective(model, g, frozen_R=True)
    qtotal = base + float(guard_scale) * guard
    if closure_enabled:
        qtotal = qtotal + float(closure_alpha) * compatible
    rloss = closure_objective(model, g.detach(), frozen_R=False)
    terms = dict(q_loss=qtotal, q_base=base, q_compatible_closure=compatible,
        q_guard=guard, reconstruction=rec.mean(), subset_micro=subset[0],
        subset_coarse=subset[1:].mean(), r_loss=rloss,
        minimum_std_ratio_to_source=(std / ref).min())
    for k in range(3):
        terms[f"reconstruction_level_{k}"] = rec[k]
        terms[f"subset_level_{k}"] = subset[k]
        terms[f"std_ratio_to_source_level_{k}"] = std[k] / ref[k]
    if any(not torch.isfinite(value).all() for value in terms.values()):
        raise FloatingPointError("Nonfinite scalar closure objective")
    return qtotal, rloss, terms


def _load_completed_model(root):
    root = Path(root).resolve()
    manifest, summary, status = [_json(root / f"{name}.json")
                                 for name in ("manifest", "summary", "status")]
    if (summary.get("status") != "complete" or status.get("status") != "complete"
            or status.get("workflow_complete") is not True or status.get("returncode") != 0
            or manifest.get("theory_dim") != 1 or manifest.get("lattice_sizes") != [36, 12, 4]
            or summary.get("actual_steps") != 3000):
        raise ValueError("Expected a complete scalar 36→12→4 source endpoint")
    for key, name in (("summary_sha256", "summary.json"),
                      ("evaluation_arrays_sha256", "evaluation_arrays.npz")):
        if status.get(key) != sha256_file(root / name):
            raise ValueError(f"Source receipt mismatch: {name}")
    if manifest.get("source_sha256"):
        import recurrent_rg.phi4_qcr as model_module
        expected = [value for name, value in manifest["source_sha256"].items()
                    if str(name).endswith("/recurrent_rg/phi4_qcr.py")]
        if len(expected) != 1 or expected[0] != sha256_file(model_module.__file__):
            raise ValueError("Continuous Q/C/R architecture source differs from its frozen endpoint")
    path = root / "training_step_3000.pt"
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("step") != 3000:
        raise ValueError("Source checkpoint step mismatch")
    with torch.random.fork_rng(devices=[]):
        model = Phi4QCR(manifest["scale"], hidden=manifest.get("hidden", 96), theory_dim=1)
    model.load_state_dict(saved["model"], strict=True)
    model.eval().requires_grad_(False)
    if state_fingerprint(model) != summary["final_state_sha256"]:
        raise ValueError("Source tensor state mismatch")
    return model, manifest, summary


def load_source_for_cache(source_run, cache, features, receipt):
    """Reuse an older cache only after verifying the actual C tensor identity."""
    root = Path(source_run).resolve()
    source, manifest, summary = _load_completed_model(root)
    oldroot = Path(receipt["source_run"]).resolve()
    old, old_manifest, old_summary = _load_completed_model(oldroot)
    if (sha256_file(oldroot / "training_step_3000.pt") != receipt["source_checkpoint_sha256"]
            or sha256_file(oldroot / "manifest.json") != receipt["source_manifest_sha256"]
            or state_fingerprint(old) != receipt["source_state_sha256"]):
        raise ValueError("Statistics cache source identity mismatch")
    if (state_fingerprint(source.C) != state_fingerprint(old.C)
            or float(source.scale) != receipt["scale"]
            or manifest["data_manifest_sha256"] != receipt["source_data_manifest_sha256"]
            or old_manifest["data_manifest_sha256"] != manifest["data_manifest_sha256"]):
        raise ValueError("Cached C, normalization or physical data differ")
    with np.load(root / "evaluation_arrays.npz", allow_pickle=False) as arrays:
        for split in ("train", "test"):
            actual = .5 * (features[f"eval_{split}_first"].numpy()
                           + features[f"eval_{split}_second"].numpy())
            if not np.array_equal(actual, arrays[f"{split}_feature_target"]):
                raise ValueError("Cached evaluation targets are not exactly paired with latest source")
    protected = {}
    for run in dict.fromkeys((root, oldroot)):
        for name in ("manifest.json", "summary.json", "status.json", "evaluation_arrays.npz",
                     "training_step_3000.pt"):
            protected[str(run / name)] = sha256_file(run / name)
    for name in ("receipt.json", "status.json", "arrays.npz"):
        path = Path(cache).resolve() / name
        protected[str(path)] = sha256_file(path)
    return source, manifest, summary, protected


@torch.no_grad()
def evaluate_cached(model, features):
    arrays = {}
    model.eval()
    for split in ("train", "test"):
        a, b = (features[f"eval_{split}_{x}"] for x in ("first", "second"))
        g = .5 * (model.Q(a) + model.Q(b))
        one = torch.stack([model.advance(g[:, k]) for k in range(2)], dim=1)
        roll = [g[:, 0]]
        for _ in range(2):
            roll.append(model.advance(roll[-1]))
        roll = torch.stack(roll, dim=1)
        values = dict(direct=g, direct_full=model.Q(.5 * (a+b)), one_step=one,
            rollout=roll, feature_target=.5 * (a+b), feature_reconstruction=model.Q.decode(g),
            one_step_feature_prediction=model.Q.decode(one),
            rollout_feature_prediction=model.Q.decode(roll))
        arrays.update({f"{split}_{key}": value.numpy() for key, value in values.items()})
    return arrays


def _scalar(terms):
    return {key: float(value.detach()) for key, value in terms.items()}


def train(options):
    options.validate()
    root = Path(options.output).resolve()
    if root.exists():
        raise FileExistsError("Never overwrite, resume or rerun an ablation endpoint")
    for protected in (Path(options.source_run).resolve(), Path(options.cache).resolve()):
        if root == protected or protected in root.parents or root in protected.parents:
            raise ValueError("Output must be separate from immutable inputs")
    root.mkdir(parents=True)
    started, cpu_started = time.perf_counter(), time.process_time()
    steps = dict.fromkeys(ARM_NAMES, 0)
    source = None

    def status(value, **kw):
        write_json(root / "status.json", dict(recipe=RECIPE, status=value,
            workflow_complete=value == "complete", actual_steps_by_arm=steps.copy(),
            returncode=0 if value == "complete" else None, error=None,
            requires_user_action=False, **kw))

    def budget():
        elapsed = time.process_time() - cpu_started
        if elapsed > options.cpu_budget_seconds:
            raise BudgetHold(f"CPU cap reached: {elapsed:.3f}s > {options.cpu_budget_seconds}s")

    try:
        torch.set_num_threads(1)
        status("initializing")
        features, receipt = load_cache(options.cache)
        cache_status = _json(Path(options.cache) / "status.json")
        if (cache_status.get("status") != "complete"
                or cache_status.get("receipt_sha256") != sha256_file(Path(options.cache) / "receipt.json")):
            raise ValueError("Statistics cache is not successfully complete")
        source, source_manifest, source_summary, pins = load_source_for_cache(
            options.source_run, options.cache, features, receipt)
        source_before = state_fingerprint(source)
        modules = module_fingerprints(source)
        models = {name: copy.deepcopy(source) for name in ARM_NAMES}
        for model in models.values():
            model.Q.requires_grad_(True)
            model.R.requires_grad_(True)
        optimizers = {name: {key: torch.optim.Adam(getattr(model, key).parameters(), lr=.001)
                             for key in ("Q", "R")} for name, model in models.items()}
        train_data = features["train_per_sample"]
        if options.system_batch > train_data.shape[1] or options.sample_batch > train_data.shape[3]:
            raise ValueError("Batch exceeds existing TRAIN statistics")
        # Each of two cached C draws is averaged over saved fields, not treated
        # as another independent physical chain. No cell/coupling labels used.
        full_first = train_data[:, :, :, :train_data.shape[3]//2].mean(3)
        full_second = train_data[:, :, :, train_data.shape[3]//2:].mean(3)
        guard_features = .5 * (full_first + full_second)
        with torch.no_grad():
            # The guard reference uses the identical estimator as its future
            # evaluations, not the slightly different half-encoding estimator.
            initial_guard_g = source.Q(guard_features)
            reference_std = initial_guard_g.reshape(-1, 3, 1).std(0, unbiased=False)[:, 0]
            _, _, calibration = paired_objectives(source, full_first.flatten(0, 1),
                full_second.flatten(0, 1), reference_std, 0., 0., False, guard_features)
            guard_scale = float(calibration["q_base"])
            initial_closure = float(calibration["q_compatible_closure"])
            if min(guard_scale, initial_closure) <= 1e-15:
                raise ValueError("Degenerate TRAIN-only objective calibration")
            closure_alpha = guard_scale / initial_closure
        batches = schedule(train_data, options)
        digest = hashlib.sha256(np.asarray(batches[0], dtype=np.int64).tobytes())
        for batch in batches[1:]:
            digest.update(batch.numpy().tobytes())
        manifest = dict(recipe=RECIPE, schema_version=5, options=asdict(options),
            source_run=str(Path(options.source_run).resolve()), source_pins=pins,
            source_checkpoint_sha256=pins[str(Path(options.source_run).resolve()/"training_step_3000.pt")],
            source_actual_steps=source_summary["actual_steps"], additional_updates_per_arm=options.steps,
            initial_module_sha256=modules, data_manifest_sha256=source_manifest["data_manifest_sha256"],
            cache_receipt_sha256=sha256_file(Path(options.cache)/"receipt.json"),
            cache_arrays_sha256=receipt["arrays_sha256"], cache_C_matches_latest_source=True,
            paired_schedule_sha256=digest.hexdigest(), device="cpu", threads=1,
            optimizer="independent fresh Adam .001", learning_rate=.001,
            theory_dim=1, lattice_sizes=[36,12,4], decoder="unchanged shared affine, unit column norm",
            arms={"A":"原 Q 目标＋共同 TRAIN 方差下限；R 原闭合目标",
                  "B":"与 A 相同，增加仅更新 Q 的冻结参数 R 闭合兼容项"},
            closure_objective="all_start_anchor", target_scale_weights="uniform",
            q_closure_formula="5*e01+2.5*e12+2.5*e02, R parameters detached, inputs differentiable",
            Q_base="10 reconstruction +20 microscopic subset +5 coarse subset",
            guard=dict(reference_std=reference_std.tolist(), floor_ratio=.5, weight=guard_scale,
                formula="mean_s relu(1 - TRAIN_std_s/(.5*source_TRAIN_std_s))^2",
                calibration="source Q; both TRAIN C draws; no TEST or labels; frozen once"),
            closure_alpha=closure_alpha, alpha_calibration=dict(q_base=guard_scale, closure=initial_closure),
            C_updates=0, physical_sampling_calls=0, MI_calls=0, GPU_used=False,
            labels_used_in_training=False, training_input_members=["train_per_sample"],
            evaluation_only_members=list(EVALUATION_METADATA),
            gradient_routing=dict(Q="own representation + shared guard; B adds frozen-R-input closure",
                R="own detached-g all-start closure only", C="frozen; not evaluated"),
            default_recipe_modified=False, warm_start=True, training_C_draws=2,
            train_physical_ensembles=train_data.shape[1], test_physical_ensembles=len(features["eval_test_first"]),
            data_warnings={key:source_summary.get(key) for key in
                ("data_warning_count","sampler_schema","sampler_qualification","equilibrium_certified")},
            source_code_sha256={str(Path(__file__).resolve()):sha256_file(__file__)},
            qualification="固定最终 C 的缓存 Q/R 消融；不是新随机联合 QCR 或完整配方复现；单 seed、原采样警告保留",
            thermal_nu=None, critical_r=None, equilibrium_certified=False)
        write_json(root/"manifest.json", manifest)
        budget()
        history=[]

        def record(step, batch_metrics):
            row=dict(step=step, arms=batch_metrics, elapsed_seconds=time.perf_counter()-started)
            history.append(row)
            write_json(root/"history_progress.json",dict(recipe=RECIPE,latest_step=step,
                latest_checkpoint_step=0,actual_steps_by_arm=steps.copy(),history=history))
            print(json.dumps(dict(event="scalar_closure_progress",**row),allow_nan=False),flush=True)

        initial_metrics={}
        for name, model in models.items():
            _,_,terms=paired_objectives(model,full_first.flatten(0,1),full_second.flatten(0,1),
                reference_std,guard_scale,closure_alpha,name=="B",guard_features)
            initial_metrics[name]=_scalar(terms)
        record(0,initial_metrics)
        status("training")
        for step in range(1,options.steps+1):
            budget()
            first,second=statistic_halves(train_data,batches[0][step-1],batches[1][step-1],batches[2][step-1])
            batch_metrics={}
            for name,model in models.items():
                qloss,rloss,terms=paired_objectives(model,first,second,reference_std,
                    guard_scale,closure_alpha,name=="B",guard_features)
                params={key:list(getattr(model,key).parameters()) for key in ("Q","R")}
                grads={"Q":torch.autograd.grad(qloss,params["Q"]),
                       "R":torch.autograd.grad(rloss,params["R"])}
                for key in ("Q","R"):
                    optimizers[name][key].zero_grad(set_to_none=True)
                    for param,grad in zip(params[key],grads[key]):
                        if not torch.isfinite(grad).all():
                            raise FloatingPointError(f"Nonfinite {name}/{key} gradient")
                        param.grad=grad
                for optimizer in optimizers[name].values():
                    optimizer.step()
                model.Q.normalize_decoder_()
                if any(p.grad is not None or p.requires_grad for p in model.C.parameters()):
                    raise RuntimeError("C must remain frozen with gradients None")
                steps[name]=step
                batch_metrics[name]=_scalar(terms)
            if step==1 or step%options.log_every==0 or step==options.steps:
                record(step,batch_metrics)
                status("training",cpu_seconds=time.process_time()-cpu_started)
        status("evaluating")
        counts={}
        for name,model in models.items():
            counts[name]={}
            for key in ("Q","R"):
                values=[int(state["step"].item()) for state in optimizers[name][key].state.values()]
                if len(values)!=len(list(getattr(model,key).parameters())) or min(values)!=options.steps or max(values)!=options.steps:
                    raise RuntimeError("Adam counters must all reach the fixed endpoint")
                counts[name][key]=dict(min=min(values),max=max(values),parameters=len(values))
            for parameter in model.parameters():
                parameter.grad=None
        # Couplings are first opened here, after all optimization has finished.
        with np.load(Path(options.cache)/"arrays.npz",allow_pickle=False) as archive:
            metadata={key:np.array(archive[key],copy=True) for key in EVALUATION_METADATA}
        if set(metadata["eval_train_cell_ids"]) & set(metadata["eval_test_cell_ids"]):
            raise ValueError("TRAIN and TEST parameter cells overlap")
        all_arrays={}
        summaries={}
        for name,model in {"source":source,**models}.items():
            budget()
            arrays=evaluate_cached(model,features)
            for split in ("train","test"):
                for key in ("labels","cell_ids"):
                    arrays[f"{split}_{key}"]=metadata[f"eval_{split}_{key}"]
            cell_grid_evaluation(model,arrays,torch.device("cpu"))
            if name=="source":
                # No new source root search; reproduce only its saved flow grid.
                with np.load(Path(options.source_run)/"evaluation_arrays.npz",allow_pickle=False) as original:
                    for key in ("flow_g","flow_R","flow_displacement"):
                        arrays[key]=original[key].copy()
                roots=source_summary.get("numerical_fixed_points",[])
            else:
                roots=scalar_flow(model,arrays,torch.device("cpu"))
            final=module_fingerprints(model)
            if final["C"]!=modules["C"]:
                raise RuntimeError("Fixed C fingerprint changed")
            summaries[name]=dict(actual_steps=steps.get(name,0),source_actual_steps=3000,
                C_updates=0,optimizer_steps=counts.get(name,{}),initial_module_sha256=modules,
                final_module_sha256=final,numerical_fixed_points=roots,
                closure={split:closure_metrics(arrays[f"{split}_direct"],arrays[f"{split}_one_step"],
                    arrays[f"{split}_rollout"],arrays["train_direct"]) for split in ("train","test")},
                train_std_by_scale=arrays["train_direct"].std(0,ddof=0).ravel().tolist(),
                thermal_nu=None,critical_r=None)
            all_arrays.update({f"{name}_{key}":value for key,value in arrays.items()})
            if name!="source":
                arm=root/name
                arm.mkdir()
                torch.save(dict(recipe=RECIPE,step=options.steps,source_actual_steps=3000,
                    model=model.state_dict(),optimizers={key:value.state_dict() for key,value in optimizers[name].items()},
                    initial_module_sha256=modules),arm/f"training_step_{options.steps}.pt")
        if state_fingerprint(source)!=source_before or any(p.grad is not None for p in source.parameters()):
            raise RuntimeError("Immutable source model changed")
        for path,digest_before in pins.items():
            if sha256_file(path)!=digest_before:
                raise RuntimeError(f"Immutable source/cache changed: {path}")
        np.savez(root/"evaluation_arrays.npz",**all_arrays)
        history_value=_json(root/"history_progress.json")
        history_value["latest_checkpoint_step"]=options.steps
        write_json(root/"history_progress.json",history_value)
        summary=dict(recipe=RECIPE,schema_version=5,status="complete",actual_steps=options.steps,
            actual_steps_by_arm=steps,arms=summaries,source_pins=pins,
            paired_schedule_sha256=manifest["paired_schedule_sha256"],frozen_source_verified=True,
            source_state_sha256_before=source_before,source_state_sha256_after=state_fingerprint(source),
            source_gradients_none=True,C_updates=0,physical_sampling_calls=0,MI_calls=0,GPU_used=False,
            labels_used_in_training=False,data_warnings=manifest["data_warnings"],
            closure_alpha=closure_alpha,guard=manifest["guard"],default_recipe_modified=False,
            timing=dict(cpu_seconds=time.process_time()-cpu_started,wall_seconds=time.perf_counter()-started),
            thermal_nu=None,critical_r=None,equilibrium_certified=False)
        write_json(root/"summary.json",summary)
        budget()
        status("complete",actual_steps=options.steps,latest_checkpoint_step=options.steps,
            summary_sha256=sha256_file(root/"summary.json"),
            evaluation_arrays_sha256=sha256_file(root/"evaluation_arrays.npz"),
            cpu_seconds=time.process_time()-cpu_started)
        return summary
    except Exception as error:
        hold=isinstance(error,BudgetHold)
        write_json(root/"status.json",dict(recipe=RECIPE,status="budget_hold" if hold else "failed",
            workflow_complete=False,returncode=3 if hold else 1,error=str(error),requires_user_action=True,
            actual_steps_by_arm=steps,cpu_seconds=time.process_time()-cpu_started,
            traceback=traceback.format_exc()))
        if hold:
            return dict(status="budget_hold")
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("source-run","cache","output"):
        parser.add_argument(f"--{name}",required=True)
    for name,default in (("steps",3000),("seed",91),("system-batch",8),("sample-batch",64),("log-every",100)):
        parser.add_argument(f"--{name}",type=int,default=default)
    parser.add_argument("--cpu-budget-seconds",type=float,default=800.)
    result=train(Options(**vars(parser.parse_args())))
    if result["status"]=="budget_hold":
        raise SystemExit(3)


if __name__=="__main__":
    main()
