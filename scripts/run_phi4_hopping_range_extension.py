"""Fresh 15-kappa hopping-phi4 experiment; planning is the default.

Endpoint pilots and a new synthetic timing check are execution evidence only.
No historical data, checkpoint, failed workflow or benchmark is reused.  No
child is killed, retried, resumed, repaired or migrated by this parent.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import math
import numbers
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time

from scripts import run_phi4_hopping_qcr as base

CPU_SECONDS, GPU_SECONDS = 21600., 14400.
PILOT_BURN_IN, PILOT_SAMPLES, PILOT_SPACING = 128, 32, 1
PILOT_MIN_ACCEPTANCE = .6
EVALUATION_REPORT_RESERVE = 500.
MODULE = "scripts.run_phi4_hopping_range_extension"
DEFAULT_CONFIG = "configs/phi4_hopping_b3_g1_kappa032_104_seed91.json"
DEFAULT_SAMPLING_PLAN = "configs/phi4_hopping_kappa032_104_sampling_plan.json"
GPU_STAGES = {"gpu_preflight", "training"}


def _read(path):
    return json.loads(Path(path).read_text())


def _code_root():
    return Path(__file__).resolve().parents[1]


def _finite_json(value):
    """Retain nonfinite failure evidence as null; never turn its gate into a pass."""
    if isinstance(value,dict):
        return {key:_finite_json(item) for key,item in value.items()}
    if isinstance(value,(tuple,list)):
        return [_finite_json(item) for item in value]
    if isinstance(value,numbers.Real) and not math.isfinite(value):
        return None
    return value


def _real_python_files(code):
    return {str(path.relative_to(code)): path for folder in ("experiments", "recurrent_rg", "scripts")
            for path in (code/folder).rglob("*.py") if not path.name.startswith("._")}


def physical_options(sampling_plan=DEFAULT_SAMPLING_PLAN):
    """Only read the new configuration; its historical provenance is not opened."""
    from recurrent_rg.data.phi4_hopping_refined_pipeline import Phi4HoppingRefinedOptions
    expected = asdict(Phi4HoppingRefinedOptions())
    expected.update(kappa_values=(.32, *Phi4HoppingRefinedOptions.kappa_values, 1.04), test_offset=1)
    values = _read(sampling_plan)
    if values.get("schema_version") != 5 or values.get("sampler_entrypoint") != "scripts.prepare_phi4_hopping_refined":
        raise ValueError("Require the pinned new schema-5 refined-sampler plan")
    options = values.get("options")
    if options != json.loads(json.dumps(expected)):
        raise ValueError("Sampling settings must match all fifteen authorized kappa conditions")
    options = dict(options)
    options.update(sizes=tuple(options["sizes"]), kappa_values=tuple(options["kappa_values"]))
    result = Phi4HoppingRefinedOptions(**options)
    result.validate()
    return result


def pilot_cases(sampling_plan=DEFAULT_SAMPLING_PLAN):
    options, cases = physical_options(sampling_plan), []
    for kappa in (options.kappa_values[0], options.kappa_values[-1]):
        for initial in ("hot", "cold_positive", "cold_negative"):
            token = json.dumps(["phi4_hopping_range_extension_endpoint_pilot_v1", options.seed,
                options.dimension, 36, kappa, options.quartic_lambda, initial])
            cases.append(dict(kappa=kappa, initial=initial,
                seed=int.from_bytes(hashlib.sha256(token.encode()).digest()[:16], "little")))
    return cases


def pilot_plan(sampling_plan=DEFAULT_SAMPLING_PLAN):
    cases = pilot_cases(sampling_plan)
    cycles = (PILOT_BURN_IN+PILOT_SAMPLES*PILOT_SPACING)*len(cases)
    return dict(cases=cases, size=36, dimension=2, quartic_lambda=1., h=0.,
        step_size=.04, leapfrog_steps=24, trajectory_length=.96,
        burn_in=PILOT_BURN_IN, samples=PILOT_SAMPLES, spacing=PILOT_SPACING,
        minimum_production_acceptance=PILOT_MIN_ACCEPTANCE,
        total_hybrid_cycles=cycles, total_force_site_evaluations=cycles*25*36**2,
        seed_purpose_separated_cases=len(cases), arrays_reused_in_production=False,
        model_updates=0, cuda_used=False, adaptation=False, equilibrium_certified=False,
        qualification="仅新增端点的短CPU稳定性/接受率检查，不认证热化；pilot不进入生产；不自适应、不重试")


def pilot(output, sampling_plan=DEFAULT_SAMPLING_PLAN):
    import numpy as np
    from recurrent_rg.data import phi4_hopping as kernel
    from recurrent_rg.data.phi4_hopping_refined_pipeline import WORK_FIELDS, source_pins
    path, definition = Path(output), pilot_plan(sampling_plan)
    if path.exists():
        raise FileExistsError("Endpoint pilot is fresh-only")
    pins, started = source_pins(), time.process_time()
    receipt = dict(status="running", workflow_complete=False, returncode=None, error=None,
        requires_user_action=False, plan=definition, rows=[], source_pins=pins,
        sampling_plan_sha256=base.sha(sampling_plan), started_at=base.now(), equilibrium_certified=False)
    base.save(path, receipt)
    try:
        for index, case in enumerate(definition["cases"]):
            case_start = time.process_time()
            try:
                result = kernel.sample_chain(case["kappa"], 1., size=36, dimension=2,
                    samples=PILOT_SAMPLES, burn_in=PILOT_BURN_IN, spacing=PILOT_SPACING,
                    step_size=.04, leapfrog_steps=24, seed=case["seed"],
                    initial=case["initial"], h=0., cluster_probability=.5)
            except (FloatingPointError,OverflowError) as exc:
                receipt["rows"].append(dict(case,finite=False,numerical_instability=True,
                    cpu_seconds=time.process_time()-case_start,error=f"{type(exc).__name__}: {exc}"))
                receipt.update(status="scientific_stop",workflow_complete=True,returncode=4,
                    requires_user_action=True,numerical_instability=True,
                    error="Endpoint numerical-stability pilot gate failed; zero production/training; no adaptation")
                break
            finite = all(np.isfinite(result[name]).all() for name in
                ("configurations", "raw_observables", "delta_h", "trajectory_magnetization"))
            acceptance = float(result["production_acceptance"])
            row = dict(case, finite=finite, production_acceptance=acceptance,
                burn_in_acceptance=float(result["burn_in_acceptance"]),
                max_absolute_delta_h=float(result["max_absolute_delta_h"]),
                accepted=np.asarray(result["accepted"]).tolist(), delta_h=np.asarray(result["delta_h"]).tolist(),
                diagnostics=result["diagnostics"], warnings=result["warnings"],
                cpu_seconds=time.process_time()-case_start,
                actual_work={name: int(result[name]) for name in WORK_FIELDS})
            receipt["rows"].append(_finite_json(row))
            receipt.update(completed_cases=index+1, cpu_seconds=time.process_time()-started)
            base.save(path, receipt)
            print(json.dumps(_finite_json(dict(event="endpoint_pilot_chain_complete", case=index,
                kappa=case["kappa"], initial=case["initial"], acceptance=acceptance))), flush=True)
            if not finite or not math.isfinite(acceptance) or acceptance < PILOT_MIN_ACCEPTANCE:
                receipt.update(status="scientific_stop", workflow_complete=True, returncode=4,
                    numerical_instability=(not finite or not math.isfinite(acceptance)),
                    requires_user_action=True, error="Endpoint finite/acceptance gate failed; zero production/training; no integrator adaptation")
                break
        else:
            if source_pins() != pins or base.sha(sampling_plan) != receipt["sampling_plan_sha256"]:
                raise RuntimeError("Pinned endpoint pilot source/configuration changed")
            receipt.update(status="complete", workflow_complete=True, returncode=0,
                           requires_user_action=False, error=None)
    except Exception as exc:
        receipt.update(status="failed", workflow_complete=False, returncode=1,
                       requires_user_action=True, error=f"{type(exc).__name__}: {exc}")
    receipt.update(cpu_seconds=time.process_time()-started, ended_at=base.now())
    base.save(path, receipt)
    return receipt["returncode"]


def validate_pilot(root, sampling_plan=DEFAULT_SAMPLING_PLAN):
    from recurrent_rg.data.phi4_hopping_refined_pipeline import generation_plan, source_pins
    receipt, definition = _read(Path(root)/"sampler_pilot.json"), pilot_plan(sampling_plan)
    rows = receipt.get("rows", [])
    if (receipt.get("status") != "complete" or receipt.get("workflow_complete") is not True
            or receipt.get("returncode") != 0 or receipt.get("error") is not None
            or receipt.get("requires_user_action") is not False or receipt.get("source_pins") != source_pins()
            or receipt.get("sampling_plan_sha256") != base.sha(sampling_plan)
            or receipt.get("plan") != definition or len(rows) != len(definition["cases"])
            or any({key: row.get(key) for key in case} != case for row, case in zip(rows, definition["cases"]))
            or any(not row.get("finite") or not math.isfinite(row.get("production_acceptance", math.nan))
                   or row["production_acceptance"] < PILOT_MIN_ACCEPTANCE
                   or not math.isfinite(row.get("cpu_seconds", math.nan)) or row["cpu_seconds"] <= 0
                   or row.get("actual_work", {}).get("hybrid_cycles") != 160
                   or row.get("actual_work", {}).get("hmc_trajectories") != 160
                   or row.get("actual_work", {}).get("force_site_evaluations") != 5184000 for row in rows)):
        raise RuntimeError("Incomplete fixed six-case endpoint pilot contract")
    # Slowest measured endpoint rate with 2x headroom. The interior is not
    # timed by this short pilot: this estimate is explicitly not a guarantee.
    work = generation_plan(physical_options(sampling_plan))["total_hybrid_cycles"]
    projection = 2*max(row["cpu_seconds"] for row in rows)*work/160
    return dict(sampler_pilot_sha256=base.sha(Path(root)/"sampler_pilot.json"),
        pilot_min_production_acceptance=min(row["production_acceptance"] for row in rows),
        remaining_production_cpu_projection_seconds=projection,
        production_cpu_projection=dict(seconds=projection, conservative_factor=2.,
            rate="slowest of six endpoint cases", production_hybrid_cycles=work,
            interior_throughput_measured=False, estimate_not_guarantee=True))


def benchmark_projection(initialization_seconds, timed_updates):
    if (not math.isfinite(initialization_seconds) or initialization_seconds < 0
            or len(timed_updates) != 5 or any(not math.isfinite(x) or x <= 0 for x in timed_updates)):
        raise ValueError("Require separate finite initialization and all five timed updates")
    return 2*initialization_seconds+1.5*max(timed_updates)*3000+EVALUATION_REPORT_RESERVE


def benchmark(output, gpu_uuid):
    """New synthetic same-model/same-batch seven-step check, never science."""
    started = time.perf_counter()
    import torch
    from experiments.phi4_hopping_qcr import (BudgetHold, _device_receipt, independent_step,
        initialize_q_decoder, module_fingerprints, optimizer_step_counts)
    from recurrent_rg.phi4_aligned_qcr import AlignedPhi4QCR
    path = Path(output)
    if path.exists():
        raise FileExistsError("Synthetic check is fresh-only")
    receipt = dict(status="running", synthetic=True, physical_result=False,
        checkpoint_saved=False, formal_model_reused=False, physical_data_read=False,
        training_batch=[32, 64, 36, 36, 1], initialization_samples_per_chain=128,
        warmup_updates=2, timed_updates=5, error=None, returncode=None,
        training_cap_seconds=GPU_SECONDS, evaluation_report_reserve_seconds=EVALUATION_REPORT_RESERVE)
    try:
        from scripts.report_phi4_hopping_qcr import select_chinese_font
        receipt["chinese_font"] = str(select_chinese_font(os.environ.get("QCR_ZH_FONT")))
        torch.set_num_threads(1)
        torch.manual_seed(91)
        device = torch.device("cuda:0")
        receipt["device"] = _device_receipt(device, gpu_uuid)
        torch.cuda.reset_peak_memory_stats(device)
        fields = torch.randn((32, 128, 36, 36, 1), dtype=torch.float64,
            generator=torch.Generator().manual_seed(20261031))
        model = AlignedPhi4QCR(float(fields.square().mean().sqrt())).to(device)
        constructor = module_fingerprints(model)
        initialize_q_decoder(model, fields, list(range(32)), seed=91)
        optimizers = {name: torch.optim.Adam(getattr(model, name).parameters(), lr=.001)
                      for name in ("Q", "C", "R")}
        batch = fields[:, :64].to(device)
        generator = torch.Generator(device=device).manual_seed(293)
        torch.cuda.synchronize(device)
        initialization_seconds = time.perf_counter()-started
        times = []
        for index in range(7):
            torch.cuda.synchronize(device)
            update_started = time.perf_counter()
            metrics = independent_step(model, batch, optimizers, generator=generator, information_pairs=128)
            torch.cuda.synchronize(device)
            if index >= 2:
                times.append(time.perf_counter()-update_started)
            peak = max(torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device))
            if peak > base.PEAK_BYTES:
                raise BudgetHold("Synthetic own CUDA peak exceeds 8 GiB")
        projection = benchmark_projection(initialization_seconds, times)
        passed = projection <= GPU_SECONDS
        receipt.update(status="complete" if passed else "budget_hold", returncode=0 if passed else 3,
            workflow_complete=passed, requires_user_action=not passed, actual_updates=7,
            optimizer_steps=optimizer_step_counts(optimizers, 7),
            constructor_module_sha256=constructor, final_module_sha256=module_fingerprints(model),
            initialization_seconds=initialization_seconds, timed_update_seconds=times,
            conservative_full_training_projection_seconds=projection,
            initialization_not_multiplied_by_training_steps=True, projection_factor=1.5,
            initialization_factor=2., peak_cuda_bytes=int(peak), last_metrics=metrics)
    except Exception as exc:
        hold = isinstance(exc, BudgetHold)
        receipt.update(status="budget_hold" if hold else "failed", workflow_complete=False,
            returncode=3 if hold else 1, requires_user_action=True, error=f"{type(exc).__name__}: {exc}")
    receipt.update(wall_seconds=time.perf_counter()-started, ended_at=base.now())
    base.save(path, receipt)
    return receipt["returncode"]


def validate_benchmark(root, gpu_uuid):
    receipt = _read(Path(root)/"gpu_preflight.json")
    clocks = receipt.get("optimizer_steps", {})
    projection = benchmark_projection(receipt.get("initialization_seconds", math.nan),
                                      receipt.get("timed_update_seconds", []))
    if (receipt.get("status") != "complete" or receipt.get("workflow_complete") is not True
            or receipt.get("returncode") != 0 or receipt.get("error") is not None
            or receipt.get("requires_user_action") is not False or receipt.get("synthetic") is not True
            or receipt.get("physical_result") is not False or receipt.get("physical_data_read") is not False
            or receipt.get("checkpoint_saved") is not False or receipt.get("formal_model_reused") is not False
            or receipt.get("actual_updates") != 7 or receipt.get("warmup_updates") != 2
            or receipt.get("timed_updates") != 5 or receipt.get("training_batch") != [32,64,36,36,1]
            or receipt.get("device", {}).get("CUDA_VISIBLE_DEVICES") != gpu_uuid
            or receipt.get("peak_cuda_bytes", base.PEAK_BYTES+1) > base.PEAK_BYTES
            or receipt.get("training_cap_seconds") != GPU_SECONDS
            or receipt.get("evaluation_report_reserve_seconds") != EVALUATION_REPORT_RESERVE
            or receipt.get("conservative_full_training_projection_seconds") != projection
            or projection > GPU_SECONDS or set(clocks) != {"Q","C","R"}
            or any(row.get("min") != 7 or row.get("max") != 7 or row.get("parameters",0) <= 0
                   for row in clocks.values())):
        raise RuntimeError("New synthetic check returned zero without the complete benchmark contract")
    return projection


def sampling_command(root, options):
    command = [sys.executable, "-u", "-m", "scripts.prepare_phi4_hopping_refined", "--execute",
               "--output", str(Path(root)/"data")]
    for key, value in asdict(options).items():
        if key == "cluster_probability":
            if value != .5:
                raise ValueError("Refined CLI fixes cluster probability .5")
            continue
        command.append("--"+key.replace("_","-"))
        command.extend(str(x) for x in value) if isinstance(value,(tuple,list)) else command.append(str(value))
    return command


def plan(output, config=DEFAULT_CONFIG, gpu_index=None, gpu_uuid=None, sampling_plan=DEFAULT_SAMPLING_PLAN):
    from experiments.phi4_hopping_qcr import TrainingOptions, load_config, training_plan
    from recurrent_rg.data.phi4_hopping_refined_pipeline import generation_plan
    root, code = Path(output).resolve(), _code_root()
    config, sampling_plan = Path(config).resolve(), Path(sampling_plan).resolve()
    if root.exists() or root == code or root in code.parents or code in root.parents:
        raise ValueError("Require fresh output separate from frozen code; no resume or overwrite")
    if type(gpu_index) is not int or gpu_index < 0 or not isinstance(gpu_uuid,str) or not gpu_uuid.startswith("GPU-"):
        raise ValueError("Require explicit physical GPU index and UUID; no selection or migration")
    options = physical_options(sampling_plan)
    values = load_config(config)
    values.update(data=str(root/"data"),output=str(root/"training"),device="cuda:0",expected_gpu_uuid=gpu_uuid)
    expected = dict(steps=3000,system_batch=32,sample_batch=64,seed=91,threads=1,log_every=100,
        checkpoint_every=1000,information_pairs=128,max_training_seconds=GPU_SECONDS,max_cuda_bytes=base.PEAK_BYTES)
    training_options = TrainingOptions(**values)
    if any(getattr(training_options,name) != value for name,value in expected.items()):
        raise ValueError("Require unchanged aligned 3000-step seed91 model/loss/training contract")
    commands = dict(sampler_pilot=[sys.executable,"-u","-m",MODULE,"--pilot-output",
            str(root/"sampler_pilot.json"),"--sampling-plan",str(sampling_plan)],
        sampling=sampling_command(root,options),
        gpu_preflight=[sys.executable,"-u","-m",MODULE,"--benchmark-output",
            str(root/"gpu_preflight.json"),"--gpu-uuid",gpu_uuid],
        training=[sys.executable,"-u","-m","recurrent_rg","train","--config",
            str(root/"training_config.json"),"--output",str(root/"training"),"--device","cuda:0"])
    physical, pilot_definition = generation_plan(options), pilot_plan(sampling_plan)
    return dict(schema_version=5,created_at=base.now(),execution_requested=False,host=base.HOST,
        code_root=str(code),output=str(root),GPU=dict(index=gpu_index,uuid=gpu_uuid),
        config_source=str(config),config_source_sha256=base.sha(config),
        sampling_plan_source=str(sampling_plan),sampling_plan_sha256=base.sha(sampling_plan),
        physical=physical,pilot=pilot_definition,training=training_plan(training_options),
        work_accounting=dict(production_force_site_evaluations=physical["total_force_site_evaluations"],
            production_force_site_cap=options.max_force_site_evaluations,
            separate_endpoint_pilot_force_site_evaluations=pilot_definition["total_force_site_evaluations"],
            total_including_endpoint_pilot_force_site_evaluations=(physical["total_force_site_evaluations"]
                +pilot_definition["total_force_site_evaluations"]),
            four_billion_is_production_dataset_cap_not_entire_workflow_cap=True),
        stages=list(commands),commands=commands,
        benchmark=dict(synthetic=True,physical_result=False,warmup_updates=2,timed_updates=5,
            projection_factor=1.5,initialization_factor=2.,evaluation_report_reserve_seconds=EVALUATION_REPORT_RESERVE,
            training_batch=[32,64,36,36,1],no_physical_sampling=True,no_checkpoint_saved=True),
        resources=dict(cpu_seconds=CPU_SECONDS,gpu_card_seconds=GPU_SECONDS,max_training_seconds=GPU_SECONDS,
            max_cuda_bytes=base.PEAK_BYTES,directory_bytes=base.DIRECTORY_BYTES,cpu_threads=1,nice=10,
            min_memory_bytes=36*1024**3,max_load_per_cpu=1.25,max_io_full_avg10=20.,
            idle_check_spacing_seconds=10.,min_gpu_free_bytes=12*1024**3,require_no_compute_processes=True),
        historical_data_read=False,historical_model_reused=False,historical_benchmark_reused=False,
        original_thirteen_conditions_use_same_parameter_keyed_seed=True,
        resampled_original_conditions_are_not_new_independent_physical_replicates=True,
        production_is_fresh=True,no_resume=True,no_retry=True,no_auto_repair=True,no_migration=True,
        qualification="范围扩展的独立新实验；短pilot不证明平衡；采样警告原样保留；时间投影不保证墙时；不认证临界指数")


def validate_sampling(root, sampling_plan=DEFAULT_SAMPLING_PLAN):
    from recurrent_rg.data.phi4_hopping_loader import load_dataset
    from recurrent_rg.data.phi4_hopping_refined_pipeline import generation_plan
    folder = Path(root)/"data"
    expected = json.loads(json.dumps(generation_plan(physical_options(sampling_plan))))
    manifest, actual_plan = _read(folder/"manifest.json"), _read(folder/"plan.json")
    if (actual_plan != expected or manifest.get("options") != expected["options"]
            or manifest.get("source_pins") != expected["source_pins"]
            or manifest.get("total_hmc_trajectories") != expected["total_hmc_trajectories"]
            or manifest.get("total_force_site_evaluations") != expected["total_force_site_evaluations"]):
        raise RuntimeError("Production plan/work/source differs from the authorized fifteen-condition dataset")
    dataset = load_dataset(folder)  # Strict hashes, all shards and configuration-only schema.
    if (dataset.configurations.shape != (60,128,36,36,1)
            or len(dataset.train_indices) != 40 or len(dataset.test_indices) != 20):
        raise RuntimeError("Wrong fifteen-condition dataset or whole-cell TRAIN/TEST split")
    result = dict(data_manifest_sha256=dataset.manifest_sha256,data_warning_count=dataset.warning_count)
    del dataset
    quality = _read(folder/"quality_summary.json")
    rows = quality.get("conditions",[])
    acceptances = [value for row in rows for value in row.get("production_acceptance",[])]
    if (len(rows) != 15 or len(acceptances) != 60 or
            any(not math.isfinite(value) or value <= 0 for value in acceptances)):
        raise RuntimeError("Zero/nonfinite production acceptance or incomplete conditions; no training")
    result["quality_warnings_retained"] = True
    return result


def budget_ok(cpu_seconds, gpu_seconds, directory_bytes, *, cpu_projection=0., gpu_projection=0.):
    values = (cpu_seconds,gpu_seconds,directory_bytes,cpu_projection,gpu_projection)
    return (all(math.isfinite(x) and x >= 0 for x in values)
            and cpu_seconds+cpu_projection+gpu_projection <= CPU_SECONDS
            and gpu_seconds+gpu_projection <= GPU_SECONDS and directory_bytes <= base.DIRECTORY_BYTES)


def _public_files(root, *, completed=False):
    names = ["job.json","plan.json","training_config.json"]
    names += [name for name in ("sampler_pilot.json","gpu_preflight.json") if (root/name).is_file()]
    public = {name:base.sha(root/name) for name in names}
    if completed:
        for name in _read(root/"data/manifest.json")["public_files"]:
            path = (root/"data"/name).resolve()
            if root/"data" not in path.parents or path.suffix != ".json":
                raise RuntimeError("Invalid public physical metadata path")
            public[f"data/{name}"] = base.sha(path)
        public["training/status.json"] = base.sha(root/"training/status.json")
        for row in _read(root/"training/status.json")["public_files"]:
            path = (root/"training"/row["path"]).resolve()
            if (root/"training" not in path.parents or path.suffix not in (".json",".png",".npz")
                    or base.sha(path) != row["sha256"]):
                raise RuntimeError("Invalid public training artifact binding")
            public[f"training/{row['path']}"] = row["sha256"]
    base.save(root/"public_files.json",public)


def run(output, config=DEFAULT_CONFIG, gpu_index=None, gpu_uuid=None, sampling_plan=DEFAULT_SAMPLING_PLAN):
    cpu_start, child_start = time.process_time(), resource.getrusage(resource.RUSAGE_CHILDREN)
    definition = plan(output,config,gpu_index,gpu_uuid,sampling_plan)
    if socket.gethostname() != base.HOST or os.getpriority(os.PRIO_PROCESS,0) != 10:
        raise ValueError("Execution requires pinned 225 host and exactly one inherited nice10 layer")
    root, code = Path(output).resolve(), _code_root()
    pins = {str(path):base.sha(path) for path in _real_python_files(code).values()}
    pins.update({str(Path(path).resolve()):base.sha(path) for path in (config,sampling_plan)})
    root.mkdir(parents=True,exist_ok=False)
    (root/"logs").mkdir()
    configuration = copy.deepcopy(_read(config))
    configuration.update(data=str(root/"data"),output=str(root/"training"))
    configuration["training"].update(device="cuda:0",expected_gpu_uuid=gpu_uuid)
    base.save(root/"training_config.json",configuration)
    pins[str(root/"training_config.json")] = base.sha(root/"training_config.json")
    base.save(root/"plan.json",dict(definition,execution_requested=True,source_sha256=pins))
    pins[str(root/"plan.json")] = base.sha(root/"plan.json")
    job = dict(status="prepared",workflow_complete=False,parent_pid=os.getpid(),current=None,
        returncode=None,error=None,requires_user_action=False,started_at=base.now(),stages=[],
        plan_sha256=base.sha(root/"plan.json"),cpu_seconds=0.,gpu_seconds=0.,formal_training_started=False,
        formal_optimizer_updates=0,historical_data_read=False,source_sha256=pins)
    gpu_seconds, active, active_stage = 0., None, None

    def checkpoint():
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        job["cpu_seconds"] = (time.process_time()-cpu_start+children.ru_utime-child_start.ru_utime
                             +children.ru_stime-child_start.ru_stime)
        if active is not None and active.poll() is None:
            try:
                fields = Path(f"/proc/{active.pid}/stat").read_text().split(")",1)[1].split()
                job["cpu_seconds"] += (int(fields[11])+int(fields[12]))/os.sysconf("SC_CLK_TCK")
            except (FileNotFoundError,ProcessLookupError):
                pass
        job["gpu_seconds"] = gpu_seconds+(time.monotonic()-active_stage["monotonic_start"]
            if active is not None and active.poll() is None and active_stage["name"] in GPU_STAGES else 0.)
        job["directory_bytes"] = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
        base.save(root/"job.json",job)

    def verify():
        if any(base.sha(path) != digest for path,digest in pins.items()):
            raise RuntimeError("Pinned source/configuration/evidence changed; no repair/restart")

    def hold(message):
        job.update(status="budget_hold",workflow_complete=False,current=None,returncode=3,
                   requires_user_action=True,error=message,ended_at=base.now())
        checkpoint()
        _public_files(root)
        return 3

    def wait_resources(name):
        nonlocal gpu_seconds
        gpu = name in GPU_STAGES
        job.update(status="waiting_resources",current=name)
        while True:
            first = dict(host=base.host_resources())
            if gpu:
                first["gpu"] = base.gpu_resources(gpu_index,gpu_uuid)
            job["resources"] = first
            checkpoint()
            if not budget_ok(job["cpu_seconds"],job["gpu_seconds"],job["directory_bytes"]):
                return False
            if not base.ready(first,gpu):
                time.sleep(30)
                continue
            if not gpu:
                return True
            started = time.monotonic()
            time.sleep(10)
            second = dict(host=base.host_resources(),gpu=base.gpu_resources(gpu_index,gpu_uuid))
            spacing = time.monotonic()-started
            gpu_seconds += spacing
            job["gpu_idle_checks"] = dict(first=first,second=second,spacing_seconds=spacing)
            checkpoint()
            if spacing >= 10 and base.ready(second,True):
                return True

    checkpoint()
    try:
        from scripts.report_phi4_hopping_qcr import select_chinese_font
        job["chinese_font"] = str(select_chinese_font(os.environ.get("QCR_ZH_FONT")))
        for name,command in definition["commands"].items():
            verify()
            if not wait_resources(name):
                return hold("Registered budget exhausted while waiting; no next stage")
            verify()
            checkpoint()
            projection = job.get("training_projection_seconds",0.) if name == "training" else 0.
            cpu_projection = job.get("remaining_production_cpu_projection_seconds",0.) if name == "sampling" else 0.
            if not budget_ok(job["cpu_seconds"],job["gpu_seconds"],job["directory_bytes"],
                             cpu_projection=cpu_projection,gpu_projection=projection):
                return hold("Actual cumulative use plus remaining conservative projection exceeds registered budget")
            stage = dict(name=name,status="running",returncode=None,started_at=base.now(),
                         command=command,monotonic_start=time.monotonic())
            active_stage = stage
            job["stages"].append(stage)
            job.update(status="running",current=name)
            if name == "training":
                job["formal_training_started"] = True
            with (root/"logs"/f"{name}.log").open("xb") as log:
                active = subprocess.Popen(command,cwd=code,
                    env=base.child_environment(gpu_uuid if name in GPU_STAGES else None),
                    stdout=log,stderr=subprocess.STDOUT)
                stage["pid"] = active.pid
                while active.poll() is None:
                    child_path = (root/"sampler_pilot.json" if name == "sampler_pilot" else
                        root/"gpu_preflight.json" if name == "gpu_preflight" else
                        root/("data" if name == "sampling" else "training")/"status.json")
                    try:
                        stage["child_status"] = _read(child_path)
                        if name == "training":
                            job["formal_optimizer_updates"] = stage["child_status"].get("actual_steps",0)
                    except (FileNotFoundError,json.JSONDecodeError):
                        pass
                    checkpoint()
                    if not budget_ok(job["cpu_seconds"],job["gpu_seconds"],job["directory_bytes"]):
                        job.update(requires_user_action=True,
                            budget_exceeded_with_child_running=True,
                            budget_note="Running child preserved; no kill/restart. No further stage will be admitted.")
                    time.sleep(5)
                rc = active.returncode
            child_path = (root/"sampler_pilot.json" if name == "sampler_pilot" else
                root/"gpu_preflight.json" if name == "gpu_preflight" else
                root/("data" if name == "sampling" else "training")/"status.json")
            try:
                stage["child_status"] = _read(child_path)
            except (FileNotFoundError,json.JSONDecodeError):
                pass
            elapsed = time.monotonic()-stage["monotonic_start"]
            if name in GPU_STAGES:
                gpu_seconds += elapsed
            state = "complete" if rc == 0 else "budget_hold" if rc == 3 else "scientific_stop" if rc == 4 else "failed"
            stage.update(status=state,returncode=rc,ended_at=base.now(),wall_seconds=elapsed)
            active, active_stage = None,None
            verify()
            if rc:
                job.update(status=state,workflow_complete=(name == "sampler_pilot" and rc == 4),
                    current=None,returncode=rc,requires_user_action=True,
                    error=f"{name} exited {rc}; completed outputs preserved; no retry/adaptation",ended_at=base.now())
                checkpoint()
                _public_files(root)
                return rc
            if name == "sampler_pilot":
                job.update(validate_pilot(root,sampling_plan))
                job["actual_endpoint_pilot_force_site_evaluations"] = sum(
                    row["actual_work"]["force_site_evaluations"] for row in _read(root/"sampler_pilot.json")["rows"])
                pins[str(root/"sampler_pilot.json")] = base.sha(root/"sampler_pilot.json")
            elif name == "sampling":
                job.update(validate_sampling(root,sampling_plan))
                job["actual_production_force_site_evaluations"] = _read(root/"data/manifest.json")["total_force_site_evaluations"]
                job["actual_total_force_site_evaluations_including_endpoint_pilot"] = (
                    job["actual_production_force_site_evaluations"]+job["actual_endpoint_pilot_force_site_evaluations"])
                job["remaining_production_cpu_projection_seconds"] = 0.
                pins[str(root/"data/manifest.json")] = base.sha(root/"data/manifest.json")
            elif name == "gpu_preflight":
                job["training_projection_seconds"] = validate_benchmark(root,gpu_uuid)
                pins[str(root/"gpu_preflight.json")] = base.sha(root/"gpu_preflight.json")
            else:
                base.validate_training(root,job["data_manifest_sha256"])
                job["formal_optimizer_updates"] = 3000
            checkpoint()
        if not budget_ok(job["cpu_seconds"],job["gpu_seconds"],job["directory_bytes"]):
            return hold("Completed outputs exceed registered cumulative resources")
        job.update(status="complete",workflow_complete=True,current=None,returncode=0,
                   error=None,requires_user_action=False,ended_at=base.now())
        checkpoint()
        _public_files(root,completed=True)
        return 0
    except BaseException as exc:
        if active is not None and active.poll() is None:
            job.update(status="waiting_running_child_after_parent_error",requires_user_action=True,
                       error=f"{type(exc).__name__}: {exc}")
            checkpoint()
            active.wait()  # Never kill a running child or anybody else's process.
        job.update(status="failed",workflow_complete=False,current=None,returncode=1,
                   requires_user_action=True,error=f"{type(exc).__name__}: {exc}",ended_at=base.now())
        checkpoint()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--config",default=DEFAULT_CONFIG)
    parser.add_argument("--sampling-plan",default=DEFAULT_SAMPLING_PLAN)
    parser.add_argument("--gpu-index",type=int)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--pilot-output")
    parser.add_argument("--benchmark-output")
    parser.add_argument("--execute",action="store_true")
    args = parser.parse_args(argv)
    if args.pilot_output is not None:
        if args.output or args.benchmark_output or args.execute or args.gpu_index is not None or args.gpu_uuid:
            parser.error("Internal CPU pilot cannot be combined with parent/GPU arguments")
        return pilot(args.pilot_output,args.sampling_plan)
    if args.benchmark_output is not None:
        if args.output or args.execute or args.gpu_index is not None or not args.gpu_uuid:
            parser.error("Internal benchmark requires only output and pinned GPU UUID")
        return benchmark(args.benchmark_output,args.gpu_uuid)
    if not args.output or args.gpu_index is None or not args.gpu_uuid:
        parser.error("Require --output, --gpu-index and --gpu-uuid; dry unless --execute")
    values = (args.output,args.config,args.gpu_index,args.gpu_uuid,args.sampling_plan)
    if args.execute:
        return run(*values)
    print(json.dumps(plan(*values),ensure_ascii=False,indent=2,allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
