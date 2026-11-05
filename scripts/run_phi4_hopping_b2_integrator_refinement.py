"""Bounded, explicitly authorized L128 integrator refinement; dry by default.

Exactly three fixed candidates are tried in order. A finite acceptance failure
may advance to the next candidate; numerical/execution failures never do. The
first fifteen-case pass is frozen before a cumulative budget gate and fresh
production/training. No historical model/data, benchmark or optimizer resumes.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time

from scripts import run_phi4_hopping_qcr as base
from scripts import run_phi4_hopping_b2_large as large
from scripts import continue_phi4_hopping_b2_large_budget as previous

MODULE = "scripts.run_phi4_hopping_b2_integrator_refinement"
CANDIDATES = ((.02, 48), (.01, 96), (.005, 192))
CPU_SECONDS, GPU_SECONDS, DIRECTORY_BYTES = 86400., 57600., 2*1024**3
GPU_UUID = previous.GPU_UUID
STOPPED_SHA256 = dict(
    job="ecc7aa6e99647d7e0db69018b157fbcf236c647e1553fe1b17f6a0765e29cb8e",
    plan="c466dfdede8cd52240f1cbf9e651ffbdf3c2620d3f21f9d2573c6ac9583b3065",
    sampler_pilot="a91e604adfe46f108d8d293982245ddc49265ae52bec4efb98ac40a6f68f4513")
NEW_SOURCES = {"scripts/run_phi4_hopping_b2_integrator_refinement.py",
    "scripts/prepare_phi4_hopping_b2_refined.py", "recurrent_rg/data/phi4_hopping_b2_refined_pipeline.py"}


def _read(path):
    return json.loads(Path(path).read_text())


def _code_root():
    return Path(__file__).resolve().parents[1]


def budget_ok(cpu, gpu, size, *, cpu_projection=0., gpu_projection=0.):
    return (all(math.isfinite(x) and x >= 0 for x in (cpu,gpu,size,cpu_projection,gpu_projection))
        and cpu+cpu_projection+gpu_projection <= CPU_SECONDS
        and gpu+gpu_projection <= GPU_SECONDS and size <= DIRECTORY_BYTES)


def refinement_options(step_size, leapfrog_steps, sampling_plan):
    from recurrent_rg.data.phi4_hopping_b2_refined_pipeline import Phi4HoppingB2RefinedOptions
    if (step_size,leapfrog_steps) not in CANDIDATES:
        raise ValueError("Integrator pair is outside the three authorized candidates")
    original = asdict(large.physical_options(sampling_plan))
    original.update(step_size=step_size,leapfrog_steps=leapfrog_steps)
    # The refined wrapper fixes only an adequate force-work cap; other caps and
    # physical seeds/sample counts remain the original explicitly pinned values.
    original.pop("max_force_site_evaluations",None)
    options = Phi4HoppingB2RefinedOptions(**original)
    options.validate()
    return options


def pilot_plan(sampling_plan):
    cases = large.pilot_cases(sampling_plan)
    per_candidate_cycles = len(cases)*160
    return dict(cases=cases,candidates=[dict(step_size=step,leapfrog_steps=leap,
            trajectory_length=.96) for step,leap in CANDIDATES],size=128,dimension=2,
        quartic_lambda=1.,h=0.,burn_in=128,samples=32,spacing=1,
        minimum_production_acceptance=.6,maximum_candidates=3,maximum_case_executions=45,
        maximum_hybrid_cycles=per_candidate_cycles*3,
        maximum_force_site_evaluations=per_candidate_cycles*128**2*sum(leap+1 for _,leap in CANDIDATES),
        case_seeds_shared_between_integrators=True,original_pilot_seed_policy=True,
        arrays_reused_in_production=False,production_seed_changed=False,model_updates=0,cuda_used=False,
        selection="first complete fifteen-case finite acceptance pass; stop after first pass",
        acceptance_failure_may_advance=True,numerical_or_execution_failure_may_advance=False,
        additional_adaptation=False,equilibrium_certified=False)


def _pilot_pins():
    from recurrent_rg.data.phi4_hopping_b2_refined_pipeline import source_pins
    return {**source_pins(), "scripts/run_phi4_hopping_b2_integrator_refinement.py":base.sha(__file__)}


def pilot_candidate(path, sampling_plan, index, max_cpu_seconds=math.inf):
    """One fixed integrator, original fifteen seeds; only finite rejection is rc4."""
    import numpy as np
    from recurrent_rg.data import phi4_hopping as kernel
    from recurrent_rg.data.phi4_hopping_pipeline import WORK_FIELDS
    path = Path(path)
    if path.exists():
        raise FileExistsError("Candidate pilot is fresh-only")
    definition, pins, started = pilot_plan(sampling_plan), _pilot_pins(), time.process_time()
    step,leap = CANDIDATES[index]
    refinement_options(step,leap,sampling_plan)
    receipt = dict(status="running",workflow_complete=False,returncode=None,error=None,
        requires_user_action=False,started_at=base.now(),candidate_index=index,
        integrator=dict(step_size=step,leapfrog_steps=leap,trajectory_length=.96),
        plan=definition,source_pins=pins,sampling_plan_sha256=base.sha(sampling_plan),rows=[],
        completed_cases=0,actual_force_site_evaluations=0,model_updates=0,
        arrays_reused_in_production=False,finite_acceptance_failure=False,equilibrium_certified=False)
    base.save(path,receipt)
    sampling_call_active=False
    try:
        for number,case in enumerate(definition["cases"]):
            if time.process_time()-started >= max_cpu_seconds:
                receipt.update(status="budget_hold",returncode=3,requires_user_action=True,
                    error="Bounded pilot CPU budget exhausted; no next case or production")
                break
            if _pilot_pins()!=pins or base.sha(sampling_plan)!=receipt["sampling_plan_sha256"]:
                raise RuntimeError("Pinned candidate source or sampling plan changed")
            receipt.update(current_case=number)
            base.save(path,receipt)
            case_start=time.process_time()
            sampling_call_active=True
            result=kernel.sample_chain(case["kappa"],1.,size=128,dimension=2,samples=32,
                burn_in=128,spacing=1,step_size=step,leapfrog_steps=leap,
                seed=case["seed"],initial=case["initial"],h=0.,cluster_probability=.5)
            sampling_call_active=False
            finite=all(np.isfinite(result[name]).all() for name in
                ("configurations","raw_observables","delta_h","trajectory_magnetization"))
            acceptance=float(result["production_acceptance"])
            work={name:int(result[name]) for name in WORK_FIELDS}
            if (work["hybrid_cycles"]!=160 or work["hmc_trajectories"]!=160
                    or work["force_site_evaluations"]!=160*(leap+1)*128**2):
                raise RuntimeError("Pilot actual work differs from the fixed integrator contract")
            row=dict(case,finite=bool(finite and math.isfinite(acceptance)),production_acceptance=acceptance,
                burn_in_acceptance=float(result["burn_in_acceptance"]),
                max_absolute_delta_h=float(result["max_absolute_delta_h"]),
                accepted=np.asarray(result["accepted"]).tolist(),delta_h=np.asarray(result["delta_h"]).tolist(),
                diagnostics=result["diagnostics"],warnings=result["warnings"],
                cpu_seconds=time.process_time()-case_start,actual_work=work)
            receipt["rows"].append(large._finite_json(row))
            receipt.update(completed_cases=number+1,cpu_seconds=time.process_time()-started,
                actual_force_site_evaluations=sum(x["actual_work"]["force_site_evaluations"] for x in receipt["rows"]))
            base.save(path,receipt)
            if (not finite or not math.isfinite(acceptance) or not 0<=acceptance<=1
                    or not math.isfinite(float(result["burn_in_acceptance"]))
                    or not math.isfinite(float(result["max_absolute_delta_h"]))):
                raise FloatingPointError("Nonfinite/invalid pilot result; no integrator fallback or production")
            print(json.dumps(dict(event="integrator_pilot_case_complete",candidate_index=index,
                case=number,step_size=step,leapfrog_steps=leap,kappa=case["kappa"],
                initial=case["initial"],acceptance=acceptance)),flush=True)
            if acceptance < .6:
                receipt.update(status="acceptance_rejected",workflow_complete=True,returncode=4,
                    requires_user_action=False,finite_acceptance_failure=True,
                    error="Finite production acceptance below fixed 0.6 threshold; candidate rejected")
                break
        else:
            receipt.update(status="complete",workflow_complete=True,returncode=0,error=None,
                requires_user_action=False)
        if _pilot_pins()!=pins or base.sha(sampling_plan)!=receipt["sampling_plan_sha256"]:
            raise RuntimeError("Pinned candidate source or sampling plan changed")
    except Exception as exc:
        receipt.update(status="failed",workflow_complete=False,returncode=1,requires_user_action=True,
            finite_acceptance_failure=False,error=f"{type(exc).__name__}: {exc}",
            partial_work_unknown=sampling_call_active,
            force_work_accounting=("completed cases only; interrupted case work unavailable"
                if sampling_call_active else "accounted returned cases; no interrupted kernel call"))
    receipt.update(cpu_seconds=time.process_time()-started,current_case=None,ended_at=base.now())
    base.save(path,receipt)
    return receipt["returncode"]


def pilot_selection(root,sampling_plan,max_cpu_seconds):
    root=Path(root)
    if (root/"integrator_selection.json").exists():
        raise FileExistsError("Integrator selection is fresh-only")
    started=time.process_time()
    selection=dict(status="running",workflow_complete=False,returncode=None,error=None,
        requires_user_action=False,started_at=base.now(),plan=pilot_plan(sampling_plan),
        candidate_receipts=[],selected_candidate=None,actual_force_site_evaluations=0,
        maximum_cpu_seconds=max_cpu_seconds,additional_adaptation=False,training_started=False)
    base.save(root/"integrator_selection.json",selection)
    try:
        for index in range(3):
            path=root/f"pilot_{index:02d}.json"
            rc=pilot_candidate(path,sampling_plan,index,max_cpu_seconds-(time.process_time()-started))
            row=_read(path)
            if row.get("partial_work_unknown"):
                selection.update(partial_work_unknown=True,
                    force_work_accounting="completed cases only; interrupted case work unavailable")
            selection["candidate_receipts"].append(dict(index=index,path=path.name,sha256=base.sha(path),
                status=row["status"],returncode=rc,completed_cases=row["completed_cases"]))
            selection["actual_force_site_evaluations"]+=row["actual_force_site_evaluations"]
            if selection["actual_force_site_evaluations"]>selection["plan"]["maximum_force_site_evaluations"]:
                raise RuntimeError("Authorized aggregate pilot force-work cap exceeded")
            base.save(root/"integrator_selection.json",selection)
            if rc==0:
                selection.update(status="complete",workflow_complete=True,returncode=0,error=None,
                    requires_user_action=False,selected_candidate=index,selected_integrator=row["integrator"])
                break
            if rc!=4 or row.get("finite_acceptance_failure") is not True:
                selection.update(status="budget_hold" if rc==3 else "failed",returncode=rc,
                    workflow_complete=False,requires_user_action=True,error=row["error"])
                break
        else:
            selection.update(status="scientific_stop",workflow_complete=True,returncode=4,
                requires_user_action=True,error="All three preregistered integrators failed fixed acceptance; no production/training")
    except Exception as exc:
        selection.update(status="failed",workflow_complete=False,returncode=1,requires_user_action=True,
            error=f"{type(exc).__name__}: {exc}")
    selection.update(cpu_seconds=time.process_time()-started,ended_at=base.now())
    base.save(root/"integrator_selection.json",selection)
    return selection["returncode"]


def validate_selection(root,sampling_plan):
    from recurrent_rg.data.phi4_hopping_b2_refined_pipeline import generation_plan
    root=Path(root)
    selection=_read(root/"integrator_selection.json")
    index=selection.get("selected_candidate")
    if (selection.get("status")!="complete" or selection.get("workflow_complete") is not True
            or selection.get("returncode")!=0 or selection.get("error") is not None
            or selection.get("requires_user_action") is not False or type(index) is not int or not 0<=index<3
            or selection.get("plan")!=pilot_plan(sampling_plan)
            or len(selection.get("candidate_receipts",[]))!=index+1):
        raise RuntimeError("Incomplete bounded integrator selection contract")
    total=0
    for number,binding in enumerate(selection["candidate_receipts"]):
        path=root/f"pilot_{number:02d}.json"
        row=_read(path)
        step,leap=CANDIDATES[number]
        rows=row.get("rows",[])
        if (binding.get("index")!=number or binding.get("path")!=path.name
                or binding.get("sha256")!=base.sha(path) or row.get("candidate_index")!=number
                or binding.get("status")!=row.get("status") or binding.get("returncode")!=row.get("returncode")
                or binding.get("completed_cases")!=len(rows) or row.get("completed_cases")!=len(rows)
                or row.get("integrator")!=dict(step_size=step,leapfrog_steps=leap,trajectory_length=.96)
                or row.get("plan")!=pilot_plan(sampling_plan) or row.get("source_pins")!=_pilot_pins()
                or row.get("sampling_plan_sha256")!=base.sha(sampling_plan)
                or row.get("workflow_complete") is not True or not rows
                or any({key:r.get(key) for key in case}!=case for r,case in zip(rows,pilot_plan(sampling_plan)["cases"]))
                or any(r.get("finite") is not True or not math.isfinite(r.get("production_acceptance",math.nan))
                    or not math.isfinite(r.get("cpu_seconds",math.nan)) or r["cpu_seconds"]<=0
                    or r.get("actual_work",{}).get("hmc_trajectories")!=160
                    or r.get("actual_work",{}).get("hybrid_cycles")!=160
                    or r.get("actual_work",{}).get("force_site_evaluations")!=160*(leap+1)*128**2 for r in rows)):
            raise RuntimeError("Candidate pilot identity/source/seed/work/finite contract differs")
        if number<index:
            if (row.get("status")!="acceptance_rejected" or row.get("returncode")!=4
                    or row.get("finite_acceptance_failure") is not True or len(rows)>15
                    or any(r["production_acceptance"]<.6 for r in rows[:-1])
                    or rows[-1]["production_acceptance"]>=.6):
                raise RuntimeError("Only finite threshold rejection may precede selected candidate")
        elif (row.get("status")!="complete" or row.get("returncode")!=0 or row.get("error") is not None
                or row.get("requires_user_action") is not False or len(rows)!=15
                or any(r["production_acceptance"]<.6 for r in rows)):
            raise RuntimeError("Selected candidate did not pass all fifteen fixed cases")
        total+=sum(r["actual_work"]["force_site_evaluations"] for r in rows)
    if (selection.get("selected_integrator")!=row["integrator"]
            or selection.get("actual_force_site_evaluations")!=total
            or total>pilot_plan(sampling_plan)["maximum_force_site_evaluations"]):
        raise RuntimeError("Integrator selection aggregate force accounting differs")
    options=refinement_options(*CANDIDATES[index],sampling_plan)
    physical=json.loads(json.dumps(generation_plan(options)))
    projection=2*max(r["cpu_seconds"] for r in rows)*physical["total_hybrid_cycles"]/160
    return dict(selected_candidate=index,selected_integrator=asdict(options),selected_physical_plan=physical,
        integrator_selection_sha256=base.sha(root/"integrator_selection.json"),
        actual_endpoint_pilot_force_site_evaluations=total,remaining_production_cpu_projection_seconds=projection,
        production_cpu_projection=dict(seconds=projection,conservative_factor=2.,
            rate="slowest of fifteen passing fixed cases",estimate_not_guarantee=True))


def prerequisites(stopped_root,preflight_root,config,sampling_plan,gpu_index,gpu_uuid):
    stopped,preflight,code=Path(stopped_root).resolve(),Path(preflight_root).resolve(),_code_root()
    if stopped==preflight or gpu_index!=3 or gpu_uuid!=GPU_UUID:
        raise ValueError("Require distinct stopped/preflight roots and pinned GPU3")
    for name,digest in STOPPED_SHA256.items():
        if base.sha(stopped/f"{name}.json")!=digest:
            raise ValueError("Stopped workflow terminal evidence fingerprint differs")
    job,old_plan,pilot=[_read(stopped/f"{name}.json") for name in ("job","plan","sampler_pilot")]
    stages=job.get("stages",[])
    old_pilot_rows=pilot.get("rows",[])
    if (job.get("status")!="scientific_stop" or job.get("workflow_complete") is not True
            or job.get("returncode")!=4 or job.get("current") is not None
            or job.get("requires_user_action") is not True or job.get("formal_training_started") is not False
            or job.get("formal_optimizer_updates")!=0 or len(stages)!=1 or stages[0].get("name")!="sampler_pilot"
            or stages[0].get("status")!="scientific_stop" or stages[0].get("returncode")!=4
            or job.get("plan_sha256")!=base.sha(stopped/"plan.json")
            or old_plan.get("host")!=base.HOST or old_plan.get("output")!=str(stopped)
            or old_plan.get("GPU")!=dict(index=3,uuid=GPU_UUID)
            or pilot.get("status")!="scientific_stop" or pilot.get("workflow_complete") is not True
            or pilot.get("returncode")!=4 or pilot.get("numerical_instability") is not False
            or len(old_pilot_rows)!=2 or any(r.get("finite") is not True for r in old_pilot_rows)
            or old_pilot_rows[1].get("initial")!="cold_positive" or old_pilot_rows[1].get("production_acceptance")!=0
            or any((stopped/name).exists() or (preflight/name).exists() for name in ("data","training"))):
        raise ValueError("Prior workflow is not the finite cold-acceptance scientific stop before production")
    inherited=old_plan["budget_extension"]
    if (inherited.get("prior_root")!=str(preflight)
            or inherited.get("prior_plan_sha256")!=base.sha(preflight/"plan.json")
            or inherited.get("prior_job_sha256")!=base.sha(preflight/"job.json")
            or base.sha(preflight/"gpu_preflight.json")!=previous.PREFLIGHT_SHA256
            or base.sha(stopped/"reused_preflight.json")!=previous.PREFLIGHT_SHA256):
        raise ValueError("Original seven-step preflight is not immutably bound to stopped workflow")
    receipt=_read(preflight/"gpu_preflight.json")
    projection=large.benchmark_projection(receipt.get("initialization_seconds",math.nan),receipt.get("timed_update_seconds",[]))
    clocks=receipt.get("optimizer_steps",{})
    if (receipt.get("status")!="budget_hold" or receipt.get("returncode")!=3
            or receipt.get("workflow_complete") is not False or receipt.get("error") is not None
            or receipt.get("actual_updates")!=7 or receipt.get("warmup_updates")!=2 or receipt.get("timed_updates")!=5
            or receipt.get("training_cap_seconds")!=14400. or receipt.get("synthetic") is not True
            or receipt.get("physical_result") is not False or receipt.get("physical_data_read") is not False
            or receipt.get("checkpoint_saved") is not False or receipt.get("formal_model_reused") is not False
            or receipt.get("training_batch")!=[32,64,128,128,1]
            or receipt.get("geometry")!=dict(block_size=2,depth=3,sizes=[128,64,32,16])
            or receipt.get("device",{}).get("CUDA_VISIBLE_DEVICES")!=GPU_UUID
            or receipt.get("peak_cuda_bytes",base.PEAK_BYTES+1)>base.PEAK_BYTES
            or receipt.get("conservative_full_training_projection_seconds")!=projection
            or set(clocks)!={"Q","C","R"}
            or any(r.get("min")!=7 or r.get("max")!=7 or r.get("parameters",0)<=0 for r in clocks.values())):
        raise ValueError("Original receipt is not the fully executed synthetic timing-only check")
    if (base.sha(config)!=old_plan["config_source_sha256"]
            or base.sha(sampling_plan)!=old_plan["sampling_plan_sha256"]):
        raise ValueError("Training configuration or original sampling plan changed")
    old_code=Path(old_plan["code_root"]).resolve()
    bindings,old_real={},{}
    for filename,digest in old_plan.get("source_sha256",{}).items():
        path=Path(filename)
        if path.name.startswith("._"):
            continue
        if base.sha(path)!=digest:
            raise ValueError(f"Prior pinned source/configuration/evidence changed: {path}")
        bindings[str(path)]=digest
        if old_code in path.parents and path.suffix==".py":
            rel=str(path.relative_to(old_code))
            if not (code/rel).is_file() or base.sha(code/rel)!=digest:
                raise ValueError(f"Frozen computational source differs: {rel}")
            old_real[rel]=digest
    present=set(large._real_python_files(code))
    if not old_real or not NEW_SOURCES.issubset(present) or present-NEW_SOURCES!=set(old_real):
        raise ValueError("Only the three authorized refinement Python files may be added")
    inventories={}
    files={}
    for root in (stopped,preflight):
        inventories[str(root)]=sorted(str(p.relative_to(root)) for p in root.rglob("*"))
        for path in root.rglob("*"):
            if path.is_file():
                files[str(path.resolve())]=path
                bindings[str(path)]=base.sha(path)
    cpu,gpu=float(job.get("cpu_seconds",math.inf)),float(job.get("gpu_seconds",math.inf))
    size=sum(p.stat().st_size for p in files.values())
    if not budget_ok(cpu,gpu,size,gpu_projection=projection+70):
        raise ValueError("Stopped cumulative resource use plus projection exceeds authorized budget")
    return dict(stopped_root=str(stopped),preflight_root=str(preflight),prior_cpu_seconds=cpu,
        prior_gpu_seconds=gpu,prior_directory_bytes=size,projection_seconds=projection,
        prior_failed_pilot_force_site_evaluations=sum(r["actual_work"]["force_site_evaluations"] for r in old_pilot_rows),
        prior_source_bindings=bindings,prior_inventories=inventories,
        prior_cpu_gpu_already_include_preflight=True,preflight_added_again=False,
        additional_benchmark_updates=0,model_loaded=False,old_roots_modified=False)


def prior_live_processes(stopped_root,preflight_root):
    matches=[]
    modules={previous.NEW_SOURCE[:-3].replace("/","."),large.MODULE,"scripts.prepare_phi4_hopping_b2_large","recurrent_rg"}
    for folder in (Path(stopped_root),Path(preflight_root)):
        job=_read(folder/"job.json")
        for pid in {job.get("parent_pid"),*[r.get("pid") for r in job.get("stages",[])]}:
            if type(pid) is not int or pid<=0 or pid==os.getpid():
                continue
            try:
                args=Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace").split("\0")
            except FileNotFoundError:
                continue
            if modules.intersection(args) and any(a==str(folder) or a.startswith(str(folder)+"/") for a in args):
                matches.append(dict(pid=pid,command=args))
    return matches


def sampling_command(root,options):
    command=large.sampling_command(root,options)
    command[command.index("-m")+1]="scripts.prepare_phi4_hopping_b2_refined"
    return command


def validate_sampling(root,options):
    from recurrent_rg.data.phi4_hopping_loader import load_dataset
    from recurrent_rg.data.phi4_hopping_b2_refined_pipeline import generation_plan
    root=Path(root)
    expected=json.loads(json.dumps(generation_plan(options)))
    manifest,actual=_read(root/"data/manifest.json"),_read(root/"data/plan.json")
    if (actual!=expected or manifest.get("options")!=expected["options"]
            or manifest.get("source_pins")!=expected["source_pins"]
            or manifest.get("total_hmc_trajectories")!=expected["total_hmc_trajectories"]
            or manifest.get("total_force_site_evaluations")!=expected["total_force_site_evaluations"]):
        raise RuntimeError("Fresh production data differs from selected integrator/physical/work plan")
    data=load_dataset(root/"data",block_size=2,depth=3)
    if data.configurations.shape!=(60,128,128,128,1) or len(data.train_indices)!=40 or len(data.test_indices)!=20:
        raise RuntimeError("Wrong fresh production array shape or split")
    result=dict(data_manifest_sha256=data.manifest_sha256,data_warning_count=data.warning_count,quality_warnings_retained=True)
    del data
    rows=_read(root/"data/quality_summary.json").get("conditions",[])
    accepted=[v for r in rows for v in r.get("production_acceptance",[])]
    if len(rows)!=15 or len(accepted)!=60 or any(not math.isfinite(v) or v<=0 for v in accepted):
        raise RuntimeError("Incomplete/zero/nonfinite production acceptance; no training")
    return result


def plan(stopped_root,preflight_root,output,config=previous.DEFAULT_CONFIG,
         sampling_plan=large.DEFAULT_SAMPLING_PLAN,gpu_index=3,gpu_uuid=GPU_UUID):
    from experiments.phi4_hopping_qcr import TrainingOptions,load_config,training_plan
    from recurrent_rg.data.phi4_hopping_b2_refined_pipeline import generation_plan
    root,code=Path(output).resolve(),_code_root()
    config,sampling_plan=Path(config).resolve(),Path(sampling_plan).resolve()
    if root.exists() or root==code or root in code.parents or code in root.parents:
        raise ValueError("Require fresh output separate from frozen code")
    evidence=prerequisites(stopped_root,preflight_root,config,sampling_plan,gpu_index,gpu_uuid)
    values=load_config(config)
    values.update(data=str(root/"data"),output=str(root/"training"),device="cuda:0",expected_gpu_uuid=gpu_uuid)
    candidates=[json.loads(json.dumps(generation_plan(refinement_options(step,leap,sampling_plan)))) for step,leap in CANDIDATES]
    return dict(schema_version=5,created_at=base.now(),execution_requested=False,code_root=str(code),output=str(root),
        host=base.HOST,GPU=dict(index=gpu_index,uuid=gpu_uuid),config_source=str(config),config_source_sha256=base.sha(config),
        sampling_plan_source=str(sampling_plan),sampling_plan_sha256=base.sha(sampling_plan),
        prior_evidence=evidence,pilot=pilot_plan(sampling_plan),physical_candidates=candidates,
        work_accounting=dict(prior_failed_pilot_force_site_evaluations=evidence["prior_failed_pilot_force_site_evaluations"],
            maximum_new_pilot_force_site_evaluations=pilot_plan(sampling_plan)["maximum_force_site_evaluations"],
            candidate_production_force_site_evaluations=[p["total_force_site_evaluations"] for p in candidates],
            maximum_cumulative_force_site_evaluations_including_prior_pilot=(evidence["prior_failed_pilot_force_site_evaluations"]
                +pilot_plan(sampling_plan)["maximum_force_site_evaluations"]+max(p["total_force_site_evaluations"] for p in candidates)),
            production_work_cap_not_entire_workflow_cap=True),
        stages=["integrator_pilots","sampling","training"],training=training_plan(TrainingOptions(**values)),
        training_command_template=[sys.executable,"-u","-m","recurrent_rg","train","--config",
            str(root/"training_runtime_config.json"),"--output",str(root/"training"),"--device","cuda:0"],
        resources=dict(cpu_seconds=CPU_SECONDS,gpu_card_seconds=GPU_SECONDS,directory_bytes=DIRECTORY_BYTES,
            directory_scope="new root plus unique files of both immutable old roots",cpu_threads=1,nice=10,
            max_cuda_bytes=base.PEAK_BYTES,min_memory_bytes=36*1024**3,max_load_per_cpu=1.25,max_io_full_avg10=20.,
            idle_check_spacing_seconds=10.,min_gpu_free_bytes=12*1024**3,require_no_compute_processes=True),
        training_budget_admission=dict(only_changed_field="training.max_training_seconds",
            runtime_cap="min(remaining cumulative CPU,remaining cumulative GPU)-60 seconds",planned_steps=3000),
        no_resume=True,no_retry=True,no_auto_repair=True,no_migration=True,no_new_benchmark=True,
        qualification="固定三候选有限积分检验；首个全15-case通过后才预算门控生产；不自动扩展候选、不认证平衡或临界指数")


def _public_files(root,completed=False):
    names=["job.json","plan.json","training_config.json","prior_evidence.json","reused_preflight.json"]
    names += [name for name in ("integrator_selection.json","selected_physical_plan.json","training_runtime_config.json",
        "training_admission.json") if (root/name).is_file()]
    names += [p.name for p in sorted(root.glob("pilot_*.json"))]
    public={name:base.sha(root/name) for name in names}
    if completed:
        for name in _read(root/"data/manifest.json")["public_files"]:
            path=(root/"data"/name).resolve()
            if root/"data" not in path.parents or path.suffix!=".json":
                raise RuntimeError("Invalid public physical metadata path")
            public[f"data/{name}"]=base.sha(path)
        public["training/status.json"]=base.sha(root/"training/status.json")
        for row in _read(root/"training/status.json")["public_files"]:
            path=(root/"training"/row["path"]).resolve()
            if (root/"training" not in path.parents or path.suffix not in (".json",".npz",".png")
                    or base.sha(path)!=row["sha256"]):
                raise RuntimeError("Invalid public training artifact binding")
            public[f"training/{row['path']}"]=row["sha256"]
    base.save(root/"public_files.json",public)


def run(stopped_root,preflight_root,output,config=previous.DEFAULT_CONFIG,
        sampling_plan=large.DEFAULT_SAMPLING_PLAN,gpu_index=3,gpu_uuid=GPU_UUID):
    cpu_start,child_start=time.process_time(),resource.getrusage(resource.RUSAGE_CHILDREN)
    definition=plan(stopped_root,preflight_root,output,config,sampling_plan,gpu_index,gpu_uuid)
    if socket.gethostname()!=base.HOST or os.getpriority(os.PRIO_PROCESS,0)!=10:
        raise ValueError("Execution requires pinned 225 host and one inherited nice10 layer")
    if prior_live_processes(stopped_root,preflight_root):
        raise ValueError("Prior workflow still running; no duplicate start")
    root,code,evidence=Path(output).resolve(),_code_root(),definition["prior_evidence"]
    pins={str(path):base.sha(path) for path in large._real_python_files(code).values()}
    pins.update(evidence["prior_source_bindings"])
    pins.update({str(Path(p).resolve()):base.sha(p) for p in (config,sampling_plan)})
    root.mkdir(parents=True,exist_ok=False);(root/"logs").mkdir()
    configuration=copy.deepcopy(_read(config));configuration.update(data=str(root/"data"),output=str(root/"training"))
    configuration["training"].update(device="cuda:0",expected_gpu_uuid=gpu_uuid)
    base.save(root/"training_config.json",configuration);base.save(root/"prior_evidence.json",evidence)
    (root/"reused_preflight.json").write_bytes((Path(preflight_root)/"gpu_preflight.json").read_bytes())
    for name in ("training_config.json","prior_evidence.json","reused_preflight.json"):
        pins[str(root/name)]=base.sha(root/name)
    base.save(root/"plan.json",dict(definition,execution_requested=True,source_sha256=pins))
    pins[str(root/"plan.json")]=base.sha(root/"plan.json")
    job=dict(status="prepared",workflow_complete=False,parent_pid=os.getpid(),current=None,returncode=None,
        error=None,requires_user_action=False,started_at=base.now(),stages=[],plan_sha256=base.sha(root/"plan.json"),
        prior_cpu_seconds=evidence["prior_cpu_seconds"],prior_gpu_seconds=evidence["prior_gpu_seconds"],
        cpu_seconds=evidence["prior_cpu_seconds"],gpu_seconds=evidence["prior_gpu_seconds"],
        training_projection_seconds=evidence["projection_seconds"],additional_benchmark_updates=0,
        prior_failed_pilot_force_site_evaluations=evidence["prior_failed_pilot_force_site_evaluations"],
        actual_new_pilot_force_site_evaluations=0,
        cumulative_total_force_site_evaluations=evidence["prior_failed_pilot_force_site_evaluations"],
        formal_training_started=False,formal_optimizer_updates=0,production_started=False)
    gpu_seconds,active,active_stage=evidence["prior_gpu_seconds"],None,None

    def checkpoint():
        child=resource.getrusage(resource.RUSAGE_CHILDREN)
        job["cpu_seconds"]=(evidence["prior_cpu_seconds"]+time.process_time()-cpu_start
            +child.ru_utime-child_start.ru_utime+child.ru_stime-child_start.ru_stime)
        if active is not None and active.poll() is None:
            try:
                fields=Path(f"/proc/{active.pid}/stat").read_text().split(")",1)[1].split()
                job["cpu_seconds"]+=(int(fields[11])+int(fields[12]))/os.sysconf("SC_CLK_TCK")
            except (FileNotFoundError,ProcessLookupError):
                pass
        job["gpu_seconds"]=gpu_seconds+(time.monotonic()-active_stage["monotonic_start"]
            if active is not None and active.poll() is None and active_stage["name"]=="training" else 0.)
        job["new_directory_bytes"]=sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
        job["directory_bytes"]=job["new_directory_bytes"]+evidence["prior_directory_bytes"]
        base.save(root/"job.json",job)

    def verify():
        if any(base.sha(path)!=digest for path,digest in pins.items()):
            raise RuntimeError("Pinned source/configuration/evidence changed; no repair")
        for folder,inventory in evidence["prior_inventories"].items():
            if sorted(str(p.relative_to(folder)) for p in Path(folder).rglob("*"))!=inventory:
                raise RuntimeError("Immutable old root inventory changed")

    def admitted(name=None):
        return budget_ok(job["cpu_seconds"],job["gpu_seconds"],job["directory_bytes"],
            cpu_projection=job.get("remaining_production_cpu_projection_seconds",0.) if name=="sampling" else 0.,
            gpu_projection=job["training_projection_seconds"] if name else 0.)

    def hold(message):
        job.update(status="budget_hold",workflow_complete=False,current=None,returncode=3,
            error=message,requires_user_action=True,ended_at=base.now())
        checkpoint();_public_files(root);return 3

    def wait_resources(name):
        nonlocal gpu_seconds
        gpu=name=="training";job.update(status="waiting_resources",current=name)
        while True:
            first=dict(host=base.host_resources())
            if gpu:first["gpu"]=base.gpu_resources(gpu_index,gpu_uuid)
            job["resources"]=first;checkpoint()
            if not admitted(name):return False
            if not base.ready(first,gpu):time.sleep(30);continue
            if not gpu:return True
            started=time.monotonic();time.sleep(10)
            second=dict(host=base.host_resources(),gpu=base.gpu_resources(gpu_index,gpu_uuid))
            spacing=time.monotonic()-started;gpu_seconds+=spacing
            job["gpu_idle_checks"]=dict(first=first,second=second,spacing_seconds=spacing);checkpoint()
            if not admitted(name):return False
            if spacing>=10 and base.ready(second,True):return True

    def child_status(name,stage):
        path=root/"integrator_selection.json" if name=="integrator_pilots" else root/("data" if name=="sampling" else "training")/"status.json"
        try:
            stage["child_status"]=_read(path)
            if name=="training":job["formal_optimizer_updates"]=stage["child_status"].get("actual_steps",0)
            if name=="integrator_pilots":
                job["actual_new_pilot_force_site_evaluations"]=stage["child_status"].get("actual_force_site_evaluations",0)
                job["cumulative_total_force_site_evaluations"]=(evidence["prior_failed_pilot_force_site_evaluations"]
                    +job["actual_new_pilot_force_site_evaluations"])
        except (FileNotFoundError,json.JSONDecodeError):pass

    checkpoint()
    try:
        from scripts.report_phi4_hopping_qcr import select_chinese_font
        job["chinese_font"]=str(select_chinese_font(os.environ.get("QCR_ZH_FONT")))
        for name in definition["stages"]:
            verify()
            if not wait_resources(name):return hold("Cumulative budget plus remaining projection exhausted while waiting")
            verify();checkpoint()
            if not admitted(name):return hold("Actual cumulative CPU/GPU plus conservative remaining work exceeds budget")
            if name=="integrator_pilots":
                available=CPU_SECONDS-job["cpu_seconds"]-job["training_projection_seconds"]-60.
                if available<=0:return hold("No bounded CPU allowance remains for preregistered pilots")
                command=[sys.executable,"-u","-m",MODULE,"--pilot-root",str(root),"--sampling-plan",str(sampling_plan),
                    "--max-pilot-cpu-seconds",str(available)]
            elif name=="sampling":
                command=sampling_command(root,refinement_options(*CANDIDATES[job["selected_candidate"]],sampling_plan))
            else:
                cap=min(GPU_SECONDS-job["gpu_seconds"],CPU_SECONDS-job["cpu_seconds"])-60.
                if cap<job["training_projection_seconds"]:return hold("Remaining runtime cap cannot admit projected 3000-step training")
                runtime=copy.deepcopy(configuration);runtime["training"]["max_training_seconds"]=cap
                base.save(root/"training_runtime_config.json",runtime)
                admission=dict(at=base.now(),runtime_cap_seconds=cap,configured_cap_seconds=GPU_SECONDS,
                    only_changed_field="training.max_training_seconds",planned_optimizer_updates=3000,
                    source_config_sha256=base.sha(root/"training_config.json"),runtime_config_sha256=base.sha(root/"training_runtime_config.json"),
                    cumulative_cpu_seconds=job["cpu_seconds"],cumulative_gpu_seconds=job["gpu_seconds"],boundary_reserve_seconds=60.)
                base.save(root/"training_admission.json",admission);job["training_admission"]=admission
                for p in ("training_runtime_config.json","training_admission.json"):pins[str(root/p)]=base.sha(root/p)
                command=definition["training_command_template"]
            stage=dict(name=name,status="running",returncode=None,command=command,started_at=base.now(),monotonic_start=time.monotonic())
            active_stage=stage;job["stages"].append(stage);job.update(status="running",current=name)
            with (root/"logs"/f"{name}.log").open("xb") as log:
                active=subprocess.Popen(command,cwd=code,env=base.child_environment(gpu_uuid if name=="training" else None),stdout=log,stderr=subprocess.STDOUT)
                stage["pid"]=active.pid
                if name=="training":job["formal_training_started"]=True
                if name=="sampling":job["production_started"]=True
                while active.poll() is None:
                    child_status(name,stage);checkpoint()
                    if not admitted():job.update(requires_user_action=True,budget_exceeded_with_child_running=True,
                        budget_note="Child retained; no kill/restart. No next stage admitted.")
                    time.sleep(5)
                rc=active.returncode
            child_status(name,stage);elapsed=time.monotonic()-stage["monotonic_start"]
            if name=="training":gpu_seconds+=elapsed
            state="complete" if rc==0 else "budget_hold" if rc==3 else "scientific_stop" if rc==4 else "failed"
            stage.update(status=state,returncode=rc,ended_at=base.now(),wall_seconds=elapsed);active,active_stage=None,None
            verify()
            if rc:
                job.update(status=state,workflow_complete=(name=="integrator_pilots" and rc==4),current=None,
                    returncode=rc,error=f"{name} exited {rc}; completed outputs preserved",requires_user_action=True,ended_at=base.now())
                checkpoint();_public_files(root);return rc
            if name=="integrator_pilots":
                job.update(validate_selection(root,sampling_plan))
                job["actual_new_pilot_force_site_evaluations"]=job["actual_endpoint_pilot_force_site_evaluations"]
                job["cumulative_total_force_site_evaluations"]=(evidence["prior_failed_pilot_force_site_evaluations"]
                    +job["actual_new_pilot_force_site_evaluations"])
                base.save(root/"selected_physical_plan.json",job["selected_physical_plan"])
                for p in [root/"integrator_selection.json",root/"selected_physical_plan.json",*root.glob("pilot_*.json")]:pins[str(p)]=base.sha(p)
            elif name=="sampling":
                job.update(validate_sampling(root,refinement_options(*CANDIDATES[job["selected_candidate"]],sampling_plan)))
                job["actual_production_force_site_evaluations"]=_read(root/"data/manifest.json")["total_force_site_evaluations"]
                job["actual_total_force_site_evaluations_including_pilots"]=job["actual_production_force_site_evaluations"]+job["actual_endpoint_pilot_force_site_evaluations"]
                job["cumulative_total_force_site_evaluations"]=(evidence["prior_failed_pilot_force_site_evaluations"]
                    +job["actual_total_force_site_evaluations_including_pilots"])
                job["remaining_production_cpu_projection_seconds"]=0.;pins[str(root/"data/manifest.json")]=base.sha(root/"data/manifest.json")
            else:
                base.validate_training(root,job["data_manifest_sha256"]);job["formal_optimizer_updates"]=3000
            checkpoint()
        if not admitted():return hold("Completed work exceeds registered cumulative resource budget")
        job.update(status="complete",workflow_complete=True,current=None,returncode=0,error=None,requires_user_action=False,ended_at=base.now())
        checkpoint();_public_files(root,completed=True);return 0
    except BaseException as exc:
        if active is not None and active.poll() is None:
            job.update(status="waiting_running_child_after_parent_error",requires_user_action=True,error=f"{type(exc).__name__}: {exc}")
            checkpoint();active.wait()
        if active is not None and active_stage is not None:
            elapsed=time.monotonic()-active_stage["monotonic_start"]
            if active_stage["name"]=="training":gpu_seconds+=elapsed
            child_status(active_stage["name"],active_stage)
            active_stage.update(status="child_exited_after_parent_error",returncode=active.returncode,ended_at=base.now(),wall_seconds=elapsed)
            active,active_stage=None,None
        job.update(status="failed",workflow_complete=False,current=None,returncode=1,requires_user_action=True,
            error=f"{type(exc).__name__}: {exc}",ended_at=base.now());checkpoint();_public_files(root);raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stopped-root");parser.add_argument("--preflight-root");parser.add_argument("--output")
    parser.add_argument("--config",default=previous.DEFAULT_CONFIG)
    parser.add_argument("--sampling-plan",default=large.DEFAULT_SAMPLING_PLAN)
    parser.add_argument("--gpu-index",type=int,default=3);parser.add_argument("--gpu-uuid",default=GPU_UUID)
    parser.add_argument("--pilot-root");parser.add_argument("--max-pilot-cpu-seconds",type=float)
    parser.add_argument("--execute",action="store_true")
    a=parser.parse_args(argv)
    if a.pilot_root:
        if (a.stopped_root or a.preflight_root or a.output or a.execute or a.max_pilot_cpu_seconds is None
                or not math.isfinite(a.max_pilot_cpu_seconds) or a.max_pilot_cpu_seconds<=0):
            parser.error("Internal bounded pilot requires only --pilot-root, original sampling plan, and positive CPU allowance")
        return pilot_selection(a.pilot_root,a.sampling_plan,a.max_pilot_cpu_seconds)
    if not a.stopped_root or not a.preflight_root or not a.output or a.max_pilot_cpu_seconds is not None:
        parser.error("Require --stopped-root, --preflight-root and fresh --output")
    values=(a.stopped_root,a.preflight_root,a.output,a.config,a.sampling_plan,a.gpu_index,a.gpu_uuid)
    if a.execute:return run(*values)
    print(json.dumps(plan(*values),ensure_ascii=False,indent=2,allow_nan=False));return 0


if __name__=="__main__":
    raise SystemExit(main())
