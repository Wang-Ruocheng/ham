"""Independent work envelope for the authorized finer-HMC follow-up.

Only the numerical integration schedule and this new child work envelope
change. The action, hybrid kernel, initial-state policy and historical caps
remain in their original modules. The parent still gates total CPU at7200 s.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from . import phi4 as base
from . import phi4_hybrid_pipeline as frozen
from .phi4_ru_pipeline import Phi4RUOptions


FORCE_SITE_CAP = 8_000_000_000


@dataclass(frozen=True)
class Phi4FineOptions(Phi4RUOptions):
    sizes: tuple[int, ...] = (108,)
    r_values: tuple[float, ...] = (-.7625, -.696875, -.63125)
    u_values: tuple[float, ...] = (.5,)
    step_size: float = .04
    leapfrog_steps: int = 24
    max_force_site_evaluations: int = FORCE_SITE_CAP

    def validate(self):
        # Do not change or monkeypatch the previous independent 7e9 envelope.
        base.Phi4Options.validate(self)
        if self.dimension != 2 or base._finite(self.h, "h") != 0:
            raise ValueError("Fine follow-up requires two-dimensional h=0 fields")
        if base._finite(self.cluster_probability, "cluster_probability") != .5:
            raise ValueError("The fixed lazy cluster probability remains .5")
        if any(base._finite(u, "u") <= 0 for u in self.u_values):
            raise ValueError("Strictly positive u is required")
        schedule = (base._finite(self.step_size, "step_size"), self.leapfrog_steps)
        if schedule not in ((.08, 12), (.04, 24)):
            raise ValueError("Only the predeclared .96-length integration schedules are allowed")
        for name, cap in (("max_trajectories", 500_000),
                          ("max_force_site_evaluations", FORCE_SITE_CAP),
                          ("max_cluster_updates", 500_000),
                          ("max_cluster_neighbor_visits", 2_000_000_000),
                          ("max_output_bytes", 1024**3),
                          ("max_memory_bytes", 512*1024**2)):
            if base._integer(getattr(self,name),name,1) > cap:
                raise ValueError(f"New fine child work cap exceeded: {name}")
        return self


def generation_plan(options):
    if not isinstance(options,Phi4FineOptions):
        raise TypeError("Require the explicit fine follow-up work envelope")
    return frozen.generation_plan(options)


def source_pins():
    return dict(frozen.source_pins(), **{
        "recurrent_rg/data/phi4_ru_fine_pipeline.py":base.sha256_file(__file__),
        "recurrent_rg/data/phi4_ru_pipeline.py":base.sha256_file(Path(__file__).with_name("phi4_ru_pipeline.py"))})


def generate_dataset(output,options):
    plan=generation_plan(options)
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError("New fine output never overwrites or resumes an earlier experiment")
    before=source_pins()
    result=frozen.generate_dataset(output,options)
    after=source_pins()
    base._write_json(output/"fine_wrapper_receipt.json",dict(schema_version=5,
        status="complete" if before==after else "failed", historical_kernel_unchanged=True,
        historical_work_cap_unchanged=True, child_force_site_cap=FORCE_SITE_CAP,
        production_force_site_evaluations=plan["total_force_site_evaluations"],
        integration=dict(step_size=options.step_size,leapfrog_steps=options.leapfrog_steps,
                         trajectory_length=options.step_size*options.leapfrog_steps),
        source_pins_before=before,source_pins_after=after,
        qualification="用户授权继续；L108采用预先检验的细积分，历史数据及旧7e9工作包络保持原件。"))
    if before!=after:
        status=json.loads((output/"status.json").read_text())
        status.update(status="failed",workflow_complete=False,returncode=1,
                      error="Fine wrapper or frozen sampler source changed",requires_user_action=True)
        base._write_json(output/"status.json",status)
        raise RuntimeError("Fine wrapper or frozen sampler source changed")
    # Publish this new wrapper's receipt without changing any historical output.
    result["public_files"] = list(result["public_files"]) + ["fine_wrapper_receipt.json"]
    base._write_json(output/"manifest.json",result)
    status=json.loads((output/"status.json").read_text())
    status["manifest_sha256"]=base.sha256_file(output/"manifest.json")
    base._write_json(output/"status.json",status)
    return result
