"""Explicitly authorized two-parameter phi4 data plan; historical caps stay intact.

Only the independent work envelope is new. Sampling delegates unchanged to the
frozen HMC + lazy embedded-Wolff implementation. No adaptation, extra samples,
sector augmentation, legacy-recipe switches, or historical output mutation.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from . import phi4 as base
from . import phi4_hybrid_pipeline as frozen


SCHEMA = frozen.SCHEMA
SAMPLE_SEMANTICS = frozen.SAMPLE_SEMANTICS
R_VALUES = tuple(-1.55 + 1.05 * i / 8 for i in range(9))
U_VALUES = tuple(.5 + .5 * i / 4 for i in range(5))
FORCE_SITE_CAP = 7_000_000_000
TRAJECTORY_CAP = 500_000
NEIGHBOR_VISIT_CAP = 2_000_000_000


@dataclass(frozen=True)
class Phi4RUOptions(base.Phi4Options):
    sizes: tuple[int, ...] = (36,)
    dimension: int = 2
    r_values: tuple[float, ...] = R_VALUES
    u_values: tuple[float, ...] = U_VALUES
    h: float = 0.
    chains: int = 4
    samples: int = 128
    burn_in: int = 1024
    spacing: int = 8
    step_size: float = .08
    leapfrog_steps: int = 12
    seed: int = 20261029
    test_every: int = 4
    test_offset: int = 1
    # Optional explicit physical points for a matched nonrectangular scan.
    parameter_cells: tuple[tuple[float, float], ...] | None = None
    cluster_probability: float = .5
    max_trajectories: int = TRAJECTORY_CAP
    max_force_site_evaluations: int = FORCE_SITE_CAP
    max_cluster_updates: int = TRAJECTORY_CAP
    max_cluster_neighbor_visits: int = NEIGHBOR_VISIT_CAP
    max_output_bytes: int = 1024 ** 3
    max_memory_bytes: int = 512 * 1024 ** 2

    def validate(self):
        super().validate()
        if self.dimension != 2 or base._finite(self.h, "h") != 0.:
            raise ValueError("The authorized r-u experiment is two-dimensional and h=0")
        if base._finite(self.cluster_probability, "cluster_probability") != .5:
            raise ValueError("Fixed lazy embedded-Wolff probability must equal .5")
        if any(base._finite(u, "u") <= 0 for u in self.u_values):
            raise ValueError("This r-u experiment requires strictly positive u")
        for name, cap in (("max_trajectories", TRAJECTORY_CAP),
                          ("max_force_site_evaluations", FORCE_SITE_CAP),
                          ("max_cluster_updates", TRAJECTORY_CAP),
                          ("max_cluster_neighbor_visits", NEIGHBOR_VISIT_CAP),
                          ("max_output_bytes", 1024 ** 3),
                          ("max_memory_bytes", 512 * 1024 ** 2)):
            value = base._integer(getattr(self, name), name, 1)
            if value > cap:
                raise ValueError(f"Authorized r-u work cap exceeded: {name} > {cap}")
        return self


def generation_plan(options):
    if not isinstance(options, Phi4RUOptions):
        raise TypeError("Require the independent Phi4RUOptions work envelope")
    # Polymorphic options validation selects this explicitly authorized envelope;
    # no global constants or historical sampler options are patched.
    return frozen.generation_plan(options)


def source_pins():
    return dict(frozen.source_pins(), **{
        "recurrent_rg/data/phi4_ru_pipeline.py": base.sha256_file(__file__)})


def generate_dataset(output, options):
    plan = generation_plan(options)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Existing r-u data is never overwritten or resumed")
    before = source_pins()
    result = frozen.generate_dataset(output, options)
    after = source_pins()
    receipt = dict(schema_version=5, status="complete" if before == after else "failed",
        workflow_complete=before == after, returncode=0 if before == after else 1,
        authorization="User approved the 45-condition r-u scan with dg=2 and its new sampling work cap",
        historical_kernel_unchanged=True, historical_work_cap_unchanged=True,
        new_force_site_cap=FORCE_SITE_CAP, new_hybrid_cycle_cap=TRAJECTORY_CAP,
        production_force_site_evaluations=plan["total_force_site_evaluations"],
        production_hmc_trajectories=plan["total_hmc_trajectories"],
        source_pins_before=before, source_pins_after=after,
        labels_used_in_training=False, equilibrium_certified=False,
        warnings_retained_without_resampling=True, automatic_repair=False)
    base._write_json(output / "wrapper_receipt.json", receipt)
    if before != after:
        status = json.loads((output / "status.json").read_text())
        status.update(status="failed", workflow_complete=False, returncode=1,
                      error="Authorized r-u wrapper or sampler source changed", requires_user_action=True)
        base._write_json(output / "status.json", status)
        raise RuntimeError("Authorized r-u wrapper or sampler source changed")
    return result
