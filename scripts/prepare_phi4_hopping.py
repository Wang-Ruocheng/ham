"""固定lambda、零场扫描kappa的phi4数据；默认只规划，不采样或训练。"""
import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import json
from pathlib import Path

from recurrent_rg.data.phi4_hopping_pipeline import Phi4HoppingOptions, generation_plan, generate_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true", help="显式执行采样；仅创建全新输出目录")
    parser.add_argument("--sizes", type=int, nargs="+", default=list(Phi4HoppingOptions.sizes))
    parser.add_argument("--dimension", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--kappa-values", type=float, nargs="+", default=list(Phi4HoppingOptions.kappa_values))
    parser.add_argument("--quartic-lambda", type=float, default=1.)
    parser.add_argument("--h", type=float, choices=(0.,), default=0.)
    parser.add_argument("--step-size", type=float, default=.08, help="内部psi坐标的leapfrog步长")
    for name in ("chains", "samples", "burn_in", "spacing", "leapfrog_steps", "seed", "test_every",
                 "test_offset", "max_trajectories", "max_force_site_evaluations", "max_cluster_updates",
                 "max_cluster_neighbor_visits", "max_output_bytes", "max_memory_bytes"):
        parser.add_argument("--"+name.replace("_", "-"), type=int, default=getattr(Phi4HoppingOptions, name))
    values = vars(parser.parse_args(argv))
    output, execute = values.pop("output"), values.pop("execute")
    for name in ("sizes", "kappa_values"):
        values[name] = tuple(values[name])
    options = Phi4HoppingOptions(**values)
    plan = generation_plan(options)
    if not execute:
        print(json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False))
        return plan
    if output is None:
        parser.error("--execute requires --output")
    manifest = generate_dataset(output, options)
    print(json.dumps(dict(status="complete", schema=manifest["schema"], output=str(output.resolve()),
        conditions=len(manifest["shards"]), warning_count=manifest["warning_count"],
        training_updates=0, equilibrium_certified=False), ensure_ascii=False))
    return manifest


if __name__ == "__main__":
    main()
