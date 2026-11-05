"""规划或生成新的phi4 lazy HMC＋embedded Wolff数据；默认仅打印计划。"""
import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
import json
from pathlib import Path

from recurrent_rg.data.phi4_hybrid_pipeline import Phi4HybridOptions, generate_dataset, generation_plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true", help="显式生成全新目录，不覆盖或续采")
    parser.add_argument("--sizes", type=int, nargs="+", default=list(Phi4HybridOptions.sizes))
    parser.add_argument("--dimension", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--r-values", type=float, nargs="+", default=list(Phi4HybridOptions.r_values))
    parser.add_argument("--u-values", type=float, nargs="+", default=[.2])
    parser.add_argument("--h", type=float, default=0.)
    parser.add_argument("--step-size", type=float, default=.08)
    parser.add_argument("--cluster-probability", type=float, choices=(.5,), default=.5)
    for name in ("chains", "samples", "burn_in", "spacing", "leapfrog_steps", "seed", "test_every",
                 "test_offset", "max_trajectories", "max_force_site_evaluations", "max_cluster_updates",
                 "max_cluster_neighbor_visits", "max_output_bytes", "max_memory_bytes"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(Phi4HybridOptions, name))
    values = vars(parser.parse_args(argv))
    output, execute = values.pop("output"), values.pop("execute")
    for name in ("sizes", "r_values", "u_values"):
        values[name] = tuple(values[name])
    options = Phi4HybridOptions(**values)
    plan = generation_plan(options)
    if not execute:
        print(json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False))
        return plan
    if output is None:
        parser.error("--execute requires --output")
    result = generate_dataset(output, options)
    print(json.dumps(dict(status="complete", output=str(output.resolve()),
        schema=result["schema"], conditions=len(result["shards"]),
        warning_count=result["warning_count"], training_updates=0,
        equilibrium_certified=False), ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
