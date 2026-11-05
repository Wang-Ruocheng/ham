"""规划或生成欧氏 phi4 连续场数据；默认只打印计划，不采样、不训练。"""
import argparse
import json
from pathlib import Path

from recurrent_rg.data.phi4 import Phi4Options, generate_dataset, generation_plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true", help="显式执行新数据采样；现有目录绝不覆盖")
    parser.add_argument("--dimension", type=int, default=2, choices=(1, 2, 3))
    parser.add_argument("--sizes", type=int, nargs="+", default=[36])
    parser.add_argument("--r-values", type=float, nargs="+", default=list(Phi4Options.r_values))
    parser.add_argument("--u-values", type=float, nargs="+", default=[1.])
    parser.add_argument("--h", type=float, default=0.)
    for name in ("chains", "samples", "burn_in", "spacing", "leapfrog_steps", "seed",
                 "test_every", "test_offset", "max_trajectories", "max_force_site_evaluations",
                 "max_output_bytes", "max_memory_bytes"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(Phi4Options, name))
    parser.add_argument("--step-size", type=float, default=.08)
    args = parser.parse_args(argv)
    values = vars(args).copy()
    output, execute = values.pop("output"), values.pop("execute")
    for name in ("sizes", "r_values", "u_values"):
        values[name] = tuple(values[name])
    options = Phi4Options(**values)
    plan = generation_plan(options)
    if not execute:
        print(json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False))
        return plan
    if output is None:
        parser.error("--execute requires --output")
    result = generate_dataset(output, options)
    print(json.dumps(dict(status="complete", output=str(output.resolve()),
        conditions=len(result["shards"]), warning_count=result["warning_count"],
        training_updates=0, current_binary_qcr_compatible=False), ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
