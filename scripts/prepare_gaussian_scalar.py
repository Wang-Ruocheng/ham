"""规划或生成二维自由高斯场；默认仅打印计划，--execute 才生成新目录。"""
import os

# Set before importing NumPy. The FFT sampler does not use dense BLAS kernels.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
import json
from pathlib import Path

from recurrent_rg.data.gaussian_scalar import (
    GaussianScalarOptions, generate_dataset, generation_plan,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true", help="仅在新目录显式执行；不覆盖、不续采")
    parser.add_argument("--sizes", type=int, nargs="+", default=[36])
    parser.add_argument("--r-values", type=float, nargs="+", default=list(GaussianScalarOptions.r_values))
    for name in ("groups", "samples", "seed", "test_every", "test_offset", "max_output_bytes", "max_memory_bytes"):
        parser.add_argument("--" + name.replace("_", "-"), type=int,
                            default=getattr(GaussianScalarOptions, name))
    args = vars(parser.parse_args(argv))
    output, execute = args.pop("output"), args.pop("execute")
    args["sizes"], args["r_values"] = tuple(args["sizes"]), tuple(args["r_values"])
    options = GaussianScalarOptions(**args)
    plan = generation_plan(options)
    if not execute:
        print(json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False))
        return plan
    if output is None:
        parser.error("--execute requires --output")
    result = generate_dataset(output, options)
    print(json.dumps(dict(status="complete", output=str(output.resolve()),
        conditions=len(result["shards"]), schema=result["schema"],
        sample_semantics=result["sample_semantics"], training_updates=0), ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
