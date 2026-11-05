"""生成新的三维零场 EA 数据：平行回火＋温度选择性等能簇更新。"""

import argparse
import json
from pathlib import Path

from recurrent_rg.data.spin_glass_pipeline import GenerationOptions
from recurrent_rg.data.spin_glass3d_pipeline import NUMERIC_LIMIT, generate_dataset, generation_plan


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dimension", type=int, default=3)
    geometry = p.add_mutually_exclusive_group()
    geometry.add_argument("--length", type=int, default=None, help="三个周期轴长度，默认16")
    geometry.add_argument("--shape", type=int, nargs=3)
    p.add_argument("--temperatures", type=float, nargs="+", required=True)
    p.add_argument("--disorders", type=int, default=9)
    p.add_argument("--replicas", type=int, default=4)
    p.add_argument("--samples", type=int, default=512)
    p.add_argument("--burn-in", type=int, default=8192)
    p.add_argument("--spacing", type=int, default=16)
    p.add_argument("--exchange-interval", type=int, default=1)
    p.add_argument("--cluster-interval", type=int, default=1)
    p.add_argument("--trace-interval", type=int, default=16)
    p.add_argument("--icm-max-temperature", type=float, default=1.0)
    p.add_argument("--initial", choices=("hot", "cold"), default="hot")
    p.add_argument("--algorithm", choices=("pt", "icm_pt"), default="icm_pt")
    p.add_argument("--distribution", choices=("bimodal", "gaussian"), default="bimodal")
    p.add_argument("--bond-scale", type=float, default=1.0)
    p.add_argument("--bond-mean", type=float, default=0.0)
    p.add_argument("--positive-probability", type=float, default=0.5)
    p.add_argument("--field", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=20261019)
    p.add_argument("--test-every", type=int, default=3)
    p.add_argument("--test-offset", type=int, default=1)
    p.add_argument("--dry-run", action="store_true", help="只校验固定计划，不采样或写文件")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.dimension != 3:
        p.error("新管线仅支持dimension=3；历史二维数据不更改")
    shape = tuple(args.shape) if args.shape is not None else (16 if args.length is None else args.length,) * 3
    options = GenerationOptions(
        shape=shape, temperatures=tuple(args.temperatures), disorders=args.disorders,
        replicas=args.replicas, samples=args.samples, burn_in=args.burn_in, spacing=args.spacing,
        exchange_interval=args.exchange_interval, distribution=args.distribution,
        bond_scale=args.bond_scale, bond_mean=args.bond_mean,
        positive_probability=args.positive_probability, field=args.field, seed=args.seed,
        test_every=args.test_every, test_offset=args.test_offset,
        max_spin_proposals=NUMERIC_LIMIT, max_output_bytes=NUMERIC_LIMIT,
        max_memory_bytes=NUMERIC_LIMIT, max_sites=NUMERIC_LIMIT,
    )
    extras = dict(cluster_interval=args.cluster_interval, trace_interval=args.trace_interval,
                  initial=args.initial, algorithm=args.algorithm, icm_max_temperature=args.icm_max_temperature)
    try:
        plan = generation_plan(options, **extras)
    except ValueError as exc:
        p.error(str(exc))
    if args.dry_run:
        print(json.dumps(dict(dry_run=True, output=str(args.output), **plan), indent=2,
                         ensure_ascii=False, allow_nan=False))
        return plan
    manifest = generate_dataset(args.output, options, **extras)
    print(json.dumps(dict(status=manifest["status"], output=str(args.output),
                          sampler_schema=manifest["sampler_schema"],
                          disorders=len(manifest["shards"]), training_updates=0), ensure_ascii=False))
    return manifest


if __name__ == "__main__":
    main()
