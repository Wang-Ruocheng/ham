"""生成新的二维 Houdayer 簇更新＋平行回火数据；不启动模型训练。"""

import argparse
import json
from pathlib import Path

from recurrent_rg.data.spin_glass_pipeline import GenerationOptions
from recurrent_rg.data.spin_glass_houdayer_pipeline import (
    NUMERIC_LIMIT,
    generate_dataset,
    generation_plan,
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dimension", type=int, default=2, help="当前Houdayer管线只支持二维")
    geometry = p.add_mutually_exclusive_group()
    geometry.add_argument("--length", type=int, default=None, help="两个周期空间轴的长度；默认36")
    geometry.add_argument("--shape", type=int, nargs=2, help="二维逐轴长度，例如36 36")
    p.add_argument("--temperatures", type=float, nargs="+", required=True)
    p.add_argument("--disorders", type=int, default=9)
    p.add_argument("--replicas", type=int, default=4, help="簇更新相互作用的真实副本数；不视为独立链")
    p.add_argument("--samples", type=int, default=512)
    p.add_argument("--burn-in", type=int, default=8192)
    p.add_argument("--spacing", type=int, default=16)
    p.add_argument("--exchange-interval", type=int, default=1)
    p.add_argument("--cluster-interval", type=int, default=1)
    p.add_argument("--trace-interval", type=int, default=16)
    p.add_argument("--initial", choices=("hot", "cold"), default="hot")
    p.add_argument("--algorithm", choices=("pt", "houdayer_pt"), default="houdayer_pt")
    p.add_argument("--distribution", choices=("bimodal", "gaussian"), default="bimodal")
    p.add_argument("--bond-scale", type=float, default=1.)
    p.add_argument("--bond-mean", type=float, default=0.)
    p.add_argument("--positive-probability", type=float, default=.5)
    p.add_argument("--field", type=float, default=0.)
    p.add_argument("--seed", type=int, default=20261019)
    p.add_argument("--test-every", type=int, default=3)
    p.add_argument("--test-offset", type=int, default=1)
    p.add_argument("--dry-run", action="store_true", help="只验证固定工作计划，不采样、不写目录；无行政时间或内存上限")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.dimension != 2:
        p.error("Houdayer管线目前只支持dimension=2")
    shape = tuple(args.shape) if args.shape is not None else (36 if args.length is None else args.length,) * 2
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
                  initial=args.initial, algorithm=args.algorithm)
    try:
        plan = generation_plan(options, **extras)
    except ValueError as exc:
        p.error(str(exc))
    if args.dry_run:
        print(json.dumps(dict(dry_run=True, output=str(args.output), **plan),
                         indent=2, ensure_ascii=False, allow_nan=False))
        return plan
    manifest = generate_dataset(args.output, options, **extras)
    print(json.dumps(dict(status=manifest["status"], output=str(args.output),
                          sampler_schema=manifest["sampler_schema"],
                          disorders=len(manifest["shards"]), training_updates=0), ensure_ascii=False))
    return manifest


if __name__ == "__main__":
    main()
