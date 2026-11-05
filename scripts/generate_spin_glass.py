"""生成任意有限维周期 EA 自旋玻璃数据；不启动 QCR 训练。"""
import argparse
import json
from pathlib import Path

from recurrent_rg.data.spin_glass_pipeline import GenerationOptions, generate_dataset, generation_plan


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dimension", type=int, default=None, help="空间维数；默认2，或由--shape推断")
    geometry = p.add_mutually_exclusive_group()
    geometry.add_argument("--length", type=int, help="每个空间轴的长度；默认9")
    geometry.add_argument("--shape", type=int, nargs="+", help="逐轴长度，例如9 12 15")
    p.add_argument("--temperatures", type=float, nargs="+", required=True,
                   help="严格正且递增的平行回火温度梯度；不支持T=0")
    p.add_argument("--disorders", type=int, default=8)
    p.add_argument("--replicas", type=int, default=2, help="同一无序下的独立热副本/PT梯度数")
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--burn-in", type=int, default=2048, help="保存前每温度的单点Metropolis整格sweep数")
    p.add_argument("--spacing", type=int, default=16, help="保存构型间隔的整格sweep数")
    p.add_argument("--exchange-interval", type=int, default=1)
    p.add_argument("--distribution", choices=("bimodal", "gaussian"), default="bimodal")
    p.add_argument("--bond-scale", type=float, default=1., help="双峰绝对耦合/高斯标准差，能量单位")
    p.add_argument("--bond-mean", type=float, default=0., help="仅高斯耦合的均值")
    p.add_argument("--positive-probability", type=float, default=.5, help="仅双峰耦合的P(J>0)")
    p.add_argument("--field", type=float, default=0.)
    p.add_argument("--seed", type=int, default=91)
    p.add_argument("--test-every", type=int, default=3)
    p.add_argument("--test-offset", type=int, default=1)
    p.add_argument("--max-spin-proposals", type=int, default=50_000_000)
    p.add_argument("--max-output-gib", type=float, default=2.)
    p.add_argument("--max-memory-gib", type=float, default=.5)
    p.add_argument("--max-sites", type=int, default=1_000_000)
    p.add_argument("--dry-run", action="store_true", help="只验证计划与预算，不采样、不写目录")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.dimension is not None and not 1 <= args.dimension <= 16:
        p.error("dimension必须在1到16之间")
    if args.shape is not None:
        shape = tuple(args.shape)
        if args.dimension is not None and len(shape) != args.dimension:
            p.error("dimension与shape的轴数不符")
    else:
        shape = (9 if args.length is None else args.length,) * (args.dimension or 2)
    import math
    if not math.isfinite(args.max_output_gib) or args.max_output_gib <= 0:
        p.error("max-output-gib必须有限且为正")
    if not math.isfinite(args.max_memory_gib) or args.max_memory_gib <= 0:
        p.error("max-memory-gib必须有限且为正")
    options = GenerationOptions(shape=shape, temperatures=tuple(args.temperatures),
        disorders=args.disorders, replicas=args.replicas, samples=args.samples,
        burn_in=args.burn_in, spacing=args.spacing, exchange_interval=args.exchange_interval,
        distribution=args.distribution, bond_scale=args.bond_scale, bond_mean=args.bond_mean,
        positive_probability=args.positive_probability, field=args.field, seed=args.seed,
        test_every=args.test_every, test_offset=args.test_offset,
        max_spin_proposals=args.max_spin_proposals,
        max_output_bytes=int(args.max_output_gib * 1024**3),
        max_memory_bytes=int(args.max_memory_gib * 1024**3), max_sites=args.max_sites)
    try:
        plan = generation_plan(options)
    except ValueError as exc:
        p.error(str(exc))
    if args.dry_run:
        print(json.dumps(dict(dry_run=True, output=str(args.output), **plan),
                         indent=2, ensure_ascii=False, allow_nan=False))
        return plan
    manifest = generate_dataset(args.output, options)
    print(json.dumps(dict(status=manifest["status"], output=str(args.output),
                         disorders=len(manifest["shards"]), training_updates=0), ensure_ascii=False))
    return manifest


if __name__ == "__main__":
    main()
