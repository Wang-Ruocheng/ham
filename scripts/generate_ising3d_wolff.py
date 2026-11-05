"""生成新的三维最近邻零场铁磁 Ising Wolff 数据；不启动模型训练。"""
import argparse
import json
from pathlib import Path

import numpy as np

from recurrent_rg.ising3d import Wolff3DOptions, generate_dataset, generation_plan


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--size", type=int, default=16)
    p.add_argument("--coupling-min", type=float, default=.12)
    p.add_argument("--coupling-max", type=float, default=.36)
    p.add_argument("--coupling-count", type=int, default=24)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--burn-in-clusters", type=int, default=4096)
    p.add_argument("--clusters-between", type=int, default=64)
    p.add_argument("--seed", type=int, default=20261019)
    p.add_argument("--test-every", type=int, default=3)
    p.add_argument("--test-offset", type=int, default=1)
    p.add_argument("--no-global-z2", action="store_true")
    p.add_argument("--max-working-bytes", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    options = Wolff3DOptions(size=args.size, couplings=tuple(np.linspace(args.coupling_min,
        args.coupling_max, args.coupling_count)), chains=args.chains, samples=args.samples,
        burn_in_clusters=args.burn_in_clusters, clusters_between=args.clusters_between,
        seed=args.seed, test_every=args.test_every, test_offset=args.test_offset,
        augment_global_z2=not args.no_global_z2, max_working_bytes=args.max_working_bytes)
    try:
        plan = generation_plan(options)
    except ValueError as exc:
        p.error(str(exc))
    if args.dry_run:
        print(json.dumps(dict(dry_run=True, output=str(args.output), **plan), ensure_ascii=False, allow_nan=False))
        return plan
    manifest = generate_dataset(args.output, options)
    print(json.dumps(dict(status=manifest["status"], conditions=len(manifest["shards"]),
        output=str(args.output), training_updates=0), ensure_ascii=False, allow_nan=False))
    return manifest


if __name__ == "__main__":
    main()
