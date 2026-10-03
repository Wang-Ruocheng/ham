"""Generate four-state AT ensembles along U/J=2, with sampling diagnostics."""
import argparse
import json
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from recurrent_rg.data.ashkin_teller import sample_ashkin_teller, observables
from recurrent_rg.data.ensembles import EnsembleDataset, grouped_split


def coupling_grid(mode='uniform'):
    if mode == 'uniform':
        return np.linspace(.08, .65, 48)
    if mode == 'middle_dense':
        # Finite-L transition neighborhood, not an exact theoretical phase label.
        return np.concatenate([np.linspace(.08,.185,16),
                               np.linspace(.19,.22,16),
                               np.linspace(.225,.65,16)])
    raise ValueError('Unknown coupling grid')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--length', type=int, default=16)
    p.add_argument('--ratio', type=float, default=2.)
    p.add_argument('--grid', choices=('uniform','middle_dense'), default='uniform')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    xs, labels, audit = [], [], []
    grid = coupling_grid(args.grid)
    for i, K in enumerate(grid):
        for replica in range(3):
            x = sample_ashkin_teller(K, args.length, 128, 91+1000*i+replica,
                                    ratio=args.ratio, burn=1024, spacing=16)
            xs.append(torch.nn.functional.one_hot(torch.from_numpy(x.astype(np.int64)), 4).to(torch.uint8))
            labels.append([K])
            audit.append(dict(K=float(K), replica=replica,
                              **{k: float(v.mean()) for k, v in observables(x,args.ratio).items()}))
        if i % 8 == 0: print(f'Generated {i+1}/48 couplings', flush=True)
    train, test = grouped_split(EnsembleDataset(torch.stack(xs), torch.tensor(labels, dtype=torch.double)))
    torch.save(dict(schema_version=1, train=asdict(train), test=asdict(test)), args.output/'ensembles.pt')
    probes = []
    probe_grid = (.12,.19,.20,.21,.22,.28) if args.grid=='middle_dense' else (.12,.22,.28,.36,.48,.60)
    for K in probe_grid:
        for ordered in (False, True):
            x = sample_ashkin_teller(K, args.length, 1024, 40000+int(K*1000)+int(ordered),
                                    ratio=args.ratio, burn=2048, spacing=16, ordered=ordered)
            stats = {}
            for name, values in observables(x,args.ratio).items():
                batches = values.reshape(-1,32).mean(1)
                stats[name] = dict(mean=float(values.mean()), batch_se=float(batches.std(ddof=1)/np.sqrt(len(batches))))
            probes.append(dict(K=K, ordered_start=ordered, **stats))
    metadata = dict(model='Ashkin-Teller', ratio=args.ratio, L=args.length, K_grid=grid.tolist(),
                    seed=91, replicas=3, samples=128, burn=1024, spacing=16, grid_mode=args.grid,
                    train_ensembles=len(train.configurations), test_ensembles=len(test.configurations),
                    audit=audit, probes=probes)
    (args.output/'sampling.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print('Sampling complete', flush=True)


if __name__ == '__main__':
    main()
