"""Generate a two-parameter Ising ensemble grid, including independent +/-B chains."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recurrent_rg.ising2d_field import sample_field_chain, chain_statistics


def parameter_grid(b_step=.05):
    if b_step not in (.05, .1):
        raise ValueError('B step must be 0.05 or 0.1')
    return np.linspace(.2, .68, 33), np.linspace(-.5, .5, round(1/b_step)+1)


def sample_row(task):
    i, k, bs = task
    result = []
    for j, b in enumerate(bs):
        held_out = (i+abs(j-len(bs)//2)) % 3 == 1
        for r in range(3):
            seed = 910000+i*len(bs)*3+j*3+r
            x = sample_field_chain(k, b, 32, 256, 512, 16, seed)
            record = {'K':float(k), 'B':float(b), 'replica':r, 'test':held_out,
                      'chain_seed':seed, **chain_statistics(x)}
            result.append((x, record))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--b-step', type=float, choices=(.05,.1), default=.05)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    if args.output.exists() or args.output.with_suffix('.json').exists():
        raise FileExistsError(args.output)
    if args.workers < 1:
        p.error('--workers must be positive')
    ks, bs = parameter_grid(args.b_step)
    report = {'sampler': 'Swendsen-Wang cluster heat bath, fixed sweeps, field v1',
              'grid': 'uniform_v2', 'K_step': .015, 'B_step':args.b_step,
              'critical_point_replacement':False,
              'K': ks.tolist(), 'B': bs.tolist(), 'sites': 32, 'replicas': 3, 'draws': 256,
              'thermalization_sweeps': 512, 'sweeps_between': 16,
              'global_flip_augmentation': False, 'pilot': [], 'ensembles': []}
    for i, (k, b) in enumerate(((.2, .5), (.44, 0.), (.44, args.b_step), (.68, 0.), (.68, args.b_step), (.68, -args.b_step))):
        diagnostics = []
        for j, initial in enumerate((0, 1, -1)):
            x = sample_field_chain(k, b, 32, 1024, 512, 16, 192000+i*3+j, initial)
            diagnostics.append({'initial': initial, **chain_statistics(x)})
        report['pilot'].append({'K': k, 'B': b, 'chains': diagnostics})
        print('pilot', k, b, diagnostics, flush=True)
    train, test, train_labels, test_labels = [], [], [], []
    # Fixed row order and per-chain seeds make output independent of worker count.
    # Hold out entire cells, including every replica and the +/-B counterpart.
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(sample_row, [(i,k,bs) for i,k in enumerate(ks)])):
            for x, record in row:
                report['ensembles'].append(record)
                target, labels = (test, test_labels) if record['test'] else (train, train_labels)
                target.append(torch.from_numpy(x).unsqueeze(-1))
                labels.append((record['K'], record['B']))
            print(f'K row {i+1}/{len(ks)}', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save((torch.stack(train), torch.tensor(train_labels, dtype=torch.float64),
                torch.stack(test), torch.tensor(test_labels, dtype=torch.float64)), args.output)
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    print('saved', args.output, flush=True)


if __name__ == '__main__':
    main()
