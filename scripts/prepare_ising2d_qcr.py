"""Fixed-cluster-count Wolff data, with hot/cold and autocorrelation diagnostics."""
import argparse
import json
from pathlib import Path
import numpy as np
from numba import njit
import torch


@njit
def chain(k, size, draws, equil, gap, seed, cold=False):
    np.random.seed(seed)
    s = np.ones((size, size), dtype=np.int8)
    if not cold:
        for i in range(size):
            for j in range(size):
                s[i, j] = 1 if np.random.random() < .5 else -1
    queue = np.empty(size * size, dtype=np.int64)
    p = 1 - np.exp(-2 * k)
    result = np.empty((draws, size, size), dtype=np.int8)
    for step in range(equil + draws * gap):
        start = np.random.randint(size * size)
        i, j = start // size, start % size
        target = s[i, j]
        s[i, j] = -target
        queue[0] = start
        head, tail = 0, 1
        while head < tail:
            z = queue[head]
            head += 1
            i, j = z // size, z % size
            for ni, nj in (((i-1) % size, j), ((i+1) % size, j),
                           (i, (j-1) % size), (i, (j+1) % size)):
                if s[ni, nj] == target and np.random.random() < p:
                    s[ni, nj] = -target
                    queue[tail] = ni * size + nj
                    tail += 1
        if step >= equil and (step-equil+1) % gap == 0:
            result[(step-equil)//gap] = s
    return result


def observations(x):
    return np.stack(((x * np.roll(x, 1, 1) + x * np.roll(x, 1, 2)).mean((1, 2)),
                     x.mean((1, 2)) ** 2), axis=1)


def stats(x):
    y = observations(x)
    result = {}
    for name, z in zip(('bond_sum_per_site', 'magnetization_squared'), y.T):
        centered = z-z.mean()
        variance = np.mean(centered**2)
        rho = [float(np.mean(centered[:-lag]*centered[lag:])/variance)
               if variance > 0 else 0. for lag in range(1, min(65, len(z)//2))]
        tau = 1.
        for a in range(0, len(rho)-1, 2):
            pair = rho[a]+rho[a+1]
            if pair <= 0:
                break
            tau += 2*pair
        result[name] = {'mean': float(z.mean()), 'lag1': rho[0],
                        'ess_estimate': len(z)/tau,
                        'se_estimate': float(np.sqrt(variance*tau/len(z)))}
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    kc = .5*np.log(1+np.sqrt(2))
    # 64 points, including the exact critical coupling for post-hoc diagnostics.
    knots = np.linspace(.2, .68, 64)
    knots[np.argmin(abs(knots-kc))] = kc
    report = {'sampler': 'Wolff fixed cluster count, numba v2', 'sites': 32,
              'equilibration_cluster_updates': 4096,
              'gap_cluster_updates': '512 for K <= 0.36; 128 otherwise (fixed by K, not chain state)',
              'seed': 91, 'couplings': knots.tolist(), 'pilot': [], 'ensembles': []}
    for i, k in enumerate((.2, kc, .68)):
        pair = []
        for cold in (False, True):
            x = chain(k, 32, 1024, 4096, 512 if k <= .36 else 128, 91000+i*2+int(cold), cold)
            pair.append(stats(x))
        report['pilot'].append({'K': k, 'hot': pair[0], 'cold': pair[1]})
        print('pilot', k, pair, flush=True)
    train, test, train_k, test_k = [], [], [], []
    for i, k in enumerate(knots):
        for replica in range(3):
            x = chain(k, 32, 128, 4096, 512 if k <= .36 else 128, 910000+i*3+replica)
            report['ensembles'].append({'K': k, 'replica': replica, **stats(x)})
            # Sampling augmentation only; even observables are unaffected.
            rng = np.random.default_rng(1910000+i*3+replica)
            x = x * rng.choice(np.array([-1, 1], dtype=np.int8), size=(128, 1, 1))
            target, labels = (test, test_k) if i % 3 == 1 else (train, train_k)
            target.append(torch.from_numpy(x).double().unsqueeze(-1))
            labels.append(k)
        print(f'data {i+1}/64 K={k:.7f}', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save((torch.stack(train), torch.tensor(train_k, dtype=torch.float64),
                torch.stack(test), torch.tensor(test_k, dtype=torch.float64)), args.output)
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    print('saved', args.output, flush=True)


if __name__ == '__main__':
    main()
