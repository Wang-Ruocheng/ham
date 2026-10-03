"""Minute-scale exploratory QCR run with a field grid concentrated near zero.

Uses the unchanged data-geometry objective. No coupling labels enter training.
This changes lattice size and sampling as well as budget: not a controlled
comparison with the large uniform-grid experiment.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recurrent_rg.ising2d_field import sample_field_chain, chain_statistics
from recurrent_rg.model_nd import RecurrentRGNetwork2D
from recurrent_rg.training import train
from recurrent_rg.data.statistics import plaquette_distribution


def focused_grid():
    ks = np.linspace(.2, .68, 17)
    positive = np.array([.0005, .001, .002, .005, .01, .02, .05, .2, .5])
    return ks, np.concatenate((-positive[::-1], [0.], positive))


def sample_cell(task):
    i, j, k, b, seed, sites, draws = task
    rows = []
    for replica in range(2):
        chain_seed = seed*100000+i*1000+j*2+replica
        x = sample_field_chain(k, b, sites, draws, 512, 16, chain_seed,
                               1 if replica == 0 else -1)
        record = dict(K=float(k), B=float(b), replica=replica, chain_seed=chain_seed,
                      test=bool((i+abs(j-9))%3 == 1), **chain_statistics(x))
        rows.append((x, record))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=91)
    args = parser.parse_args()
    if args.epochs < 1 or args.workers < 1:
        parser.error('Epochs and workers must be positive')
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(2)
    sites, draws, levels = 16, 64, 3
    ks, bs = focused_grid()
    audit = dict(K=ks.tolist(), B=bs.tolist(), sites=sites, draws=draws, replicas=2,
                 seed=args.seed, sampler='finite-field Swendsen-Wang cluster heat bath',
                 thermalization_sweeps=512, sweeps_between=16,
                 split='Hold out all replicas and both field signs of each selected cell',
                 pilot=[], ensembles=[])
    for index, (k,b) in enumerate(((.44, .002), (.68, .0005))):
        rows = []
        for j, initial in enumerate((0, 1, -1)):
            x = sample_field_chain(k, b, sites, 384, 512, 16, args.seed*100000+90000+3*index+j, initial)
            rows.append(dict(initial=initial, **chain_statistics(x)))
        audit['pilot'].append(dict(K=k, B=b, chains=rows))
        print('pilot', k, b, rows, flush=True)
    tx, ty, vx, vy = [], [], [], []
    tasks = [(i,j,float(k),float(b),args.seed,sites,draws) for i,k in enumerate(ks) for j,b in enumerate(bs)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n, rows in enumerate(pool.map(sample_cell, tasks), 1):
            for x, record in rows:
                audit['ensembles'].append(record)
                data, labels = (vx,vy) if record['test'] else (tx,ty)
                data.append(torch.from_numpy(x).unsqueeze(-1))
                labels.append((record['K'], record['B']))
            if n%19 == 0:
                print(f'Data: {n}/{len(tasks)} cells, elapsed {time.perf_counter()-started:.1f}s', flush=True)
    tx, vx = torch.stack(tx), torch.stack(vx)
    ty, vy = torch.tensor(ty,dtype=torch.float64), torch.tensor(vy,dtype=torch.float64)
    cache = out/'ensembles.pt'
    torch.save((tx,ty,vx,vy), cache)
    (out/'sampling_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    data_seconds = time.perf_counter()-started
    torch.manual_seed(args.seed)
    model = RecurrentRGNetwork2D(levels=levels, hidden=60, theory_dim=2).double()
    model.theory_encoder.binary_lookup = model.rg_kernel.binary_lookup = True
    objective = dict(objective_mode='data_geometry', geometry_scale=4., noise_multiplier=2.,
                     rollout_weight=5., kernel_noncollapse_mode='variance_floor',
                     kernel_variance_ratio=.25, coordinate_reduction='sum')
    cfg = dict(dimension=2, sites=sites, levels=levels, theory_dim=2, hidden=60,
               kernel_layout='nonoverlapping', binary_lookup=True, learnable_kernel=True,
               seed=args.seed, epochs=args.epochs, system_batch=8, sample_batch=draws,
               eval_systems=24, eval_samples=draws, learning_rate=.001, threads=2,
               data_cache=str(cache.resolve()), spin_flip_augmentation=False,
               coordinate_reduction='sum', objective_kwargs=objective, report_steps=[args.epochs])
    summary = dict(config=cfg, data=dict(train_shape=list(tx.shape),test_shape=list(vx.shape)),
                   optimized_parameter_count=sum(p.numel() for mod in (model.theory_encoder,model.rg_kernel,model.theory_rg) for p in mod.parameters()),
                   training_contract=dict(couplings_used_in_loss=False, action_used=False, rsmi_used=False,
                       optimized_modules=['Q','C','R'], hard_symmetry_constraints=False),
                   caution='Exploratory finite-size run; multiple settings changed relative to large baseline; no causal attribution to sampling alone.',
                   timing=dict(data_seconds=data_seconds))
    (out/'run_config.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('TRAIN',cfg,'optimized_parameters',summary['optimized_parameter_count'],flush=True)
    training_started = time.perf_counter()
    summary['history'] = train(model,tx,vx,pattern_fn=plaquette_distribution,epochs=args.epochs,
        system_batch=cfg['system_batch'],sample_batch=draws,eval_samples=draws,eval_systems=24,
        learning_rate=.001,seed=args.seed,progress_dir=out,spin_flip_augment=False,objective_kwargs=objective)
    summary['timing']['training_seconds'] = time.perf_counter()-training_started
    torch.save({k:v for k,v in model.state_dict().items() if not k.startswith('action_network.')},out/'qcr_model.pt')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    plotting_started = time.perf_counter()
    subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'scripts/plot_ising2d_field_qcr.py'),
                    '--run',str(out),'--output-dir',str(out/f'step{args.epochs}'),'--threads','1'],check=True)
    summary['timing']['plotting_seconds'] = time.perf_counter()-plotting_started
    summary['timing']['total_seconds'] = time.perf_counter()-started
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('TIMING',json.dumps(summary['timing']),flush=True)


if __name__ == '__main__':
    main()
