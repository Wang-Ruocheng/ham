"""Single-seed equal-time snapshot QCR for finite-temperature square TFIM."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time
import subprocess

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recurrent_rg.tfim2d import sample_tfim2d_chain
from recurrent_rg.ising2d_field import chain_statistics
from recurrent_rg.model_nd import RecurrentRGNetwork2D
from recurrent_rg.training import train, evaluate
from recurrent_rg.data.statistics import plaquette_distribution
from experiments.qcr_scalar_report import scalar_report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--theory-dim', type=int, choices=[1,2], default=1)
    parser.add_argument('--reuse-pilot-from', type=Path,
                        help='Reuse matching sampler pilot from a completed run, not its weights')
    parser.add_argument('--seed', type=int, default=91)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--pilot-only', action='store_true')
    parser.add_argument('--dtau', type=float, default=.025)
    parser.add_argument('--beta', type=float, default=16.)
    args = parser.parse_args()
    if not np.isfinite(args.dtau) or not np.isfinite(args.beta) or args.dtau <= 0 or args.beta <= 0:
        raise ValueError('beta and dtau must be finite and positive')
    slices = round(args.beta / args.dtau)
    if not np.isclose(slices*args.dtau,args.beta):
        raise ValueError('beta/dtau must be a positive integer')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(dimension=2, sites=16, levels=3, theory_dim=args.theory_dim, hidden=60,
               replicas=3, ensemble_size=128, time_slices=slices, beta_over_slices=args.dtau,
               equilibration_sweeps=256, decorrelation_sweeps=8, epochs=args.epochs,
               system_batch=32, sample_batch=128, eval_samples=48, eval_systems=24,
               learning_rate=.001, seed=args.seed, threads=args.threads,
               objective_mode='data_geometry', geometry_scale=4., noise_multiplier=2.,
               rollout_weight=5., kernel_layout='nonoverlapping', coordinate_reduction='sum', binary_lookup=True,
               reference_critical_coupling=3.04438, physical_model='2D TFIM',
               output_dir=str(args.output_dir), fixed_kernel=False)
    contract = dict(optimized_modules=['Q','C','R'], couplings_used_in_loss=False,
                    action_used=False, rsmi_used=False, inverse_temperature=args.beta,
                    sampler='2+1D anisotropic Swendsen-Wang Trotter Ising',
                    snapshot='one random equal-time slice per saved sweep interval',
                    ground_state_convergence_claimed=False,
                    continuum_convergence_claimed=False,
                    spin_flip_augmentation=True, rollout_horizons=[1,2])
    (args.output_dir/'run_config.json').write_text(json.dumps(dict(config=cfg, training_contract=contract), indent=2)+'\n')
    # Compile outside worker pool. Each independent chain has an explicit seed.
    sample_tfim2d_chain(3., size=2, slices=2, draws=2, equil=2, gap=1)
    pilots = []
    for h in (2., 3.04438, 4.):
        for nt, dtau, initial in ((slices,args.dtau,0),(slices,args.dtau,1),
                                 (2*slices,args.dtau/2,0),(2*slices,args.dtau,0)):
            pilots.append((h,nt,dtau,initial))
    def pilot(job):
        h,slices,dtau,initial = job
        start = time.monotonic()
        x = sample_tfim2d_chain(h, size=16, slices=slices, dtau=dtau,
                               draws=256, equil=256, gap=8,
                               seed=args.seed+10000+pilots.index(job), initial=initial)
        row = dict(h_over_J=h, time_slices=slices, dtau=dtau, beta=slices*dtau,
                   initial=initial, statistics=chain_statistics(x), seconds=time.monotonic()-start)
        print('PILOT',json.dumps(row),flush=True)
        return row
    pilot_path = args.output_dir/'sampler_pilot.json'
    if args.reuse_pilot_from is not None:
        previous=json.loads((args.reuse_pilot_from/'summary.json').read_text())['config']
        for key in ['sites','time_slices','beta_over_slices','seed','equilibration_sweeps','decorrelation_sweeps']:
            if previous[key] != cfg[key]:
                raise ValueError(f'Pilot configuration mismatch: {key}')
        pilot_rows=json.loads((args.reuse_pilot_from/'sampler_pilot.json').read_text())
        expected={(float(h),int(nt),float(dt),int(init)) for h,nt,dt,init in pilots}
        observed={(float(r['h_over_J']),int(r['time_slices']),float(r['dtau']),int(r['initial'])) for r in pilot_rows}
        if expected != observed:
            raise ValueError('Pilot jobs do not match the requested sampler')
        pilot_path.write_text(json.dumps(pilot_rows,indent=2)+'\n')
        (args.output_dir/'pilot_provenance.json').write_text(json.dumps({
            'source':str(args.reuse_pilot_from),'reused_sampler_audit_only':True,
            'model_weights_reused':False},indent=2)+'\n')
    if not pilot_path.exists():
        with ThreadPoolExecutor(args.workers) as pool:
            results = list(pool.map(pilot,pilots))
        pilot_path.write_text(json.dumps(results,indent=2)+'\n')
    if args.pilot_only:
        return
    dtau_tag = str(args.dtau).replace('.', 'p')
    cache = Path('data_cache')/f'tfim2d_sw_v1_L16_M{slices}_dtau{dtau_tag}_eq256_gap8_K64_rep3_n128_seed{args.seed}.pt'
    cfg['data_cache'] = str(cache)
    if cache.exists():
        tx,th,vx,vh = torch.load(cache,weights_only=True)
    else:
        grid = np.linspace(.5,5.,64)
        jobs = [(i,r,float(h)) for i,h in enumerate(grid) for r in range(3)]
        def sample(job):
            i,r,h = job
            x=sample_tfim2d_chain(h,slices=slices,dtau=args.dtau,seed=args.seed+1000+i*3+r)
            print(f'DATA {i+1}/64 replica={r+1} h/J={h:.4f}',flush=True)
            return x,chain_statistics(x)
        with ThreadPoolExecutor(args.workers) as pool:
            data=list(pool.map(sample,jobs))
        x=torch.from_numpy(np.stack([v[0] for v in data])).unsqueeze(-1).double()
        labels=torch.tensor([v[2] for v in jobs],dtype=torch.float64)
        mask=torch.tensor([v[0]%3 != 1 for v in jobs])
        tx,th,vx,vh=x[mask],labels[mask],x[~mask],labels[~mask]
        torch.save((tx,th,vx,vh),cache)
        cache.with_suffix('.json').write_text(json.dumps(dict(config=cfg, jobs=jobs,
                       statistics=[v[1] for v in data]),indent=2)+'\n')
    torch.manual_seed(args.seed); torch.set_num_threads(args.threads)
    model=RecurrentRGNetwork2D(levels=3,theory_dim=args.theory_dim,hidden=60).double()
    model.theory_encoder.binary_lookup=True; model.rg_kernel.binary_lookup=True
    count=sum(p.numel() for module in (model.theory_encoder,model.theory_rg,model.rg_kernel) for p in module.parameters())
    print(f'TRAIN data={tuple(tx.shape)}/{tuple(vx.shape)} optimized_parameters={count}',flush=True)
    history=train(model,tx,vx,pattern_fn=plaquette_distribution,epochs=args.epochs,
                  system_batch=32,sample_batch=128,eval_samples=48,eval_systems=24,
                  learning_rate=.001,seed=args.seed,progress_dir=args.output_dir,
                  objective_kwargs=dict(objective_mode='data_geometry',geometry_scale=4.,
                                        noise_multiplier=2.,rollout_weight=5.,coordinate_reduction='sum'))
    summary=dict(config=cfg,training_contract=contract,optimized_parameters=count,
                 data=dict(train_shape=list(tx.shape),test_shape=list(vx.shape)),history=history)
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    torch.save({k:v for k,v in model.state_dict().items() if not k.startswith('action_network.')},args.output_dir/'qcr_model.pt')
    if args.theory_dim==1:
        scalar_report(args.output_dir,cache,dimension=2)
    else:
        for step in sorted({s for s in [2000,4000,args.epochs] if s<=args.epochs}):
            path=args.output_dir/f'training_step_{step}.pt'
            saved=torch.load(path if path.exists() else args.output_dir/'qcr_model.pt',weights_only=True)
            state=saved.get('model',saved)
            model.load_state_dict(state,strict=False)
            row={'epoch':step}
            for name,x in [('train',tx),('test',vx)]:
                ids=torch.linspace(0,len(x)-1,min(cfg['eval_systems'],len(x))).long()
                vals=evaluate(model,x[ids],plaquette_distribution,cfg['eval_samples'],objective_kwargs=dict(
                    objective_mode='data_geometry',geometry_scale=4.,noise_multiplier=2.,rollout_weight=5.,coordinate_reduction='sum'))
                row.update({name+'_'+k:v for k,v in vals.items()})
            dest=args.output_dir/f'step{step}';dest.mkdir(exist_ok=True)
            torch.save(state,dest/'qcr_model.pt')
            partial=dict(summary,config=dict(cfg,epochs=step,output_dir=str(dest)),
                         history=[r for r in history if r['epoch']<step]+[row])
            (dest/'summary.json').write_text(json.dumps(partial,indent=2)+'\n')
            subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'scripts/plot_qcr_vector_flow.py'),
                            '--run',str(dest),'--cache',str(cache)],check=True)


if __name__=='__main__':
    main()
