"""Joint unsupervised QCR on two-parameter classical Ising ensembles."""
import argparse
import json
from pathlib import Path
import sys
import subprocess
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recurrent_rg.model_nd import RecurrentRGNetwork2D, FixedDecimationCoarseGrainer
from experiments.joint_qcr_ising import train, plaquette_distribution


def build_model(learnable_kernel=True):
    # Construct the baseline first so removing C does not shift Q/R's RNG
    # initialization. This makes the fixed-C run a paired-seed control.
    model = RecurrentRGNetwork2D(levels=4, hidden=60, theory_dim=2).double()
    if not learnable_kernel:
        model.rg_kernel = FixedDecimationCoarseGrainer(model.geometry)
    model.theory_encoder.binary_lookup = model.rg_kernel.binary_lookup = True
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=2000)
    p.add_argument('--seed', type=int, default=91)
    p.add_argument('--coordinate-reduction', choices=('mean', 'sum'), default='sum',
                   help='Use mean explicitly to reproduce historical vector loss normalization')
    p.add_argument('--learnable-kernel', action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    train_x, train_kb, test_x, test_kb = torch.load(args.cache, weights_only=True)
    assert train_kb.shape[1] == test_kb.shape[1] == 2
    assert train_x.shape[1:] == test_x.shape[1:] == (256, 32, 32, 1)
    model = build_model(args.learnable_kernel)
    objective = dict(objective_mode='data_geometry', geometry_scale=4., noise_multiplier=2.,
                     rollout_weight=5., kernel_noncollapse_mode='variance_floor', kernel_variance_ratio=.25,
                     coordinate_reduction=args.coordinate_reduction)
    cfg = {'dimension': 2, 'sites': 32, 'levels': 4, 'theory_dim': 2, 'hidden': 60,
           'kernel_layout': 'nonoverlapping', 'binary_lookup': True,
           'coordinate_reduction': args.coordinate_reduction,
           'learnable_kernel': args.learnable_kernel,
           'initialization': 'Q/R initialized identically to learned-C baseline at the same seed',
           'seed': args.seed, 'epochs': args.epochs, 'system_batch': 32, 'sample_batch': 256,
           'eval_systems': 48, 'eval_samples': 128, 'learning_rate': .001, 'threads': 4,
           'data_cache': str(args.cache.resolve()), 'spin_flip_augmentation': False,
           'objective_kwargs': objective}
    contract = {'couplings_used_in_loss': False, 'rsmi_used': False, 'action_used': False,
                'optimized_modules': ['Q', 'C', 'R'] if args.learnable_kernel else ['Q', 'R'],
                'kernel': 'learned' if args.learnable_kernel else 'fixed positive decimation: X[..., ::2, ::2, :]',
                'explicit_Z2_constraint': not args.learnable_kernel,
                'symmetry_note': 'Fixed decimation is Z2-equivariant by construction; Q/R remain unconstrained.' if not args.learnable_kernel else 'No hard symmetry constraints',
                'global_flip_augmentation': False, 'coordinate_names': ['K=beta*J', 'B=beta*h'],
                'kernel_noncollapse': 'mean_k relu(|mean(D[k+1])|-sqrt(1-0.25*(1-stopgrad(mean(D[k]))^2)))^2; global batch means',
                'separation': 'equal-weight local distribution geometry at all four scales; detached noise-adjusted targets',
                'loss_weights': {'anchor': 2, 'coarse_consistency': 5, 'kernel_noncollapse': 2,
                                 'separation': 10, 'commutator': 5, 'rollout': 5}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {'config': cfg, 'training_contract': contract,
               'data': {'train_shape': list(train_x.shape), 'test_shape': list(test_x.shape)},
               'parameter_count': sum(p.numel() for p in model.parameters()),
               'optimized_parameter_count': sum(p.numel() for mod in (model.theory_encoder, model.rg_kernel, model.theory_rg) for p in mod.parameters())}
    (args.output_dir/'run_config.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)
    history = train(model, train_x, test_x, pattern_fn=plaquette_distribution, epochs=args.epochs,
                    system_batch=32, sample_batch=256, eval_samples=128, eval_systems=48,
                    learning_rate=.001, seed=args.seed, progress_dir=args.output_dir,
                    spin_flip_augment=False, objective_kwargs=objective)
    summary['history'] = history
    torch.save({k: v for k, v in model.state_dict().items() if not k.startswith('action_network.')},
               args.output_dir/'qcr_model.pt')
    (args.output_dir/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('finished', history[-1], flush=True)
    plotter = Path(__file__).resolve().parents[1]/'scripts/plot_ising2d_field_qcr.py'
    for step in sorted({1000, args.epochs}):
        if step > args.epochs:
            continue
        command = [sys.executable, str(plotter), '--run', str(args.output_dir),
                   '--output-dir', str(args.output_dir/f'step{step}')]
        if step != args.epochs:
            command += ['--checkpoint', str(args.output_dir/f'training_step_{step}.pt')]
        subprocess.run(command, check=True)
        subprocess.run([sys.executable, str(plotter.with_name('plot_field_qcr_scale_trajectories.py')),
                        '--run', str(args.output_dir/f'step{step}')], check=True)
    subprocess.run([sys.executable, str(plotter.with_name('report_ising2d_field_run.py')),
                    '--run', str(args.output_dir)], check=True)


if __name__ == '__main__':
    main()
