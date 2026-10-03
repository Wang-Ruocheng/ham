"""Train C solely by RSMI, then freeze it and fit the existing Q/R objective.

No coupling labels enter either optimizer. Known rules get separately fitted
critics with the same architecture and batch schedule for diagnostic baselines.
"""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recurrent_rg.config import RunConfig
from recurrent_rg.checkpoints import build_model, qcr_state_dict
from recurrent_rg.runner import prepared_data, file_fingerprint, run_training, evaluate_run
from recurrent_rg.rsmi import (EnvironmentCritic, enumerated_infonce, kernel_probabilities,
                               reference_probabilities, region_offsets, sample_regions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=2400)
    parser.add_argument('--baseline-steps', type=int, default=2400)
    parser.add_argument('--qr-steps', type=int, default=2400)
    parser.add_argument('--buffer', type=int, default=1)
    args = parser.parse_args()
    cfg = RunConfig.load(args.config)
    if cfg.model.fixed_kernel or cfg.model.categorical_states or cfg.model.kernel_layout != 'nonoverlapping':
        raise ValueError('This RSMI experiment requires a learnable binary nonoverlapping C')
    if cfg.model.block_factor != 2 or min(args.steps, args.baseline_steps) < 1:
        raise ValueError('Invalid run settings')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(cfg.training.seed)
    model = build_model(cfg.model, cfg.training.max_rg_steps)
    train, test = prepared_data(cfg)
    tx, vx = train.configurations, test.configurations
    pairs = min(64, tx.shape[1], vx.shape[1])
    systems = min(16, len(tx))
    _, env_offsets = region_offsets(cfg.model.dimension, buffer=args.buffer)
    env_size = len(env_offsets)
    started = time.perf_counter()
    manifest = dict(source_config=cfg.to_dict(), objective='negative exact-H InfoNCE, within ensemble',
        seed=cfg.training.seed, steps=args.steps, baseline_steps=args.baseline_steps,
        qr_steps=args.qr_steps, systems=systems, pairs=pairs, buffer=args.buffer,
        visible_sites=2**cfg.model.dimension, environment_sites=env_size,
        data_sha256=file_fingerprint(cfg.data.cache), train_shape=list(tx.shape), test_shape=list(vx.shape),
        labels_used_in_loss=False, rsmi_scales=[0], critic_hidden=64,
        kernel_auxiliary_losses=False, gradient='exact sum over binary H; no ST in RSMI',
        note='Shared kernel and critic across ensembles; not a verbatim reproduction of a paper experiment.')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

    def get_p(v, rule):
        return kernel_probabilities(model.rg_kernel, v) if rule == 'learned' else reference_probabilities(v, rule)

    def evaluate(critic, x, rule, *, shuffled=False, repeats=8):
        gen = torch.Generator().manual_seed(818)
        bounds, entropy = [], []
        with torch.no_grad():
            for _ in range(repeats):
                v,e = sample_regions(x, min(32,len(x)), pairs, gen, buffer=args.buffer,
                                     flip=cfg.data.spin_flip_augment)
                p = get_p(v,rule)
                if shuffled:
                    # Independent within-ensemble permutations; never change marginals.
                    e = torch.stack([row[torch.randperm(pairs,generator=gen)] for row in e])
                bounds.extend((enumerated_infonce(p,critic(e))/math.log(2)).tolist())
                marginal = p.mean(1).clamp_min(1e-15)
                entropy.extend((-(marginal*marginal.log()).sum(-1)/math.log(2)).tolist())
        return dict(bound_bits=float(np.mean(bounds)), output_entropy_bits=float(np.mean(entropy)))

    reports = {}
    learned_history = None
    for rule in ('learned', 'decimation', 'majority'):
        # Matched initial critic parameters and batches across reference rules.
        torch.manual_seed(cfg.training.seed+1)
        critic = EnvironmentCritic(env_size).double()
        parameters = list(critic.parameters())
        if rule == 'learned':
            parameters += list(model.rg_kernel.parameters())
        optimizer = torch.optim.Adam(parameters,lr=.001)
        gen = torch.Generator().manual_seed(cfg.training.seed+2)
        history = []
        steps = args.steps if rule == 'learned' else args.baseline_steps
        for step in range(steps+1):
            if step:
                v,e = sample_regions(tx, systems, pairs, gen, buffer=args.buffer,
                                     flip=cfg.data.spin_flip_augment)
                loss = -enumerated_infonce(get_p(v,rule),critic(e)).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite RSMI loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            if step % 200 == 0 or step == steps:
                tr,te = evaluate(critic,tx,rule),evaluate(critic,vx,rule)
                row = dict(step=step,train=tr,test=te)
                history.append(row)
                (args.output/f'{rule}_history.json').write_text(json.dumps(history,indent=2)+'\n')
                print(f'{rule} {step}/{steps}: train I>={tr["bound_bits"]:.5f} test I>={te["bound_bits"]:.5f} bits',flush=True)
        reports[rule] = dict(train=tr,test=te,shuffled_test=evaluate(critic,vx,rule,shuffled=True))
        torch.save(dict(critic=critic.state_dict(),optimizer=optimizer.state_dict(),step=steps),
                   args.output/f'{rule}_critic.pt')
        if rule == 'learned':
            learned_history = history
            torch.save(qcr_state_dict(model),args.output/'rsmi_qcr_init.pt')
            width = 2**cfg.model.dimension
            patterns = (2*((torch.arange(2**width)[:,None]//(2**torch.arange(width)))%2)-1).double()
            with torch.no_grad():
                p = kernel_probabilities(model.rg_kernel,patterns[None])[0,:,1]
            reports['kernel'] = dict(patterns=patterns.tolist(),p_plus=p.tolist(),
                uniform_output_separation=float(abs(p[-1]-p[0])),
                mean_sampling_variance_uniform_patterns=float((4*p*(1-p)).mean()))
    reports['random'] = dict(population_mi_bits=0, note='Input-independent fair coin; analytic zero MI')
    reports['rsmi_seconds'] = time.perf_counter()-started
    (args.output/'rsmi_results.json').write_text(json.dumps(reports,indent=2)+'\n')
    fig,axes = plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for split in ('train','test'):
        axes[0].plot([r['step'] for r in learned_history],[-r[split]['bound_bits'] for r in learned_history],label=split)
    axes[0].set(xlabel='C + critic updates',ylabel='RSMI loss = -InfoNCE (bits)',title='Pure RSMI; lower is better')
    axes[0].legend();axes[0].grid(alpha=.2)
    names=['learned','decimation','majority']
    axes[1].bar(names,[reports[n]['test']['bound_bits'] for n in names])
    axes[1].axhline(0,color='k',lw=.6)
    axes[1].set(ylabel='Held-out InfoNCE (bits)',title='Separate critics; estimates, not exact MI')
    fig.savefig(args.output/'rsmi_loss.png',dpi=170);plt.close(fig)
    if args.qr_steps:
        qr_cfg = replace(cfg,schema_version=4,
            loss=replace(cfg.loss,kernel_sign_weight=0.,weights=replace(cfg.loss.weights,kernel_noncollapse=0.)),
            training=replace(cfg.training,epochs=args.qr_steps,train_modules=('Q','R'),
                system_batch=32,sample_batch=pairs,checkpoint_every=400,log_every=100),
            evaluation=replace(cfg.evaluation,samples=pairs,systems=32))
        (args.output/'qr_config.json').write_text(json.dumps(qr_cfg.to_dict(),indent=2)+'\n')
        run_training(qr_cfg,args.output/'qr',initialize_from=args.output/'rsmi_qcr_init.pt')
        evaluate_run(args.output/'qr',output=args.output/'qr'/'report')
        # Explicitly verify no downstream objective changed C.
        final = torch.load(args.output/'qr'/'qcr_model.pt',weights_only=True)
        initial = torch.load(args.output/'rsmi_qcr_init.pt',weights_only=True)
        assert all(torch.equal(final[k],v) for k,v in initial.items() if k.startswith('rg_kernel.'))
        reports['C_unchanged_during_QR'] = True
        h=json.loads((args.output/'qr'/'history_progress.json').read_text())
        fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
        for split in ('train','test'):
            ax.plot([r['epoch'] for r in h],[r[split+'_total'] for r in h],label=split)
        ax.set(xlabel='Q/R updates (C frozen)',ylabel='Total Q/R loss',yscale='log');ax.legend()
        fig.savefig(args.output/'qr'/'report'/'loss.png',dpi=170);plt.close(fig)
    reports['total_seconds'] = time.perf_counter()-started
    (args.output/'rsmi_results.json').write_text(json.dumps(reports,indent=2)+'\n')
    print(json.dumps(reports,indent=2),flush=True)


if __name__ == '__main__':
    main()
