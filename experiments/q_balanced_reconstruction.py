"""Paired Q-only warm-start ablation; source probabilistic C/R remain frozen.

One common stochastic C draw is cached as exact sparse per-sample local
histograms. Full Q, including its local MLP, remains trainable. No MI is
computed and physical labels enter evaluation/plots only.
"""
import copy
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
from scipy import sparse
import torch

from recurrent_rg.balanced_reconstruction import fit_balanced_metric, balanced_reconstruction_loss
from recurrent_rg.checkpoints import load_run, qcr_state_dict
from recurrent_rg.config import RunConfig
from recurrent_rg.data.ensembles import load_ensembles
from recurrent_rg.data.statistics import categorical_plaquette_distribution
from recurrent_rg.representation import normalize_affine_decoder_, linear_distribution_reconstruction_loss
from recurrent_rg.runner import file_fingerprint
from experiments.q_only_cached_sweep import sample_counts


def encode_sparse(q, frequencies, states, volume):
    """Exact encoder on observed categorical patterns, including Q gradients."""
    h = frequencies.tocoo()
    codes, inverse = np.unique(h.col, return_inverse=True)
    powers = states ** torch.arange(volume)
    digits = (torch.from_numpy(codes)[:, None] // powers) % states
    patterns = torch.nn.functional.one_hot(digits, states).flatten(1).double()
    indices = torch.from_numpy(np.stack([h.row, inverse])).long()
    hist = torch.sparse_coo_tensor(indices, torch.from_numpy(h.data).double(),
                                   (h.shape[0], len(codes))).coalesce()
    return q.head(torch.sparse.mm(hist, q.local_mlp(patterns)))


def make_cache(model, datasets, steps, states):
    cache = {}
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(9191)
        for split, dataset in datasets.items():
            counts, probabilities = [], []
            for i, x in enumerate(dataset.configurations):
                levels = model.coarse_grain(x[None].double(), max_rg_steps=steps)
                # Canonicalize harmless STE roundoff to hard one-hot states.
                levels = [torch.nn.functional.one_hot(v.argmax(-1), states).double() for v in levels]
                counts.append([sample_counts(v, model.geometry, states) for v in levels])
                probabilities.append(torch.stack([
                    categorical_plaquette_distribution(v.squeeze(0).unsqueeze(1)) for v in levels]).numpy())
                if (i + 1) % 100 == 0:
                    print(f'cache {split}: {i+1}/{len(dataset.configurations)}', flush=True)
            cache[split] = (counts, np.stack(probabilities))
    return cache


def batch(cache, split, ids, halves):
    counts, probabilities = cache[split]
    rows = [sparse.csr_matrix(counts[i][k][half].mean(0))
            for i in ids for k in range(len(counts[i])) for half in halves]
    h = sparse.vstack(rows, format='csr')
    p = np.stack([probabilities[ids][:, :, half].mean(2) for half in halves], axis=2)
    return h, torch.from_numpy(p)


def objective(q, h, p, metrics, beta, states, volume, weights):
    g = encode_sparse(q, h, states, volume).reshape(*p.shape[:3], -1)
    direct = g.mean(2)
    anchor = (g[:, 0, 0] - g[:, 0, 1]).square().sum(-1).mean()
    consistency = (g[:, 1:, 0] - g[:, 1:, 1]).square().sum(-1).mean()
    decoder = q.distribution_decoder
    rec = torch.stack([linear_distribution_reconstruction_loss(decoder, direct[:, k], p[:, k].mean(1))
                       for k in range(p.shape[1])]).mean()
    balanced = torch.stack([balanced_reconstruction_loss(decoder, direct[:, k], p[:, k].mean(1), metrics[k])
                            for k in range(p.shape[1])]).mean()
    total = weights[0] * anchor + weights[1] * consistency + weights[2] * rec + beta * balanced
    return total, dict(anchor=anchor, consistency=consistency, reconstruction=rec,
                       balanced=balanced, total=total), direct


def run(source, output, epochs=400, *, max_wall_seconds=540., beta=1.):
    started = time.monotonic()
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    loaded = load_run(source); cfg = RunConfig.from_dict(loaded.config)
    if not cfg.model.categorical_states or not cfg.model.affine_reconstruction or cfg.model.theory_dim != 2:
        raise ValueError('This pilot requires an existing categorical affine dimg=2 run')
    if cfg.data.spin_flip_augment or cfg.training.sample_batch % 2:
        raise ValueError('Pilot does not change augmentation or support odd subset sizes')
    model = loaded.model
    source_state = {k: v.clone() for k, v in qcr_state_dict(model).items()}
    for parameter in model.parameters(): parameter.requires_grad_(False)
    train, test = load_ensembles(cfg.data.cache)
    datasets = dict(train=train, test=test)
    steps = cfg.training.max_rg_steps; states = cfg.model.categorical_states
    cache = make_cache(model, datasets, steps, states)
    ns = cfg.training.sample_batch
    halves = [np.arange(ns//2), np.arange(ns//2, ns)]
    evals = {s: batch(cache, s, np.arange(len(d.configurations)), halves) for s, d in datasets.items()}
    train_p = evals['train'][1]
    metrics = [fit_balanced_metric(train_p[:, k, 0], train_p[:, k, 1],
                                   max_directions=cfg.model.theory_dim) for k in range(steps+1)]
    arms = dict(baseline=0., balanced=beta)
    qs = {name: copy.deepcopy(model.theory_encoder) for name in arms}
    for q in qs.values():
        for parameter in q.parameters(): parameter.requires_grad_(True)
    opts = {name: torch.optim.Adam(q.parameters(), lr=cfg.training.learning_rate) for name, q in qs.items()}
    weights = (10*cfg.loss.weights.anchor, cfg.loss.weights.coarse_consistency, cfg.loss.weights.separation)
    # Verify both forward and full encoder gradient against the uncached model.
    original = copy.deepcopy(qs['baseline'])
    cached = copy.deepcopy(original)
    h, p = batch(cache, 'train', [0], [np.arange(ns)])
    a = original(train.configurations[:1, :ns].double())
    b = encode_sparse(cached, h[:1], states, model.geometry.patch_volume)
    torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)
    params_a = [v for n, v in original.named_parameters() if not n.startswith('distribution_decoder.')]
    params_b = [v for n, v in cached.named_parameters() if not n.startswith('distribution_decoder.')]
    ga = torch.autograd.grad(a.sum(), params_a); gb = torch.autograd.grad(b.sum(), params_b)
    gradient_error = max(float((x-y).abs().max()) for x, y in zip(ga, gb))
    if gradient_error > 1e-9: raise AssertionError('Sparse cache changed Q gradients')
    effective = replace(cfg, training=replace(cfg.training, epochs=epochs, train_modules=('Q',)))
    manifest = dict(config=effective.to_dict(), source=str(source), source_sha256=file_fingerprint(loaded.checkpoint),
                    data_sha256=file_fingerprint(cfg.data.cache),
                    ablation='Paired Q-only warm start; frozen source C/R; reset paired Adam',
                    additional_balanced_weights=arms, original_Q_weights=list(weights),
                    metrics_by_scale=[m.description() for m in metrics], coarse_draw_seed=9191,
                    model_initialization='Unchanged source Q/C/R; no PCA/model reinitialization',
                    cache_gradient_error=gradient_error, labels_used_in_training=False,
                    limitation='One common fixed stochastic-C draw; subset noise is heuristic',
                    requested_updates=epochs, max_setup_training_seconds=max_wall_seconds)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('Cache ready; metric ranks '+str([m.axes.shape[1] for m in metrics])+
          f'; elapsed {time.monotonic()-started:.1f}s', flush=True)
    history = {name: [] for name in arms}

    def evaluate(step):
        with torch.no_grad():
            for name, q in qs.items():
                row = dict(epoch=step)
                for split, (h, p) in evals.items():
                    _, terms, _ = objective(q, h, p, metrics, arms[name], states, model.geometry.patch_volume, weights)
                    row.update({split+'_'+k: float(v) for k, v in terms.items()})
                history[name].append(row)
        (out/'history_progress.json').write_text(json.dumps(history, indent=2)+'\n')

    evaluate(0)
    generator = torch.Generator().manual_seed(cfg.training.seed)
    completed = 0
    for step in range(1, epochs+1):
        if time.monotonic()-started > max_wall_seconds: break
        ids = torch.randperm(len(train.configurations), generator=generator)[:cfg.training.system_batch].numpy()
        permutation = torch.randperm(train.configurations.shape[1], generator=generator)[:ns].numpy()
        h, p = batch(cache, 'train', ids, [permutation[:ns//2], permutation[ns//2:]])
        for name, q in qs.items():
            opts[name].zero_grad(set_to_none=True)
            loss, _, _ = objective(q, h, p, metrics, arms[name], states, model.geometry.patch_volume, weights)
            if not torch.isfinite(loss): raise FloatingPointError(f'{name}: nonfinite loss')
            loss.backward(); opts[name].step(); normalize_affine_decoder_(q)
        completed = step
        if step % 50 == 0:
            evaluate(step)
            print(f'paired update {step}/{epochs}; elapsed {time.monotonic()-started:.1f}s', flush=True)
    if history['baseline'][-1]['epoch'] != completed: evaluate(completed)
    results = {}; encodings = {}
    for name in ['source', *arms]:
        q = model.theory_encoder if name == 'source' else qs[name]
        z = {}
        with torch.no_grad():
            for split, (h, p) in evals.items():
                _, _, direct = objective(q, h, p, metrics, arms.get(name, 0.), states, model.geometry.patch_volume, weights)
                z[split] = direct.numpy()
        encodings[name] = z
        np.savez_compressed(out/f'{name}_encodings.npz', **z,
                            train_labels=train.couplings.numpy(), test_labels=test.couplings.numpy())
        results[name] = representation_metrics(q, z, evals, metrics)
        if name != 'source':
            trained = copy.deepcopy(model); trained.theory_encoder = q
            for key, value in qcr_state_dict(trained).items():
                if not key.startswith('theory_encoder.'):
                    torch.testing.assert_close(value, source_state[key], atol=0, rtol=0)
            metadata = copy.deepcopy(manifest); metadata['arm'] = name
            torch.save(dict(model=qcr_state_dict(trained), epoch=completed, metadata=metadata,
                            optimizer=opts[name].state_dict(), batch_generator_state=generator.get_state()),
                       out/f'{name}.pt')
    results['metadata'] = dict(completed_updates=completed, requested_updates=epochs,
                              setup_training_seconds=time.monotonic()-started,
                              C_R_bitwise_unchanged=True, loss_weights=list(weights), additional_beta=beta,
                              R_caveat='Source R is not refitted to the changed Q coordinates; no new RG-flow claim')
    (out/'metrics.json').write_text(json.dumps(results, indent=2)+'\n')
    plot(out, model, encodings, datasets, history)
    print(json.dumps(results, indent=2), flush=True)
    return out


def representation_metrics(q, z, evals, metrics):
    result = []
    W = q.distribution_decoder.weight.detach().numpy()
    basis = np.linalg.qr(W)[0]
    for k, metric in enumerate(metrics):
        g = z['train'][:, k]; p = evals['train'][1][:, k].mean(1).numpy()
        pt = evals['test'][1][:, k].mean(1).numpy()
        centered = p-p.mean(0); eig = np.linalg.eigvalsh(np.cov(g.T, bias=True))
        residual = centered-centered@basis@basis.T
        probes = []
        for axis in metric.axes.numpy().T:
            y = centered@axis; yt = (pt-p.mean(0))@axis
            x = np.column_stack([g, np.ones(len(g))]); xt = np.column_stack([z['test'][:, k], np.ones(len(pt))])
            coefficient = np.linalg.lstsq(x, y, rcond=1e-8)[0]
            probes.append(dict(train_r2=float(1-np.mean((x@coefficient-y)**2)/np.var(y)),
                               test_r2=float(1-np.mean((xt@coefficient-yt)**2)/np.var(yt))))
        result.append(dict(covariance_eigenvalues=eig.tolist(), weak_main_ratio=float(eig[0]/max(eig[-1], 1e-12)),
                           participation_rank=float(eig.sum()**2/max((eig*eig).sum(), 1e-24)),
                           train_decoder_plane_captured_variance=float(1-np.sum(residual**2)/np.sum(centered**2)),
                           observable_direction_linear_probes=probes))
    return dict(scales=result)


def plot(out, model, encodings, datasets, history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = list(encodings); levels = next(iter(encodings.values()))['train'].shape[1]
    kc = float(np.log(1+np.sqrt(2))/2)
    labels = np.concatenate([datasets[s].couplings.numpy() for s in ['train', 'test']])
    phase = (labels[:, 0]>kc).astype(int)+2*(labels[:, 1]>kc).astype(int)
    colors = ['#444444', '#0072b2', '#d55e00', '#009e73']
    titles = ['both disordered', 'layer 1 ordered', 'layer 2 ordered', 'both ordered']
    fig, axes = plt.subplots(levels, len(names), figsize=(5*len(names), 4.5*levels), layout='constrained', squeeze=False)
    for col, name in enumerate(names):
        direct = np.concatenate([encodings[name][s] for s in ['train', 'test']])
        for k in range(levels):
            ax = axes[k, col]
            for i, color in enumerate(colors):
                ax.scatter(*direct[phase==i, k].T, color=color, s=10, alpha=.65, label=titles[i])
            ax.set(xlabel='g1', ylabel='g2', title=f'{name}, k={k}', aspect='equal')
            ax.legend(fontsize=7); ax.grid(alpha=.2)
    fig.suptitle('Q-only ablation; phase labels are evaluation annotations, not supervision')
    fig.savefig(out/'Q_comparison.png', dpi=150); plt.close(fig)
    from scripts.dual_ising_grid_diagnostics import grid_encodings
    for name in names:
        direct = np.concatenate([encodings[name][s] for s in ['train', 'test']])
        grids, values = grid_encodings(labels, direct)
        fig, axes = plt.subplots(2, levels, figsize=(5*levels, 8), layout='constrained', squeeze=False)
        for j in range(2):
            for k in range(levels):
                ax = axes[j, k]
                image = ax.pcolormesh(*grids, values[:, :, k, j].T, shading='nearest')
                ax.set(xlabel='K1', ylabel='K2', title=f'{name}: g{j+1}, k={k}')
                fig.colorbar(image, ax=ax)
        fig.savefig(out/f'{name}_coupling_by_scale.png', dpi=140); plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), layout='constrained')
    for name, rows in history.items():
        for split, style in [('train', '-'), ('test', '--')]:
            for ax, term in zip(axes, ['reconstruction', 'balanced', 'anchor', 'consistency']):
                ax.plot([r['epoch'] for r in rows], [r[split+'_'+term] for r in rows], style, label=name+' '+split)
                ax.set(title=term+' (unweighted)', xlabel='additional Q updates', yscale='log')
    for ax in axes: ax.legend(fontsize=7); ax.grid(alpha=.2)
    fig.savefig(out/'loss_comparison.png', dpi=150); plt.close(fig)
    source = encodings['source']['train'][:, :-1].reshape(-1, 2)
    lower, upper = source.min(0), source.max(0); padding = .15*np.maximum(upper-lower, .05)
    mesh = np.meshgrid(*[np.linspace(a, b, 25) for a, b in zip(lower-padding, upper+padding)])
    g = np.stack(mesh, -1).reshape(-1, 2)
    with torch.no_grad(): displacement = model.advance_theory(torch.from_numpy(g)).numpy()-g
    speed = np.linalg.norm(displacement, axis=1)
    direction = displacement/np.maximum(speed[:, None], 1e-12)
    fig, ax = plt.subplots(figsize=(7, 6), layout='constrained')
    ax.quiver(*g.T, *direction.T, color='.7'); ax.scatter(*source.T, s=6)
    ax.set(xlabel='g1', ylabel='g2', title='Frozen source R(g)-g; NOT refitted to new Q\nOff-data flow is extrapolation')
    fig.savefig(out/'frozen_R_reference_only.png', dpi=150); plt.close(fig)
