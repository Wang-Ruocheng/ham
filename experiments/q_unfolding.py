"""Label-free Q loss/architecture ablation on identical frozen C trajectories.

All physical labels are accessed only after optimization. Not a QCR default.
"""
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from recurrent_rg.checkpoints import load_run, build_model, qcr_state_dict
from recurrent_rg.data.ensembles import load_ensembles
from recurrent_rg.data.statistics import categorical_plaquette_distribution
from recurrent_rg.distribution_encoder import LocalDistributionEncoder
from recurrent_rg.manifold import GraphCoordinate
from recurrent_rg.representation import linear_distribution_reconstruction_loss
from recurrent_rg.runner import file_fingerprint
from experiments.q_only_cached_sweep import sample_counts, encode, representation_loss
from experiments.q_only_candidates import OPTIONS, validate_frozen_levels


def unit_decoder(decoder):
    with torch.no_grad():
        decoder.weight.div_(decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-12))


def run(source, cache_source, output, epochs):
    started = time.monotonic()
    torch.set_num_threads(2)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    loaded = load_run(source)
    torch.manual_seed(91)
    model = build_model(loaded.model_config, 2)
    model.rg_kernel.load_state_dict(loaded.model.rg_kernel.state_dict())
    model.theory_rg.load_state_dict(loaded.model.theory_rg.state_dict())
    for p in model.parameters():
        p.requires_grad_(False)
    frozen = {k: v.clone() for k, v in qcr_state_dict(model).items()}
    tr, te = load_ensembles(loaded.config['data']['cache'])
    datasets = dict(train=tr, test=te)
    cachefile = Path(cache_source)/'frozen_levels.pt'
    old_manifest = json.loads((Path(cache_source)/'manifest.json').read_text())
    if old_manifest['data_sha256'] != file_fingerprint(loaded.config['data']['cache']):
        raise ValueError('Cache/data fingerprint mismatch')
    if old_manifest['cached_levels_sha256'] != file_fingerprint(cachefile):
        raise ValueError('Cache contents changed')
    if old_manifest['source_sha256'] != file_fingerprint(old_manifest['source_checkpoint']):
        raise ValueError('Original cache-source checkpoint changed')
    cache_reference = load_run(old_manifest['source']).model
    for key, value in model.rg_kernel.state_dict().items():
        torch.testing.assert_close(value, cache_reference.rg_kernel.state_dict()[key], atol=0, rtol=0)
    del cache_reference
    levels = torch.load(cachefile, weights_only=True)
    cache = {}
    for split, ls in levels.items():
        validate_frozen_levels(ls)
        torch.testing.assert_close(ls[0], datasets[split].configurations)
        counts, descriptors = [], []
        for i in range(len(ls[0])):
            xs = [v[i:i+1].double() for v in ls]
            counts.append([sample_counts(x, model.geometry, 3) for x in xs])
            descriptors.append(torch.stack([
                categorical_plaquette_distribution(x.squeeze(0).unsqueeze(1)) for x in xs
            ]).numpy())
        cache[split] = counts, np.stack(descriptors)
    del levels
    codes = (torch.arange(3**9)[:, None]//(3**torch.arange(9))) % 3
    patterns = torch.nn.functional.one_hot(codes, 3).flatten(1).double()

    def batch(split, ids, halves):
        counts, ps = cache[split]
        hist = np.array([[np.stack([
            np.asarray(counts[i][k][half].mean(0)).ravel() for half in halves
        ]) for k in range(3)] for i in ids])
        p = np.stack([ps[ids][:, :, half].mean(2) for half in halves], axis=2)
        return torch.from_numpy(hist), torch.from_numpy(p)

    evals = {s: batch(s, np.arange(len(d.configurations)),
                     [np.arange(32), np.arange(32, 64)]) for s, d in datasets.items()}
    full_p = {s: p.mean(2) for s, (_, p) in evals.items()}
    chart = GraphCoordinate.fit(full_p['train'][:, 0].numpy(), neighbors=4)
    targets = {s: torch.from_numpy(chart.transform(p.numpy()))[..., None] for s, p in full_p.items()}
    # Reference analytic linear reconstruction optimum; TRAIN data only.
    ptrain = full_p['train'].reshape(-1, 81)
    center = ptrain.mean(0)
    _, _, vh = torch.linalg.svd(ptrain-center, full_matrices=False)
    pca_axis = vh[0]
    reference = {s: ((p-center)@pca_axis).numpy() for s, p in full_p.items()}
    base = copy.deepcopy(model.theory_encoder)
    for p in base.parameters():
        p.requires_grad_(True)
    qs = {n: copy.deepcopy(base) for n in ['original', 'graph_coordinate', 'neural_linear_reconstruction']}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(91)
        qs['histogram_linear_reconstruction'] = LocalDistributionEncoder(3, 2).double()
        decoders = {n: torch.nn.Linear(1, 81).double() for n in qs if 'reconstruction' in n}
        for decoder in decoders.values():
            decoder.bias.data.copy_(center)
            unit_decoder(decoder)
    opts = {n: torch.optim.Adam(list(q.parameters()) +
            (list(decoders[n].parameters()) if n in decoders else []), lr=.001) for n, q in qs.items()}

    def predict(name, h, p):
        if name == 'histogram_linear_reconstruction':
            return qs[name].from_distribution(p)
        return encode(qs[name], patterns, h)

    def objective(name, h, p, target):
        if name == 'original':
            total, terms = representation_loss(qs[name], patterns, h, p, 10, OPTIONS)
            return total, dict(total=total, representation=terms['geometry'],
                              anchor=terms['anchor'], consistency=terms['consistency'])
        g = predict(name, h, p)
        anchor = (g[:, 0, 0]-g[:, 0, 1]).square().sum(-1).mean()
        consistency = (g[:, 1:, 0]-g[:, 1:, 1]).square().sum(-1).mean()
        if name == 'graph_coordinate':
            representation = (g-target[:, :, None]).square().mean()
        else:
            representation = linear_distribution_reconstruction_loss(
                decoders[name], g.mean(2), p.mean(2), mode='squared')
        total = 10*representation + 20*anchor + 5*consistency
        return total, dict(total=total, representation=representation, anchor=anchor, consistency=consistency)

    # Verify structural branch matches direct configurations exactly.
    with torch.no_grad():
        torch.testing.assert_close(qs['histogram_linear_reconstruction'](te.configurations[:2].double()),
            qs['histogram_linear_reconstruction'].from_distribution(full_p['test'][:2, 0]), atol=1e-12, rtol=1e-12)
    manifest = dict(source=str(loaded.checkpoint), source_sha256=file_fingerprint(loaded.checkpoint),
        cache_sha256=file_fingerprint(cachefile), data_sha256=file_fingerprint(loaded.config['data']['cache']),
        seed=91, requested_epochs=epochs, labels_used_in_training=False, C_R_frozen=True,
        protocol='Four paired Q-only arms; same batches/Adam0.001; original neural Q fresh seed91; no C/R updates',
        objectives=dict(original='Original interval relative_local geometry',
            graph_coordinate='10 mean squared regression of both subset g to full-D data-derived chart; 20 micro+5 coarse subset consistency',
            neural_linear_reconstruction='Original neural Q; 10 sum-bin squared affine reconstruction of same-scale p; 20/5 subset',
            histogram_linear_reconstruction='Exact generic oriented 2x2 categorical histogram -> learned linear 81->1; same affine reconstruction/20/5'),
        decoder_gauge='Reconstruction decoder column Euclidean norm projected to1 after every update; no latent-scale shrinking gauge',
        graph_subset_note='Both-half target MSE adds 5/6 to effective micro subset weight and 5/3 to coarse subset weight, because representation averages three scales; raw objective is logged explicitly',
        chart=dict(reference='Micro TRAIN descriptors only', neighbors=4, endpoints=chart.endpoints,
                   diameter=chart.diameter, normalized='[-1,1] signed geodesic diameter coordinate',
                   queries='Each coarse/test full ensemble independently attached; no test shortcuts',
                   limitation='Interval-like data chart; not a general multidimensional/branched-manifold solution'),
        params={n: dict(Q=sum(p.numel() for p in q.parameters()),
                    auxiliary=sum(p.numel() for p in decoders[n].parameters()) if n in decoders else 0) for n, q in qs.items()},
        limitations='One training seed, one frozen C draw. New coordinates invalidate source R; no new R phase diagram or RG success claimed.')
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    history = {n: [] for n in qs}

    def evaluate(step):
        with torch.no_grad():
            for n in qs:
                row = dict(epoch=step)
                for s, (h, p) in evals.items():
                    _, terms = objective(n, h, p, targets[s])
                    row[s] = {k: float(v) for k, v in terms.items()}
                history[n].append(row)
        (out/'history.json').write_text(json.dumps(history, indent=2))

    evaluate(0)
    gen = torch.Generator().manual_seed(91)
    print('Label-free chart and exact histogram verification complete; four paired arms.', flush=True)
    completed = 0
    for step in range(1, epochs+1):
        ids = torch.randperm(len(tr.configurations), generator=gen)[:32].numpy()
        perm = torch.randperm(64, generator=gen).numpy()
        h, p = batch('train', ids, [perm[:32], perm[32:]])
        for name in qs:
            opts[name].zero_grad(set_to_none=True)
            total, _ = objective(name, h, p, targets['train'][ids])
            if not torch.isfinite(total):
                raise FloatingPointError(name)
            total.backward()
            opts[name].step()
            if name in decoders:
                unit_decoder(decoders[name])
        completed = step
        if step % 100 == 0 or step == epochs:
            evaluate(step)
            print(f'epoch {step}/{epochs}; elapsed {time.monotonic()-started:.1f}s; test '+
                  str({n: round(history[n][-1]['test']['total'], 5) for n in qs}), flush=True)
            torch.save(dict(Q={n:q.state_dict() for n,q in qs.items()},
                decoder={n:d.state_dict() for n,d in decoders.items()}, optimizer={n:o.state_dict() for n,o in opts.items()},
                generator=gen.get_state(), epoch=step), out/'checkpoint_last.pt')
    for k, v in qcr_state_dict(model).items():
        torch.testing.assert_close(v, frozen[k], atol=0, rtol=0)
    # Physical coupling labels are first accessed HERE, for held-out diagnostics.
    labels = {s: d.couplings.reshape(len(d.configurations), -1)[:, 0].numpy() for s, d in datasets.items()}
    values, halves = {}, {}
    with torch.no_grad():
        for n in qs:
            values[n] = {s: predict(n, h.mean(2, keepdim=True), p.mean(2, keepdim=True)).numpy()[:, :, 0, 0]
                         for s, (h, p) in evals.items()}
            halves[n] = {s: predict(n, h, p).numpy()[..., 0] for s, (h, p) in evals.items()}
    values['graph_target'] = {s: v.numpy()[..., 0] for s, v in targets.items()}
    values['raw_histogram_PCA'] = reference
    metrics = {}
    for name, gs in values.items():
        scalestats = []
        for j in range(3):
            K, g = labels['test'], gs['test'][:, j]
            unique = np.unique(K)
            curve = np.array([g[K == k].mean() for k in unique])
            delta = np.diff(curve)
            sign = 1 if spearmanr(labels['train'], gs['train'][:, j]).statistic >= 0 else -1
            ordered_left, ordered_right = sign*g[K < -.6], sign*g[K > .6]
            stat = dict(abs_spearman_K=float(abs(spearmanr(K, g).statistic)),
                reverse_variation_fraction=float(min(delta.clip(min=0).sum(), (-delta).clip(min=0).sum())/max(abs(delta).sum(), 1e-12)),
                ordered_branch_gap=float(ordered_right.min()-ordered_left.max()),
                ordered_centroid_distance=float(ordered_right.mean()-ordered_left.mean()),
                g_std=float(g.std()), phase_centroids=[float(g[K<-.6].mean()), float(g[abs(K)<.2].mean()), float(g[K>.6].mean())])
            if name in halves:
                rms = float(np.sqrt(np.mean(np.diff(halves[name]['test'][:, j], axis=-1)**2)))
                stat['subset_RMS'] = rms
                stat['subset_relative_RMS'] = rms/max(float(g.std()), 1e-12)
                stat['ordered_separation_to_subset_RMS'] = stat['ordered_centroid_distance']/max(rms, 1e-12)
            scalestats.append(stat)
        metrics[name] = dict(scales=scalestats, final_loss=history[name][-1] if name in history else None)
    for n, q in qs.items():
        if n == 'histogram_linear_reconstruction':
            torch.save(dict(Q=q.state_dict(), decoder=decoders[n].state_dict(), architecture='LocalDistributionEncoder(3,2,1)',
                            metadata=manifest, epoch=completed), out/(n+'.pt'))
        else:
            result_model = copy.deepcopy(model)
            result_model.theory_encoder = q
            torch.save(dict(model=qcr_state_dict(result_model), metadata=dict(config=loaded.config, **manifest, arm=n),
                epoch=completed, decoder=decoders[n].state_dict() if n in decoders else None), out/(n+'.pt'))
    report = dict(**manifest, completed_epochs=completed, elapsed_seconds=time.monotonic()-started, results=metrics)
    (out/'metrics.json').write_text(json.dumps(report, indent=2))
    np.savez_compressed(out/'encodings.npz', **{f'{n}_{s}':v for n,gs in values.items() for s,v in gs.items()},
        **{f'{s}_K':v for s,v in labels.items()}, **{f'{n}_{s}_halves':v for n,gs in halves.items() for s,v in gs.items()})
    torch.save(dict(reference=torch.from_numpy(chart.reference), paths=torch.from_numpy(chart.paths),
                    endpoints=chart.endpoints, neighbors=chart.neighbors, PCA_center=center, PCA_axis=pca_axis), out/'data_chart.pt')
    plot(out, labels, values, history)
    print(json.dumps(dict(elapsed_seconds=report['elapsed_seconds'], results=metrics), indent=2), flush=True)


def plot(out, labels, values, history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout='constrained')
    K = labels['test']; unique = np.unique(K)
    for ax, (name, gs) in zip(axes.flat, values.items()):
        for j in range(3):
            mean = np.array([gs['test'][K==k, j].mean() for k in unique])
            ax.plot(unique, mean, '-o', ms=3, label=f'k={j}')
        ax.set(title=name.replace('_', ' '), xlabel='K (held-out; evaluation only)', ylabel='g')
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.suptitle('Q-only signed spin-1: SAME frozen C; no coupling labels in optimization')
    fig.savefig(out/'K_to_g.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    for ax, name, title in zip(axes,
            ['original', 'neural_linear_reconstruction', 'histogram_linear_reconstruction'],
            ['Original Q + original objective', 'SAME Q + affine reconstruction', 'Histogram Q + affine reconstruction']):
        for j in range(3):
            mean = np.array([values[name]['test'][K==k, j].mean() for k in unique])
            ax.plot(unique, mean, '-o', ms=3, label=f'k={j}')
        ax.set(title=title, xlabel='Held-out K (evaluation only)', ylabel='g')
        ax.legend(); ax.grid(alpha=.2)
    fig.suptitle('1200 Q-only updates; frozen identical C/R; NO coupling supervision')
    fig.savefig(out/'Q_fix_summary.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    for name, rows in history.items():
        for ax, key in zip(axes, ['representation', 'anchor', 'consistency']):
            for s, style in [('train', '-'), ('test', '--')]:
                ax.plot([r['epoch'] for r in rows], [r[s][key] for r in rows], style, label=f'{name} {s}')
            ax.set(title=key+' (unweighted; objectives DIFFER)', xlabel='updates', yscale='log')
    axes[0].legend(fontsize=6); fig.savefig(out/'loss.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), layout='constrained')
    for ax, name in zip(axes, history):
        g=values[name]['test']
        for j in [0, 1]:
            sc=ax.scatter(g[:, j], g[:, j+1], c=K, cmap='coolwarm', vmin=-.9, vmax=.9,
                          marker=['o','x'][j], label=f'{j}->{j+1}')
        ax.set(title=name.replace('_',' '), xlabel='Q(C^k D)', ylabel='Q(C^(k+1) D)')
        ax.legend(); ax.grid(alpha=.2); fig.colorbar(sc, ax=ax, label='K (evaluation)')
    fig.suptitle('Direct successors only; R is NOT refitted')
    fig.savefig(out/'direct_successors.png', dpi=150); plt.close(fig)
