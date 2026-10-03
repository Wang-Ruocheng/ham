"""Resolve geometry-loss violations by scale and ensemble pair, without training."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recurrent_rg.model_nd import RecurrentRGNetwork2D
from recurrent_rg.data.statistics import plaquette_distribution


def encode(model, x, samples):
    records = {key: [] for key in ('g', 'p', 'noise', 'stats')}
    with torch.no_grad():
        for start in range(0, len(x), 4):
            levels = model.coarse_grain(x[start:start+4, :samples].double())
            batch = {key: [] for key in records}
            for v in levels:
                a, b = v[:, :samples//2], v[:, samples//2:]
                pa, pb = plaquette_distribution(a), plaquette_distribution(b)
                batch['g'].append(.5*(model.theory_encoder(a)+model.theory_encoder(b)))
                batch['p'].append(.5*(pa+pb))
                batch['noise'].append(.25*(pa-pb).square().sum(-1))
                m = v.mean((2, 3, 4))
                bond = .5*(v*torch.roll(v, -1, 2)+v*torch.roll(v, -1, 3)).mean((1, 2, 3, 4))
                aligned = .5*(pa[:, 0]+pb[:, 0]+pa[:, 15]+pb[:, 15])
                batch['stats'].append(torch.stack((m.mean(1), m.abs().mean(1), bond, aligned), -1))
            for key in records:
                records[key].append(torch.stack(batch[key], 1).numpy())
            if start % 100 == 0:
                print(f'encoded {min(start+4,len(x))}/{len(x)} ensembles, samples={samples}', flush=True)
    return {key: np.concatenate(parts) for key, parts in records.items()}


def analyze(encoded, labels, indices, k, scale, multiplier):
    g, p, noise = [encoded[key][indices, k] for key in ('g', 'p', 'noise')]
    a, b = np.triu_indices(len(indices), 1)
    distance = np.linalg.norm(p[a]-p[b], axis=1)
    allowance = multiplier*np.sqrt(noise[a]+noise[b])
    lower, upper = scale*np.maximum(distance-allowance, 0), scale*(distance+allowance)
    latent = np.linalg.norm(g[a]-g[b], axis=1)
    under, over = np.maximum(lower-latent, 0)**2, np.maximum(latent-upper, 0)**2
    loss = under+over
    order = np.argsort(-loss)
    top, seen = [], set()
    for pair in order:
        i, j = int(indices[a[pair]]), int(indices[b[pair]])
        key = tuple(sorted((tuple(labels[i]), tuple(labels[j]))))
        if key in seen:
            continue
        seen.add(key)
        top.append(dict(i=i, j=j, KB_i=labels[i].tolist(), KB_j=labels[j].tolist(),
                        loss=float(loss[pair]), under=float(under[pair]), over=float(over[pair]),
                        latent_distance=float(latent[pair]), lower=float(lower[pair]), upper=float(upper[pair]),
                        stats_i=encoded['stats'][i,k].tolist(), stats_j=encoded['stats'][j,k].tolist()))
        if len(top) == 20:
            break
    kc = .5*np.log1p(np.sqrt(2))
    # Diagnostic categories based only on microscopic coupling labels, not used in training.
    categories = np.array([(0 if abs(b) < 1e-12 else 1 if abs(b) <= .008 else 2)*2
                           + int(k >= kc-1e-12) for k,b in labels[indices]])
    sums, counts = np.zeros((6,6)), np.zeros((6,6), dtype=int)
    ca, cb = np.minimum(categories[a], categories[b]), np.maximum(categories[a], categories[b])
    np.add.at(sums, (ca,cb), loss)
    np.add.at(counts, (ca,cb), 1)
    category_rows = []
    for i in range(6):
        for j in range(i,6):
            if counts[i,j]:
                category_rows.append(dict(a=i,b=j,pairs=int(counts[i,j]),
                                          mean_loss=float(sums[i,j]/counts[i,j]),
                                          loss_share=float(sums[i,j]/loss.sum())))
    n_top = max(1, int(np.ceil(.01*len(loss))))
    return dict(k=k, sites=32//2**k, ensembles=len(indices), pairs=len(loss),
                mean_loss=float(loss.mean()), under_mean=float(under.mean()), over_mean=float(over.mean()),
                under_fraction=float(np.mean(under>0)), over_fraction=float(np.mean(over>0)),
                top_one_percent_loss_share=float(loss[order[:n_top]].sum()/loss.sum()),
                categories=category_rows, top_pairs=top)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    summary = json.loads((args.run/'summary.json').read_text())
    cfg = summary['config']; obj = cfg['objective_kwargs']
    torch.set_num_threads(2)
    model = RecurrentRGNetwork2D(levels=cfg['levels'],hidden=cfg['hidden'],theory_dim=cfg['theory_dim'],
                                learnable_kernel=cfg.get('learnable_kernel',True)).double().eval()
    model.theory_encoder.binary_lookup = model.rg_kernel.binary_lookup = True
    model.load_state_dict(torch.load(args.run/'qcr_model.pt', weights_only=True), strict=False)
    _, _, x, labels = torch.load(cfg['data_cache'], weights_only=True)
    labels = labels.numpy()
    out = args.run/'geometry_diagnostic'; out.mkdir(exist_ok=True)
    report = dict(checkpoint=cfg['epochs'],
                  stats_order=['mean_m','mean_abs_m','mean_nearest_neighbor_product','uniform_plaquette_probability'],
                  category_names=['B=0, K<Kc','B=0, K>=Kc','0<|B|<=.008, K<Kc',
                                  '0<|B|<=.008, K>=Kc','|B|>=.02, K<Kc','|B|>=.02, K>=Kc'],
                  note='All unordered ensemble pairs; three replicas per cell. Categories are post-hoc and do not establish finite-size phases.')
    eval_ids = torch.linspace(0,len(x)-1,min(cfg['eval_systems'],len(x))).long().numpy()
    sets = {}
    if args.plot_only:
        report = json.loads((out/'diagnostic.json').read_text())
    for samples in (() if args.plot_only else (128,256)):
        encoded = encode(model,x,samples)
        sets[samples] = encoded
        report[f'full_test_n{samples}'] = [analyze(encoded,labels,np.arange(len(x)),k,
            obj['geometry_scale'],obj['noise_multiplier']) for k in range(cfg['levels'])]
        if samples == cfg['eval_samples']:
            exact = [analyze(encoded,labels,eval_ids,k,obj['geometry_scale'],obj['noise_multiplier'])
                     for k in range(cfg['levels'])]
            expected = [summary['history'][-1][f'test_data_geometry_level_{k}'] for k in range(cfg['levels'])]
            np.testing.assert_allclose([r['mean_loss'] for r in exact],expected,rtol=1e-9,atol=1e-12)
            report['logged_test_subset'] = exact
    for name in ('logged_test_subset','full_test_n128','full_test_n256'):
        total = sum(r['mean_loss'] for r in report[name])
        for row in report[name]:
            row['share_of_geometry'] = row['mean_loss']/total
            row['weighted_contribution_to_total'] = 10/cfg['levels']*row['mean_loss']
    (out/'diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
    fig, axes = plt.subplots(1,3,figsize=(14,4.5),sharey=True)
    for ax,name,title in zip(axes,('logged_test_subset','full_test_n128','full_test_n256'),
                             ('Logged subset: 48 ensembles, 128 samples','Full test: 429 ensembles, 128 samples',
                              'Full test: 429 ensembles, 256 samples')):
        rows=report[name]; under=np.array([r['under_mean'] for r in rows]); over=np.array([r['over_mean'] for r in rows])
        ax.bar(range(4),under,label='g distances too small',color='#377eb8')
        ax.bar(range(4),over,bottom=under,label='g distances too large',color='#ff9d45')
        ax.set(title=title,xlabel='RG scale k',xticks=range(4),ylabel='Unweighted geometry loss')
        ax.legend(fontsize=8); ax.grid(axis='y',alpha=.2)
    fig.suptitle('Fixed decimation, seed 91, step 2000: geometry loss by scale')
    fig.tight_layout();fig.savefig(out/'geometry_by_scale.png',dpi=190);plt.close(fig)
    # Show actual configurations for the largest pair at the largest-loss scale.
    worst=max(report['full_test_n256'],key=lambda r:r['mean_loss']); pair=worst['top_pairs'][0]; k=worst['k']
    assert not cfg.get('learnable_kernel',True), 'Configuration display currently assumes fixed positive decimation'
    fig,axes=plt.subplots(2,4,figsize=(12,7.4))
    for row,i in enumerate((pair['i'],pair['j'])):
        raw=x[i,...,0].numpy(); order=np.argsort(raw.mean((1,2)))
        picks=[int(order[len(order)//4]),int(order[3*len(order)//4])]
        for column,(s,level) in enumerate([(picks[0],0),(picks[1],0),(picks[0],k),(picks[1],k)]):
            axes[row,column].imshow(raw[s,::2**level,::2**level],cmap='gray',vmin=-1,vmax=1,interpolation='nearest')
            axes[row,column].set_title(f'sample {s}, k={level}',fontsize=9)
            axes[row,column].set_xticks([]);axes[row,column].set_yticks([])
        axes[row,0].set_ylabel(f'Ensemble {i}\nK={labels[i,0]:.5f}, B={labels[i,1]:+.3f}',fontsize=9)
    fig.suptitle(f"Largest-loss pair at k={k}: g distance {pair['latent_distance']:.3f}; allowed [{pair['lower']:.3f}, {pair['upper']:.3f}]")
    fig.text(.5,.015,'Configurations selected at the 25th/75th percentiles of sample magnetization.\n'
             'Loss compares ensemble pattern distributions, not these individual images. White=+1; black=-1.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.085,right=.99,bottom=.13,top=.88,wspace=.10,hspace=.27)
    fig.savefig(out/'largest_pair_configurations.png',dpi=190);plt.close(fig)
    for name in ('logged_test_subset','full_test_n128','full_test_n256'):
        print(name,[(r['k'],r['mean_loss'],r['under_mean'],r['over_mean']) for r in report[name]],flush=True)
    print('largest example',json.dumps(pair),flush=True)


if __name__ == '__main__':
    main()
