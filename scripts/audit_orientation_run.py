"""Audit local direction regularization, full binary kernel, and local KB inverses."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from recurrent_rg.checkpoints import load_run
from recurrent_rg.losses import local_orientation_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    loaded = load_run(args.run)
    model = loaded.model
    folder = args.run/'full_report'
    arrays = np.load(folder/'encodings.npz')
    output = {}
    with torch.no_grad():
        for split in ('train', 'test'):
            loss, levels = local_orientation_loss(model, torch.from_numpy(arrays[split]),
                                                  loaded.config['loss'].get('orientation_radius', .02))
            output[split] = {'orientation':float(loss), 'orientation_by_scale':levels.tolist()}
    kernel = json.loads((folder/'metrics.json').read_text())['kernel']
    x, p = np.array(kernel['patterns']), np.array(kernel['p_plus'])
    ids = {tuple(v):i for i,v in enumerate(x)}
    symmetry = np.array([p[i]+p[ids[tuple(-v)]]-1 for i,v in enumerate(x)])
    edges = []
    for i,v in enumerate(x):
        for j in range(x.shape[1]):
            if v[j] < 0:
                w = v.copy(); w[j] = 1
                edges.append(p[ids[tuple(w)]]-p[i])
    edges = np.array(edges)
    mask = x.sum(1) != 0
    alignment = np.where(x.sum(1)>0, p, 1-p)
    output['kernel'] = dict(increasing_edges=int((edges>1e-8).sum()),
        decreasing_edges=int((edges< -1e-8).sum()), edges=len(edges),
        min_edge=float(edges.min()), max_edge=float(edges.max()),
        mean_Z2_probability_error=float(abs(symmetry).mean()),
        max_Z2_probability_error=float(abs(symmetry).max()),
        majority_alignment_non_ties=float(alignment[mask].mean()),
        uniform_minus_retention=float(1-p[0]), uniform_plus_retention=float(p[-1]),
        note='Uniform weighting over patterns/edges, not empirical data weighting')
    # Only invert triangles connecting adjacent physical parameter-grid cells.
    g = np.concatenate([arrays['train'], arrays['test']])[:,0]
    labels = np.concatenate([arrays['train_labels'], arrays['test_labels']])
    ks, bs = np.unique(labels[:,0]), np.unique(labels[:,1])
    means = np.array([[g[np.all(labels==[k,b],axis=1)].mean(0) for b in bs] for k in ks])
    roots = json.loads((folder/'fixed_points.json').read_text())['roots']
    output['roots'] = roots
    for root in roots:
        for reverse in (False, True):
            hits = []
            for i in range(len(ks)-1):
                for j in range(len(bs)-1):
                    a,b,c,d = (i,j),(i+1,j),(i+1,j+1),(i,j+1)
                    for triangle in ((a,b,d),(b,c,d)) if reverse else ((a,b,c),(a,c,d)):
                        vertices = np.array([means[u,v] for u,v in triangle])
                        matrix = np.vstack([vertices.T,np.ones(3)])
                        if abs(np.linalg.det(matrix)) < 1e-12:
                            continue
                        bary = np.linalg.solve(matrix,np.r_[root['g'],1])
                        if bary.min() >= -1e-8 and bary.max() <= 1+1e-8:
                            kb = bary@np.array([[ks[u],bs[v]] for u,v in triangle])
                            hits.append(kb.tolist())
            root['inverse_KB_alternate' if reverse else 'inverse_KB'] = hits
    output['inverse_note'] = 'Empirical microscopic interpolation; multiple hits indicate ambiguity. No extrapolation. Not confidence intervals.'
    (folder/'orientation_audit.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
