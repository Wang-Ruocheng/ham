"""Select K>=0 without resampling or changing the existing ensemble split."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def select_nonnegative(x, y):
    if len(x) != len(y) or y.ndim != 2 or y.shape[1] != 2:
        raise ValueError('Expected aligned configurations and (K,B) labels')
    mask = y[:, 0] >= 0
    if not mask.any():
        raise ValueError('No nonnegative K ensembles')
    return x[mask], y[mask]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    tx, ty, vx, vy = torch.load(args.source, weights_only=True)
    tx, ty = select_nonnegative(tx, ty)
    vx, vy = select_nonnegative(vx, vy)
    train_cells = set(map(tuple, ty.tolist()))
    test_cells = set(map(tuple, vy.tolist()))
    if train_cells & test_cells:
        raise ValueError('Physical cells overlap between train and test')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.save((tx, ty, vx, vy), args.output/'ensembles.pt')
    digest = hashlib.sha256()
    with args.source.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    labels = torch.cat([ty, vy])
    audit = dict(source=str(args.source.resolve()), source_sha256=digest.hexdigest(),
                 selection='K >= 0; original samples, labels, order and split preserved',
                 train_shape=list(tx.shape), test_shape=list(vx.shape),
                 K=torch.unique(labels[:, 0]).tolist(), B=torch.unique(labels[:, 1]).tolist())
    (args.output/'selection_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    main()
