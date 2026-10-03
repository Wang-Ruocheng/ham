"""Compare independent training runs without modifying either checkpoint."""
import argparse
import json
from pathlib import Path
import torch
from recurrent_rg.config import RunConfig


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    manifests = [json.loads((d/'manifest.json').read_text()) for d in (args.reference, args.run)]
    histories = [json.loads((d/'history_progress.json').read_text()) for d in (args.reference, args.run)]
    old = {r['epoch']: r for r in histories[0]}
    rows = []
    for row in histories[1]:
        ref = old.get(row['epoch'])
        same_keys = ref is not None and row.keys() == ref.keys()
        rows.append(dict(epoch=row['epoch'], all_terms_exact=same_keys and equal(ref, row),
                         max_absolute_difference=max(abs(row[k]-ref[k]) for k in row) if same_keys else None))
    checkpoints = []
    for path in sorted(args.run.glob('training_step_*.pt'), key=lambda x: int(x.stem.split('_')[-1])):
        ref_path = args.reference/path.name
        if not ref_path.exists():
            continue
        a, b = [torch.load(f, map_location='cpu', weights_only=True) for f in (ref_path, path)]
        keys_equal = a['model'].keys() == b['model'].keys()
        checkpoints.append(dict(epoch=b['epoch'], weights_bitwise_equal=equal(a['model'], b['model']),
                                max_weight_difference=max((a['model'][k]-b['model'][k]).abs().max().item() for k in a['model']) if keys_equal else None,
                                optimizer_equal=equal(a['optimizer'], b['optimizer']),
                                batch_rng_equal=equal(a['batch_generator_state'], b['batch_generator_state']),
                                torch_rng_equal=equal(a['torch_rng_state'], b['torch_rng_state'])))
    report = dict(reference=str(args.reference), run=str(args.run),
                  normalized_configs_equal=equal(*[RunConfig.from_dict(m['config']).to_dict() for m in manifests]),
                  data_sha256_equal=manifests[0]['data_sha256'] == manifests[1]['data_sha256'],
                  fresh_run=manifests[1]['resume'] is None and manifests[1]['initialize_from'] is None,
                  history_length=[len(h) for h in histories], history=rows, checkpoints=checkpoints)
    final_paths = [d/'qcr_model.pt' for d in (args.reference, args.run)]
    if all(f.exists() for f in final_paths):
        report['final_weights_bitwise_equal'] = equal(*[torch.load(f, map_location='cpu', weights_only=True) for f in final_paths])
    (args.run/'reproduction_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'history'}, indent=2))
    print('All observed history rows exact:', all(r['all_terms_exact'] for r in rows))


if __name__ == '__main__':
    main()
