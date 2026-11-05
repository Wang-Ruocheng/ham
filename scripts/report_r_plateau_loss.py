"""Read-only R-only closure curves from one saved checkpoint's history."""
import argparse
import json
from pathlib import Path

import torch

from experiments.frozen_qc_r_plateau import plot_loss
from recurrent_rg.checkpoints import load_run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--checkpoint', type=int)
    parser.add_argument('--output', type=Path, help='Separate loss-report directory')
    parser.add_argument('--threads', type=int, help='Limit this diagnostic process only')
    args = parser.parse_args(argv)
    if args.threads is not None:
        if args.threads < 1:
            parser.error('Diagnostic threads must be positive')
        torch.set_num_threads(args.threads)

    checkpoint = args.checkpoint if args.checkpoint is not None else 'checkpoint_last.pt'
    loaded = load_run(args.run, checkpoint)
    history = loaded.payload['history']
    if not history:
        raise ValueError('Selected checkpoint has no R-only loss history')
    out = args.output or args.run / 'report'
    out.mkdir(parents=True, exist_ok=True)
    plot_loss(history, out)
    records = dict(
        checkpoint=str(loaded.checkpoint.resolve()),
        epoch=loaded.payload.get('epoch'),
        history_source='checkpoint.payload.history',
        history=history,
    )
    (out / 'loss_metrics.json').write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps(dict(checkpoint=records['checkpoint'], epoch=records['epoch'],
                          records=len(history), output=str(out)), indent=2))
    return records


if __name__ == '__main__':
    main()
