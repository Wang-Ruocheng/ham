"""Plot a scalar map on an explicit range, retaining data-support diagnostics."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recurrent_rg.checkpoints import load_run
from recurrent_rg.evaluation import scalar_flow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--lower', type=float, required=True)
    parser.add_argument('--upper', type=float, required=True)
    args = parser.parse_args()
    if args.lower >= args.upper:
        parser.error('lower must be less than upper')
    torch.set_num_threads(2)
    loaded = load_run(args.run)
    if loaded.config['model']['theory_dim'] != 1:
        parser.error('Requires dim_g=1')
    out = args.run / 'full_report'
    with np.load(out / 'encodings.npz') as data:
        inputs = data['train'][:, :-1].ravel()
        micro = np.concatenate([data['train'][:, 0], data['test'][:, 0]]).ravel()
    flow = scalar_flow(loaded.model, args.lower, args.upper)
    fig, ax = plt.subplots(figsize=(8, 6), layout='constrained')
    ax.plot(flow['grid'], flow['mapped'], label='R(g)')
    ax.plot(flow['grid'], flow['grid'], 'k--', label='Identity')
    ax.axvspan(inputs.min(), inputs.max(), color='green', alpha=.12, label='Training R input range')
    ax.axvspan(micro.min(), micro.max(), color='orange', alpha=.18, label='Microscopic Q(D) range')
    for item in flow['roots']:
        g = item['g'][0]
        slope = item['eigenvalues'][0]['real']
        item['inside_training_input_range'] = bool(inputs.min() <= g <= inputs.max())
        item['inside_microscopic_range'] = bool(micro.min() <= g <= micro.max())
        ax.scatter(g, g, color='green' if item['stability'] == 'stable' else 'red', zorder=5)
        ax.annotate(f"{item['stability']}\ng={g:.3f}, slope={slope:.3f}", (g, g),
                    xytext=(8, -35), textcoords='offset points', fontsize=9)
    ax.set(xlabel='g', ylabel='R(g)', title='Scalar RG map: roots outside shading are extrapolations')
    ax.legend(); ax.grid(alpha=.2)
    fig.savefig(out / 'R_vs_g_wide.png', dpi=170); plt.close(fig)
    audit = dict(roots=flow['roots'], search_range=[args.lower, args.upper],
                 training_input_range=[float(inputs.min()), float(inputs.max())],
                 microscopic_range=[float(micro.min()), float(micro.max())],
                 search_note=flow['search_note'])
    (out / 'fixed_points_wide.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
