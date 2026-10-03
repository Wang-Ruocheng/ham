"""Track binary C truth tables and parameter changes from initialization."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recurrent_rg.config import RunConfig
from recurrent_rg.checkpoints import build_model, load_run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    cfg = RunConfig.from_dict(json.loads((args.run/'manifest.json').read_text())['config'])
    torch.manual_seed(cfg.training.seed)
    initial = build_model(cfg.model, cfg.training.max_rg_steps)
    volume = cfg.model.block_factor**cfg.model.dimension
    patterns = (2*((torch.arange(2**volume)[:, None]//(2**torch.arange(volume)))%2)-1).double()
    x = patterns.reshape(len(patterns), 1, *([cfg.model.block_factor]*cfg.model.dimension), 1)
    initial_parameters = torch.cat([v.detach().flatten() for v in initial.rg_kernel.parameters()])
    checkpoints = sorted(args.run.glob('training_step_*.pt'), key=lambda a: int(a.stem.split('_')[-1]))
    rows = []
    for step, model in [(0, initial)]+[(int(f.stem.split('_')[-1]), load_run(f).model) for f in checkpoints]:
        with torch.no_grad():
            soft = model.rg_kernel.field_head.relaxed(model.rg_kernel.local_features(x)).flatten()
            hard = model.rg_kernel(x).flatten().sign()
        weights = torch.cat([v.detach().flatten() for v in model.rg_kernel.parameters()])
        rows.append(dict(step=step, truth=hard.tolist(), relaxed=soft.tolist(),
                         positive_decimation_matches=int((hard==patterns[:, 0]).sum()),
                         all_minus_output=int(hard[0]), all_plus_output=int(hard[-1]),
                         parameter_distance_from_initial=float(torch.linalg.vector_norm(weights-initial_parameters))))
    out = args.run/'kernel_audit';out.mkdir(exist_ok=True)
    (out/'report.json').write_text(json.dumps(dict(patterns=patterns.tolist(), checkpoints=rows), indent=2)+'\n')
    fig, ax = plt.subplots(figsize=(11, 5), layout='constrained')
    im = ax.imshow(np.array([r['relaxed'] for r in rows]).T, aspect='auto', vmin=-1, vmax=1, cmap='coolwarm')
    ax.set(xticks=range(len(rows)), xticklabels=[r['step'] for r in rows],
           yticks=range(len(patterns)), yticklabels=[''.join('+' if v>0 else '-' for v in r) for r in patterns],
           xlabel='Training update', ylabel='Block spins (flattened spatial order)',
           title='Trainable C: relaxed outputs throughout training')
    fig.colorbar(im, ax=ax, label='Before hard sign threshold')
    fig.savefig(out/'kernel_evolution.png', dpi=170);plt.close(fig)
    print(json.dumps([{k:v for k,v in r.items() if k not in ('truth','relaxed')} for r in rows], indent=2))


if __name__ == '__main__':
    main()
