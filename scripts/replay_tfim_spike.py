"""Reconstruct unsaved diagnostic checkpoints without altering the source run."""
import argparse
import json
from pathlib import Path
import torch
from recurrent_rg.checkpoints import load_run, qcr_state_dict
from recurrent_rg.config import RunConfig
from recurrent_rg.runner import prepared_data, evaluate_run
from recurrent_rg.training import train
from recurrent_rg.data.statistics import two_spin_distribution
from recurrent_rg.evaluation import scalar_flow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    loaded = load_run(args.run, 1200)
    cfg = RunConfig.from_dict(loaded.config)
    torch.set_num_threads(cfg.training.threads)
    tx, vx = prepared_data(cfg)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.run/'manifest.json').read_text())
    manifest['diagnostic_replay'] = dict(source=str(args.run), start=1200, stop=2000,
                                       checkpoint_every=100, note='No hyperparameter changes')
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    t, e = cfg.training, cfg.evaluation
    train(loaded.model, tx.configurations, vx.configurations,
          pattern_fn=two_spin_distribution, epochs=2000,
          system_batch=t.system_batch, sample_batch=t.sample_batch,
          eval_samples=e.samples, eval_systems=e.systems,
          learning_rate=t.learning_rate, seed=t.seed, progress_dir=args.output,
          spin_flip_augment=cfg.data.spin_flip_augment,
          objective_kwargs=cfg.loss.objective_kwargs(), resume_state=loaded.payload,
          initial_history=loaded.payload['history'], train_modules=t.train_modules,
          checkpoint_every=100, checkpoint_metadata=loaded.payload['metadata'],
          record_every=t.log_every)
    original = qcr_state_dict(load_run(args.run, 2000).model)
    replayed = qcr_state_dict(loaded.model)
    exact = all(torch.equal(original[k], replayed[k]) for k in original)
    maximum = max(float((original[k]-replayed[k]).abs().max()) for k in original)
    result = dict(bitwise_equal_at_2000=exact, max_abs_parameter_difference=maximum)
    if not exact:
        (args.output/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
        raise RuntimeError('Replay failed exact endpoint verification')
    patterns = torch.tensor([[-1,-1],[-1,1],[1,-1],[1,1]], dtype=torch.float64).reshape(4,1,2,1)
    result['checkpoints'] = {}
    for step in [1500,1600,1700,1800,1900,2000]:
        m = load_run(args.output, step).model
        with torch.no_grad(): hard = m.rg_kernel(patterns).flatten().tolist()
        result['checkpoints'][step] = dict(kernel=hard, roots=scalar_flow(m,-5,5)['roots'])
    (args.output/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    for step in [1500,1800,1900,2000]:
        evaluate_run(args.output, checkpoint=step, output=args.output/f'step{step}_report')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
