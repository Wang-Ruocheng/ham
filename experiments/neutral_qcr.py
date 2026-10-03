"""Paired neutral-weight ablation using the existing Ising caches and joint QCR."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recurrent_rg.config import RunConfig
from recurrent_rg.runner import run_training, evaluate_run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dimension',type=int,choices=(1,2),required=True)
    p.add_argument('--weight',type=float,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=2400)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    source=root/'configs'/f'rsmi_ising{args.dimension}d_g{args.dimension}_seed91.json'
    cfg=RunConfig.load(source)
    cfg=replace(cfg,loss=replace(cfg.loss,kernel_neutral_weight=args.weight,kernel_sign_weight=0.,
                weights=replace(cfg.loss.weights,kernel_noncollapse=2.)),
                training=replace(cfg.training,epochs=args.epochs,train_modules=('Q','C','R')))
    cfg.validate()
    args.output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    (args.output/'config.json').write_text(json.dumps(cfg.to_dict(),indent=2)+'\n')
    run_training(cfg,args.output/'qr')
    evaluate_run(args.output/'qr',output=args.output/'qr'/'report')
    # Reuse physical diagnostics, with an explicit correct protocol label.
    with (args.output/'physical_report.log').open('w') as log:
        subprocess.run([sys.executable,str(root/'scripts'/'report_rsmi_experiment.py'),
                        '--run',str(args.output),'--flow-title',f'Joint QCR, neutral weight {args.weight:g}'],
                        cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
    (args.output/'timing.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start),indent=2)+'\n')
    print(f'Completed {args.output}',flush=True)


if __name__=='__main__':main()
