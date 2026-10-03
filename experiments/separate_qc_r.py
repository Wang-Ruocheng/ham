"""From-scratch paired-seed ablation: QC and R receive only their own losses."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from recurrent_rg.config import RunConfig
from recurrent_rg.runner import run_training, evaluate_run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--distinct-alpha',type=float,default=0.)
    p.add_argument('--distinct-margin',type=float,default=.8)
    p.add_argument('--fixed-decimation',action='store_true')
    args=p.parse_args()
    cfg=RunConfig.from_dict(json.loads((args.source/'manifest.json').read_text())['config'])
    cfg=replace(cfg,training=replace(cfg.training,gradient_mode='qc_r_separate'),
                loss=replace(cfg.loss,kernel_distinct_alpha=args.distinct_alpha,kernel_distinct_margin=args.distinct_margin))
    if args.fixed_decimation:
        if args.distinct_alpha:raise ValueError('Fixed C has no distinct loss')
        cfg=replace(cfg,model=replace(cfg.model,fixed_kernel=True,kernel_sampling='deterministic',kernel_initialization='random'),
            training=replace(cfg.training,train_modules=('Q','R')),
            loss=replace(cfg.loss,kernel_sign_weight=0.,kernel_neutral_weight=0.,weights=replace(cfg.loss.weights,kernel_noncollapse=0.)))
    run_training(cfg,args.output)
    evaluate_run(args.output,output=args.output/'report')


if __name__=='__main__':main()
