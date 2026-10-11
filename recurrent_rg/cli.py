"""Installed command and python -m entry point for the shared QCR pipeline."""
import argparse
from .config import RunConfig
from .runner import run_training, evaluate_run


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('train',help='Train from a JSON config and an explicit data cache')
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--resume');p.add_argument('--initialize-from')
    p.add_argument('--device',default='cpu',help='Default QCR execution device: cpu or cuda:<index>')
    p.add_argument('--benchmark-warmup',type=int,help='Time actual updates, excluding this many initial updates from steady-step statistics')
    p.add_argument('--allow-legacy-training',action='store_true',help='Explicit ablation: bypass the locked empirical-MI independent-gradient recipe')
    for name in ('evaluate','plot'):
        p=sub.add_parser(name,help='Evaluate/report an existing versioned run, without training')
        p.add_argument('--run',required=True);p.add_argument('--checkpoint',type=int)
        p.add_argument('--cache');p.add_argument('--max-rg-steps',type=int);p.add_argument('--output')
    args=parser.parse_args(argv)
    if args.command=='train':
        if args.allow_legacy_training:
            if args.device!='cpu' or args.benchmark_warmup is not None:
                parser.error('CUDA/timing support is for the locked default trainer, not the legacy runner')
            print(run_training(RunConfig.load(args.config),args.output,resume=args.resume,
                               initialize_from=args.initialize_from,allow_legacy_training=True))
        else:
            if args.initialize_from:
                parser.error('Default training starts randomly; use --resume for an identical-recipe checkpoint. Warm starts require an explicit ablation.')
            from experiments.rsmi_separate_qcr import main as default_train
            command=['--config',args.config,'--output',args.output]
            if args.device!='cpu':command+=['--device',args.device]
            if args.benchmark_warmup is not None:command+=['--benchmark-warmup',str(args.benchmark_warmup)]
            if args.resume:command+=['--resume-checkpoint',args.resume]
            default_train(command)
    else:
        print(evaluate_run(args.run,checkpoint=args.checkpoint,cache=args.cache,
                           max_rg_steps=args.max_rg_steps,output=args.output,make_plots=args.command=='plot'))


if __name__=='__main__':main()
