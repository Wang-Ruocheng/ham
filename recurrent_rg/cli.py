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
    for name in ('evaluate','plot'):
        p=sub.add_parser(name,help='Evaluate/report an existing versioned run, without training')
        p.add_argument('--run',required=True);p.add_argument('--checkpoint',type=int)
        p.add_argument('--cache');p.add_argument('--max-rg-steps',type=int);p.add_argument('--output')
    args=parser.parse_args(argv)
    if args.command=='train':
        print(run_training(RunConfig.load(args.config),args.output,resume=args.resume,initialize_from=args.initialize_from))
    else:
        print(evaluate_run(args.run,checkpoint=args.checkpoint,cache=args.cache,
                           max_rg_steps=args.max_rg_steps,output=args.output,make_plots=args.command=='plot'))


if __name__=='__main__':main()
