"""Installed command and python -m entry point for the shared QCR pipeline."""
import argparse
import json
from pathlib import Path
from .config import RunConfig
from .runner import run_training, evaluate_run


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('train',help='Train from a JSON config and an explicit data cache')
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--resume');p.add_argument('--initialize-from')
    p.add_argument('--device',default=None,help='Default QCR execution device: cpu or cuda:<index>')
    p.add_argument('--benchmark-warmup',type=int,help='Time actual updates, excluding this many initial updates from steady-step statistics')
    p.add_argument('--reference-execution',action='store_true',help='Use unoptimized execution for a same-recipe performance comparison')
    p.add_argument('--mi-cache-dir',help='GPU static empirical-MI cache directory; defaults beside the data cache')
    p.add_argument('--allow-legacy-training',action='store_true',help='Explicit ablation: bypass the locked empirical-MI independent-gradient recipe')
    for name in ('evaluate','plot'):
        p=sub.add_parser(name,help='Evaluate/report an existing versioned run, without training')
        p.add_argument('--run',required=True);p.add_argument('--checkpoint',type=int)
        p.add_argument('--cache');p.add_argument('--max-rg-steps',type=int);p.add_argument('--output')
    args=parser.parse_args(argv)
    if args.command=='train':
        explicit_device = args.device is not None
        args.device = args.device if explicit_device else 'cpu'
        # An explicitly named continuous recipe has its own strict loader and
        # probability model. Never infer it from field values or reinterpret a
        # historical discrete/r-u config as hopping-kappa data.
        config_path = Path(args.config)
        config_header = json.loads(config_path.read_text()) if config_path.is_file() else {}
        if isinstance(config_header, dict) and config_header.get('recipe') == 'phi4_ru_b3_g2_ensemble_vae_terminal_v1':
            if (args.allow_legacy_training or args.resume or args.initialize_from
                    or args.benchmark_warmup is not None or args.reference_execution or args.mi_cache_dir):
                parser.error('The explicit continuous ensemble-VAE experiment is fresh-only')
            from experiments.phi4_ru_vae_qcr import run
            return run(args.config, args.output, device=args.device)
        if isinstance(config_header, dict) and config_header.get('recipe') == 'xy2d_b3_local_g2_terminal_fixed_QC_R_v1':
            if (args.allow_legacy_training or args.resume or args.initialize_from
                    or args.benchmark_warmup is not None or args.reference_execution or args.mi_cache_dir
                    or (explicit_device and args.device != 'cuda:0')):
                parser.error('The XY terminal R fit uses its pinned completed source and original R Adam')
            from experiments.xy_qcr_g2_r_only import run
            return run(args.config,args.output)
        if isinstance(config_header, dict) and config_header.get('recipe') == 'xy2d_b3_local_periodic_freegain_g2_matched1250_seed91_v1':
            if (args.allow_legacy_training or args.resume or args.initialize_from
                    or args.benchmark_warmup is not None or args.reference_execution or args.mi_cache_dir
                    or (explicit_device and args.device != 'cuda:0')):
                parser.error('The XY dimension control uses fresh initialization and pinned existing data')
            from experiments.xy_qcr_g2 import run
            return run(args.config,args.output)
        if isinstance(config_header, dict) and config_header.get('recipe') == 'xy2d_b3_local_g1_joint_train_plateau_continue_v1':
            if (args.allow_legacy_training or args.resume or args.initialize_from
                    or args.benchmark_warmup is not None or args.reference_execution or args.mi_cache_dir
                    or (explicit_device and args.device != 'cuda:0')):
                parser.error('The XY continuation uses its pinned source and original optimizer states')
            from experiments.xy_qcr_continue import run
            return run(args.config,args.output)
        if isinstance(config_header, dict) and config_header.get('recipe') == 'xy2d_b3_local_periodic_freegain_g1_seed91_v1':
            if (args.allow_legacy_training or args.resume or args.initialize_from
                    or args.benchmark_warmup is not None or args.reference_execution or args.mi_cache_dir
                    or (explicit_device and args.device != 'cuda:0')):
                parser.error('The explicit XY trial is fresh-only with a pinned periodic adapter and masked GPU')
            from experiments.xy_qcr import run
            return run(args.config,args.output)
        if isinstance(config_header, dict) and config_header.get('recipe') == 'phi4_hopping_b3_local_Q_free_gain_unsupervised1250_seed91_v1':
            if (args.allow_legacy_training or args.resume or args.initialize_from
                    or args.benchmark_warmup is not None or args.reference_execution or args.mi_cache_dir
                    or (explicit_device and args.device != 'cuda:0')):
                parser.error('The explicit local-Q phi4 experiment is fresh-only with pinned data/device')
            from experiments.phi4_hopping_local_current import run
            return run(args.config, args.output)
        if isinstance(config_header, dict) and config_header.get('recipe') == 'ising2d_KB_local_motif_g2_supervision10_matched1250_seed91_v1':
            if (args.allow_legacy_training or args.resume or args.initialize_from
                    or args.benchmark_warmup is not None or args.reference_execution or args.mi_cache_dir
                    or (explicit_device and args.device != 'cuda:0')):
                parser.error('The matched local-Q KB ablation is fresh-only with pinned data and device')
            from experiments.ising_kb_local_motif_supervised import run
            return run(args.config,args.output)
        if isinstance(config_header, dict) and config_header.get('recipe') == 'ising2d_B0_local_motif_g1_supervision10_matched550_seed91_v1':
            if (args.allow_legacy_training or args.resume or args.initialize_from
                    or args.benchmark_warmup is not None or args.reference_execution or args.mi_cache_dir
                    or (explicit_device and args.device != 'cuda:0')):
                parser.error('The matched local-Q ablation is fresh-only with pinned data and device')
            from experiments.ising_b0_local_motif_supervised import run
            return run(args.config,args.output)
        if isinstance(config_header, dict) and config_header.get('recipe') == 'ising2d_B0_cnn_g1_supervision10_free_decoder_10min_seed91_v1':
            if (args.allow_legacy_training or args.resume or args.initialize_from
                    or args.benchmark_warmup is not None or args.reference_execution or args.mi_cache_dir
                    or (explicit_device and args.device != 'cuda:0')):
                parser.error('The scalar supervised 10-minute experiment has pinned data and device')
            from experiments.ising_b0_supervised_short import run
            return run(args.config,args.output)
        if isinstance(config_header, dict) and config_header.get('recipe') == 'ising2d_KB_local_motif_R_plateau_continue_v1':
            if (args.allow_legacy_training or args.resume or args.initialize_from
                    or args.benchmark_warmup is not None or args.reference_execution or args.mi_cache_dir
                    or (explicit_device and args.device != 'cuda:0')):
                parser.error('The plateau continuation inherits the pinned final R/Adam and frozen Q/C')
            from experiments.ising_kb_r_plateau_continue import run
            return run(args.config,args.output)
        if isinstance(config_header, dict) and config_header.get('recipe') == 'ising2d_KB_local_motif_frozen_QC_fullbatch_R_v1':
            if (args.allow_legacy_training or args.resume or args.initialize_from
                    or args.benchmark_warmup is not None or args.reference_execution or args.mi_cache_dir
                    or (explicit_device and args.device != 'cuda:0')):
                parser.error('The local-Q frozen-QC diagnostic has a pinned source and masked GPU')
            from experiments.ising_kb_local_motif_frozen_r import run
            return run(args.config,args.output)
        if isinstance(config_header, dict) and config_header.get('recipe') == 'ising2d_KB_free_decoder_frozen_QC_fullbatch_R_v1':
            if (args.allow_legacy_training or args.resume or args.initialize_from
                    or args.benchmark_warmup is not None or args.reference_execution or args.mi_cache_dir
                    or (explicit_device and args.device != 'cuda:0')):
                parser.error('The explicit frozen-QC R diagnostic has pinned source and device')
            from experiments.ising_kb_frozen_r import run
            return run(args.config, args.output)
        if isinstance(config_header, dict) and config_header.get('recipe') == 'ising2d_KB_cnn_b3_g2_supervision10_free_decoder_matched1250_seed91_v1':
            if (args.allow_legacy_training or args.resume or args.initialize_from
                    or args.benchmark_warmup is not None or args.reference_execution or args.mi_cache_dir
                    or (explicit_device and args.device != 'cuda:0')):
                parser.error('The free-decoder KB ablation is fresh-only with a fixed UUID-masked GPU')
            from experiments.ising_kb_free_decoder_matched import run
            return run(args.config, args.output)
        if isinstance(config_header, dict) and config_header.get('recipe') == 'ising2d_KB_cnn_b3_g2_supervision1_matched1250_seed91_v1':
            if (args.allow_legacy_training or args.resume or args.initialize_from
                    or args.benchmark_warmup is not None or args.reference_execution or args.mi_cache_dir
                    or (explicit_device and args.device != 'cuda:0')):
                parser.error('The weight-one KB ablation is fresh-only with a fixed UUID-masked GPU')
            from experiments.ising_kb_weak_supervision_matched import run
            return run(args.config, args.output)
        if isinstance(config_header, dict) and config_header.get('recipe') == 'ising2d_KB_cnn_b3_g2_unsupervised_matched1250_seed91_v1':
            if (args.allow_legacy_training or args.resume or args.initialize_from
                    or args.benchmark_warmup is not None or args.reference_execution or args.mi_cache_dir
                    or (explicit_device and args.device != 'cuda:0')):
                parser.error('The matched unsupervised KB trial is fresh-only with a fixed UUID-masked GPU')
            from experiments.ising_kb_unsupervised_matched import run
            return run(args.config, args.output)
        if isinstance(config_header, dict) and config_header.get('recipe') == 'ising2d_KB_cnn_b3_g2_supervised_short_seed91_v1':
            if (args.allow_legacy_training or args.resume or args.initialize_from
                    or args.benchmark_warmup is not None or args.reference_execution or args.mi_cache_dir
                    or (explicit_device and args.device != 'cuda:0')):
                parser.error('The supervised KB short trial is fresh-only with a fixed UUID-masked GPU')
            from experiments.ising_kb_supervised_short import run
            return run(args.config, args.output)
        if isinstance(config_header, dict) and config_header.get('recipe') == 'ising2d_KB_cnn_b3_g2_Rlr003_seed91_v1':
            if (args.allow_legacy_training or args.resume or args.initialize_from
                    or args.benchmark_warmup is not None or args.reference_execution or args.mi_cache_dir
                    or (explicit_device and args.device != 'cuda:0')):
                parser.error('The explicit KB CNN trial is fresh-only with a fixed UUID-masked GPU')
            from experiments.ising_kb_cnn_qcr import run
            return run(args.config, args.output)
        if isinstance(config_header, dict) and config_header.get('recipe') in (
                'phi4_ru_b3_cnn_g2_Rlr003_v1', 'phi4_ru_b3_cnn_g2_Rlr003_saddle_window_v1',
                'phi4_ru_b3_cnn_g2_Rlr003_nonlinear_decoder_v1',
                'phi4_ru_b3_cnn_g2_Rlr003_nonlinear_local_window_v1',
                'phi4_kappa_h_b3_cnn_g2_nonlinear_seed91_v1'):
            if (args.allow_legacy_training or args.resume or args.initialize_from
                    or args.benchmark_warmup is not None or args.reference_execution or args.mi_cache_dir
                    or (explicit_device and args.device != 'cuda:0')):
                parser.error('The explicit ru CNN trial is fresh-only with a fixed UUID-masked GPU')
            from experiments.phi4_ru_cnn_qcr import run
            return run(args.config, args.output)
        if isinstance(config_header, dict) and config_header.get('recipe') == 'phi4_hopping_b3_configuration_cnn_refinedR_v1':
            if (args.allow_legacy_training or args.resume or args.initialize_from
                    or args.benchmark_warmup is not None or args.reference_execution or args.mi_cache_dir
                    or (explicit_device and args.device != 'cuda:0')):
                parser.error('Refined continuous CNN trial is fresh-only, with fixed UUID-masked cuda:0 and a complete 20-minute budget')
            from experiments.phi4_hopping_cnn_qcr import main as cnn_train
            return cnn_train(['--config', args.config, '--output', args.output, '--execute'])
        if isinstance(config_header, dict) and config_header.get('recipe') == 'phi4_hopping_raw_patch_joint625_v1':
            if (args.allow_legacy_training or args.resume or args.initialize_from
                    or args.benchmark_warmup is not None or args.reference_execution or args.mi_cache_dir):
                parser.error('The explicit continuous hopping recipe is fresh-only and has no legacy, resume or binary-MI-cache mode')
            from experiments.phi4_hopping_qcr import main as continuous_train
            command = ['--config', args.config, '--output', args.output, '--execute']
            if explicit_device:
                command += ['--device', args.device]
            continuous_train(command)
            return
        if args.allow_legacy_training:
            if args.device!='cpu' or args.benchmark_warmup is not None or args.reference_execution or args.mi_cache_dir:
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
            if args.reference_execution:command+=['--reference-execution']
            if args.mi_cache_dir:command+=['--mi-cache-dir',args.mi_cache_dir]
            if args.resume:command+=['--resume-checkpoint',args.resume]
            default_train(command)
    else:
        print(evaluate_run(args.run,checkpoint=args.checkpoint,cache=args.cache,
                           max_rg_steps=args.max_rg_steps,output=args.output,make_plots=args.command=='plot'))


if __name__=='__main__':main()
