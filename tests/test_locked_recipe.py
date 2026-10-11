"""Default entrypoints cannot silently omit MI or change gradient ownership."""
from dataclasses import replace
import pytest
from experiments.rsmi_separate_qcr import build_parser, validate_recipe
from recurrent_rg.config import RunConfig, DataConfig
from recurrent_rg.runner import run_training
from recurrent_rg.cli import main


def arguments(*extra):
    return build_parser().parse_args(['--config','unused.json','--output','unused',*extra])


def test_locked_defaults():
    args=arguments()
    cfg=RunConfig(data=DataConfig(cache='unused',physical_model='test'))
    validate_recipe(args,cfg)
    assert (args.estimator,args.rsmi_weight,args.variance_weight,args.color_weight)==('histogram',2,2,1)
    assert args.variance_ratio==1 and args.variance_per_ensemble and args.independent_c
    assert not args.mi_initialize


def test_specific_frozen_c_ablation_does_not_unlock_other_defaults():
    cfg=RunConfig(data=DataConfig(cache='unused',physical_model='test'))
    cfg=replace(cfg,training=replace(cfg.training,train_modules=('Q','R')))
    with pytest.raises(ValueError,match='train_modules'):
        validate_recipe(arguments(),cfg)
    validate_recipe(arguments('--freeze-c-checkpoint','source.pt'),cfg)
    with pytest.raises(ValueError,match='color_weight'):
        validate_recipe(arguments('--freeze-c-checkpoint','source.pt','--color-weight','0'),cfg)


@pytest.mark.parametrize('flags',[
    ['--color-weight','0'],['--estimator','infonce'],['--variance-weight','10'],
    ['--no-independent-c'],['--no-variance-per-ensemble'],['--mi-initialize'],
])
def test_deviations_require_explicit_ablation(flags):
    cfg=RunConfig(data=DataConfig(cache='unused',physical_model='test'))
    with pytest.raises(ValueError,match='Locked default'):
        validate_recipe(arguments(*flags),cfg)
    validate_recipe(arguments(*flags,'--allow-nondefault-recipe'),cfg)


def test_model_initialization_is_locked():
    cfg=RunConfig(data=DataConfig(cache='unused',physical_model='test'))
    cfg=replace(cfg,model=replace(cfg.model,kernel_initialization='positive_decimation'))
    with pytest.raises(ValueError,match='kernel_initialization'):
        validate_recipe(arguments(),cfg)


def test_legacy_runner_fails_before_reading_data(tmp_path):
    with pytest.raises(ValueError,match='legacy runner'):
        run_training(None,tmp_path/'unused')


def test_cli_dispatches_to_independent_empirical_recipe(monkeypatch):
    calls=[]
    monkeypatch.setattr('experiments.rsmi_separate_qcr.main',lambda argv:calls.append(argv))
    main(['train','--config','c.json','--output','out','--resume','checkpoint.pt'])
    assert calls==[['--config','c.json','--output','out','--resume-checkpoint','checkpoint.pt']]
