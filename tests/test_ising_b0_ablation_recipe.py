"""Named B0 ablations are narrow authorizations, not a global recipe escape."""
from dataclasses import replace
import json

import pytest
import torch

from experiments import rsmi_separate_qcr as driver
from recurrent_rg.checkpoints import build_model
from recurrent_rg.config import DataConfig, LossConfig, ModelConfig, RunConfig, TrainingConfig


def _args(name=None, *extra):
    values=['--config','unused.json','--output','unused']
    if name is not None:
        values += ['--ising-b0-ablation',name]
    return driver.build_parser().parse_args([*values,*extra])


def _config():
    return RunConfig(data=DataConfig(cache='unused.pt',physical_model='ising2d',
        coupling_names=('K',)),model=ModelConfig(dimension=2,theory_dim=5,hidden=96),
        loss=LossConfig(closure_objective='one_step_and_rollout'),
        training=TrainingConfig(max_rg_steps=2,gradient_mode='qc_r_separate'))


def _case(name):
    cfg=_config()
    options={
        'c_no_mi':['--rsmi-weight','0'],
        'c_no_variance':['--variance-weight','0'],
        'c_no_color':['--color-weight','0']}.get(name,[])
    if name == 'q_no_reconstruction':
        cfg=replace(cfg,loss=replace(cfg.loss,weights=replace(cfg.loss.weights,separation=0.)))
    elif name == 'q_no_subset':
        cfg=replace(cfg,loss=replace(cfg.loss,weights=replace(cfg.loss.weights,
            anchor=0.,coarse_consistency=0.)))
    elif name == 'r_no_rollout':
        cfg=replace(cfg,loss=replace(cfg.loss,rollout_weight=0.))
    return _args(name,*options),cfg


@pytest.mark.parametrize('name',driver.ISING_B0_ABLATIONS)
def test_each_named_ablation_passes_cpu_and_cuda_option_gates(name):
    args,cfg=_case(name)
    driver.validate_ising_b0_ablation(args,cfg)
    driver.validate_rsmi_options(args)
    driver.validate_cuda_options(args)
    driver.validate_recipe(args,cfg)
    contract=driver.ising_b0_ablation_contract(args,cfg)
    assert contract['name']==name
    assert contract['baseline']==(name=='none')
    assert bool(contract['removed_terms'])==(name!='none')
    assert contract['labels_used_in_training'] is False
    assert contract['MI_evaluated']==(name!='c_no_mi')
    assert contract['effective_weights']['R_one_step']==5.


@pytest.mark.parametrize('name',[name for name in driver.ISING_B0_ABLATIONS if name!='none'])
def test_marker_does_not_silently_zero_a_config_or_cli_weight(name):
    with pytest.raises(ValueError,match='ising-b0-ablation'):
        driver.validate_ising_b0_ablation(_args(name),_config())


@pytest.mark.parametrize('extra',[
    ['--allow-nondefault-recipe'], ['--resume-checkpoint','old.pt'],
    ['--mi-initialize'], ['--freeze-c-checkpoint','old.pt'],
    ['--affine-pca-init'], ['--theory-dim','5'], ['--q-coupling-weight','1'],
    ['--q-future-weight','1'], ['--q-future-control'],
    ['--q-future-pairs','pairs.json'], ['--q-future-normalizer','norm.json'],
    ['--q-only-candidates'], ['--q-only-weight-sweep'], ['--q-balanced-reconstruction'],
    ['--q-last-layer-audit'], ['--q-head-structure-audit'], ['--q-feature-bottleneck'],
    ['--q-original-plateau'], ['--q-initialization-pilot'], ['--q-unfolding'],
    ['--r-only-continue'], ['--r-only-refit'], ['--r-only-plateau'],
    ['--joint-train-plateau'], ['--frozen-cache','cache.pt']])
def test_named_gate_rejects_other_ablations_before_data(extra):
    with pytest.raises(ValueError,match='fresh standalone'):
        driver.validate_ising_b0_ablation(_args('none',*extra))


def test_named_gate_rejects_source_without_loading_it():
    args=driver.build_parser().parse_args(['--source','old','--output','new',
                                         '--ising-b0-ablation','none'])
    with pytest.raises(ValueError,match='fresh standalone'):
        driver.validate_ising_b0_ablation(args)


@pytest.mark.parametrize('extra',[
    ['--rsmi-weight','1'], ['--variance-weight','0'], ['--color-weight','0'],
    ['--variance-ratio','.5'], ['--no-variance-per-ensemble'], ['--no-independent-c'],
    ['--estimator','infonce'], ['--environment','local_pairs']])
def test_named_baseline_locks_other_cli_objectives(extra):
    with pytest.raises(ValueError,match='only its named loss'):
        driver.validate_ising_b0_ablation(_args('none',*extra),_config())


@pytest.mark.parametrize('name',driver.ISING_B0_ABLATIONS)
def test_each_named_gate_rejects_second_loss_change(name):
    args,cfg=_case(name)
    cfg=replace(cfg,loss=replace(cfg.loss,isotropy_weight=.01))
    with pytest.raises(ValueError,match='loss.isotropy_weight'):
        driver.validate_recipe(args,cfg)


def test_historical_B0_campaign_does_not_adopt_the_new_closure_default():
    cfg = _config()
    assert cfg.loss.closure_objective == 'one_step_and_rollout'
    changed = replace(cfg, loss=replace(cfg.loss, closure_objective='all_start_anchor'))
    with pytest.raises(ValueError, match='closure_objective'):
        driver.validate_recipe(_args('none'), changed)


@pytest.mark.parametrize('change',[
    dict(theory_dim=1),dict(dimension=1),dict(block_factor=2),dict(width=5),
    dict(fixed_kernel=True),dict(kernel_initialization='positive_decimation'),
    dict(kernel_sampling='deterministic'),dict(kernel_temperature=.5),
    dict(binary_lookup=False),dict(affine_reconstruction=False),
    dict(theory_head='linear')])
def test_named_gate_locks_default_model_except_explicit_capacity_axes(change):
    cfg=_config()
    cfg=replace(cfg,model=replace(cfg.model,**change))
    with pytest.raises(ValueError,match='locked recipe'):
        driver.validate_recipe(_args('none'),cfg)


def test_named_gate_rejects_hidden_neutral_auxiliary():
    cfg=_config()
    cfg=replace(cfg,loss=replace(cfg.loss,kernel_neutral_weight=1.))
    with pytest.raises(ValueError,match='kernel_neutral_weight'):
        driver.validate_recipe(_args('none'),cfg)


def test_uniform_scale_weights_and_existing_multistep_schedule_are_allowed():
    cfg=_config()
    cfg=replace(cfg,loss=replace(cfg.loss,closure_scale_weights=(3.,3.)),
        training=replace(cfg.training,lr_schedule='multistep',
            lr_milestones=(6000,9000),lr_gamma=.3))
    driver.validate_recipe(_args('none'),cfg)
    cfg=replace(cfg,loss=replace(cfg.loss,closure_scale_weights=(1.8,.2)))
    with pytest.raises(ValueError,match='uniform_closure'):
        driver.validate_recipe(_args('none'),cfg)


@pytest.mark.parametrize('extra',[['--mi-cache-dir','unused'],['--mi-test-cpu']])
def test_named_no_mi_rejects_unused_cache_and_placement(extra):
    with pytest.raises(ValueError,match='unused_mi'):
        driver.validate_rsmi_options(_args('c_no_mi','--rsmi-weight','0',*extra))


def test_absent_marker_keeps_historical_recipe_and_zero_mi_opt_in():
    args=_args(None,'--rsmi-weight','0','--allow-nondefault-recipe')
    driver.validate_rsmi_options(args)
    driver.validate_cuda_options(args)
    driver.validate_recipe(args,_config())
    assert driver.ising_b0_ablation_contract(args,_config()) is None
    # Deliberately do not retroactively reject historical serialized Q/R
    # weights: only the explicit campaign marker gains the stricter gate.
    cfg=_config()
    cfg=replace(cfg,loss=replace(cfg.loss,rollout_weight=0.))
    driver.validate_recipe(_args(),cfg)


def test_historical_future_normalizer_omitted_features_means_exactly_twelve():
    cfg=_config()
    model=build_model(cfg.model,2)
    old_model=cfg.to_dict()['model']
    old_model.pop('local_features')
    contract=dict(weight=1.,normalizer_provenance=dict(model_config=old_model,
        initial_C_sha256=driver.future_module_sha(model.rg_kernel),
        scales=[36,12,4],samples=4))
    data=torch.zeros(2,4,36,36,1)
    driver.validate_future_normalizer_model(contract,cfg,model,data)
    changed=replace(cfg,model=replace(cfg.model,local_features=24))
    with pytest.raises(ValueError,match='Frozen future normalizer'):
        driver.validate_future_normalizer_model(contract,changed,model,data)


def test_named_zero_mi_smoke_skips_mi_and_records_exact_objective(tmp_path,monkeypatch):
    generator=torch.Generator().manual_seed(45)
    train=(2*torch.randint(2,(3,4,36,36,1),generator=generator)-1).double()
    test=(2*torch.randint(2,(2,4,36,36,1),generator=generator)-1).double()
    cache=tmp_path/'data.pt'
    torch.save((train,torch.tensor([.2,.3,.4]),test,torch.tensor([.25,.35])),cache)
    cfg=_config()
    cfg=replace(cfg,data=replace(cfg.data,cache=str(cache)),
        model=replace(cfg.model,hidden=5),
        training=replace(cfg.training,epochs=2,threads=1,system_batch=2,sample_batch=4,
            checkpoint_every=1,log_every=1),
        evaluation=replace(cfg.evaluation,max_rg_steps=2,samples=4,systems=2))
    path=tmp_path/'config.json'
    path.write_text(json.dumps(cfg.to_dict()))
    def forbidden(*args,**kwargs):
        pytest.fail('Named no-MI arm must not compute or cache MI')
    for name in ('EnvironmentCritic','region_offsets','sample_regions',
        'kernel_probabilities','categorical_kernel_probabilities','enumerated_infonce',
        'binary_chain_joint_histograms','histogram_mutual_information',
        'binary_lattice_mi_statistics','categorical_lattice_mi_statistics',
        'categorical_local_pair_mi_statistics','best_binary_pair_kernel',
        'best_deterministic_binary_rule','optimize_categorical_rule',
        'initialize_kernel_from_table','evaluate_information'):
        monkeypatch.setattr(driver,name,forbidden)
    import recurrent_rg.mi_cache as mi_cache
    monkeypatch.setattr(mi_cache,'load_or_build_mi_statistics',forbidden)
    monkeypatch.setattr(mi_cache,'lattice_statistics_cache_key',forbidden)
    models=[]
    def capture(*args,**kwargs):
        model=build_model(*args,**kwargs);models.append(model);return model
    monkeypatch.setattr(driver,'build_model',capture)
    output=tmp_path/'run'
    driver.main(['--config',str(path),'--output',str(output),
        '--ising-b0-ablation','c_no_mi','--rsmi-weight','0','--skip-final-report'])
    manifest=json.loads((output/'manifest.json').read_text())
    summary=json.loads((output/'summary.json').read_text())
    for value in (manifest,summary['contract']):
        assert value['ising_b0_ablation']['name']=='c_no_mi'
        assert value['ising_b0_ablation']['effective_weights']==dict(C_MI=0.,
            C_variance=2.,C_color=1.,Q_reconstruction=10.,Q_microscopic_subset=20.,
            Q_coarse_subset=5.,R_one_step=5.,R_multistep=5.)
        assert value['execution_optimizations']['mi_statistics_cache']=={}
        assert value['rsmi_enabled'] is False and value['mi_evaluated'] is False
    assert summary['actual_epochs']==2
    final=driver.qcr_module_sha256(models[0])
    assert all(final[name]!=manifest['training_start_module_sha256'][name]
               for name in ('Q','C','R'))
    assert all(row['train_rsmi'] is None for row in summary['history'])
    assert not (tmp_path/'.qcr_mi_cache').exists()


def test_all_named_arms_real_cpu_smoke_preserve_paired_initialization(tmp_path):
    generator=torch.Generator().manual_seed(45)
    train=(2*torch.randint(2,(3,4,12,12,1),generator=generator)-1).double()
    test=(2*torch.randint(2,(2,4,12,12,1),generator=generator)-1).double()
    cache=tmp_path/'data.pt'
    torch.save((train,torch.tensor([.2,.3,.4]),test,torch.tensor([.25,.35])),cache)
    initial=[]
    for name in driver.ISING_B0_ABLATIONS:
        args,cfg=_case(name)
        cfg=replace(cfg,data=replace(cfg.data,cache=str(cache)),
            model=replace(cfg.model,hidden=5),
            training=replace(cfg.training,max_rg_steps=1,epochs=1,threads=1,
                system_batch=2,sample_batch=4,checkpoint_every=1,log_every=1),
            evaluation=replace(cfg.evaluation,max_rg_steps=1,samples=4,systems=2))
        path=tmp_path/f'{name}.json'
        path.write_text(json.dumps(cfg.to_dict()))
        options=[]
        if not args.rsmi_weight:options += ['--rsmi-weight','0']
        if not args.variance_weight:options += ['--variance-weight','0']
        if not args.color_weight:options += ['--color-weight','0']
        output=tmp_path/name
        driver.main(['--config',str(path),'--output',str(output),
            '--ising-b0-ablation',name,'--skip-final-report',*options])
        manifest=json.loads((output/'manifest.json').read_text())
        summary=json.loads((output/'summary.json').read_text())
        assert summary['actual_epochs']==1
        assert manifest['ising_b0_ablation']['name']==name
        initial.append(manifest['training_start_module_sha256'])
    assert all(value==initial[0] for value in initial)
