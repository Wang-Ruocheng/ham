"""Explicit MI-free ablation: no MI work, while independently routed Q/C/R train."""
import json

import pytest
import torch

from experiments import rsmi_separate_qcr as driver
from experiments.potts_frozen_qc_continue import pattern_function
from recurrent_rg.checkpoints import build_model
from recurrent_rg.config import DataConfig,RunConfig
from recurrent_rg.losses import qcr_objective


def _args(*extra):
    return driver.build_parser().parse_args([
        '--config', 'unused.json', '--output', 'unused', *extra])


def _config(tmp_path, *, epochs=3):
    generator=torch.Generator().manual_seed(31)
    train=(2*torch.randint(2,(3,4,16,16,16,1),generator=generator)-1).double()
    test=(2*torch.randint(2,(2,4,16,16,16,1),generator=generator)-1).double()
    cache=tmp_path/'data.pt'
    torch.save((train,torch.tensor([.2,.3,.4],dtype=torch.double),
                test,torch.tensor([.25,.35],dtype=torch.double)),cache)
    config=RunConfig.from_dict(dict(schema_version=5,
        data=dict(cache=str(cache),physical_model='ising3d',coupling_names=['K']),
        model=dict(dimension=3,width=2,theory_dim=1,hidden=5,block_factor=2,
                   affine_reconstruction=True),
        training=dict(max_rg_steps=2,epochs=epochs,seed=91,system_batch=2,
                      sample_batch=4,checkpoint_every=1,log_every=1,
                      gradient_mode='qc_r_separate',threads=1),
        evaluation=dict(max_rg_steps=2,samples=4,systems=2)))
    path=tmp_path/'config.json'
    path.write_text(json.dumps(config.to_dict()))
    return config,path


def _forbid_mi(monkeypatch):
    def forbidden(*args,**kwargs):
        pytest.fail('Zero RSMI must not construct/read statistics, critic, or evaluate MI')
    for name in ('EnvironmentCritic','region_offsets','sample_regions',
        'kernel_probabilities','categorical_kernel_probabilities','enumerated_infonce',
        'binary_chain_joint_histograms','histogram_mutual_information',
        'binary_lattice_mi_statistics','categorical_lattice_mi_statistics',
        'categorical_local_pair_mi_statistics','best_binary_pair_kernel',
        'best_deterministic_binary_rule','optimize_categorical_rule',
        'initialize_kernel_from_table','evaluate_information'):
        monkeypatch.setattr(driver,name,forbidden)
    import recurrent_rg.mi_cache as cache
    monkeypatch.setattr(cache,'load_or_build_mi_statistics',forbidden)
    monkeypatch.setattr(cache,'lattice_statistics_cache_key',forbidden)


def test_default_recipe_remains_two_and_zero_requires_explicit_ablation():
    default=_args()
    assert default.rsmi_weight==2.
    driver.validate_rsmi_options(default)
    driver.validate_cuda_options(default)
    zero=_args('--rsmi-weight','0')
    with pytest.raises(ValueError,match='allow-nondefault-recipe'):
        driver.validate_rsmi_options(zero)
    zero.allow_nondefault_recipe=True
    driver.validate_rsmi_options(zero)
    driver.validate_cuda_options(zero)
    with pytest.raises(ValueError,match='Locked default recipe'):
        driver.validate_recipe(_args('--rsmi-weight','0'),RunConfig(data=DataConfig(cache='unused.pt',physical_model='test')))


@pytest.mark.parametrize('weight',['-1','nan','inf'])
def test_invalid_rsmi_weights_fail_before_work(weight):
    with pytest.raises(ValueError,match='Nonnegative finite RSMI'):
        driver.validate_rsmi_options(_args('--rsmi-weight',weight))


@pytest.mark.parametrize('option',[
    ['--mi-initialize'],['--mi-cache-dir','unused'],['--mi-test-cpu'],
    ['--estimator','infonce'],['--environment','local_pairs'],
    ['--no-independent-c'],['--freeze-c-checkpoint','unused.pt'],
    ['--q-future-weight','1'],['--joint-train-plateau']])
def test_zero_rsmi_rejects_mi_options_or_other_incompatible_ablations(option):
    with pytest.raises(ValueError):
        driver.validate_rsmi_options(_args('--rsmi-weight','0',
                                          '--allow-nondefault-recipe',*option))


def test_zero_rsmi_3d_smoke_skips_every_mi_path_and_updates_all_modules(tmp_path,monkeypatch,capsys):
    _,config=_config(tmp_path)
    _forbid_mi(monkeypatch)
    models=[]
    def capture(*args,**kwargs):
        model=build_model(*args,**kwargs);models.append(model);return model
    monkeypatch.setattr(driver,'build_model',capture)
    output=tmp_path/'no_mi'
    driver.main(['--config',str(config),'--output',str(output),'--rsmi-weight','0',
                 '--allow-nondefault-recipe','--skip-final-report'])
    manifest=json.loads((output/'manifest.json').read_text())
    summary=json.loads((output/'summary.json').read_text())
    assert summary['actual_epochs']==3
    for contract in (manifest,summary['contract']):
        assert contract['rsmi_enabled'] is False and contract['mi_evaluated'] is False
        assert contract['rsmi_weight']==0 and contract['estimator'] is None
        assert contract['environment_mode'] is None and contract['rsmi_scales']==[]
        assert contract['critic_parameters']==0 and contract['critic_hidden'] is None
        assert contract['execution_optimizations']['mi_statistics_cache']=={}
        assert contract['execution_optimizations']['selected_empirical_mi'] is False
        assert 'MI is not computed' in contract['note']
    for row in summary['history']:
        for split in ('train','test'):
            assert row[f'{split}_rsmi'] is None
            assert row[f'{split}_rsmi_bits'] is None
            assert row[f'{split}_shuffled_rsmi_bits'] is None
            assert row[f'{split}_total']==row[f'{split}_qcr_total']
    final=driver.qcr_module_sha256(models[0])
    assert all(final[name]!=manifest['training_start_module_sha256'][name] for name in ('Q','C','R'))
    checkpoint=torch.load(output/'checkpoint_last.pt',map_location='cpu',weights_only=True)
    assert 'critic' not in checkpoint and 'critic_optimizer' not in checkpoint
    assert not (tmp_path/'.qcr_mi_cache').exists()
    log=capsys.readouterr().out
    assert 'RSMI disabled; MI not evaluated' in log and 'I_histogram=' not in log


def test_zero_rsmi_preserves_initialization_and_qcr_sampling_rng_against_default(tmp_path,monkeypatch):
    _,config=_config(tmp_path,epochs=1)
    original_objective=driver.qcr_objective
    streams=[]
    def capture(*args,**kwargs):
        streams.append(torch.get_rng_state().clone())
        return original_objective(*args,**kwargs)
    monkeypatch.setattr(driver,'qcr_objective',capture)
    for name,options in [('default',[]),('zero',['--rsmi-weight','0','--allow-nondefault-recipe'])]:
        driver.main(['--config',str(config),'--output',str(tmp_path/name),
                     '--skip-final-report',*options])
    baseline=json.loads((tmp_path/'default'/'manifest.json').read_text())
    ablation=json.loads((tmp_path/'zero'/'manifest.json').read_text())
    for field in ('random_construction_module_sha256','pre_affine_initialization_module_sha256',
                  'training_start_module_sha256'):
        assert baseline[field]==ablation[field]
    # Driver qcr_objective is called only for actual updates, not evaluate().
    assert len(streams)==2 and torch.equal(streams[0],streams[1])


def test_zero_rsmi_exact_resume_reproduces_uninterrupted_training(tmp_path,monkeypatch):
    _,config=_config(tmp_path,epochs=3)
    _forbid_mi(monkeypatch)
    common=['--config',str(config),'--rsmi-weight','0','--allow-nondefault-recipe',
            '--skip-final-report']
    driver.main([*common,'--output',str(tmp_path/'full')])
    driver.main([*common,'--output',str(tmp_path/'prefix'),'--epochs','1'])
    driver.main([*common,'--output',str(tmp_path/'resumed'),'--resume-checkpoint',
                 str(tmp_path/'prefix'/'checkpoint_last.pt')])
    a=torch.load(tmp_path/'full'/'checkpoint_last.pt',map_location='cpu',weights_only=True)
    b=torch.load(tmp_path/'resumed'/'checkpoint_last.pt',map_location='cpu',weights_only=True)
    assert a['history']==b['history']
    for name,value in a['model'].items():
        torch.testing.assert_close(value,b['model'][name],rtol=0,atol=0)
    assert torch.equal(a['batch_generator_state'],b['batch_generator_state'])
    assert torch.equal(a['rsmi_generator_state'],b['rsmi_generator_state'])
    assert torch.equal(a['torch_rng_state'],b['torch_rng_state'])


def test_zero_rsmi_real_3d_objectives_keep_independent_gradients(tmp_path):
    cfg,_=_config(tmp_path)
    torch.manual_seed(91);torch.set_num_threads(1)
    model=build_model(cfg.model,2);model.detach_kernel_representation=True
    batch=(2*torch.randint(2,(2,4,8,8,8,1))-1).double()
    total,terms=qcr_objective(model,batch,pattern_fn=pattern_function(cfg),
        kernel_noncollapse_mode='variance_floor_ensemble',kernel_variance_ratio=1.,
        kernel_sign_weight=1.)
    closure=5*(terms['commutator']+terms['rollout'])
    c_objective=2*terms['kernel_noncollapse']+terms['kernel_sign']
    q=list(model.theory_encoder.parameters());c=list(model.rg_kernel.parameters())
    r=list(model.theory_rg.parameters())
    expected=[torch.autograd.grad(objective,parameters,retain_graph=True,allow_unused=True)
              for objective,parameters in ((total-closure,q),(c_objective,c),(closure,r))]
    driver.independent_component_gradients(total,closure,c_objective,q,c,r)
    for parameters,gradients in zip((q,c,r),expected):
        assert any(gradient is not None and gradient.abs().sum()>0 for gradient in gradients)
        for parameter,gradient in zip(parameters,gradients):
            if gradient is None:assert parameter.grad is None
            else:torch.testing.assert_close(parameter.grad,gradient)


def test_retired_distinct_is_not_silently_zeroed_or_allowed_to_reach_data(tmp_path,monkeypatch):
    _,config=_config(tmp_path,epochs=1)
    raw=json.loads(config.read_text())
    raw['loss']['kernel_distinct_alpha']=1.
    config.write_text(json.dumps(raw))
    monkeypatch.setattr(driver,'prepared_data',lambda *a,**k:pytest.fail('Retired loss must fail before reading data'))
    with pytest.raises(ValueError,match='retired'):
        driver.main(['--config',str(config),'--output',str(tmp_path/'invalid'),
                     '--rsmi-weight','0','--allow-nondefault-recipe','--skip-final-report'])
    assert not (tmp_path/'invalid').exists()


def test_component_c_logging_does_not_multiply_unmeasured_mi():
    row=dict(epoch=1,train_anchor=1.,train_coarse_consistency=2.,train_separation=3.,
             train_rsmi=None,train_kernel_noncollapse=.4,train_kernel_sign=.5,
             train_commutator=.6,train_rollout=.7)
    result=driver.train_component_losses(row,RunConfig(data=DataConfig(cache='unused.pt',physical_model='test')),
        _args('--rsmi-weight','0','--allow-nondefault-recipe'))
    assert result['component_C']==pytest.approx(1.3)
