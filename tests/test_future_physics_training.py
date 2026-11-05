"""Synthetic-only integration checks for the explicit future-Q trainer path."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from experiments import rsmi_separate_qcr as driver
from recurrent_rg.checkpoints import build_model, isolated_rng, load_run
from recurrent_rg.config import ModelConfig, RunConfig
from recurrent_rg.data.statistics import plaquette_distribution
from recurrent_rg.future_physics import FuturePhysicsPredictor, BINARY2D_OBSERVABLE_NAMES
from recurrent_rg.losses import qcr_objective


def arguments(*extra):
    return driver.build_parser().parse_args(['--config','unused.json','--output','unused',*extra])


def pair_files(tmp_path, *, data_sha='a'*64, count=4):
    raw=dict(schema_version=1,data_sha256=data_sha)
    for split,offset in [('train',0),('test',100)]:
        raw[split]=dict(condition_ids=[i//2+offset for i in range(count)],
            chain_ids=list(range(offset,offset+count)),
            pairs=[[i,i+1] for i in range(0,count,2)])
    pairs=tmp_path/'pairs.json';pairs.write_text(json.dumps(raw))
    norm=tmp_path/'normalizer.json'
    cfg=future_config('unused.pt')
    with isolated_rng('cpu',cfg.training.seed):
        initial=build_model(cfg.model,2)
    norm.write_text(json.dumps(dict(schema_version=1,data_sha256=data_sha,
        pairs_sha256=driver.file_fingerprint(pairs),fitting_split='train',
        target_mean=[[.012345678901234]*5]*3,target_std=[[.02]*5]*3,
        target_std_unfloored=[[.02]*5]*3,std_floor=.01,paths=[[0,1],[1,2],[0,2]],
        observable_names=list(BINARY2D_OBSERVABLE_NAMES),fitting_ensembles=count,C_draws=5,
        model_pretraining_updates=0,QCR_updates=0,calibration_seed=2026101719,
        initial_C_sha256=driver.future_module_sha(initial.rg_kernel),model_config=cfg.to_dict()['model'],
        scales=[108,36,12],samples=4)))
    return pairs,norm,raw


def test_zero_default_has_no_files_predictor_rng_or_contract(monkeypatch):
    args=arguments()
    assert args.q_future_weight==0 and not args.q_future_control
    before=torch.get_rng_state().clone()
    monkeypatch.setattr(Path,'read_text',lambda *a,**k:pytest.fail('Zero default must not read future files'))
    driver.validate_future_options(args)
    assert driver.load_future_inputs(args,'unused',{'train':0,'test':0})==(None,None)
    driver.validate_resume_future({'metadata':{}},None,'cpu',{})
    assert torch.equal(before,torch.get_rng_state())


def test_paired_control_validates_pairs_but_does_not_read_normalizer(tmp_path):
    pairs,_,_=pair_files(tmp_path)
    args=arguments('--q-future-control','--q-future-pairs',str(pairs),
                   '--q-future-normalizer',str(tmp_path/'does-not-exist.json'))
    driver.validate_future_options(args)
    identities,contract=driver.load_future_inputs(args,'a'*64,{'train':4,'test':4})
    assert contract['mode']=='paired_control' and contract['weight']==0
    assert contract['target_mean'] is contract['target_std'] is contract['normalizer_sha256'] is None
    assert identities['train']['pairs'].shape==(2,2)
    assert contract['observables']==list(BINARY2D_OBSERVABLE_NAMES)


@pytest.mark.parametrize('extra',[
    ['--q-future-control','--q-future-weight','1'],
    ['--q-future-weight','-1'],['--q-future-weight','nan'],
    ['--q-future-pairs','p.json'],['--q-future-weight','1'],
])
def test_invalid_future_cli_fails_closed(extra):
    with pytest.raises(ValueError):driver.validate_future_options(arguments(*extra))


@pytest.mark.parametrize('option,value',[
    ('q_coupling_weight',1.),('source',Path('old')),('allow_nondefault_recipe',True),
    ('mi_initialize',True),('freeze_c_checkpoint',Path('old')),('affine_pca_init',True),
    ('independent_c',False),('estimator','infonce'),('environment','local_pairs'),
    ('q_future_hidden',64),('r_only_continue',True),('q_unfolding',True),
    ('joint_train_plateau',True),('frozen_cache',Path('old')),
])
def test_future_or_control_cannot_enable_labels_or_other_recipes(option,value):
    args=arguments('--q-future-control','--q-future-pairs','pairs.json')
    setattr(args,option,value)
    with pytest.raises(ValueError):driver.validate_future_options(args)


@pytest.mark.parametrize('mutation',[
    'data_sha','duplicate_pair','pair_cross_condition','duplicate_chain','cross_split_chain','K_field',
])
def test_pair_provenance_rejects_mismatch_overlap_and_labels(tmp_path,mutation):
    pairs,_,raw=pair_files(tmp_path)
    if mutation=='data_sha':raw['data_sha256']='b'*64
    if mutation=='duplicate_pair':raw['train']['pairs']=[[0,1],[0,3]]
    if mutation=='pair_cross_condition':raw['train']['pairs']=[[0,2],[1,3]]
    if mutation=='duplicate_chain':raw['train']['chain_ids'][1]=0
    if mutation=='cross_split_chain':raw['test']['chain_ids'][0]=0
    if mutation=='K_field':raw['train']['K']=[.42,.42,.46,.46]
    pairs.write_text(json.dumps(raw))
    args=arguments('--q-future-control','--q-future-pairs',str(pairs))
    with pytest.raises(ValueError):driver.load_future_inputs(args,'a'*64,{'train':4,'test':4})


@pytest.mark.parametrize('mutation',['split','pair_sha','data_sha','shape','negative','nonfinite'])
def test_normalizer_is_frozen_train_bound_and_finite(tmp_path,mutation):
    pairs,norm,_=pair_files(tmp_path)
    raw=json.loads(norm.read_text())
    if mutation=='split':raw['fitting_split']='test'
    if mutation=='pair_sha':raw['pairs_sha256']='b'*64
    if mutation=='data_sha':raw['data_sha256']='b'*64
    if mutation=='shape':raw['target_mean']=[[0.]*5]
    if mutation=='negative':raw['target_std'][0][0]=-.1
    if mutation=='nonfinite':raw['target_mean'][0][0]=float('nan')
    norm.write_text(json.dumps(raw))
    args=arguments('--q-future-weight','1','--q-future-pairs',str(pairs),'--q-future-normalizer',str(norm))
    with pytest.raises(ValueError):driver.load_future_inputs(args,'a'*64,{'train':4,'test':4})


def test_fixed_normalizer_precision_and_floor_are_checkpointed(tmp_path):
    pairs,norm,_=pair_files(tmp_path)
    args=arguments('--q-future-weight','1','--q-future-pairs',str(pairs),'--q-future-normalizer',str(norm))
    _,contract=driver.load_future_inputs(args,'a'*64,{'train':4,'test':4})
    predictor=FuturePhysicsPredictor(1,target_mean=contract['target_mean'],
        target_std=[[0.]*5]*3,std_floor=contract['std_floor'])
    assert predictor.target_mean.dtype==torch.float64
    assert predictor.target_mean[0,0].item()==.012345678901234
    assert torch.equal(predictor.target_std,torch.full((3,5),.01,dtype=torch.float64))
    assert predictor.state_dict()['_extra_state']['std_floor']==.01


@pytest.mark.parametrize('change',['initial_C','model','scale','samples'])
def test_normalizer_binds_initial_c_model_and_geometry(tmp_path,change):
    pairs,norm,_=pair_files(tmp_path)
    args=arguments('--q-future-weight','1','--q-future-pairs',str(pairs),'--q-future-normalizer',str(norm))
    _,contract=driver.load_future_inputs(args,'a'*64,{'train':4,'test':4})
    cfg=future_config('unused.pt')
    with isolated_rng('cpu',cfg.training.seed):model=build_model(cfg.model,2)
    data=torch.empty((4,4,108,108,1),device='meta')
    driver.validate_future_normalizer_model(contract,cfg,model,data)
    provenance=contract['normalizer_provenance']
    if change=='initial_C':provenance['initial_C_sha256']='b'*64
    if change=='model':provenance['model_config']['hidden']=96
    if change=='scale':provenance['scales']=[108,54,27]
    if change=='samples':provenance['samples']=64
    with pytest.raises(ValueError):driver.validate_future_normalizer_model(contract,cfg,model,data)


def test_pair_sampling_is_complete_and_resume_reproducible(tmp_path):
    pairs,_,_=pair_files(tmp_path,count=8)
    args=arguments('--q-future-control','--q-future-pairs',str(pairs))
    identities,_=driver.load_future_inputs(args,'a'*64,{'train':8,'test':8})
    generator=torch.Generator().manual_seed(91)
    saved=generator.get_state().clone()
    first=driver.future_batch_ids(identities['train'],6,generator)
    generator.set_state(saved)
    assert torch.equal(first,driver.future_batch_ids(identities['train'],6,generator))
    assert len(first.unique())==6
    rows=first.reshape(-1,2)
    assert torch.equal(rows[:,1],rows[:,0]+1)
    assert torch.equal(driver.future_batch_ids(identities['train'],4),torch.tensor([0,1,6,7]))


@pytest.mark.parametrize('shape',[(2,4,36,36,1),(2,4,108,54,1),(2,4,81,81,1),(2,4,135,135,1)])
def test_adapter_domain_rejected_before_training(shape):
    with pytest.raises(ValueError):driver.validate_future_data(torch.empty(shape,device='meta'))


def future_config(cache):
    return RunConfig.from_dict(dict(schema_version=5,
        data=dict(cache=str(cache),physical_model='ising2d',coupling_names=['K']),
        model=dict(dimension=2,theory_dim=1,hidden=5,block_factor=3,affine_reconstruction=True),
        training=dict(max_rg_steps=2,epochs=2,system_batch=2,sample_batch=4,
            checkpoint_every=1,log_every=1,gradient_mode='qc_r_separate',threads=1),
        evaluation=dict(max_rg_steps=2,samples=4,systems=2)))


@pytest.mark.parametrize('change',['block','depth','pool','tangent','orientation','weights','metric'])
def test_future_config_rejects_unsupported_recipe(change):
    cfg=future_config('unused.pt')
    if change=='block':cfg=replace(cfg,model=replace(cfg.model,block_factor=2))
    if change=='depth':cfg=replace(cfg,training=replace(cfg.training,max_rg_steps=1))
    if change=='pool':cfg=replace(cfg,data=replace(cfg.data,train_pool_replicas=2))
    if change=='tangent':cfg=replace(cfg,loss=replace(cfg.loss,tangent_weight=1.))
    if change=='orientation':cfg=replace(cfg,loss=replace(cfg.loss,orientation_weight=1.))
    if change=='weights':cfg=replace(cfg,loss=replace(cfg.loss,weights=replace(cfg.loss.weights,separation=2.)))
    if change=='metric':cfg=replace(cfg,loss=replace(cfg.loss,closure_metric='noise_aware'))
    with pytest.raises(ValueError):
        driver.validate_future_options(arguments('--q-future-control','--q-future-pairs','p.json'),cfg)


def test_auxiliary_gradient_is_q_only_and_reuses_existing_c_samples():
    torch.set_num_threads(1);torch.manual_seed(54)
    model=build_model(ModelConfig(dimension=2,hidden=5,block_factor=3,affine_reconstruction=True),2)
    model.detach_kernel_representation=True
    batch=(2*torch.randint(2,(2,4,108,108,1))-1).double()
    with isolated_rng('cpu',999):
        predictor=FuturePhysicsPredictor(1,target_mean=torch.zeros(3,5,dtype=torch.double),
            target_std=torch.ones(3,5,dtype=torch.double),std_floor=.01)
    levels=[];base,terms=qcr_objective(model,batch,pattern_fn=plaquette_distribution,levels_sink=levels)
    identities=dict(condition_ids=torch.tensor([0,0]),chain_ids=torch.tensor([0,1]))
    before=torch.get_rng_state().clone()
    auxiliary,aux_terms=driver.future_batch_loss(predictor,model,levels,identities,torch.tensor([0,1]),1.)
    assert torch.equal(before,torch.get_rng_state())
    closure=5*(terms['commutator']+terms['rollout'])
    q=list(model.theory_encoder.parameters());p=list(predictor.parameters())
    c=list(model.rg_kernel.parameters());r=list(model.theory_rg.parameters())
    assert all(v is None for v in torch.autograd.grad(auxiliary,c+r,allow_unused=True,retain_graph=True))
    qaux=torch.autograd.grad(auxiliary,q+p,allow_unused=True,retain_graph=True)
    assert any(g is not None and g.abs().sum()>0 for g in qaux[:len(q)])
    assert all(g is not None and torch.isfinite(g).all() for g in qaux[len(q):])
    c_objective=terms['kernel_noncollapse']+sum(v.square().sum() for v in c)
    expected_q=torch.autograd.grad(base+auxiliary-closure,q+p,allow_unused=True,retain_graph=True)
    expected_c=torch.autograd.grad(c_objective,c,retain_graph=True)
    expected_r=torch.autograd.grad(closure,r,retain_graph=True)
    driver.independent_component_gradients(base+auxiliary,closure,c_objective,q+p,c,r)
    for parameters,gradients in ((q+p,expected_q),(c,expected_c),(r,expected_r)):
        for parameter,gradient in zip(parameters,gradients):
            if gradient is None:assert parameter.grad is None
            else:torch.testing.assert_close(parameter.grad,gradient,atol=0,rtol=0)
    assert aux_terms['future_physics_nrmse'].isfinite()


def assert_nested_equal(left,right):
    if isinstance(left,torch.Tensor):torch.testing.assert_close(left,right,atol=0,rtol=0)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left:assert_nested_equal(left[key],right[key])
    elif isinstance(left,(list,tuple)):
        assert len(left)==len(right)
        for a,b in zip(left,right):assert_nested_equal(a,b)
    else:assert left==right


def synthetic_cli_inputs(tmp_path):
    generator=torch.Generator().manual_seed(711)
    # Synthetic binary configurations, not physical experiment data.
    train=(2*torch.randint(2,(2,4,108,108,1),generator=generator)-1).double()
    test=(2*torch.randint(2,(2,4,108,108,1),generator=generator)-1).double()
    cache=tmp_path/'synthetic.pt'
    # No coupling labels exist; explicit group IDs permit split validation.
    torch.save(dict(schema_version=1,train=dict(configurations=train,couplings=None,group_ids=torch.tensor([0,0])),
        test=dict(configurations=test,couplings=None,group_ids=torch.tensor([100,100]))),cache)
    cfg=future_config(cache);cfg=replace(cfg,data=replace(cfg.data,coupling_names=()))
    config=tmp_path/'config.json'
    config.write_text(json.dumps(cfg.to_dict()))
    pairs,norm,_=pair_files(tmp_path,data_sha=driver.file_fingerprint(cache),count=2)
    return config,pairs,norm


@pytest.mark.parametrize('control',[False,True])
def test_full_cli_auxiliary_checkpoint_exact_cpu_resume_and_control(tmp_path,monkeypatch,control):
    config,pairs,norm=synthetic_cli_inputs(tmp_path)
    monkeypatch.setattr(driver,'evaluate_run',lambda *a,**k:pytest.fail('Synthetic test requests no report'))
    common=['--config',str(config),'--q-future-pairs',str(pairs),'--skip-final-report']
    common+=['--q-future-control'] if control else ['--q-future-weight','1','--q-future-normalizer',str(norm)]
    whole,first,resumed=(tmp_path/name for name in ('whole','first','resumed'))
    driver.main(common+['--output',str(whole)])
    driver.main(common+['--output',str(first),'--epochs','1'])
    source=first/'checkpoint_last.pt';before=source.read_bytes()
    driver.main(common+['--output',str(resumed),'--resume-checkpoint',str(source)])
    assert source.read_bytes()==before
    a=torch.load(whole/'checkpoint_last.pt',map_location='cpu',weights_only=True)
    b=torch.load(resumed/'checkpoint_last.pt',map_location='cpu',weights_only=True)
    assert a['epoch']==b['epoch']==2
    for key in ('model','optimizer','history','torch_rng_state','batch_generator_state',
                'rsmi_generator_state','lr_scheduler_state'):
        assert_nested_equal(a[key],b[key])
    assert a['metadata']['labels_used_in_training'] is False
    assert ('future_physics_predictor_state' in a) is not control
    assert not any('future_physics' in key for key in a['model'])
    assert load_run(whole).model_config.block_factor==3  # Standard QCR report loader stays compatible.
    if control:
        assert a['metadata']['auxiliary_future_parameters']==0
        assert not any('future_physics' in key for key in a['history'][0])
    else:
        assert_nested_equal(a['future_physics_predictor_state'],b['future_physics_predictor_state'])
        assert a['metadata']['auxiliary_future_parameters']>0
        for split in ('train','test'):
            assert a['history'][0][f'{split}_future_physics_nrmse']>=0
            assert a['history'][0][f'{split}_weighted_future_physics']>=0
            assert a['history'][0][f'{split}_future_physics_raw_rmse']>=0


def test_full_cli_matched_control_preserves_c_and_initial_r_update(tmp_path,monkeypatch):
    config,pairs,norm=synthetic_cli_inputs(tmp_path)
    monkeypatch.setattr(driver,'evaluate_run',lambda *a,**k:pytest.fail('Synthetic test requests no report'))
    common=['--config',str(config),'--q-future-pairs',str(pairs),'--skip-final-report']
    control,future=tmp_path/'control',tmp_path/'future'
    driver.main(common+['--output',str(control),'--q-future-control'])
    driver.main(common+['--output',str(future),'--q-future-weight','1','--q-future-normalizer',str(norm)])
    first_control=torch.load(control/'training_step_1.pt',map_location='cpu',weights_only=True)
    first_future=torch.load(future/'training_step_1.pt',map_location='cpu',weights_only=True)
    for key,value in first_control['model'].items():
        if key.startswith(('rg_kernel.','theory_rg.')):
            torch.testing.assert_close(value,first_future['model'][key],atol=0,rtol=0)
    a=torch.load(control/'checkpoint_last.pt',map_location='cpu',weights_only=True)
    b=torch.load(future/'checkpoint_last.pt',map_location='cpu',weights_only=True)
    for key,value in a['model'].items():
        if key.startswith('rg_kernel.'):
            torch.testing.assert_close(value,b['model'][key],atol=0,rtol=0)
    assert torch.equal(a['torch_rng_state'],b['torch_rng_state'])
    assert torch.equal(a['batch_generator_state'],b['batch_generator_state'])
    assert torch.equal(a['rsmi_generator_state'],b['rsmi_generator_state'])
    assert any(not torch.equal(value,b['model'][key]) for key,value in a['model'].items()
               if key.startswith('theory_encoder.'))


def test_resume_requires_exact_future_mode_auxiliary_and_rng():
    contract=dict(weight=1.,mode='future_physics',pairs_sha256='a'*64)
    execution=dict(version=1,enabled=False,precision='torch.float64')
    metadata=dict(q_future_physics=contract,execution_device='cpu',precision='torch.float64',
        independent_c=True,training_driver='rsmi_separate_qcr',execution_optimizations=execution)
    good=dict(metadata=metadata,model={},optimizer={},history=[],torch_rng_state={},
        batch_generator_state={},rsmi_generator_state={},lr_scheduler_state={},future_physics_predictor_state={})
    driver.validate_resume_future(good,contract,'cpu',execution)
    for key in ('optimizer','batch_generator_state','rsmi_generator_state','future_physics_predictor_state'):
        bad=deepcopy(good);bad.pop(key)
        with pytest.raises(ValueError):driver.validate_resume_future(bad,contract,'cpu',execution)
    with pytest.raises(ValueError):driver.validate_resume_future(good,None,'cpu',execution)
    control=deepcopy(good);control['metadata']['q_future_physics']=dict(contract,weight=0.,mode='paired_control')
    with pytest.raises(ValueError):
        driver.validate_resume_future(control,control['metadata']['q_future_physics'],'cpu',execution)
    control.pop('future_physics_predictor_state')
    driver.validate_resume_future(control,control['metadata']['q_future_physics'],'cpu',execution)


def test_mi_test_cpu_is_explicit_cuda_histogram_execution_only():
    default=arguments()
    assert default.mi_test_cpu is False
    driver.validate_mi_test_cpu(default,'cpu')
    args=arguments('--mi-test-cpu')
    with pytest.raises(ValueError,match='CUDA'):driver.validate_mi_test_cpu(args,'cpu')
    driver.validate_mi_test_cpu(args,'cuda:2')  # Validation does not initialize any GPU.
    args.estimator='infonce'
    with pytest.raises(ValueError):driver.validate_mi_test_cpu(args,'cuda:2')
    args.estimator='histogram';args.environment='local_pairs'
    with pytest.raises(ValueError):driver.validate_mi_test_cpu(args,'cuda:2')


def test_test_histogram_never_materializes_on_cuda_when_offloaded():
    class HistogramSpy:
        def __init__(self):self.moves=[]
        def to(self,device):self.moves.append(str(device));return self
    train,test=HistogramSpy(),HistogramSpy()
    result=driver.place_mi_histograms({'train':train,'test':test},torch.device('cuda:2'),test_cpu=True)
    assert result=={'train':train,'test':test}
    assert train.moves==['cuda:2'] and test.moves==['cpu']
    train,test=HistogramSpy(),HistogramSpy()
    driver.place_mi_histograms({'train':train,'test':test},torch.device('cuda:2'))
    assert train.moves==test.moves==['cuda:2']


def test_offloaded_logged_mi_equals_full_cpu_estimator_without_affecting_train_gradient():
    from recurrent_rg.empirical_mi import binary_lattice_mi_statistics
    generator=torch.Generator().manual_seed(902)
    batch=(2*torch.randint(2,(3,4,9,9,1),generator=generator)-1).double()
    histogram=binary_lattice_mi_statistics(batch,block=3,flip=False)
    logits=torch.randn(512,2,generator=generator,dtype=torch.double,requires_grad=True)
    table=logits.softmax(-1)
    expected=histogram.information(table)
    expected_gradient=torch.autograd.grad(expected.mean(),logits,retain_graph=True)[0]
    before_rng=torch.get_rng_state().clone()
    table_for_test=driver.logged_mi_table(table,'test',test_cpu=True)
    with torch.no_grad():actual=histogram.information(table_for_test)
    torch.testing.assert_close(actual,expected.detach(),rtol=0,atol=0)
    assert not table_for_test.requires_grad and not actual.requires_grad
    assert torch.equal(before_rng,torch.get_rng_state())
    train_table=driver.logged_mi_table(table,'train',test_cpu=True)
    assert train_table is table
    assert driver.logged_mi_table(table,'test') is table  # Default path is identical.
    actual_gradient=torch.autograd.grad(histogram.information(train_table).mean(),logits)[0]
    torch.testing.assert_close(actual_gradient,expected_gradient,rtol=0,atol=0)
    assert logits.grad is None


def test_exact_resume_cannot_switch_test_mi_device_and_legacy_default_is_compatible():
    old_default={'metadata':{'execution_optimizations':{}}}
    cpu_test={'metadata':{'execution_optimizations':{'mi_test_device':'cpu'}}}
    driver.validate_resume_mi_placement(old_default,{})
    driver.validate_resume_mi_placement(cpu_test,{'mi_test_device':'cpu'})
    with pytest.raises(ValueError):driver.validate_resume_mi_placement(old_default,{'mi_test_device':'cpu'})
    with pytest.raises(ValueError):driver.validate_resume_mi_placement(cpu_test,{})
    contract=dict(weight=0.,mode='paired_control')
    execution=dict(mi_test_device='cpu')
    resume=dict(metadata=dict(q_future_physics=contract,execution_device='cpu',precision='torch.float64',
        independent_c=True,training_driver='rsmi_separate_qcr',execution_optimizations=execution),
        model={},optimizer={},history=[],torch_rng_state={},batch_generator_state={},
        rsmi_generator_state={},lr_scheduler_state={})
    driver.validate_resume_future(resume,contract,'cpu',execution)
    with pytest.raises(ValueError,match='execution conventions'):
        driver.validate_resume_future(resume,contract,'cpu',{})


def test_future_chunk_execution_only_on_enabled_cuda_future_and_resume_locked():
    optimized=dict(enabled=True,trusted_internal_binary_checks=True)
    expected=dict(encoding_batch_chunk=4,validated_binary_inputs=True)
    assert driver.future_encoding_execution(1.,optimized)==expected
    assert driver.future_encoding_execution(0.,optimized)=={}  # Default and paired control.
    assert driver.future_encoding_execution(1.,dict(enabled=False))=={}  # CPU/reference.
    contract=dict(weight=1.,mode='future_physics',encoding_execution=expected)
    execution=dict(optimized,future_encoding_batch_chunk=4,future_validated_binary_inputs=True)
    resume=dict(metadata=dict(q_future_physics=contract,execution_device='cuda:2',precision='torch.float64',
        independent_c=True,training_driver='rsmi_separate_qcr',execution_optimizations=execution),
        model={},optimizer={},history=[],torch_rng_state={},batch_generator_state={},
        rsmi_generator_state={},lr_scheduler_state={},future_physics_predictor_state={},
        cuda_rng_state={},cuda_device_index=2)
    driver.validate_resume_future(resume,contract,'cuda:2',execution)
    for key,value in [('future_encoding_batch_chunk',8),('future_validated_binary_inputs',False)]:
        with pytest.raises(ValueError,match='execution conventions'):
            driver.validate_resume_future(resume,contract,'cuda:2',dict(execution,**{key:value}))
    with pytest.raises(ValueError,match='contract'):
        driver.validate_resume_future(resume,dict(contract,encoding_execution={}), 'cuda:2',execution)


def test_memory_trace_only_benchmark_cuda_first_step_without_rng(monkeypatch,capsys):
    for name,value in [('memory_allocated',10),('memory_reserved',20),
                       ('max_memory_allocated',30),('max_memory_reserved',40)]:
        monkeypatch.setattr(torch.cuda,name,lambda device,value=value:value)
    rng=torch.get_rng_state().clone()
    driver.benchmark_memory_snapshot(None,'cuda:2','ignored',step=1)
    driver.benchmark_memory_snapshot(0,'cpu','ignored',step=1)
    driver.benchmark_memory_snapshot(0,'cuda:2','ignored',step=2)
    assert capsys.readouterr().out==''
    driver.benchmark_memory_snapshot(0,'cuda:2','after_setup')
    driver.benchmark_memory_snapshot(0,'cuda:2','afterbase',step=1)
    rows=[json.loads(line)['benchmark_memory'] for line in capsys.readouterr().out.splitlines()]
    assert [row['phase'] for row in rows]==['after_setup','afterbase']
    assert rows[1]==dict(phase='afterbase',step=1,allocated_bytes=10,reserved_bytes=20,
                         max_allocated_bytes=30,max_reserved_bytes=40)
    assert torch.equal(rng,torch.get_rng_state())
