from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from experiments import rsmi_separate_qcr as driver
from experiments.rsmi_separate_qcr import microscopic_coupling_loss,build_parser
from recurrent_rg.checkpoints import build_model
from recurrent_rg.config import ModelConfig, RunConfig
from recurrent_rg.data.statistics import plaquette_distribution
from recurrent_rg.execution import configure_cuda_execution
from recurrent_rg.learning_rate import LearningRatePolicy
from recurrent_rg.losses import qcr_objective

def test_supervision_shape_and_detached_labels():
    encoder=torch.nn.Linear(2,1).double()
    x=torch.randn(4,2,dtype=torch.double)
    labels=torch.randn(4,1,dtype=torch.double,requires_grad=True)
    loss=microscopic_coupling_loss(encoder,x,labels)
    torch.testing.assert_close(loss,((encoder(x)-labels.detach())**2).mean())
    loss.backward()
    assert labels.grad is None
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.parameters())

def test_unsupervised_default_unchanged():
    args=build_parser().parse_args(['--config','dummy.json','--output','dummy'])
    assert args.q_coupling_weight==0
    assert args.skip_final_report is False
    driver.validate_cuda_options(args)
    contract=dict(note='No critic or coupling labels.',gradient_QC='Q: representation losses only; C: original')
    before=deepcopy(contract)
    driver.add_supervision_contract(contract,args.q_coupling_weight)
    assert contract==before


def test_supervision_never_silently_selects_a_column_for_scalar_q():
    encoder=torch.nn.Linear(2,1).double()
    with pytest.raises(ValueError,match='shape mismatch'):
        microscopic_coupling_loss(encoder,torch.ones(3,2,dtype=torch.double),
                                  torch.ones(3,2,dtype=torch.double))


@pytest.mark.parametrize('device,trusted', [
    pytest.param('cpu',False,id='cpu'),
    pytest.param('cuda:0',False,id='cuda',
                 marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='Requires CUDA')),
    pytest.param('cuda:0',True,id='cuda-trusted',
                 marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='Requires CUDA'))])
def test_supervision_adds_only_q_gradients_under_independent_routing(device,trusted):
    torch.set_num_threads(1);torch.manual_seed(11)
    model=build_model(ModelConfig(dimension=2,hidden=5,block_factor=2,
        affine_reconstruction=True,kernel_sampling='bernoulli'),2).to(device)
    model.detach_kernel_representation=True
    batch=(2*torch.randint(2,(3,8,16,16,1))-1).double().to(device)
    labels=torch.tensor([.2,.3,.4],dtype=torch.double,device=device,requires_grad=True)
    if trusted:
        execution=configure_cuda_execution(model,device,independent_c=True)
        assert execution['shared_binary_Q_tables'] and execution['trusted_internal_binary_checks']
    base,terms=qcr_objective(model,batch,pattern_fn=plaquette_distribution)
    closure=5*(terms['commutator']+terms['rollout'])
    supervised=microscopic_coupling_loss(model.theory_encoder,batch,labels)
    q=list(model.theory_encoder.parameters());c=list(model.rg_kernel.parameters())
    r=list(model.theory_rg.parameters())
    c_objective=sum(parameter.square().sum() for parameter in c)
    expected_q=torch.autograd.grad(base-closure+10*supervised,q,retain_graph=True,allow_unused=True)
    expected_c=torch.autograd.grad(c_objective,c,retain_graph=True)
    expected_r=torch.autograd.grad(closure,r,retain_graph=True)
    supervision_q=torch.autograd.grad(supervised,q,retain_graph=True,allow_unused=True)
    assert any(value is not None and value.abs().sum()>0 for value in supervision_q)
    driver.independent_component_gradients(base+10*supervised,closure,c_objective,q,c,r)
    for parameters,gradients in ((q,expected_q),(c,expected_c),(r,expected_r)):
        for parameter,gradient in zip(parameters,gradients):
            if gradient is None: assert parameter.grad is None
            else: torch.testing.assert_close(parameter.grad,gradient)
    assert labels.grad is None


def test_cuda_allows_supervised_resume_subject_to_checkpoint_validation():
    args=build_parser().parse_args(['--config','dummy.json','--output','dummy','--q-coupling-weight','10'])
    driver.validate_cuda_options(args)
    args.resume_checkpoint=Path('checkpoint_last.pt')
    driver.validate_cuda_options(args)
    args.q_coupling_weight=0
    driver.validate_cuda_options(args)


@pytest.mark.parametrize('option,value', [
    ('source',Path('old')),('allow_nondefault_recipe',True),('mi_initialize',True),
    ('freeze_c_checkpoint',Path('old.pt')),('affine_pca_init',True),
    ('q_only_weight_sweep',True),('q_only_candidates',True),('q_balanced_reconstruction',True),
    ('q_last_layer_audit',True),('q_head_structure_audit',True),('q_feature_bottleneck',True),
    ('q_original_plateau',True),('q_initialization_pilot',True),('q_unfolding',True),
    ('r_only_continue',True),('r_only_refit',True),('r_only_plateau',True),
    ('joint_train_plateau',True),('frozen_cache',Path('cache')),('estimator','infonce'),
    ('independent_c',False),('q_coupling_weight',-1.),('q_coupling_weight',float('nan'))])
def test_cuda_supervision_does_not_unlock_other_ablations(option,value):
    args=build_parser().parse_args(['--config','dummy.json','--output','dummy','--q-coupling-weight','10'])
    setattr(args,option,value)
    with pytest.raises(ValueError):
        driver.validate_cuda_options(args)


@pytest.mark.parametrize('note', ['No critic or coupling labels.',
    'Compressed empirical MI; no critic or coupling labels. Large finite-sample bias is possible.'])
def test_supervised_metadata_explicitly_describes_labels_and_q_gradient(note):
    contract=dict(note=note,gradient_QC='Q: representation losses only; C: unchanged')
    driver.add_supervision_contract(contract,10.)
    assert 'no critic or coupling labels' not in contract['note'].lower()
    assert 'TRAIN coupling labels' in contract['note'] and 'TEST labels are diagnostic only' in contract['note']
    assert 'microscopic coupling MSE' in contract['gradient_QC']
    assert 'C: unchanged' in contract['gradient_QC']
    assert contract['q_supervision']['weight']==10.
    assert contract['q_supervision']['fitting_split']=='TRAIN only'


@pytest.mark.parametrize('skip',[False,True])
def test_final_report_skipping_is_explicit_and_preserves_checkpoint(tmp_path,monkeypatch,skip):
    checkpoint=tmp_path/'checkpoint_last.pt';checkpoint.write_bytes(b'final model')
    (tmp_path/'summary.json').write_text(json.dumps(dict(timing={})))
    calls=[]
    monkeypatch.setattr(driver,'evaluate_run',lambda *args,**kwargs:calls.append((args,kwargs)))
    driver.final_report(tmp_path,skip=skip)
    summary=json.loads((tmp_path/'summary.json').read_text())
    assert len(calls)==(0 if skip else 1)
    assert summary['final_report_skipped'] is skip
    assert summary['timing']['report_seconds']>=0
    assert checkpoint.read_bytes()==b'final model'


def test_supervised_cpu_smoke_saves_final_model_summary_and_explicit_report_skip(tmp_path,monkeypatch):
    """One synthetic update exercises metadata and the real TRAIN-only label path."""
    generator=torch.Generator().manual_seed(19)
    train=(2*torch.randint(2,(2,4,8,8,1),generator=generator)-1).double()
    test=(2*torch.randint(2,(2,4,8,8,1),generator=generator)-1).double()
    cache=tmp_path/'data.pt'
    torch.save((train,torch.tensor([.2,.3],dtype=torch.double),
                test,torch.tensor([.4,.5],dtype=torch.double)),cache)
    cfg=RunConfig.from_dict(dict(schema_version=5,
        data=dict(cache=str(cache),physical_model='ising2d',coupling_names=['K']),
        model=dict(dimension=2,theory_dim=1,hidden=5,block_factor=2,affine_reconstruction=True),
        training=dict(max_rg_steps=1,epochs=1,system_batch=2,sample_batch=4,
                      checkpoint_every=1,log_every=1,gradient_mode='qc_r_separate',threads=1),
        evaluation=dict(max_rg_steps=1,samples=4,systems=2)))
    config=tmp_path/'config.json';config.write_text(json.dumps(cfg.to_dict()))
    output=tmp_path/'run'
    monkeypatch.setattr(driver,'evaluate_run',lambda *a,**k:pytest.fail('Explicit report skip must be honored'))
    driver.main(['--config',str(config),'--output',str(output),
                 '--q-coupling-weight','10','--skip-final-report'])
    manifest=json.loads((output/'manifest.json').read_text())
    summary=json.loads((output/'summary.json').read_text())
    assert manifest['labels_used_in_training'] is True
    assert manifest['skip_final_report'] is summary['contract']['skip_final_report'] is True
    assert manifest['q_supervision']['coupling_names']==['K']
    assert summary['contract']['q_supervision']==manifest['q_supervision']
    assert summary['actual_epochs']==1 and summary['final_report_skipped'] is True
    assert summary['history'][0]['train_coupling_supervision']>=0
    assert summary['history'][0]['test_coupling_supervision']>=0
    assert (output/'checkpoint_last.pt').is_file() and (output/'qcr_model.pt').is_file()
    assert not (output/'report').exists()


def resume_contract_fixture():
    cfg=RunConfig.from_dict(dict(schema_version=5,
        data=dict(cache='unused.pt',physical_model='ising2d',coupling_names=['K']),
        model=dict(dimension=2,theory_dim=1,block_factor=2),
        training=dict(epochs=12000,gradient_mode='qc_r_separate')))
    args=build_parser().parse_args(['--config','unused.json','--output','unused',
                                    '--q-coupling-weight','10','--resume-checkpoint','old.pt'])
    execution=dict(version=1,enabled=False,reference=False,shared_binary_Q_tables=False,
        trusted_internal_binary_checks=False,precision='torch.float64',selected_empirical_mi=False)
    metadata=dict(note='',gradient_QC='',data_sha256='a'*64,labels_used_in_training=True,
        independent_c=True,training_driver='rsmi_separate_qcr',precision='torch.float64',
        execution_device='cpu',execution_optimizations=deepcopy(execution))
    driver.add_supervision_contract(metadata,10.)
    metadata['q_supervision']['coupling_names']=['K']
    resume=dict(metadata=metadata,model={},optimizer={},history=[],torch_rng_state={},
                batch_generator_state={},rsmi_generator_state={},lr_scheduler_state={})
    return resume,args,cfg,dict(device='cpu',data_sha256='a'*64,execution=execution)


def test_supervised_resume_contract_preserves_labels_and_allows_report_setting_change():
    resume,args,cfg,checks=resume_contract_fixture()
    resume['metadata']['skip_final_report']=True
    args.skip_final_report=False
    driver.validate_resume_supervision(resume,args,cfg,**checks)
    args.skip_final_report=True
    driver.validate_resume_supervision(resume,args,cfg,**checks)


@pytest.mark.parametrize('change', ['weight','scale','target','names','labels','missing_supervision',
    'remove_supervision','device','data','independent','execution','missing_adam','missing_rng','missing_lr'])
def test_exact_supervised_resume_rejects_objective_or_state_changes(change):
    resume,args,cfg,checks=resume_contract_fixture()
    old=resume['metadata']
    if change=='weight': args.q_coupling_weight=5.
    if change=='scale': old['q_supervision']['scale']=1
    if change=='target': old['q_supervision']['target']='coarse labels'
    if change=='names': old['q_supervision']['coupling_names']=['B']
    if change=='labels': old['labels_used_in_training']=False
    if change=='missing_supervision': old.pop('q_supervision')
    if change=='remove_supervision': args.q_coupling_weight=0.
    if change=='device': old['execution_device']='cuda:0'
    if change=='data': checks['data_sha256']='b'*64
    if change=='independent': old['independent_c']=False
    if change=='execution': old['execution_optimizations']['shared_binary_Q_tables']=True
    if change=='missing_adam': resume.pop('optimizer')
    if change=='missing_rng': resume.pop('batch_generator_state')
    if change=='missing_lr': resume.pop('lr_scheduler_state')
    with pytest.raises(ValueError):
        driver.validate_resume_supervision(resume,args,cfg,**checks)


def test_default_unsupervised_resume_remains_compatible_but_cannot_add_supervision():
    resume,args,cfg,checks=resume_contract_fixture()
    resume['metadata'].pop('q_supervision')
    resume['metadata']['labels_used_in_training']=False
    args.q_coupling_weight=0.
    # Historical unsupervised runs need not contain the new supervised audit fields.
    for field in ('execution_device','precision','execution_optimizations'):
        resume['metadata'].pop(field)
    resume.pop('lr_scheduler_state')
    driver.validate_resume_supervision(resume,args,cfg,**checks)
    args.q_coupling_weight=10.
    with pytest.raises(ValueError,match='supervision'):
        driver.validate_resume_supervision(resume,args,cfg,**checks)


def small_supervised_config(tmp_path):
    generator=torch.Generator().manual_seed(37)
    train=(2*torch.randint(2,(3,4,16,16,1),generator=generator)-1).double()
    test=(2*torch.randint(2,(2,4,16,16,1),generator=generator)-1).double()
    cache=tmp_path/'data.pt'
    torch.save((train,torch.tensor([.2,.3,.4],dtype=torch.double),
                test,torch.tensor([.25,.35],dtype=torch.double)),cache)
    cfg=RunConfig.from_dict(dict(schema_version=5,
        data=dict(cache=str(cache),physical_model='ising2d',coupling_names=['K']),
        model=dict(dimension=2,theory_dim=1,hidden=5,block_factor=2,affine_reconstruction=True),
        training=dict(max_rg_steps=2,epochs=4,system_batch=2,sample_batch=4,
            checkpoint_every=1,log_every=1,gradient_mode='qc_r_separate',threads=1,
            lr_schedule='multistep',lr_milestones=[2,3],lr_gamma=.3),
        evaluation=dict(max_rg_steps=2,samples=4,systems=2)))
    config=tmp_path/'config.json';config.write_text(json.dumps(cfg.to_dict()))
    return config,LearningRatePolicy(cfg.training)


def assert_nested_close(left,right,*,atol=0,rtol=0):
    if isinstance(left,torch.Tensor):
        torch.testing.assert_close(left,right,atol=atol,rtol=rtol)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left: assert_nested_close(left[key],right[key],atol=atol,rtol=rtol)
    elif isinstance(left,(list,tuple)):
        assert len(left)==len(right)
        for a,b in zip(left,right): assert_nested_close(a,b,atol=atol,rtol=rtol)
    elif isinstance(left,float):
        assert left==pytest.approx(right,abs=atol,rel=rtol)
    else:
        assert left==right


@pytest.mark.parametrize('device', [pytest.param('cpu',id='cpu'),
    pytest.param('cuda:0',id='cuda',
                 marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='Requires CUDA'))])
def test_supervised_exact_resume_matches_uninterrupted_all_qcr_adam_rng_and_lr(tmp_path,monkeypatch,device):
    config,policy=small_supervised_config(tmp_path)
    report_calls=[]
    monkeypatch.setattr(driver,'evaluate_run',lambda *a,**k:report_calls.append(a[0]))
    common=['--config',str(config),'--device',device,'--q-coupling-weight','10']
    if device!='cpu': common+=['--mi-cache-dir',str(tmp_path/'mi_cache')]
    whole,first,resumed=(tmp_path/name for name in ('whole','first','resumed'))
    driver.main(common+['--output',str(whole),'--skip-final-report'])
    # A capped pilot ends before its requested total; the continuation omits the old cap.
    driver.main(common+['--output',str(first),'--epochs','3',
                        '--max-training-seconds','0.000000001','--skip-final-report'])
    source=first/'checkpoint_last.pt';before=source.read_bytes()
    pilot=json.loads((first/'summary.json').read_text())
    assert pilot['actual_epochs']==1 and pilot['requested_epochs']==3
    assert pilot['stopped_by_training_limit'] is True
    driver.main(common+['--output',str(resumed),'--resume-checkpoint',str(source)])
    assert source.read_bytes()==before and report_calls==[resumed]
    a=torch.load(whole/'checkpoint_last.pt',map_location='cpu',weights_only=True)
    b=torch.load(resumed/'checkpoint_last.pt',map_location='cpu',weights_only=True)
    tolerance=dict(atol=1e-10,rtol=1e-9) if device!='cpu' else dict(atol=0,rtol=0)
    assert a['epoch']==b['epoch']==4
    assert a['lr_scheduler_state']==b['lr_scheduler_state']==policy.checkpoint_state(4)
    for key in ('model','optimizer','history'):
        assert_nested_close(a[key],b[key],**tolerance)
    for key in ('torch_rng_state','batch_generator_state','rsmi_generator_state'):
        assert torch.equal(a[key],b[key])
    if device!='cpu': assert torch.equal(a['cuda_rng_state'],b['cuda_rng_state'])
    assert a['metadata']['q_supervision']==b['metadata']['q_supervision']
    assert b['metadata']['resumed_from']==str(source)
    assert b['metadata']['independent_c'] is b['metadata']['labels_used_in_training'] is True
    assert [row['learning_rate'] for row in b['history']]==[policy.rate_for_update(u) for u in range(1,5)]
    summary=json.loads((resumed/'summary.json').read_text())
    assert summary['actual_epochs']==summary['requested_epochs']==4
    assert summary['training_limit_seconds'] is None and summary['stopped_by_training_limit'] is False
    assert summary['final_report_skipped'] is False


@pytest.mark.parametrize('mode',['changed_weight','remove_labels','add_labels','changed_data'])
def test_resume_rejects_supervision_changes_before_loading_model_or_adam(tmp_path,monkeypatch,mode):
    config,_=small_supervised_config(tmp_path)
    first=tmp_path/'first'
    driver.main(['--config',str(config),'--output',str(first),'--epochs','1',
                 '--q-coupling-weight','10','--skip-final-report'])
    checkpoint=torch.load(first/'checkpoint_last.pt',map_location='cpu',weights_only=True)
    weight='10'
    if mode=='changed_weight': weight='5'
    if mode=='remove_labels': weight='0'
    if mode=='add_labels':
        checkpoint['metadata']['labels_used_in_training']=False
        checkpoint['metadata'].pop('q_supervision')
    if mode=='changed_data': checkpoint['metadata']['data_sha256']='b'*64
    source=tmp_path/'source.pt';torch.save(checkpoint,source)
    monkeypatch.setattr(driver,'load_model_state',lambda *a,**k:pytest.fail('Validate resume before loading model/Adam'))
    if mode=='changed_data':
        monkeypatch.setattr(driver,'binary_lattice_mi_statistics',
            lambda *a,**k:pytest.fail('Reject changed resume data before loading/building MI statistics'))
    with pytest.raises(ValueError,match='supervision|data changed'):
        driver.main(['--config',str(config),'--output',str(tmp_path/'bad'),
            '--resume-checkpoint',str(source),'--q-coupling-weight',weight,'--skip-final-report'])
    assert not (tmp_path/'bad').exists()
