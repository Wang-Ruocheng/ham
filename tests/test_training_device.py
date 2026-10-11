"""Execution-device changes must not relax the physics recipe."""
import json
from pathlib import Path
import pytest
import torch

from recurrent_rg.cli import main as cli
from experiments.rsmi_separate_qcr import main, build_parser


def test_default_cpu_and_opt_in_timing():
    args=build_parser().parse_args(['--config','unused','--output','unused'])
    assert args.device=='cpu' and args.benchmark_warmup is None
    assert not args.reference_execution and args.mi_cache_dir is None


def test_cli_forwards_cuda_without_ablation(monkeypatch):
    calls=[]
    monkeypatch.setattr('experiments.rsmi_separate_qcr.main',lambda argv:calls.append(argv))
    cli(['train','--config','c','--output','out','--device','cuda:0','--benchmark-warmup','2'])
    assert calls==[['--config','c','--output','out','--device','cuda:0','--benchmark-warmup','2']]


def test_cli_forwards_same_recipe_execution_controls(monkeypatch):
    calls=[]
    monkeypatch.setattr('experiments.rsmi_separate_qcr.main',lambda argv:calls.append(argv))
    cli(['train','--config','c','--output','out','--device','cuda:0',
         '--reference-execution'])
    cli(['train','--config','c','--output','out','--device','cuda:0',
         '--mi-cache-dir','statistics'])
    assert calls==[
        ['--config','c','--output','out','--device','cuda:0','--reference-execution'],
        ['--config','c','--output','out','--device','cuda:0','--mi-cache-dir','statistics']]


def test_bad_device_and_timing_fail_before_reading_data():
    with pytest.raises(ValueError,match='Supported execution devices'):
        main(['--config','unused','--output','unused','--device','mps'])
    with pytest.raises(ValueError,match='nonnegative'):
        main(['--config','unused','--output','unused','--benchmark-warmup','-1'])


@pytest.mark.skipif(torch.cuda.is_available(),reason='Checks unavailable-CUDA fail-closed behavior')
def test_requested_unavailable_cuda_does_not_fall_back_to_cpu():
    with pytest.raises(RuntimeError,match='unavailable'):
        main(['--config','unused','--output','unused','--device','cuda:0'])


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Requires CUDA')
def test_cuda_driver_actual_training_reports_and_resume_state(tmp_path):
    """One cache fixture covers fresh reference/optimized and exact-mode resume.

    CUDA atomic reductions can vary their floating summation order. The
    tolerance is strict float64 numerical equivalence, not bitwise weights;
    parameter devices, sampling RNG and recipe metadata must match exactly.
    """
    config=json.loads((Path(__file__).resolve().parents[1]/'configs/remote_smoke_225.json').read_text())
    gen=torch.Generator().manual_seed(817)
    train=(2*torch.randint(2,(3,8,36,36,1),generator=gen)-1).double()
    test=(2*torch.randint(2,(2,8,36,36,1),generator=gen)-1).double()
    cache=tmp_path/'cache.pt'
    torch.save((train,torch.tensor([.1,.3,.5],dtype=torch.float64),
                test,torch.tensor([.2,.4],dtype=torch.float64)),cache)
    config['data'].update(cache=str(cache))
    config['model']['hidden']=8
    config['training'].update(epochs=4,threads=1,checkpoint_every=2,log_every=2)
    cfg=tmp_path/'config.json';cfg.write_text(json.dumps(config))
    reference=tmp_path/'reference'
    whole=tmp_path/'whole';first=tmp_path/'first';resumed=tmp_path/'resumed'
    statistics=tmp_path/'mi_statistics'
    common=['--config',str(cfg),'--device','cuda:0','--benchmark-warmup','1']
    main(common+['--output',str(reference),'--reference-execution'])
    optimized=common+['--mi-cache-dir',str(statistics)]
    main(optimized+['--output',str(whole)])  # Fresh optimizer/seed; cache miss.
    main(optimized+['--output',str(first),'--epochs','2'])  # Same statistics; cache hit.
    main(optimized+['--output',str(resumed),'--resume-checkpoint',str(first/'checkpoint_last.pt')])
    original=torch.load(reference/'checkpoint_last.pt',weights_only=True)
    a=torch.load(whole/'checkpoint_last.pt',weights_only=True)
    b=torch.load(resumed/'checkpoint_last.pt',weights_only=True)
    for payload in (original,a,b):
        assert {str(v.device) for v in payload['model'].values()}=={'cuda:0'}
        assert {v.dtype for v in payload['model'].values()}=={torch.float64}
        assert payload['metadata']['execution_device']=='cuda:0'
        assert payload['metadata']['precision']=='torch.float64'
        assert payload['metadata']['independent_c'] and not payload['metadata']['labels_used_in_training']
        assert payload['metadata']['data_sha256']==a['metadata']['data_sha256']
    assert original['model'].keys()==a['model'].keys()==b['model'].keys()
    for key,value in a['model'].items():
        # CUDA index_add_ may use nondeterministic atomic reduction order.
        # Restore Adam/RNG exactly, but do not promise bitwise weight equality.
        torch.testing.assert_close(value,b['model'][key],atol=1e-10,rtol=1e-9)
        torch.testing.assert_close(value,original['model'][key],atol=1e-10,rtol=1e-9)
    assert torch.equal(a['cuda_rng_state'],b['cuda_rng_state'])
    assert torch.equal(a['cuda_rng_state'],original['cuda_rng_state'])
    assert torch.equal(a['batch_generator_state'],original['batch_generator_state'])
    assert torch.equal(a['rsmi_generator_state'],original['rsmi_generator_state'])
    assert len(a['history'])==len(original['history'])
    for optimized_row,reference_row in zip(a['history'],original['history']):
        assert optimized_row.keys()==reference_row.keys()
        assert optimized_row['epoch']==reference_row['epoch']
        for key in optimized_row:
            if key!='epoch':
                torch.testing.assert_close(torch.tensor(optimized_row[key],dtype=torch.float64),
                    torch.tensor(reference_row[key],dtype=torch.float64),atol=1e-10,rtol=1e-9)

    reference_execution=original['metadata']['execution_optimizations']
    assert reference_execution['reference'] and not reference_execution['enabled']
    assert not reference_execution['shared_binary_Q_tables']
    assert not reference_execution['trusted_internal_binary_checks']
    assert reference_execution['mi_statistics_cache']=={}
    assert not reference_execution['training_data']['resident']
    assert reference_execution['training_data']['reason']=='reference_or_cpu'
    for directory,expected_hit in ((whole,False),(first,True),(resumed,True)):
        manifest=json.loads((directory/'manifest.json').read_text())
        execution=manifest['execution_optimizations']
        assert execution['enabled'] and not execution['reference']
        assert execution['shared_binary_Q_tables'] and execution['trusted_internal_binary_checks']
        assert execution['precision']=='torch.float64'
        assert execution['training_data']['resident']
        assert execution['training_data']['reason']=='fits_safe_budget'
        assert execution['training_data']['source_dtype']=='torch.float64'
        assert execution['training_data']['bytes']==train.numel()*train.element_size()
        assert execution['mi_statistics_cache'].keys()=={'train','test'}
        for record in execution['mi_statistics_cache'].values():
            assert record['hit'] is expected_hit
            assert not record['publication_race']
            assert Path(record['path']).parent==statistics
            assert Path(record['path']).is_file()
            assert (record['load_seconds']==0.) is (not expected_hit)
            assert (record['build_seconds']==0.) is expected_hit
    summary=json.loads((whole/'summary.json').read_text())
    assert summary['actual_epochs']==4
    assert summary['timing']['measured_updates']==3
    assert all(value>0 for value in summary['timing']['step_seconds_all'])
    assert (whole/'report'/'R_vs_g.png').is_file()
