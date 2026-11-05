import json
from pathlib import Path

import pytest

from scripts import run_ising_finite_perturbation as pipe


def test_fingerprint_refuses_large_physical_data(tmp_path):
    path=tmp_path/'old_physical.pt'
    with path.open('wb') as stream:
        stream.truncate(33*2**20)
    with pytest.raises(ValueError,match='大型物理'):
        pipe.fingerprint(path)


def test_fingerprint_refuses_symlinks(tmp_path):
    path=tmp_path/'data.json'; path.write_text('{}')
    link=tmp_path/'alias.json'; link.symlink_to(path)
    with pytest.raises(ValueError,match='普通'):
        pipe.fingerprint(link)


def test_atomic_status_preserves_fields(tmp_path):
    path=tmp_path/'job.json'
    pipe.save(path,dict(status='running',source_unchanged=True))
    assert pipe.read(path)==dict(status='running',source_unchanged=True)
    assert not path.with_name('job.json.writing').exists()


def test_resources_never_admits_gpu_zero_or_one():
    code=Path(pipe.__file__).read_text()
    assert "index not in (2, 3)" in code
    assert "choices=['cpu','cuda:2','cuda:3']" in code
    assert 'terminate(' not in code and 'kill(' not in code


def test_no_mutation_of_previous_jobs():
    code=Path(pipe.__file__).read_text()
    assert 'update_sequence' not in code
    assert "save(previous" not in code
    assert 'source_checkpoint_sha256' in code
    assert 'scientific_stop' in code


def fake_sources(tmp_path):
    source=tmp_path/'phase2'; previous=tmp_path/'phase3'
    run=source/'dim3_control/run'; run.mkdir(parents=True); previous.mkdir()
    summary=dict(actual_epochs=12000,config=dict(schema_version=5,
        model=dict(theory_dim=3,hidden=96,width=3,block_factor=2,
            kernel_layout='nonoverlapping',fixed_kernel=False,kernel_initialization='random'),
        training=dict(max_rg_steps=2)))
    for name,obj in [('summary.json',summary),('manifest.json',dict(labels_used_in_training=False))]:
        pipe.save(run/name,obj)
    (run/'training_step_12000.pt').write_bytes(b'new model, not physical data')
    sha=pipe.fingerprint(run/'training_step_12000.pt')
    pipe.save(source/'job.json',dict(status='complete',workflow_complete=True,
        training=dict(dim3_control=dict(actual_epochs=12000,labels_used_in_training=False,
                                        checkpoint_sha256=sha))))
    pipe.save(previous/'job.json',dict(status='complete',workflow_complete=True,smoke_only=False))
    return source,previous,run,sha


def test_formal_source_validates_actual_weights(tmp_path):
    source,previous,run,sha=fake_sources(tmp_path)
    assert pipe.validate_source(source,previous)==(run,sha)
    (run/'training_step_12000.pt').write_bytes(b'changed model')
    with pytest.raises(ValueError,match='指纹'):
        pipe.validate_source(source,previous)


@pytest.mark.parametrize('failure',['supervision','geometry','hidden','short','smoke_previous'])
def test_invalid_source_cannot_start(tmp_path,failure):
    source,previous,run,_=fake_sources(tmp_path)
    if failure=='supervision':
        path=run/'manifest.json'; d=pipe.read(path); d['labels_used_in_training']=True
    elif failure=='smoke_previous':
        path=previous/'job.json';d=pipe.read(path);d['smoke_only']=True
    else:
        path=run/'summary.json';d=pipe.read(path)
        if failure=='geometry': d['config']['model']['block_factor']=3
        elif failure=='hidden': d['config']['model']['hidden']=64
        else: d['actual_epochs']=11999
    pipe.save(path,d)
    with pytest.raises(ValueError):pipe.validate_source(source,previous)


def test_budget_numbers_and_fixed_training():
    source=Path(pipe.__file__).read_text()
    assert '26181632' in source and '818176' in source and '6701294592' in source
    assert "'--updates','6000'" in source
    assert "args.seed" in source
    assert pipe.ARMS==('A','B','C','D','E')


def test_execution_budget_includes_smoke_and_collector_card_time():
    projected, consumed=pipe.execution_budget_totals(
        dict(projected_collection_gpu_seconds_with_margin=300.,elapsed_gpu_card_seconds=3.),
        dict(projected_training_gpu_seconds_with_margin=600.,elapsed_gpu_budget_seconds=7.),
        dict(gpu_card_seconds=11.))
    assert projected==900. and consumed==21.


def test_remaining_stage_budget_does_not_launch_training_without_room():
    plan=dict(execution_preflight=dict(projected_collection_gpu_seconds_with_margin=300.,
        projected_training_gpu_seconds_with_margin=600.))
    assert pipe.remaining_gpu_projection(plan,'collection')==900.
    assert pipe.remaining_gpu_projection(plan,'training')==600.
    assert pipe.remaining_gpu_projection(plan,'sampling')==0.
    assert 3100+pipe.remaining_gpu_projection(plan,'training')>3600


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-1.])
def test_execution_budget_refuses_invalid_receipts(bad):
    with pytest.raises(ValueError,match='有限非负'):
        pipe.execution_budget_totals(dict(projected_collection_gpu_seconds_with_margin=bad),
            dict(projected_training_gpu_seconds_with_margin=1.),dict(gpu_card_seconds=0.))


def test_previous_execution_gpu_cost_is_explicit_and_not_silently_zero():
    assert pipe.consumed_gpu_seconds(dict(gpu_card_seconds=0.))==0.
    assert pipe.consumed_gpu_seconds(dict(elapsed_gpu_card_seconds=2.5))==2.5
    with pytest.raises(ValueError,match='缺少实际'):
        pipe.consumed_gpu_seconds({})
