"""CPU/synthetic checks only: range-only settings and safe orchestration."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import phi4_ru_cnn_local_window as spec
from experiments import phi4_ru_cnn_qcr as trainer
from scripts import run_phi4_ru_cnn_local_window as runner


def test_only_range_and_provenance_change():
    from experiments.phi4_ru_cnn_nonlinear import preset as control
    old=control();new=spec.preset('a'*64)
    for key in ('recipe','data_root','data_manifest_sha256'):
        old.pop(key);new.pop(key)
    old.pop('physical_options');new.pop('physical_options')
    assert old==new
    from recurrent_rg.data.phi4_ru_pipeline import Phi4RUOptions
    from dataclasses import asdict
    a,b=asdict(Phi4RUOptions()),spec.physical_options()
    for key in ('r_values','u_values'):
        a.pop(key);b.pop(key)
    assert json.loads(json.dumps(a))==b


def test_new_preset_is_strict_historical_preset_still_strict(tmp_path):
    path=tmp_path/'config.json'
    config=spec.preset('b'*64)
    path.write_text(json.dumps(config))
    assert trainer.validate_config(path)==config
    for key,value in [('steps',3000),('data_root','/tmp/wrong'),('recipe','bad'),('seed',92)]:
        changed=copy.deepcopy(config);changed[key]=value
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError):trainer.validate_config(path)
    with pytest.raises(ValueError):spec.preset('not-a-sha')
    assert trainer.validate_config('configs/phi4_ru_b3_cnn_g2_Rlr003_seed91.json')==trainer.preset()


def test_plan_same_work_coverage_and_train_only_pilot(tmp_path):
    code=Path(__file__).resolve().parents[1]
    plan=runner.build_plan(tmp_path/'absent',code)
    assert not (tmp_path/'absent').exists()
    p=plan['physical']
    assert p['conditions']==45 and p['configurations']==23040
    assert p['total_hmc_trajectories']==368640
    assert p['total_force_site_evaluations']==6210846720
    assert plan['work_including_pilots']['force_site_evaluations']==6243194880
    assert min(spec.R_VALUES)<-1.97-.12<-1.97+.12<max(spec.R_VALUES)
    assert min(spec.U_VALUES)<2.13-.13<2.13+.13<max(spec.U_VALUES)
    assert set(plan['pilot']['cells'])<=set(p['train_cell_indices'])
    assert len(p['train_cell_indices'])==34 and len(p['test_cell_indices'])==11
    assert p['options']['step_size']==.08 and p['options']['leapfrog_steps']==12
    assert plan['training_template']['steps']==6000
    assert plan['training_template']['learning_rates']=={'Q':.001,'C':.001,'R':.003}
    assert not plan['scientific_policy']['labels_used_in_training']


def test_standard_entrypoint_routes_new_preset_without_legacy_flags(tmp_path,monkeypatch):
    from recurrent_rg.cli import main
    path=tmp_path/'config.json';path.write_text(json.dumps(spec.preset('c'*64)))
    called=[]
    monkeypatch.setattr(trainer,'run',lambda *args:called.append(args))
    main(['train','--config',str(path),'--output',str(tmp_path/'run')])
    assert len(called)==1
    for flag in ('--resume','--initialize-from'):
        with pytest.raises(SystemExit):main(['train','--config',str(path),'--output','x',flag,'x'])


def test_execute_completes_sequence_and_seals_new_manifest(tmp_path,monkeypatch):
    root=tmp_path/'fresh'
    monkeypatch.setattr(spec,'ROOT',str(root))
    monkeypatch.setattr(spec,'SNAPSHOT',str(Path(runner.__file__).resolve().parents[1]))
    monkeypatch.setattr(runner,'check_host',lambda:None)
    calls=[]
    def child(root,job,name,command,wall_budget,gpu=False):
        calls.append(name)
        job['stages'].append(dict(name=name,status='complete',returncode=0))
        if name=='sampler_preflight':
            runner.save(root/'sampler_preflight.json',dict(budget_passed=True))
        elif name=='physical_sampling':
            (root/'data').mkdir()
            runner.save(root/'data/manifest.json',dict(options=spec.physical_options()))
            np.savez(root/'data/data.npz',private=np.ones(1))
        else:
            assert gpu
            config=trainer.validate_config(root/'training_config.json')
            assert config['data_manifest_sha256']==runner.sha(root/'data/manifest.json')
            (root/'run').mkdir()
            runner.save(root/'run/summary.json',dict(actual_steps=6000))
            runner.save(root/'run/status.json',dict(status='complete',workflow_complete=True,returncode=0,actual_steps=6000,
                public_files={'summary.json':runner.sha(root/'run/summary.json')}))
    monkeypatch.setattr(runner,'run_child',child)
    monkeypatch.setattr(runner.previous,'sampling_validity',lambda *a:dict(status='complete',warning_count=7))
    assert runner.execute(root)==0
    job=runner.read(root/'job.json')
    assert calls==['sampler_preflight','physical_sampling','training_and_report']
    assert job['workflow_complete'] and job['data_warning_count']==7
    assert 'data/data.npz' not in job['public_files']
    with pytest.raises(FileExistsError):runner.execute(root)


def test_failed_budget_pilot_does_not_start_production(tmp_path,monkeypatch):
    root=tmp_path/'hold'
    monkeypatch.setattr(spec,'ROOT',str(root))
    monkeypatch.setattr(spec,'SNAPSHOT',str(Path(runner.__file__).resolve().parents[1]))
    monkeypatch.setattr(runner,'check_host',lambda:None)
    calls=[]
    def child(root,job,name,*a,**kw):
        calls.append(name)
        runner.save(root/'sampler_preflight.json',dict(budget_passed=False))
    monkeypatch.setattr(runner,'run_child',child)
    assert runner.execute(root)==3
    assert calls==['sampler_preflight']
    assert not (root/'data').exists() and not (root/'run').exists()
    assert runner.read(root/'job.json')['status']=='budget_hold'


def test_resource_wait_requires_two_clean_checks_and_never_selects_other_gpu(tmp_path,monkeypatch):
    rows=iter([dict(gpu_ready=False),dict(gpu_ready=True),dict(gpu_ready=True)])
    monkeypatch.setattr(runner,'snapshot',lambda:next(rows))
    monkeypatch.setattr(runner,'write_state',lambda *a,**kw:None)
    waits=[]
    monkeypatch.setattr(runner.time,'sleep',lambda t:waits.append(t))
    runner.admission(tmp_path,{},'training_and_report',True)
    assert waits==[30,10]
    assert runner.GPU_UUID==spec.GPU_UUID and spec.GPU_INDEX==2


def test_no_reference_coordinates_enter_training_config(tmp_path):
    plan=runner.build_plan(tmp_path/'absent',tmp_path)
    config=spec.preset('b'*64)
    assert set(config)==set(__import__('experiments.phi4_ru_cnn_nonlinear',fromlist=['preset']).preset())
    assert plan['scientific_policy']['reference_is_not_training_or_root_target']
    assert spec.R_VALUES[4]==spec.REFERENCE_RU[0] and spec.U_VALUES[2]==spec.REFERENCE_RU[1]
    assert config['loss']['closure_scale_weights'] is None
