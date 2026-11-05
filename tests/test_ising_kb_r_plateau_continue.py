import copy
import json
from pathlib import Path
import numpy as np
import pytest
from experiments import ising_kb_r_plateau_continue as run

PUBLIC=Path('results/ising2d_KB_local_motif_frozen_R_seed91_vm225_20261104_monitor/final')


def test_pinned_resume_dispatch(tmp_path,monkeypatch):
    config='configs/ising2d_KB_local_motif_R_plateau_continue_seed91.json'
    assert run.validate_config(config)==run.preset()
    assert run.SOURCE_ADAM_STEPS==run.SOURCE_R_UPDATES+1250
    assert run.SOURCE_PINS['latest.pt']=='d25a15ac2eb8dec2cdc389119b0cef6da2d036405f37c89873c7e4f704f339d1'
    from recurrent_rg.cli import main
    called=[];monkeypatch.setattr(run,'run',lambda *a:called.append(a))
    main(['train','--config',config,'--output','unused']);assert len(called)==1
    with pytest.raises(SystemExit):main(['train','--config',config,'--output','unused','--resume','wrong'])
    bad=run.preset();bad['learning_rate']=.001;(tmp_path/'bad.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError):run.validate_config(tmp_path/'bad.json')


def test_preserved_history_and_plateau_guard(tmp_path):
    rows=json.loads((PUBLIC/'history.json').read_text());last=rows[-1]['train']['total']
    inherited=run.inherited_history(PUBLIC/'history.json',last)
    assert inherited==rows and not run.plateau_reached(inherited,window=5000,minimum=10000)
    with pytest.raises(AssertionError):run.inherited_history(PUBLIC/'history.json',last*.9)
    # Carry source history and require three genuinely flat complete 5k windows.
    for step in range(61000,68000,1000):
        inherited.append(dict(extra_R_updates=step,train={'total':last},best_train_closure=last))
    assert run.plateau_reached(inherited,window=5000,minimum=10000)
    inherited[-1]['train']['total']=last*1.06
    assert not run.plateau_reached(inherited,window=5000,minimum=10000)


def test_root_matching_handles_creation_and_class_change():
    def root(x,y,c):return dict(g=[x,y],classification=c)
    central=root(.079,-.077,'repelling');other=root(1,1,'saddle')
    snapshots=[dict(extra_R_updates=60000,roots=[central,other]),
               dict(extra_R_updates=61000,roots=[root(.080,-.078,'repelling'),root(1.001,1,'attracting'),root(-1,-1,'saddle')])]
    result=run.root_stability(snapshots,np.array([[-2,2],[-2,2]]),[central,other])
    a,b,c=result['final_root_tracks']
    assert a['present_all'] and a['same_classification']
    assert b['present_all'] and not b['same_classification']
    assert not c['present_all'] and c['matched_indices']==[None,2]
    assert [v['point']['classification'] for v in result['central_track']]==['repelling','repelling']


def test_late_report_array_only(tmp_path):
    from scripts import report_ising_kb_r_plateau_continue as report
    from scripts.report_phi4_hopping_qcr import closure_metrics
    a=dict(np.load(PUBLIC/'evaluation_arrays.npz',allow_pickle=False));roots=json.loads((PUBLIC/'roots.json').read_text())
    for k in ('one_step_g','rollout_g','all_start_g','flow_g','flow_R','flow_displacement'):a['source_'+k]=a[k].copy()
    np.savez(tmp_path/'evaluation_arrays.npz',**a)
    s=json.loads((PUBLIC/'summary.json').read_text());s.update(new_R_updates=6000,actual_extra_R_updates=66000,train_plateau_reached=True,source_roots=roots['roots'])
    snapshots=[dict(extra_R_updates=n,roots=copy.deepcopy(roots['roots'])) for n in range(61000,67000,1000)]
    stability=run.root_stability(snapshots,a['flow_bounds'],roots['roots'])
    for name,obj in [('manifest',dict(labels_used_in_training=False,test_used_in_training=False)),('summary',s),('roots',roots),('root_stability',stability),('history',json.loads((PUBLIC/'history.json').read_text()))]:
        (tmp_path/(name+'.json')).write_text(json.dumps(obj))
    result=report.render(tmp_path)
    assert result['new_R_updates']==6000 and result['train_plateau_reached']
    assert len(list((tmp_path/'report').rglob('*.png')))==9


def test_central_tracking_does_not_jump_to_transient_neighbor():
    st=json.loads((PUBLIC.parent.parent/'ising2d_KB_local_motif_R_plateau_continue_seed91_vm225_20261104_monitor/final/root_stability.json').read_text())
    p=PUBLIC.parent.parent/'ising2d_KB_local_motif_R_plateau_continue_seed91_vm225_20261104_monitor/final/evaluation_arrays.npz'
    with np.load(p,allow_pickle=False) as a:bounds=a['flow_bounds'].copy()
    source=json.loads((PUBLIC/'roots.json').read_text())['roots']
    result=run.root_stability(st['snapshots'],bounds,source)
    assert [r['root_index'] for r in result['central_track']]==[2,2,4,2,4,2]
    assert all(r['point']['classification']=='repelling' for r in result['central_track'])
    assert result['snapshots']==st['snapshots']  # No numerical roots changed.
