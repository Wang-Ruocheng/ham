"""Synthetic end-to-end report validation, without model loading or inference."""
import json

import numpy as np
import pytest

from scripts import report_phi4_hopping_qonly_nonlinear as report


def fixture(tmp_path):
    rng=np.random.default_rng(11)
    splits=np.repeat(['train','train','test']*5,4);train=splits=='train'
    p=rng.random((60,4,625));p/=p.sum(-1,keepdims=True)
    arrays=dict(target_features=p,splits=splits,cell_ids=np.repeat(range(15),4),
        labels=np.column_stack((np.repeat(np.linspace(.32,1.04,15),4),np.ones(60),np.zeros(60))),
        train_target_mean=p[train].mean(0),train_target_std=np.sqrt(p[train].var(0).mean(-1)),
        source_flow_grid=np.linspace(-1,1,101)[:,None],source_flow_next=np.linspace(-.9,.9,101)[:,None])
    metrics={s:{} for s in ('train','test')}
    for name in report.MODELS:
        g=rng.normal(0,.1,(60,4,1));direction=np.ones(625)/25
        arrays.update({name+'_direct_g':g,name+'_direct_full_g':g,name+'_g_first':g,name+'_g_second':g,
            name+'_decoded':g*direction+1/625,name+'_decoder_jacobian':np.broadcast_to(direction,(60,4,625)),
            name+'_decoder_grid':np.linspace(-.5,.5,201)[:,None],name+'_decoder_grid_jacobian':np.broadcast_to(direction,(201,625))})
        for split in metrics:
            mask=splits==split
            rmse=np.sqrt(((arrays[name+'_decoded'][mask]-p[mask])**2).mean((0,2)))
            const=np.sqrt(((arrays['train_target_mean']-p[mask])**2).mean((0,2)))
            metrics[split][name]=dict(rmse=rmse.tolist(),nrmse=(rmse/arrays['train_target_std']).tolist(),
                constant_train_mean_rmse=const.tolist(),constant_train_mean_nrmse=(const/arrays['train_target_std']).tolist())
    np.savez(tmp_path/'evaluation_arrays.npz',**arrays)
    history=[dict(step=s,arms={n:dict(q_total=.1,reconstruction_raw=.01,subset_micro=.001,reconstruction_by_scale=[.01]*4) for n in report.MODELS[1:]}) for s in (0,300)]
    summary=dict(status='complete',actual_steps=300,theory_dim=1,C_updates=0,R_updates=0,labels_used_in_training=False,old_R_valid_for_new_Q=False)
    for name,val in [('plan',{}),('summary',summary),('metrics',metrics),('history',history)]:
        (tmp_path/(name+'.json')).write_text(json.dumps(val))
    return arrays


def test_render_preserves_evidence_and_rejects_overwrite(tmp_path):
    fixture(tmp_path)
    pins={n:report.common._sha(tmp_path/n) for n in report.INPUTS}
    public=report.render_report(tmp_path)
    assert len(public)==6 and all((tmp_path/p).stat().st_size>0 for p in public)
    assert pins=={n:report.common._sha(tmp_path/n) for n in report.INPUTS}
    with pytest.raises(FileExistsError):report.render_report(tmp_path)


def test_metrics_inconsistency_rejected(tmp_path):
    fixture(tmp_path)
    metrics=json.loads((tmp_path/'metrics.json').read_text())
    metrics['test']['B_nonlinear']['rmse'][0]*=2
    (tmp_path/'metrics.json').write_text(json.dumps(metrics))
    with pytest.raises(ValueError,match='metrics'):report.load_run(tmp_path)


def test_dimension_change_rejected(tmp_path):
    fixture(tmp_path)
    summary=json.loads((tmp_path/'summary.json').read_text());summary['theory_dim']=2
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError):report.load_run(tmp_path)
