import json
import numpy as np
import pytest
from scripts import report_phi4_hopping_qonly_encoders as report


def fixture(tmp_path):
    rng=np.random.default_rng(11);splits=np.repeat(['train','train','test']*5,4);train=splits=='train'
    p=rng.random((60,4,625));p/=p.sum(-1,keepdims=True)
    a=dict(target_features=p,splits=splits,cell_ids=np.repeat(range(15),4),
        labels=np.column_stack((np.repeat(np.linspace(.32,1.04,15),4),np.ones(60),np.zeros(60))),
        train_target_mean=p[train].mean(0),train_target_std=np.sqrt(p[train].var(0).mean(-1)),
        source_flow_grid=np.linspace(-1,1,101)[:,None],source_flow_next=np.linspace(-.9,.9,101)[:,None])
    metrics={s:{} for s in ('train','test')}
    for name in report.MODELS:
        g=rng.normal(0,.1,(60,4,1));direction=np.ones(625)/25
        a.update({name+'_direct_g':g,name+'_direct_full_g':g,name+'_decoded':g*direction+1/625})
        if name!='source':a.update({name+'_g_first':g,name+'_g_second':g,name+'_decoder_jacobian':np.broadcast_to(direction,(60,4,625))})
        for split in metrics:
            mask=splits==split;rmse=np.sqrt(((a[name+'_decoded'][mask]-p[mask])**2).mean((0,2)));const=np.sqrt(((a['train_target_mean']-p[mask])**2).mean((0,2)))
            metrics[split][name]=dict(rmse=rmse.tolist(),nrmse=(rmse/a['train_target_std']).tolist(),constant_train_mean_rmse=const.tolist(),constant_train_mean_nrmse=(const/a['train_target_std']).tolist())
    np.savez(tmp_path/'evaluation_arrays.npz',**a)
    history=[dict(step=s,arms={n:dict(q_total=.1,reconstruction_raw=.01,subset_micro=.001,reconstruction_by_scale=[.01]*4) for n in report.MODELS[1:]}) for s in range(0,301,50)]
    summary=dict(status='complete',actual_steps=300,theory_dim=1,C_updates=0,R_updates=0,labels_used_in_training=False,test_used_in_training=False,old_R_valid_for_new_Q=False,all_tested_encoders_fresh=True,thermal_nu=None)
    for name,v in [('plan',{}),('summary',summary),('metrics',metrics),('history',history)]: (tmp_path/(name+'.json')).write_text(json.dumps(v))


def test_render_and_no_overwrite(tmp_path):
    fixture(tmp_path);pins={n:report.common._sha(tmp_path/n) for n in report.INPUTS}
    public=report.render_report(tmp_path)
    assert len(public)==6 and all((tmp_path/p).stat().st_size>0 for p in public)
    assert pins=={n:report.common._sha(tmp_path/n) for n in report.INPUTS}
    with pytest.raises(FileExistsError):report.render_report(tmp_path)


@pytest.mark.parametrize('mutation',['metrics','steps','warmstart','labels','dimension'])
def test_corrupt_evidence_rejected(tmp_path,mutation):
    fixture(tmp_path)
    name='metrics' if mutation=='metrics' else 'summary';v=json.loads((tmp_path/(name+'.json')).read_text())
    if mutation=='metrics':v['test']['C_hierarchical']['rmse'][0]*=2
    elif mutation=='steps':v['actual_steps']=200
    elif mutation=='warmstart':v['all_tested_encoders_fresh']=False
    elif mutation=='labels':v['labels_used_in_training']=True
    else:v['theory_dim']=2
    (tmp_path/(name+'.json')).write_text(json.dumps(v))
    with pytest.raises(ValueError):report.load_run(tmp_path)
