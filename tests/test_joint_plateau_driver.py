from types import SimpleNamespace
import pytest

from experiments.rsmi_separate_qcr import build_parser, train_component_losses
from recurrent_rg.config import RunConfig


def test_joint_plateau_is_opt_in_and_does_not_enable_recipe_overrides():
    parser=build_parser()
    default=parser.parse_args(['--config','case.json','--output','out'])
    assert not default.joint_train_plateau
    enabled=parser.parse_args(['--config','case.json','--output','out','--joint-train-plateau'])
    assert enabled.joint_train_plateau
    assert not enabled.allow_nondefault_recipe
    assert not enabled.r_only_plateau


def test_component_objectives_use_train_metrics_and_actual_routing_weights():
    cfg=RunConfig.from_dict({'schema_version':5,'data':{'cache':'cache.pt','physical_model':'Ising'},
        'loss':{'weights':{'anchor':2,'separation':10,'coarse_consistency':5,'commutator':5},'rollout_weight':5}})
    args=SimpleNamespace(rsmi_weight=2.,variance_weight=2.,color_weight=1.)
    row=dict(epoch=200,train_anchor=.1,train_coarse_consistency=.2,train_separation=.3,
        train_rsmi=-.4,train_kernel_noncollapse=.5,train_kernel_sign=.6,
        train_commutator=.7,train_rollout=.8,test_total=-1e99)
    result=train_component_losses(row,cfg,args)
    assert result==pytest.approx(dict(epoch=200,component_Q=4.2,component_C=.8,component_R=7.5))
    row['test_total']=1e99
    assert train_component_losses(row,cfg,args)==result
