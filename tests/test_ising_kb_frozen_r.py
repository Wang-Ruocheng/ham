import copy
import numpy as np
import torch
import pytest
from recurrent_rg.checkpoints import build_model,qcr_state_dict
from recurrent_rg.config import ModelConfig
from experiments.ising_kb_frozen_r import freeze_and_continue,update,r_loss,training_targets,predictions,validate_loaded_state


def test_checkpoint_tensor_check_excludes_only_unsaved_unused_action():
    model=build_model(ModelConfig(dimension=2,theory_dim=2,hidden=5,block_factor=3),2)
    saved=copy.deepcopy(qcr_state_dict(model));payload={'model':saved}
    with torch.no_grad():
        for p in model.action_network.parameters():p.add_(1)
    assert validate_loaded_state(model,payload)['all_saved_tensors_bitwise_equal']
    with torch.no_grad():next(model.theory_rg.parameters()).add_(1)
    with pytest.raises(ValueError,match='tensor differs'):validate_loaded_state(model,payload)


def test_optimizer_continuation_freezing_and_allstart():
    torch.set_num_threads(1);torch.manual_seed(19)
    model=build_model(ModelConfig(dimension=2,theory_dim=2,hidden=96,block_factor=3,kernel_layout='nonoverlapping'),2)
    targets=torch.randn(5,3,2,dtype=torch.float64)
    opt=torch.optim.Adam(model.theory_rg.parameters(),lr=.003);update(model,targets,opt)
    frozen=copy.deepcopy(model);continued=freeze_and_continue(frozen,dict(epoch=1,optimizers={'R':opt.state_dict()}))
    before={k:v.clone() for k,v in frozen.state_dict().items() if not k.startswith('theory_rg.')}
    update(model,targets,opt);update(frozen,targets,continued)
    for x,y in zip(model.theory_rg.parameters(),frozen.theory_rg.parameters()):torch.testing.assert_close(x,y,rtol=0,atol=0)
    for k,v in before.items():torch.testing.assert_close(v,frozen.state_dict()[k],rtol=0,atol=0)
    assert all(p.grad is None for m in (frozen.theory_encoder,frozen.rg_kernel) for p in m.parameters())
    assert all(int(s['step'])==2 for s in continued.state.values())
    g=targets.clone().requires_grad_();a=frozen.advance_theory(g[:,0]);b=frozen.advance_theory(g[:,1]);c=frozen.advance_theory(a)
    expected=5*(a-g[:,1]).square().sum(-1).mean()+2.5*((b-g[:,2]).square().sum(-1).mean()+(c-g[:,2]).square().sum(-1).mean())
    torch.testing.assert_close(r_loss(frozen,g),expected);r_loss(frozen,g).backward();assert g.grad is None


def test_train_selection_never_reads_labels_or_test_targets():
    rng=np.random.default_rng(1);a={'direct_g':rng.normal(size=(714,3,2)), 'splits':np.array(['train']*474+['test']*240)}
    a['train_direct']=a['direct_g'][:474].copy();x=training_targets(a,'cpu')
    a['direct_g'][474:]=np.nan;a['labels']='never accessed'
    torch.testing.assert_close(x,training_targets(a,'cpu'));a['train_direct'][0,0,0]+=1
    with pytest.raises(ValueError):training_targets(a,'cpu')


def test_predictions_keep_frozen_targets():
    class Model(torch.nn.Module):
        def __init__(self):super().__init__();self.x=torch.nn.Parameter(torch.ones((),dtype=torch.float64))
        def advance_theory(self,x):return 2*x
    g=np.arange(18.).reshape(3,3,2);a=dict(direct_g=g,all_start_pairs=np.array([[0,1],[0,2],[1,2]]))
    result=predictions(Model(),a)
    np.testing.assert_array_equal(result['direct_g'],g)
    np.testing.assert_array_equal(result['all_start_g'][:,1],4*g[:,0])
    np.testing.assert_array_equal(result['all_start_g'][:,2],2*g[:,1])
