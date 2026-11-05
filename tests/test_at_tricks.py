import copy
import numpy as np
import pytest
import torch
from recurrent_rg.checkpoints import build_model
from recurrent_rg.config import ModelConfig
from recurrent_rg.empirical_mi import categorical_local_pair_mi_statistics, lattice_mi_statistics
from experiments.at_predictive_q import prediction_loss, future_geometry_loss
from recurrent_rg.config import LossConfig


def test_future_geometry_updates_only_q_and_ignores_r():
    torch.manual_seed(7);torch.set_num_threads(1)
    m=build_model(ModelConfig(dimension=2,block_factor=2,categorical_states=4,fixed_kernel=True,kernel_sampling='deterministic',theory_dim=2,hidden=5,affine_reconstruction=False),2)
    x=torch.nn.functional.one_hot(torch.randint(4,(4,8,8,8)),4).double()
    cfg=LossConfig(noise_multiplier=0.)
    loss=future_geometry_loss(m,x,cfg)
    with torch.no_grad():
        for p in m.theory_rg.parameters():p.add_(100.)
    torch.testing.assert_close(loss,future_geometry_loss(m,x,cfg))
    loss.backward()
    assert sum(float(p.grad.abs().sum()) for p in m.theory_encoder.parameters())>0
    assert all(p.grad is None for p in m.theory_rg.parameters())
    assert torch.isfinite(loss)


def test_prediction_only_updates_q():
    torch.manual_seed(2);torch.set_num_threads(1)
    m=build_model(ModelConfig(dimension=2,block_factor=2,categorical_states=4,fixed_kernel=True,kernel_sampling='deterministic',theory_dim=2,hidden=5,affine_reconstruction=False),2)
    teacher=copy.deepcopy(m.theory_encoder).requires_grad_(False)
    x=torch.nn.functional.one_hot(torch.randint(4,(3,8,8,8)),4).double()
    loss=prediction_loss(m,teacher,x);loss.backward()
    assert sum(float(p.grad.abs().sum()) for p in m.theory_encoder.parameters())>0
    assert all(p.grad is None for p in m.theory_rg.parameters())
    assert all(p.grad is None for p in teacher.parameters())


def test_local_pair_buffer_mi_and_gradients():
    torch.manual_seed(3);torch.set_num_threads(1)
    x=torch.nn.functional.one_hot(torch.randint(4,(2,256,8,8)),4)
    stats=categorical_local_pair_mi_statistics(x)
    assert len(stats.groups)==4
    logits=torch.randn(256,4,dtype=torch.double,requires_grad=True)
    table=torch.softmax(logits,-1)
    info=stats.information(table)
    torch.testing.assert_close(info,torch.stack([g.information(table) for g in stats.groups]).mean(0))
    assert max(float(v) for v in info)<.03
    info.mean().backward();assert torch.isfinite(logits.grad).all()
    assert all(d['environment_sites']==2 and d['buffer']==1 for d in stats.diagnostics['groups'])
    with pytest.raises(ValueError):lattice_mi_statistics(x,states=4,environment_offsets=[[0,0]])
