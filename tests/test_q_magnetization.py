import torch
from experiments.joint_qcr_ising2d_field import build_model
from experiments.supervised_q_ising2d_multiscale import patterns_for, frequencies, predict
from archive.qcr_ablations.supervised_q_magnetization import MagnetizationQ


def test_magnetization_q_initialization_and_new_weight_gradient():
    torch.manual_seed(91)
    base=build_model(False).theory_encoder
    q=MagnetizationQ(base)
    assert sum(p.numel() for p in q.parameters()) == 2450
    patterns=patterns_for(base)
    x=2*(torch.rand(3,8,8,8,1)>.2).double()-1
    h=frequencies(x,base.geometry)
    torch.testing.assert_close(q(h,patterns),predict(base,h,patterns))
    q(h,patterns).square().sum().backward()
    assert q.head[0].weight.grad[:,-1].norm()>0


def test_signed_magnetization_matches_each_scale_and_permutation():
    base=build_model(False).theory_encoder
    q=MagnetizationQ(base)
    patterns=patterns_for(base)
    x=2*torch.randint(0,2,(3,8,8,8,1)).double()-1
    for k in range(3):
        coarse=x[:,:,::2**k,::2**k]
        h=frequencies(coarse,base.geometry)
        torch.testing.assert_close(h@patterns[:,q.center],coarse.mean((1,2,3,4)))
        hp=frequencies(coarse[:,torch.randperm(8)],base.geometry)
        torch.testing.assert_close(q(h,patterns),q(hp,patterns))
