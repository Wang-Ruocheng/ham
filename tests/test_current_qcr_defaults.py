"""The main objective must stay aligned with the documented six-term loss."""
import pytest
import torch
from experiments.joint_qcr_ising import qcr_objective, two_spin_distribution
from recurrent_rg.model_nd import RecurrentRGNetwork1D


def test_default_objective_is_current_full_scale_coordinate_sum():
    torch.manual_seed(19)
    m=RecurrentRGNetwork1D(levels=3,hidden=8,theory_dim=2).double()
    x=(2*torch.randint(2,(4,8,16,1))-1).double()
    actual,t=qcr_objective(m,x,pattern_fn=two_spin_distribution)
    explicit,_=qcr_objective(m,x,pattern_fn=two_spin_distribution,
                            objective_mode='data_geometry',coordinate_reduction='sum',rollout_weight=5.)
    torch.testing.assert_close(actual,explicit,rtol=0,atol=0)
    expected=2*t['anchor']+5*t['coarse_consistency']+2*t['kernel_noncollapse']+10*t['separation']+5*t['commutator']+5*t['rollout']
    torch.testing.assert_close(actual,expected)
    assert all(f'data_geometry_level_{k}' in t for k in range(3))
    assert all(f'rollout_level_{k}' in t for k in [1,2])


def test_legacy_requires_explicit_archive_import():
    m=RecurrentRGNetwork1D(levels=2,hidden=4).double()
    with pytest.raises(ValueError,match='archive'):
        qcr_objective(m,torch.ones(2,4,8,1,dtype=torch.float64),
                      pattern_fn=two_spin_distribution,objective_mode='legacy')
