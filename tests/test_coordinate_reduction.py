import pytest
import torch
from recurrent_rg.model_nd import RecurrentRGNetwork1D
from experiments.joint_qcr_ising import qcr_objective,two_spin_distribution,latent_squared_error


@pytest.mark.parametrize('dim',[1,2,3])
def test_only_four_terms_scale_with_dimension(dim):
    torch.manual_seed(91)
    m=RecurrentRGNetwork1D(levels=3,hidden=8,theory_dim=dim).double()
    x=(2*torch.randint(0,2,(5,12,16,1))-1).double()
    args=dict(pattern_fn=two_spin_distribution,objective_mode='data_geometry',rollout_weight=5.)
    old,a=qcr_objective(m,x,coordinate_reduction='mean',**args)
    new,b=qcr_objective(m,x,coordinate_reduction='sum',**args)
    for key in ('anchor','coarse_consistency','commutator','rollout','rollout_level_1','rollout_level_2'):
        torch.testing.assert_close(b[key],dim*a[key])
    for key in ('separation','kernel_noncollapse','data_geometry_level_0'):
        torch.testing.assert_close(a[key],b[key])
    expected=old+(dim-1)*(2*a['anchor']+5*a['coarse_consistency']+5*a['commutator']+5*a['rollout'])
    torch.testing.assert_close(new,expected)
    new.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for mod in (m.theory_encoder,m.rg_kernel,m.theory_rg) for p in mod.parameters())


def test_sum_invariant_under_embedding_and_rotation():
    a=torch.randn(12,1,dtype=torch.float64)
    b=torch.cat((a,torch.zeros_like(a)),1)
    rotated=torch.cat((a,a),1)/2**.5
    for x in (b,rotated):
        torch.testing.assert_close(latent_squared_error(x,torch.zeros_like(x),'sum'),latent_squared_error(a,torch.zeros_like(a),'sum'))
