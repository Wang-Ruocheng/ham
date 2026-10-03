import pytest
import torch
from recurrent_rg.losses import local_orientation_loss, qcr_objective
from recurrent_rg.config import LossConfig, ModelConfig, QCRWeights
from recurrent_rg.checkpoints import build_model
from recurrent_rg.data.statistics import two_spin_distribution


class LinearMap:
    def __init__(self, matrix):
        self.matrix = matrix

    def advance_theory(self, x):
        return x @ self.matrix.T


@pytest.mark.parametrize('dim', [1, 2, 3])
@pytest.mark.parametrize('factor', [.3, 1., 2.])
def test_all_positive_scalings_unpenalized(dim, factor):
    x = torch.randn(8, 3, dim, dtype=torch.double)
    loss, scales = local_orientation_loss(LinearMap(factor*torch.eye(dim, dtype=x.dtype)), x)
    assert loss == 0
    assert scales.shape == (2,)


def test_reversal_detected_and_no_rng_consumed():
    x = torch.randn(8, 3, 2, dtype=torch.double)
    state = torch.get_rng_state()
    loss, _ = local_orientation_loss(LinearMap(-torch.eye(2, dtype=x.dtype)), x)
    assert loss > .9999
    assert torch.equal(state, torch.get_rng_state())


def test_negative_direction_and_finite_gradient():
    x = torch.randn(8, 3, 2, dtype=torch.double, requires_grad=True)
    matrix = torch.diag(torch.tensor([1.5, -1.1], dtype=torch.double)).requires_grad_()
    loss, _ = local_orientation_loss(LinearMap(matrix), x)
    assert loss > 0
    loss.backward()
    assert torch.isfinite(matrix.grad).all() and matrix.grad.abs().sum() > 0
    assert torch.isfinite(x.grad).all()


def test_joint_total_gradients_and_disabled_compatibility():
    torch.manual_seed(3)
    model = build_model(ModelConfig(dimension=1, theory_dim=2, hidden=6, kernel_sampling='deterministic'), 2)
    x = (2*torch.randint(2, (8, 12, 16, 1))-1).double()
    weights = QCRWeights(anchor=2., separation=10.)
    base, _ = qcr_objective(model, x, pattern_fn=two_spin_distribution, weights=weights)
    off, _ = qcr_objective(model, x, pattern_fn=two_spin_distribution, weights=weights, orientation_weight=0)
    assert torch.equal(base, off)
    loss, terms = qcr_objective(model, x, pattern_fn=two_spin_distribution, weights=weights, orientation_weight=.1)
    torch.testing.assert_close(loss, base+.1*terms['orientation'])
    assert 'orientation_level_0' in terms and 'orientation_level_1' in terms
    loss.backward()
    for module in (model.theory_encoder, model.rg_kernel, model.theory_rg):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())


@pytest.mark.parametrize('kwargs', [{'orientation_weight':-1}, {'orientation_radius':0}, {'orientation_weight':float('nan')}])
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        LossConfig(**kwargs).objective_kwargs()
