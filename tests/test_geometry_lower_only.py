import pytest
import torch
from recurrent_rg.config import LossConfig
from recurrent_rg.losses import distribution_geometry_loss

@pytest.mark.parametrize('mode', ['absolute', 'relative_local'])
def test_only_lower_bound(mode):
    p = torch.eye(3, dtype=torch.double, requires_grad=True)
    g = torch.tensor([[-10.], [0.], [10.]], dtype=torch.double, requires_grad=True)
    loss, _ = distribution_geometry_loss(g,p,p,bounds='lower_only',mode=mode)
    interval, _ = distribution_geometry_loss(g,p,p,mode=mode)
    assert loss.item() == 0 and interval.item() > 0
    loss.backward()
    assert p.grad is None and torch.isfinite(g.grad).all()
    close = torch.tensor([[0.], [.1], [.2]],dtype=torch.double,requires_grad=True)
    loss, _ = distribution_geometry_loss(close,p,p,bounds='lower_only',mode=mode)
    assert loss.item() > 0
    loss.backward()
    assert close.grad.abs().sum() > 0

def test_bounds_config():
    assert LossConfig().geometry_bounds == 'interval'
    assert LossConfig(geometry_bounds='lower_only').objective_kwargs()['geometry_bounds'] == 'lower_only'
    with pytest.raises(ValueError):
        LossConfig(geometry_bounds='bad').objective_kwargs()
