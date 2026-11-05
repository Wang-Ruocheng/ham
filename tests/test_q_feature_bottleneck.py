import torch
from experiments.q_feature_bottleneck import FeatureBottleneck, objective


def test_bottleneck_detached_target_and_gradients():
    for dim in [1,2]:
        torch.manual_seed(91);model=FeatureBottleneck(dim)
        x=torch.randn(7,3,2,12,dtype=torch.double,requires_grad=True)
        value,terms=objective(model,x)
        assert abs(value.item()-(10*terms['reconstruction']+20*terms['anchor']+5*terms['consistency']))<1e-10
        value.backward()
        assert x.grad is None
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert model(x.detach()).shape==(7,3,2,dim)
