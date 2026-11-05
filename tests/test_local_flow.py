import torch
from recurrent_rg.local_flow import LocalRBFDisplacement


def test_initial_map_and_roundtrip():
    torch.manual_seed(1)
    net=torch.nn.Sequential(torch.nn.Linear(2,8),torch.nn.Tanh(),torch.nn.Linear(8,2)).double()
    x=torch.randn(40,3,2,dtype=torch.double)
    local=LocalRBFDisplacement.from_training_inputs(net,x,8)
    torch.testing.assert_close(local(x),net(x),rtol=0,atol=0)
    local.offset.data.normal_();local.linear.data.normal_()
    clone=LocalRBFDisplacement.from_training_inputs(net,x,8)
    clone.load_state_dict(local.state_dict())
    torch.testing.assert_close(local(x),clone(x))
    assert torch.autograd.gradcheck(local,(x[:2].requires_grad_(),))
    assert torch.isfinite(local(torch.full((3,2),1e6,dtype=torch.double))).all()
