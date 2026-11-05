import copy
import torch
from experiments.q_initialization_pilot import initialize,calibrate_output
from experiments.q_only_cached_sweep import encode


def test_initializer_and_train_output_calibration():
    torch.manual_seed(91)
    q=torch.nn.Module()
    q.local_mlp=torch.nn.Sequential(torch.nn.Linear(27,12),torch.nn.Tanh()).double()
    q.head=torch.nn.Sequential(torch.nn.Linear(12,96),torch.nn.Tanh(),torch.nn.Linear(96,1)).double()
    patterns=torch.randn(20,27,dtype=torch.double)
    h=torch.rand(10,20,dtype=torch.double);h=h/h.sum(1,keepdim=True)
    with torch.no_grad():pooled=h@q.local_mlp(patterns)
    for name in ['calibrated','orthogonal','xavier_tanh','covariance_aware']:
        model=copy.deepcopy(q);initialize(model,name,pooled);calibrate_output(model,patterns,h)
        g=encode(model,patterns,h)
        torch.testing.assert_close(g.std(unbiased=False),torch.tensor(.1,dtype=torch.double))
        assert abs(g.mean().item())<1e-9
        assert sum(p.numel() for p in model.parameters())==sum(p.numel() for p in q.parameters())
