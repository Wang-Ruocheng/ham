import torch
from experiments.q_last_layer_audit import loss


def test_last_layer_loss_matches_original_q_terms():
    torch.manual_seed(13)
    head=torch.nn.Linear(7,1).double()
    hidden=torch.randn(8,3,2,7,dtype=torch.double)
    p=torch.rand(8,3,2,81,dtype=torch.double)
    p=p/p.sum(-1,keepdim=True)
    value,terms=loss(head,hidden,p)
    assert abs(value.item()-(20*terms['anchor']+5*terms['consistency']+10*sum(terms['geometry'])/3))<1e-10
    value.backward()
    assert hidden.grad is None
    assert torch.isfinite(head.weight.grad).all()
