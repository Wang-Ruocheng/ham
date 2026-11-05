import torch
from recurrent_rg.losses import distribution_reconstruction_loss
from recurrent_rg.checkpoints import build_model, qcr_state_dict, load_model_state
from recurrent_rg.config import ModelConfig
from dataclasses import replace

def test_js_and_detached_target():
    decoder=torch.nn.Linear(1,3).double()
    g=torch.tensor([[-1.],[1.]],dtype=torch.double,requires_grad=True)
    p=torch.tensor([[1.,0.,0.],[0.,1.,0.]],dtype=torch.double,requires_grad=True)
    loss=distribution_reconstruction_loss(decoder,g,p)
    assert 0 <= loss < .694
    loss.backward()
    assert p.grad is None and g.grad.abs().sum()>0
    assert decoder.weight.grad.abs().sum()>0
    target=decoder(g).softmax(-1).detach()
    assert abs(distribution_reconstruction_loss(decoder,g,target).item()) < 1e-12

def test_checkpoint_and_paired_initialization():
    cfg=ModelConfig(dimension=2,categorical_states=3,kernel_sampling='categorical',hidden=8,affine_reconstruction=False)
    torch.manual_seed(91);a=build_model(cfg,2);rng=torch.get_rng_state()
    torch.manual_seed(91);b=build_model(replace(cfg,reconstruction_hidden=32),2)
    assert torch.equal(rng,torch.get_rng_state())
    for k,v in qcr_state_dict(a).items():torch.testing.assert_close(v,qcr_state_dict(b)[k])
    c=build_model(replace(cfg,reconstruction_hidden=32),2)
    load_model_state(c,qcr_state_dict(b))
    assert b.theory_encoder.distribution_decoder[-1].out_features == 81
