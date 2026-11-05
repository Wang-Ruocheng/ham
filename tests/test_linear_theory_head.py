from dataclasses import replace
import torch
from recurrent_rg.config import ModelConfig
from recurrent_rg.checkpoints import build_model, load_model_state, qcr_state_dict

def test_paired_linear_head():
    cfg=ModelConfig(dimension=2,categorical_states=3,kernel_sampling='categorical',hidden=16)
    torch.manual_seed(91);a=build_model(cfg,2);rng=torch.get_rng_state()
    torch.manual_seed(91);b=build_model(replace(cfg,theory_head='linear'),2)
    assert torch.equal(rng,torch.get_rng_state())
    for k,v in qcr_state_dict(a).items():
        if not k.startswith('theory_encoder.head.'):
            torch.testing.assert_close(v,qcr_state_dict(b)[k])
    z=torch.zeros(1,12,dtype=torch.double,requires_grad=True)
    torch.testing.assert_close(a.theory_encoder.head(z),b.theory_encoder.head(z))
    torch.testing.assert_close(torch.autograd.grad(a.theory_encoder.head(z).sum(),z)[0],
                               torch.autograd.grad(b.theory_encoder.head(z).sum(),z)[0])
    c=build_model(replace(cfg,theory_head='linear'),2)
    load_model_state(c,qcr_state_dict(b))
