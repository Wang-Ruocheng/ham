from dataclasses import replace
import torch
from recurrent_rg.config import RunConfig
from recurrent_rg.checkpoints import build_model, qcr_state_dict, load_model_state
from recurrent_rg.losses import qcr_objective
from recurrent_rg.data.statistics import categorical_plaquette_distribution
from recurrent_rg.training import separate_qc_r_gradients


def test_fixed_categorical_decimation_and_symmetry():
    cfg=RunConfig.load('configs/ashkin_teller2d_g2_fixed_decimation_seed91.json')
    cfg.validate()
    torch.manual_seed(91);m=build_model(cfg.model,2)
    torch.manual_seed(91);learned=build_model(replace(cfg.model,fixed_kernel=False,kernel_sampling='categorical'),2)
    for a,b in zip(m.theory_encoder.parameters(),learned.theory_encoder.parameters()):
        assert torch.equal(a,b)
    for a,b in zip(m.theory_rg.parameters(),learned.theory_rg.parameters()):
        assert torch.equal(a,b)
    colors=torch.randint(0,4,(3,8,16,16))
    x=torch.nn.functional.one_hot(colors,4).double()
    assert len(list(m.rg_kernel.parameters()))==0
    assert torch.equal(m.rg_kernel(x),x[:,:,::2,::2,:])
    assert torch.equal(m.coarse_grain(x)[2],x[:,:,::4,::4,:])
    for xor in (1,2,3):
        alt=torch.nn.functional.one_hot(colors^xor,4).double()
        assert torch.equal(m.rg_kernel(alt).argmax(-1),m.rg_kernel(x).argmax(-1)^xor)
    # Partially ordered configurations retain their product sector exactly.
    product_colors=3*torch.randint(0,2,colors.shape)
    reduced=m.rg_kernel(torch.nn.functional.one_hot(product_colors,4).double()).argmax(-1)
    assert torch.all((reduced==0)|(reduced==3))
    loss,terms=qcr_objective(m,x,pattern_fn=categorical_plaquette_distribution,**cfg.loss.objective_kwargs())
    closure=5*terms['commutator']+5*terms['rollout']
    separate_qc_r_gradients(loss,closure,list(m.theory_encoder.parameters()),list(m.theory_rg.parameters()))
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.theory_encoder.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.theory_rg.parameters())
    copy=build_model(cfg.model,2);load_model_state(copy,qcr_state_dict(m))
    assert torch.equal(copy.theory_encoder(x),m.theory_encoder(x))
