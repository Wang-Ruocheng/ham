import copy
import torch
import pytest

from experiments.rsmi_separate_qcr import build_parser, main
from experiments.frozen_qc_r_continue import r_objective
from recurrent_rg.checkpoints import build_model, qcr_state_dict
from recurrent_rg.config import ModelConfig, RunConfig, DataConfig
from recurrent_rg.losses import latent_squared_error, multistep_closure_loss


def test_refit_opt_in_and_conflicting_q_ablation_rejected():
    assert not build_parser().parse_args(['--source','x','--output','y']).r_only_refit
    with pytest.raises(ValueError,match='no other ablations'):
        main(['--source','x','--output','y','--r-only-refit','--q-coupling-weight','1'])


def test_vector_refit_uses_original_closure_and_preserves_q_c():
    torch.set_num_threads(1);torch.manual_seed(8)
    model=build_model(ModelConfig(dimension=2,block_factor=3,categorical_states=4,
        kernel_sampling='categorical',theory_dim=2,hidden=6,affine_reconstruction=True),2)
    cfg=RunConfig(data=DataConfig(cache='unused',physical_model='test'))
    before={k:v.clone() for k,v in qcr_state_dict(model).items()}
    for p in model.parameters():p.requires_grad_(False)
    for p in model.theory_rg.parameters():p.requires_grad_(True)
    direct=torch.randn(8,3,2,dtype=torch.double)
    one=latent_squared_error(model.advance_theory(direct[:,:-1]),direct[:,1:],'sum')
    roll,_=multistep_closure_loss(model,direct,'sum')
    torch.testing.assert_close(r_objective(model,direct,cfg),5*one+5*roll)
    opt=torch.optim.Adam(model.theory_rg.parameters(),lr=.001)
    opt.zero_grad();r_objective(model,direct,cfg).backward();opt.step()
    after=qcr_state_dict(model)
    assert any(not torch.equal(v,after[k]) for k,v in before.items() if k.startswith('theory_rg.'))
    for key,value in before.items():
        if not key.startswith('theory_rg.'):
            torch.testing.assert_close(value,after[key],atol=0,rtol=0)
