import copy
import torch

from experiments.frozen_qc_r_continue import continued_r_adam, r_objective
from experiments.rsmi_separate_qcr import build_parser
from recurrent_rg.checkpoints import build_model, qcr_state_dict
from recurrent_rg.config import DataConfig, ModelConfig, RunConfig


def test_r_only_flag_is_explicit_and_off_by_default():
    parser=build_parser()
    assert not parser.parse_args(['--config','x.json','--output','out']).r_only_continue
    assert parser.parse_args(['--source','src','--output','out','--r-only-continue']).r_only_continue


def test_r_adam_continues_exactly_and_q_c_are_frozen():
    torch.set_num_threads(1);torch.manual_seed(19)
    cfg=RunConfig(data=DataConfig(cache='unused.pt',physical_model='test'))
    model=build_model(ModelConfig(hidden=5,theory_dim=1),2)
    original=torch.optim.Adam(list(model.theory_encoder.parameters())+list(model.theory_rg.parameters()),lr=.001)
    targets=torch.randn(4,3,1,dtype=torch.double)
    original.zero_grad();r_objective(model,targets,cfg).backward();original.step()
    payload=dict(optimizer=copy.deepcopy(original.state_dict()),metadata=dict(
        config=dict(training=dict(train_modules=['Q','R'])),frozen_C=dict(checkpoint='x')))
    resumed=copy.deepcopy(model)
    for p in resumed.parameters():p.requires_grad_(False)
    for p in resumed.theory_rg.parameters():p.requires_grad_(True)
    opt=continued_r_adam(resumed,payload)
    before={k:v.clone() for k,v in qcr_state_dict(resumed).items()}
    for m,o in [(model,original),(resumed,opt)]:
        o.zero_grad();r_objective(m,targets,cfg).backward();o.step()
    for a,b in zip(model.theory_rg.parameters(),resumed.theory_rg.parameters()):
        torch.testing.assert_close(a,b,atol=0,rtol=0)
    for k,v in before.items():
        if not k.startswith('theory_rg.'):
            torch.testing.assert_close(v,qcr_state_dict(resumed)[k],atol=0,rtol=0)
    assert all(p.grad is None for p in resumed.theory_encoder.parameters())
    assert all(p.grad is None for p in resumed.rg_kernel.parameters())
    assert all(float(opt.state[p]['step'])==2 for p in resumed.theory_rg.parameters())
