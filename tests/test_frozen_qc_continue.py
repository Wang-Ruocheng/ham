import copy
import torch
from experiments.potts_frozen_qc_continue import restrict_adam_state, pattern_function
from types import SimpleNamespace
from recurrent_rg.data.statistics import two_spin_distribution, plaquette_distribution, categorical_plaquette_distribution


def test_frozen_continuation_uses_system_statistics():
    for dimension,states,expected in [(1,0,two_spin_distribution),(2,0,plaquette_distribution),(2,3,categorical_plaquette_distribution)]:
        cfg=SimpleNamespace(model=SimpleNamespace(dimension=dimension,categorical_states=states))
        assert pattern_function(cfg) is expected


def test_restricted_adam_preserves_next_r_update():
    torch.manual_seed(1)
    m=torch.nn.Module()
    m.theory_encoder=torch.nn.Linear(1,2)
    m.rg_kernel=torch.nn.Linear(2,2)
    m.theory_rg=torch.nn.Linear(2,1)
    params=list(m.theory_encoder.parameters())+list(m.rg_kernel.parameters())+list(m.theory_rg.parameters())
    opt=torch.optim.Adam(params,lr=.001)
    for _ in range(3):
        opt.zero_grad();sum(p.square().sum() for p in params).backward();opt.step()
    source={'optimizer':copy.deepcopy(opt.state_dict())}
    narrowed=restrict_adam_state(source,m)
    twin=copy.deepcopy(m);r_opt=torch.optim.Adam(twin.theory_rg.parameters(),lr=.001)
    r_opt.load_state_dict(narrowed['optimizer'])
    opt.zero_grad();r_opt.zero_grad()
    sum(p.square().sum() for p in m.theory_rg.parameters()).backward()
    sum(p.square().sum() for p in twin.theory_rg.parameters()).backward()
    opt.step();r_opt.step()
    for a,b in zip(m.theory_rg.parameters(),twin.theory_rg.parameters()):assert torch.equal(a,b)
    assert len(source['optimizer']['param_groups'][0]['params'])==6
