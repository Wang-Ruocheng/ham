import pytest
import torch
from pathlib import Path
from recurrent_rg.checkpoints import build_model
from recurrent_rg.config import ModelConfig
from recurrent_rg.losses import qcr_objective
from recurrent_rg.data.statistics import plaquette_distribution


@pytest.mark.parametrize('name', ['tfim1d_long.json', 'tfim1d_short.json',
    'tfim1d_g2_isotropy20.json', 'ising2d_joint_isotropy20.json'])
def test_current_training_presets_retain_covariance(name):
    from recurrent_rg.config import RunConfig
    config = RunConfig.load(Path(__file__).resolve().parents[1]/'configs'/name)
    config.validate()
    assert config.loss.isotropy_weight == 20


@pytest.mark.parametrize('theory_dim', [1, 2])
def test_opt_in_covariance_term_and_gradient(theory_dim):
    torch.manual_seed(4)
    model=build_model(ModelConfig(dimension=2,theory_dim=theory_dim,hidden=6,kernel_sampling='deterministic'),1)
    x=(2*torch.randint(2,(4,8,8,8,1))-1).double()
    base,terms=qcr_objective(model,x,pattern_fn=plaquette_distribution)
    boosted,extra=qcr_objective(model,x,pattern_fn=plaquette_distribution,isotropy_weight=20)
    assert 'isotropy' not in terms
    a=model.theory_encoder(x[:,:4]);b=model.theory_encoder(x[:,4:]);g=(a+b)/2
    g=g-g.mean(0);expected=(g.T@g/len(g)-torch.eye(theory_dim)).square().sum()/theory_dim
    torch.testing.assert_close(extra['isotropy'],expected)
    torch.testing.assert_close(boosted,base+20*expected)
    boosted.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.theory_encoder.parameters())
    with pytest.raises(ValueError):qcr_objective(model,x,pattern_fn=plaquette_distribution,isotropy_weight=-1)
