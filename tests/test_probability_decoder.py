from dataclasses import replace
import math
import torch
from recurrent_rg.config import ModelConfig
from recurrent_rg.checkpoints import build_model, qcr_state_dict, load_model_state
from recurrent_rg.representation import project_probability_decoder_, probability_distribution_reconstruction_loss
from recurrent_rg.data.statistics import categorical_plaquette_distribution
from recurrent_rg.losses import qcr_objective
from experiments.rsmi_separate_qcr import independent_component_gradients


def configs():
    base = ModelConfig(dimension=2, theory_dim=2, hidden=5, categorical_states=4,
                       kernel_sampling='categorical', affine_reconstruction=True)
    new = replace(base, affine_reconstruction=False, reconstruction_hidden=8,
                  reconstruction_hidden_layers=2, reconstruction_probability_loss='squared',
                  reconstruction_spectral_bound=math.sqrt(2))
    return base, new


def test_paired_initialization_rng_and_checkpoint_roundtrip():
    torch.set_num_threads(1)
    torch.manual_seed(91); a = build_model(configs()[0], 1); rng = torch.get_rng_state()
    torch.manual_seed(91); b = build_model(configs()[1], 1)
    assert torch.equal(rng, torch.get_rng_state())
    for k, v in qcr_state_dict(a).items():
        if 'distribution_decoder' not in k:
            torch.testing.assert_close(v, qcr_state_dict(b)[k], atol=0, rtol=0)
    c = build_model(configs()[1], 1); load_model_state(c, qcr_state_dict(b))
    z = torch.randn(7, 2, dtype=torch.double)
    torch.testing.assert_close(b.theory_encoder.distribution_decoder(z), c.theory_encoder.distribution_decoder(z), atol=0, rtol=0)


def test_probability_mse_detaches_targets_and_preserves_bin_sum():
    m = build_model(configs()[1], 1); d = m.theory_encoder.distribution_decoder
    g = torch.randn(4, 2, dtype=torch.double, requires_grad=True)
    p = torch.rand(4, 256, dtype=torch.double).softmax(-1).requires_grad_()
    loss = probability_distribution_reconstruction_loss(d, g, p)
    torch.testing.assert_close(loss, ((d(g).softmax(-1)-p)**2).sum(-1).mean())
    loss.backward()
    assert p.grad is None and g.grad.abs().sum() > 0
    assert all(v.grad is not None for v in d.parameters())
    torch.testing.assert_close(d(g).softmax(-1).sum(-1), torch.ones(4, dtype=torch.double))


def test_exact_cap_after_optimizer_update_and_eval_is_read_only():
    m = build_model(configs()[1], 1); e = m.theory_encoder; d = e.distribution_decoder
    with torch.no_grad():
        for layer in d:
            if isinstance(layer, torch.nn.Linear): layer.weight.mul_(100)
    opt = torch.optim.Adam(d.parameters(), lr=.1)
    sum(v.square().sum() for v in d.parameters()).backward(); opt.step()
    project_probability_decoder_(e)
    for layer in d:
        if isinstance(layer, torch.nn.Linear):
            assert torch.linalg.matrix_norm(layer.weight, 2) <= math.sqrt(2)*(1+1e-12)
    before = {k:v.clone() for k,v in d.state_dict().items()}
    g = torch.randn(2, dtype=torch.double)
    jac = torch.autograd.functional.jacobian(lambda x:d(x).softmax(-1), g)
    assert torch.linalg.matrix_norm(jac, 2) <= math.sqrt(2)*(1+1e-12)
    d.eval(); d(g)
    for k,v in d.state_dict().items(): torch.testing.assert_close(before[k],v,atol=0,rtol=0)


def test_decoder_change_preserves_c_and_r_gradients():
    torch.set_num_threads(1); torch.manual_seed(19)
    x = torch.nn.functional.one_hot(torch.randint(4, (3,8,9,9)),4).double()
    gradients=[]
    for cfg in configs():
        torch.manual_seed(91); m=build_model(cfg,1); m.detach_kernel_representation=True
        torch.manual_seed(31)
        total,terms=qcr_objective(m,x,pattern_fn=categorical_plaquette_distribution,
            kernel_noncollapse_mode='variance_floor_ensemble',kernel_variance_ratio=1.,kernel_sign_weight=1.)
        closure=5*(terms['commutator']+terms['rollout'])
        c_loss=2*terms['kernel_noncollapse']+terms['kernel_sign']
        independent_component_gradients(total,closure,c_loss,m.theory_encoder.parameters(),m.rg_kernel.parameters(),m.theory_rg.parameters())
        gradients.append([[p.grad.clone() for p in mod.parameters()] for mod in [m.rg_kernel,m.theory_rg]])
        assert 'reconstruction_mse_level_0' in terms and 'reconstruction_js_level_0' not in terms
    for ga,gb in zip(*gradients):
        for a,b in zip(ga,gb): torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
