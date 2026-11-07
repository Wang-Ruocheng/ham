import copy
from dataclasses import replace
import torch
import pytest
from recurrent_rg.config import RunConfig
from recurrent_rg.checkpoints import build_model, qcr_state_dict, load_model_state
from recurrent_rg.ensemble_vae import EnsembleVAEQ
from recurrent_rg.losses import qcr_objective
from recurrent_rg.data.statistics import categorical_plaquette_distribution, plaquette_distribution
from recurrent_rg.encoding import half_subset_theory
from recurrent_rg.terminal_r_fit import fingerprint, fit_r
from experiments.rsmi_separate_qcr import independent_component_gradients


@pytest.fixture(autouse=True)
def one_thread(): torch.set_num_threads(1)


def cfg(binary=False):
    name = 'ising2d_KB_weak_kl_vae_g2_seed91_20261106' if binary else 'dual_ising2d_weak_kl_vae_g2_seed91_20261105'
    return RunConfig.load('configs/'+name+'.json')


def test_matches_successful_qonly_architecture_and_objective():
    from experiments.dual_ising_q_vae import EnsembleVAE, objective
    p=torch.rand(7,256,dtype=torch.double).softmax(-1)
    center=p.mean(0);scale=p.std(0,unbiased=False).clamp_min(.001)
    torch.manual_seed(19);old=EnsembleVAE(center,scale)
    torch.manual_seed(19);new=EnsembleVAEQ();new.calibrate(p);new.set_objective_step(600)
    a,b=p[:3],p[3:6];eps=torch.randn(6,2,dtype=torch.double)
    x,_=objective(old,a,b,eps,torch.zeros(3,dtype=torch.long),.01,0)
    y,_=new.cross_subset_objective(a,b,epsilon=eps)
    torch.testing.assert_close(x,y,atol=1e-13,rtol=1e-13)
    for x,y in zip(old.parameters(),new.parameters()):torch.testing.assert_close(x,y,atol=0,rtol=0)
    xg=torch.autograd.grad(objective(old,a,b,eps,torch.zeros(3,dtype=torch.long),.01,0)[0],list(old.parameters()))
    yg=torch.autograd.grad(new.cross_subset_objective(a,b,epsilon=eps)[0],list(new.parameters()))
    for x,y in zip(xg,yg):torch.testing.assert_close(x,y,atol=1e-13,rtol=1e-13)


def test_private_noise_no_input_grad_and_eval_state_mutation():
    e=EnsembleVAEQ();e.set_objective_step(300)
    a=torch.rand(4,256,dtype=torch.double).softmax(-1).requires_grad_();b=a.detach().clone().requires_grad_()
    state=torch.get_rng_state();before=fingerprint(e)
    loss,terms=e.cross_subset_objective(a,b,scale_index=1);loss.backward()
    assert a.grad is None and b.grad is None
    assert torch.equal(state,torch.get_rng_state()) and fingerprint(e)==before
    assert terms['vae_effective_KL_weight']==.005
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in e.parameters())
    e.eval();loss2,_=e.cross_subset_objective(a,b,scale_index=1)
    torch.testing.assert_close(loss,loss2,atol=0,rtol=0)


@pytest.mark.parametrize('binary', [False, True])
def test_initial_C_R_and_rng_preserved_and_checkpoint_roundtrip(binary):
    c=cfg(binary);c.validate()
    old=replace(c.model,q_encoder='local_motif',affine_reconstruction=True)
    torch.manual_seed(91);a=build_model(old,2);rng=torch.get_rng_state()
    torch.manual_seed(91);b=build_model(c.model,2)
    assert torch.equal(rng,torch.get_rng_state())
    for part in ['rg_kernel','theory_rg']:
        assert fingerprint(getattr(a,part))==fingerprint(getattr(b,part))
    b.theory_encoder.set_objective_step(3000)
    twin=build_model(c.model,2);load_model_state(twin,qcr_state_dict(b))
    assert fingerprint(twin.theory_encoder)==fingerprint(b.theory_encoder)
    x=(2*torch.randint(2,(3,8,9,9,1))-1).double() if binary else torch.nn.functional.one_hot(torch.randint(4,(3,8,9,9)),4).double()
    state=torch.get_rng_state();z=b.theory_encoder(x);torch.testing.assert_close(z,twin.theory_encoder(x),atol=0,rtol=0)
    assert torch.equal(state,torch.get_rng_state())
    with pytest.raises(ValueError,match='replaces'):
        replace(c,loss=replace(c.loss,weights=replace(c.loss.weights,anchor=2))).validate()


@pytest.mark.parametrize('binary', [False, True])
def test_multiscale_objective_gradient_routes_and_terminal_freeze(binary):
    c=cfg(binary);spec=replace(c.model,hidden=5)
    torch.manual_seed(91);m=build_model(spec,2);m.detach_kernel_representation=True
    x=(2*torch.randint(2,(3,8,36,36,1))-1).double() if binary else torch.nn.functional.one_hot(torch.randint(4,(3,8,36,36)),4).double()
    pattern_fn=plaquette_distribution if binary else categorical_plaquette_distribution
    gradients=[]
    for step in (1,600):
        m.theory_encoder.set_objective_step(step);torch.manual_seed(66)
        total,t=qcr_objective(m,x,pattern_fn=pattern_fn,**c.loss.objective_kwargs())
        ce=torch.stack([t[f'vae_cross_entropy_level_{k}'] for k in range(3)]).mean()
        kl=torch.stack([t[f'vae_KL_level_{k}'] for k in range(3)]).mean()
        torch.testing.assert_close(t['separation'],ce+.01*min(step/600,1)*kl)
        c_loss=2*t['kernel_noncollapse']+t['kernel_sign'];closure=t['closure_total']
        independent_component_gradients(total,closure,c_loss,m.theory_encoder.parameters(),m.rg_kernel.parameters(),m.theory_rg.parameters())
        gradients.append([[p.grad.clone() for p in mod.parameters()] for mod in [m.rg_kernel,m.theory_rg]])
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.theory_encoder.parameters())
    for aa,bb in zip(*gradients):
        for a,b in zip(aa,bb):torch.testing.assert_close(a,b,atol=0,rtol=0)
    opt=torch.optim.Adam(list(m.theory_encoder.parameters())+list(m.rg_kernel.parameters())+list(m.theory_rg.parameters()),lr=.001)
    opt.step();before={n:fingerprint(mod) for n,mod in [('Q',m.theory_encoder),('C',m.rg_kernel)]}
    with torch.no_grad():targets=half_subset_theory(m,m.coarse_grain(x))
    result=fit_r(m,m.theory_rg,{'Q':m.theory_encoder,'C':m.rg_kernel},opt,targets,c.loss,max_steps=2,check_every=1)
    assert result['Q_C_unchanged'] and result['actual_extra_R_updates']==2
    for n,mod in [('Q',m.theory_encoder),('C',m.rg_kernel)]:assert before[n]==fingerprint(mod)


def test_binary_statistics_calibration_and_encoding_match_two_state_onehot():
    from recurrent_rg.ensemble_vae import calibrate_from_train
    e=EnsembleVAEQ(states=2)
    x=(2*torch.randint(2,(5,8,9,9,1))-1).double().requires_grad_()
    onehot=torch.nn.functional.one_hot((x.detach()[...,0]>0).long(),2).double()
    torch.testing.assert_close(e.probabilities(x),categorical_plaquette_distribution(onehot),atol=0,rtol=0)
    receipt=calibrate_from_train(e,x,chunk_size=2)
    assert receipt['labels_used'] is False
    torch.testing.assert_close(e.center,plaquette_distribution(x).mean(0))
    torch.testing.assert_close(e(x),e(onehot),atol=0,rtol=0)
    e(x).sum().backward()
    assert x.grad is None
