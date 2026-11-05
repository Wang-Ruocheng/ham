import copy
import torch
import numpy as np
from experiments.dual_ising_q_vae import per_configuration_histograms,EnsembleVAE,mixture_tc,gaussian_kl,objective
from recurrent_rg.data.statistics import categorical_plaquette_distribution
from scripts.report_dual_ising_q_vae import factor_diagnostics


def test_cached_statistics_are_exact_periodic_probabilities():
    torch.set_num_threads(1);torch.manual_seed(3)
    x=torch.nn.functional.one_hot(torch.randint(4,(3,8,6,6)),4).to(torch.uint8)
    p=per_configuration_histograms(x)
    torch.testing.assert_close(p.mean(1),categorical_plaquette_distribution(x.double()),rtol=0,atol=1e-16)
    torch.testing.assert_close(p.sum(-1),torch.ones(3,8,dtype=torch.double))
    torch.testing.assert_close(p,per_configuration_histograms(x.roll(1,2)),rtol=0,atol=0)


def test_exact_mixture_tc_distinguishes_independent_from_correlated():
    torch.set_num_threads(1);torch.manual_seed(4)
    z=torch.randn(200,2,dtype=torch.double)
    independent=mixture_tc(z,torch.zeros(6,2,dtype=torch.double),torch.zeros(6,2,dtype=torch.double))
    assert abs(float(independent))<1e-12
    mu=torch.tensor([[-3.,-3.],[3.,3.]],dtype=torch.double,requires_grad=True)
    lv=torch.full_like(mu,-4.,requires_grad=True)
    samples=mu[torch.arange(200)%2]+torch.randn(200,2,dtype=torch.double)*torch.exp(lv[0]/2)
    tc=mixture_tc(samples,mu,lv)
    assert .65<float(tc)<.75
    tc.backward();assert torch.isfinite(mu.grad).all() and torch.isfinite(lv.grad).all()
    assert gaussian_kl(torch.zeros_like(mu),torch.zeros_like(lv))==0


def test_objective_routes_all_q_decoder_parameters_and_paired_warmup():
    torch.set_num_threads(1);torch.manual_seed(91)
    p=torch.rand(8,256,dtype=torch.double);p=p/p.sum(-1,keepdim=True)
    a,b=p[:4],p[4:];m=EnsembleVAE(p.mean(0),p.std(0).clamp_min(.001));other=copy.deepcopy(m)
    eps=torch.randn(8,2,dtype=torch.double);choices=torch.tensor([0,1,0,1])
    for model in (m,other):
        loss,terms=objective(model,a,b,eps,choices,.01,.1)
        loss.backward()
        assert torch.isfinite(loss) and all(t.grad is not None and torch.isfinite(t.grad).all() for t in model.parameters())
    for x,y in zip(m.parameters(),other.parameters()):torch.testing.assert_close(x.grad,y.grad,atol=0,rtol=0)
    for name,_ in m.named_modules():assert 'rg_kernel' not in name and 'theory_rg' not in name


def test_axis_score_requires_two_distinct_factors():
    labels=np.array([[a,b] for a in range(4) for b in range(4) for rep in range(2)],float)
    good,_,_=factor_diagnostics(labels.copy(),labels)
    torch.testing.assert_close(torch.tensor(good['parameter_main_effects'],dtype=torch.double),torch.eye(2,dtype=torch.double))
    assert good['axis_separation_score']==1
    bad,_,_=factor_diagnostics(np.repeat(labels[:,[0]],2,axis=1),labels)
    assert bad['axis_separation_score']==.5
