import torch
from recurrent_rg.rsmi import EnvironmentCritic, kernel_probabilities, enumerated_infonce
from recurrent_rg.checkpoints import build_model
from recurrent_rg.config import ModelConfig
from recurrent_rg.training import separate_qc_r_gradients
from recurrent_rg.losses import qcr_objective
from recurrent_rg.data.statistics import two_spin_distribution
from experiments.rsmi_separate_qcr import evaluate_information


def test_combined_gradient_routing_and_critic():
    torch.set_num_threads(1);torch.manual_seed(7)
    model=build_model(ModelConfig(hidden=5),2)
    critic=EnvironmentCritic(2,5).double()
    x=(2*torch.randint(2,(4,8,16,1))-1).double()
    total,terms=qcr_objective(model,x,pattern_fn=two_spin_distribution)
    closure=5*(terms['commutator']+terms['rollout'])
    r=list(model.theory_rg.parameters());q=list(model.theory_encoder.parameters());c=list(model.rg_kernel.parameters())
    expected_r=torch.autograd.grad(closure,r,retain_graph=True)
    expected_q=torch.autograd.grad(total-closure,q,retain_graph=True)
    info=enumerated_infonce(kernel_probabilities(model.rg_kernel,x[:,:,:2,0]),critic(x[:,:,4:6,0])).mean()
    expected_c=torch.autograd.grad(total-closure-2*info,c,retain_graph=True)
    separate_qc_r_gradients(total-2*info,closure,q+c+list(critic.parameters()),r)
    for ps,gs in [(q,expected_q),(c,expected_c),(r,expected_r)]:
        for p,g in zip(ps,gs):torch.testing.assert_close(p.grad,g)
    assert sum(float(p.grad.abs().sum()) for p in critic.parameters())>0


def test_information_evaluation_is_reproducible_and_rng_isolated():
    torch.set_num_threads(1);torch.manual_seed(7)
    model=build_model(ModelConfig(hidden=5),2);critic=EnvironmentCritic(2,5).double()
    x=(2*torch.randint(2,(4,8,16,1))-1).double()
    state=torch.get_rng_state()
    a=evaluate_information(model,critic,x,flip=True,pairs=8,repeats=2)
    assert torch.equal(state,torch.get_rng_state())
    assert a==evaluate_information(model,critic,x,flip=True,pairs=8,repeats=2)
