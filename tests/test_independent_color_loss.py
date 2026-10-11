import torch

from experiments.rsmi_separate_qcr import independent_component_gradients
from recurrent_rg.checkpoints import build_model
from recurrent_rg.config import ModelConfig
from recurrent_rg.data.statistics import categorical_plaquette_distribution
from recurrent_rg.losses import qcr_objective, categorical_kernel_preserve_loss


def test_uniform_color_term_changes_only_c_gradients():
    torch.set_num_threads(1);torch.manual_seed(12)
    m=build_model(ModelConfig(dimension=2,block_factor=2,categorical_states=3,kernel_sampling='categorical',hidden=5),2)
    m.detach_kernel_representation=True
    x=torch.nn.functional.one_hot(torch.randint(3,(3,8,8,8)),3).double()
    gradients=[]
    for weight in [0.,1.]:
        m.zero_grad(set_to_none=True);torch.manual_seed(31)
        total,terms=qcr_objective(m,x,pattern_fn=categorical_plaquette_distribution,
            kernel_noncollapse_mode='variance_floor_ensemble',kernel_sign_weight=weight)
        closure=5*(terms['commutator']+terms['rollout'])
        objective=2*terms['kernel_noncollapse']+weight*terms.get('kernel_sign',0.)
        independent_component_gradients(total,closure,objective,m.theory_encoder.parameters(),m.rg_kernel.parameters(),m.theory_rg.parameters())
        gradients.append([[p.grad.clone() for p in module.parameters()] for module in [m.theory_encoder,m.rg_kernel,m.theory_rg]])
    for module in [0,2]:
        for a,b in zip(gradients[0][module],gradients[1][module]):torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
    expected=torch.autograd.grad(categorical_kernel_preserve_loss(m.rg_kernel),list(m.rg_kernel.parameters()))
    for a,b,e in zip(gradients[0][1],gradients[1][1],expected):torch.testing.assert_close(b-a,e,atol=1e-12,rtol=1e-12)
