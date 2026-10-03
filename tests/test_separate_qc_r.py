import torch
import copy
from recurrent_rg.checkpoints import build_model
from recurrent_rg.config import ModelConfig
from recurrent_rg.data.statistics import two_spin_distribution
from recurrent_rg.losses import qcr_objective
from recurrent_rg.training import separate_qc_r_gradients


def test_separate_gradients_match_detached_multistep_reference():
    q=torch.tensor(0.7,dtype=torch.float64,requires_grad=True)
    c=torch.tensor(0.6,dtype=torch.float64,requires_grad=True)
    r=torch.tensor(1.2,dtype=torch.float64,requires_grad=True)
    direct=[q*c**k for k in range(4)]
    qc=(q-2).square()+(c-.4).square()
    closure=sum((r*direct[k]-direct[k+1]).square() for k in range(3))
    closure+=sum((r**k*direct[0]-direct[k]).square() for k in range(1,4))
    r_ref=r.detach().clone().requires_grad_(True)
    detached=[g.detach() for g in direct]
    reference=sum((r_ref*detached[k]-detached[k+1]).square() for k in range(3))
    reference+=sum((r_ref**k*detached[0]-detached[k]).square() for k in range(1,4))
    reference.backward()
    separate_qc_r_gradients(qc+closure,closure,[q,c],[r])
    torch.testing.assert_close(q.grad,2*(q-2))
    torch.testing.assert_close(c.grad,2*(c-.4))
    torch.testing.assert_close(r.grad,r_ref.grad)


def test_real_qc_gradients_do_not_depend_on_r_weights():
    torch.set_num_threads(1)
    torch.manual_seed(31)
    model=build_model(ModelConfig(hidden=4),2)
    altered=copy.deepcopy(model)
    with torch.no_grad():
        for p in altered.theory_rg.parameters():p.add_(.2)
    x=(2*torch.randint(2,(4,8,16,1))-1).double()
    gradients=[]
    for m in (model,altered):
        torch.manual_seed(81)
        total,terms=qcr_objective(m,x,pattern_fn=two_spin_distribution)
        qc=list(m.theory_encoder.parameters())+list(m.rg_kernel.parameters())
        separate_qc_r_gradients(total,5*(terms['commutator']+terms['rollout']),qc,list(m.theory_rg.parameters()))
        gradients.append([p.grad.clone() if p.grad is not None else None for p in qc])
    for a,b in zip(*gradients):
        if a is None:assert b is None
        else:torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-12)
