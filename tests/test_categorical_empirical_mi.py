import math
import torch
from recurrent_rg.empirical_mi import EmpiricalBinaryMI,categorical_lattice_mi_statistics,optimize_categorical_rule
from recurrent_rg.rsmi import histogram_mutual_information,initialize_kernel_from_table
from recurrent_rg.losses import categorical_kernel_variance_floor
from recurrent_rg.checkpoints import build_model
from recurrent_rg.config import ModelConfig


def test_categorical_information_matches_joint_and_gradient():
    torch.manual_seed(8)
    joint=torch.rand(2,9,4,dtype=torch.double);joint/=joint.sum((1,2),keepdim=True)
    pe=joint.sum(1);conditional=(joint/pe[:,None,:]).permute(0,2,1).reshape(8,9)
    weights=torch.zeros(2,8,dtype=torch.double)
    weights[0,:4]=pe[0];weights[1,4:]=pe[1]
    stats=EmpiricalBinaryMI(conditional,weights.to_sparse(),joint.sum(-1),{})
    z=torch.randn(9,3,dtype=torch.double,requires_grad=True);table=z.softmax(-1)
    a=stats.information(table);b=histogram_mutual_information(joint,table)
    torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
    ga=torch.autograd.grad(a.sum(),z,retain_graph=True)[0];gb=torch.autograd.grad(b.sum(),z)[0]
    torch.testing.assert_close(ga,gb,atol=1e-12,rtol=1e-12)


def test_four_state_block3_full_shell_sparse_counts_and_gradient():
    torch.set_num_threads(1);torch.manual_seed(28)
    x=torch.nn.functional.one_hot(torch.randint(4,(2,2,7,7)),4).double()
    stats=categorical_lattice_mi_statistics(x,block=3,buffer=1)
    assert stats.diagnostics['visible_sites']==9 and stats.diagnostics['environment_sites']==24
    assert stats.diagnostics['sparse_conditionals'] and stats.diagnostics['buffer']==1
    assert stats.marginal.shape==(2,4**9)
    torch.testing.assert_close(stats.marginal.sum(-1),torch.ones(2,dtype=torch.double))
    logits=torch.randn(4**9,4,dtype=torch.double,requires_grad=True)
    information=stats.information(logits.softmax(-1))
    assert torch.isfinite(information).all()
    assert torch.all(information>=-1e-12) and torch.all(information<=math.log(4)+1e-12)
    information.sum().backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum()>0
    unseen=stats.marginal.sum(0)==0
    assert logits.grad[unseen].abs().max()==0


def test_categorical_counts_and_per_ensemble_floor():
    colors=torch.stack([torch.zeros(2,6,6,dtype=torch.long),torch.ones(2,6,6,dtype=torch.long)])
    x=torch.nn.functional.one_hot(colors,3).double()
    stats=categorical_lattice_mi_statistics(x)
    assert stats.marginal.shape==(2,81)
    assert stats.marginal[0,0]==1 and stats.marginal[1,40]==1
    torch.testing.assert_close(stats.information(torch.softmax(torch.randn(81,3,dtype=torch.double),-1)),torch.zeros(2,dtype=torch.double),atol=1e-12,rtol=0)
    source=torch.eye(3,dtype=torch.double)[None].repeat(2,1,1).reshape(2,3,1,1,3).requires_grad_()
    target=torch.stack([torch.tensor([1.,0.,0.]),torch.tensor([0.,1.,0.])]).double()[:,None,None,None,:].repeat(1,3,1,1,1).requires_grad_()
    assert categorical_kernel_variance_floor([source,target])==0
    loss=categorical_kernel_variance_floor([source,target],per_ensemble=True);assert loss>0
    loss.backward();assert source.grad is None and target.grad.abs().sum()>0


def test_full_categorical_variance_floor_is_color_neutral():
    source=torch.eye(3,dtype=torch.double).reshape(1,3,1,1,3)
    biased=source[:,[0,0,1]].clone().requires_grad_()
    assert categorical_kernel_variance_floor([source,biased],.25,per_ensemble=True)==0
    loss=categorical_kernel_variance_floor([source,biased],1.,per_ensemble=True)
    assert loss>0
    perm=[2,0,1]
    torch.testing.assert_close(loss,categorical_kernel_variance_floor(
        [source[...,perm],biased[...,perm]],1.,per_ensemble=True))
    assert categorical_kernel_variance_floor([source,source],1.,per_ensemble=True)==0
    constant=source[:,[0,0,0]]
    assert categorical_kernel_variance_floor([constant,constant],1.,per_ensemble=True)==0
    loss.backward();assert biased.grad.abs().sum()>0


def test_categorical_search_and_initialization():
    torch.set_num_threads(1)
    cond=torch.eye(3,dtype=torch.double)
    stats=EmpiricalBinaryMI(cond,torch.full((1,3),1/3,dtype=torch.double).to_sparse(),torch.full((1,3),1/3,dtype=torch.double),{})
    table,report=optimize_categorical_rule(stats,3,starts=4,steps=100)
    assert report['best_MI_nats']>math.log(3)-1e-10
    cfg=ModelConfig(dimension=2,block_factor=2,hidden=8,categorical_states=3,kernel_sampling='categorical')
    model=build_model(cfg,2)
    colors=(torch.arange(81)[:,None]//(3**torch.arange(4)))%3
    rule=torch.nn.functional.one_hot(colors[:,0],3).double()
    fit=initialize_kernel_from_table(model.rg_kernel,rule,steps=400)
    assert fit['max_probability_error']<.02


def test_independent_detachment_preserves_all_assigned_gradients():
    from recurrent_rg.losses import qcr_objective
    from recurrent_rg.data.statistics import categorical_plaquette_distribution
    from experiments.rsmi_separate_qcr import independent_component_gradients
    torch.set_num_threads(1);torch.manual_seed(6)
    model=build_model(ModelConfig(dimension=2,block_factor=2,hidden=5,categorical_states=3,kernel_sampling='categorical'),2)
    x=torch.nn.functional.one_hot(torch.randint(3,(3,6,8,8)),3).double()
    gradients=[];values=[]
    for detach in [False,True]:
        model.detach_kernel_representation=detach;model.zero_grad(set_to_none=True);torch.manual_seed(19)
        total,terms=qcr_objective(model,x,pattern_fn=categorical_plaquette_distribution,kernel_noncollapse_mode='variance_floor_ensemble')
        c_loss=2*terms['kernel_noncollapse']+sum(p.square().sum()*.001 for p in model.rg_kernel.parameters())
        closure=5*(terms['commutator']+terms['rollout'])
        independent_component_gradients(total+c_loss,closure,c_loss,model.theory_encoder.parameters(),model.rg_kernel.parameters(),model.theory_rg.parameters())
        gradients.append([p.grad.clone() for module in [model.theory_encoder,model.rg_kernel,model.theory_rg] for p in module.parameters()]);values.append(total.detach())
    torch.testing.assert_close(values[0],values[1],rtol=0,atol=0)
    for a,b in zip(*gradients):torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
