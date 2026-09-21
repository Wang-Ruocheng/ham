import torch

from experiments.joint_qcr_ising2d_field import build_model
from experiments.supervised_q_ising2d_multiscale import frequencies, patterns_for
from archive.qcr_ablations.supervised_q_structure_ablation import QVariant


def test_equal_parameter_counts():
    base=build_model(False).theory_encoder
    for name in ['baseline','normalized','hierarchical']:
        assert sum(p.numel() for p in QVariant(base,name).parameters())==2390


def test_hierarchical_exact_cache_matches_direct_outputs_and_gradients():
    torch.manual_seed(91)
    base=build_model(False).theory_encoder
    q=QVariant(base,'hierarchical')
    x=(2*torch.randint(0,2,(2,5,4,4,1))-1).double()
    h=frequencies(x.reshape(10,1,4,4,1),base.geometry).reshape(2,1,5,512)
    predicted=q(h,patterns_for(base))[:,0]
    patches=base.geometry.extract_patches(x)
    u=q.local_mlp[:4](patches).mean((2,3))
    direct=q.head(q.local_mlp[4:](u).mean(1))
    torch.testing.assert_close(predicted,direct,rtol=1e-10,atol=1e-12)
    a=torch.autograd.grad(predicted.square().sum(),q.parameters())
    b=torch.autograd.grad(direct.square().sum(),q.parameters())
    for u,v in zip(a,b):torch.testing.assert_close(u,v,rtol=1e-10,atol=1e-12)
    torch.testing.assert_close(q(h[:,:,torch.tensor([4,1,3,0,2])],patterns_for(base))[:,0],direct)


def test_normalization_calibration_and_eval_do_not_use_test_batch_statistics():
    torch.manual_seed(91)
    base=build_model(False).theory_encoder
    q=QVariant(base,'normalized')
    h=torch.rand(8,4,512,dtype=torch.float64);h/=h.sum(-1,keepdim=True)
    patterns=patterns_for(base)
    q.calibrate(h,patterns);q.eval()
    mean,var=q.norm.running_mean.clone(),q.norm.running_var.clone()
    a=q(h[:1],patterns);b=q(h,patterns)[:1]
    torch.testing.assert_close(a,b)
    torch.testing.assert_close(q.norm.running_mean,mean)
    torch.testing.assert_close(q.norm.running_var,var)
    features=q.features(h,patterns).reshape(-1,12)
    torch.testing.assert_close(mean,features.mean(0))
    torch.testing.assert_close(var,features.var(0,unbiased=False))
