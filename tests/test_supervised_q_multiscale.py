import torch

from experiments.supervised_q_ising2d_multiscale import frequencies, patterns_for, predict, metrics
from recurrent_rg.model_nd import RecurrentRGNetwork2D


def test_cached_supervised_q_matches_direct_outputs_and_gradients():
    torch.manual_seed(12)
    q = RecurrentRGNetwork2D(hidden=7, theory_dim=2).double().theory_encoder
    # Direct local-MLP path, without the binary lookup accelerator.
    q.binary_lookup = False
    x = (2*torch.randint(0,2,(3,8,8,8,1))-1).double()
    for k in range(2):
        coarse=x[:,:,::2**k,::2**k]
        hist=frequencies(coarse,q.geometry)
        torch.testing.assert_close(hist.sum(-1),torch.ones(3,dtype=torch.float64))
        direct=q(coarse)
        cached=predict(q,hist,patterns_for(q))
        torch.testing.assert_close(direct,cached,rtol=1e-10,atol=1e-12)
        a=torch.autograd.grad(direct.square().sum(),q.parameters())
        b=torch.autograd.grad(cached.square().sum(),q.parameters())
        for u,v in zip(a,b):
            torch.testing.assert_close(u,v,rtol=1e-10,atol=1e-12)


def test_metrics_inverts_label_normalization_and_keeps_scales():
    target=torch.tensor([[.2,-.1],[.6,.1]],dtype=torch.float64)
    mean,scale=target.mean(0),target.std(0,unbiased=False)
    pred=((target-mean)/scale)[:,None,:].repeat(1,4,1)
    result=metrics(pred,target,mean,scale)
    assert result['normalized_mse']==[0.]*4
    assert result['r2']==[[1.,1.]]*4
