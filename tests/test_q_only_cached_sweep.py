import torch
import numpy as np
from recurrent_rg.checkpoints import build_model
from recurrent_rg.config import ModelConfig
from experiments.q_only_cached_sweep import sample_counts, encode

def test_cached_subsets_match_original_encoder():
    torch.manual_seed(3)
    m=build_model(ModelConfig(dimension=2,categorical_states=3,kernel_sampling='categorical',hidden=8,affine_reconstruction=False),1)
    x=torch.nn.functional.one_hot(torch.randint(3,(1,4,6,6)),3).double()
    counts=sample_counts(x,m.geometry,3)
    codes=(torch.arange(3**9)[:,None]//(3**torch.arange(9)))%3
    patterns=torch.nn.functional.one_hot(codes,3).flatten(1).double()
    h=torch.from_numpy(np.asarray(counts[[0,3]].mean(0)))
    a=m.theory_encoder(x[:,[0,3]])
    b=encode(m.theory_encoder,patterns,h)
    torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-10)
    ga=torch.autograd.grad(a.sum(),tuple(m.theory_encoder.parameters()),retain_graph=True)
    gb=torch.autograd.grad(b.sum(),tuple(m.theory_encoder.parameters()))
    for a,b in zip(ga,gb):torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-10)
