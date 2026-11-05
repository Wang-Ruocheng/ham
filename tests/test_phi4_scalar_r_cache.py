import numpy as np
import torch

from experiments.phi4_scalar_r_cache import subset, seed
from experiments.phi4_ru_r_cache import encode_chain
from recurrent_rg.phi4_qcr import Phi4QCR


def test_private_scalar_subsets_are_deterministic_separated_and_unique():
    torch.manual_seed(10)
    before=torch.get_rng_state().clone()
    a=subset("fit",0,3)
    np.testing.assert_array_equal(a,subset("fit",0,3))
    assert len(a)==len(np.unique(a))==64 and a.min()>=0 and a.max()<128
    assert not np.array_equal(a,subset("eval",0,3))
    assert seed("fit",0,3,"C") != seed("fit",0,3,"subset")
    assert torch.equal(before,torch.get_rng_state())


def test_existing_collector_handles_scalar_frozen_QC_without_updates():
    torch.manual_seed(3)
    model=Phi4QCR(1.,hidden=8,theory_dim=1).double().eval().requires_grad_(False)
    field=torch.randn(128,9,9,1,dtype=torch.float64)
    before={k:v.clone() for k,v in model.state_dict().items()}
    values=encode_chain(model,field,subset("fit",0,0),c_seed=seed("fit",0,0,"C"))
    assert values["direct"].shape==(3,1)
    assert values["source_one_step"].shape==(2,1)
    assert values["source_rollout"].shape==(3,1)
    for k,v in model.state_dict().items():
        torch.testing.assert_close(v,before[k],rtol=0,atol=0)
    assert all(p.grad is None for p in model.parameters())
