import torch
from recurrent_rg.empirical_mi import lattice_mi_statistics


def test_sparse_matches_dense_information_and_gradients():
    torch.set_num_threads(1);torch.manual_seed(72)
    x=torch.nn.functional.one_hot(torch.randint(3,(2,8,6,6)),3).double()
    dense=lattice_mi_statistics(x,states=3,sparse_conditionals=False)
    sparse=lattice_mi_statistics(x,states=3,sparse_conditionals=True)
    z=torch.randn(81,3,dtype=torch.double,requires_grad=True);table=z.softmax(-1)
    a=dense.information(table);b=sparse.information(table)
    torch.testing.assert_close(a,b,rtol=1e-11,atol=1e-11)
    ga=torch.autograd.grad(a.sum(),z,retain_graph=True)[0];gb=torch.autograd.grad(b.sum(),z)[0]
    torch.testing.assert_close(ga,gb,rtol=1e-10,atol=1e-11)


def test_eight_states_uses_sparse_conditionals():
    x=torch.nn.functional.one_hot(torch.randint(8,(2,4,6,6)),8).double()
    stats=lattice_mi_statistics(x,states=8)
    assert stats.conditional.is_sparse and stats.conditional.shape[1]==4096
    assert stats.conditional._nnz()<=2*4*6*6
    values=stats.information(torch.full((4096,8),1/8,dtype=torch.double))
    torch.testing.assert_close(values,torch.zeros_like(values),atol=1e-12,rtol=0)
