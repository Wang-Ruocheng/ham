import torch
from recurrent_rg.empirical_mi import binary_lattice_mi_statistics
from recurrent_rg.rsmi import region_offsets,histogram_mutual_information


def test_block3_empirical_mi_and_gradient_match_uncompressed_counts():
    torch.set_num_threads(1);torch.manual_seed(41)
    x=2*torch.randint(2,(2,4,9,9,1))-1
    stats=binary_lattice_mi_statistics(x,block=3)
    v,e=region_offsets(2,block=3,buffer=1)
    assert len(v)==9 and len(e)==24
    assert stats.conditional.shape[1]==512
    logits=torch.randn(512,2,dtype=torch.double,requires_grad=True)
    table=logits.softmax(-1);expected=[]
    for ensemble in x[...,0]:
        bits=(ensemble>0).long()
        def encode(offsets):
            return sum(torch.roll(bits,tuple(-int(t) for t in o),(-2,-1))*2**i for i,o in enumerate(offsets)).flatten()
        vc=encode(v);_,ec=torch.unique(encode(e),return_inverse=True)
        ne=int(ec.max())+1
        counts=torch.bincount(vc*ne+ec,minlength=512*ne).reshape(512,ne).double()
        expected.append(histogram_mutual_information((counts/counts.sum())[None],table)[0])
    actual=stats.information(table);reference=torch.stack(expected)
    torch.testing.assert_close(actual,reference,atol=1e-12,rtol=1e-10)
    ga=torch.autograd.grad(actual.sum(),logits,retain_graph=True)[0]
    gb=torch.autograd.grad(reference.sum(),logits)[0]
    torch.testing.assert_close(ga,gb,atol=1e-12,rtol=1e-9)


def test_default_block2_unchanged():
    torch.manual_seed(9);x=2*torch.randint(2,(2,4,8,8,1))-1
    a=binary_lattice_mi_statistics(x);b=binary_lattice_mi_statistics(x,block=2)
    for key in ('conditional','marginal'):
        torch.testing.assert_close(getattr(a,key),getattr(b,key),atol=0,rtol=0)
    torch.testing.assert_close(a.weights.to_dense(),b.weights.to_dense(),atol=0,rtol=0)
