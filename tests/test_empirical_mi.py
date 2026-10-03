import itertools,math
import numpy as np
import pytest
import torch
from recurrent_rg.empirical_mi import binary_lattice_mi_statistics,best_deterministic_binary_rule
from recurrent_rg.rsmi import binary_chain_joint_histograms,histogram_mutual_information,region_offsets


def test_compressed_chain_matches_dense_values_and_gradients():
    torch.manual_seed(31)
    x=(2*torch.randint(2,(3,9,12,1))-1).double()
    for flip in [False,True]:
        sparse=binary_lattice_mi_statistics(x,flip=flip)
        dense=binary_chain_joint_histograms(x,flip=flip)
        z=torch.randn(4,dtype=torch.double,requires_grad=True);p=z.sigmoid();table=torch.stack([1-p,p],1)
        a=sparse.information(table);b=histogram_mutual_information(dense,table)
        torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
        ga=torch.autograd.grad(a.sum(),z,retain_graph=True)[0];gb=torch.autograd.grad(b.sum(),z)[0]
        torch.testing.assert_close(ga,gb,atol=1e-12,rtol=1e-12)
        selected,report=best_deterministic_binary_rule(sparse)
        scores=[]
        for code in range(16):
            p=((code//(2**torch.arange(4)))%2).double()
            scores.append(float(histogram_mutual_information(dense,torch.stack([1-p,p],1)).mean()))
        assert abs(report['best_MI_nats']-max(scores))<1e-12


def test_square_full_shell_matches_explicit_region_dictionary():
    torch.manual_seed(9)
    x=(2*torch.randint(2,(2,3,6,6,1))-1).double()
    stats=binary_lattice_mi_statistics(x)
    v,e=region_offsets(2)
    assert len(v)==4 and len(e)==20
    z=torch.randn(16,dtype=torch.double,requires_grad=True);p=z.sigmoid();table=torch.stack([1-p,p],1)
    expected=[]
    for ensemble in x:
        pairs=[]
        for sample in ensemble:
            for i,j in itertools.product(range(6),repeat=2):
                def code(offsets):return sum(int(sample[(i+a)%6,(j+b)%6,0]>0)<<k for k,(a,b) in enumerate(offsets.tolist()))
                pairs.append((code(v),code(e)))
        categories=sorted(set(b for _,b in pairs));ids={e:i for i,e in enumerate(categories)}
        joint=torch.zeros(1,16,len(categories),dtype=torch.double)
        for a,b in pairs:joint[0,a,ids[b]]+=1/len(pairs)
        expected.append(histogram_mutual_information(joint,table)[0])
    torch.testing.assert_close(stats.information(table),torch.stack(expected),atol=1e-12,rtol=1e-12)
