import math
import torch
from recurrent_rg.rsmi import binary_chain_joint_histograms, histogram_mutual_information


def test_histogram_matches_direct_regions_and_flip():
    torch.manual_seed(4)
    x=(2*torch.randint(2,(3,7,12,1))-1)
    expected=torch.zeros(3,4,4,dtype=torch.double)
    for s in range(3):
        for n in range(7):
            for a in range(12):
                bits=(x[s,n,:,0]>0).long()
                v=int(bits[a]+2*bits[(a+1)%12]);e=int(bits[(a-2)%12]+2*bits[(a+3)%12])
                expected[s,v,e]+=1/84
    torch.testing.assert_close(binary_chain_joint_histograms(x),expected)
    torch.testing.assert_close(binary_chain_joint_histograms(x,flip=True),.5*(expected+expected.flip((1,2))))


def test_information_values_and_gradients():
    joint=torch.eye(4,dtype=torch.double)[None]/4
    table=torch.tensor([[1.,0.],[0.,1.],[1.,0.],[0.,1.]],dtype=torch.double)
    torch.testing.assert_close(histogram_mutual_information(joint,table),torch.tensor([math.log(2)],dtype=torch.double))
    torch.testing.assert_close(histogram_mutual_information(torch.ones_like(joint)/16,table),torch.zeros(1,dtype=torch.double))
    torch.manual_seed(5)
    logits=torch.randn(4,dtype=torch.double,requires_grad=True)
    def objective(z):
        p=z.sigmoid();return histogram_mutual_information(joint,torch.stack((1-p,p),1))
    assert torch.autograd.gradcheck(objective,(logits,))


def test_no_cross_ensemble_mi():
    joint=torch.zeros(2,4,4,dtype=torch.double);joint[0,0,0]=1;joint[1,3,3]=1
    table=torch.tensor([[1.,0.],[1.,0.],[1.,0.],[0.,1.]],dtype=torch.double)
    torch.testing.assert_close(histogram_mutual_information(joint,table),torch.zeros(2,dtype=torch.double))
