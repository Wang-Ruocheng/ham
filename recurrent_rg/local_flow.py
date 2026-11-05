"""Smooth local affine corrections to an existing latent displacement network."""
import copy
import torch
from torch import nn


class LocalRBFDisplacement(nn.Module):
    """Returns a displacement, not the full R(g); add g exactly once outside."""
    def __init__(self, global_network, centers, origin, scale, widths):
        super().__init__()
        self.global_network = copy.deepcopy(global_network)
        for name,value in [('centers',centers),('origin',origin),('scale',scale),('widths',widths)]:
            self.register_buffer(name,value.detach().clone())
        count,dim = centers.shape
        self.offset = nn.Parameter(centers.new_zeros(count,dim))
        self.linear = nn.Parameter(centers.new_zeros(count,dim,dim))

    @classmethod
    def from_training_inputs(cls, network, inputs, count=32):
        x=inputs.detach().reshape(-1,inputs.shape[-1])
        origin=x.mean(0);scale=x.std(0).clamp_min(1e-6)
        z=(x-origin)/scale
        chosen=[int(z.square().sum(-1).argmin())]
        distance=(z-z[chosen[0]]).square().sum(-1)
        for _ in range(min(count,len(z))-1):
            idx=int(distance.argmax())
            if distance[idx]<1e-12:break
            chosen.append(idx)
            distance=torch.minimum(distance,(z-z[idx]).square().sum(-1))
        centers=z[chosen]
        if len(centers)>1:
            distances=torch.cdist(centers,centers)
            distances.fill_diagonal_(float('inf'))
            widths=.75*distances.topk(min(3,len(centers)-1),largest=False).values[:,-1]
            widths=widths.clamp_min(.15)
        else:widths=z.new_ones(1)
        return cls(network,centers,origin,scale,widths)

    def forward(self, g):
        delta=((g-self.origin)/self.scale).unsqueeze(-2)-self.centers
        weights=torch.exp(-.5*delta.square().sum(-1)/self.widths.square())
        affine=self.offset+torch.einsum('...ji,joi->...jo',delta,self.linear)
        correction=(weights.unsqueeze(-1)*affine).sum(-2)/(weights.sum(-1,keepdim=True)+1e-8)
        return self.global_network(g)+correction
