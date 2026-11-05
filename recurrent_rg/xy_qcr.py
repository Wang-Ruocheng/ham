"""Periodic XY adapter of the current local-Q continuous QCR recipe.

Q is invariant and C is equivariant under a common spin rotation. Reflection
symmetry is not hard constrained. No temperature, vortex or stiffness labels.
Full-shell information uses circular kernel smoothing, not exact histogram MI.
"""
import math
import torch
from torch import nn
from .phi4_aligned_qcr import _chunk_amplitudes, PATCH_OFFSETS, PLAQUETTE_OFFSETS
from .phi4_qcr import block_vectors, sample_regions
from .losses import all_start_anchor_closure_loss


def wrap(x): return torch.atan2(torch.sin(x),torch.cos(x))


def relative_embedding(angles, reference):
    d=angles-reference
    return torch.cat((torch.cos(d),torch.sin(d)),-1)


def circular_bin_probabilities(mean, sigma=.25, bins=16):
    edges=torch.linspace(-math.pi,math.pi,bins+1,device=mean.device,dtype=mean.dtype)
    means=wrap(mean)[...,None,None]
    shifts=2*math.pi*torch.arange(-2,3,device=mean.device,dtype=mean.dtype)
    z=(edges[:,None]+shifts-means)/(sigma*math.sqrt(2))
    cdf=.5*(1+torch.erf(z))
    p=(cdf[...,1:,:]-cdf[...,:-1,:]).sum(-1).clamp_min(0)
    return p/p.sum(-1,keepdim=True).clamp_min(1e-15)


class CircularC(nn.Module):
    def __init__(self,hidden=96,sigma=.25):
        super().__init__(); self.register_buffer('noise_std',torch.tensor(sigma,dtype=torch.double))
        self.net=nn.Sequential(nn.Linear(18,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,1)).double()
    def mean_blocks(self,blocks):
        center=blocks[...,4:5]
        return center+self.net(relative_embedding(blocks,center))
    def mean_field(self,x): return self.mean_blocks(block_vectors(x))
    def forward(self,x,generator=None):
        mean=self.mean_field(x)
        return wrap(mean+self.noise_std*torch.randn(mean.shape,dtype=mean.dtype,device=mean.device,generator=generator))
    def uniform_nll(self):
        a=torch.linspace(-math.pi,math.pi,17,device=self.noise_std.device,dtype=self.noise_std.dtype)[:-1]
        d=wrap(self.mean_blocks(a[:,None].expand(-1,9))[:,0]-a)
        shifts=2*math.pi*torch.arange(-2,3,device=d.device,dtype=d.dtype)
        # Wrapped-normal NLL relative to its modal value; additive constant only.
        return (-torch.logsumexp(-.5*((d[:,None]+shifts)/self.noise_std)**2,-1)
                +torch.logsumexp(-.5*(shifts/self.noise_std)**2,0)).mean()


def circular_features(x,chunk_size=131072):
    e,n,L,_,_=x.shape; obs=n*L*L; flat=x[...,0].reshape(e,obs)
    centers=torch.arange(8,device=x.device,dtype=x.dtype)*(2*math.pi/8)-math.pi
    accum=x.new_zeros(e,512); size=max(1,chunk_size//e)
    for s in range(0,obs,size):
        a=_chunk_amplitudes(flat,s,min(s+size,obs),L,PLAQUETTE_OFFSETS)
        d=a[...,1:]-a[...,:1]
        p=torch.softmax(8*torch.cos(d[...,None]-centers),-1)
        joint=(p[:,:,0,:,None,None]*p[:,:,1,None,:,None]*p[:,:,2,None,None,:]).flatten(-3)
        accum=accum+joint.sum(1)
    return accum/obs


class CircularQ(nn.Module):
    def __init__(self,hidden=96,dim=1,chunk_size=32768):
        super().__init__();self.chunk_size=chunk_size
        self.local_mlp=nn.Sequential(nn.Linear(18,hidden),nn.Tanh(),nn.Linear(hidden,12),nn.Tanh(),nn.Linear(12,12),nn.Tanh()).double()
        self.head=nn.Sequential(nn.Linear(12,hidden),nn.Tanh(),nn.Linear(hidden,dim)).double()
        self.decoder=nn.Linear(dim,512).double()
        with torch.no_grad(): self.decoder.weight.div_(self.decoder.weight.norm(dim=0,keepdim=True))
    def forward(self,x):
        from torch.utils.checkpoint import checkpoint
        e,n,L,_,_=x.shape;obs=n*L*L;flat=x[...,0].reshape(e,obs);size=max(1,self.chunk_size//e)
        def part(flat,s,t):
            a=_chunk_amplitudes(flat,s,t,L,PATCH_OFFSETS)
            return self.local_mlp(relative_embedding(a,a[...,4:5])).sum(1)
        vals=[]
        for s in range(0,obs,size):
            t=min(s+size,obs)
            vals.append(checkpoint(part,flat,s,t,use_reentrant=False) if torch.is_grad_enabled() else part(flat,s,t))
        return self.head(torch.stack(vals).sum(0)/obs)
    def decode(self,g):return self.decoder(g)
    def normalize_decoder_(self):pass # Explicit inheritance of current free-gain recipe.


class XYQCR(nn.Module):
    def __init__(self,seed=91,dim=1,hidden=96):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.C=CircularC(hidden);self.Q=CircularQ(hidden,dim)
            self.R=nn.Sequential(nn.Linear(dim,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,dim)).double()
    def features(self,x):return circular_features(x)
    def coarse_grain(self,x,generator=None):
        if x.shape[2]%9:raise ValueError('Two b3 transitions require L divisible by9')
        first=self.C(x,generator);return [x,first,self.C(first,generator)]
    def advance(self,g):return g+self.R(g)
    def advance_theory(self,g):return self.advance(g)


def circular_information(mean,environment,sigma=.25,shuffle=False):
    # Environment contains every site of the buffered 24-site shell.
    e=environment.detach();z=torch.cat((torch.cos(e),torch.sin(e)),-1)
    distance=(z.square().sum(-1)[...,None]+z.square().sum(-1)[:,None,:]-2*torch.bmm(z,z.transpose(1,2))).clamp_min(0)/24
    diagonal=torch.eye(mean.shape[1],device=mean.device,dtype=torch.bool)[None]
    weights=torch.softmax((-distance/(2*.5**2)).masked_fill(diagonal,-torch.inf),-1)
    p=circular_bin_probabilities(mean,sigma)
    if shuffle:p=p.roll(1,1) # Diagnostic deterministic within-ensemble mismatch only.
    conditional=torch.bmm(weights,p);marginal=conditional.mean(1,keepdim=True)
    return (conditional*(conditional.clamp_min(1e-15).log()-marginal.clamp_min(1e-15).log())).sum(-1).mean()


def objectives(model,x,generator=None,information_pairs=128):
    if x.shape[1]%2:raise ValueError('Even sample batch required')
    levels=model.coarse_grain(x,generator);half=x.shape[1]//2
    gs=[];rec=[];sub=[]
    for field in levels:
        field=field.detach();a=model.Q(field[:,:half]);b=model.Q(field[:,half:]);g=.5*(a+b)
        gs.append(g.detach())
        with torch.no_grad():target=model.features(field)
        rec.append((model.Q.decode(g)-target).square().sum(-1).mean());sub.append((a-b).square().sum(-1).mean())
    qr=torch.stack(rec).mean();sc=torch.stack(sub[1:]).mean();ql=10*qr+20*sub[0]+5*sc
    cl,_,pairs=all_start_anchor_closure_loss(model,torch.stack(gs,1),coordinate_reduction='sum');rl=10*cl
    blocks,env=sample_regions(x.detach(),information_pairs,generator)
    mi=circular_information(model.C.mean_blocks(blocks)[...,0],env,float(model.C.noise_std))
    variance=[]
    for field in levels[:-1]:
        mean=model.C.mean_field(field)
        vi=1-(field.detach().cos().flatten(1).mean(1)**2+field.detach().sin().flatten(1).mean(1)**2)
        vo=1-(mean.cos().flatten(1).mean(1)**2+mean.sin().flatten(1).mean(1)**2)
        variance.append((vi-vo).clamp_min(0).square().mean())
    var=torch.stack(variance).mean();color=model.C.uniform_nll();cc=-2*mi+2*var+color
    m=dict(q_total=ql,c_total=cc,r_total=rl,q_reconstruction=qr,q_subset_micro=sub[0],q_subset_coarse=sc,
           c_information=mi,c_variance=var,c_uniform=color,closure_all_start=cl,
           closure_one_step=.5*(pairs[0,1]+pairs[1,2]),closure_rollout=.5*(pairs[0,1]+pairs[0,2]))
    m.update({f'closure_start_{s}_target_{t}':v for (s,t),v in pairs.items()})
    return ql,cc,rl,{k:float(v.detach()) for k,v in m.items()}


def independent_step(model,x,opts,generator=None,information_pairs=128):
    q,c,r,m=objectives(model,x,generator,information_pairs)
    grads={name:torch.autograd.grad(loss,list(getattr(model,name).parameters())) for name,loss in zip(('Q','C','R'),(q,c,r))}
    for name in ('Q','C','R'):
        opts[name].zero_grad(set_to_none=True)
        for p,g in zip(getattr(model,name).parameters(),grads[name]):
            if not torch.isfinite(g).all():raise FloatingPointError('Nonfinite gradient')
            p.grad=g
    for opt in opts.values():opt.step()
    if not all(math.isfinite(v) for v in m.values()):raise FloatingPointError('Nonfinite objective')
    return m
