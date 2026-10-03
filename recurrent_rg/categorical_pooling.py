"""Exact first-order lookup for hard one-hot patches, including input VJPs.

Only observed patterns are evaluated. Discrete indexing is used in the forward
pass, but the backward is the same local MLP derivative as dense evaluation.
"""
import torch
import torch.nn.functional as F


def pattern_table(patches, states):
    shape = patches.shape
    if shape[-1] % states:
        raise ValueError('One-hot patch width must be divisible by states')
    sites = patches.reshape(*shape[:-1], -1, states)
    if not torch.all((sites-sites.round()).abs() < 1e-10) or not torch.all((sites.sum(-1)-1).abs() < 1e-10) or sites.min() < -1e-10:
        raise ValueError('Expected hard one-hot patches')
    powers = states ** torch.arange(sites.shape[-2], device=patches.device)
    codes = (sites.argmax(-1)*powers).sum(-1)
    unique, indices = torch.unique(codes, sorted=True, return_inverse=True)
    patterns = F.one_hot((unique[:,None]//powers)%states, states).flatten(1).to(patches.dtype)
    return patterns, indices.reshape(codes.shape)


class CategoricalMean(torch.autograd.Function):
    @staticmethod
    def forward(ctx, patches, states, module, *parameters):
        patterns, indices = pattern_table(patches, states)
        indices = indices.reshape(patches.shape[0], -1)
        offsets = torch.arange(len(indices),device=indices.device)[:,None]*len(patterns)
        counts = torch.bincount((indices+offsets).flatten(), minlength=len(indices)*len(patterns))
        freq = counts.reshape(len(indices),len(patterns)).to(patches.dtype)/indices.shape[1]
        ctx.module, ctx.patch_shape = module, patches.shape
        ctx.save_for_backward(patterns, indices, freq, *parameters)
        return freq @ module(patterns)

    @staticmethod
    def backward(ctx, gradient):
        patterns, indices, freq, *parameters = ctx.saved_tensors
        need = ctx.needs_input_grad[0]
        with torch.enable_grad():
            patterns = patterns.detach().requires_grad_(need)
            values = ctx.module(patterns)
            if need:
                jac = torch.stack([torch.autograd.grad(values[:,f].sum(),patterns,retain_graph=True)[0]
                                   for f in range(values.shape[-1])],dim=1)
            grads = torch.autograd.grad(values, parameters, freq.T @ gradient)
        gx = None
        if need:
            table = torch.einsum('sf,pfw->spw',gradient,jac)
            gx = table.gather(1,indices[...,None].expand(-1,-1,patterns.shape[1]))
            gx = (gx/indices.shape[1]).reshape(ctx.patch_shape)
        return (gx,None,None,*grads)


class CategoricalLocal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, patches, states, function, *parameters):
        patterns, indices = pattern_table(patches, states)
        ctx.function = function
        ctx.save_for_backward(patterns, indices, *parameters)
        return function(patterns)[indices]

    @staticmethod
    def backward(ctx, gradient):
        patterns, indices, *parameters = ctx.saved_tensors
        need = ctx.needs_input_grad[0]
        with torch.enable_grad():
            patterns = patterns.detach().requires_grad_(need)
            values = ctx.function(patterns)
            weights = torch.zeros_like(values)
            weights.index_add_(0,indices.flatten(),gradient.reshape(-1,values.shape[-1]))
            if need:
                jac = torch.stack([torch.autograd.grad(values[:,f].sum(),patterns,retain_graph=True)[0]
                                   for f in range(values.shape[-1])],dim=1)
            grads = torch.autograd.grad(values,parameters,weights)
        gx = (jac[indices]*gradient[...,None]).sum(-2) if need else None
        return (gx,None,None,*grads)
