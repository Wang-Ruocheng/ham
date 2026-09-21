"""Exact first-order autograd for mean local MLP features on binary patches.

Evaluate each of the 2**width patterns once. The input VJP is retained, so
straight-through gradients still reach C. This is not a histogram-only loss.
"""
import torch


class BinaryMean(torch.autograd.Function):
    @staticmethod
    def forward(ctx, patches, module, *parameters):
        width = patches.shape[-1]
        if width > 12 or not torch.all((patches.abs()-1).abs() < 1e-12):
            raise ValueError('BinaryMean requires binary patches with width <= 12')
        flat = patches.reshape(patches.shape[0], -1, width)
        powers = 2 ** torch.arange(width, device=patches.device)
        indices = ((flat > 0).long()*powers).sum(-1)
        patterns = 2*((torch.arange(2**width, device=patches.device)[:, None] // powers) % 2)-1
        patterns = patterns.to(patches.dtype)
        offsets = torch.arange(len(flat), device=patches.device)[:, None]*len(patterns)
        counts = torch.bincount((indices+offsets).flatten(), minlength=len(flat)*len(patterns))
        frequencies = counts.reshape(len(flat), len(patterns)).to(patches.dtype)/flat.shape[1]
        values = module(patterns)
        ctx.module = module
        ctx.patch_shape = patches.shape
        ctx.save_for_backward(patterns, indices, frequencies, *parameters)
        return frequencies @ values

    @staticmethod
    def backward(ctx, gradient):
        patterns, indices, frequencies, *parameters = ctx.saved_tensors
        need_input = ctx.needs_input_grad[0]
        with torch.enable_grad():
            patterns = patterns.detach().requires_grad_(need_input)
            values = ctx.module(patterns)
            jac = None
            if need_input:
                jac = torch.stack([torch.autograd.grad(values[:, f].sum(), patterns,
                                   retain_graph=True)[0] for f in range(values.shape[1])], dim=1)
            parameter_gradients = torch.autograd.grad(values, parameters, frequencies.T @ gradient)
        input_gradient = None
        if need_input:
            # Contract before gathering: [system, pattern, width], avoiding
            # a [system, sample, site, feature, width] intermediate.
            table = torch.einsum('sf,pfw->spw', gradient, jac)
            input_gradient = table.gather(1, indices[..., None].expand(-1, -1, patterns.shape[1]))
            input_gradient = (input_gradient/indices.shape[1]).reshape(ctx.patch_shape)
        return (input_gradient, None, *parameter_gradients)


def binary_mean(patches, module):
    return BinaryMean.apply(patches, module, *module.parameters())


class BinaryLocal(torch.autograd.Function):
    """Binary-pattern evaluation of a pointwise module, including its STE head."""
    @staticmethod
    def forward(ctx, patches, function, *parameters):
        width = patches.shape[-1]
        if width > 12 or not torch.all((patches.abs()-1).abs() < 1e-12):
            raise ValueError('BinaryLocal requires binary patches with width <= 12')
        powers = 2 ** torch.arange(width, device=patches.device)
        indices = ((patches > 0).long()*powers).sum(-1)
        patterns = (2*((torch.arange(2**width, device=patches.device)[:, None] // powers) % 2)-1).to(patches.dtype)
        values = function(patterns)
        ctx.function = function
        ctx.save_for_backward(patterns, indices, *parameters)
        return values[indices]

    @staticmethod
    def backward(ctx, gradient):
        patterns, indices, *parameters = ctx.saved_tensors
        need_input = ctx.needs_input_grad[0]
        with torch.enable_grad():
            patterns = patterns.detach().requires_grad_(need_input)
            values = ctx.function(patterns)
            weights = torch.zeros_like(values)
            weights.index_add_(0, indices.flatten(), gradient.reshape(-1, values.shape[-1]))
            jac = None
            if need_input:
                jac = torch.stack([torch.autograd.grad(values[:, f].sum(), patterns,
                                  retain_graph=True)[0] for f in range(values.shape[-1])], dim=1)
            parameter_gradients = torch.autograd.grad(values, parameters, weights)
        input_gradient = None
        if need_input:
            input_gradient = (jac[indices]*gradient[..., None]).sum(-2)
        return (input_gradient, None, *parameter_gradients)
