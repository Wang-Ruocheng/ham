"""Exact first-order autograd for mean local MLP features on binary patches.

Evaluate each of the 2**width patterns once. The input VJP is retained, so
straight-through gradients still reach C. This is not a histogram-only loss.
"""
import torch
from functools import lru_cache


@lru_cache(maxsize=16)
def _cached_pattern_table(width, device_type, device_index, dtype):
    """Only immutable enumeration constants are cached, never MLP outputs."""
    device = torch.device(device_type, device_index)
    powers = 2 ** torch.arange(width, device=device)
    patterns = (2*((torch.arange(2**width, device=device)[:, None] // powers) % 2)-1).to(dtype)
    return powers, patterns


def binary_pattern_table(width, reference, *, cached=False):
    if width > 12 or width < 1:
        raise ValueError('Binary lookup requires patch width between 1 and 12')
    if cached:
        device = reference.device
        return _cached_pattern_table(width, device.type, device.index, reference.dtype)
    powers = 2 ** torch.arange(width, device=reference.device)
    patterns = (2*((torch.arange(2**width, device=reference.device)[:, None] // powers) % 2)-1).to(reference.dtype)
    return powers, patterns


def _check_binary(patches, name, validated_binary_inputs):
    width = patches.shape[-1]
    if width > 12 or width < 1 or (not validated_binary_inputs and
            not torch.all((patches.abs()-1).abs() < 1e-12)):
        raise ValueError(f'{name} requires binary patches with width <= 12')


def binary_frequencies(patches, *, validated_binary_inputs=False):
    """Exact frequencies for detached inputs, used by the shared-Q fast path."""
    if patches.requires_grad:
        raise ValueError('Shared binary frequencies require detached inputs')
    _check_binary(patches, 'BinaryMean', validated_binary_inputs)
    width = patches.shape[-1]
    powers, patterns = binary_pattern_table(width, patches, cached=True)
    flat = patches.reshape(patches.shape[0], -1, width)
    indices = ((flat > 0).long()*powers).sum(-1)
    offsets = torch.arange(len(flat), device=patches.device)[:, None]*len(patterns)
    counts = torch.bincount((indices+offsets).flatten(), minlength=len(flat)*len(patterns))
    return counts.reshape(len(flat), len(patterns)).to(patches.dtype)/flat.shape[1], patterns


class BinaryMean(torch.autograd.Function):
    @staticmethod
    def forward(ctx, patches, module, *parameters):
        return _binary_mean_forward(ctx, patches, module, parameters,
                                    validated_binary_inputs=False)

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


def _binary_mean_forward(ctx, patches, module, parameters, *, validated_binary_inputs):
    _check_binary(patches, 'BinaryMean', validated_binary_inputs)
    width = patches.shape[-1]
    flat = patches.reshape(patches.shape[0], -1, width)
    powers, patterns = binary_pattern_table(width, patches, cached=validated_binary_inputs)
    indices = ((flat > 0).long()*powers).sum(-1)
    offsets = torch.arange(len(flat), device=patches.device)[:, None]*len(patterns)
    counts = torch.bincount((indices+offsets).flatten(), minlength=len(flat)*len(patterns))
    frequencies = counts.reshape(len(flat), len(patterns)).to(patches.dtype)/flat.shape[1]
    values = module(patterns)
    ctx.module = module
    ctx.patch_shape = patches.shape
    ctx.save_for_backward(patterns, indices, frequencies, *parameters)
    return frequencies @ values


class _TrustedBinaryMean(BinaryMean):
    @staticmethod
    def forward(ctx, patches, module, *parameters):
        return _binary_mean_forward(ctx, patches, module, parameters,
                                    validated_binary_inputs=True)


def binary_mean(patches, module, *, validated_binary_inputs=False):
    implementation = _TrustedBinaryMean if validated_binary_inputs else BinaryMean
    return implementation.apply(patches, module, *module.parameters())


class BinaryLocal(torch.autograd.Function):
    """Binary-pattern evaluation of a pointwise module, including its STE head."""
    @staticmethod
    def forward(ctx, patches, function, *parameters):
        return _binary_local_forward(ctx, patches, function, parameters,
                                     validated_binary_inputs=False)

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


def _binary_local_forward(ctx, patches, function, parameters, *, validated_binary_inputs):
    _check_binary(patches, 'BinaryLocal', validated_binary_inputs)
    powers, patterns = binary_pattern_table(patches.shape[-1], patches,
                                           cached=validated_binary_inputs)
    indices = ((patches > 0).long()*powers).sum(-1)
    values = function(patterns)
    ctx.function = function
    ctx.save_for_backward(patterns, indices, *parameters)
    return values[indices]


class _TrustedBinaryLocal(BinaryLocal):
    @staticmethod
    def forward(ctx, patches, function, *parameters):
        return _binary_local_forward(ctx, patches, function, parameters,
                                     validated_binary_inputs=True)


def binary_local(patches, function, parameters, *, validated_binary_inputs=False):
    """Trusted internal inputs may skip the redundant value check, not the VJP."""
    implementation = _TrustedBinaryLocal if validated_binary_inputs else BinaryLocal
    return implementation.apply(patches, function, *parameters)
