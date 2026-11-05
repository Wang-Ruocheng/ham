"""Probability-level regularizers for finite-alphabet local kernels."""
import torch
from .adapters import BinaryFieldHead, CategoricalFieldHead


def uniform_block_probabilities(kernel):
    """Rows P_C(.|a,...,a); q probes, no sampling or sign preference."""
    head = getattr(kernel, 'field_head', None)
    binary = isinstance(head, BinaryFieldHead)
    if not binary and not isinstance(head, CategoricalFieldHead):
        raise ValueError('Uniform-block probes require a learnable finite-alphabet kernel')
    if binary and head.sampling_mode != 'bernoulli':
        raise ValueError('Uniform-block probes require Bernoulli probabilities')
    first = kernel.local_encoder[0]
    q = 2 if binary else head.output_channels
    channels = 1 if binary else q
    if kernel.input_channels != channels or first.in_features % channels:
        raise ValueError('Inconsistent finite-alphabet input encoding')
    sites = first.in_features // channels
    states = torch.arange(q,device=first.weight.device)[:,None].expand(q,sites)
    inputs = ((2*states-1).to(first.weight.dtype) if binary else
              torch.nn.functional.one_hot(states,q).to(first.weight.dtype).flatten(1))
    logits = head.logits(kernel.local_encoder(inputs))
    if binary:
        plus = torch.sigmoid(2*logits)
        return torch.cat((1-plus,plus),dim=-1)
    return torch.softmax(logits,dim=-1)


def uniform_reference_marginal(kernel, *, max_patterns=65536, chunk_size=4096):
    """Exact E_{V~Uniform(q)^m} P_C(H|V), with differentiable probabilities.

    Independent of data and lattice dimension. Binary fields use +/-1 inputs;
    q-state fields use the repository's one-hot encoding. Enumerating q**m
    patterns is exponential: refuse oversized requests rather than silently
    replacing the specified loss with an approximation. No RNG is consumed.
    """
    if not isinstance(max_patterns, int) or max_patterns < 1 or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError('Reference enumeration limits must be positive integers')
    head = getattr(kernel, 'field_head', None)
    binary = isinstance(head, BinaryFieldHead)
    if not binary and not isinstance(head, CategoricalFieldHead):
        raise ValueError('Neutral loss requires a trainable finite-alphabet kernel')
    if binary and head.sampling_mode != 'bernoulli':
        raise ValueError('Binary neutral loss requires Bernoulli kernel probabilities')
    first = kernel.local_encoder[0]
    q = 2 if binary else head.output_channels
    channels = 1 if binary else q
    if kernel.input_channels != channels or first.in_features % channels:
        raise ValueError('Inconsistent finite-alphabet input encoding')
    sites = first.in_features // channels
    count = q**sites
    if count > max_patterns:
        raise ValueError(f'Neutral loss needs {q}^{sites}={count} reference patterns; '
                         f'exceeds max_patterns={max_patterns}. Increase the explicit limit '
                         'only if affordable; no sampling approximation is implicit.')
    powers = q**torch.arange(sites, device=first.weight.device)
    marginal = first.weight.new_zeros(q)
    for start in range(0, count, chunk_size):
        codes = torch.arange(start,min(count,start+chunk_size),device=first.weight.device)
        states = (codes[:,None]//powers) % q
        inputs = ((2*states-1).to(first.weight.dtype) if binary else
                  torch.nn.functional.one_hot(states,q).to(first.weight.dtype).flatten(1))
        logits = head.logits(kernel.local_encoder(inputs))
        if binary:
            plus = torch.sigmoid(2*logits)
            probabilities = torch.cat((1-plus,plus),dim=-1)
        else:
            probabilities = torch.softmax(logits,dim=-1)
        marginal = marginal + probabilities.sum(0)/count
    return marginal


def kernel_neutral_loss(kernel, *, max_patterns=65536):
    """1/2 ||E_uniform P_C(.|V) - Uniform(q)||^2.

    For q=2 this is exactly (mean P(+|V)-1/2)^2. Invariant to relabeling
    states. Does not imply equivariance, information retention, or prevent
    an input-independent uniform random kernel.
    """
    marginal = uniform_reference_marginal(kernel, max_patterns=max_patterns)
    return .5*(marginal-1/len(marginal)).square().sum()
