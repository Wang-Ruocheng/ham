"""Stable probability normalization without the fused CUDA softmax kernels.

PyTorch 2.6's float64 CUDA softmax/log-softmax can double-count part of
misaligned rows (notably 625 bins). Separate reductions avoid that kernel in
both the forward and backward pass; the mathematical objective is unchanged.
"""
import torch


def stable_log_softmax(logits, dim=-1):
    # Shift before logsumexp so even large common offsets do not reduce the
    # precision of the final subtraction. A detached shift has the same exact
    # derivative, since log-softmax is invariant to a common shift.
    shifted = logits - logits.amax(dim=dim, keepdim=True).detach()
    return shifted - torch.logsumexp(shifted, dim=dim, keepdim=True)


def stable_softmax(logits, dim=-1):
    return stable_log_softmax(logits, dim=dim).exp()
