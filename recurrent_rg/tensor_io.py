"""Bounded-temporary validation and private file-backed CPU tensor loading.

These helpers change storage/validation execution only. They never rewrite a
cache, cast its tensors, draw random numbers, or omit part of a validity check.
Mapped datasets must be treated as immutable by callers.
"""
from pathlib import Path
import mmap as _mmap

import torch


VALIDATION_CHUNK_ELEMENTS = 1 << 20


def load_cpu_tensors(path, *, mmap=False):
    """Load safe tensor payloads, optionally mapping modern torch.save files.

    The old non-ZIP serialization format cannot be memory mapped. Detect that
    format before loading instead of catching arbitrary errors and retrying a
    potentially corrupt modern cache. The legacy fallback still uses
    weights_only=True. No data file or serialization identity is changed.
    """
    path = Path(path)
    kwargs = dict(map_location='cpu', weights_only=True)
    if mmap:
        with path.open('rb') as stream:
            modern = stream.read(4) == b'PK\x03\x04'
        if modern:
            get_options = getattr(torch.serialization, 'get_default_mmap_options', None)
            if get_options is not None and get_options() != _mmap.MAP_PRIVATE:
                raise ValueError('Low-RAM cache loading requires private, not shared, mmap')
            kwargs['mmap'] = True
    # torch 2.1's mmap path accepts str but rejects pathlib.Path.
    return torch.load(str(path), **kwargs)


def tensor_chunks(tensor, *, max_elements=VALIDATION_CHUNK_ELEMENTS, keep_last_dim=False):
    """Yield bounded strided views, including for noncontiguous inputs.

    Avoid flatten/reshape: those operations can silently copy an entire
    noncontiguous dataset before validation even starts. Narrowing is a view.
    """
    if isinstance(max_elements, bool) or not isinstance(max_elements, int) or max_elements < 1:
        raise ValueError('Validation chunk size must be a positive integer')
    if tensor.layout != torch.strided:
        raise ValueError('Bounded validation requires a strided tensor')
    if keep_last_dim and (tensor.ndim == 0 or tensor.shape[-1] > max_elements):
        raise ValueError('Validation chunk must fit the intact channel axis')
    if tensor.numel() <= max_elements:
        yield tensor
        return
    axes = tensor.shape[:-1] if keep_last_dim else tensor.shape
    axis = next(axis for axis, width in enumerate(axes) if width > 1)
    elements_per_slice = tensor.numel() // tensor.shape[axis]
    width = max(1, max_elements // elements_per_slice)
    for start in range(0, tensor.shape[axis], width):
        view = tensor.narrow(axis, start, min(width, tensor.shape[axis] - start))
        yield from tensor_chunks(view, max_elements=max_elements, keep_last_dim=keep_last_dim)


def all_finite(tensor, *, max_elements=VALIDATION_CHUNK_ELEMENTS):
    """Same finite predicate as isfinite(...).all(), with bounded temporaries."""
    return all(bool(torch.isfinite(chunk).all())
               for chunk in tensor_chunks(tensor, max_elements=max_elements))
