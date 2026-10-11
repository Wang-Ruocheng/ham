"""Execution-only accelerations; no changes to QCR parameters or objectives."""
import torch
import torch.nn.functional as F


def reference_binary_pattern_distribution(configurations):
    """Original one-hot motif implementation for same-recipe timing audits."""
    dimensions = configurations.ndim - 3
    bits = (configurations.squeeze(-1) > 0).long()
    if dimensions == 1:
        pattern = 2 * bits + torch.roll(bits, -1, dims=2)
        return F.one_hot(pattern, 4).to(configurations.dtype).mean(dim=(1, 2))
    if dimensions != 2:
        raise ValueError('Reference motifs support one- or two-dimensional fields')
    pattern = (8 * bits + 4 * torch.roll(bits, -1, dims=2)
               + 2 * torch.roll(bits, -1, dims=3)
               + torch.roll(torch.roll(bits, -1, dims=2), -1, dims=3))
    return F.one_hot(pattern, 16).to(configurations.dtype).mean(dim=(1, 2, 3))


def configure_cuda_execution(model, device, *, reference=False, independent_c=True):
    """Enable trusted internal binary paths only after prepared_data validates D.

    The caller must use the default hard Bernoulli C and detached Q inputs.
    This cannot waive validation for public standalone encoder calls, alter C
    draws, or remove C's cross-scale straight-through input VJP.
    """
    from .adapters import BinaryFieldHead
    from .model_nd import LocalCoarseGrainer

    enabled = torch.device(device).type == 'cuda' and not reference
    kernel = model.rg_kernel
    binary = (enabled and independent_c and isinstance(kernel, LocalCoarseGrainer)
              and kernel.input_channels == 1
              and isinstance(kernel.field_head, BinaryFieldHead)
              and kernel.field_head.sampling_mode == 'bernoulli'
              and getattr(kernel, 'binary_lookup', False)
              and getattr(model.theory_encoder, 'binary_lookup', False)
              and not getattr(model.theory_encoder, 'categorical_states', 0))
    model.share_binary_q_tables = bool(binary)
    model.fast_validated_statistics = bool(binary)
    kernel.validated_binary_inputs = bool(binary)
    return dict(version=1, enabled=enabled, reference=bool(reference),
                shared_binary_Q_tables=bool(binary),
                trusted_internal_binary_checks=bool(binary),
                precision='torch.float64',
                note='Same objectives/parameters/RNG draws; floating reduction order may differ. '
                     'C samples are never cached; C input VJP remains active.')


def resident_training_data(configurations, device, *, enabled, max_bytes=1 << 30):
    """Keep small raw caches on CUDA without changing dtype or batch RNG.

    Bound residency by both one GiB and ten percent of currently free memory.
    Larger caches keep the established CPU selection/selected-batch transfer.
    CPU generators still choose exactly the same indices in the same order.
    """
    device = torch.device(device)
    size = configurations.numel() * configurations.element_size()
    metadata = dict(resident=False, bytes=size, source_dtype=str(configurations.dtype))
    if not enabled or device.type != 'cuda':
        metadata['reason'] = 'reference_or_cpu'
        return configurations, metadata
    free, _ = torch.cuda.mem_get_info(device)
    budget = min(max_bytes, free // 10)
    metadata['budget_bytes'] = budget
    if size > budget:
        metadata['reason'] = 'cache_exceeds_safe_budget'
        return configurations, metadata
    try:
        resident = configurations.to(device=device)
    except torch.cuda.OutOfMemoryError:
        # Residency is optional. Selected batches must still train on CUDA;
        # this is not an automatic CPU fallback for the requested trainer.
        metadata['reason'] = 'residency_allocation_failed'
        return configurations, metadata
    metadata.update(resident=True, reason='fits_safe_budget')
    return resident, metadata


def select_training_batch(configurations, ids, indices, device):
    """Preserve historical two-stage indexing and the original CPU RNG order."""
    if configurations.device.type == 'cuda':
        ids = ids.to(configurations.device)
        indices = indices.to(configurations.device)
    return configurations[ids][:, indices].to(device=device, dtype=torch.float64)
