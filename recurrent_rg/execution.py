"""Execution-only accelerations; no changes to QCR parameters or objectives."""
from itertools import product
import torch
import torch.nn.functional as F


def reference_binary_pattern_distribution(configurations):
    """Original one-hot motif implementation for same-recipe timing audits."""
    dimensions = configurations.ndim - 3
    bits = (configurations.squeeze(-1) > 0).long()
    if dimensions == 1:
        pattern = 2 * bits + torch.roll(bits, -1, dims=2)
        return F.one_hot(pattern, 4).to(configurations.dtype).mean(dim=(1, 2))
    if dimensions == 3:
        pattern = torch.zeros_like(bits)
        for offset in product(range(2), repeat=3):
            pattern = 2 * pattern + torch.roll(bits, shifts=tuple(-i for i in offset), dims=(2, 3, 4))
        return F.one_hot(pattern, 256).to(configurations.dtype).mean(dim=(1, 2, 3, 4))
    if dimensions != 2:
        raise ValueError('Reference motifs support one-, two-, or three-dimensional fields')
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


def low_ram_execution_policy(*, reference=False):
    """Version storage/selection execution without changing the QCR recipe.

    This record does not claim cross-device bitwise determinism. The reference
    branch remains available for same-device, same-index regression checks.
    """
    enabled = not reference
    return dict(version=1, enabled=enabled,
        physical_cache_mmap=enabled,
        physical_validation='all_elements_in_bounded_chunks',
        batch_selection='joint_ensemble_sample_gather' if enabled else 'historical_two_stage',
        evaluation_panels='indices_only_materialized_at_log' if enabled else 'eager_full_panels',
        mi_cache_mmap=enabled,
        mi_split_loading='load_prepare_place_release_each_split' if enabled else 'load_all_then_place',
        source_dtype_unchanged=True, computation_dtype='torch.float64',
        CPU_index_generator_order_unchanged=True, C_sampling_order_unchanged=True,
        pin_memory=False, asynchronous_prefetch=False)


def select_training_batch(configurations, ids, indices, device, *, reference=False):
    """Gather the exact ordered ensemble/sample Cartesian product once.

    CPU-generated IDs, duplicates, negative indices, raw storage dtype and
    final float64 transfer are unchanged. No RNG, pinning, prefetching,
    residency decision or arithmetic enters this selection. The historical
    two-stage path can still be requested for execution-equivalence audits.
    """
    if configurations.device.type == 'cuda':
        ids = ids.to(configurations.device)
        indices = indices.to(configurations.device)
    if reference:
        selected = configurations[ids][:, indices]
    elif (isinstance(ids, torch.Tensor) and isinstance(indices, torch.Tensor)
          and ids.ndim == indices.ndim == 1
          and ids.dtype in (torch.int32, torch.int64)
          and indices.dtype in (torch.int32, torch.int64)):
        selected = configurations[ids[:, None], indices[None, :]]
    else:
        # Preserve less-common historical mask/list indexing semantics rather
        # than silently interpreting a boolean mask as integer IDs.
        selected = configurations[ids][:, indices]
    return selected.to(device=device, dtype=torch.float64)
