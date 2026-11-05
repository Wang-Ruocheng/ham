"""Device-local hard-spin statistics and small-patch matched-baseline helpers.

No NumPy conversion, model fitting, physical sampling, labels, or data loading
occurs here.  In particular native_pair_batch indexes only the requested V/E
pixels from the full configuration tensor (which may be a CPU mmap view).
It does not materialize selected full physical chains before extracting them.
"""
from __future__ import annotations

import math

import torch

from .rsmine import buffered_shell_offsets


def _spins(value, *, minimum_dimensions=2):
    if not isinstance(value, torch.Tensor):
        raise TypeError("spins must be a torch.Tensor")
    if value.ndim < minimum_dimensions or any(n < 1 for n in value.shape):
        raise ValueError("Need nonempty trailing two-dimensional spin fields")
    if value.dtype in (torch.bool, torch.uint8) or value.is_complex():
        raise ValueError("Spins must use real +/-1 encoding")
    if not bool((value.abs().sub(1).abs() <= 1e-12).all()):
        raise ValueError("Spins must be binary +/-1 up to 1e-12 roundoff")
    return value


def _rand(shape, *, generator, device, dtype=torch.float64):
    if not isinstance(generator, torch.Generator):
        raise ValueError("Every stochastic operation needs an explicit torch.Generator")
    return torch.rand(shape, generator=generator, device=generator.device, dtype=dtype).to(device)


def _randint(high, shape, *, generator, device):
    if not isinstance(generator, torch.Generator):
        raise ValueError("Every stochastic operation needs an explicit torch.Generator")
    return torch.randint(high, shape, generator=generator, device=generator.device).to(device)


@torch.no_grad()
def motif_probabilities_tensor(spins, width=3, sample_axis=-3):
    """Periodic motif frequencies [...,512/16], with exact readout.py ordering.

    Input [...,M,H,W] is averaged over M,H,W, never over leading ensemble/C
    axes.  width3 uses centered lexicographic (-1,0,1)^2 offsets, LSB first.
    width2 uses anchored (00,10,01,11), weights (8,4,2,1).  Computation retains
    just an integer code field and small histograms, never a patch expansion
    or a per-site one-hot tensor.  Result stays on the input device, float64.
    """
    values = _spins(spins, minimum_dimensions=3)
    if type(sample_axis) is not int or not -values.ndim <= sample_axis < values.ndim:
        raise ValueError("sample_axis must name an existing nonspatial axis")
    axis = sample_axis % values.ndim
    if axis >= values.ndim - 2:
        raise ValueError("sample_axis cannot be a spatial axis")
    values = values.movedim(axis, -3)
    if width == 3:
        offsets = [(row, column, 1 << bit)
                   for bit, (row, column) in enumerate(
                       (r, c) for r in (-1, 0, 1) for c in (-1, 0, 1))]
        bins = 512
    elif width == 2:
        offsets = [(0, 0, 8), (1, 0, 4), (0, 1, 2), (1, 1, 1)]
        bins = 16
    else:
        raise ValueError("Only Q width3 and decoder width2 are defined")
    prefix = values.shape[:-3]
    rows = math.prod(prefix)
    indices = torch.zeros(values.shape, device=values.device, dtype=torch.long)
    bits = values > 0
    for row, column, weight in offsets:
        indices.add_(torch.roll(bits, shifts=(-row, -column), dims=(-2, -1)).long(), alpha=weight)
    flat = indices.reshape(rows, -1)
    offsets = bins * torch.arange(rows, device=values.device)[:, None]
    counts = torch.bincount((flat + offsets).reshape(-1), minlength=rows * bins)
    counts = counts.reshape(rows, bins).to(torch.float64) / flat.shape[1]
    return counts.reshape(*prefix, bins)


@torch.no_grad()
def native_pair_batch(configurations, chain_indices, pairs_per_chain=64, *, generator):
    """Extract V[G,B,9]/E[G,B,24] from CPU or GPU [all_chains,M,L,L].

    chain_indices selects the first axis, not a condition/label input.  Each
    output group remains one physical chain, with B DIFFERENT configurations
    sampled without replacement.  Each configuration gets an independent
    periodic top-left origin for its 3x3 V; the shell spans offsets [-2,4]^2.
    Training patch origins may be arbitrary; actual coarse lattice application
    is nonoverlapping stride3, implemented separately by coarsen_tensor.

    Only small pixel index arrays and returned patches are materialized.  A
    CPU generator and CPU configurations give CPU patches, suitable for moving
    only the patches (not full chains) to a GPU.
    """
    if not isinstance(configurations, torch.Tensor):
        raise TypeError("configurations must be a torch.Tensor")
    if configurations.ndim != 4 or any(n < 1 for n in configurations.shape):
        raise ValueError("Need configurations [all_chains,samples,L,L]")
    groups, samples, height, width = configurations.shape
    if height != width or height < 7:
        raise ValueError("Need square L>=7 to avoid aliased buffered shell sites")
    if (not isinstance(chain_indices, torch.Tensor) or chain_indices.ndim != 1
            or chain_indices.numel() < 1
            or chain_indices.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)):
        raise ValueError("chain_indices must be a nonempty integer vector")
    if type(pairs_per_chain) is not int or not 2 <= pairs_per_chain <= samples:
        raise ValueError("Need 2<=pairs_per_chain<=available configurations")
    if not isinstance(generator, torch.Generator):
        raise ValueError("An explicit torch.Generator is required")
    device = configurations.device
    ids = chain_indices.to(device=device, dtype=torch.long)
    if bool(((ids < 0) | (ids >= groups)).any()):
        raise ValueError("chain_indices contains an unavailable physical chain")
    sample_ids = torch.stack([
        torch.randperm(samples, generator=generator, device=generator.device)[:pairs_per_chain]
        for _ in range(len(ids))
    ]).to(device)
    origins = _randint(height, (len(ids), pairs_per_chain, 2), generator=generator, device=device)
    visible, environment = buffered_shell_offsets()

    def gather(offsets):
        positions = (origins[:, :, None, :] + offsets.to(device)) % height
        return configurations[
            ids[:, None, None], sample_ids[:, :, None],
            positions[..., 0], positions[..., 1],
        ]

    # Validate only the extracted pixels, not every cell of a huge mmap cache.
    v, e = gather(visible), gather(environment)
    for patch in (v, e):
        if patch.dtype in (torch.bool, torch.uint8) or patch.is_complex() or not bool(
                (patch.abs().sub(1).abs() <= 1e-12).all()):
            raise ValueError("Extracted configurations must contain +/-1 spins")
    return v, e


def _model_values(spins, model):
    if model is None:
        raise ValueError("This coarse-graining method requires its frozen model")
    try:
        parameter = next(model.parameters())
    except (AttributeError, StopIteration) as exc:
        raise ValueError("Need a model with parameters identifying device and dtype") from exc
    if parameter.device != spins.device:
        raise ValueError("Model and configurations must already be on the same device")
    return spins.to(dtype=parameter.dtype)


@torch.no_grad()
def coarsen_tensor(spins, method, model=None, probabilities=None, generator=None):
    """One hard binary 3x3/stride3 RG step, preserving prefix/device/dtype.

    Methods: majority/decimation (also fixed_majority/fixed_decimation),
    rsmine (also rsmi_ne), nmcrg, and qcr_c_pca.  QCR's frozen probability
    table is indexed by top-left row-major 3x3 bits, first bit least significant;
    it may be [512] P(+), or [512,2] probabilities ordered [-,+].  The native
    NMCRG model requires its explicit generator to share the model's device;
    this helper never clones or reseeds a generator to change that contract.
    """
    values = _spins(spins)
    height, width = values.shape[-2:]
    if height != width or height % 3:
        raise ValueError("Need square spatial size divisible by the fixed block=3")
    aliases = {"fixed_majority": "majority", "fixed_decimation": "decimation",
               "rsmi_ne": "rsmine"}
    method = aliases.get(method, method)
    if method == "decimation":
        return values[..., ::3, ::3].sign().clone()
    if method == "majority":
        positive = torch.zeros_like(values[..., ::3, ::3], dtype=torch.long)
        for row in range(3):
            for column in range(3):
                positive.add_(values[..., row::3, column::3] > 0)
        return (positive >= 5).to(values.dtype).mul(2).sub(1)
    if method not in ("rsmine", "nmcrg", "qcr_c_pca"):
        raise ValueError("Unknown matched-baseline coarse-graining method")
    if not isinstance(generator, torch.Generator):
        raise ValueError("Stochastic coarse-graining requires an explicit torch.Generator")
    if method == "nmcrg":
        model_input = _model_values(values, model)
        if getattr(model, "stride", 3) != 3:
            raise ValueError("Matched-data NMCRG must use stride=3")
        return model.coarse_grain(model_input, generator=generator).to(values.dtype)
    if method == "rsmine":
        # Accept either the complete native C+critic model or the C alone.
        coarse_grainer = getattr(model, "coarse_grainer", model)
        model_input = _model_values(values, coarse_grainer)
        if getattr(coarse_grainer, "block", None) != 3:
            raise ValueError("Matched-data RSMI-NE must use block=3")
        patches = torch.stack([
            model_input[..., row::3, column::3]
            for row in range(3) for column in range(3)
        ], dim=-1)
        return coarse_grainer.sample_spins(patches, generator=generator).squeeze(-1).to(values.dtype)
    if not isinstance(probabilities, torch.Tensor) or probabilities.device != values.device:
        raise ValueError("QCR probability table must be a tensor already on the input device")
    if probabilities.shape == (512, 2):
        if not bool(torch.allclose(probabilities.sum(-1), torch.ones_like(probabilities[:, 0]),
                                   rtol=0, atol=1e-6)):
            raise ValueError("QCR probability table rows must sum to one")
        probability_plus = probabilities[:, 1]
    elif probabilities.shape == (512,):
        probability_plus = probabilities
    else:
        raise ValueError("QCR table must have shape [512] or [512,2]")
    if (not probabilities.is_floating_point() or not bool(torch.isfinite(probabilities).all())
            or bool(((probabilities < 0) | (probabilities > 1)).any())):
        raise ValueError("QCR table must contain finite probabilities in [0,1]")
    codes = torch.zeros_like(values[..., ::3, ::3], dtype=torch.long)
    for bit, (row, column) in enumerate((r, c) for r in range(3) for c in range(3)):
        codes.add_((values[..., row::3, column::3] > 0).long(), alpha=1 << bit)
    p = probability_plus[codes]
    uniforms = _rand(p.shape, generator=generator, device=p.device, dtype=p.dtype)
    return (uniforms < p).to(values.dtype).mul(2).sub(1)
