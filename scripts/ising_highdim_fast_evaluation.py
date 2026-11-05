"""Frozen-model CPU binary trajectories without materialized float64 patches.

This is an evaluation-only specialization, not a training implementation.  It
retains the original ensemble/chunk/scale order and calls the original
Bernoulli field head on expanded float64 logits, including its STE arithmetic.
It deliberately rejects unsupported models instead of silently changing their
evaluation.  No coupling labels enter this module.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
import hashlib
import weakref

import numpy as np
import torch

from recurrent_rg.adapters import BinaryFieldHead
from recurrent_rg.binary_pooling import binary_pattern_table
from recurrent_rg.checkpoints import isolated_rng
from recurrent_rg.model_nd import EnsembleTheoryEncoder, LocalCoarseGrainer


def _model_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(memoryview(array).cast('B'))
    return digest.hexdigest()


class FrozenBinaryCollector:
    """Exact lookup evaluator for frozen binary 2D b=2/3, Q-motif3 models.

    ``progress(stage, completed, total, extra)`` is called after every completed
    chunk. ``deadline.check()`` is also called inside encoding and coarse
    graining.  On a deadline exception, ``partial_arrays``, ``partial_rng_state``
    and ``partial_options`` expose the completed chunk prefix for a receipt.
    A partial pass must not be treated as a completed full evaluation.
    """

    def __init__(self, model, *, progress=None, deadline=None):
        self.model = model
        self.progress = progress
        self.deadline = deadline
        self._validate_model()
        self.model_sha256 = _model_digest(model)
        self._semantic_signature = self._semantics()
        self._micro_cache = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        self.partial_rng_state = None
        self.partial_options = None
        self._partial_direct = []
        self._partial_rollout = []
        with torch.no_grad():
            reference = next(model.parameters())
            _, patterns_q = binary_pattern_table(9, reference, cached=True)
            self._q_values = model.theory_encoder.local_mlp(patterns_q).detach()
            _, patterns_c = binary_pattern_table(model.geometry.block_volume, reference, cached=True)
            self._c_logits = model.rg_kernel.field_head.logits(
                model.rg_kernel.local_encoder(patterns_c)).detach()

    def _validate_model(self):
        model = self.model
        q, c = model.theory_encoder, model.rg_kernel
        geometry = model.geometry
        def standard_mlp(module, kinds):
            return (type(module) is torch.nn.Sequential and
                    tuple(type(layer) for layer in module) == kinds)
        linear, tanh = torch.nn.Linear, torch.nn.Tanh
        safe_q_head = (type(q.head) is linear or
                      standard_mlp(q.head, (linear, tanh, linear)))
        if (not isinstance(q, EnsembleTheoryEncoder) or
                not isinstance(c, LocalCoarseGrainer) or
                type(c.field_head) is not BinaryFieldHead or
                not getattr(q, 'binary_lookup', False) or
                not getattr(c, 'binary_lookup', False) or
                getattr(q, 'categorical_states', 0) or
                getattr(c, 'categorical_states', 0) or
                c.input_channels != 1 or c.kernel_layout != 'nonoverlapping' or
                geometry.dimensions != 2 or geometry.widths != (3, 3) or
                geometry.factors not in ((2, 2), (3, 3)) or not geometry.periodic or
                q.geometry != geometry or c.geometry != geometry or
                c.field_head.sampling_mode not in ('bernoulli', 'deterministic') or
                not standard_mlp(q.local_mlp, (linear, tanh, linear, tanh, linear, tanh)) or
                not standard_mlp(c.local_encoder, (linear, tanh, linear, tanh)) or
                not safe_q_head or
                not standard_mlp(model.theory_rg, (linear, tanh, linear, tanh, linear))):
            raise ValueError('Fast report requires binary lookup, periodic 2D Q-motif3 and nonoverlapping b=2/3 C')
        if any(value.device.type != 'cpu' or value.dtype != torch.float64
               for value in model.parameters()):
            raise ValueError('Fast report requires a CPU float64 frozen model')

    def _semantics(self):
        head = self.model.rg_kernel.field_head
        return (self.model.geometry, head.sampling_mode, head.relaxation_temperature,
                self.model.levels, self.model.theory_dim)

    def _check(self):
        if self.deadline is not None:
            self.deadline.check()

    def _unchanged(self):
        self._validate_model()
        if self._semantics() != self._semantic_signature or _model_digest(self.model) != self.model_sha256:
            raise RuntimeError('Frozen evaluation model changed; lookup/cache reuse refused')

    @property
    def partial_arrays(self):
        """Completed prefix only; concatenation happens on request, not per chunk."""
        if not self._partial_direct:
            return {}
        return {'direct': np.concatenate(self._partial_direct),
                'rollout': np.concatenate(self._partial_rollout)}

    def _binary_view(self, chunk, samples, spin_flip_augment):
        # Validate in bounded slices; never allocate a float64 patch stack.
        values = chunk[:, :samples, ..., 0].detach().numpy()
        for ensemble in values:
            flat = ensemble.reshape(-1)
            for start in range(0, flat.size, 262144):
                self._check()
                part = flat[start:start + 262144]
                if not np.all(np.abs(np.abs(part) - 1.) < 1e-12):
                    raise ValueError('Fast binary evaluation received nonbinary or nonfinite configurations')
        bits = np.greater(values, 0)
        if spin_flip_augment:
            half = samples // 2
            first, second = bits[:, :half], bits[:, half:samples]
            bits = np.concatenate((first, ~first, second, ~second), axis=1)
        return np.ascontiguousarray(bits)

    def _encode(self, bits):
        # The least-significant bit is the first lexicographic patch offset,
        # exactly as geometry.extract_patches + binary_mean enumerates it.
        self._check()
        indices = np.zeros(bits.shape, dtype=np.uint16)
        for bit, offsets in enumerate(self.model.geometry.patch_offsets()):
            self._check()
            shifted = np.roll(bits, tuple(-offset for offset in offsets), axis=(2, 3))
            indices |= shifted.astype(np.uint16) << bit
        counts = np.empty((len(bits), 512), dtype=np.int64)
        for index, pattern_ids in enumerate(indices):
            self._check()
            counts[index] = np.bincount(pattern_ids.reshape(-1), minlength=512)
        frequency = torch.from_numpy(counts).to(dtype=torch.float64) / int(np.prod(bits.shape[1:]))
        return self.model.theory_encoder.head(frequency @ self._q_values)

    def _coarsen(self, bits):
        self._check()
        factor = self.model.geometry.factors[0]
        if bits.shape[2] % factor or bits.shape[3] % factor:
            raise ValueError(f'Every requested spatial size must be divisible by b={factor}')
        # Preserve the original uint8 b=2 path; nine-bit b=3 indices need uint16.
        dtype = np.uint8 if factor == 2 else np.uint16
        indices = np.zeros((bits.shape[0], bits.shape[1], bits.shape[2] // factor,
                            bits.shape[3] // factor), dtype=dtype)
        offsets = ((row, column) for row in range(factor) for column in range(factor))
        for bit, (row, column) in enumerate(offsets):
            indices |= bits[:, :, row::factor, column::factor].astype(dtype) << bit
        logits = self._c_logits[torch.from_numpy(indices.astype(np.int64))]
        head = self.model.rg_kernel.field_head
        if head.sampling_mode == 'bernoulli':
            # Keep precisely the original shape, CPU dtype, random draw order,
            # and hard + soft - soft.detach floating-point operation order.
            field = head.sample_logits(logits)
        else:
            relaxed = torch.tanh(logits)
            hard = torch.where(relaxed >= 0, 1., -1.)
            field = hard.detach() + relaxed - relaxed.detach()
        return np.ascontiguousarray(field.numpy()[..., 0] > 0)

    def _dataset_cache(self, configurations, options):
        # Retain only two current datasets (normally TRAIN and TEST), weakly.
        # Per-chunk bit digests below also catch mutation via NumPy aliases,
        # which PyTorch's tensor version counter alone cannot detect.
        key = (id(configurations), options)
        entry = self._micro_cache.get(key)
        if entry is None or entry[0]() is not configurations:
            entry = (weakref.ref(configurations), {})
            self._micro_cache[key] = entry
        self._micro_cache.move_to_end(key)
        while len(self._micro_cache) > 2:
            self._micro_cache.popitem(last=False)
        return entry[1]

    def collect(self, configurations, *, samples, max_rg_steps=None,
                spin_flip_augment=False, chunk_size=16, sampling_seed=12345):
        """Return the original ``direct/rollout/one_step`` NumPy arrays."""
        self._check()
        self._unchanged()
        if (not isinstance(configurations, torch.Tensor) or configurations.device.type != 'cpu' or
                configurations.ndim != 5 or configurations.shape[-1] != 1 or
                configurations.requires_grad):
            raise ValueError('Expected detached CPU [ensembles,samples,H,W,1] configurations')
        if chunk_size < 1 or not len(configurations):
            raise ValueError('chunk_size and number of ensembles must be positive')
        if samples < 2 or samples % 2 or samples > configurations.shape[1]:
            raise ValueError('Need an even sample count available in every ensemble')
        steps = self.model.levels - 1 if max_rg_steps is None else max_rg_steps
        if not isinstance(steps, int) or steps < 0:
            raise ValueError('max_rg_steps must be a nonnegative integer')
        factor = self.model.geometry.factors[0]
        if any(length % (factor ** steps) for length in configurations.shape[2:4]):
            raise ValueError(f'Spatial sizes must be divisible at every requested b={factor} scale')
        options = (samples, spin_flip_augment, chunk_size, tuple(configurations.shape))
        cache = self._dataset_cache(configurations, options)
        self._partial_direct, self._partial_rollout = [], []
        self.partial_rng_state = None
        self.partial_options = dict(samples=samples, max_rg_steps=steps,
            spin_flip_augment=spin_flip_augment, chunk_size=chunk_size,
            sampling_seed=sampling_seed, completed_ensembles=0)
        was_training = self.model.training
        stochastic = self.model.rg_kernel.field_head.sampling_mode == 'bernoulli'
        rng = isolated_rng('cpu', sampling_seed) if stochastic else nullcontext()
        try:
            self.model.eval()
            with rng, torch.no_grad():
                self.partial_rng_state = torch.get_rng_state().clone()
                for offset, chunk in enumerate(configurations.split(chunk_size)):
                    self._check()
                    bits = self._binary_view(chunk, samples, spin_flip_augment)
                    bit_digest = hashlib.sha256(memoryview(bits).cast('B')).hexdigest()
                    previous = cache.get(offset)
                    if previous is not None and previous[0] == bit_digest:
                        g0 = previous[1].clone()
                        self.cache_hits += 1
                    else:
                        g0 = self._encode(bits)
                        cache[offset] = (bit_digest, g0.detach().clone())
                        self.cache_misses += 1
                    direct = [g0]
                    for _ in range(steps):
                        bits = self._coarsen(bits)
                        direct.append(self._encode(bits))
                    d = torch.stack(direct, dim=1)
                    rollout = [g0]
                    for _ in range(steps):
                        self._check()
                        rollout.append(self.model.advance_theory(rollout[-1]))
                    self._partial_direct.append(d.numpy())
                    self._partial_rollout.append(torch.stack(rollout, dim=1).numpy())
                    completed = min((offset + 1) * chunk_size, len(configurations))
                    self.partial_options['completed_ensembles'] = completed
                    self.partial_rng_state = torch.get_rng_state().clone()
                    if self.progress is not None:
                        self.progress('collect_trajectories', completed, len(configurations),
                            dict(chunk_index=offset, sampling_seed=sampling_seed,
                                 rg_steps=steps, samples=samples,
                                 cache_hits=self.cache_hits, cache_misses=self.cache_misses))
                self._check()
                arrays = self.partial_arrays
                arrays['one_step'] = self.model.advance_theory(
                    torch.from_numpy(arrays['direct'][:, :-1])).numpy()
            self._unchanged()
            return arrays
        finally:
            self.model.train(was_training)
