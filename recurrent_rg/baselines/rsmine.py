"""Independent PyTorch implementation of the RSMI-NE binary objective.

Algorithm reference: Gokmen et al., Phys. Rev. E 104, 064106 (2021),
https://arxiv.org/abs/2103.16887.  Mathematical/API cross-check: the authors'
Apache-2.0 TensorFlow implementation, RSMI-NE/RSMI-NE at commit
dd8c937496a7c866dbfe686494b98e0397ffb43d, specifically cg_layers.py,
critics.py and VBMI_bounds.py.  This is a framework migration, not a claim of
bitwise reproduction of the TensorFlow implementation or paper experiments.

The native loss jointly updates a linear binary coarse-grainer and a separable
critic through RelaxedBernoulli samples (the binary Gumbel-softmax).  There is
no exact-H enumeration, variance/color regularizer, Q, R, or closure term.
Defaults of block=3 and environment_size=24 are the disclosed matched-data
geometry, not the paper's original block/environment sizes.  Each negative
sample must belong to the same (condition, physical chain) as its anchor.
"""

from dataclasses import dataclass
from itertools import product
import math

import torch
from torch import nn


def buffered_shell_offsets(block=3, buffer=1):
    """Local 2D V/full-shell indices, with one empty buffer layer by default.

    For b=3, buffer=1: V has 9 sites; E is the 24-site perimeter of the 7x7
    enclosing square.  Applying these indices periodically to actual lattice
    configurations, with a non-aliasing lattice size, belongs to the caller.
    """
    for name, value, lower in (("block", block, 1), ("buffer", buffer, 0)):
        if not isinstance(value, int) or isinstance(value, bool) or value < lower:
            raise ValueError(f"{name} must be an integer >= {lower}")
    low, high = -buffer - 1, block + buffer
    visible = list(product(range(block), repeat=2))
    environment = [
        point for point in product(range(low, high + 1), repeat=2)
        if low in point or high in point
    ]
    return torch.tensor(visible, dtype=torch.long), torch.tensor(environment, dtype=torch.long)


def _uniform(shape, *, generator, device, dtype):
    """Permit an explicit CPU RNG even when model/data reside on an accelerator."""
    rng_device = generator.device if generator is not None else device
    return torch.rand(shape, generator=generator, device=rng_device, dtype=dtype).to(device)


def _normal(shape, *, generator, device, dtype):
    rng_device = generator.device if generator is not None else device
    return torch.randn(shape, generator=generator, device=rng_device, dtype=dtype).to(device)


class _Dense(nn.Module):
    """Dense with Keras Dense's Glorot-uniform/zero-bias initialization.

    Unlike constructing nn.Linear and overwriting it, this does not consume
    the global RNG when an explicit generator has been supplied.
    """

    def __init__(self, input_size, output_size, *, generator, device, dtype):
        super().__init__()
        limit = math.sqrt(6.0 / (input_size + output_size))
        weight = _uniform(
            (output_size, input_size), generator=generator, device=device, dtype=dtype
        )
        self.weight = nn.Parameter((2 * weight - 1) * limit)
        self.bias = nn.Parameter(torch.zeros(output_size, device=device, dtype=dtype))

    def forward(self, values):
        return torch.nn.functional.linear(values, self.weight, self.bias)


class LinearBinaryCoarseGrainer(nn.Module):
    """One bias-free linear logit, followed by native binary Concrete sampling.

    A logit ``a = w dot V`` gives P(H=1|V)=sigmoid(a), *not* sigmoid(2*a).
    Native H values supplied to the critic are relaxed 0/1 values.  Physical
    evaluation uses sample_spins(), which returns exact -1/+1 samples.
    """

    def __init__(
        self, block=3, *, initial_temperature=0.5, minimum_temperature=0.1,
        relaxation_rate=0.005, generator=None, device=None, dtype=torch.float32,
    ):
        super().__init__()
        if not isinstance(block, int) or isinstance(block, bool) or block < 1:
            raise ValueError("block must be a positive integer")
        if not 0 < minimum_temperature <= initial_temperature:
            raise ValueError("Need 0 < minimum_temperature <= initial_temperature")
        if not math.isfinite(initial_temperature) or not math.isfinite(relaxation_rate):
            raise ValueError("Temperature and relaxation rate must be finite")
        if relaxation_rate < 0:
            raise ValueError("relaxation_rate must be nonnegative")
        if not dtype.is_floating_point:
            raise ValueError("RSMI-NE requires a floating dtype")
        self.block = block
        self.visible_size = block * block
        self.initial_temperature = float(initial_temperature)
        self.minimum_temperature = float(minimum_temperature)
        self.relaxation_rate = float(relaxation_rate)
        # tf.random_normal_initializer() has default stddev=0.05.
        self.weight = nn.Parameter(0.05 * _normal(
            (self.visible_size,), generator=generator, device=device, dtype=dtype
        ))

    def temperature(self, step):
        """Explicit step avoids a Python-scalar temperature frozen inside a graph."""
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError("step must be a nonnegative integer")
        return max(
            self.minimum_temperature,
            self.initial_temperature * math.exp(-self.relaxation_rate * step),
        )

    def logits(self, visible):
        if visible.ndim < 2 or visible.shape[-1] != self.visible_size:
            raise ValueError(f"Visible blocks need final dimension {self.visible_size}")
        if visible.dtype != self.weight.dtype or visible.device != self.weight.device:
            raise ValueError("Visible blocks must match the model's device and dtype")
        return (visible * self.weight).sum(-1, keepdim=True)

    def probabilities(self, visible):
        return self.logits(visible).sigmoid()

    def forward(self, visible, *, step, generator=None):
        logits = self.logits(visible)
        uniform = _uniform(
            logits.shape, generator=generator, device=logits.device, dtype=logits.dtype
        )
        # One logistic draw equals the difference of two independent Gumbels.
        eps = torch.finfo(logits.dtype).eps
        uniform = uniform.clamp(eps, 1 - eps)
        logistic = uniform.log() - torch.log1p(-uniform)
        return torch.sigmoid((logits + logistic) / self.temperature(step))

    @torch.no_grad()
    def sample_spins(self, visible, *, generator=None):
        """Exact Bernoulli output in physical -1/+1 encoding; shape [...,1]."""
        probabilities = self.probabilities(visible)
        uniform = _uniform(
            probabilities.shape, generator=generator,
            device=probabilities.device, dtype=probabilities.dtype,
        )
        return (uniform < probabilities).to(probabilities.dtype).mul(2).sub(1)


class SeparableCritic(nn.Module):
    """Two independent MLPs, 2 x 32 ReLU hidden units, embedding dimension 8.

    scores[...,i,j] = u(E_i) dot v(H_j).  This is the transpose of the authors'
    H-by-E score matrix: our column-wise logsumexp over E is exactly their
    TensorFlow Probability row-wise logsumexp over E.  No dropout is used in
    the paper defaults.
    """

    def __init__(
        self, environment_size=24, *, hidden_dim=32, embed_dim=8, layers=2,
        generator=None, device=None, dtype=torch.float32,
    ):
        super().__init__()
        for name, value in (("environment_size", environment_size), ("hidden_dim", hidden_dim),
                            ("embed_dim", embed_dim), ("layers", layers)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.environment_size = environment_size

        def mlp(input_size):
            modules = []
            for _ in range(layers):
                modules.extend([
                    _Dense(input_size, hidden_dim, generator=generator, device=device, dtype=dtype),
                    nn.ReLU(),
                ])
                input_size = hidden_dim
            modules.append(_Dense(
                input_size, embed_dim, generator=generator, device=device, dtype=dtype
            ))
            return nn.Sequential(*modules)

        self.environment_net = mlp(environment_size)
        self.hidden_net = mlp(1)

    def forward(self, environment, hidden):
        if environment.ndim < 2 or hidden.ndim != environment.ndim:
            raise ValueError("Need matching [...,batch,feature] environment/hidden arrays")
        if environment.shape[:-1] != hidden.shape[:-1]:
            raise ValueError("Environment and hidden batch/group shapes must match")
        if environment.shape[-1] != self.environment_size or hidden.shape[-1] != 1:
            raise ValueError("Invalid environment or hidden feature dimension")
        embedding_e = self.environment_net(environment)
        embedding_h = self.hidden_net(hidden)
        return embedding_e @ embedding_h.transpose(-1, -2)


def infonce_from_scores(scores):
    """Native InfoNCE bound in nats, for [..., E replicas, H anchors].

    Returns one bound per leading group; the scalar result for an ungrouped
    square score matrix is also useful for mathematical cross-checks.
    """
    if scores.ndim < 2 or scores.shape[-1] != scores.shape[-2] or scores.shape[-1] < 2:
        raise ValueError("InfoNCE needs square score matrices with at least two replicas")
    batch = scores.shape[-1]
    denominator = torch.logsumexp(scores, dim=-2) - math.log(batch)
    return (scores.diagonal(dim1=-2, dim2=-1) - denominator).mean(-1)


@dataclass
class RSMINELoss:
    loss: torch.Tensor
    mi_estimate: torch.Tensor
    group_estimates: torch.Tensor
    group_sizes: tuple
    temperature: float
    relaxed_hidden: torch.Tensor


def _group_indices(visible, condition_ids, chain_ids):
    if visible.ndim == 3:
        if condition_ids is not None or chain_ids is not None:
            raise ValueError("For grouped inputs, supply groups directly, not redundant ids")
        if visible.shape[0] < 1 or visible.shape[1] < 2:
            raise ValueError("Each explicit group needs at least two replicas")
        return None
    if visible.ndim != 2 or condition_ids is None or chain_ids is None:
        raise ValueError("Flat inputs require integer condition_ids AND physical chain_ids")
    if len(visible) < 2:
        raise ValueError("At least two replicas are required")
    integer_dtypes = {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}
    for ids in (condition_ids, chain_ids):
        if not isinstance(ids, torch.Tensor) or ids.shape != (len(visible),):
            raise ValueError("Group ids must be tensors with shape [replicas]")
        if ids.dtype not in integer_dtypes:
            raise ValueError("Group ids must be integer identifiers, not coupling values")
    labels = torch.stack((condition_ids.to(visible.device), chain_ids.to(visible.device)), -1)
    _, inverse = torch.unique(labels, dim=0, sorted=True, return_inverse=True)
    indices = [(inverse == i).nonzero(as_tuple=True)[0] for i in range(int(inverse.max()) + 1)]
    if any(len(index) < 2 for index in indices):
        raise ValueError("Every (condition, physical chain) needs at least two replicas")
    return indices


class RSMINE(nn.Module):
    """The native coarse-grainer plus critic; no QCR regularization or prediction head."""

    def __init__(
        self, block=3, environment_size=24, *, hidden_dim=32, embed_dim=8, layers=2,
        initial_temperature=0.5, minimum_temperature=0.1, relaxation_rate=0.005,
        generator=None, device=None, dtype=torch.float32,
    ):
        super().__init__()
        self.coarse_grainer = LinearBinaryCoarseGrainer(
            block, initial_temperature=initial_temperature,
            minimum_temperature=minimum_temperature, relaxation_rate=relaxation_rate,
            generator=generator, device=device, dtype=dtype,
        )
        self.critic = SeparableCritic(
            environment_size, hidden_dim=hidden_dim, embed_dim=embed_dim, layers=layers,
            generator=generator, device=device, dtype=dtype,
        )

    def mi_loss(
        self, visible, environment, *, step, generator=None,
        condition_ids=None, chain_ids=None,
    ):
        return mi_loss(
            self, visible, environment, step=step, generator=generator,
            condition_ids=condition_ids, chain_ids=chain_ids,
        )

    def forward(self, visible, environment, **kwargs):
        return self.mi_loss(visible, environment, **kwargs)


def mi_loss(
    model, visible, environment, *, step, generator=None,
    condition_ids=None, chain_ids=None,
):
    """Negative native sampled InfoNCE, averaged equally over physical groups.

    Inputs are [groups,replicas,9]/[groups,replicas,24], with each explicit group
    ONE condition AND ONE physical chain, or flat [replicas,features] with both
    integer id arrays.  Ids only restrict negatives; they never enter a network.
    For ragged flat groups, each group has equal weight, not weight proportional
    to its number of draws.  This matched-data grouping differs from any
    original-paper use of already-independent equilibrium replicas.  In
    particular a finite, possibly correlated chain cache is not a certified
    population mutual-information estimate despite the InfoNCE functional.
    """
    if environment.shape[:-1] != visible.shape[:-1]:
        raise ValueError("Visible and environment batch/group shapes must match")
    parameter = model.coarse_grainer.weight
    if environment.device != parameter.device or environment.dtype != parameter.dtype:
        raise ValueError("Environment must match the model's device and dtype")
    if environment.shape[-1] != model.critic.environment_size:
        raise ValueError("Incorrect environment feature count")
    indices = _group_indices(visible, condition_ids, chain_ids)
    hidden = model.coarse_grainer(visible, step=step, generator=generator)
    if indices is None:
        estimates = infonce_from_scores(model.critic(environment, hidden))
        sizes = (visible.shape[1],) * visible.shape[0]
    else:
        estimates = torch.stack([
            infonce_from_scores(model.critic(environment[index], hidden[index]))
            for index in indices
        ])
        sizes = tuple(len(index) for index in indices)
    estimate = estimates.mean()
    return RSMINELoss(
        loss=-estimate, mi_estimate=estimate, group_estimates=estimates,
        group_sizes=sizes, temperature=model.coarse_grainer.temperature(step),
        relaxed_hidden=hidden,
    )
