"""Single-factor follow-ups to the explicit continuous-field b=2 experiment.

This module does not change historical models or the standard b=3 recipe.
``cubic_decoder`` changes only Q's decoder, and ``odd_C`` changes only C's
mean function. The experiments still use the historical continuous-field
Gaussian information proxy, not exact empirical Shannon mutual information.
The objectives themselves are reused unchanged from :mod:`phi4_b2_qcr`.
"""
from __future__ import annotations

import hashlib

import torch
from torch import nn
from torch.nn import functional as F

from recurrent_rg.phi4_b2_qcr import (
    GaussianCoarseGrainerB2,
    Phi4B2QCR,
    continuous_objectives_b2,
)
from recurrent_rg.phi4_qcr import FEATURE_DIM, ContinuousTheoryEncoder


VARIANTS = ("baseline", "cubic_decoder", "odd_C")


class CubicTheoryEncoder(ContinuousTheoryEncoder):
    """An affine decoder plus zero-initialized quadratic and cubic terms.

    The original g coordinate and normalized linear decoder are retained:
    ``decode(g) = W g + b + U2 g^2 + U3 g^3``. Neither high-order coefficient
    is normalized, and there is no PCA, coordinate rescaling, or TEST-derived
    initialization. For this explicit one-dimensional control, U2 and U3
    each have shape [105, 1], adding exactly 210 trainable parameters.
    """

    def __init__(self, hidden: int = 96, theory_dim: int = 1):
        if theory_dim != 1:
            raise ValueError("The cubic-decoder validation requires theory_dim=1")
        super().__init__(hidden=hidden, theory_dim=theory_dim)
        self.quadratic_weight = nn.Parameter(torch.zeros(FEATURE_DIM, 1, dtype=torch.double))
        self.cubic_weight = nn.Parameter(torch.zeros(FEATURE_DIM, 1, dtype=torch.double))

    def decode(self, g: torch.Tensor) -> torch.Tensor:
        return (super().decode(g)
                + F.linear(g.square(), self.quadratic_weight)
                + F.linear(g.pow(3), self.cubic_weight))


class OddGaussianCoarseGrainerB2(GaussianCoarseGrainerB2):
    """Exactly odd conditional mean with unchanged Gaussian output noise.

    For an unconstrained randomly initialized base net f,
    ``m(v) = (f(v) - f(-v))/2``. This guarantees Z2 equivariance in distribution
    because the zero-mean fixed Gaussian noise is symmetric. Individual draws
    are antithetic, not identically seeded: C(-x, -epsilon) = -C(x, epsilon).
    No ensemble-dependent amplitude normalization is performed.
    """

    def mean_blocks(self, blocks: torch.Tensor) -> torch.Tensor:
        positive = super().mean_blocks(blocks)
        negative = super().mean_blocks(-blocks)
        return .5 * (positive - negative)


def module_fingerprint(module: nn.Module) -> str:
    """Read-only deterministic SHA of named state, including all buffers."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class ValidationModel(Phi4B2QCR):
    """Construct the historical fresh model, then change one declared factor.

    Replacement constructors run inside a CPU RNG fork and load the exact
    common state just created by ``Phi4B2QCR``. Thus same-seed C/Q-net/R
    initial parameters and the global RNG stream match the baseline. There
    is no pretrained/selected C or reuse of a trained source checkpoint.
    The odd-C variant intentionally changes C's initial *function* while
    preserving every C net parameter; the cubic variant initially leaves
    the complete model function unchanged.
    """

    def __init__(self, scale: float, hidden: int = 96, theory_dim: int = 1,
                 variant: str = "baseline"):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}; expected one of {VARIANTS}")
        if theory_dim != 1:
            raise ValueError("This validation is explicitly restricted to theory_dim=1")
        super().__init__(scale=scale, hidden=hidden, theory_dim=theory_dim)
        self.variant = variant
        original = {
            "C": module_fingerprint(self.C),
            "Q_net": module_fingerprint(self.Q.net),
            "Q_affine_decoder": module_fingerprint(self.Q.decoder),
            "R": module_fingerprint(self.R),
        }
        if variant == "cubic_decoder":
            with torch.random.fork_rng(devices=[]):
                replacement = CubicTheoryEncoder(hidden=hidden, theory_dim=theory_dim)
            receipt = replacement.load_state_dict(self.Q.state_dict(), strict=False)
            if set(receipt.missing_keys) != {"quadratic_weight", "cubic_weight"} or receipt.unexpected_keys:
                raise RuntimeError("Cubic decoder did not preserve exactly the common Q state")
            self.Q = replacement
        elif variant == "odd_C":
            with torch.random.fork_rng(devices=[]):
                replacement = OddGaussianCoarseGrainerB2(
                    scale=scale, hidden=hidden, noise_ratio=float(self.C.noise_ratio))
            replacement.load_state_dict(self.C.state_dict(), strict=True)
            self.C = replacement
        after = {
            "C": module_fingerprint(self.C),
            "Q_net": module_fingerprint(self.Q.net),
            "Q_affine_decoder": module_fingerprint(self.Q.decoder),
            "R": module_fingerprint(self.R),
        }
        if original != after:
            raise RuntimeError("Validation replacement altered common initial parameters")
        self.initial_common_sha256 = original

    def validation_receipt(self) -> dict:
        """JSON-safe scientific contract; not an equilibrium or RG certificate."""
        return {
            "variant": self.variant,
            "single_factor": True,
            "block_size": 2,
            "stride": 2,
            "theory_dim": 1,
            "closure_objective": "all_start_anchor",
            "closure_scale_weights": None,
            "gradient_routing": "Q representation only; C its own objective only; R closure only",
            "Q_closure_feedback": False,
            "C_selected_or_pretrained": False,
            "source_checkpoint_reused": False,
            "same_seed_common_initial_parameters": True,
            "replacement_construction_preserves_global_rng": True,
            "initial_common_sha256": dict(self.initial_common_sha256),
            "decoder": ("affine_plus_raw_g_quadratic_and_cubic"
                        if self.variant == "cubic_decoder" else "shared_affine"),
            "extra_decoder_parameters": 2 * FEATURE_DIM if self.variant == "cubic_decoder" else 0,
            "high_order_coefficients_initially_zero": self.variant == "cubic_decoder",
            "only_linear_decoder_columns_normalized": True,
            "decoder_coordinate_rescaled": False,
            "decoder_TEST_or_PCA_initialization": False,
            "C_mean_exactly_odd": self.variant == "odd_C",
            "C_noise_ratio": float(self.C.noise_ratio.detach().cpu()),
            "C_noise_distribution": "fixed_zero_mean_Gaussian",
            "C_per_ensemble_amplitude_normalization": False,
            "information_objective": "historical_full_buffered_20_site_Gaussian_information_proxy",
            "exact_Shannon_MI": False,
            "labels_used_for_training": False,
            "default_recipe_modified": False,
        }


__all__ = [
    "VARIANTS", "CubicTheoryEncoder", "OddGaussianCoarseGrainerB2",
    "ValidationModel", "continuous_objectives_b2", "module_fingerprint",
]
