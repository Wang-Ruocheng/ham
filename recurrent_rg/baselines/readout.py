"""Label-free common readouts for the data-aligned baseline comparison.

These are explicit comparison adapters, not the Q/R components of the original
RSMI or Neural-MCRG papers.  No labels, checkpoints, data files or samplers are
read here.  Fits require a caller-declared TRAIN split and the same fitted
coordinate map is subsequently reused at every scale and on TEST.

There are deliberately TWO motif orderings.  Q's centered 3x3 binary lookup
uses lexicographic offsets with the first offset as the least significant bit.
The existing affine decoder targets ``plaquette_distribution``: anchored 2x2
offsets (00,10,01,11) with weights (8,4,2,1).  They must not be interchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn


Q_MOTIF_ORDERING = "centered_3x3_lexicographic_offsets_lsb_first_v1"
TARGET_MOTIF_ORDERING = "anchored_2x2_offsets_00_10_01_11_msb_first_v1"


def _array(value, name="array"):
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.requires_grad:
            raise ValueError(f"{name} must be detached CPU data, not a model graph")
        value = value.detach().numpy()
    result = np.asarray(value)
    if result.dtype.kind not in "iuf" or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite real numbers")
    return result


def validate_spins(spins):
    """Validate +/-1 fields ``[...,height,width]`` without changing the input.

    The 1e-12 tolerance only admits roundoff in a hard straight-through sample;
    it does not turn relaxed fields or zero/one bits into observed spins.
    """
    values = _array(spins, "spins")
    if values.ndim < 2 or any(length < 1 for length in values.shape):
        raise ValueError("Need nonempty periodic two-dimensional spin fields")
    if not np.all(np.abs(np.abs(values) - 1) <= 1e-12):
        raise ValueError("Spins must be binary +/-1 up to 1e-12 roundoff")
    return values


def fixed_coarsen(spins, rule, *, block=3):
    """Nonoverlapping top-left decimation or odd-block majority, no RNG.

    Arbitrary leading axes are retained.  A CPU tensor input returns a CPU
    tensor of its original dtype; a NumPy input returns a NumPy array.  Inputs
    are never modified.  Majority with an even block is rejected rather than
    silently defining a tie rule.
    """
    values = validate_spins(spins)
    if type(block) is not int or block < 2:
        raise ValueError("block must be an integer >=2")
    if any(length % block for length in values.shape[-2:]):
        raise ValueError("Both spatial sizes must be divisible by the block")
    if rule == "decimation":
        result = np.where(values[..., ::block, ::block] > 0, 1, -1)
    elif rule == "majority":
        if block % 2 != 1:
            raise ValueError("Majority requires an odd block; no hidden tie rule")
        positive = np.zeros(values[..., ::block, ::block].shape, dtype=np.int64)
        for row in range(block):
            for column in range(block):
                positive += values[..., row::block, column::block] > 0
        result = np.where(2 * positive > block * block, 1, -1)
    else:
        raise ValueError("rule must be decimation or majority")
    result = np.ascontiguousarray(result, dtype=values.dtype)
    if isinstance(spins, torch.Tensor):
        return torch.from_numpy(result)
    return result


def motif_probabilities(spins, width=3, *, sample_axis=-3):
    """Per-ensemble periodic motif frequencies, never a site-by-bin one-hot.

    Standard input is ``[...ensemble_axes,samples,H,W]`` and output is
    ``[...ensemble_axes,512]`` (width3) or ``[...,16]`` (width2).  The sample
    axis and both spatial axes are averaged, never different ensembles or C
    repeats.  ``sample_axis`` can move the sample axis within the leading
    nonspatial axes.  Both orderings match the existing Q/decoder exactly.
    """
    values = validate_spins(spins)
    if values.ndim < 3 or type(sample_axis) is not int:
        raise ValueError("Need an explicit sample axis and two spatial axes")
    if not -values.ndim <= sample_axis < values.ndim:
        raise ValueError("sample_axis is out of range")
    axis = sample_axis % values.ndim
    if axis >= values.ndim - 2:
        raise ValueError("sample_axis cannot be a spatial axis")
    values = np.moveaxis(values, axis, -3)
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
    bits = values > 0
    indices = np.zeros(values.shape, dtype=np.uint16)
    for row, column, weight in offsets:
        indices += np.roll(bits, (-row, -column), axis=(-2, -1)).astype(np.uint16) * weight
    flat = indices.reshape(rows, -1)
    counts = np.empty((rows, bins), dtype=np.float64)
    for index, row in enumerate(flat):
        counts[index] = np.bincount(row, minlength=bins)
    return (counts / flat.shape[1]).reshape(*prefix, bins)


def _finite_matrix(value, name, *, minimum_rows=1):
    result = _array(value, name).astype(np.float64, copy=False)
    if result.ndim < 2 or not result.shape[-1] or any(n < 1 for n in result.shape):
        raise ValueError(f"{name} needs nonempty rows and a feature axis")
    rows = result.reshape(-1, result.shape[-1])
    if len(rows) < minimum_rows:
        raise ValueError(f"{name} has too few fitting rows")
    return result, rows


def _train_only(split):
    if split != "train":
        raise ValueError("Fit is TRAIN-only; TEST must only use the frozen transform")


def _rcond(value):
    if isinstance(value, bool) or not math.isfinite(value) or not 0 < value < 1:
        raise ValueError("rcond must be finite and strictly between zero and one")
    return float(value)


def _readonly(value):
    result = np.array(value, dtype=np.float64, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class PCAReadout:
    """One frozen center/basis shared over scales; rank is not forced full."""

    center: np.ndarray
    basis: np.ndarray
    singular_values: np.ndarray
    rank: int
    sample_count: int
    train_shape: tuple
    rcond: float
    weak_relative_threshold: float
    warnings: tuple = ()

    def __post_init__(self):
        for name in ("center", "basis", "singular_values"):
            object.__setattr__(self, name, _readonly(_array(getattr(self, name), name)))
        if (self.center.shape != (512,) or self.basis.ndim != 2
                or self.basis.shape[0] != 512 or not self.basis.shape[1]
                or self.singular_values.ndim != 1 or not len(self.singular_values)
                or np.any(self.singular_values < 0)
                or np.any(np.diff(self.singular_values) > 1e-10)
                or type(self.rank) is not int or not 0 <= self.rank <= len(self.singular_values)
                or type(self.sample_count) is not int or self.sample_count < 2):
            raise ValueError("Invalid frozen PCA shapes, spectrum, rank or row count")
        if len(self.singular_values) < self.basis.shape[1]:
            raise ValueError("PCA cannot invent unobserved basis directions")
        np.testing.assert_allclose(self.basis.T @ self.basis, np.eye(self.basis.shape[1]),
                                   atol=1e-10, rtol=1e-10)
        _rcond(self.rcond)
        if not math.isfinite(self.weak_relative_threshold) or not 0 < self.weak_relative_threshold < 1:
            raise ValueError("Invalid weak-direction warning threshold")
        expected_rank = int(np.sum(self.singular_values > self.rcond * self.singular_values[0]))
        if self.rank != expected_rank or math.prod(self.train_shape[:-1]) != self.sample_count:
            raise ValueError("PCA rank or original TRAIN shape is inconsistent")
        if not self.train_shape or self.train_shape[-1] != 512:
            raise ValueError("PCA TRAIN shape must record all 512 motif frequencies")
        object.__setattr__(self, "train_shape", tuple(self.train_shape))
        object.__setattr__(self, "warnings", tuple(self.warnings))

    @property
    def dimension(self):
        return self.basis.shape[1]

    def transform(self, frequencies):
        value = _array(frequencies, "frequencies")
        if value.ndim < 1 or value.shape[-1] != len(self.center):
            raise ValueError("PCA transform needs the original 512-frequency axis")
        return (value - self.center) @ self.basis

    def inverse_transform(self, coordinates):
        value = _array(coordinates, "coordinates")
        if value.ndim < 1 or value.shape[-1] != self.dimension:
            raise ValueError("PCA inverse needs its original coordinate dimension")
        return value @ self.basis.T + self.center

    def to_dict(self):
        power = self.singular_values ** 2
        return dict(kind="TRAIN_shared_3x3_PCA_v1", fit_split="train", labels_used=False,
            motif_ordering=Q_MOTIF_ORDERING, center=self.center.tolist(), basis=self.basis.tolist(),
            singular_values=self.singular_values.tolist(), rank=self.rank,
            dimension=self.dimension, sample_count=self.sample_count,
            train_shape=list(self.train_shape), rcond=self.rcond,
            weak_relative_threshold=self.weak_relative_threshold, warnings=list(self.warnings),
            explained_variance_ratio=(power / power.sum()).tolist() if power.sum() > 0 else [0.] * len(power),
            sign_convention="Largest-absolute loading in each column positive; first index breaks ties",
            degeneracy_caution="Repeated singular subspaces have no uniquely physical orientation",
            scale_policy="One joint TRAIN center and basis; no per-scale or TEST refit")

    @classmethod
    def from_dict(cls, value):
        if (value.get("kind") != "TRAIN_shared_3x3_PCA_v1" or value.get("fit_split") != "train"
                or value.get("labels_used") is not False or value.get("motif_ordering") != Q_MOTIF_ORDERING):
            raise ValueError("Unknown or non-TRAIN PCA receipt")
        result = cls(center=value["center"], basis=value["basis"],
            singular_values=value["singular_values"], rank=value["rank"],
            sample_count=value["sample_count"], train_shape=tuple(value["train_shape"]),
            rcond=value["rcond"], weak_relative_threshold=value["weak_relative_threshold"],
            warnings=tuple(value.get("warnings", ())))
        if result.dimension != value.get("dimension"):
            raise ValueError("PCA receipt dimension differs from the saved basis")
        return result


def fit_train_pca(frequencies, dimension=5, *, split="train", rcond=1e-12,
                  weak_relative_threshold=1e-3):
    """Fit once to every supplied TRAIN ensemble/scale row, weighted equally.

    Input ``[TRAIN ensembles,scales,512]`` jointly fits all scales, rather than
    fitting incompatible axes separately.  Additional C draws, if supplied,
    remain conditional fitting rows and are not independent physical samples.
    """
    _train_only(split)
    original, rows = _finite_matrix(frequencies, "TRAIN frequencies", minimum_rows=2)
    rcond = _rcond(rcond)
    if (rows.shape[-1] != 512 or type(dimension) is not int or
            not 1 <= dimension <= min(rows.shape)):
        raise ValueError("Need 512-frequency rows and an available positive PCA dimension")
    if (np.any(rows < -1e-12) or np.any(rows > 1 + 1e-12)
            or not np.allclose(rows.sum(-1), 1., rtol=1e-10, atol=1e-10)):
        raise ValueError("PCA input must be actual normalized motif frequencies")
    if (isinstance(weak_relative_threshold, bool) or not math.isfinite(weak_relative_threshold)
            or not 0 < weak_relative_threshold < 1):
        raise ValueError("Need a finite weak-direction threshold between zero and one")
    center = rows.mean(0)
    _, singular, vt = np.linalg.svd(rows - center, full_matrices=False)
    basis = vt[:dimension].T.copy()
    for column in range(dimension):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0:
            basis[:, column] *= -1
    rank = int(np.sum(singular > rcond * singular[0]))
    ratios = singular[:dimension] / singular[0] if singular[0] > 0 else np.zeros(dimension)
    warnings = []
    if rank < dimension:
        warnings.append("TRAIN rank is below the requested dimension; null directions are not physical evidence")
    if np.any(ratios < weak_relative_threshold):
        warnings.append("At least one retained direction is weak relative to TRAIN PC1")
    if np.any(np.isclose(np.diff(singular[:dimension]), 0., atol=1e-14, rtol=0.)):
        warnings.append("Repeated singular values: sign fixing does not identify a unique subspace orientation")
    return PCAReadout(center, basis, singular, rank, len(rows), tuple(original.shape),
                      rcond, float(weak_relative_threshold), tuple(warnings))


@dataclass(frozen=True)
class AffineReadout:
    """Frozen TRAIN-only 16-motif readout; raw outputs are never clipped."""

    weight: np.ndarray
    bias: np.ndarray
    singular_values: np.ndarray
    rank: int
    sample_count: int
    ridge: float
    rcond: float

    def __post_init__(self):
        for name in ("weight", "bias", "singular_values"):
            object.__setattr__(self, name, _readonly(_array(getattr(self, name), name)))
        if (self.weight.ndim != 2 or not self.weight.shape[0] or self.weight.shape[1] != 16
                or self.bias.shape != (16,) or self.singular_values.ndim != 1
                or not len(self.singular_values) or np.any(self.singular_values < 0)
                or type(self.rank) is not int or not 0 <= self.rank <= len(self.singular_values)
                or type(self.sample_count) is not int or self.sample_count < 1
                or isinstance(self.ridge, bool) or not math.isfinite(self.ridge) or self.ridge < 0):
            raise ValueError("Invalid affine readout shapes, rank or regularization")
        _rcond(self.rcond)
        expected_rank = int(np.sum(self.singular_values > self.rcond * self.singular_values[0]))
        if expected_rank != self.rank:
            raise ValueError("Affine rank differs from the saved fitting spectrum")

    def predict(self, coordinates):
        value = _array(coordinates, "coordinates")
        if value.ndim < 1 or value.shape[-1] != self.weight.shape[0]:
            raise ValueError("Affine readout coordinate dimension differs")
        return value @ self.weight + self.bias

    __call__ = predict

    def to_dict(self):
        return dict(kind="TRAIN_affine_16_motif_readout_v1", fit_split="train", labels_used=False,
            motif_ordering=TARGET_MOTIF_ORDERING, weight=self.weight.tolist(), bias=self.bias.tolist(),
            singular_values=self.singular_values.tolist(), rank=self.rank, sample_count=self.sample_count,
            ridge=self.ridge, rcond=self.rcond,
            objective="sum_i ||y_i-(g_i W+b)||^2 + ridge*||W||_F^2; intercept unpenalized; SVD cutoff explicit",
            clipped=False, renormalized=False)

    @classmethod
    def from_dict(cls, value):
        if (value.get("kind") != "TRAIN_affine_16_motif_readout_v1" or value.get("fit_split") != "train"
                or value.get("labels_used") is not False
                or value.get("motif_ordering") != TARGET_MOTIF_ORDERING
                or value.get("clipped") is not False or value.get("renormalized") is not False):
            raise ValueError("Unknown, non-TRAIN or postprocessed affine readout")
        return cls(value["weight"], value["bias"], value["singular_values"],
                   value["rank"], value["sample_count"], value["ridge"], value["rcond"])


def fit_affine_readout(coordinates, target, *, ridge=0., rcond=1e-12, split="train"):
    """Fit a shared 16-motif readout using truncated-SVD minimum norm/ridge.

    ``ridge=0`` is the deterministic minimum-norm least-squares solution.
    Positive ridge penalizes slopes, not the intercept, in the explicitly
    recorded SUM-of-squares convention.  No TEST/model/label data is accessed.
    """
    _train_only(split)
    original_x, x = _finite_matrix(coordinates, "TRAIN coordinates")
    original_y, y = _finite_matrix(target, "TRAIN target")
    rcond = _rcond(rcond)
    if original_x.shape[:-1] != original_y.shape[:-1] or y.shape[1] != 16:
        raise ValueError("Coordinates/16-motif targets need identical fitting row axes")
    if (isinstance(ridge, bool) or not math.isfinite(ridge) or ridge < 0):
        raise ValueError("ridge must be a finite nonnegative number")
    if (np.any(y < -1e-12) or np.any(y > 1 + 1e-12)
            or not np.allclose(y.sum(-1), 1., rtol=1e-10, atol=1e-10)):
        raise ValueError("Affine targets must be normalized observed motif probabilities")
    x_center, y_center = x.mean(0), y.mean(0)
    u, singular, vt = np.linalg.svd(x - x_center, full_matrices=False)
    keep = singular > rcond * singular[0]
    factor = np.zeros_like(singular)
    if ridge:
        factor[keep] = singular[keep] / (singular[keep] ** 2 + ridge)
    else:
        factor[keep] = 1 / singular[keep]
    weight = (vt.T * factor) @ (u.T @ (y - y_center))
    bias = y_center - x_center @ weight
    return AffineReadout(weight, bias, singular, int(keep.sum()), len(x), float(ridge), rcond)


class ResidualFlow(nn.Module):
    """The shared comparison R(g)=g+MLP(g), matching current R architecture.

    This is an amortized common-readout adapter, NOT a claim that original
    RSMI/Neural-MCRG papers trained such an R.  Independent construction uses
    a different RNG consumption order from constructing an entire Q/C/R.
    """

    def __init__(self, dimension=5, hidden=96):
        super().__init__()
        if type(dimension) is not int or dimension < 1 or type(hidden) is not int or hidden < 1:
            raise ValueError("Positive integer latent dimension and hidden width required")
        self.theory_dim = dimension
        self.theory_rg = nn.Sequential(nn.Linear(dimension, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, dimension)).double()

    def forward(self, coordinates):
        return coordinates + self.theory_rg(coordinates)

    def advance_theory(self, coordinates):
        return self(coordinates)


def build_r(dimension=5, hidden=96, *, seed=None):
    """Build CPU float64 R; an explicit seed does not consume caller RNG."""
    if seed is None:
        return ResidualFlow(dimension, hidden)
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return ResidualFlow(dimension, hidden)


def _draw_axes(value, name, c_axis):
    array = _array(value, name).astype(np.float64, copy=False)
    if c_axis is None:
        if array.ndim != 3:
            raise ValueError(f"{name} needs [physical_chain,scale,coordinate]")
        array = array[:, None]
    elif array.ndim == 4 and type(c_axis) is int and c_axis % 4 not in (0, 3):
        array = np.moveaxis(array, c_axis, 1)
    else:
        raise ValueError(f"{name} needs [physical_chain,C_draw,scale,coordinate]")
    if any(length < 1 for length in array.shape):
        raise ValueError(f"{name} cannot contain an empty axis")
    return array


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 1e-12 else None


def path_metrics(direct, r_one_step, rollout, reference_train, *, c_axis=1):
    """Latent paths and identity with C errors averaged within physical chains.

    Direct/rollout are [chain,C,scale,coordinate]; r_one_step may omit its unused
    final scale.  Report all adjacent paths and microscopic-start rollouts
    (exactly all three paths for the planned 108->36->12 experiment).

    ``reference_train`` must be the explicitly selected base-C TRAIN encoding
    [chain,scale,coordinate], as in the existing five-seed report.  We never
    silently replace its original normalization by a C-averaged reference.
    These metrics compare R vs identity WITHIN one coordinate system, not a
    coordinate-invariant ranking of different Q/C methods.
    """
    observed = _draw_axes(direct, "direct", c_axis)
    one = _draw_axes(r_one_step, "r_one_step", c_axis)
    recurrent = _draw_axes(rollout, "rollout", c_axis)
    reference = _array(reference_train, "reference_train").astype(np.float64, copy=False)
    n, draws, scales, dimension = observed.shape
    if (scales < 2 or recurrent.shape != observed.shape
            or one.shape not in (observed.shape, (n, draws, scales - 1, dimension))
            or reference.ndim != 3 or reference.shape[1:] != (scales, dimension)
            or len(reference) < 2):
        raise ValueError("Path arrays or base-C TRAIN normalization have incompatible shapes")
    if not np.allclose(recurrent[:, :, 0], observed[:, :, 0], rtol=1e-10, atol=1e-12):
        raise ValueError("Microscopic rollout must start at the actual direct g0")
    sigma = reference.std(axis=0, ddof=0)
    paths = sorted(set([(scale, scale + 1) for scale in range(scales - 1)]
                       + [(0, scale) for scale in range(1, scales)]))
    results = []
    for start, end in paths:
        prediction = one[:, :, start] if end == start + 1 else recurrent[:, :, end]
        target = observed[:, :, end]
        row = dict(start=start, end=end,
            target_train_rms_coordinate_std=float(np.sqrt(np.mean(sigma[end] ** 2))))
        for method, predicted in (("R", prediction), ("identity", observed[:, :, start])):
            per_chain_coordinate_mse = ((predicted - target) ** 2).mean(axis=1)
            coordinate_rmse = np.sqrt(per_chain_coordinate_mse.mean(axis=0))
            rmse = float(np.sqrt(per_chain_coordinate_mse.mean()))
            row[method] = dict(rmse=rmse,
                nrmse=_ratio(rmse, row["target_train_rms_coordinate_std"]),
                coordinate_rmse=coordinate_rmse.tolist(),
                coordinate_nrmse=[_ratio(x, y) for x, y in zip(coordinate_rmse, sigma[end])],
                per_physical_chain_mse=per_chain_coordinate_mse.mean(-1).tolist())
        row["R_to_identity_rmse_ratio"] = _ratio(row["R"]["rmse"], row["identity"]["rmse"])
        results.append(row)
    return dict(paths=results, physical_chain_count=n, C_draw_count=draws,
        independent_sample_count=n, TRAIN_target_coordinate_std=sigma.tolist(),
        normalization_source="Explicit base-C TRAIN encoding; population std ddof0",
        aggregation="Within each physical chain average C-draw squared errors, then average chains/coordinates and take sqrt",
        path_policy="Adjacent steps plus microscopic-start all-horizon rollout; other nonmicroscopic multistep paths not invented",
        comparison_caution="Latent metrics are only within this Q/C; different coordinates and coarse grainers are not a common physical target")


def physical_error_decomposition(observed, decoded_direct, decoded_prediction, *,
                                 c_axis=1, average_C_before_error=False):
    """Separate common-motif reconstruction and R transport including cross term.

    Default: compute errors for each C draw, average within each physical chain,
    then average physical chains.  ``average_C_before_error=True`` instead
    evaluates predictions of conditional C means and is explicitly labeled;
    it is not the same metric.  No probabilities are clipped/renormalized.
    """
    values = [_array(value, name).astype(np.float64, copy=False) for value, name in zip(
        (observed, decoded_direct, decoded_prediction), ("observed", "decoded_direct", "decoded_prediction"))]
    if (any(value.shape != values[0].shape for value in values)
            or values[0].ndim < 2 or any(length < 1 for length in values[0].shape)):
        raise ValueError("Physical statistic arrays need identical nonempty shapes")
    if type(average_C_before_error) is not bool:
        raise ValueError("average_C_before_error must be an explicit boolean")
    if c_axis is not None:
        if (type(c_axis) is not int or not -values[0].ndim <= c_axis < values[0].ndim
                or c_axis % values[0].ndim in (0, values[0].ndim - 1)):
            raise ValueError("C axis must be distinct from physical chain and statistic axes")
        values = [np.moveaxis(value, c_axis, 1) for value in values]
        if average_C_before_error:
            values = [value.mean(axis=1) for value in values]
    elif average_C_before_error:
        raise ValueError("Cannot average a nonexistent C axis")
    observed, decoded_direct, decoded_prediction = values
    reconstruction = decoded_direct - observed
    transport = decoded_prediction - decoded_direct
    total = decoded_prediction - observed
    average_axes = tuple(range(1, total.ndim))
    per_chain = dict(total_mse=(total ** 2).mean(axis=average_axes),
        readout_mse=(reconstruction ** 2).mean(axis=average_axes),
        R_transport_mse=(transport ** 2).mean(axis=average_axes),
        cross_term=(2 * reconstruction * transport).mean(axis=average_axes))
    results = {key: float(value.mean()) for key, value in per_chain.items()}
    for name in ("total", "readout", "R_transport"):
        results[name + "_rmse"] = float(np.sqrt(results[name + "_mse"]))
    if not np.isclose(results["total_mse"], results["readout_mse"] + results["R_transport_mse"]
                      + results["cross_term"], rtol=1e-10, atol=1e-14):
        raise ArithmeticError("Physical error decomposition identity failed")
    results.update(per_physical_chain={key: value.tolist() for key, value in per_chain.items()},
        physical_chain_count=len(total), independent_sample_count=len(total),
        C_averaged_before_error=average_C_before_error,
        aggregation=("C means before squaring; then equal physical-chain weights" if average_C_before_error
                     else "Squared errors per C draw, conditional C average within each chain, equal chain weights"),
        clipped=False, renormalized=False)
    return results
