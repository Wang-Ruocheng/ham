"""Independent finite-lattice Gaussian scalar fields, without Markov chains.

For S=.5*r*sum(phi**2)+.5*sum_(x,a)(phi[x+e_a]-phi[x])**2,
the periodic precision spectrum is r+4*sum_a sin(pi*n_a/L)**2.
Real iid white noise is filtered using orthonormal FFTs.  The zero mode is
retained; no sample/condition normalization, zero-mean projection, or clipping
is applied.  Couplings and exact-reference diagnostics never enter data.npz.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import operator
from pathlib import Path
import time

import numpy as np


SCHEMA = "euclidean_gaussian_fft_v1"
SAMPLE_SEMANTICS = "independent_iid_groups_not_markov_chains"
QUALIFICATION = (
    "二维周期格点自由实标量场，u=h=0、r>0；实独立标准高斯白噪声经正交FFT精确线性滤波。"
    "exact finite-lattice sampling up to floating-point arithmetic and pseudorandom RNG；"
    "保留零模，不逐构型或逐条件归一化，不做Z2增强。"
    "独立样本仍有有限样本Monte Carlo不确定性；解析有限格点参考不是连续极限或临界指数认证。"
    "chains只是兼容旧加载器的字段名，含义为独立iid样本组，不是HMC链、热冷初态或热化诊断。"
    "r/u/h只用于生成与评价；构型NPZ仅含configurations，不含训练监督标签。"
)


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not bool")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(result)


def _positive_r(value):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or np.iscomplexobj(value):
        raise ValueError("r must be a finite positive real scalar")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("r must be finite and >0; the periodic r=0 zero mode is not normalizable")
    return value


def lattice_spectrum(size, r, dimension=2):
    """Precision eigenvalues in NumPy FFT order, including q=0 with value r."""
    size = _integer(size, "size", 3)
    dimension = _integer(dimension, "dimension", 1)
    if dimension != 2:
        raise ValueError("This independently scoped Gaussian pipeline supports dimension=2 only")
    r = _positive_r(r)
    axis = 4. * np.sin(np.pi * np.arange(size, dtype=np.float64) / size) ** 2
    return r + axis[:, None] + axis[None, :]


def exact_covariance(size, r):
    """C[d]=E(phi[x]*phi[x+d]), with ordinary inverse-DFT normalization 1/V."""
    return np.fft.ifftn(1. / lattice_spectrum(size, r)).real


def filter_white_noise(white, r):
    """Filter [...,L,L] real white noise without modifying or centering it."""
    if np.iscomplexobj(white):
        raise ValueError("White noise must be real")
    white = np.asarray(white, dtype=np.float64)
    if white.ndim < 2 or white.shape[-1] != white.shape[-2] or not np.isfinite(white).all():
        raise ValueError("Require finite [...,L,L] real white noise on a square torus")
    spectrum = lattice_spectrum(white.shape[-1], r)
    transformed = np.fft.fftn(white, axes=(-2, -1), norm="ortho")
    result = np.fft.ifftn(transformed / np.sqrt(spectrum), axes=(-2, -1), norm="ortho")
    # The multiplier is even in momentum, so conjugate symmetry preserves reality.
    tolerance = 128 * np.finfo(np.float64).eps * max(1., float(np.max(np.abs(result.real))))
    if float(np.max(np.abs(result.imag))) > tolerance:
        raise FloatingPointError("Gaussian FFT filtering failed the real-field conjugacy check")
    return np.ascontiguousarray(result.real)


def sample_gaussian_fields(r, *, size, samples, seed):
    """Return float64 [samples,L,L,1] independent draws with the zero mode intact."""
    size = _integer(size, "size", 3)
    samples = _integer(samples, "samples", 1)
    seed = _integer(seed, "seed", 0)
    _positive_r(r)
    white = np.random.default_rng(seed).standard_normal((samples, size, size))
    return filter_white_noise(white, r)[..., None]


@dataclass(frozen=True)
class GaussianScalarOptions:
    sizes: tuple[int, ...] = (36,)
    dimension: int = 2
    r_values: tuple[float, ...] = tuple(float(x) for x in np.geomspace(.02, 2., 12))
    groups: int = 4
    samples: int = 128
    seed: int = 91
    test_every: int = 3
    test_offset: int = 1
    max_output_bytes: int = 1024 ** 3
    max_memory_bytes: int = 256 * 1024 ** 2

    def validate(self):
        for name, minimum in (("dimension", 2), ("groups", 1), ("samples", 4), ("seed", 0),
                ("test_every", 2), ("test_offset", 0), ("max_output_bytes", 1),
                ("max_memory_bytes", 1)):
            _integer(getattr(self, name), name, minimum)
        if self.dimension != 2:
            raise ValueError("This pipeline requires dimension=2")
        if self.samples % 2:
            raise ValueError("samples must be even for disjoint saved-sample-half evaluation")
        if not isinstance(self.sizes, (tuple, list)) or not self.sizes:
            raise ValueError("sizes must be a nonempty sequence")
        for size in self.sizes:
            _integer(size, "size", 3)
        if len(set(self.sizes)) != len(self.sizes):
            raise ValueError("Duplicate sizes")
        if not isinstance(self.r_values, (tuple, list)) or not self.r_values:
            raise ValueError("r_values must be a nonempty sequence")
        values = [_positive_r(r) for r in self.r_values]
        if len(set(values)) != len(values):
            raise ValueError("Duplicate r_values")
        if self.test_offset >= self.test_every:
            raise ValueError("test_offset must be smaller than test_every")
        splits = {"test" if i % self.test_every == self.test_offset else "train"
                  for i in range(len(values))}
        if splits != {"train", "test"}:
            raise ValueError("Whole-r-cell split must contain both train and test cells")
        if self.max_output_bytes > 1024 ** 3:
            raise ValueError("This small calibration is capped at <=1 GiB output")
        return self

    def manifest_options(self):
        # chains is an explicit loader-compatibility alias, never a physical claim.
        return dict(asdict(self), chains=self.groups, u_values=[0.], h=0.,
                    chains_field_semantics=SAMPLE_SEMANTICS)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def group_seed(options, size, cell, group):
    """Stable separated streams, keyed by physical r rather than its array index."""
    message = [SCHEMA, int(options.seed), 2, int(size), float(options.r_values[cell]), int(group)]
    return int.from_bytes(hashlib.sha256(json.dumps(message).encode()).digest()[:16], "little")


def generation_plan(options):
    """Pure preflight: no RNG draws, output creation, or FFT allocation."""
    options.validate()
    cells = len(options.r_values)
    volumes = [size ** 2 for size in options.sizes]
    data_bytes = cells * options.groups * options.samples * sum(volumes) * 8
    # Uncompressed NPZ arrays plus public covariance/structure-factor JSON arrays.
    output_bytes = data_bytes + cells * sum(512 * volume + 131072 for volume in volumes) + 262144
    # One complete condition + one-group FFT workspaces + diagnostics; never all cells.
    working_bytes = (options.groups * options.samples * max(volumes) * 8
                     + options.samples * max(volumes) * 128 + max(volumes) * 256 + 8 * 1024 ** 2)
    for value, cap, name in ((output_bytes, options.max_output_bytes, "output"),
                             (working_bytes, options.max_memory_bytes, "working memory")):
        if value > cap:
            raise ValueError(f"Projected {name} byte budget exceeded: {value} > {cap}")
    return dict(schema=SCHEMA, sample_semantics=SAMPLE_SEMANTICS,
        options=options.manifest_options(), action="S=.5*r*sum(phi^2)+.5*sum_(x,a)(phi[x+e_a]-phi[x])^2",
        sampler="real iid Gaussian white noise, orthonormal FFT, divide by sqrt(r+4*sum sin^2(q_a/2)), inverse orthonormal FFT",
        boundary="periodic", dtype="float64", zero_mode_retained=True,
        no_per_configuration_or_condition_normalization=True,
        parameter_cells=cells, conditions=cells * len(volumes),
        configurations=cells * len(volumes) * options.groups * options.samples,
        projected_output_bytes=output_bytes, projected_working_bytes=working_bytes,
        cpu_threads_requested=1, fft_work_unit="one forward and one inverse FFT per independent draw",
        split_unit="entire r cell across all iid groups and sizes",
        train_cell_indices=[i for i in range(cells) if i % options.test_every != options.test_offset],
        test_cell_indices=[i for i in range(cells) if i % options.test_every == options.test_offset],
        axes=["independent_iid_group", "independent_draw", "x", "y", "channel=1"],
        labels_used_in_training=False, couplings_in_configuration_arrays=False,
        training_updates=0, source_sha256=sha256_file(__file__), qualification=QUALIFICATION)


def gaussian_diagnostics(configurations, r):
    """Public analytic-reference checks only; no alteration of the sampled fields."""
    values = np.asarray(configurations, dtype=np.float64)
    if (values.ndim != 5 or values.shape[-1] != 1 or values.shape[2] != values.shape[3]
            or not np.isfinite(values).all()):
        raise ValueError("Require finite [groups,samples,L,L,1] configurations")
    groups, samples, size, _, _ = values.shape
    volume, count = size ** 2, groups * samples
    spectrum = lattice_spectrum(size, r)
    # Stream one group here as during generation; avoid a full-condition complex copy.
    structure_sum = np.zeros((size, size), dtype=np.float64)
    mean_fields, twice_actions = [], []
    for group in values[..., 0]:
        transformed = np.fft.fftn(group, axes=(-2, -1), norm="ortho")
        power = np.abs(transformed) ** 2
        structure_sum += power.sum(axis=0)
        mean_fields.append(group.mean(axis=(-2, -1)))
        twice_actions.append((power * spectrum).sum(axis=(-2, -1)))
    mean_fields, twice_actions = np.stack(mean_fields), np.stack(twice_actions)
    measured_structure = structure_sum / count
    reference_structure = 1. / spectrum
    reference_covariance = exact_covariance(size, r)
    measured_covariance = np.fft.ifftn(measured_structure).real
    mean_variance = 1. / (volume * float(r))
    coordinate = np.arange(size)
    self_conjugate_axis = (coordinate == 0) | ((size % 2 == 0) & (coordinate == size // 2))
    self_conjugate = self_conjugate_axis[:, None] & self_conjugate_axis[None, :]
    structure_se = reference_structure * np.sqrt(np.where(self_conjugate, 2., 1.) / count)
    return dict(schema=SCHEMA, sample_semantics=SAMPLE_SEMANTICS, size=size, dimension=2,
        r=float(r), u=0., h=0., independent_draws=count, groups=groups, samples_per_group=samples,
        fourier_convention="phi_hat=fftn(phi,norm='ortho'); E|phi_hat(q)|^2=1/(r+4sum sin^2(q/2))",
        covariance_convention="unconnected zero-theory-mean C(d)=E[phi(x)phi(x+d)]; no empirical centering",
        exact_covariance_by_displacement=reference_covariance.tolist(),
        empirical_covariance_by_displacement=measured_covariance.tolist(),
        exact_structure_factor=reference_structure.tolist(),
        empirical_structure_factor=measured_structure.tolist(),
        exact_structure_factor_standard_error_iid=structure_se.tolist(),
        point_variance=dict(exact=float(reference_covariance[0, 0]),
                            empirical=float(measured_covariance[0, 0])),
        mean_field=dict(exact_mean=0., exact_variance=mean_variance,
            empirical_mean=float(mean_fields.mean()),
            empirical_second_moment=float(np.mean(mean_fields ** 2)),
            empirical_variance_ddof0=float(np.var(mean_fields, ddof=0)),
            independent_mean_standard_error=math.sqrt(mean_variance / count),
            mean_z_score=float(mean_fields.mean() / math.sqrt(mean_variance / count)),
            second_moment_z_score=float((np.mean(mean_fields ** 2) / mean_variance - 1.) * math.sqrt(count / 2.)),
            group_means=mean_fields.mean(axis=1).tolist()),
        twice_action_chi_squared=dict(distribution=f"2S ~ chi_squared(df={volume}) per independent draw",
            degrees_of_freedom=volume, exact_mean=volume, exact_variance=2 * volume,
            empirical_mean=float(twice_actions.mean()),
            empirical_variance_ddof1=float(np.var(twice_actions, ddof=1)),
            mean_standard_error=math.sqrt(2 * volume / count),
            mean_z_score=float((twice_actions.mean() - volume) / math.sqrt(2 * volume / count)),
            empirical_quantiles_025_050_975=np.quantile(twice_actions, (.025, .5, .975)).tolist(),
            group_means=twice_actions.mean(axis=1).tolist()),
        warnings=[], warning_policy="descriptive checks only; no automatic resampling, seed replacement or threshold-based rejection",
        uncertainty="Finite iid Monte Carlo uncertainty remains. Momentum bins are conjugate-correlated; per-bin standard errors are not a joint confidence region.",
        qualification=QUALIFICATION)


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def generate_dataset(output, options):
    """Generate into a fresh directory; preserve every failed/partial run, never resume."""
    plan = generation_plan(options)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "plan.json", plan)
    (output / "conditions").mkdir()
    shards, public_files, current = [], ["plan.json"], None
    wall, cpu = time.perf_counter(), time.process_time()
    try:
        for size in options.sizes:
            for cell, r in enumerate(options.r_values):
                split = "test" if cell % options.test_every == options.test_offset else "train"
                current = f"L{size}_c{cell:04d}"
                directory = output / "conditions" / current
                directory.mkdir()
                array = np.empty((options.groups, options.samples, size, size, 1), dtype=np.float64)
                seeds = []
                for group in range(options.groups):
                    progress = dict(schema=SCHEMA, status="running", workflow_complete=False,
                        current=current, iid_group_index=group, completed_conditions=len(shards),
                        error=None, requires_user_action=False, sample_semantics=SAMPLE_SEMANTICS)
                    _write_json(output / "status.json", progress)
                    _write_json(directory / "status.json", progress)
                    seed = group_seed(options, size, cell, group)
                    seeds.append(seed)
                    array[group] = sample_gaussian_fields(r, size=size, samples=options.samples, seed=seed)
                data_path = directory / "data.npz"
                np.savez(data_path, configurations=array)
                diagnostics_path = directory / "diagnostics.json"
                _write_json(diagnostics_path, gaussian_diagnostics(array, r))
                sidecar = dict(schema=SCHEMA, status="complete", cell_index=cell, size=size,
                    dimension=2, split=split, r=float(r), u=0., h=0.,
                    sample_semantics=SAMPLE_SEMANTICS, independent_iid_groups=options.groups,
                    independent_draws_per_group=options.samples, group_seeds=seeds,
                    data_sha256=sha256_file(data_path), diagnostics="diagnostics.json",
                    diagnostics_sha256=sha256_file(diagnostics_path), zero_mode_retained=True,
                    labels_used_in_training=False, qualification=QUALIFICATION)
                sidecar_path = directory / "sidecar.json"
                _write_json(sidecar_path, sidecar)
                status_path = directory / "status.json"
                _write_json(status_path, dict(schema=SCHEMA, status="complete", workflow_complete=True,
                    returncode=0, error=None, requires_user_action=False, sample_semantics=SAMPLE_SEMANTICS))
                shards.append(dict(size=size, cell_index=cell, split=split,
                    path=str(data_path.relative_to(output)), sidecar=str(sidecar_path.relative_to(output)),
                    sha256=sidecar["data_sha256"], sidecar_sha256=sha256_file(sidecar_path),
                    bytes=data_path.stat().st_size))
                public_files.extend(str(path.relative_to(output)) for path in
                                    (diagnostics_path, sidecar_path, status_path))
                print(json.dumps(dict(event="gaussian_iid_condition_complete", condition=current,
                    independent_draws=options.groups * options.samples, split=split)), flush=True)
                del array
        public_files.extend(("manifest.json", "status.json"))
        manifest = dict(schema=SCHEMA, status="complete", options=options.manifest_options(),
            sample_semantics=SAMPLE_SEMANTICS, shards=shards, public_files=public_files,
            plan_sha256=sha256_file(output / "plan.json"), source_sha256=plan["source_sha256"],
            warning_count=0, labels_used_in_training=False, zero_mode_retained=True,
            qualification=QUALIFICATION)
        _write_json(output / "manifest.json", manifest)
        _write_json(output / "status.json", dict(schema=SCHEMA, status="complete", workflow_complete=True,
            current=None, completed_conditions=len(shards), returncode=0, error=None,
            requires_user_action=False, manifest_sha256=sha256_file(output / "manifest.json"),
            sample_semantics=SAMPLE_SEMANTICS, warning_count=0,
            wall_seconds=time.perf_counter() - wall, cpu_seconds=time.process_time() - cpu))
        return manifest
    except Exception as exc:
        failure = dict(schema=SCHEMA, status="failed", workflow_complete=False,
            current=current, completed_conditions=len(shards), returncode=1,
            error=f"{type(exc).__name__}: {exc}", requires_user_action=True)
        _write_json(output / "status.json", failure)
        if current is not None:
            _write_json(output / "conditions" / current / "status.json", failure)
        raise


# Explicit descriptive aliases for external orchestration; no alternate algorithm.
plan_dataset = generation_plan
prepare_dataset = generate_dataset
