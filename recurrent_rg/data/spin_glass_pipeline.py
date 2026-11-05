"""Auditable, dimension-independent quenched EA data, separate from legacy caches.

One shard is one quenched disorder realization at every temperature.  Real
replicas are independent PT ladders, not temperatures of the same ladder.
Only explicit spin/overlap views are model inputs; J and T remain metadata.
"""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from .spin_glass import energy, generate_bonds, sample_parallel_tempering


SCHEMA = "edwards_anderson_pt_v1"
QUALIFICATION = (
    "Finite correlated MCMC data; diagnostics do not certify equilibration or "
    "independence. No automatic extension, reseeding, training, or critical-point fit."
)


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


def _finite(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class GenerationOptions:
    shape: tuple[int, ...] = (9, 9)
    temperatures: tuple[float, ...] = (.5, 1., 1.5, 2.)
    disorders: int = 8
    replicas: int = 2
    samples: int = 128
    burn_in: int = 2048
    spacing: int = 16
    exchange_interval: int = 1
    distribution: str = "bimodal"
    bond_scale: float = 1.
    bond_mean: float = 0.
    positive_probability: float = .5
    field: float = 0.
    seed: int = 91
    test_every: int = 3
    test_offset: int = 1
    max_spin_proposals: int = 50_000_000
    max_output_bytes: int = 2 * 1024**3
    max_memory_bytes: int = 512 * 1024**2
    max_sites: int = 1_000_000

    def validate(self):
        if not isinstance(self.shape, (tuple, list)) or not self.shape:
            raise ValueError("shape must contain at least one spatial axis")
        # NumPy limits the rank of the resulting configurations/diagnostic arrays.
        if len(self.shape) > 16:
            raise ValueError("At most 16 spatial dimensions are supported")
        for side in self.shape:
            _integer(side, "lattice side", 3)
        for name, minimum in (("disorders", 3), ("replicas", 2), ("samples", 4),
                              ("burn_in", 0), ("spacing", 1), ("exchange_interval", 1),
                              ("seed", 0), ("test_every", 2), ("test_offset", 0),
                              ("max_spin_proposals", 1), ("max_output_bytes", 1),
                              ("max_memory_bytes", 1),
                              ("max_sites", 1)):
            _integer(getattr(self, name), name, minimum)
        if self.test_offset >= self.test_every:
            raise ValueError("test_offset must be smaller than test_every")
        # O(1), even for adversarial counts: budgets reject before any ID lists.
        tests = (1 + (int(self.disorders) - 1 - int(self.test_offset)) // int(self.test_every)
                 if self.test_offset < self.disorders else 0)
        if not 0 < tests < self.disorders:
            raise ValueError("The disorder split must contain both TRAIN and TEST")
        if not isinstance(self.temperatures, (list, tuple, np.ndarray)):
            raise ValueError("temperatures must be a positive increasing ladder")
        ts = np.asarray(self.temperatures, dtype=np.float64)
        if ts.ndim != 1 or len(ts) < 2 or not np.isfinite(ts).all():
            raise ValueError("At least two finite temperatures are required")
        if (ts <= 0).any() or (np.diff(ts) <= 0).any() or not np.isfinite(1. / ts).all():
            raise ValueError("Temperatures must be strictly positive and increasing; T=0 is unsupported")
        if self.distribution not in ("bimodal", "gaussian"):
            raise ValueError("distribution must be bimodal or gaussian")
        scale = _finite(self.bond_scale, "bond_scale")
        mean = _finite(self.bond_mean, "bond_mean")
        probability = _finite(self.positive_probability, "positive_probability")
        _finite(self.field, "field")
        if scale <= 0 or not 0 <= probability <= 1:
            raise ValueError("Require positive bond_scale and 0 <= positive_probability <= 1")
        if self.distribution == "bimodal" and mean != 0:
            raise ValueError("bond_mean applies only to Gaussian bonds")
        if self.distribution == "bimodal" and probability in (0., 1.):
            raise ValueError("Disorder-held-out data requires nondegenerate bimodal bonds (0 < p < 1)")
        if self.distribution == "gaussian" and probability != .5:
            raise ValueError("positive_probability applies only to bimodal bonds")
        if math.prod(int(side) for side in self.shape) > self.max_sites:
            raise ValueError("Site-count budget exceeded before allocation")
        if self.disorders > 2**31:
            raise ValueError("Too many disorders for collision-free uint32 seed namespaces")
        return self


def generation_plan(options):
    """Validate and project work without allocating spins or touching output."""
    options.validate()
    n, d = math.prod(int(side) for side in options.shape), len(options.shape)
    nt = len(options.temperatures)
    nr, ns, nd = int(options.replicas), int(options.samples), int(options.disorders)
    sweeps = int(options.burn_in) + ns * int(options.spacing)
    proposals = nd * nr * nt * n * sweeps
    # Uncompressed NPZ payload, conservative ZIP/JSON overhead and diagnostics.
    shard_bytes = (nt * nr * ns * (n + 8) + 8 * d * n + 8 * nt
                   + 32 * nr * nt + 8 * nr * nt * nt + 16 * nr * (nt - 1))
    diagnostic_bytes = nt * (nr * 2048 + nr * (nr - 1) // 2 * (4096 + 512 * d))
    projected_bytes = nd * (shard_bytes + diagnostic_bytes + 131_072) + 131_072
    # Generation streams one disorder; allow multiple copies for validation/FFT,
    # all temperature walkers, and forward/backward neighbor tables. The estimate
    # is a conservative preflight bound, not a measured process RSS guarantee.
    working_bytes = 4 * shard_bytes + 16 * d * n + nd * diagnostic_bytes + 1_048_576
    if proposals > options.max_spin_proposals:
        raise ValueError(f"Spin-proposal budget exceeded: {proposals} > {options.max_spin_proposals}")
    if projected_bytes > options.max_output_bytes:
        raise ValueError("Projected output exceeds byte budget before sampling")
    if working_bytes > options.max_memory_bytes:
        raise ValueError("Projected working memory exceeds budget before allocation")
    return dict(
        schema=SCHEMA, options=asdict(options), dimension=d, sites=n,
        sweeps_per_temperature_per_real_replica=sweeps,
        total_single_spin_visits=proposals, total_flip_proposals_upper_bound=proposals,
        projected_output_bytes=projected_bytes,
        projected_working_bytes=working_bytes,
        train_disorders=[i for i in range(nd) if i % options.test_every != options.test_offset],
        test_disorders=[i for i in range(nd) if i % options.test_every == options.test_offset],
        split_unit="entire quenched disorder, including all temperatures and real replicas",
        axes=dict(bonds=["direction", "*spatial"],
                  configurations=["temperature", "real_replica", "sample", "*spatial"],
                  energies=["temperature", "real_replica", "sample"]),
        hamiltonian="H=-sum_(x,a) J[a,x]*s[x]*s[x+e_a]-h*sum_x s[x]; k_B=1",
        bond_convention="One forward edge per site/direction; periodic axes of length >=3",
        initialization="mixed: even real replicas hot, odd real replicas cold (+1)",
        exchange="Within each real replica only; alternating even/odd neighboring temperature pairs",
        diagnostic_counts_include_burn_in=True,
        local_update="Random-order lazy Metropolis: propose a spin flip at each visit with probability 1/2",
        local_attempts_definition="Actual flip proposals; local_accepts counts accepted flips",
        round_trip_definition="Per walker: first endpoint -> other endpoint -> original endpoint",
        seed_layout="SHA256(master,shape) base; bijective uint32 map of 2*disorder_id+purpose_tag",
        duplicate_bonds="Identical bond fingerprints across splits are rejected, never resampled",
        training_updates=0, labels_used_in_training=False, qualification=QUALIFICATION,
    )


def _seed(master, purpose, disorder, shape):
    message = json.dumps(["EA-PT-v1", int(master), [int(side) for side in shape]])
    base = int.from_bytes(hashlib.sha256(message.encode()).digest()[:4], "little")
    tag = {"quenched_bonds": 0, "thermal_ladders": 1}[purpose]
    return ((base + 2 * int(disorder) + tag) * 0x9E3779B1) % 2**32


def _bond_sha(bonds):
    return hashlib.sha256(np.ascontiguousarray(bonds, dtype="<f8").tobytes()).hexdigest()


def _sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _json(path, payload):
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("w") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False,
                  default=lambda x: x.tolist() if isinstance(x, (np.ndarray, np.generic)) else str(x))
        stream.write("\n")
    temporary.replace(path)


def series_diagnostics(values):
    """Descriptive initial-positive-pair autocorrelation estimate, not a certificate."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Expected a nonempty finite scalar series")
    result = dict(samples=len(x), mean=float(x.mean()), std=float(x.std()),
                  lag1=None, ess=None, iat_saved_intervals=None)
    if len(x) < 4 or x.std() < 1e-14:
        result["warning"] = "Too short or constant series; mixing/ESS unidentifiable"
        return result
    centered = x - x.mean()
    transform = np.fft.rfft(centered, n=2 ** (2 * len(x) - 1).bit_length())
    cov = np.fft.irfft(transform * transform.conjugate())[:len(x)] / np.arange(len(x), 0, -1)
    rho = cov / cov[0]
    result["lag1"] = float(rho[1])
    positive_sum = 0.
    for k in range(1, len(x) - 1, 2):
        pair = float(rho[k] + rho[k + 1])
        if pair <= 0:
            break
        positive_sum += pair
    inefficiency = max(1., 1. + 2 * positive_sum)
    result.update(ess=float(len(x) / inefficiency), iat_saved_intervals=.5 * inefficiency)
    return result


def sampling_diagnostics(result, shape, temperatures):
    """Keep within-J thermal statistics separate from disorder averaging."""
    spins, energies = result["configurations"], result["energies"]
    nt, nr, ns = spins.shape[:3]
    n = math.prod(shape)
    rows, warnings = [], []
    for t in range(nt):
        thermal = []
        for r in range(nr):
            e = series_diagnostics(energies[t, r] / n)
            m = series_diagnostics(spins[t, r].reshape(ns, n).mean(1))
            thermal.append(dict(real_replica=r, energy_density=e, magnetization=m))
            for name, series in (("energy", e), ("magnetization", m)):
                if series["ess"] is None or series["ess"] < min(32, ns / 2):
                    warnings.append(f"T index {t}, real replica {r}, {name}: limited ESS evidence")
        pairs = []
        for a in range(nr):
            for b in range(a + 1, nr):
                q_field = (spins[t, a] * spins[t, b]).reshape(ns, *shape)
                q = q_field.reshape(ns, n).mean(1)
                q2, q4 = float(np.mean(q**2)), float(np.mean(q**4))
                overlap = series_diagnostics(q)
                squared = series_diagnostics(q**2)
                if squared["ess"] is None or squared["ess"] < min(32, ns / 2):
                    warnings.append(f"T index {t}, pair {a}/{b}: limited q^2 ESS evidence")
                # Axis-resolved, finite-box overlap correlation, not an R prediction.
                correlations = []
                for axis, side in enumerate(shape):
                    distances = list(range(1, min(side // 2, 4) + 1))
                    correlations.append(dict(axis=axis, distances=distances,
                        mean=[float(np.mean(q_field * np.roll(q_field, shift, axis=axis + 1)))
                              for shift in distances]))
                pairs.append(dict(real_replicas=[a, b], q=overlap, q_squared=squared,
                    q2=q2, q4=q4, raw_overlap_susceptibility=n * q2,
                    binder_1_minus_q4_over_3q2sq=(1 - q4 / (3 * q2**2)) if q2 > 0 else None,
                    overlap_correlations=correlations))
        rows.append(dict(temperature=float(temperatures[t]), thermal=thermal, overlap_pairs=pairs,
                         q2_pair_mean=float(np.mean([p["q2"] for p in pairs]))))
    swap_attempts, swap_accepts = result["swap_attempts"], result["swap_accepts"]
    swap_rates = [[float(a / n) if n else None for a, n in zip(accepted, attempted)]
                  for accepted, attempted in zip(swap_accepts, swap_attempts)]
    if any(v is None or v < .05 for row in swap_rates for v in row):
        warnings.append("Some neighboring temperatures have absent/low exchange acceptance")
    if not np.asarray(result["round_trips"]).any():
        warnings.append("No full temperature round trip observed")
    return dict(temperatures=rows, swap_acceptance=swap_rates,
                round_trips=np.asarray(result["round_trips"]).tolist(), warnings=warnings,
                pair_combinations_are_not_independent=True,
                susceptibility_convention="N*<q^2> within one disorder; not connected in a field",
                quality_certified=False, qualification=QUALIFICATION)


def generate_dataset(output, options):
    """Generate new shards only. Never overwrite, resume, train, or auto-extend."""
    plan = generation_plan(options)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    (root / "shards").mkdir()
    _json(root / "plan.json", plan)
    started = time.monotonic()
    status = dict(schema=SCHEMA, status="running", workflow_complete=False,
                  started_at=datetime.now(timezone.utc).isoformat(), completed_disorders=0,
                  training_updates=0, equilibrium_certified=False)
    _json(root / "status.json", status)
    manifest = dict(schema=SCHEMA, status="running", plan_sha256=_sha(root / "plan.json"),
                    options=asdict(options), shards=[], training_updates=0,
                    qualification=QUALIFICATION)
    diagnostics = []
    bond_splits = {}
    try:
        for disorder in range(options.disorders):
            status["current_disorder"] = disorder
            _json(root / "status.json", status)
            bond_seed = _seed(options.seed, "quenched_bonds", disorder, options.shape)
            sampling_seed = _seed(options.seed, "thermal_ladders", disorder, options.shape)
            bonds = generate_bonds(options.shape, bond_seed, distribution=options.distribution,
                scale=options.bond_scale, mean=options.bond_mean,
                positive_probability=options.positive_probability)
            split = "test" if disorder % options.test_every == options.test_offset else "train"
            bond_fingerprint = _bond_sha(bonds)
            if bond_fingerprint in bond_splits and bond_splits[bond_fingerprint] != split:
                raise ValueError("Identical quenched bonds occurred across TRAIN/TEST; preserving partial data, no automatic reseed")
            bond_splits[bond_fingerprint] = split
            result = sample_parallel_tempering(bonds, options.temperatures, samples=options.samples,
                burn_in=options.burn_in, spacing=options.spacing, seed=sampling_seed,
                replicas=options.replicas, exchange_interval=options.exchange_interval,
                field=options.field, initial="mixed")
            relative = f"shards/disorder_{disorder:06d}.npz"
            target = root / relative
            arrays = {key: result[key] for key in (
                "configurations", "energies", "swap_attempts", "swap_accepts",
                "local_attempts", "local_accepts", "round_trips", "temperature_visits")}
            arrays["site_visits"] = result["site_visits"]
            arrays.update(bonds=bonds, temperatures=np.asarray(options.temperatures, np.float64),
                          disorder_id=np.array(disorder, np.int64),
                          bond_seed=np.array(bond_seed, np.uint32),
                          sampling_seed=np.array(sampling_seed, np.uint32))
            if "replica_seeds" in result:
                arrays["replica_seeds"] = np.asarray(result["replica_seeds"], np.uint32)
            temporary = target.with_suffix(".npz.writing")
            with temporary.open("wb") as stream:
                np.savez(stream, **arrays)
            temporary.replace(target)
            row = dict(disorder_id=disorder, split=split, path=relative,
                       sha256=_sha(target), bytes=target.stat().st_size,
                       bond_sha256=bond_fingerprint,
                       bond_seed=bond_seed, sampling_seed=sampling_seed)
            manifest["shards"].append(row)
            diagnostics.append(dict(disorder_id=disorder, split=split,
                **sampling_diagnostics(result, options.shape, options.temperatures)))
            _json(root / "manifest.json", manifest)
            _json(root / "sampling.json", dict(schema=SCHEMA, status="running",
                disorders=diagnostics, qualification=QUALIFICATION))
            status.update(completed_disorders=disorder + 1, current_disorder=None,
                          elapsed_seconds=time.monotonic() - started)
            _json(root / "status.json", status)
            actual = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
            if actual > options.max_output_bytes:
                raise RuntimeError("Actual output byte budget exceeded; no further sampling")
        _json(root / "sampling.json", dict(schema=SCHEMA, disorders=diagnostics,
            disorder_average=[dict(temperature=float(t), q2=float(np.mean([
                row["temperatures"][i]["q2_pair_mean"] for row in diagnostics])))
                for i, t in enumerate(options.temperatures)],
            warning_count=sum(len(row["warnings"]) for row in diagnostics),
            aggregation="Thermal/pair means within J first, then equal-weight disorder mean",
            equilibrium_certified=False, qualification=QUALIFICATION))
        manifest.update(status="complete", sampling_sha256=_sha(root / "sampling.json"),
            distinct_bond_realizations=len(bond_splits),
            code_sha256={p.name: _sha(p) for p in (Path(__file__),
                Path(__file__).with_name("spin_glass.py"))})
        _json(root / "manifest.json", manifest)
        if sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) > options.max_output_bytes:
            raise RuntimeError("Final output byte budget exceeded; preserving all generated shards")
        status.update(status="complete", workflow_complete=True,
                      elapsed_seconds=time.monotonic() - started,
                      manifest_sha256=_sha(root / "manifest.json"))
        _json(root / "status.json", status)
    except Exception as exc:
        status.update(status="failed", workflow_complete=False,
                      error=f"{type(exc).__name__}: {exc}",
                      elapsed_seconds=time.monotonic() - started,
                      requires_user_action=True)
        _json(root / "status.json", status)
        raise
    return manifest


@dataclass(frozen=True)
class SpinGlassRecord:
    disorder_id: int
    split: str
    configurations: np.ndarray
    bonds: np.ndarray
    temperatures: np.ndarray
    energies: np.ndarray

    def spin_view(self, replica=0):
        """Label-free [T,sample,*spatial,1]; T indexes ensembles, not channels."""
        replica = _integer(replica, "real replica", 0)
        if replica >= self.configurations.shape[1]:
            raise ValueError("Real replica index out of bounds")
        return self.configurations[:, replica, ..., None]

    def overlap(self, a=0, b=1):
        a, b = _integer(a, "real replica a", 0), _integer(b, "real replica b", 0)
        if a == b or max(a, b) >= self.configurations.shape[1]:
            raise ValueError("Overlap requires two different real replicas of the same disorder")
        return self.configurations[:, a] * self.configurations[:, b]

    def overlap_view(self, a=0, b=1):
        """Label-free [T,sample,*spatial,1], not automatic QCR compatibility."""
        return self.overlap(a, b)[..., None]


def iter_spin_glass(output, split=None, *, verify_hashes=True):
    """Stream validated shards; no labels are bundled into spin/overlap views.

    This dedicated loader intentionally does not call ``load_ensembles``:
    quenched-disorder holds out J, not temperature, and >=3D training is not
    implemented by adding a data generator.
    """
    if split not in (None, "train", "test"):
        raise ValueError("split must be train, test, or None")
    root = Path(output).resolve()
    with (root / "status.json").open() as stream:
        status = json.load(stream)
    if status.get("schema") != SCHEMA or status.get("status") != "complete" or not status.get("workflow_complete"):
        raise ValueError("Spin-glass dataset is not complete")
    if verify_hashes and _sha(root / "manifest.json") != status.get("manifest_sha256"):
        raise ValueError("Manifest SHA256 mismatch")
    with (root / "manifest.json").open() as stream:
        manifest = json.load(stream)
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "complete":
        raise ValueError("Unsupported spin-glass schema/status")
    if verify_hashes and (_sha(root / "plan.json") != manifest.get("plan_sha256") or
                          _sha(root / "sampling.json") != manifest.get("sampling_sha256")):
        raise ValueError("Plan or sampling receipt SHA256 mismatch")
    options = GenerationOptions(**manifest["options"])
    options.validate()
    shape, nt = tuple(options.shape), len(options.temperatures)
    shards = manifest["shards"]
    if (len(shards) != options.disorders or
            any(row["disorder_id"] != index for index, row in enumerate(shards))):
        raise ValueError("Disorder identity/order mismatch")
    seen_bonds = {}
    for row in shards:
        expected_split = "test" if row["disorder_id"] % options.test_every == options.test_offset else "train"
        if row["split"] != expected_split:
            raise ValueError("Disorder split mismatch")
        bond_sha = row.get("bond_sha256")
        if not isinstance(bond_sha, str) or len(bond_sha) != 64:
            raise ValueError("Missing bond fingerprint")
        if bond_sha in seen_bonds and seen_bonds[bond_sha] != row["split"]:
            raise ValueError("Identical bonds leaked across disorder splits")
        seen_bonds[bond_sha] = row["split"]
    # Validate the complete split metadata before yielding any model input.
    for row in shards:
        if split is not None and row["split"] != split:
            continue
        path = (root / row["path"]).resolve()
        if path.parent != (root / "shards").resolve():
            raise ValueError("Shard path escapes dataset")
        if verify_hashes and (path.stat().st_size != row["bytes"] or _sha(path) != row["sha256"]):
            raise ValueError("Shard SHA256/size mismatch")
        with np.load(path, allow_pickle=False) as saved:
            x, bonds = saved["configurations"], saved["bonds"]
            ts, energies = saved["temperatures"], saved["energies"]
            if int(saved["disorder_id"]) != row["disorder_id"]:
                raise ValueError("Shard disorder identity mismatch")
            if _bond_sha(bonds) != row["bond_sha256"]:
                raise ValueError("Bond fingerprint mismatch")
            if (x.shape != (nt, options.replicas, options.samples, *shape) or x.dtype != np.int8
                    or bonds.shape != (len(shape), *shape) or not np.isfinite(bonds).all()
                    or not np.isin(x, (-1, 1)).all() or not np.array_equal(ts, options.temperatures)
                    or energies.shape != x.shape[:3] or not np.isfinite(energies).all()):
                raise ValueError("Malformed spin-glass shard arrays")
            # Check semantics as well as hashes: stored energies belong to these spins/J.
            # Do not promote a full multi-temperature shard to float64 at once.
            for t in range(nt):
                for r in range(options.replicas):
                    for start in range(0, options.samples, 16):
                        sl = slice(start, start + 16)
                        if not np.allclose(energy(x[t, r, sl], bonds, options.field),
                                           energies[t, r, sl], rtol=1e-11, atol=1e-10):
                            raise ValueError("Saved energies do not match spins, bonds, and field")
            yield SpinGlassRecord(row["disorder_id"], row["split"], x, bonds, ts, energies)
