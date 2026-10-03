# Developing the QCR pipeline

## Dependency direction

`experiments/scripts -> runner/training/evaluation -> model/loss/data utilities`.
The reusable package must not import experiments or plotting from its losses.
Existing `experiments.joint_qcr_ising` and `experiments._utils` imports remain
compatibility exports. Historical experiment recipes and result directories
remain intact; new reusable logic belongs in the package, not another recipe.

| Module | Responsibility |
|---|---|
| `config.py` | Versioned dataclasses; positive/even counts, explicit symmetry policy, scale validation |
| `data/samplers.py` | Discover existing physics-specific samplers; no ambiguous common sweep unit |
| `data/ensembles.py` | Dataset metadata, whole-group train/test splitting, replica pooling, legacy caches |
| `data/views.py` | Within-ensemble sampling and explicitly requested spin-flip augmentation |
| `data/statistics.py` | Local distribution descriptors consumed by the geometry loss |
| `model_nd.py`, `geometry.py`, `adapters.py` | Shared Q/C/R, geometry, field space; existing state-dict keys unchanged |
| `losses.py` | Current objective algebra and fixed-C sufficient-statistic entry point |
| `training.py` | Shared optimizer, module freezing, batches, checkpoints and deterministic resume |
| `checkpoints.py` | One model factory; strict Q/C/R loading; legacy architecture adapter |
| `evaluation.py` | Direct/recursive trajectories, labeled normalization, fixed-point Jacobians |
| `plotting/` | Generic R maps, coupling plots and trajectories using diagnostic arrays |
| `runner.py`, `cli.py` | Configuration-driven workflow, data fingerprint and reports |

## Starting a run

No new dependency or configuration framework is needed. Config files are JSON;
cache paths in them resolve relative to the config file. The examples use
existing local datasets and are not self-contained download recipes.

```bash
python -m recurrent_rg train --config configs/tfim1d_short.json --output results/my_short_run
python -m recurrent_rg plot --run results/my_short_run
python -m recurrent_rg evaluate --run results/my_short_run \
  --cache results/tfim_short_vs_long_g1_seed91/ensembles_L128.pt --max-rg-steps 5 \
  --output results/my_short_run/long_test
```

`plot` calculates the same diagnostics as `evaluate` and adds figures; neither
trains. Couplings are used only for grouping, splitting and optional plotting.
The installed `recurrent-rg` entry point provides the same commands.
Use explicit output directories when comparing different evaluation caches.
Current generic runner: CPU float64, binary single-channel 1D/2D data,
or one-hot categorical 2D data with categorical sampling and variance-floor loss.
Generic flow plotting supports latent dimension1 or2. Existing field adapters
are retained; continuous training still needs suitable data statistics and
anti-collapse losses. The [Potts pilot](potts.md) implements the categorical
statistics, sampler, preservation loss and a complete runnable configuration.

Train depth is `training.max_rg_steps`; test depth is
`evaluation.max_rg_steps` or `--max-rg-steps`. Internal `levels=steps+1`.
`model.coarse_grain(X,max_rg_steps=...)` never changes `model.levels`.
The runner checks every block division and a minimum final spatial size4.
Testing a short-trained model for more steps requires an appropriately large
**evaluation** cache; it does not enlarge training configurations.

## Data and ensemble organization

Signed nearest-neighbor 2D Ising sampling is available through
`sample_signed_field_chain` in `data.samplers`. For negative K it requires
an even periodic square lattice and uses the exact checkerboard gauge map:
ferromagnetic |K| in staggered fields B*eta. Cluster recoloring depends on
the sum of eta over the cluster, not its size. Returned samples are physical
spins, not gauge-transformed spins. Positive K delegates to the original
sampler. `python -m scripts.generate_symmetric_ising2d --output NEW_DIRECTORY`
generates the symmetric signed-K/focused-B pilot grid with sign-grouped splits.
This symmetry of the sampling grid is not a model constraint or a claim that
the phase diagram is symmetric under K->-K at nonzero uniform B.

Existing `(train_X,train_labels,test_X,test_labels)` tensor caches are supported.
New unlabeled caches can use:

```python
{
    "schema_version": 1,
    "train": {"configurations": train_X, "group_ids": train_source_ids},
    "test":  {"configurations": test_X,  "group_ids": test_source_ids},
}
```

`group_ids` identify source ensemble families across replicas, not physical
coupling labels. They must be globally consistent and disjoint across splits.
Optional `couplings` metadata can be supplied in each split. Without labels,
omit `coupling_names`; QCR training and R/trajectory plotting still work.

Use `EnsembleDataset.pool(n)` only after group-level splitting. It concatenates
the sample axis within the same source group, never spatial sites, and checks
optional coupling metadata for conflicts. `train_pool_replicas` and
`test_pool_replicas` are separate. For the short example, pooling8 training
replicas gives3 ensembles of1,024 samples per coupling; also set training
`sample_batch=1024` to actually use all samples. The test cache has only3
replicas per coupling, so leave its pooling factor1.

Spin-flip augmentation requires `spin_flip_symmetric=true` explicitly.
For nonzero longitudinal field, use no spin flips and select
`loss.kernel_noncollapse_mode="variance_floor"`. Declaring a symmetry is the
dataset author's responsibility; the runner does not guess it from filenames.
Sampling algorithms retain their own equilibration and decorrelation units.

## Loss changes and controlled experiments

Edit current loss algebra in `losses.py`, local descriptors in
`data/statistics.py`, and configure weights/conventions explicitly. Keep labels
out of the objective. Preserve the distinction between summed latent error
and per-coordinate mean error. Binary lookup is an execution optimization,
not a different objective; value/gradient tests cover it.
Optional `loss.weights` uses the historical QCRWeights names: `anchor=2`,
`coarse_consistency=5`, `kernel_noncollapse=2`, `separation=10`, `commutator=5`.
The historical anchor term contains an internal factor10, so its effective
microscopic split coefficient is20. `loss.rollout_weight=5` is separate.

New experiments should explicitly retain `loss.isotropy_weight=20`, as in all
current `configs/` presets. This adds `||Cov(g_0)-I||_F^2 / dim_g`, using the
microscopic half-encoding midpoint and population covariance across ensembles.
It does not replace split consistency or closure and is not applied at coarse
scales. Low-level defaults remain zero for historical reproducibility; copying
an old recipe without this explicit option does not enable the new term.
Full global covariance is not evidence of two independent physical parameters.

Experimental alternative: `loss.geometry_mode="relative_local"` with
`isotropy_weight=0` (see `configs/ablations/*relative_local.json`). This preserves
the original interval targets and all closure/split/kernel terms, but normalizes
pair penalties by `target_distance**2 + noise_allowance**2 + floor**2`, caps
inverse weights at100, and weights pairs by target-distance squared divided by
that denominator. The default experimental floor is .05 in scaled-target units.
Half of the objective averages all pairs; half averages the symmetrized union
of4 nearest *resolved* data-statistic neighbors per ensemble. Unresolved pairs
do not consume neighbor slots. Without any resolved neighbor the global term
is used alone. Targets, weights and neighbor selection are detached from Q/C.
These are experimental heuristics, not confidence bounds or dimensionality
guarantees. Absolute geometry remains the backward-compatible default.

`training.train_modules` selects any nonempty subset of Q,C,R. Excluded modules
are frozen. Fixed positive decimation is `model.fixed_kernel=true`; construct
the learned model first and replace C to preserve paired Q/R initialization.

For a **trainable** C starting at positive decimation, set
`model.kernel_initialization="positive_decimation"` with `fixed_kernel=false`
and `kernel_layout="nonoverlapping"`. The binary map initially selects the
first spin of every block exactly, using a unit-weight path through the MLP.
Other hidden rows retain random weights; their output weights start at zero.
All C weights remain trainable, with no orientation penalty or sign constraint.
The default is `"random"`; this initialization consumes no extra RNG draws and
preserves the paired Q/R initialization. Checkpoint loading overwrites initial
weights as usual; initialization does not reset C during training or resume.
Frozen-Q/C covariance regression remains a separate experimental objective;
selecting R-only with the default runner does not silently enable that metric.

## Checkpoints and compatibility

New runs contain a manifest, data SHA256, complete resume checkpoints, weights,
history and a final `checkpoint_last.pt`. Checkpoints include optimizer state,
minibatch RNG, CPU Torch RNG and history. The shared current loop uses only
these RNGs during training; adding another RNG requires checkpoint support.
New runs have a fixed `training.log_every` cadence independent of final epoch
count. On continuation, an off-cadence former stopping endpoint is omitted from
the continued history (the source history is retained), matching continuous
logging. Historical entry points retain their old automatic logging cadence.

For exact continuation, increase only `training.epochs` in the config and run:

```bash
python -m recurrent_rg train --config configs/extended.json \
  --resume results/my_short_run/checkpoint_last.pt --output results/my_continuation
```

Exact CLI resume rejects changed data/configuration; output must be new.
`--initialize-from .../qcr_model.pt` instead starts a fresh optimizer from
weights, and is deliberately not called resume. Old training entry points
retain their historical resume interface and checkpoint keys.

`load_run(path,checkpoint=2000)` supports old `summary.json`/`run_config.json`
directories and new manifests. Weight-only historical runs need explicit
`overrides={"levels":3,...}` where architecture cannot be recovered. Missing
Q/C/R keys are errors; missing unused action-network weights are allowed.
Physical model names/units are not inferred from paths. Old reports remain
available for old run schemas; the generic CLI reporter requires a new manifest.

## Evaluation conventions

Scalar plots display R(g), vector arrows display R(g)-g. Stability always uses
eigenvalues of the full discrete R Jacobian. Numerical root searches have
explicit bounds and are not exhaustive. Training R inputs exclude the final
target-only scale. Encoding-range shading is not proof of dense data support.

`closure_metrics` labels whether normalization uses evaluated targets or an
explicit reference set. Zero target variance gives null normalized error,
not a falsely good score. Existing specialized reports retain their historical
normalization conventions for reproducibility. No physical correctness is
inferred solely from closure or proximity of a latent root to a coupling curve.

## Required checks

```bash
python -m pytest
python -m recurrent_rg --help
```

Regression tests cover loss conventions, binary execution values/gradients,
strict checkpoint loading, old imports, pooled/split data, unlabeled data,
frozen modules, separate evaluation depth, data-fingerprint checks, discrete
stability, exact resumed weights/history and scalar/vector report generation.
When extending sampling, add a small exact-measure or known-statistic check.
Do not delete historical scripts/checkpoints to make a refactor pass.
