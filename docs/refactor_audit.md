# Modularization and core-code audit

## Scope

Shared objective, data views/statistics, training, checkpoints, diagnostics and
plotting now live in the installable `recurrent_rg` package. The package does
not import experiment scripts.28 active files were mechanically migrated to
package imports; old import paths remain compatibility exports. Specialized
historical sampling/report recipes remain available and are not all rewritten
into the generic runner. Existing data, results and checkpoints were preserved.

## Numerical verification

- Before extraction, captured loss, every diagnostic term, and **all** Q/C/R
  parameter gradients for four seeded cases:1D/2D, balanced augmentation with
  balance penalty / no augmentation with variance-floor penalty.
- After extraction, compared these tensors with zero absolute and relative
  tolerance: all matched exactly. Q/C/R architecture and default objective
  formulas were not changed.
- Full test suite: **106 passed**, one existing PyTorch TypedStorage deprecation
  warning, no failures. Baseline before refactor:90 passed.
- Tests cover complete scalar/vector training-report workflows, exact resume
  from final and intermediate checkpoints, two logging cadences, frozen
  modules, independent evaluation depth, pooling/split safeguards, unlabeled
  datasets, cache mutation checks, binary lookup gradients and loss defaults.
- Loaded real historical scalar TFIM, vector TFIM, learned-C 2D field Ising,
  and fixed-C 2D field Ising checkpoints; all saved Q/C/R tensors matched.
- Both long/short example configurations validated against their real caches.
- `python -m recurrent_rg --help` and train help work; `compileall` over package,
  experiments and scripts passed; `git diff --check` passed.
- Inspected scalar/vector report images produced by synthetic smoke tests.
  These are interface tests, not new physical experiment results.

## Issues addressed during the core review

1. Shared scientific logic depended on an experiment entry point. It now has
   one package implementation and compatibility aliases.
2. `sample_batch` could silently truncate or round down. Invalid/unavailable
   sample counts now fail explicitly; nonfinite training loss also stops.
3. Generic checkpoint consumers could ignore missing Q/C/R weights. Strict
   Q/C/R validation now allows only absent unused action-network weights.
4. Scalar legacy reconstruction did not uniformly handle fixed-C2D models.
   The common loader handles fixed/learned C in both dimensions and checks
   configuration against saved weights.
5. Evaluation changed model depth in the short-chain diagnostic. Optional
   `max_rg_steps` now expands trajectories without modifying training depth.
6. New-run log cadence was tied to final epoch count. An explicit fixed cadence
   and endpoint-history handling make continued logs match continuous logs.
   Legacy entry points retain their historical cadence behavior.
7. Packaging declared a missing CLI module. The train/evaluate/plot entry point
   and `python -m recurrent_rg` now exist inside the packaged source tree.
8. Coupling labels must not become a requirement of the unsupervised pipeline.
   New caches can use source group IDs without coupling labels; plotting those
   labels is optional. Pooling and train/test isolation are checked separately.
9. Zero target variance must not masquerade as small relative closure error.
   Generic diagnostics preserve raw errors and report undefined NRMSE as null.
10. Discrete stability must use the full R Jacobian. The shared diagnostic is
    tested with a negative multiplier whose magnitude exceeds one. Scalar
    numerical-identity maps are not reported as hundreds of isolated roots.

## Limits retained deliberately

The generic runner currently targets CPU float64, binary1D/2D systems and
scalar/vector flow figures. Field adapters remain extensible, but additional
field spaces require appropriate distribution statistics and losses. Sampler
algorithms retain their physics-specific parameters and update units.

The CLI's exact-resume contract requires new versioned manifests and unchanged
data/configuration except increased epochs. Old weights load through the common
loader and can initialize a new run; historical exact continuation interfaces
remain available. Root searches are bounded and non-exhaustive. No full-scale
physics experiment was rerun or claimed improved by this structural refactor.
