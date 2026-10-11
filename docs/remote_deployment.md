# Remote execution smoke test

The deployment syncs a verified Git commit, not the dirty local working tree.
Generated caches, checkpoints and historical results are not part of the source
sync. Use a new, project-local environment; do not change the shared server's
Python environments, NVIDIA driver or other jobs.

QCR defaults to CPU for historical compatibility. The locked default trainer
also supports explicit `--device cuda:0`: the model, selected batches, C table
and empirical-MI tensor algebra use CUDA, while histogram construction and
final plotting stay on CPU. Architecture, float64 precision, losses and
independent Q/C/R gradients are unchanged. Optional historical ablation drivers
are not silently routed to CUDA.

A successful CUDA matrix check is only a hardware/runtime check, not evidence
of GPU QCR training. Verify `execution_device` in the manifest and the actual
devices serialized in the training checkpoint.

## Isolated environment

With a pre-existing Python 3.10+ interpreter, run from the repository:

```bash
bash scripts/setup_remote_env.sh /absolute/path/to/python /absolute/path/to/ham/.env
```

The script refuses to overwrite an existing environment. It installs the
[official PyTorch 2.6.0 CUDA 11.8 build](https://pytorch.org/get-started/previous-versions/)
without installing torchvision or torchaudio. CUDA driver compatibility must
still be checked by actual execution; see
[NVIDIA's compatibility limitations](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
The deployment server's compatible-wheel list excludes 2.7.1; this version pin
is an environment compatibility choice, not a model or objective change.

## Smoke test, not a physics experiment

```bash
CUDA_VISIBLE_DEVICES=0 .env/bin/python scripts/remote_smoke.py \
  --output tmp/remote_smoke_225 --check-cuda
```

This uses real, very short zero-field Ising chains on a 36x36 lattice, the
default random/trainable probabilistic 3x3 C and affine-reconstruction Q, and
three simultaneous independent-gradient Q/C/R updates. The limited data and
updates verify execution only: they do not establish critical behavior,
convergence, or a reproduction of any research result.

Outputs include the actual Git revision, Python/PyTorch versions, CUDA check,
training history, R(g) versus identity, coupling-to-g and loss reports. The script refuses
to overwrite an existing output directory. Coupling labels are not used by the
training loss. No long-running training is started by this test.

## Actual GPU training and same-cache timing

Prepare one cache, then copy that exact file to the server. Use fresh outputs:

```bash
python -m scripts.benchmark_qcr_device --prepare --output tmp/device_data
python -m scripts.benchmark_qcr_device --cache tmp/device_data/ensembles.pt \
  --device cpu --output tmp/device_cpu
python -m scripts.benchmark_qcr_device --cache /remote/path/ensembles.pt \
  --device cuda:0 --output tmp/device_cuda \
  --compare /remote/path/local_cpu_benchmark.json
```

The short benchmark performs 60 real default QCR updates, excluding the first
10 only from steady timing (not from training). It uses dimg2, hidden96, 3x3 C,
36→12→4, batch8x64, float64 and two CPU threads. Labels remain evaluation-only.
CUDA is synchronized at update timing boundaries. Setup, logging, checkpoints
and CPU reports are recorded separately; all R(g)-g, coupling and loss plots
are retained. CPU/CUDA random draws and library versions can differ, so this
is a performance comparison, not a bitwise physics reproduction.

For a normal default run:

```bash
python -m recurrent_rg train --config /path/to/config.json \
  --output /new/run --device cuda:0
```

No automatic CPU fallback is allowed after an explicit CUDA request. GPU
checkpoints save CUDA RNG together with Adam and CPU selection RNG state;
continuation must use the same device. CUDA atomic reductions can still be
numerically nondeterministic. CPU evaluations preserve both CPU and CUDA
training RNG streams instead of reseeding an active GPU.

## Execution-only GPU accelerations

Default CUDA execution now shares Q's learned binary pattern table across
the detached inputs in each update. It recomputes the table after every update,
never caches trainable outputs across updates, and retains C's cross-scale
straight-through input gradients and independent Bernoulli draws. Small motif
histograms use integer counts instead of allocating site-by-bin one-hot arrays.
Public inputs remain strictly validated; only already-validated binary trainer
inputs and internally normalized histograms skip repeated GPU value checks.
No model, objective, precision, coupling supervision, or gradient routing changes.

Small raw training caches can stay on GPU, bounded by one GiB and ten percent
of currently free memory. Larger caches retain CPU batch selection followed by
selected-batch transfer. CPU generators choose the same batch indices in either
case. This optional residency fallback is not a CPU training fallback.

Data-only empirical MI statistics are built on CPU once and saved beside the
input cache in `.qcr_mi_cache/` (or `--mi-cache-dir /path`). Keys include the
actual data SHA256, split, alphabet, dimensions, block, buffer, environment
offsets, pooling, augmentation, dtype and implementation version. Treat input
data caches as immutable while a run is starting. No C output/parameters are
cached; cache hits primarily reduce startup time, not per-update MI work.
Concurrent publication is atomic and corrupt files fail explicitly rather than
being overwritten. Historical data/checkpoints/results are not modified.

For a same-device, same-cache A/B audit, use separate fresh output directories:

```bash
python -m scripts.benchmark_qcr_device --cache /path/ensembles.pt \
  --device cuda:0 --reference-execution --output /new/reference
python -m scripts.benchmark_qcr_device --cache /path/ensembles.pt \
  --device cuda:0 --output /new/optimized \
  --compare-reference /new/reference/benchmark.json
```

The comparison validates execution flags, cache/config hashes, dtype, updates
and timing warmup, and separates update, setup and report timing. An unchanged
normal trainer does not add benchmark synchronization. Reduction order can
change floating-point rounding; equivalence is tested with explicit tolerances,
not a CPU/GPU bitwise promise. Exact CUDA resume must preserve the execution
mode: pre-optimization CUDA checkpoints require `--reference-execution`.
Categorical/historical paths without compatible binary lookup retain their
existing Q/C operations; static statistics caching and bounded residency do not
change their objectives. Optional legacy ablations remain guarded.

### Measured audit: RTX 3090, 2026-10-11

Optimization commit `292f750` was deployed on SSH225 and tested with the same
cache/config/seed, 60 real updates and 10-update timing warmup described above.
Cache SHA256: `169ba064b80b00becc3ab64573ca4a3315db30505351d4953d9d369e1d5f001c`.
Server artifacts: `tmp/cuda_optimization_292f750_seed91/`.

| Execution | Mean update | Median update | Setup |
| --- | ---: | ---: | ---: |
| Reference | 71.64 ms | 71.71 ms | 24.42 s |
| Optimized, MI cache miss | 57.17 ms | 57.18 ms | 21.06 s |
| Optimized, MI cache hit | 54.94 ms | 47.13 ms | 13.56 s |

The mean-update throughput improved **1.25–1.30x**. Median ratios were
1.25–1.52x, but the warm run had a broad timing distribution; these are short
audits, not a universal speedup promise. Cache hits do not accelerate the MI
algebra inside an update: CPU statistics build took 8.04 s in total, versus
0.49 s to read the cached statistics. Import/filesystem/clock warmth also affects
setup and CPU report timings, so whole-job changes cannot all be attributed to
the code optimizations.

Local targeted regression: 193 passed, 37 skipped. Server CUDA audit: 186 passed,
1 skipped (the unavailable-CUDA check). It includes objective/gradient, C input
VJP, stochastic RNG and resume comparisons. After the real 60-update A/B runs,
the maximum parameter difference was `2.22e-16`; maximum difference across the
recorded loss rows was `4.44e-16`. CPU, CUDA, batch-selection and MI RNG states
matched exactly. All execution, R(g)-g, coupling and loss reports were retained.
These short runs validate acceleration/equivalence, not physical convergence.
