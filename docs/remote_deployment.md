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
