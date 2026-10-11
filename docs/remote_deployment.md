# Remote execution smoke test

The deployment syncs a verified Git commit, not the dirty local working tree.
Generated caches, checkpoints and historical results are not part of the source
sync. Use a new, project-local environment; do not change the shared server's
Python environments, NVIDIA driver or other jobs.

Current QCR training is **CPU-only**: the trainer does not yet provide device
routing. A successful CUDA matrix check is a separate hardware/runtime check,
not evidence of GPU QCR training.

## Isolated environment

With a pre-existing Python 3.10+ interpreter, run from the repository:

```bash
bash scripts/setup_remote_env.sh /absolute/path/to/python /absolute/path/to/ham/.env
```

The script refuses to overwrite an existing environment. It installs the
[official PyTorch 2.7.1 CUDA 11.8 build](https://pytorch.org/get-started/previous-versions/)
without installing torchvision or torchaudio. CUDA driver compatibility must
still be checked by actual execution; see
[NVIDIA's compatibility limitations](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

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
training history, R(g), R(g)-g, coupling-to-g and loss reports. The script refuses
to overwrite an existing output directory. Coupling labels are not used by the
training loss. No long-running training is started by this test.
