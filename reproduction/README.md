# Fixed-point SU(3) gauge-action reproduction

This directory contains a compact Apple-silicon reproduction of selected
checks from *Machine-Learned Renormalization-Group-Improved Gauge Actions and
Classically Perfect Gradient Flows* (PRL 136, 031901, 2026).

```bash
conda activate ham
python reproduction/reproduce.py all
```

The workflow validates the released fixed-point action, local SU(3) gauge
invariance, force finite differences, and the published continuum-limit fit.
It does not regenerate the paper-scale HMC training ensembles. The local
upstream checkout is pinned under `third_party/lge-cnn/`; its CUDA workflow is
not part of the recurrent RG package.

