# Auxiliary experiments

`rsmi_ising_1d.py` is a self-contained exact-enumeration check of a learned
one-bit block-spin kernel. It is intentionally separate from the main recurrent
RG training pipeline:

```bash
python experiments/rsmi_ising_1d.py
```

The default computes the small-chain joint distribution by exact enumeration.
To train the same RSMI kernel from configurations alone, use the empirical
joint distribution:

```bash
python experiments/rsmi_ising_1d.py \
  --joint-source empirical \
  --samples 100000
```

Historical intermediate Ising prototypes live under `archive/legacy_ising/`.

`ensemble_theory_encoder_ising_1d.py` isolates the existing ensemble encoder
`g = Q(D)` from the full model. It learns a scalar theory coordinate using only
split-ensemble consistency, noncollapse, and distances between empirical local
pattern distributions. Couplings and observables are used only after training:

```bash
python experiments/ensemble_theory_encoder_ising_1d.py
```

`joint_qcr_ising_1d.py` initializes from that encoder and then jointly updates
Q, the hard binary coarse-grainer C, and the shared theory-flow cell R. It uses
one-step anti-collapse and the commuting diagram, with no RSMI or action loss:

```bash
python experiments/joint_qcr_ising_1d.py
```

`analyze_qcr_basins.py` locates fixed points of the learned scalar R map,
evaluates their linear stability, and separates the data-supported input range
from neural-network extrapolation:

```bash
python experiments/analyze_qcr_basins.py
```

`joint_qcr_ising_2d.py` is the corresponding zero-field square-lattice
experiment. It uses no RSMI, action target, coupling label, or hard-coded RG
kernel in its default learned-C mode. Each finite ensemble view is exactly
paired as `(X, -X)` so that Q cannot use the sign of spontaneous magnetization
as a shortcut. The sampler also randomizes the recorded global spin
orientation, an exact zero-field Z2 operation that prevents ordered-sector
locking. Select the fixed majority control with `--kernel majority`:

```bash
python experiments/joint_qcr_ising_2d.py --kernel learned
python experiments/joint_qcr_ising_2d.py \
  --kernel majority \
  --output-dir results/fixed_majority_qr_ising_2d
```

`unsupervised_recurrent_rg_ising_1d.py` jointly trains the RSMI kernel and the
theory encoder/flow/action modules from configurations only. Couplings are kept
out of the training objective and are loaded only for post-training evaluation:

```bash
python experiments/unsupervised_recurrent_rg_ising_1d.py
```

The corresponding fully label-free 2D Ising pipeline uses periodic Wolff
sampling and the dimension-independent recurrent-RG core with square-lattice,
binary-field, and binary-local-conditional adapters. Its RSMI loss is a neural
InfoNCE bound rather than a finite binary environment histogram:

```bash
python experiments/unsupervised_recurrent_rg_ising_2d.py
```
