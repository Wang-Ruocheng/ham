# Local direction consistency experiment

Optional loss configuration: `orientation_weight` (default 0),
`orientation_radius` (default .02). Implemented in
`recurrent_rg.losses.local_orientation_loss`.

At every observed R-input scale k=0,...,kmax-1, form g from the mean of
the two subset encodings, and evaluate deltaR=R(g+delta)-R(g).
The raw loss is the average squared negative part of
dot(delta,deltaR)/(norm(delta)*norm(deltaR)+1e-12).
Two-dimensional probes use 16 equally spaced circle directions, with
radius .02 times the detached per-scale RMS Euclidean spread (floor1e-3).
No RNG is consumed. Probe centers remain differentiable through Q/C;
this is joint regularization, not fitting R with a frozen representation.
The terminal scale is excluded because it has no observed successor.

This is a coordinate-dependent orientation prior, not a continuity theorem,
spin symmetry constraint or contraction bound. Positive scalar expansions
and contractions both receive zero penalty. Sparse probes do not guarantee
all directions preserve orientation, and probes can leave the data manifold.
Collapsed output increments can evade the penalty; existing geometry and
closure terms remain necessary. In 1D (and pure isotropic reversal), the
cosine has flat regions with little useful gradient; the present experiment
targets dim_g=2, not a validated scalar optimization prescription.

The controlled 2D Ising trial uses weight .1, seed91, 2400 joint updates,
random Bernoulli C, otherwise unchanged relative_local/variance_floor2
settings. No additional C orientation term is included. Inspect full kernel
probabilities, representation rank, DR eigenvalues and closure errors rather
than treating a lower orientation loss as success.

```bash
python -m recurrent_rg train --config configs/ablations/ising2d_g2_bernoulliC_orientation0p1_relative_variance2_2400.json --output results/ising2d_g2_bernoulliC_orientation0p1_relative_variance2_2400_seed91
python -m scripts.report_joint_isotropy --run results/ising2d_g2_bernoulliC_orientation0p1_relative_variance2_2400_seed91
python -m scripts.audit_orientation_run --run results/ising2d_g2_bernoulliC_orientation0p1_relative_variance2_2400_seed91
```

Use a fresh output directory to repeat training. Tests cover positive scaling,
negative eigen-directions, finite joint gradients, zero-weight compatibility,
configuration validation and RNG preservation.
