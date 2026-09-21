"""Write reproducible experiment provenance and results for the two-parameter run."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    summary = json.loads((args.run/'summary.json').read_text())
    result = json.loads((args.run/'field_diagnostics.json').read_text())
    cfg = summary['config']
    last = summary['history'][-1]
    axes_path = args.run/'axis_diagnostics.json'
    axes_report = json.loads(axes_path.read_text()) if axes_path.exists() else None
    rows = '\n'.join(f"| {k} | {result['metrics']['train']['rollout_nrmse'][k]:.6f} | {result['metrics']['test']['rollout_nrmse'][k]:.6f} |" for k in range(1, 4))
    root_lines = '\n'.join(f"- F{i+1}: g={r['g']}, {r['stability']}, DR eigenvalues={r['eigenvalues']}; distance to observed R inputs={r['distance_to_sampled_R_input']:.6g}." for i, r in enumerate(result['fixed_points']))
    text = f'''# Two-parameter 2D classical Ising: joint QCR, dim(g)=2

## Model and data

Probability proportional to exp(K sum_nn s_i s_j + B sum_i s_i), K=beta J, B=beta h.
At fixed temperature, scanning these is equivalent to varying J and h independently.
Ferromagnetic isotropic nearest-neighbor lattice, periodic 32x32 spins.
K grid: 33 values in [0.20,0.68], with the closest point replaced by exact Kc=log(1+sqrt(2))/2.
B grid: [-0.2,-0.1,-0.05,-0.02,-0.008,-0.002,0,0.002,0.008,0.02,0.05,0.1,0.2].

429 parameter cells, 3 independently sampled replicas per cell, 256 configurations per replica.
Total 329472 configurations (previous zero-field run: 24576); spins per configuration remain 1024.
Training: 286 cells / 858 ensembles / 219648 configurations.
Test: 143 cells / 429 ensembles / 109824 configurations.
Whole cells and all their replicas are held out together, on a deterministic checker pattern symmetric in B.
The exact (Kc, B=0) cell is held out for testing.
Positive/negative-field chains are independently sampled, not mirrored.

## Sampling validity

Finite-field Swendsen-Wang update: bonds between equal neighbors are occupied with probability 1-exp(-2K).
Each connected cluster of n sites is independently recolored + with probability sigmoid(2 B n).
512 thermalization sweeps; 16 full sweeps between draws, fixed counts rather than state-dependent stopping.
Pilot chains from random, all-plus and all-minus starts: 1024 recorded configurations for each of six (K,B) choices.
Their largest pairwise mean discrepancy across the tested observables was approximately 2.03 estimated standard errors.
Saved-chain lag-1 autocorrelations across the full dataset had median near zero and 95th percentile absolute value about 0.12.
These are finite diagnostics, not a proof that every observable is independent or equilibrated.

Sampler tests compare positive, zero and negative fields with exact 2x2 enumeration, and K=0 magnetization with tanh(B).
General finite-field cluster-dynamics reference: https://arxiv.org/abs/2205.01985 .
The exact implementation and tests are in `recurrent_rg/ising2d_field.py` and `tests/test_ising2d_field.py`.
Full diagnostics are in `{Path(cfg['data_cache']).with_suffix('.json')}`.

## Training contract

One seed ({cfg['seed']}), {cfg['epochs']} Adam updates, lr=0.001, system batch 32, sample batch 256.
No coupling/field labels in loss; no RSMI; no explicit Z2 architecture constraint; no action training.
Q+C+R has {summary['optimized_parameter_count']} optimized parameters; total instantiated count {summary['parameter_count']} includes unused action parameters.
Same hidden=59, dim(g)=2, local Q width 3x3 and learned nonoverlapping C blocks 2x2 as the previous zero-field experiment.
Scales: 32 -> 16 -> 8 -> 4. The final 4x4 scale is particularly susceptible to finite-size effects.
Binary-pattern lookup evaluates the same networks and retains first-order gradients through C.

Loss remains 2 LQ + 5 Lsplit + 2 LC + 10 Lsep + 5 Lcomm.
LC balances the global batch marginal, not the magnetization of each ensemble.
The parameter distribution includes both signs of B, but batches are ordinary random system batches, not symmetry-projected batches.

Critical finite-field change: NO per-ensemble global-spin-flip augmentation, including at B=0 for a uniform data pipeline.
Pairing s and -s within a fixed nonzero-B ensemble would change its distribution and erase field information.
Each Q view now receives 128 independent saved configurations rather than 64 configurations plus their flips.
Logged loss uses 48 fixed train/test systems and 128 configurations, so absolute train/test-loss comparisons to earlier logs are not controlled.
Relative to the previous zero-field experiment, data coverage, saved samples per ensemble and update count increase; parameter count is unchanged.
This is an exploratory experiment, not a controlled ablation isolating only the effect of adding B.

## Results

Final train loss: {last['train_total']:.9g}; final test loss: {last['test_total']:.9g}.
The logged curve is not smoothly converged: at update 920 test loss rose to about 6.04, dominated by commutator error and coarse marginal imbalance, and later recovered. No checkpoint selection or extra seed was used.
Full-ensemble metrics use every saved configuration and every train/test ensemble without flip augmentation.
NRMSE divides per-coordinate rollout RMSE by pooled coordinate SD at the corresponding scale.

| RG step | Train NRMSE | Test NRMSE |
|---|---:|---:|
{rows}

Test one-step error divided by the identity-map baseline error: {result['metrics']['test']['one_step_error_relative_to_identity']}.
Values below one indicate better one-step prediction than R(g)=g; this guards against treating small loss from almost-stationary encodings as successful dynamics.

Axis-resolved rollout diagnostics: {axes_report}.
These explicitly test the weak direction rather than hiding it under the variance of the dominant direction.

Microscopic covariance eigenvalues: {result['microscopic_covariance_eigenvalues']}.
Second-PC variance fraction: {result['microscopic_second_pc_fraction']:.9g}.
Participation rank: {result['microscopic_participation_rank']:.9g}.
Global PCA rank is not an intrinsic-dimension or sufficiency proof; a curved one-dimensional representation can also have nonzero second PC.
Replica signal/noise diagnostic, smallest-PC first: {result['replica_diagnostic']}.
An extremely thin second direction may still carry reproducible information; conversely, nonzero variance can be sampling noise.
Post-hoc quadratic readout test R2: {result['posthoc_quadratic_observable_readout_test_R2']}.
This readout fits measured observables on training cells only after Q is frozen; it never supplies a training signal to QCR.

Critical microscopic data and its actual coarse trajectory:
{result['critical_direct_trajectory']}

Numerical roots within the plotted box (121 initial guesses, not exhaustive):

{root_lines or 'No root found in the searched box.'}

Kernel spin-flip-oddness violations: {result['kernel_oddness_violations']} / 16 input patterns.
The complete truth table is in `field_diagnostics.json`.

## Interpretation limits

`physical_parameter_map.png` overlays the KNOWN coexistence line B=0,K>Kc and the known critical point.
Those lines are references, not inferred phase boundaries. The colored g panels and latent grid are what the network learned.
At finite L=32 all equilibrium observables are rounded; no thermodynamic-limit phase-boundary extraction is claimed.
At nonzero B the sign of magnetization distinguishes field preference, not a new finite-temperature phase boundary.
Numerical R roots and nearest physical parameter cells are not an established physical fixed-point identification.
Closeness to sampled R inputs is a descriptive support test, not a confidence interval on R or its derivatives.
Recovering two microscopic parameters does not establish that two variables close the full RG theory space.

## Artifacts and reproduction

- `physical_parameter_map.png`: measured magnetization and both learned coordinates over (K,B).
- `learned_parameter_grid.png`: original parameter grid mapped to the g plane.
- `whitened_latent_grid.png`: PCA-rescaled view of the same grid; weak second direction is amplified, not newly learned.
- `R_vector_field.png`: discrete RG field and actual-versus-predicted trajectories.
- `loss_components.png`: weighted training and held-out loss terms.
- `field_diagnostics.json`, `field_flow_data.npz`: raw metrics and trajectories.
- `axis_diagnostics.json`: strong/weak-axis rollout errors and root support check.
- `qcr_model.pt`: final Q/C/R weights.

```sh
python scripts/prepare_ising2d_field_qcr.py --output {cfg['data_cache']}
python experiments/joint_qcr_ising2d_field.py --cache {cfg['data_cache']} --output-dir {args.run} --epochs {cfg['epochs']} --seed {cfg['seed']}
MPLBACKEND=Agg python scripts/plot_ising2d_field_qcr.py --run {args.run}
python scripts/diagnose_field_qcr_axes.py --run {args.run}
python scripts/report_ising2d_field_qcr.py --run {args.run}
```

Skip preparation if reusing the cache; the preparation script refuses to overwrite an existing cache.
'''
    (args.run/'REPORT.md').write_text(text)
    print(args.run/'REPORT.md')


if __name__ == '__main__':
    main()
