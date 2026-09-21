# Square-lattice quantum TFIM, scalar QCR, seed 91

## Scope and data

Pauli convention H=-J sum_<ij> Z_i Z_j - h sum_i X_i, J=1, no longitudinal
field. Periodic spatial L=16; beta J=16, dtau J=.025, 640 time slices.
Only equal-time 16x16 Z snapshots enter Q/C/R. This is not classical K/B
Ising, and the time slices are not treated as independent training samples.

64 uniformly spaced h/J values from .5 to 5; three independently seeded
Swendsen-Wang chains per value; 256 warmup sweeps, eight sweeps between each
of 128 saved configurations. 129 train/63 test ensembles, 16512/8064 raw
configurations. All replicas of a coupling stay in one split; coupling sets
do not overlap. The theoretical critical coupling is not inserted in the grid.
The cache is `data_cache/tfim2d_sw_v1_L16_M640_dtau0p025_eq256_gap8_K64_rep3_n128_seed91.pt`;
its JSON sidecar records configuration, chain identifiers and autocorrelation diagnostics.

## Sampling checks

The initial dtau=.1, beta=8 pilot was rejected for visibly large Trotter errors
near the transition. The retained .025/beta16 pilot compares hot/cold starts,
dtau=.0125 at fixed beta16, and beta32 at fixed dtau=.025 at h/J=2,3.04438,4.
For bond correlation and m2, all comparisons with the baseline fall within
1.32 combined estimated standard errors. Each pilot has 256 saved samples;
agreement is a limited statistical check, not a continuum or ground-state proof.
Full details: `sampler_pilot.json`. Three sampler tests cover periodic neighbors,
reproducibility, exact enumeration of a small Trotter lattice and small-dtau
comparison to exact diagonalization of a quantum Hamiltonian (the first test
combines topology and reproducibility).

## Training contract

dim(g)=1, hidden60, 10191 optimized Q/C/R parameters; local learned nonoverlapping
2x2 C, local 3x3 Q features, shared autonomous residual R. Spatial scales16,8,4
and all available rollout horizons1,2 enter the unchanged data_geometry loss.
Weights: split0=20, coarse split=5, C balance=2, geometry=10, one-step closure=5,
multistep closure=5. No action, RSMI, coupling labels in the loss, or hard Z2
constraint; global spin flips are data augmentation. Adam lr=.001, 2000
predeclared updates, system_batch32/sample_batch128. Evaluation loss uses
24 systems with 48 original samples plus flips; scalar reports evaluate all
ensembles with 128 original samples plus flips. Beyond k=2 is extrapolation.

## Interpretation

The zero-temperature, infinite-size reference is h/J approximately3.04438,
nu approximately.629971, from
https://pmc.ncbi.nlm.nih.gov/articles/PMC9481121/ . It is a reporting reference,
not a label or enforced training target. Finite beta, finite L, snapshot-only
coarse-graining and noninvertible latent encoding can invalidate direct physical
interpretation of log(2)/log(R'(g*)). This is an exploratory single-seed run.

## Completed results

| Updates | Logged train loss | Logged test loss | Full-test NRMSE k1 | k2 |
|---|---:|---:|---:|---:|
| 1000 | .0752533 | .0765092 | .0337676 | .0617277 |
| 2000 | .0680631 | .0713779 | .0286401 | .0537815 |

At2000 the unstable root is g=-1.886425195 with R'=1.433307986,
y=log2(R')=.519348646, conditional nu_eff=1.925488799. At1000 the corresponding
numbers are g=-1.946413626, R'=1.446488702 and nu_eff=1.877740135. Both roots
are within the observed scalar coordinate range. The plotted final stable root
g=.358932210, R'=.253539610 is just outside observed support. No second stable
root was found in the plotted interval; this is not an exhaustive global search.

The final g0(h/J) interpolation crosses the unstable root at h/J=3.128862005,
about2.78% above the zero-temperature reference. This inverse interpolation
is not a physical critical-point identification. At the reference3.04438,
the interpolated direct Q coordinates across k0,1,2 are
[-1.800312947,-1.793616994,-1.693936417], so they do not remain at the root.
The reference field is not a sampled full-dataset grid value; these are
interpolated means (not a fresh critical-field Q evaluation).

Figures: `step2000/R_vs_g.png`, `step2000/coupling_vs_g.png`,
`loss_and_closure.png`; earlier figures are in `step1000/`. The first curve
is R(g); the companion panel is R(g)-g. Coupling plots show actual Q(C^kD),
not autonomous R rollouts. `critical_exponent_audit.json` records fixed-point
residuals and finite-difference/autograd agreement. The final state is in
`qcr_model.pt`, with optimizer/RNG checkpoints every500 updates.

Validation: all30 targeted sampler, coordinate-reduction, data-geometry and
nonoverlapping-kernel tests passed; git diff --check passed. All three final
figures were visually inspected. The physical critical exponent has not been
recovered despite low value-closure errors.
