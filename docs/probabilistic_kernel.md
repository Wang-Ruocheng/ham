# Default probabilistic binary C

New ModelConfig and schema-version-2 configurations default to random C
initialization and Bernoulli sampling. The unsuccessful local orientation
regularizer remains off by default. Explicitly specify the current preset with:

```json
"model": {
  "kernel_initialization": "random",
  "kernel_sampling": "bernoulli",
  "kernel_temperature": 1.0
}
```

Schema-version-1 configs (including files without a schema version) retain
their historical deterministic default when sampling is omitted. Old saved
checkpoints are also interpreted deterministically if they lack this field.
This prevents changing historical experiments silently. Fixed C requires
deterministic sampling; set it explicitly when constructing ModelConfig in Python.

For each block, C computes logit z and samples a binary output with P(+1)=sigmoid(2z). Draws are independent between blocks, samples, and RG steps. Sampling stays active in evaluation; inference must not silently replace the kernel by a deterministic threshold.

The implementation draws logistic noise eta=log(u)-log(1-u), forms soft=tanh((z+eta/2)/temperature), and uses sign(z+eta/2) in the forward pass with the soft derivative in the backward pass. This is a **biased straight-through relaxation**, not an unbiased gradient of the discrete expected loss. Temperature changes only the relaxation gradient, not the Bernoulli probability. The probability law changes continuously with parameters, but individual draws remain discrete and noisy.

Binary lookup computes deterministic logits for each pattern and expands them to all block occurrences **before drawing noise**. Sampling inside the pattern table would incorrectly correlate all identical blocks; tests guard against this. Q's binary-feature lookup remains valid because forward samples remain +/-1.

The CPU training evaluator uses a forked RNG with seed 12345, preserving the training sampling RNG. `collect_trajectories(..., sampling_seed=12345)` likewise provides reproducible coarse trajectories without consuming training randomness. Different `sampling_seed` values audit inference sampling noise; they are not independent training seeds. Deterministic model behavior is unchanged. Old checkpoint configs default to deterministic sampling; model parameter keys/count are unchanged. Fixed decimation rejects stochastic sampling.

Scalar reports include P(+1) for each pattern; a single sampled output table is explicitly labeled as a random draw, not a truth table. They repeat coarse-trajectory evaluation with five sampling seeds. R itself and microscopic Q(D) remain deterministic once trained.

First experiment: `configs/ablations/tfim1d_g1_bernoulliC_relative_variance2_2400.json`. Result and limitations: `results/tfim1d_g1_bernoulliC_relative_variance2_2400_seed91/REPORT.md`.

Matched 2D Ising test: `configs/ablations/ising2d_g2_bernoulliC_relative_variance2_2400.json`. Result: `results/ising2d_g2_bernoulliC_relative_variance2_2400_seed91/REPORT.md`. It has a central discrete repeller but retains an alternating-sign unstable direction; stochastic C does not impose an orientation convention.
