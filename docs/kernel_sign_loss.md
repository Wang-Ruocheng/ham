# Optional uniform-block sign convention

New configurations (schema version 4) default `loss.kernel_sign_weight` to 0.
Version 3 retains its historical default 0.1 when the field is absent;
versions 1/2 retain 0. Explicit values are always preserved.
Fixed/deterministic kernels loaded from configuration default this term to 0.
Direct Python `LossConfig()` also defaults to weight 0.
For a learnable single-channel Bernoulli C with logits z and
P(+|x)=sigmoid(2z(x)), the raw penalty is

L_sign = [softplus(2 z(all-minus)) + softplus(-2 z(all-plus))]/2.

This is exactly minus the average log probability of retaining the two
uniform signs. It uses no Monte Carlo draws and no coupling/energy labels.
Gradients go directly to C. Since the same C is reused at every RG scale,
the penalty is counted once per training batch rather than multiplied by
the number of scales. It is a physical orientation convention, not a fully
prior-free loss, and does not constrain mixed blocks, enforce Z2 symmetry,
or guarantee the absence of RG cycles. C remains randomly initialized and
trainable. This is separate from the failed R local-direction penalty.

Configurations/checkpoints predating this option retain weight0 on resume.
The implementation rejects deterministic or fixed kernels for this
probabilistic NLL interpretation. Probability calculations use softplus
directly, avoiding log-of-zero at saturated probabilities.

Initial controlled experiment:
`configs/ablations/ising1d_g2_bernoulliC_sign0p1_relative_variance2_2400.json`.
Same seed91/data/2400 joint updates as the sign-weight0 baseline, with no
R orientation regularizer. Inspect C probabilities, R/R2 fixed points and
cycles, representation geometry and closure together.
