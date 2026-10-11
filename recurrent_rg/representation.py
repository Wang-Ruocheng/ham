"""Label-free distribution reconstruction for ensemble representations."""

import torch
from torch import nn
from torch.nn import functional as F


@torch.no_grad()
def normalize_affine_decoder_(encoder):
    """Unit decoder columns fix coordinate scales without forcing full rank."""
    if getattr(encoder, 'distribution_reconstruction_mode', '') != 'affine_mse':
        return
    decoder = encoder.distribution_decoder
    decoder.weight.div_(decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-12))


@torch.no_grad()
def initialize_affine_decoder_pca_(encoder, train_histograms):
    """Explicit ablation: initialize a scalar reconstruction axis from TRAIN only.

    The caller supplies probabilities, not coupling labels. This is only an
    initialization: the decoder remains trainable, and Q is not replaced by PCA.
    """
    if getattr(encoder, 'distribution_reconstruction_mode', '') != 'affine_mse':
        raise ValueError('PCA initialization requires the scalar affine decoder')
    decoder = encoder.distribution_decoder
    if decoder.in_features != 1:
        raise ValueError('PCA initialization ablation currently supports scalar g only')
    if train_histograms.ndim != 2 or train_histograms.shape[1] != decoder.out_features:
        raise ValueError('Expected a matrix of training histograms')
    p = train_histograms.detach().to(decoder.weight)
    if len(p) < 2 or not torch.isfinite(p).all() or (p < 0).any():
        raise ValueError('Invalid training histograms')
    if not torch.allclose(p.sum(-1), torch.ones(len(p), dtype=p.dtype, device=p.device)):
        raise ValueError('Training histograms must sum to one')
    center = p.mean(0)
    _, singular, vh = torch.linalg.svd(p-center, full_matrices=False)
    if singular.square().sum() <= 1e-15:
        raise ValueError('Training histograms have no variance')
    axis = vh[0]
    # Orientation is a gauge, chosen from the existing decoder, never K labels.
    if axis @ decoder.weight[:, 0] < 0:
        axis = -axis
    decoder.weight.copy_(axis[:, None])
    decoder.bias.copy_(center)
    return dict(training_histograms=len(p),
                explained_variance_fraction=float(singular[0].square()/singular.square().sum()),
                labels_used=False, decoder_remains_trainable=True)


def linear_distribution_reconstruction_loss(
    decoder: nn.Linear,
    theory: torch.Tensor,
    target: torch.Tensor,
    *,
    mode: str = "squared",
    validate: bool = True,
) -> torch.Tensor:
    """Reconstruct a detached histogram through an affine latent decoder.

    ``theory`` has shape ``[..., dim]`` and ``target`` has shape
    ``[..., bins]``. Both objectives sum over bins, then average over all
    leading dimensions; changing the alphabet therefore does not silently
    change the objective to a per-bin mean.

    ``squared`` reconstructs probabilities with the raw affine output. It
    is ordinary linear reconstruction, without an output simplex constraint.
    ``kl`` interprets that output as multinomial logits and computes
    KL(target || softmax(logits)). Zero target probabilities are supported.
    Targets never receive gradients. The caller controls gradients into the
    encoder supplying ``theory``, including any upstream coarse-graining.
    """
    if not isinstance(decoder, nn.Linear):
        raise TypeError("Distribution decoder must be nn.Linear")
    if mode not in ("squared", "kl"):
        raise ValueError("Reconstruction mode must be 'squared' or 'kl'")
    if theory.ndim < 1 or target.ndim != theory.ndim:
        raise ValueError("Theory and histogram must have matching batch dimensions")
    if theory.shape[:-1] != target.shape[:-1]:
        raise ValueError("Theory and histogram batch shapes differ")
    if theory.shape[-1] != decoder.in_features or target.shape[-1] != decoder.out_features:
        raise ValueError("Decoder dimensions do not match theory and histogram")

    probabilities = target.detach()
    # Public calls validate strictly. The default CUDA trainer can waive only
    # repeated numeric checks for its internally counted, normalized motifs;
    # shape checks above and per-update finite-loss checks remain mandatory.
    if validate:
        with torch.no_grad():
            if not torch.isfinite(probabilities).all() or (probabilities < 0).any():
                raise ValueError("Target histograms must be finite and nonnegative")
            total = probabilities.sum(dim=-1)
            if not torch.allclose(total, torch.ones_like(total), atol=1e-5, rtol=1e-5):
                raise ValueError("Target histograms must sum to one")

    prediction = decoder(theory)
    if mode == "squared":
        return (prediction - probabilities).square().sum(dim=-1).mean()
    log_probabilities = F.log_softmax(prediction, dim=-1)
    return (
        torch.xlogy(probabilities, probabilities) - probabilities * log_probabilities
    ).sum(dim=-1).mean()
