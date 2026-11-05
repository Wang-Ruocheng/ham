"""Explicit per-configuration CNN encoder ablation of historical b3 phi4 QCR.

Only Q's encoder changes. The shared unit-column affine decoder, physical
descriptor, random Gaussian C, independent gradients, and all-start R remain.
This is not a change to the package's default models or historical checkpoints.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .phi4_aligned_qcr import AlignedPhi4QCR, FEATURE_DIM


class ConfigurationCNNQ(nn.Module):
    def __init__(self, scale, frame_chunk=64, patch_chunk_size=262144, *, theory_dim=1):
        super().__init__()
        if scale <= 0 or frame_chunk < 1 or patch_chunk_size < 1:
            raise ValueError("Positive normalization and chunk sizes required")
        if type(theory_dim) is not int or theory_dim < 1:
            raise ValueError("theory_dim must be a positive integer")
        self.frame_chunk = frame_chunk
        self.patch_chunk_size = patch_chunk_size
        self.register_buffer("scale", torch.tensor(float(scale), dtype=torch.float64))
        self.spatial = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, padding_mode="circular"), nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=2, dilation=2, groups=16,
                      padding_mode="circular"), nn.SiLU()).double()
        self.frame_head = nn.Sequential(nn.Linear(32, 32), nn.SiLU(),
                                        nn.Linear(32, 16), nn.SiLU()).double()
        self.head = nn.Sequential(nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, theory_dim)).double()
        self.decoder = nn.Linear(theory_dim, FEATURE_DIM).double()
        with torch.no_grad():
            self.head[-1].weight.normal_(0., .01)
            self.head[-1].bias.zero_()
        self.normalize_decoder_()

    def frame_features(self, frames):
        value = self.spatial(frames / self.scale)
        pooled = torch.cat((value.mean((-2, -1)), value.square().mean((-2, -1))), -1)
        return self.frame_head(pooled)

    def forward(self, fields):
        if fields.ndim != 5 or fields.shape[-1] != 1 or fields.shape[2] != fields.shape[3]:
            raise ValueError("Expected [ensembles, configurations, L, L, 1]")
        e, n, side, _, _ = fields.shape
        if n < 1 or side < 2:
            raise ValueError("Nonempty ensemble and side>=2 required")
        frames = fields.detach().reshape(e*n, side, side, 1).permute(0, 3, 1, 2).contiguous()
        pieces = [checkpoint(self.frame_features, part, use_reentrant=False)
                  if torch.is_grad_enabled() else self.frame_features(part)
                  for part in frames.split(self.frame_chunk)]
        value = torch.cat(pieces).reshape(e, n, 16)
        return self.head(torch.cat((value.mean(1), value.var(1, unbiased=False)), -1))

    def decode(self, g):
        return self.decoder(g)

    @torch.no_grad()
    def normalize_decoder_(self):
        self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-12))


class CNNPhi4QCR(AlignedPhi4QCR):
    def __init__(self, scale, seed=91, frame_chunk=64, *, theory_dim=1):
        # Preserve the historical random C/R initialization stream. The old Q
        # is discarded untrained, and a separate fresh CNN Q is constructed.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            super().__init__(scale, hidden=96, theory_dim=theory_dim, local_features=12,
                             patch_chunk_size=262144, block_size=3, steps=2)
            torch.manual_seed(seed)
            self.Q = ConfigurationCNNQ(scale, frame_chunk=frame_chunk, theory_dim=theory_dim)
