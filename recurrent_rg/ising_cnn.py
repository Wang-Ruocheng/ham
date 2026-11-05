"""Explicit binary Ising adapter of the per-configuration phi4 CNN encoder."""
from .phi4_cnn_qcr import ConfigurationCNNQ


class IsingConfigurationCNN(ConfigurationCNNQ):
    def __init__(self, geometry, *, theory_dim=1):
        # Spins already have fixed unit amplitude. No per-frame normalization.
        super().__init__(1., frame_chunk=64, theory_dim=theory_dim)
        del self.decoder  # The standard builder installs the 16-bin affine decoder.
        self.geometry = geometry
        self.binary_lookup = False  # CNN features cannot use the local motif Q table.

    def forward_many(self, configurations, **kwargs):
        return [self(x) for x in configurations]

    def decode(self, g):
        return self.distribution_decoder(g)
