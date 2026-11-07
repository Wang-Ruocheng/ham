"""Opt-in weak-KL ensemble Q, shared across scales; not a QCR default.

The encoder consumes complete categorical plaquette probabilities. Coupling
labels are never inputs. Cross-half CE predicts normalized local statistics,
not a full-configuration likelihood. R always sees deterministic posterior means.
"""
import torch
from torch import nn
from .data.statistics import categorical_plaquette_distribution, plaquette_distribution
from .probability_normalization import stable_log_softmax, stable_softmax


class EnsembleVAEQ(nn.Module):
    distribution_reconstruction_mode = 'ensemble_vae_cross_subset'
    kl_weight = .01
    warmup_steps = 600

    def __init__(self, theory_dim=2, states=4):
        super().__init__()
        self.categorical_states = states
        bins = states**4
        self.register_buffer('center', torch.full((bins,), 1./bins, dtype=torch.double))
        self.register_buffer('scale', torch.ones(bins, dtype=torch.double))
        self.register_buffer('objective_step', torch.tensor(0, dtype=torch.int64))
        self.register_buffer('noise_seed', torch.tensor(torch.initial_seed() % (2**62), dtype=torch.int64))
        self.encoder = nn.Sequential(nn.Linear(bins,96), nn.LeakyReLU(.1),
                                     nn.Linear(96,96), nn.LeakyReLU(.1))
        self.mean = nn.Linear(96,theory_dim)
        self.logvar = nn.Linear(96,theory_dim)
        self.distribution_decoder = nn.Sequential(nn.Linear(theory_dim,64),nn.LeakyReLU(.1),
            nn.Linear(64,64),nn.LeakyReLU(.1),nn.Linear(64,bins))
        self.double()
        with torch.no_grad():
            self.mean.weight.normal_(0,.02); self.mean.bias.zero_()
            self.logvar.weight.zero_(); self.logvar.bias.fill_(-2.)
            self.distribution_decoder[-1].weight.normal_(0,.02)
            self.distribution_decoder[-1].bias.copy_(self.center.log())

    def encode(self, probabilities):
        h = self.encoder((probabilities-self.center)/self.scale)
        return self.mean(h), self.logvar(h).clamp(-8,3)

    def decode_probabilities(self, latent):
        return stable_softmax(self.distribution_decoder(latent))

    def probabilities(self, configurations):
        # Counts are detached explicitly, including when called outside the
        # independent-gradient trainer. Q cannot optimize C through this path.
        x = configurations.detach()
        if self.categorical_states == 2 and x.shape[-1] == 1:
            return plaquette_distribution(x)
        if x.shape[-1] != self.categorical_states:
            raise ValueError('Field channels do not match the VAE motif alphabet')
        return categorical_plaquette_distribution(x)

    def forward(self, configurations):
        return self.encode(self.probabilities(configurations))[0]

    @torch.no_grad()
    def calibrate(self, train_histograms):
        p = train_histograms.detach().to(self.center)
        if (p.ndim != 2 or p.shape[1] != len(self.center) or len(p) < 2
                or not torch.isfinite(p).all() or (p < 0).any()
                or not torch.allclose(p.sum(-1),torch.ones(len(p),device=p.device,dtype=p.dtype))):
            raise ValueError('Finite normalized TRAIN motif histograms required')
        self.center.copy_(p.mean(0))
        self.scale.copy_(p.std(0,unbiased=False).clamp_min(.001))
        self.distribution_decoder[-1].bias.copy_(self.center.clamp_min(1e-6).log())

    @torch.no_grad()
    def set_objective_step(self, step):
        if type(step) is not int or step < 0: raise ValueError('Nonnegative update required')
        self.objective_step.fill_(step)

    def cross_subset_objective(self, first, second, *, scale_index=0, epsilon=None):
        a, b = first.detach(), second.detach()
        mu, lv = self.encode(torch.cat([a,b]))
        if epsilon is None:
            # Private, checkpointed update/scale-indexed noise. Adding VAE Q
            # does not consume the global C-sampling or data-selection stream.
            gen = torch.Generator().manual_seed(
                (int(self.noise_seed)+(int(self.objective_step)+1)*104729+scale_index) % (2**63-1))
            epsilon = torch.randn(mu.shape,generator=gen,dtype=torch.double).to(mu)
        z = mu+(lv*.5).exp()*epsilon
        target = torch.cat([b,a])
        log_probabilities = stable_log_softmax(self.distribution_decoder(z))
        ce = -(target*log_probabilities).sum(-1).mean()
        kl = .5*(mu.square()+lv.exp()-lv-1).sum(-1).mean()
        coefficient = self.kl_weight*min(int(self.objective_step)/self.warmup_steps,1.)
        decoded = self.decode_probabilities(mu)
        mse = (decoded-target).square().sum(-1).mean()
        return ce+coefficient*kl, dict(vae_cross_entropy=ce,vae_KL=kl,
            vae_effective_KL_weight=ce.new_tensor(coefficient),reconstruction_mse=mse,
            decoder_probability_sum_error=(decoded.detach().sum(-1)-1).abs().max(),
            decoder_log_probability_sum_error=(log_probabilities.detach().exp().sum(-1)-1).abs().max())


@torch.no_grad()
def calibrate_from_train(encoder, configurations, chunk_size=8):
    """One microscopic TRAIN calibration, frozen thereafter across ALL scales."""
    hist = []
    for start in range(0,len(configurations),chunk_size):
        x = configurations[start:start+chunk_size].to(encoder.center)
        hist.append(encoder.probabilities(x))
    encoder.calibrate(torch.cat(hist))
    return dict(physical_chains=len(configurations), fitting_split='TRAIN', scale=0,
                std_floor=.001, decoder_bias_probability_floor=1e-6,
                labels_used=False, no_C_forward=True, no_pretraining=True,
                frozen_across_scales=True)
