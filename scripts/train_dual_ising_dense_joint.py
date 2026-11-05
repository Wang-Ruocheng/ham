"""From-scratch joint run with equivalent dense VJP at coarse categorical levels.

Lookup remains enabled for microscopic inputs. If the input requires a C
gradient, ordinary dense MLP evaluation avoids the lookup's explicit Jacobian.
This changes execution only, not the model, loss, or straight-through estimator.
"""
from pathlib import Path
from recurrent_rg.config import RunConfig
from recurrent_rg.model_nd import EnsembleTheoryEncoder
from recurrent_rg.runner import run_training


def install_dense_coarse_vjp():
    original=EnsembleTheoryEncoder.forward
    def forward(self, configurations):
        if not (getattr(self,'categorical_states',0) and configurations.requires_grad):
            return original(self,configurations)
        patches=self.geometry.extract_patches(configurations)
        spatial_dims=tuple(range(1,patches.ndim-1))
        return self.head(self.local_mlp(patches).mean(dim=spatial_dims))
    EnsembleTheoryEncoder.forward=forward


def main():
    install_dense_coarse_vjp()
    out=Path('results/dual_ising2d_g2_dense_joint_fast_2400_seed91')
    run_training(RunConfig.load('configs/dual_ising2d_g2_dense_joint_seed91.json'),out)
    from report_dual_ising_dense_joint import main as report
    report(out)


if __name__=='__main__':main()
