"""The B=0 d_g=5/b=3 campaign retains the independent default objectives."""
from dataclasses import replace

import pytest
import torch

from experiments import rsmi_separate_qcr as driver
from recurrent_rg.binary_pooling import binary_pattern_table
from recurrent_rg.checkpoints import build_model
from recurrent_rg.config import DataConfig, LossConfig, ModelConfig, RunConfig, TrainingConfig
from recurrent_rg.data.statistics import plaquette_distribution
from recurrent_rg.empirical_mi import binary_lattice_mi_statistics
from recurrent_rg.losses import qcr_objective
from recurrent_rg.representation import normalize_affine_decoder_
from recurrent_rg.rsmi import kernel_probabilities


@pytest.mark.parametrize('seed', [91, 92])
def test_baseline_uniform_losses_and_gradient_routing_remain_default(seed):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        cfg = RunConfig(data=DataConfig(cache='not-opened.pt', physical_model='ising2d',
            coupling_names=('K',)), model=ModelConfig(dimension=2, theory_dim=5, hidden=96),
            loss=LossConfig(closure_objective='one_step_and_rollout'),
            training=TrainingConfig(max_rg_steps=2, seed=seed, gradient_mode='qc_r_separate'))
        args = driver.build_parser().parse_args(['--config', 'not-opened.json',
            '--output', 'not-created', '--ising-b0-ablation', 'none'])
        driver.validate_recipe(args, cfg)
        contract = driver.ising_b0_ablation_contract(args, cfg)
        assert not args.allow_nondefault_recipe and args.independent_c
        assert cfg.schema_version == 5 and cfg.model.block_factor == 3
        assert cfg.model.local_features == 12 and cfg.loss.closure_scale_weights is None
        assert contract['baseline'] and not contract['removed_terms']
        assert not contract['labels_used_in_training']
        assert contract['effective_weights'] == dict(C_MI=-2., C_variance=2., C_color=1.,
            Q_reconstruction=10., Q_microscopic_subset=20., Q_coarse_subset=5.,
            R_one_step=5., R_multistep=5.)

        torch.manual_seed(seed)
        model = build_model(cfg.model, 2)
        model.detach_kernel_representation = True
        model.share_binary_q_tables = True
        microscopic = (2 * torch.randint(2, (2, 4, 36, 36, 1)) - 1).double()
        effective = replace(cfg.loss, kernel_noncollapse_mode='variance_floor_ensemble',
            kernel_variance_ratio=1., kernel_sign_weight=1.)
        base, terms = qcr_objective(model, microscopic,
            pattern_fn=plaquette_distribution, **effective.objective_kwargs())
        for name, prefix in [('commutator', 'one_step'), ('rollout', 'rollout')]:
            expected = torch.stack([terms[f'{prefix}_level_{k}'] for k in (1, 2)]).mean()
            torch.testing.assert_close(terms[name], expected, rtol=1e-14, atol=1e-14)

        # This is the same full-shell, buffer-one empirical histogram term as
        # the driver, built only from configurations (no K or other labels).
        histogram = binary_lattice_mi_statistics(microscopic, block=3, buffer=1)
        _, patterns = binary_pattern_table(9, microscopic)
        table = kernel_probabilities(model.rg_kernel, patterns[None])[0]
        information = histogram.information(table).mean()
        c_objective = -2 * information + 2 * terms['kernel_noncollapse'] + terms['kernel_sign']
        q_objective = 2 * terms['anchor'] + 5 * terms['coarse_consistency'] + 10 * terms['separation']
        closure = 5 * (terms['commutator'] + terms['rollout'])
        total = base - 2 * information
        q, c, r = (list(module.parameters()) for module in
                   (model.theory_encoder, model.rg_kernel, model.theory_rg))

        # Q's representation graph has no path back into the sampled C.
        assert all(value is None for value in torch.autograd.grad(
            q_objective, c, retain_graph=True, allow_unused=True))
        expected_q = torch.autograd.grad(q_objective, q, retain_graph=True)
        expected_c = torch.autograd.grad(c_objective, c, retain_graph=True)
        expected_r = torch.autograd.grad(closure, r, retain_graph=True)
        driver.independent_component_gradients(total, closure, c_objective, q, c, r)
        # These equalities verify assigned R gradients never feed Q/C, even
        # though the shared closure graph contains Q values before selection.
        for parameters, expected in ((q, expected_q), (c, expected_c), (r, expected_r)):
            for parameter, gradient in zip(parameters, expected):
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                torch.testing.assert_close(parameter.grad, gradient, rtol=1e-10, atol=1e-12)
        decoder = model.theory_encoder.distribution_decoder
        assert isinstance(decoder, torch.nn.Linear) and decoder.weight.shape == (16, 5)
        torch.testing.assert_close(decoder.weight.norm(dim=0), torch.ones(5, dtype=torch.double))
        torch.optim.Adam(q + c + r, lr=.001).step()
        normalize_affine_decoder_(model.theory_encoder)
        torch.testing.assert_close(decoder.weight.norm(dim=0), torch.ones(5, dtype=torch.double))
    finally:
        torch.set_num_threads(previous_threads)
