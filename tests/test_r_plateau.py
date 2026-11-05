import pytest
import torch

from experiments.frozen_qc_r_continue import r_objective
from experiments.frozen_qc_r_plateau import frozen_subset_direct, plateau_reached
from recurrent_rg.checkpoints import build_model, qcr_state_dict
from recurrent_rg.config import DataConfig, ModelConfig, RunConfig
from recurrent_rg.data.statistics import plaquette_distribution
from recurrent_rg.losses import latent_squared_error, multistep_closure_loss, qcr_objective


def small_vector_model_and_batch():
    torch.set_num_threads(1)
    torch.manual_seed(19)
    cfg = RunConfig(
        data=DataConfig(cache='unused.pt', physical_model='test'),
        model=ModelConfig(dimension=2, theory_dim=2, hidden=5, block_factor=3),
    )
    model = build_model(cfg.model, 2)
    batch = (2 * torch.randint(2, (3, 8, 36, 36, 1)) - 1).double()
    return cfg, model, batch


def test_vector_half_subset_closure_matches_joint_qcr_with_same_c_draws():
    cfg, model, batch = small_vector_model_and_batch()
    assert model.rg_kernel.field_head.sampling_mode == 'bernoulli'
    assert all(p.requires_grad for p in model.rg_kernel.parameters())
    assert cfg.loss.weights.commutator == cfg.loss.rollout_weight == 5.

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31)
        direct = frozen_subset_direct(model, batch)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31)
        total, terms = qcr_objective(
            model, batch, pattern_fn=plaquette_distribution,
            **cfg.loss.objective_kwargs(),
        )

    assert direct.shape == (3, 3, 2)
    one_step = latent_squared_error(
        model.advance_theory(direct[:, :-1]), direct[:, 1:],
        cfg.loss.coordinate_reduction,
    )
    rollout, _ = multistep_closure_loss(model, direct, cfg.loss.coordinate_reduction)
    torch.testing.assert_close(one_step, terms['commutator'], atol=0, rtol=0)
    torch.testing.assert_close(rollout, terms['rollout'], atol=0, rtol=0)
    closure = cfg.loss.weights.commutator * terms['commutator'] + cfg.loss.rollout_weight * terms['rollout']
    torch.testing.assert_close(r_objective(model, direct, cfg), closure, atol=0, rtol=0)
    assert 'reconstruction_mse_level_0' in terms
    assert total.detach() > closure.detach()


def test_frozen_targets_allow_r_updates_without_q_c_or_input_gradients():
    cfg, model, batch = small_vector_model_and_batch()
    batch.requires_grad_(True)
    before = {name: value.clone() for name, value in qcr_state_dict(model).items()}
    direct = frozen_subset_direct(model, batch)
    assert not direct.requires_grad and direct.grad_fn is None

    optimizer = torch.optim.Adam(model.theory_rg.parameters(), lr=.001)
    optimizer.zero_grad(set_to_none=True)
    r_objective(model, direct, cfg).backward()
    assert batch.grad is None
    assert all(p.grad is None for p in model.theory_encoder.parameters())
    assert all(p.grad is None for p in model.rg_kernel.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.theory_rg.parameters())
    assert any(p.grad.abs().sum() > 0 for p in model.theory_rg.parameters())
    optimizer.step()

    after = qcr_state_dict(model)
    for name, value in before.items():
        if name.startswith(('theory_encoder.', 'rg_kernel.')):
            torch.testing.assert_close(value, after[name], atol=0, rtol=0)
    assert any(not torch.equal(value, after[name]) for name, value in before.items()
               if name.startswith('theory_rg.'))


def plateau_rows(last_step, improvement_per_log=.0001):
    return [dict(extra_R_updates=step,
                 best_train_closure=1. - improvement_per_log * (step // 250),
                 train=dict(total=1. - improvement_per_log * (step // 250)))
            for step in range(0, last_step + 1, 250)]


def test_plateau_requires_a_full_window_for_three_consecutive_logs():
    rows = plateau_rows(2500)
    assert not plateau_reached(rows[:-3])
    assert not plateau_reached(rows[:-2])
    assert not plateau_reached(rows[:-1])
    assert plateau_reached(rows)
    assert not plateau_reached(rows[-3:])
    assert not plateau_reached(rows, minimum=3000)


def test_plateau_rejects_continuing_or_recent_significant_improvement():
    assert not plateau_reached(plateau_rows(3000, improvement_per_log=.01))
    rows = plateau_rows(2500)
    assert plateau_reached(rows)
    rows[-1]['best_train_closure'] = .9
    rows[-1]['train']['total'] = .9
    assert not plateau_reached(rows)


def test_plateau_requires_improvement_strictly_below_one_percent():
    rows = plateau_rows(2500, improvement_per_log=0.)
    rows[-1]['best_train_closure'] = .99
    rows[-1]['train']['total'] = .99
    assert not plateau_reached(rows)
    rows[-1]['best_train_closure'] = .991
    rows[-1]['train']['total'] = .991
    assert plateau_reached(rows)


@pytest.mark.parametrize('confirmation_index', [-3, -2, -1])
def test_plateau_requires_current_loss_within_five_percent_of_best_at_each_confirmation(confirmation_index):
    rows = plateau_rows(2500)
    assert plateau_reached(rows)
    row = rows[confirmation_index]
    row['train']['total'] = row['best_train_closure'] * 1.051
    assert not plateau_reached(rows)
    row['train']['total'] = row['best_train_closure'] * 1.05
    assert plateau_reached(rows)
