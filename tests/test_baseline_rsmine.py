"""CPU synthetic mathematical tests, not a physical RSMI-NE experiment."""

import math

import pytest
import torch

from recurrent_rg.baselines.rsmine import (
    LinearBinaryCoarseGrainer, RSMINE, SeparableCritic, buffered_shell_offsets,
    infonce_from_scores, mi_loss,
)


def seeded(seed):
    return torch.Generator().manual_seed(seed)


def data(groups=3, replicas=8):
    rng = seeded(123)
    v = (2 * torch.randint(2, (groups, replicas, 9), generator=rng) - 1).double()
    e = (2 * torch.randint(2, (groups, replicas, 24), generator=rng) - 1).double()
    return v, e


def test_matched_geometry_has_all_24_buffered_environment_sites_without_aliases():
    v, e = buffered_shell_offsets()
    assert v.shape == (9, 2) and e.shape == (24, 2)
    assert len(torch.unique(e, dim=0)) == 24
    assert torch.cdist(v.double(), e.double(), p=float("inf")).min() == 2
    assert ((e == -2) | (e == 4)).any(-1).all()
    for block, buffer in ((0, 1), (3, -1), (3.0, 1)):
        with pytest.raises(ValueError):
            buffered_shell_offsets(block, buffer)


def test_initialization_and_sampling_are_seeded_without_touching_global_rng():
    state = torch.random.get_rng_state().clone()
    a = RSMINE(generator=seeded(91), dtype=torch.double)
    b = RSMINE(generator=seeded(91), dtype=torch.double)
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pa, pb)
    v, e = data()
    result_a = a(v, e, step=6, generator=seeded(92))
    result_b = b(v, e, step=6, generator=seeded(92))
    torch.testing.assert_close(result_a.loss, result_b.loss)
    torch.testing.assert_close(result_a.relaxed_hidden, result_b.relaxed_hidden)
    assert torch.equal(state, torch.random.get_rng_state())


def test_binary_logit_has_no_factor_two_no_bias_and_flip_probability_symmetry():
    cg = LinearBinaryCoarseGrainer(generator=seeded(1), dtype=torch.double)
    with torch.no_grad():
        cg.weight.copy_(torch.arange(9, dtype=torch.double) / 100)
    v = torch.ones(2, 9, dtype=torch.double)
    assert len(list(cg.parameters())) == 1
    torch.testing.assert_close(cg.logits(v), torch.full((2, 1), 0.36, dtype=torch.double))
    torch.testing.assert_close(cg.probabilities(v), torch.sigmoid(cg.logits(v)))
    torch.testing.assert_close(cg.probabilities(-v), 1 - cg.probabilities(v))


def test_relaxation_matches_explicit_logistic_reparameterization_and_anneals():
    cg = LinearBinaryCoarseGrainer(generator=seeded(1), dtype=torch.double)
    v, _ = data()
    uniform = torch.rand((3, 8, 1), generator=seeded(18), dtype=torch.double)
    expected = torch.sigmoid(
        (cg.logits(v) + uniform.log() - torch.log1p(-uniform)) / cg.temperature(33)
    )
    torch.testing.assert_close(cg(v, step=33, generator=seeded(18)), expected)
    assert cg.temperature(0) == 0.5
    assert cg.temperature(100) == pytest.approx(0.5 * math.exp(-0.5))
    assert cg.temperature(100000) == 0.1
    early = cg(v, step=0, generator=seeded(7))
    late = cg(v, step=100000, generator=seeded(7))
    assert not torch.equal(early, late)
    assert ((late - 0.5).abs() >= (early - 0.5).abs()).all()


def test_hard_coarse_graining_is_reproducible_exact_binary_and_detached():
    cg = LinearBinaryCoarseGrainer(generator=seeded(1), dtype=torch.double)
    v, _ = data()
    h = cg.sample_spins(v, generator=seeded(5))
    torch.testing.assert_close(h, cg.sample_spins(v, generator=seeded(5)))
    assert ((h == -1) | (h == 1)).all()
    assert not h.requires_grad
    with torch.no_grad():
        cg.weight.zero_()
    many = torch.ones(10000, 9, dtype=torch.double)
    assert abs(float(cg.sample_spins(many, generator=seeded(91)).mean())) < 0.04


def test_critic_is_two_independent_relu_mlps_and_dot_product():
    critic = SeparableCritic(generator=seeded(4), dtype=torch.double)
    assert sum(isinstance(m, torch.nn.ReLU) for m in critic.modules()) == 4
    assert not ({id(p) for p in critic.hidden_net.parameters()} &
                {id(p) for p in critic.environment_net.parameters()})
    _, e = data()
    h = torch.ones(3, 8, 1, dtype=torch.double)
    expected = torch.einsum("gic,gjc->gij", critic.environment_net(e), critic.hidden_net(h))
    torch.testing.assert_close(critic(e, h), expected)


def test_infonce_column_normalization_matches_explicit_formula_and_gradients():
    scores = torch.randn(3, 5, 5, generator=seeded(4), dtype=torch.double, requires_grad=True)
    explicit = torch.stack([
        sum(scores[g, j, j] - torch.logsumexp(scores[g, :, j], 0) + math.log(5)
            for j in range(5)) / 5 for g in range(3)
    ])
    actual = infonce_from_scores(scores)
    torch.testing.assert_close(actual, explicit)
    ga, = torch.autograd.grad(actual.sum(), scores, retain_graph=True)
    ge, = torch.autograd.grad(explicit.sum(), scores)
    torch.testing.assert_close(ga, ge)
    assert (actual <= math.log(5) + 1e-12).all()
    # Column-wise addition must cancel; row-wise addition generally must not.
    offsets = torch.arange(5, dtype=torch.double)[None, None, :]
    torch.testing.assert_close(infonce_from_scores(scores + offsets), actual)
    # Author's critic returns H-by-E and TFP reduces over its last (E) axis.
    author_scores = scores.transpose(-1, -2)
    tfp_equivalent = (
        author_scores.diagonal(dim1=-2, dim2=-1).mean(-1)
        - torch.logsumexp(author_scores, -1).mean(-1) + math.log(5)
    )
    torch.testing.assert_close(actual, tfp_equivalent)


def test_independent_and_perfect_binary_score_limits():
    torch.testing.assert_close(infonce_from_scores(torch.zeros(4, 4)), torch.tensor(0.0))
    binary = torch.tensor([-1.0, 1.0, -1.0, 1.0], dtype=torch.double)
    assert float(infonce_from_scores(40 * binary[:, None] * binary[None, :])) == pytest.approx(
        math.log(2), abs=1e-12
    )


def test_native_loss_updates_only_c_and_critic_not_unrelated_heads():
    model = RSMINE(generator=seeded(3), dtype=torch.double)
    unrelated = torch.nn.Parameter(torch.ones(3, dtype=torch.double))
    v, e = data()
    result = mi_loss(model, v, e, step=9, generator=seeded(8))
    assert result.group_estimates.shape == (3,)
    assert result.group_sizes == (8, 8, 8)
    torch.testing.assert_close(result.loss, -result.group_estimates.mean())
    result.loss.backward()
    assert model.coarse_grainer.weight.grad.abs().sum() > 0
    assert sum(float(p.grad.abs().sum()) for p in model.critic.parameters()) > 0
    assert unrelated.grad is None
    assert not any("variance" in k or "color" in k or "closure" in k for k in model.state_dict())


def test_explicit_groups_and_flat_composite_ids_have_identical_loss_and_gradients():
    model = RSMINE(generator=seeded(4), dtype=torch.double)
    v, e = data(groups=4)
    conditions = torch.tensor([0, 0, 1, 1]).repeat_interleave(8)
    chains = torch.tensor([0, 1, 0, 1]).repeat_interleave(8)
    grouped = model(v, e, step=30, generator=seeded(8))
    flat = model(v.flatten(0, 1), e.flatten(0, 1), step=30, generator=seeded(8),
                 condition_ids=conditions, chain_ids=chains)
    torch.testing.assert_close(grouped.loss, flat.loss)
    gg = torch.autograd.grad(grouped.loss, tuple(model.parameters()), retain_graph=True)
    gf = torch.autograd.grad(flat.loss, tuple(model.parameters()))
    for a, b in zip(gg, gf):
        torch.testing.assert_close(a, b)


def test_negatives_never_cross_condition_or_chain_and_ragged_groups_equal_weight():
    model = RSMINE(generator=seeded(4), dtype=torch.double)
    v, e = data()
    v = v.flatten(0, 1)[:9]
    e = e.flatten(0, 1)[:9]
    conditions = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1])
    chains = torch.tensor([0, 0, 1, 1, 1, 0, 0, 0, 0])
    result = model(v, e, step=0, generator=seeded(8),
                   condition_ids=conditions, chain_ids=chains)
    assert result.group_sizes == (2, 3, 4)
    expected = torch.stack([
        infonce_from_scores(model.critic(e[index], result.relaxed_hidden[index]))
        for index in (slice(0, 2), slice(2, 5), slice(5, 9))
    ])
    torch.testing.assert_close(result.group_estimates, expected)
    torch.testing.assert_close(result.loss, -expected.mean())


@pytest.mark.parametrize("bad_ids", [None, torch.zeros(8), torch.zeros(7, dtype=torch.long)])
def test_flat_inputs_reject_missing_or_coupling_like_ids(bad_ids):
    model = RSMINE(generator=seeded(4), dtype=torch.double)
    v, e = data(groups=1)
    with pytest.raises(ValueError):
        model(v[0], e[0], step=0, generator=seeded(8),
              condition_ids=bad_ids, chain_ids=torch.zeros(8, dtype=torch.long))


def test_bad_shapes_singletons_dtypes_and_steps_are_rejected():
    model = RSMINE(generator=seeded(4), dtype=torch.double)
    v, e = data(groups=1)
    for vv, ee in ((v[:, :1], e[:, :1]), (v[..., :8], e), (v, e[..., :23]), (v, e.float())):
        with pytest.raises(ValueError):
            model(vv, ee, step=0, generator=seeded(1))
    with pytest.raises(ValueError):
        model(v[0], e[0], step=0, condition_ids=torch.arange(8), chain_ids=torch.arange(8))
    for step in (-1, 0.5, True):
        with pytest.raises(ValueError):
            model(v, e, step=step, generator=seeded(1))
    with pytest.raises(ValueError):
        model(v[0, :0], e[0, :0], step=0, condition_ids=torch.empty(0, dtype=torch.long),
              chain_ids=torch.empty(0, dtype=torch.long))
