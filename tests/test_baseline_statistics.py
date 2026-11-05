"""CPU synthetic checks of device-local baseline statistics and patch indexing."""
import numpy as np
import pytest
import torch

from recurrent_rg.baselines.readout import fixed_coarsen, motif_probabilities
from recurrent_rg.baselines.rsmine import RSMINE, buffered_shell_offsets
from recurrent_rg.baselines.nmcrg import PeriodicConvRBM
from recurrent_rg.baselines.statistics import (
    motif_probabilities_tensor, native_pair_batch, coarsen_tensor,
)


def rng(seed=91):
    return torch.Generator().manual_seed(seed)


def spins(shape):
    return (2 * torch.randint(2, shape, generator=rng()) - 1).to(torch.int8)


@pytest.mark.parametrize("width", [2, 3])
@pytest.mark.parametrize("shape,sample_axis", [((7, 9, 9), -3), ((2, 3, 7, 9, 9), -3),
                                               ((7, 2, 3, 9, 9), 0)])
def test_gpu_style_histograms_match_numpy_readout_exactly(width, shape, sample_axis):
    x = spins(shape)
    actual = motif_probabilities_tensor(x, width, sample_axis)
    expected = motif_probabilities(x, width, sample_axis=sample_axis)
    assert actual.dtype == torch.float64 and actual.device == x.device
    np.testing.assert_array_equal(actual.numpy(), expected)
    torch.testing.assert_close(actual.sum(-1), torch.ones_like(actual[..., 0]))


@pytest.mark.parametrize("width,last_bin", [(3, 511), (2, 15)])
def test_uniform_and_periodically_repeated_motifs(width, last_bin):
    x = torch.ones(2, 5, 3, 3)
    x[0].neg_()
    p = motif_probabilities_tensor(x, width)
    assert p[0, 0] == 1 and p[1, last_bin] == 1
    tiled = x.repeat_interleave(2, -1).repeat_interleave(2, -2)
    np.testing.assert_array_equal(motif_probabilities_tensor(tiled, width).numpy(),
                                  motif_probabilities(tiled, width))


def test_hists_do_not_merge_ensemble_or_C_axes_and_support_noncontiguous_inputs():
    x = spins((2, 3, 8, 9, 12)).transpose(-1, -2)
    p = motif_probabilities_tensor(x)
    assert p.shape == (2, 3, 512)
    for ensemble in range(2):
        for draw in range(3):
            np.testing.assert_array_equal(p[ensemble, draw].numpy(),
                                          motif_probabilities(x[ensemble, draw]))


def test_pair_sampler_matches_explicit_configuration_and_periodic_indices():
    x = spins((5, 12, 9, 9))
    selected = torch.tensor([4, 1])
    before = x.clone()
    global_state = torch.random.get_rng_state().clone()
    v, e = native_pair_batch(x, selected, 8, generator=rng(5))
    reference_rng = rng(5)
    sample_ids = torch.stack([torch.randperm(12, generator=reference_rng)[:8] for _ in selected])
    origins = torch.randint(9, (2, 8, 2), generator=reference_rng)
    vo, eo = buffered_shell_offsets()
    for group in range(2):
        assert len(sample_ids[group].unique()) == 8
        for sample in range(8):
            for offsets, actual in ((vo, v), (eo, e)):
                positions = (origins[group, sample] + offsets) % 9
                expected = x[selected[group], sample_ids[group, sample], positions[:, 0], positions[:, 1]]
                torch.testing.assert_close(actual[group, sample], expected)
    assert torch.equal(x, before)
    assert torch.equal(global_state, torch.random.get_rng_state())
    assert v.shape == (2, 8, 9) and e.shape == (2, 8, 24)
    assert v.device.type == e.device.type == "cpu"


def test_pair_sampler_keeps_chains_separate_and_handles_channel_strided_view():
    channel_data = torch.ones(4, 10, 9, 9, 2, dtype=torch.int8)
    channel_data[1, ..., 0] = -1
    x = channel_data[..., 0]
    assert not x.is_contiguous()
    v, e = native_pair_batch(x, torch.tensor([1, 3]), 10, generator=rng())
    assert (v[0] == -1).all() and (e[0] == -1).all()
    assert (v[1] == 1).all() and (e[1] == 1).all()


def test_pair_sampler_never_validates_or_materializes_unselected_full_chains():
    # Unselected groups are invalid by design: scanning the whole cache would
    # reject them, whereas the small-patch-only path need not touch them.
    x = torch.zeros(4, 10, 9, 9, dtype=torch.int8)
    x[2] = -1
    v, e = native_pair_batch(x, torch.tensor([2]), 8, generator=rng())
    assert (v == -1).all() and (e == -1).all()


@pytest.mark.parametrize("method", ["majority", "decimation", "fixed_majority", "fixed_decimation"])
def test_fixed_coarsening_matches_numpy_and_preserves_prefix_dtype(method):
    x = spins((2, 3, 4, 9, 9))
    actual = coarsen_tensor(x, method)
    expected = fixed_coarsen(x, method.replace("fixed_", ""))
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (2, 3, 4, 3, 3) and actual.dtype == x.dtype


def test_rsmine_coarsening_matches_explicit_top_left_patch_application():
    x = spins((2, 4, 9, 9))
    model = RSMINE(generator=rng(), dtype=torch.float32)
    patches = torch.stack([x[..., r::3, c::3].float() for r in range(3) for c in range(3)], -1)
    expected = model.coarse_grainer.sample_spins(patches, generator=rng(13)).squeeze(-1).to(x.dtype)
    actual = coarsen_tensor(x, "rsmine", model=model, generator=rng(13))
    torch.testing.assert_close(actual, expected)
    assert not actual.requires_grad


def test_nmcrg_coarsening_calls_native_sample_and_preserves_dtype():
    x = spins((2, 4, 9, 9))
    model = PeriodicConvRBM(generator=rng(), dtype=torch.float64)
    expected = model.coarse_grain(x.double(), generator=rng(12)).to(x.dtype)
    actual = coarsen_tensor(x, "nmcrg", model=model, generator=rng(12))
    torch.testing.assert_close(actual, expected)


def test_probability_table_uses_C_top_left_row_major_lsb_first():
    x = spins((2, 4, 9, 9))
    # Select the top-left (bit0) or bottom-right (bit8) spin deterministically.
    codes = torch.arange(512)
    for bit, row, column in ((0, 0, 0), (8, 2, 2)):
        p = ((codes >> bit) & 1).double()
        expected = x[..., row::3, column::3]
        for table in (p, torch.stack((1 - p, p), -1)):
            actual = coarsen_tensor(x, "qcr_c_pca", probabilities=table, generator=rng(2))
            torch.testing.assert_close(actual, expected)


def test_probability_sampling_is_reproducible_without_global_rng_changes():
    x = spins((2, 4, 9, 9))
    state = torch.random.get_rng_state().clone()
    table = torch.full((512,), 0.5, dtype=torch.double)
    a = coarsen_tensor(x, "qcr_c_pca", probabilities=table, generator=rng(12))
    b = coarsen_tensor(x, "qcr_c_pca", probabilities=table, generator=rng(12))
    torch.testing.assert_close(a, b)
    assert torch.equal(state, torch.random.get_rng_state())


def test_validation_rejects_invalid_statistics_and_sampler_requests():
    x = spins((2, 10, 9, 9))
    for width, axis in ((1, -3), (3, -1), (3, 6)):
        with pytest.raises(ValueError):
            motif_probabilities_tensor(x, width, axis)
    with pytest.raises(ValueError):
        motif_probabilities_tensor(torch.zeros_like(x))
    with pytest.raises(ValueError):
        motif_probabilities_tensor(torch.ones_like(x, dtype=torch.uint8))
    for indices, pairs in ((torch.tensor([2]), 8), (torch.tensor([0.0]), 8),
                           (torch.tensor([0]), 11), (torch.tensor([0]), 1)):
        with pytest.raises(ValueError):
            native_pair_batch(x, indices, pairs, generator=rng())
    with pytest.raises(ValueError):
        native_pair_batch(x[..., :6, :6], torch.tensor([0]), 8, generator=rng())
    with pytest.raises(ValueError):
        native_pair_batch(x, torch.tensor([0]), 8, generator=None)


def test_coarsening_rejects_missing_rng_invalid_tables_and_invalid_shapes():
    x = spins((2, 10, 9, 9))
    for method in ("rsmine", "nmcrg", "qcr_c_pca"):
        with pytest.raises(ValueError):
            coarsen_tensor(x, method)
    for table in (torch.full((511,), 0.5), torch.full((512,), float("nan")),
                  torch.full((512,), 2.0), torch.ones(512, 2)):
        with pytest.raises(ValueError):
            coarsen_tensor(x, "qcr_c_pca", probabilities=table, generator=rng())
    with pytest.raises(ValueError):
        coarsen_tensor(x[..., :8, :8], "majority")
    with pytest.raises(ValueError):
        coarsen_tensor(x, "unknown")
