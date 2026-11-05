import numpy as np
import pytest

from scripts.preview_phi4_b2_phase_configurations import (
    CELL_KAPPAS, METHODS, aggregate, block_mean, moments, oddness_ratio, select_fields,
)


def test_block_average_geometry_and_mean():
    x = np.arange(64, dtype=float).reshape(8, 8)
    expected = np.array([[4.5, 6.5, 8.5, 10.5], [20.5, 22.5, 24.5, 26.5],
                         [36.5, 38.5, 40.5, 42.5], [52.5, 54.5, 56.5, 58.5]])
    np.testing.assert_allclose(block_mean(x), expected)
    assert block_mean(x).mean() == x.mean()
    assert block_mean(block_mean(block_mean(x))).shape == (1, 1)


@pytest.mark.parametrize("shape", [(3, 3), (4, 2), (1, 4, 4)])
def test_block_average_invalid(shape):
    with pytest.raises(ValueError):
        block_mean(np.ones(shape))


def test_sign_independent_metrics_and_sector_preservation():
    x = np.array([[2., -1.], [3., 4.]])
    positive = moments(x, 2., x.mean())
    negative = moments(-x, 2., (-x).mean())
    assert positive["q_abs_mean_over_rms"] == negative["q_abs_mean_over_rms"]
    assert positive["majority_fraction"] == negative["majority_fraction"] == .75
    assert positive["mean"] == -negative["mean"]
    assert negative["sign_retention"] == 1.
    assert moments(-x, 2., x.mean())["sign_retention"] == 0.
    assert moments(np.zeros((2, 2)), 1.)["majority_fraction"] == .5
    assert moments(np.zeros((2, 2)), 1.)["q_abs_mean_over_rms"] == 0.


def test_block_average_does_not_claim_order_from_pooled_mean():
    x = np.array([[1., -1., 1., -1.], [-1., 1., -1., 1.],
                  [1., -1., 1., -1.], [-1., 1., -1., 1.]])
    assert moments(x, 1.)["q_abs_mean_over_rms"] == 0.
    assert moments(block_mean(x), 1.)["q_abs_mean_over_rms"] == 0.


def test_oddness_zero_for_odd_map_and_detects_bias():
    value = np.arange(4.).reshape(2, 2)
    assert oddness_ratio(value, -value) == 0.
    assert oddness_ratio(np.ones((2, 2)), np.ones((2, 2))) == 2.
    assert oddness_ratio(value + 1, -value + 1) > 0.


def test_selected_axes_are_chain_sample_spatial_and_copy():
    # Broadcast-backed toy source avoids materializing a large physical array.
    labels = (np.arange(4)[:, None, None, None, None] * 1000
              + np.arange(128)[None, :, None, None, None]).astype(np.float64)
    x = np.broadcast_to(labels, (4, 128, 128, 128, 1))
    selected = select_fields(x)
    assert selected.shape == (4, 3, 128, 128)
    np.testing.assert_allclose(selected[:, :, 0, 0],
                               [[32, 64, 96], [1032, 1064, 1096],
                                [2032, 2064, 2096], [3032, 3064, 3096]])
    assert selected.flags.owndata


def test_selection_rejects_wrong_axis_order_or_dtype():
    x = np.broadcast_to(np.float32(0), (4, 128, 128, 128, 1))
    with pytest.raises(ValueError):
        select_fields(x)
    with pytest.raises(ValueError):
        select_fields(np.zeros((1, 4, 128, 128, 128)))


def test_equal_physical_chain_aggregation():
    rows = []
    for cell, _ in CELL_KAPPAS:
        for method in METHODS:
            for chain in range(4):
                for sample in range(3):
                    rows.append(dict(cell_index=cell, method=method, chain=chain,
                                     moments=[dict(size=128 // 2**k, metric=chain + sample)
                                              for k in range(4)]))
    result = aggregate(rows)
    assert len(result) == 5 * 3 * 4
    assert result[0]["four_chain_mean"]["metric"] == 2.5
    assert [x["metric"] for x in result[0]["chains"]] == [1., 2., 3., 4.]
    with pytest.raises(ValueError):
        aggregate(rows[:-1])
