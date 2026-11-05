import numpy as np
import pytest

from scripts.preview_phi4_raw_endpoint_configurations import moments, select_fields


def test_selection_preserves_chain_saved_and_spatial_axes():
    labels = (1000 * np.arange(4)[:, None, None, None, None]
              + np.arange(128)[None, :, None, None, None]).astype(np.float64)
    values = np.broadcast_to(labels, (4, 128, 128, 128, 1))
    selected = select_fields(values)
    assert selected.shape == (4, 2, 128, 128)
    np.testing.assert_array_equal(selected[:, :, 0, 0],
                                  [[32, 96], [1032, 1096], [2032, 2096], [3032, 3096]])
    assert selected.flags.owndata


def test_selection_rejects_wrong_dtype_and_shape():
    with pytest.raises(ValueError):
        select_fields(np.broadcast_to(np.float32(0), (4, 128, 128, 128, 1)))
    with pytest.raises(ValueError):
        select_fields(np.zeros((2, 128, 128, 128, 1)))


def test_sign_invariant_summary_does_not_flip_or_pool_fields():
    x = np.array([[1., 2.], [-1., 2.]])
    before = x.copy()
    positive, negative = moments(x), moments(-x)
    assert positive["mean"] == -negative["mean"]
    assert positive["abs_mean_over_rms"] == negative["abs_mean_over_rms"]
    assert positive["majority_sign_fraction"] == negative["majority_sign_fraction"] == .75
    np.testing.assert_array_equal(x, before)
