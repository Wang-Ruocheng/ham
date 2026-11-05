import numpy as np
import pytest

from scripts.plot_saved_kb_parameter_flow import cell_means


def sample():
    return dict(direct_g=np.arange(24, dtype=float).reshape(4, 3, 2),
                labels=np.array([[.2, -.01], [.6, .01], [.2, -.01], [.6, .01]]),
                cell_ids=np.array([0, 1, 0, 1]), splits=np.array(['train', 'test', 'train', 'test']))


def test_same_parameter_means_keep_labels_and_scales():
    a = sample()
    means, labels, counts = cell_means(a)
    np.testing.assert_array_equal(means[0], a['direct_g'][[0, 2]].mean(0))
    np.testing.assert_array_equal(means[1], a['direct_g'][[1, 3]].mean(0))
    np.testing.assert_array_equal(labels, a['labels'][:2])
    assert counts == [2, 2]


@pytest.mark.parametrize('key', ['labels', 'splits'])
def test_reject_mixing_physical_cells_or_splits(key):
    a = sample()
    a[key][2] = a[key][1]
    with pytest.raises(ValueError, match='aggregate'):
        cell_means(a)


def test_nonfinite_data_rejected():
    a = sample()
    a['direct_g'][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        cell_means(a)
