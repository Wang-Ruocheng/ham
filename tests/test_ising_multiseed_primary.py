import numpy as np
import pytest

from scripts.plot_ising_multiseed_primary import arrow_segments, panel_data


def source():
    x = np.arange(18, dtype=float).reshape(3, 2, 3)/10
    y = x+np.asarray([.03, -.07, .09])
    return dict(selected=np.array([0]), x=x, y=y,
                fractions=np.array([.995, .004, .001]), center=np.zeros(3),
                components=np.eye(3), roots=[dict(root_id='F1', g=[.2, -.3, .4],
                stability='saddle', physical_support_certified=False)],
                metrics=dict(training_seed=91, lattice_sizes=[108, 36]))


def test_flow_is_projected_saved_R_minus_input_not_next_scale():
    data = source()
    panel = panel_data(data)
    np.testing.assert_allclose(panel['delta'], .03)
    assert not np.allclose(panel['delta'][:, 0], panel['x'][:, 1]-panel['x'][:, 0])


def test_arrow_endpoint_and_length_are_exact_saved_values():
    panel = panel_data(source())
    indices, start, delta, endpoint = arrow_segments(panel, 1, count=2)
    np.testing.assert_allclose(endpoint, panel['y'][indices, 1])
    np.testing.assert_allclose(endpoint-start, delta)
    np.testing.assert_allclose(delta, .03)


def test_original_coordinate_sign_and_full_root_classification_retained():
    data = source()
    data['x'] *= -1
    data['y'] *= -1
    data['components'] *= -1
    panel = panel_data(data)
    np.testing.assert_allclose(panel['x'], data['x'][..., 0])
    assert panel['roots'][0]['position'] == -.2
    assert panel['roots'][0]['stability'] == 'saddle'


def test_multiple_major_directions_cannot_silently_be_dropped():
    data = source()
    data['selected'] = np.array([0, 1])
    with pytest.raises(ValueError, match='PC1'):
        panel_data(data)
