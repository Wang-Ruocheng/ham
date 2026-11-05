"""Offline interpolation contracts; no model, SSH, sampler or full-R root search."""
from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import plot_ising_kb_pc12_stream as stream


def _points():
    x, y = np.meshgrid(np.linspace(-1., 1., 5), np.linspace(-1., 1., 5))
    return np.stack((x.ravel(), y.ravel()), axis=-1)


def _linear(points, matrix, root, *, normal=0.):
    vectors = (points - root) @ np.asarray(matrix).T
    return np.column_stack((vectors, np.full(len(points), normal)))


@pytest.mark.parametrize("matrix", [np.diag([.7, -.4]), np.diag([-.5, -.8])])
def test_affine_saddle_and_sink_zeros_are_exact_and_deduplicated(matrix):
    points = _points()
    expected_root = np.array([.17, -.23])
    full = _linear(points, matrix, expected_root)
    field = stream.InterpolatedFlow(points, full, np.eye(3)[:2])
    zeros = field.zeros()
    assert len(zeros) == 1
    np.testing.assert_allclose(zeros[0]["xy"], expected_root, atol=2e-12)
    assert zeros[0]["projected_residual"] < 1e-12
    assert zeros[0]["interpolated_full_residual"] < 1e-12
    # A zero of this two-dimensional interpolant is never a certified full-R root.
    assert zeros[0]["full_root_certified"] is False
    assert field.valid_triangles[zeros[0]["simplex_index"]]
    np.testing.assert_allclose(zeros[0]["interpolated_vector_jacobian"], matrix, atol=2e-12)
    query = np.array([[.11, -.19], [-.64, .78]])
    np.testing.assert_allclose(field.evaluate(query), _linear(query, matrix, expected_root), atol=2e-12)


def test_zero_on_shared_triangle_vertex_is_not_reported_multiple_times():
    points = _points()
    field = stream.InterpolatedFlow(points, _linear(points, np.diag([1., -1.]), np.zeros(2)), np.eye(3)[:2])
    zeros = field.zeros()
    assert len(zeros) == 1
    np.testing.assert_allclose(zeros[0]["xy"], [0., 0.], atol=2e-12)


def test_projection_zero_with_nonzero_normal_drift_cannot_be_full_root():
    points = _points()
    root = np.array([.17, -.23])
    field = stream.InterpolatedFlow(points, _linear(points, np.diag([-.5, -.8]), root, normal=1.25), np.eye(3)[:2])
    zero, = field.zeros()
    np.testing.assert_allclose(zero["xy"], root, atol=2e-12)
    assert zero["projected_residual"] < 1e-12
    assert zero["interpolated_full_residual"] == pytest.approx(1.25)
    assert zero["full_root_certified"] is False
    np.testing.assert_allclose(field.evaluate(root), [0., 0., 1.25], atol=2e-12)


def test_vectors_use_saved_basis_without_centering_or_reorienting():
    points = _points()
    full = np.column_stack((points + [.2, -.3], np.ones(len(points))))
    basis = np.array([[0., -1., 0.], [1., 0., 0.]])
    field = stream.InterpolatedFlow(points, full, basis)
    np.testing.assert_array_equal(field.vectors, full @ basis.T)
    query = np.array([[.11, -.19], [-.64, .78]])
    expected_full = np.column_stack((query + [.2, -.3], np.ones(len(query))))
    np.testing.assert_allclose(field.evaluate(query), expected_full, atol=2e-12)


def test_no_extrapolation_beyond_convex_hull_and_no_hidden_fill():
    points = _points()
    field = stream.InterpolatedFlow(points, _linear(points, np.eye(2), np.array([.1, -.2])), np.eye(3)[:2])
    query = np.array([[0., 0.], [1.01, 0.], [0., -1.01], [2., 2.]])
    result = field.evaluate(query)
    assert np.isfinite(result[0]).all()
    assert np.isnan(result[1:]).all()
    x, y, vectors = stream.make_grid(field, ([-1.2, -1.2], [1.2, 1.2]), resolution=31)
    assert x.ndim == y.ndim == 1
    assert vectors.shape == (len(y), len(x), 2)
    xx, yy = np.meshgrid(x, y)
    exterior = (abs(xx) > 1. + 1e-12) | (abs(yy) > 1. + 1e-12)
    mask = np.ma.getmaskarray(vectors)
    assert mask.shape == vectors.shape
    assert mask[exterior].all()
    assert not mask[len(y) // 2, len(x) // 2].any()


def test_long_bridge_triangles_are_masked_not_interpolated_across_gap():
    left = _points() * .025 + [-1., 0.]
    right = _points() * .025 + [1., 0.]
    points = np.concatenate((left, right))
    field = stream.InterpolatedFlow(points, _linear(points, np.eye(2), np.zeros(2)), np.eye(3)[:2], max_edge_ratio=4.)
    assert field.valid_triangles.any()
    assert (~field.valid_triangles).any()
    # The centre is in the total convex hull but only joined by unsupported long edges.
    assert np.isnan(field.evaluate(np.array([0., 0.]))).all()
    assert not field.zeros()
    assert np.isfinite(field.evaluate(np.array([-1., 0.]))).all()


def test_each_scale_has_its_own_interpolant_not_a_pooled_cancelling_field():
    points = _points()
    roots = [np.array([-.27, .18]), np.array([.37, -.33])]
    fields = [stream.InterpolatedFlow(points, _linear(points, sign * np.eye(2), root), np.eye(3)[:2])
              for sign, root in zip((1., -1.), roots)]
    for field, expected_root in zip(fields, roots):
        zero, = field.zeros()
        np.testing.assert_allclose(zero["xy"], expected_root, atol=2e-12)
    assert not np.allclose(fields[0].evaluate([0., 0.]), fields[1].evaluate([0., 0.]))


def test_hidden_state_slope_masks_folded_or_unresolved_projected_neighborhood():
    points = _points()
    full = _linear(points, np.eye(2), [.1, -.2])
    # A steep hidden-state graph is deliberately rejected even though its
    # projected data are an ordinary dense square. This is a plotting mask,
    # never a certification of physical support.
    states = np.column_stack((points, 12. * points[:, 0]))
    strict = stream.InterpolatedFlow(points, full, np.eye(3)[:2], states=states, max_hidden_slope=5.)
    relaxed = stream.InterpolatedFlow(points, full, np.eye(3)[:2], states=states, max_hidden_slope=20.)
    assert not strict.valid_triangles.any()
    assert np.isnan(strict.evaluate([0., 0.])).all()
    assert not strict.zeros()
    assert relaxed.valid_triangles.all()
    assert np.isfinite(relaxed.evaluate([0., 0.])).all()


def test_saved_full_coordinate_root_markers_are_not_snapped_to_interpolation_domain():
    center = np.array([10., -4., 3.])
    components = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    roots = {"roots": [dict(root_id=7, g=[103., -29., 51.], residual_norm=3e-12,
                            geometric_support_candidate=False, physical_support_certified=False)]}
    report = SimpleNamespace(grid={"pca_center": center, "pca_components": components}, roots=roots)
    before = copy.deepcopy(roots)
    marker, = stream.saved_root_markers(report)
    assert marker["kind"] == "full_coordinate_root_projection"
    np.testing.assert_array_equal(marker["xy"], (np.array(roots["roots"][0]["g"]) - center) @ components[:2].T)
    assert roots == before
    assert marker["physical_support_certified"] is False


def test_construction_and_zero_analysis_leave_all_public_inputs_unchanged(tmp_path):
    points = _points()
    full = _linear(points, np.diag([1., -.8]), [.21, -.13], normal=.4)
    basis = np.eye(3)[:2]
    path = tmp_path / "public_arrays.npz"
    np.savez_compressed(path, points=points, full_displacement=full, basis=basis)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"kind": "synthetic public NPZ"}), encoding="utf-8")
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (path, receipt)}
    arrays_before = [value.copy() for value in (points, full, basis)]
    field = stream.InterpolatedFlow(points, full, basis)
    field.zeros()
    field.evaluate(np.array([[0., 0.], [2., 2.]]))
    stream.make_grid(field, ([-1.1, -1.1], [1.1, 1.1]), resolution=21)
    for value, original in zip((points, full, basis), arrays_before):
        np.testing.assert_array_equal(value, original)
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (path, receipt)} == before


def _tiny_frozen_values(dimension=5):
    rng = np.random.default_rng(9821)
    weights = [rng.normal(scale=.02, size=shape)
               for shape in ((96, dimension), (96, 96), (dimension, 96))]
    biases = [rng.normal(scale=.01, size=shape)
              for shape in ((96,), (96,), (dimension,))]
    inputs = rng.normal(size=(7, dimension))
    value = np.tanh(inputs @ weights[0].T + biases[0])
    value = np.tanh(value @ weights[1].T + biases[1])
    outputs = inputs + value @ weights[2].T + biases[2]
    prefix = f"dim{dimension}_"
    values = {prefix + f"W{i}": weights[i] for i in range(3)}
    values.update({prefix + f"b{i}": biases[i] for i in range(3)})
    values[prefix + "validation_input"] = inputs
    values[prefix + "validation_output"] = outputs
    return values


def test_tiny_frozen_r_preserves_residual_connection_and_tanh_architecture():
    values = _tiny_frozen_values()
    model = stream.FrozenR(values, 5)
    inputs, outputs = values["dim5_validation_input"], values["dim5_validation_output"]
    np.testing.assert_allclose(inputs + model.displacement(inputs), outputs, atol=1e-14, rtol=0)
    assert model.validation_error < 1e-14
    # Not an absolute map incorrectly substituted for the displacement.
    assert not np.allclose(model.displacement(inputs), outputs)


def test_frozen_r_rejects_validation_mismatch_and_wrong_dtype():
    values = _tiny_frozen_values()
    values["dim5_validation_output"] = values["dim5_validation_output"] + .1
    with pytest.raises(ValueError, match="disagrees"):
        stream.FrozenR(values, 5)
    values = _tiny_frozen_values()
    values["dim5_W0"] = values["dim5_W0"].astype(np.float32)
    with pytest.raises(ValueError, match="float64"):
        stream.FrozenR(values, 5)
