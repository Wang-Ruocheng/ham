"""Synthetic, CPU-only checks of the fixed-point overlay search."""
import json

import numpy as np
import pytest
import torch

from recurrent_rg.phi4_ru_flow import full_map_and_jacobian
from scripts import plot_phi4_ru_fixed_point_overlay as overlay


class AffineMap(torch.nn.Module):
    def __init__(self, matrix, root):
        super().__init__()
        self.register_buffer("matrix", torch.as_tensor(matrix, dtype=torch.float64))
        self.register_buffer("root", torch.as_tensor(root, dtype=torch.float64))

    def advance(self, g):
        return (g - self.root) @ self.matrix.T + self.root


class TranslationMap(torch.nn.Module):
    def advance(self, g):
        return g + g.new_tensor([.1, 0.])


class NonlinearResidualMap(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.coefficient = torch.nn.Parameter(
            torch.tensor(.3, dtype=torch.float64), requires_grad=False)

    def advance(self, g):
        residual = torch.stack((self.coefficient * torch.sin(g[:, 0]) + .2 * g[:, 1],
                                .1 * g[:, 0] ** 2 - .4 * torch.tanh(g[:, 1])), dim=1)
        return g + residual


@pytest.mark.parametrize("matrix,classification", [
    ([[1.3, 0.], [0., .8]], "saddle"),
    ([[-1.2, 0.], [0., .7]], "saddle"),
    ([[.7, -.2], [.2, .7]], "attracting"),
    ([[1.2, -.3], [.3, 1.2]], "repelling"),
])
def test_find_known_affine_fixed_point_deduplicates_and_classifies_discrete_map(
        matrix, classification):
    bounds = np.array([[-.1, .5], [-4., 2.]])
    expected = np.array([.13, -.7])
    cloud = expected + np.array([[-.02, -.2], [-.02, .2], [.02, -.2], [.02, .2]])
    model = AffineMap(matrix, expected).eval().requires_grad_(False)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    roots, receipt = overlay.find_roots(model, bounds, [cloud], nx=3, ny=3,
                                        max_nfev=128)

    assert len(roots) == 1
    root = roots[0]
    np.testing.assert_allclose(root["g"], expected, rtol=0, atol=1e-8)
    np.testing.assert_allclose(root["full_DR"], matrix, rtol=0, atol=1e-12)
    assert root["classification"] == classification
    assert root["residual"] < 1e-8
    assert root["scaled_residual_inf"] <= 1e-9
    assert root["inside_evaluation_local_envelope"] is True
    actual_spectrum = [complex(item["real"], item["imag"]) for item in root["eigenvalues"]]
    np.testing.assert_allclose(np.sort_complex(actual_spectrum),
                               np.sort_complex(np.linalg.eigvals(matrix)), atol=1e-12)
    for eigenvalue in root["eigenvalues"]:
        assert eigenvalue["modulus"] == pytest.approx(
            abs(complex(eigenvalue["real"], eigenvalue["imag"])))
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in model.parameters())
    # Public diagnostics must not retain NumPy scalars/arrays or torch tensors.
    json.dumps({"roots": roots, "receipt": receipt}, allow_nan=False)


def test_optimizer_success_without_a_zero_is_not_a_fixed_point():
    roots, receipt = overlay.find_roots(TranslationMap(), [[-1., 1.], [-1., 1.]],
                                        nx=3, ny=3, max_nfev=32)
    assert roots == []
    assert any(attempt["optimizer_success"] for attempt in receipt["attempts"])
    assert all(not attempt["accepted"] for attempt in receipt["attempts"])
    assert all(attempt["scaled_residual_inf"] > 1e-9
               for attempt in receipt["attempts"])


def test_root_outside_requested_frame_is_excluded():
    model = AffineMap([[.8, 0.], [0., .6]], [2., .1])
    roots, _ = overlay.find_roots(model, [[-1., 1.], [-1., 1.]],
                                  nx=3, ny=3, max_nfev=32)
    assert roots == []


def test_root_between_separate_clouds_is_not_reported_as_covered():
    model = AffineMap([[1.3, 0.], [0., .8]], [0., 0.])
    box = np.array([[-.1, -.1], [-.1, .1], [.1, -.1], [.1, .1]])
    clouds = [box + [-.6, 0.], box + [.6, 0.]]
    roots, _ = overlay.find_roots(model, [[-1., 1.], [-1., 1.]], clouds,
                                  nx=3, ny=3, max_nfev=32)
    assert len(roots) == 1
    assert roots[0]["inside_evaluation_local_envelope"] is False


def test_full_DR_matches_central_difference_including_residual_identity():
    model = NonlinearResidualMap().eval()
    points = np.array([[.2, -.4], [-.6, .8]], dtype=np.float64)
    mapped, derivative = full_map_and_jacobian(model, points, "cpu")
    h = 1e-6
    finite_difference = np.empty_like(derivative)
    with torch.no_grad():
        for axis in range(2):
            offset = np.eye(2)[axis] * h
            plus = model.advance(torch.as_tensor(points + offset)).numpy()
            minus = model.advance(torch.as_tensor(points - offset)).numpy()
            finite_difference[:, :, axis] = (plus - minus) / (2 * h)
    np.testing.assert_allclose(derivative, finite_difference, rtol=1e-8, atol=1e-9)
    expected = np.zeros_like(derivative)
    expected[:, 0, 0] = 1 + .3 * np.cos(points[:, 0])
    expected[:, 0, 1] = .2
    expected[:, 1, 0] = .2 * points[:, 0]
    expected[:, 1, 1] = 1 - .4 / np.cosh(points[:, 1]) ** 2
    np.testing.assert_allclose(derivative, expected, rtol=1e-14, atol=1e-14)
    assert mapped.shape == points.shape
    assert model.coefficient.grad is None
