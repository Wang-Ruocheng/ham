"""Synthetic CPU-only tests of bounded two-coordinate flow diagnostics."""
import numpy as np
import pytest
import torch

from recurrent_rg.phi4_qcr import Phi4QCR
from recurrent_rg.phi4_ru_flow import bounded_flow, classify_eigenvalues, full_map_and_jacobian
from experiments.phi4_qcr import state_fingerprint


class LinearMap(torch.nn.Module):
    def __init__(self, matrix):
        super().__init__()
        self.register_buffer("matrix", torch.as_tensor(matrix, dtype=torch.float64))

    def advance(self, x):
        return x @ self.matrix.T


def arrays(rank_one=False):
    support = np.array([[-1., -1.], [1., -1.], [-1., 1.], [1., 1.], [0., 0.]])
    if rank_one:
        support[:, 1] = 2 * support[:, 0]
    direct = np.stack([support, support * .8, support * .6], axis=1)
    return dict(train_direct=direct, test_direct=direct * 100)


def test_full_DR_contains_identity_and_does_not_set_parameter_gradients():
    torch.manual_seed(11)
    model = Phi4QCR(1., hidden=4, theory_dim=2)
    with torch.no_grad():
        for parameter in model.R.parameters():
            parameter.zero_()
    before = state_fingerprint(model)
    points = np.array([[.1, .2], [-.3, .4]])
    mapped, dr = full_map_and_jacobian(model, points, "cpu")
    np.testing.assert_array_equal(mapped, points)
    np.testing.assert_array_equal(dr, np.repeat(np.eye(2)[None], 2, axis=0))
    assert state_fingerprint(model) == before
    assert all(parameter.grad is None for parameter in model.parameters())


def test_known_saddle_bounded_roots_training_only_support_and_budget_callbacks():
    values, checks = arrays(), []
    roots, receipt = bounded_flow(LinearMap([[1.4, 0], [0, .6]]), values,
                                 "cpu", budget_check=lambda: checks.append(1))
    assert len(roots) == 1
    root = roots[0]
    np.testing.assert_allclose(root["g"], [0, 0], atol=1e-8)
    assert root["classification"] == "saddle"
    assert root["within_train_hull"] and root["local_rank"] == 2
    assert root["expanding_directions"] == root["contracting_directions"] == 1
    assert root["thermal_nu"] is None and not root["physical_critical_point_certified"]
    assert receipt["actual_starts"] == 49
    assert all(row["nfev"] <= 64 for row in receipt["root_search_attempts"])
    assert len(checks) > 10
    # TEST has hundred-fold larger support but cannot expand the search.
    np.testing.assert_allclose(values["flow_bounds"], [[-1.2, 1.2], [-1.2, 1.2]])
    assert values["flow_DR"].shape == (1089, 2, 2)
    np.testing.assert_allclose(values["flow_displacement"], values["flow_R"] - values["flow_g"])
    assert values["flow_in_train_hull"].sum() < 1089
    assert values["flow_pca_explained_variance_ratio"].sum() == pytest.approx(1.)


@pytest.mark.parametrize("matrix,classification", [
    ([[-1.2, 0], [0, .5]], "saddle"),
    ([[1.1, -.4], [.4, 1.1]], "repelling"),
    ([[.2, -.4], [.4, .2]], "attracting"),
])
def test_preserve_negative_and_complex_eigenvalues(matrix, classification):
    values = arrays()
    roots, _ = bounded_flow(LinearMap(matrix), values, "cpu")
    assert len(roots) == 1 and roots[0]["classification"] == classification
    expected = np.linalg.eigvals(matrix)
    actual = np.asarray([complex(row["real"], row["imag"]) for row in roots[0]["eigenvalues"]])
    np.testing.assert_allclose(np.sort_complex(actual), np.sort_complex(expected))
    assert np.any(values["flow_eigenvalues_real"] < 0) or np.any(values["flow_eigenvalues_imag"] != 0)


def test_rank_one_is_reported_without_synthetic_two_dimensional_roots():
    values = arrays(rank_one=True)
    roots, receipt = bounded_flow(LinearMap([[1.4, 0], [0, .6]]), values, "cpu")
    assert roots == [] and receipt["train_input_rank"] == 1
    assert not receipt["hull_available"] and receipt["actual_starts"] == 0
    assert not values["flow_in_train_hull"].any()
    assert values["flow_pca_explained_variance_ratio"][1] < 1e-20


def test_frozen_search_limits_and_neutral_classification():
    assert classify_eigenvalues([1., .5])[0] == "neutral_or_unresolved"
    with pytest.raises(ValueError, match="33/7/64"):
        bounded_flow(LinearMap(np.eye(2)), arrays(), "cpu", max_nfev=65)
