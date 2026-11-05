import numpy as np
import pytest
import torch

from recurrent_rg.manifold import GraphCoordinate
from recurrent_rg.distribution_encoder import categorical_local_distribution, LocalDistributionEncoder
from recurrent_rg.data.statistics import categorical_plaquette_distribution


def test_graph_coordinate_interval_and_independent_queries():
    t = np.linspace(-1, 1, 31)
    reference = np.column_stack((t, t*t))
    graph = GraphCoordinate.fit(reference, neighbors=4)
    z = graph.training_coordinates()
    assert np.all(np.diff(z) > 0) or np.all(np.diff(z) < 0)
    query = np.array([[.12, .12**2]])
    np.testing.assert_array_equal(graph.transform(query), graph.transform(np.repeat(query, 10, axis=0))[:1])
    np.testing.assert_allclose(graph.transform(reference), z, atol=1e-10)


def test_disconnected_graph_is_not_silently_repaired():
    x = np.r_[np.arange(5), 100+np.arange(5)][:, None]
    with pytest.raises(ValueError, match='Disconnected'):
        GraphCoordinate.fit(x, neighbors=2)


@pytest.mark.parametrize('dimensions', [1, 2, 3])
def test_categorical_distribution_generalizes_dimensions(dimensions):
    state = torch.randint(3, (2, 4)+ (4,)*dimensions)
    x = torch.nn.functional.one_hot(state, 3).double().requires_grad_()
    p = categorical_local_distribution(x)
    assert p.shape == (2, 3**(2**dimensions))
    torch.testing.assert_close(p.sum(-1), torch.ones(2, dtype=torch.double))
    assert not p.requires_grad
    q = LocalDistributionEncoder(3, dimensions).double()
    q(x).sum().backward()
    assert q.projection.weight.grad is not None and x.grad is None


def test_histogram_matches_existing_oriented_plaquettes():
    state = torch.randint(3, (2, 4, 5, 5))
    x = torch.nn.functional.one_hot(state, 3).double()
    torch.testing.assert_close(categorical_local_distribution(x), categorical_plaquette_distribution(x))


def test_uniform_and_checkerboard_are_different_statistics():
    uniform = torch.zeros((1, 4, 6, 6), dtype=torch.long)
    checker = (torch.arange(6)[:, None]+torch.arange(6)[None, :])%2*2
    checker = checker[None, None].expand_as(uniform)
    p = categorical_local_distribution(torch.nn.functional.one_hot(uniform, 3).double())
    r = categorical_local_distribution(torch.nn.functional.one_hot(checker, 3).double())
    assert torch.linalg.vector_norm(p-r) > 1
