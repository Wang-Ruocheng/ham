"""Training-data-only graph coordinates for explicit representation ablations.

This is an interval-manifold chart, not a general substitute for a
multidimensional representation. No physical coupling or phase labels enter it.
"""
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.spatial.distance import cdist


@dataclass
class GraphCoordinate:
    reference: np.ndarray
    paths: np.ndarray
    endpoints: tuple[int, int]
    neighbors: int

    @classmethod
    def fit(cls, reference, *, neighbors=4):
        x = np.asarray(reference, dtype=np.float64)
        if x.ndim != 2 or len(x) <= neighbors or neighbors < 1:
            raise ValueError('Expected more training reference points than neighbors')
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite training descriptors')
        distance = cdist(x, x)
        np.fill_diagonal(distance, np.inf)
        ids = np.argsort(distance, axis=1, kind='stable')[:, :neighbors]
        edges = np.zeros_like(distance)
        for i in range(len(x)):
            edges[i, ids[i]] = np.maximum(distance[i, ids[i]], 1e-12)
        edges = np.maximum(edges, edges.T)
        graph = csr_matrix(edges)
        if connected_components(graph, directed=False)[0] != 1:
            raise ValueError('Disconnected training graph: no silent connectivity repair')
        paths = shortest_path(graph, directed=False)
        a, b = np.unravel_index(np.argmax(paths), paths.shape)
        if paths[a, b] <= 1e-10:
            raise ValueError('Degenerate graph diameter')
        return cls(x.copy(), paths, (int(a), int(b)), neighbors)

    @property
    def diameter(self):
        return float(self.paths[self.endpoints])

    def training_coordinates(self):
        a, b = self.endpoints
        return (self.paths[:, a] - self.paths[:, b]) / self.diameter

    def transform(self, descriptors):
        """Attach queries independently; they cannot create training shortcuts."""
        x = np.asarray(descriptors, dtype=np.float64)
        if x.shape[-1] != self.reference.shape[1] or not np.isfinite(x).all():
            raise ValueError('Invalid query descriptors')
        flat = x.reshape(-1, x.shape[-1])
        distance = cdist(flat, self.reference)
        ids = np.argsort(distance, axis=1, kind='stable')[:, :self.neighbors]
        attach = np.take_along_axis(distance, ids, axis=1)
        a, b = self.endpoints
        da = np.min(attach + self.paths[ids, a], axis=1)
        db = np.min(attach + self.paths[ids, b], axis=1)
        return ((da-db) / self.diameter).reshape(x.shape[:-1])
