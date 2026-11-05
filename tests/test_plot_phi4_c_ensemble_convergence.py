"""Synthetic aggregate-schema tests only; no model, raw field, or inference."""
from __future__ import annotations

import json

import numpy as np
import pytest

from scripts import plot_phi4_c_ensemble_convergence as plot


def aggregate_document():
    metrics = {name: {"mean": .4, "chain_means": [.3, .4, .4, .5]}
               for name in plot.PLOT_METRIC_NAMES}
    metrics.update(
        aligned_distance_decomposition={"mse_over_s2": .16, "energy_over_s2": 1.0},
        unaligned_distance_decomposition={"mse_over_s2": .25, "energy_over_s2": 1.0},
    )
    return {"metadata": {"n_physical_per_condition": 64, "n_c_draws": 2},
            "results": [{"kappa": .71, "method": "learned_sample", "scale": 12,
                         "metrics": metrics}]}


def test_real_layout_extra_decompositions_do_not_require_mean():
    # JSON round-trip tests the public wire format, not scientific measurements.
    document = json.loads(json.dumps(aggregate_document()))
    result = plot.canonical_rows(document)
    assert result[0]["metrics"] == {name: .4 for name in plot.PLOT_METRIC_NAMES}
    assert "aligned_distance_decomposition" in document["results"][0]["metrics"]


def test_plain_numeric_metrics_also_supported():
    document = aggregate_document()
    document["results"][0]["metrics"]["q"] = .7
    assert plot.canonical_rows(document)[0]["metrics"]["q"] == .7


@pytest.mark.parametrize("value, message", [
    ({"mse_over_s2": .4}, "Expected a mean aggregate"),
    ({"mean": np.nan}, "Nonfinite aggregate"),
    (np.inf, "Nonfinite aggregate"),
])
def test_malformed_or_nonfinite_plotted_metric_rejected(value, message):
    document = aggregate_document()
    document["results"][0]["metrics"]["q"] = value
    with pytest.raises(ValueError, match=message):
        plot.canonical_rows(document)


def test_duplicate_condition_method_scale_rejected():
    document = aggregate_document()
    document["results"].append(document["results"][0])
    with pytest.raises(ValueError, match="Duplicate aggregate"):
        plot.canonical_rows(document)
