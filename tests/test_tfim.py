import math

import numpy as np
import pytest
import torch

from recurrent_rg.tfim import (
    exact_tfim_block_rg_flow,
    sample_tfim_ensembles,
    wolff_add_probability,
)


def test_wolff_probability_has_correct_zero_and_formula() -> None:
    assert wolff_add_probability(0.0) == 0.0
    assert wolff_add_probability(0.5) == pytest.approx(1.0 - math.exp(-1.0))
    assert 0.0 < wolff_add_probability(0.2) < wolff_add_probability(1.0) < 1.0
    with pytest.raises(ValueError):
        wolff_add_probability(-0.1)


def test_tfim_sampler_contract_and_binary_spins() -> None:
    configurations, couplings = sample_tfim_ensembles(
        np.asarray([0.25, 1.5]),
        replicas=1,
        samples=4,
        sites=8,
        seed=7,
        time_slices=8,
        equilibration_sweeps=8,
        decorrelation_sweeps=2,
    )
    assert configurations.shape == (2, 4, 8, 1)
    assert torch.equal(couplings, torch.tensor([0.25, 1.5], dtype=torch.float64))
    assert set(configurations.unique().tolist()) <= {-1.0, 1.0}


def test_exact_block_flow_shape_and_zero_fixed_point() -> None:
    couplings = torch.tensor([0.0, 1.0], dtype=torch.float64)
    flow = exact_tfim_block_rg_flow(couplings, levels=4)
    assert flow.shape == (2, 4)
    assert torch.equal(flow[0], torch.zeros(4, dtype=torch.float64))
