import numpy as np
import pytest

from scripts.plot_ising_primary_components import select_components


def test_seed91_uses_only_pc1_at_fixed_99_percent_rule():
    fractions = [.9981921201536161, .0017659004611851852,
                 3.964713934594071e-5, 2.328478645351226e-6, 3.767207376741365e-9]
    assert select_components(fractions).tolist() == [0]


def test_multiple_components_are_kept_when_needed():
    assert select_components([.8, .15, .05], .95).tolist() == [0, 1]
    assert select_components([.8, .15, .05], 1).tolist() == [0, 1, 2]


@pytest.mark.parametrize('fractions,threshold', [
    ([], .99), ([.8, .1], .99), ([.1, .9], .99),
    ([1.1], .99), ([-.1, 1.1], .99), ([np.nan], .99),
    ([np.inf], .99), ([1.], 0), ([1.], 1.01),
])
def test_invalid_or_truncated_variances_cannot_claim_coverage(fractions, threshold):
    with pytest.raises(ValueError):
        select_components(fractions, threshold)
