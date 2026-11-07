import pytest
import torch
from recurrent_rg.probability_normalization import stable_log_softmax, stable_softmax
from scripts.verify_vae_normalization import verify


def test_625_bin_probability_and_gradient_contracts():
    result = verify('cpu')
    assert result['status'] == 'passed' and len(result['cases']) == 12
    assert {'random_16', 'uniform_16'} <= {row['case'] for row in result['cases']}
    assert result['binary_parameter_gradient_CPU_max_error'] < 1e-9
    assert result['model_updates'] == 0 and not result['physical_data_loaded']


def test_nonlast_dimension_and_tied_maximum():
    x = torch.zeros(2, 5, 3, dtype=torch.double, requires_grad=True)
    weight = torch.arange(30, dtype=torch.double).reshape_as(x)
    actual = stable_log_softmax(x, dim=1)
    expected = x.log_softmax(1)
    a, = torch.autograd.grad((weight*actual).sum(), x, retain_graph=True)
    b, = torch.autograd.grad((weight*expected).sum(), x)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(stable_softmax(x, dim=1).sum(1), torch.ones(2, 3, dtype=torch.double))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Run only on an admitted idle GPU')
def test_CUDA_625_bin_probability_and_gradient_contracts():
    assert verify('cuda:0')['status'] == 'passed'
