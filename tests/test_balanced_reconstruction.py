import numpy as np
import pytest
from scipy import sparse
import torch

from recurrent_rg.balanced_reconstruction import fit_balanced_metric, balanced_reconstruction_loss
from recurrent_rg.checkpoints import build_model
from recurrent_rg.config import ModelConfig
from experiments.q_balanced_reconstruction import encode_sparse
from experiments.q_only_cached_sweep import sample_counts
from experiments.rsmi_separate_qcr import build_parser, main


def test_metric_equalizes_supported_variance_without_initializing_model():
    p = torch.tensor([[.45,.15,.25,.15], [.15,.45,.15,.25],
                      [.40,.20,.15,.25], [.20,.40,.25,.15]], dtype=torch.double)
    metric = fit_balanced_metric(p, p, max_directions=2)
    assert metric.axes.shape == (4, 2)
    normalized = ((p-p.mean(0))@metric.axes).square().mean(0)/metric.denominators
    torch.testing.assert_close(normalized, torch.ones(2, dtype=torch.double))


def test_noise_only_direction_not_forced_and_target_detached():
    center = torch.full((4,4), .25, dtype=torch.double)
    delta = torch.tensor([[.01,-.01,0,0],[-.01,.01,0,0]]*2, dtype=torch.double)
    metric = fit_balanced_metric(center+delta, center-delta, max_directions=2)
    assert metric.axes.shape[1] == 0
    decoder = torch.nn.Linear(2,4).double()
    theory = torch.randn(4,2,dtype=torch.double,requires_grad=True)
    target = center.clone().requires_grad_(True)
    loss = balanced_reconstruction_loss(decoder,theory,target,metric)
    loss.backward()
    assert target.grad is None and theory.grad is not None


def test_sparse_encoder_preserves_categorical_forward_and_gradient():
    torch.set_num_threads(1); torch.manual_seed(5)
    model = build_model(ModelConfig(dimension=2,block_factor=3,categorical_states=4,
                        kernel_sampling='categorical',hidden=6,affine_reconstruction=True),1)
    x = torch.nn.functional.one_hot(torch.randint(4,(1,4,6,6)),4).double()
    h = sparse.csr_matrix(sample_counts(x,model.geometry,4)[[0,2]].mean(0))
    q = model.theory_encoder
    a=q(x[:,[0,2]]); b=encode_sparse(q,h,4,9)
    torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-10)
    parameters=[p for name,p in q.named_parameters() if not name.startswith('distribution_decoder.')]
    ga=torch.autograd.grad(a.sum(),parameters,retain_graph=True);gb=torch.autograd.grad(b.sum(),parameters)
    for aa,bb in zip(ga,gb):torch.testing.assert_close(aa,bb,atol=1e-10,rtol=1e-10)


def test_pilot_is_opt_in_and_rejects_supervision():
    args=build_parser().parse_args(['--source','x','--output','y'])
    assert not args.q_balanced_reconstruction
    with pytest.raises(ValueError,match='no other ablations'):
        main(['--source','x','--output','y','--q-balanced-reconstruction','--q-coupling-weight','20'])
