from dataclasses import replace
from pathlib import Path

import pytest
import torch

from experiments.rsmi_separate_qcr import independent_component_gradients, build_parser, validate_recipe
from recurrent_rg.balanced_reconstruction import fit_balanced_metric, refresh_training_metrics
from recurrent_rg.checkpoints import build_model, checkpoint_payload, load_run, qcr_state_dict
from recurrent_rg.config import ModelConfig, RunConfig, DataConfig
from recurrent_rg.data.statistics import categorical_plaquette_distribution
from recurrent_rg.losses import qcr_objective


def spec():
    return ModelConfig(dimension=2, theory_dim=2, block_factor=3, categorical_states=3,
                       kernel_sampling='categorical', hidden=6, affine_reconstruction=True)


def test_balanced_config_changes_only_explicit_Q_weight():
    root = Path(__file__).resolve().parents[1]
    old = RunConfig.load(root/'configs/dual_ising2d_uniformK1K2_g2_affineQ_joint_1200_seed91.json')
    new = RunConfig.load(root/'configs/dual_ising2d_uniformK1K2_g2_affineQ_balanced1_joint_1200_seed91.json')
    new.validate()
    assert new == replace(old, loss=replace(old.loss, balanced_reconstruction_weight=1.))
    validate_recipe(build_parser().parse_args(['--config','unused','--output','unused']), new)
    assert old.loss.balanced_reconstruction_weight == 0.


def test_balanced_extra_loss_changes_Q_gradients_but_not_C_R():
    torch.set_num_threads(1); torch.manual_seed(7)
    p = torch.rand(16,81,dtype=torch.double); p /= p.sum(-1,keepdim=True)
    metric = fit_balanced_metric(p,p,max_directions=2)
    x = torch.nn.functional.one_hot(torch.randint(3,(3,8,9,9)),3).double()
    gradients = []
    for weight in (0.,1.):
        torch.manual_seed(91); m = build_model(spec(),1); m.detach_kernel_representation=True
        m.balanced_reconstruction_metrics = [metric,metric]
        torch.manual_seed(31)
        total, terms = qcr_objective(m,x,pattern_fn=categorical_plaquette_distribution,
            balanced_reconstruction_weight=weight, kernel_noncollapse_mode='variance_floor_ensemble',
            kernel_variance_ratio=1.,kernel_sign_weight=1.)
        closure = 5*(terms['commutator']+terms['rollout'])
        c_loss = 2*terms['kernel_noncollapse']+terms['kernel_sign']
        independent_component_gradients(total,closure,c_loss,m.theory_encoder.parameters(),
                                         m.rg_kernel.parameters(),m.theory_rg.parameters())
        gradients.append([[p.grad.clone() for p in module.parameters()]
                          for module in (m.theory_encoder,m.rg_kernel,m.theory_rg)])
        assert ('balanced_reconstruction' in terms) == bool(weight)
    assert any(not torch.equal(a,b) for a,b in zip(gradients[0][0],gradients[1][0]))
    for ga,gb in zip(gradients[0][1:],gradients[1][1:]):
        for a,b in zip(ga,gb): torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)


def test_refresh_preserves_rng_weights_and_checkpoint_metric_roundtrip(tmp_path):
    torch.set_num_threads(1);torch.manual_seed(41)
    m = build_model(spec(),1); m.train()
    x = torch.nn.functional.one_hot(torch.randint(3,(6,8,9,9)),3).double()
    before = {k:v.clone() for k,v in qcr_state_dict(m).items()}; rng = torch.get_rng_state()
    refresh_training_metrics(m,x,categorical_plaquette_distribution,samples=8,chunk_size=2,fitted_epoch=200)
    assert m.training and torch.equal(rng,torch.get_rng_state())
    for key,value in before.items(): torch.testing.assert_close(value,qcr_state_dict(m)[key],atol=0,rtol=0)
    cfg = RunConfig(data=DataConfig(cache='unused',physical_model='test'),model=spec())
    optimizer = torch.optim.Adam(m.parameters(),lr=.001)
    payload = checkpoint_payload(m,optimizer,torch.Generator(),201,metadata={'config':cfg.to_dict()})
    path = tmp_path/'checkpoint.pt';torch.save(payload,path)
    loaded = load_run(path).model
    assert loaded.balanced_metric_fitted_epoch == 200
    for a,b in zip(m.balanced_reconstruction_metrics,loaded.balanced_reconstruction_metrics):
        for key in a.state_dict(): torch.testing.assert_close(a.state_dict()[key],b.state_dict()[key],atol=0,rtol=0)


def test_balanced_requires_fitted_train_metric_instead_of_fitting_test_data():
    torch.manual_seed(10);m=build_model(spec(),1)
    x=torch.nn.functional.one_hot(torch.randint(3,(2,4,9,9)),3).double()
    with pytest.raises(ValueError,match='TRAIN-only'):
        qcr_objective(m,x,pattern_fn=categorical_plaquette_distribution,
                      balanced_reconstruction_weight=1.,kernel_noncollapse_mode='variance_floor_ensemble')
