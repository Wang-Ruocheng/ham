import json
import torch
from recurrent_rg.config import ModelConfig, RunConfig, DataConfig, LossConfig
from recurrent_rg.checkpoints import build_model, load_run, qcr_state_dict


def test_new_and_legacy_defaults():
    data = {'cache':'unused.pt', 'physical_model':'test'}
    assert ModelConfig().kernel_sampling == 'bernoulli'
    assert ModelConfig().kernel_initialization == 'random'
    assert RunConfig.from_dict({'data':data}).model.kernel_sampling == 'deterministic'
    assert RunConfig.from_dict({'data':data,'schema_version':1}).model.kernel_sampling == 'deterministic'
    cfg = RunConfig.from_dict({'data':data,'schema_version':2})
    assert cfg.model.kernel_sampling == 'bernoulli'
    assert cfg.loss.orientation_weight == 0
    assert cfg.loss.kernel_sign_weight == 0
    assert RunConfig.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()


def test_sign_default_versioning():
    data = {'cache': 'unused', 'physical_model': 'test'}
    fresh = RunConfig(data=DataConfig(**data))
    assert fresh.schema_version == 5
    assert LossConfig().kernel_sign_weight == 0
    assert RunConfig.from_dict(fresh.to_dict()).loss.kernel_sign_weight == 0
    assert RunConfig.from_dict({'data':data,'schema_version':4}).loss.kernel_sign_weight == 0
    assert RunConfig.from_dict({'data':data,'schema_version':4,'loss':{'kernel_sign_weight':.1}}).loss.kernel_sign_weight == .1
    cfg = RunConfig.from_dict({'data': data, 'schema_version': 3})
    assert cfg.loss.kernel_sign_weight == .1
    assert RunConfig.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()
    for version in (1, 2):
        assert RunConfig.from_dict({'data': data, 'schema_version': version}).loss.kernel_sign_weight == 0
        explicit = RunConfig.from_dict({'data': data, 'schema_version': version, 'loss': {'kernel_sign_weight': .1}})
        assert explicit.loss.kernel_sign_weight == .1
    for model in ({'fixed_kernel': True}, {'kernel_sampling': 'deterministic'}):
        assert RunConfig.from_dict({'data': data, 'schema_version': 3, 'model': model}).loss.kernel_sign_weight == 0
    assert RunConfig.from_dict({'data': data, 'schema_version': 3, 'loss': {'kernel_sign_weight': 0}}).loss.kernel_sign_weight == 0


def test_fixed_config_default():
    cfg = RunConfig.from_dict({'schema_version':2,'data':{'cache':'unused','physical_model':'test'},'model':{'fixed_kernel':True}})
    assert cfg.model.kernel_sampling == 'deterministic'


def test_historical_versioned_checkpoint_keeps_deterministic_C(tmp_path):
    cfg = RunConfig.from_dict({'schema_version':1,'data':{'cache':'unused','physical_model':'test'},'model':{'hidden':4}})
    model = build_model(cfg.model, cfg.training.max_rg_steps)
    values = cfg.to_dict(); values['model'].pop('kernel_sampling')
    (tmp_path/'manifest.json').write_text(json.dumps({'config':values}))
    torch.save(qcr_state_dict(model), tmp_path/'qcr_model.pt')
    restored = load_run(tmp_path)
    assert restored.model_config.kernel_sampling == 'deterministic'
    assert restored.config['model']['kernel_sampling'] == 'deterministic'
