from dataclasses import asdict
import json
import pytest
import torch

from recurrent_rg.config import DataConfig, ModelConfig, RunConfig, ScaleConfig
from recurrent_rg.checkpoints import build_model, load_run, qcr_state_dict


def test_new_dimension_aware_defaults():
    assert ModelConfig().block_factor == 2
    assert ModelConfig(dimension=2).block_factor == 3
    assert ModelConfig(dimension=2, block_factor=2).block_factor == 2
    assert ScaleConfig((36, 36), 2).shapes() == [(36, 36), (12, 12), (4, 4)]
    with pytest.raises(ValueError, match='divisible'):
        ScaleConfig((16, 16), 2).shapes()
    cfg = RunConfig(data=DataConfig('unused', 'ising2d'), model=ModelConfig(dimension=2))
    assert cfg.schema_version == 5
    assert RunConfig.from_dict(asdict(cfg)) == cfg
    raw = asdict(cfg)
    del raw['model']['block_factor']
    assert RunConfig.from_dict(raw).model.block_factor == 3


@pytest.mark.parametrize('version', [None, 1, 2, 3, 4])
def test_old_missing_block_stays_two(version):
    raw = dict(data=dict(cache='unused', physical_model='ising2d'), model=dict(dimension=2))
    if version is not None:
        raw['schema_version'] = version
    assert RunConfig.from_dict(raw).model.block_factor == 2
    raw['model']['block_factor'] = 3
    assert RunConfig.from_dict(raw).model.block_factor == 3


@pytest.mark.parametrize('version,block', [(4, 2), (5, 3)])
def test_checkpoint_missing_block_uses_schema_default(tmp_path, version, block):
    spec = ModelConfig(dimension=2, hidden=4, block_factor=block)
    model = build_model(spec, 2)
    cfg = asdict(RunConfig(data=DataConfig('unused', 'ising2d'), model=spec))
    cfg['schema_version'] = version
    cfg['training']['max_rg_steps'] = 2
    del cfg['model']['block_factor']
    (tmp_path/'manifest.json').write_text(json.dumps(dict(config=cfg)))
    torch.save(dict(model=qcr_state_dict(model)), tmp_path/'qcr_model.pt')
    loaded = load_run(tmp_path)
    assert loaded.model_config.block_factor == block
    for key, value in qcr_state_dict(model).items():
        torch.testing.assert_close(qcr_state_dict(loaded.model)[key], value)
