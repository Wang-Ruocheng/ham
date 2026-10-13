import json

import pytest
import torch

from recurrent_rg.config import RunConfig
from recurrent_rg.mi_cache import lattice_statistics_cache_key
from recurrent_rg.rsmi import region_offsets
from scripts.prepare_qcr_mi_cache import prepare
from scripts.run_scalar_lr_sweep import make_configs


def fixture_config(tmp_path):
    cache = tmp_path / 'cache.pt'
    generator = torch.Generator().manual_seed(91)
    train = (2 * torch.randint(2, (3, 8, 12, 12, 1), generator=generator) - 1).double()
    test = (2 * torch.randint(2, (2, 8, 12, 12, 1), generator=generator) - 1).double()
    torch.save((train, torch.tensor([.2, .3, .4]), test, torch.tensor([.5, .6])), cache)
    raw = make_configs(cache, epochs=8, threads=1)['multistep']
    raw['training']['max_rg_steps'] = raw['evaluation']['max_rg_steps'] = 1
    raw['training']['sample_batch'] = raw['evaluation']['samples'] = 8
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(raw))
    return config, RunConfig.from_dict(raw), train


def test_shared_cache_identity_preserves_original_v1_fields(tmp_path):
    _, cfg, train = fixture_config(tmp_path)
    _, env = region_offsets(2, block=3, buffer=1)
    actual = lattice_statistics_cache_key(cfg, 'a' * 64, 'train', train, env,
                                         'binary_lattice_mi_statistics', {'block': 3})
    expected = dict(data_sha256='a' * 64, split='train', states=2, dimension=2,
        block=3, buffer=1, environment_offsets=env.tolist(), flip=False, pooling=1,
        dtype='torch.float64', shape=list(train.shape), environment='full_shell',
        statistics='binary_lattice_mi_statistics', statistics_kwargs={'block': 3})
    assert actual == expected


def test_preprocessing_is_data_only_reusable_and_does_not_consume_rng(tmp_path):
    config, _, _ = fixture_config(tmp_path)
    before = torch.get_rng_state().clone()
    cold = prepare(config, tmp_path / 'cold.json')
    assert torch.equal(before, torch.get_rng_state())
    assert cold['optimizer_updates'] == cold['sampling_draws'] == 0
    assert not cold['model_initialized']
    assert not any(row['cache']['hit'] for row in cold['splits'].values())
    warm = prepare(config, tmp_path / 'warm.json')
    assert all(row['cache']['hit'] for row in warm['splits'].values())
    for split in ('train', 'test'):
        assert cold['splits'][split]['cache']['key'] == warm['splits'][split]['cache']['key']
    assert json.loads((tmp_path / 'warm.json').read_text())['status'] == 'complete'
    with pytest.raises(FileExistsError):
        prepare(config, tmp_path / 'warm.json')
