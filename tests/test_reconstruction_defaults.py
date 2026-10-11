from dataclasses import replace
import json
import torch

from experiments.rsmi_separate_qcr import default_representation
from recurrent_rg.config import DataConfig, ModelConfig, RunConfig
from recurrent_rg.checkpoints import build_model, load_run, qcr_state_dict
from recurrent_rg.representation import normalize_affine_decoder_


def test_fresh_models_and_driver_default_to_reconstruction_without_changing_history():
    assert ModelConfig().affine_reconstruction
    raw=dict(schema_version=5,data=dict(cache='unused',physical_model='test'),model=dict(dimension=2))
    historical=RunConfig.from_dict(raw)
    assert not historical.model.affine_reconstruction
    assert default_representation(historical,raw).model.affine_reconstruction
    assert not default_representation(historical,raw,continuing=True).model.affine_reconstruction
    raw['model']['affine_reconstruction']=False
    assert not default_representation(historical,raw).model.affine_reconstruction


def test_multidimensional_decoder_columns_are_normalized_without_full_rank_constraint():
    m=build_model(ModelConfig(dimension=2,theory_dim=2,hidden=5),2)
    decoder=m.theory_encoder.distribution_decoder
    with torch.no_grad():decoder.weight[:,1].copy_(3*decoder.weight[:,0])
    normalize_affine_decoder_(m.theory_encoder)
    torch.testing.assert_close(decoder.weight.norm(dim=0),torch.ones(2,dtype=torch.double))
    assert torch.linalg.matrix_rank(decoder.weight)==1


def test_loading_old_schema5_weights_does_not_add_decoder(tmp_path):
    cfg=RunConfig(data=DataConfig('unused','test'),
                  model=ModelConfig(dimension=2,hidden=5,affine_reconstruction=False))
    m=build_model(cfg.model,cfg.training.max_rg_steps)
    raw=cfg.to_dict();del raw['model']['affine_reconstruction']
    (tmp_path/'manifest.json').write_text(json.dumps(dict(config=raw)))
    torch.save(qcr_state_dict(m),tmp_path/'qcr_model.pt')
    loaded=load_run(tmp_path)
    assert not loaded.model_config.affine_reconstruction
    assert not hasattr(loaded.model.theory_encoder,'distribution_decoder')
    for k,v in qcr_state_dict(m).items():
        torch.testing.assert_close(v,qcr_state_dict(loaded.model)[k],atol=0,rtol=0)
