import json

import pytest

from scripts.preview_phi4_ru_cnn_checkpoint import choose_checkpoint, sha, restore_model


def test_only_status_confirmed_checkpoint_is_selected(tmp_path):
    (tmp_path/'status.json').write_text(json.dumps(dict(actual_steps=2950, latest_checkpoint_step=2000)))
    old = tmp_path/'training_step_2000.pt'
    old.write_bytes(b'complete saved model')
    (tmp_path/'training_step_3000.pt').write_bytes(b'incomplete write')
    assert choose_checkpoint(tmp_path) == (old, 2000, sha(old))


@pytest.mark.parametrize('step', [0, 3000, True])
def test_invalid_saved_step_fails_closed(tmp_path, step):
    (tmp_path/'status.json').write_text(json.dumps(dict(actual_steps=2000, latest_checkpoint_step=step)))
    with pytest.raises(ValueError):
        choose_checkpoint(tmp_path)


@pytest.mark.parametrize('nonlinear_decoder', [False, True, 'local'])
def test_restore_exact_affine_and_nonlinear_architectures(nonlinear_decoder):
    import torch
    from recurrent_rg.phi4_cnn_qcr import CNNPhi4QCR
    from experiments import phi4_ru_cnn_nonlinear as nonlinear
    model = CNNPhi4QCR(1., seed=91, theory_dim=2)
    recipe = 'phi4_ru_b3_cnn_g2_Rlr003_saddle_window_v1'
    if nonlinear_decoder:
        nonlinear.install_decoder(model)
        recipe = nonlinear.RECIPE
        if nonlinear_decoder == 'local':
            from experiments.phi4_ru_cnn_local_window import RECIPE
            recipe = RECIPE
        with torch.no_grad():
            for parameter in model.Q.decoder.parameters():
                parameter.add_(.001)
    manifest = dict(recipe=recipe, scale=1., theory_dim=2,
        lattice_sizes=[36, 12, 4], labels_used_in_training=False)
    if nonlinear_decoder:
        manifest['decoder'] = dict(nonlinear.DECODER)
    saved = dict(recipe=recipe, manifest=manifest, model=model.state_dict())
    restored = restore_model(saved, manifest)
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])
    with pytest.raises(ValueError, match='Incompatible'):
        restore_model(dict(saved, recipe='wrong'), manifest)
    if nonlinear_decoder:
        incompatible = dict(manifest, decoder=dict(nonlinear.DECODER, hidden=[64,64]))
        with pytest.raises(ValueError, match='decoder policy'):
            restore_model(dict(saved, manifest=incompatible), incompatible)
