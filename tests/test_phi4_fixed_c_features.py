import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments import phi4_fixed_c_features as features
from experiments import phi4_qcr as original
from experiments.phi4_ru_r_cache import encode_chain
from experiments.phi4_scalar_r_cache import seed, subset
from recurrent_rg.phi4_qcr import Phi4QCR


def test_half_features_reproduce_existing_cache_without_parameter_or_rng_mutation():
    torch.manual_seed(11)
    model = Phi4QCR(1., hidden=8, theory_dim=1).double().eval().requires_grad_(False)
    fields = torch.randn(128, 9, 9, 1, dtype=torch.float64)
    indices, cseed = subset("fit", 0, 2), seed("fit", 0, 2, "C")
    before, rng = original.state_fingerprint(model), torch.get_rng_state().clone()
    expected = encode_chain(model, fields, indices, c_seed=cseed)
    actual = features.encode_half_features(model, fields, indices, c_seed=cseed)
    errors = features.compare_reference(actual, expected["feature_target"], expected["direct"])
    assert errors == (0., 0.)
    assert actual["half_features"].shape == (3, 2, 105)
    np.testing.assert_allclose(actual["half_features"].sum(-1), 1., rtol=0., atol=1e-14)
    assert original.state_fingerprint(model) == before
    assert all(parameter.grad is None for parameter in model.parameters())
    assert torch.equal(torch.get_rng_state(), rng)


def test_compare_reference_rejects_feature_or_encoding_disagreement():
    halves = np.zeros((3, 2, 105), dtype=np.float64)
    direct = np.zeros((3, 1), dtype=np.float64)
    values = dict(half_features=halves, source_direct=direct.copy())
    features.compare_reference(values, halves.mean(1), direct)
    values["half_features"][0, 0, 0] = 1e-8
    with pytest.raises(RuntimeError, match="feature mean"):
        features.compare_reference(values, np.zeros((3, 105)), direct)
    values["half_features"].fill(0.)
    values["source_direct"][0, 0] = 1e-8
    with pytest.raises(RuntimeError, match="re-encoding"):
        features.compare_reference(values, np.zeros((3, 105)), direct)
    values["source_direct"].fill(np.nan)
    with pytest.raises(ValueError, match="finite"):
        features.compare_reference(values, np.zeros((3, 105)), direct)


@pytest.mark.parametrize("bad_indices", [np.zeros(64, dtype=np.int64), np.arange(64, dtype=float)])
def test_encoder_rejects_repeated_or_noninteger_sample_identity(bad_indices):
    model = Phi4QCR(1., hidden=4).eval().requires_grad_(False)
    fields = torch.zeros(128, 9, 9, 1, dtype=torch.float64)
    with pytest.raises(ValueError, match="unique"):
        features.encode_half_features(model, fields, bad_indices, c_seed=1)


def test_encoder_requires_frozen_source_model():
    model = Phi4QCR(1., hidden=4)
    fields = torch.zeros(128, 9, 9, 1, dtype=torch.float64)
    with pytest.raises(ValueError, match="frozen"):
        features.encode_half_features(model, fields, np.arange(64), c_seed=1)


def test_inventory_rejects_unpinned_or_changed_old_cache(tmp_path):
    inventory = {name: "0" * 64 for name in ("manifest.json", "receipt.json", "status.json", "cache.npz")}
    for name in inventory:
        (tmp_path / name).write_bytes(b"changed")
    (tmp_path / "public_files.json").write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match="changed"):
        features._verify_cache(tmp_path)
    inventory["unexpected.pt"] = "0" * 64
    (tmp_path / "public_files.json").write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match="four-member"):
        features._verify_cache(tmp_path)


def test_run_preserves_immutable_output_scope_and_fixed_budget(tmp_path):
    source, cache = tmp_path / "source", tmp_path / "cache"
    source.mkdir()
    cache.mkdir()
    with pytest.raises(ValueError, match="outside"):
        features.run(source, cache, source / "nested")
    assert not (source / "nested").exists()
    with pytest.raises(ValueError, match="600"):
        features.run(source, cache, tmp_path / "output", 601.)
    assert not (tmp_path / "output").exists()
    with pytest.raises(FileExistsError):
        features.run(source, cache, tmp_path)


def test_run_writes_complete_hash_inventory_and_verifies_every_reused_draw(tmp_path, monkeypatch):
    source, cache, output = (tmp_path / name for name in ("source", "cache", "new"))
    source.mkdir()
    cache.mkdir()
    model = Phi4QCR(1., hidden=4).double().eval().requires_grad_(False)
    train, test = np.arange(32), np.arange(32, 52)
    dataset = SimpleNamespace(train_indices=train, test_indices=test,
        configurations=torch.arange(52, dtype=torch.float64)[:, None, None, None, None].expand(52, 128, 1, 1, 1),
        cell_ids=np.arange(52) // 4, labels=np.zeros((52, 3)))

    def fake_encode(model, field, indices, *, c_seed):
        value = float(field[0, 0, 0, 0]) + (c_seed % 1000) / 1000
        halves = np.full((3, 2, 105), value, dtype=np.float64)
        return dict(half_features=halves, source_direct=np.full((3, 1), value, dtype=np.float64))

    arrays = {}
    for split, selected in (("train", train), ("test", test)):
        arrays[f"{split}_cell_ids"] = dataset.cell_ids[selected]
        arrays[f"{split}_labels"] = dataset.labels[selected]
    for prefix, purpose, split in features.PREFIXES:
        selected = train if split == "train" else test
        arrays[f"{prefix}_feature_target"] = np.empty((4, len(selected), 3, 105))
        arrays[f"{prefix}_direct"] = np.empty((4, len(selected), 3, 1))
        arrays[f"{prefix}_subset_indices"] = np.empty((4, len(selected), 64), dtype=np.int64)
        arrays[f"{prefix}_c_seeds"] = np.empty((4, len(selected)), dtype=np.int64)
        for repeat in range(4):
            for row, chain in enumerate(selected):
                indices, cseed = subset(purpose, repeat, int(chain)), seed(purpose, repeat, int(chain), "C")
                value = fake_encode(model, dataset.configurations[chain], indices, c_seed=cseed)
                arrays[f"{prefix}_feature_target"][repeat, row] = value["half_features"].mean(1)
                arrays[f"{prefix}_direct"][repeat, row] = value["source_direct"]
                arrays[f"{prefix}_subset_indices"][repeat, row] = indices
                arrays[f"{prefix}_c_seeds"][repeat, row] = cseed
    np.savez_compressed(cache / "cache.npz", **arrays)
    cachepins = {str(cache / "cache.npz"): original.sha256_file(cache / "cache.npz")}
    monkeypatch.setattr(features, "_verify_cache", lambda cache: (dict(data_warning_count=13), cachepins))
    monkeypatch.setattr(features, "_load_source", lambda source, manifest: (model, dataset, {}))
    monkeypatch.setattr(features, "encode_half_features", fake_encode)
    monkeypatch.setattr(torch, "set_num_threads", lambda count: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda count: None)
    before = original.state_fingerprint(model)
    manifest = features.run(source, cache, output)
    assert manifest["old_cache_feature_target_max_abs_difference"] == 0.
    assert manifest["old_cache_direct_g_max_abs_difference"] == 0.
    assert manifest["optimizer_updates"] == dict(Q=0, C=0, R=0)
    status = json.loads((output / "status.json").read_text())
    assert status["workflow_complete"] is True and status["returncode"] == 0
    assert status["encoded_rows"] == 336
    inventory = json.loads((output / "public_files.json").read_text())
    assert set(inventory) == {"manifest.json", "receipt.json", "status.json", "features.npz"}
    for name, digest in inventory.items():
        assert original.sha256_file(output / name) == digest
    with np.load(output / "features.npz", allow_pickle=False) as result:
        assert result["fit_half_features"].shape == (4, 32, 3, 2, 105)
        assert result["eval_test_half_features"].shape == (4, 20, 3, 2, 105)
        np.testing.assert_array_equal(result["fit_subset_indices"], arrays["fit_subset_indices"])
        np.testing.assert_array_equal(result["eval_train_c_seeds"], arrays["eval_train_c_seeds"])
    assert original.state_fingerprint(model) == before
