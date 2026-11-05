from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from recurrent_rg.config import RunConfig
from recurrent_rg.data.ensembles import load_ensembles
from recurrent_rg.data.spin_glass_pipeline import GenerationOptions, generate_dataset, iter_spin_glass
from scripts.prepare_spin_glass_qcr import prepare_overlap_cache


def test_overlap_adapter_keeps_physical_identity_and_removes_labels(tmp_path):
    source, target = tmp_path / "source", tmp_path / "cache"
    options = GenerationOptions(shape=(16,), temperatures=(.7, 2.), disorders=3,
        replicas=2, samples=8, burn_in=4, spacing=1, seed=77)
    generate_dataset(source, options)
    metadata = prepare_overlap_cache(source, target)
    tx, vx = load_ensembles(target / "ensembles.pt")
    assert tx.couplings is None and vx.couplings is None
    assert tx.configurations.shape == (4, 8, 16, 1)
    assert vx.configurations.shape == (2, 8, 16, 1)
    assert set(tx.group_ids[:, 0].tolist()) == {0, 2}
    assert set(vx.group_ids[:, 0].tolist()) == {1}
    for record in iter_spin_glass(source):
        ds = tx if record.split == "train" else vx
        for t in range(2):
            index = torch.where((ds.group_ids == torch.tensor([record.disorder_id, t])).all(1))[0].item()
            np.testing.assert_array_equal(ds.configurations[index].numpy(), record.overlap_view()[t])
    assert metadata["labels_used_in_training"] is False
    assert metadata["cache_sha256"]
    assert all(row["overlap_replicas"] == [0, 1] for split in ("train", "test") for row in metadata[split])
    with pytest.raises(FileExistsError):
        prepare_overlap_cache(source, target)


def test_default_spin_glass_config_obeys_recipe():
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "configs/spin_glass1d_overlap_g1_seed91.json"
    cfg = RunConfig.load(path)
    cfg.validate()
    assert cfg.schema_version == 5 and cfg.model.dimension == 1 and cfg.model.theory_dim == 1
    assert cfg.model.block_factor == 2 and cfg.model.affine_reconstruction
    assert cfg.model.kernel_initialization == "random" and cfg.model.kernel_sampling == "bernoulli"
    assert not cfg.data.coupling_names and cfg.loss.closure_scale_weights is None
    assert cfg.training.epochs == 6000 and set(cfg.training.train_modules) == {"Q", "C", "R"}
    assert cfg.loss.weights.anchor * 10 == 20
    assert cfg.loss.weights.coarse_consistency == 5 and cfg.loss.weights.separation == 10


def test_adapter_rejects_incomplete_source_without_creating_cache(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "status.json").write_text(json.dumps(dict(status="failed", workflow_complete=False)))
    with pytest.raises(ValueError, match="complete"):
        prepare_overlap_cache(source, tmp_path / "cache")
    assert not (tmp_path / "cache").exists()
