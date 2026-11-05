import numpy as np
import pytest
import torch

from recurrent_rg.data.ensembles import load_ensembles
from recurrent_rg.data.spin_glass_pipeline import GenerationOptions, generate_dataset, iter_spin_glass
from scripts.prepare_spin_glass_qcr import prepare_overlap_cache


def test_two_dimensional_overlap_labels_and_plaquette_frustration(tmp_path):
    source, target = tmp_path / "physical", tmp_path / "data"
    generate_dataset(source, GenerationOptions(shape=(9, 12), temperatures=(1., 3.),
        disorders=3, replicas=2, samples=8, burn_in=4, spacing=1, seed=20261019))
    metadata = prepare_overlap_cache(source, target)
    train, test = load_ensembles(target / "ensembles.pt")
    assert metadata["dimension"] == 2 and metadata["lattice_shape"] == [9, 12]
    assert train.configurations.shape == (4, 8, 9, 12, 1)
    assert test.configurations.shape == (2, 8, 9, 12, 1)
    assert train.couplings is None and test.couplings is None
    assert set(train.group_ids[:, 0].tolist()).isdisjoint(test.group_ids[:, 0].tolist())
    for record in iter_spin_glass(source):
        dataset = train if record.split == "train" else test
        rows = metadata[record.split]
        bx, by = np.sign(record.bonds)
        signs = [bx[i, j] * by[(i + 1) % 9, j] * bx[i, (j + 1) % 12] * by[i, j]
                 for i in range(9) for j in range(12)]
        fraction = float(np.mean(np.asarray(signs) < 0))
        for temperature in range(2):
            index = torch.where((dataset.group_ids == torch.tensor([record.disorder_id, temperature])).all(1))[0].item()
            np.testing.assert_array_equal(dataset.configurations[index].numpy(), record.overlap_view()[temperature])
            row = rows[index]
            assert row["dimension"] == 2 and row["sites"] == 108
            assert row["lattice_shape"] == [9, 12] and "ring_flux" not in row
            assert row["frustrated_plaquette_fraction"] == fraction
            assert row["zero_plaquette_fraction"] == 0 and row["zero_bond_count"] == 0
    assert metadata["labels_used_in_training"] is False


def test_three_dimensional_data_not_silently_used_by_qcr(tmp_path):
    source = tmp_path / "physical"
    generate_dataset(source, GenerationOptions(shape=(3, 3, 3), temperatures=(1., 3.),
        disorders=3, replicas=2, samples=4, burn_in=0, spacing=1, seed=123))
    with pytest.raises(ValueError, match="one- and two-dimensional"):
        prepare_overlap_cache(source, tmp_path / "data")
    assert not (tmp_path / "data").exists()
