"""Small CPU-only checks of exact Gaussian sampling and its provenance contract."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from recurrent_rg.data import gaussian_scalar as data


def options(**changes):
    values = dict(sizes=(3,), r_values=(.1, .3, .8), groups=4, samples=8, seed=924)
    values.update(changes)
    return data.GaussianScalarOptions(**values)


def dense_precision(size, r):
    """Build the periodic nearest-neighbor quadratic form independently of FFTs."""
    precision = np.eye(size * size) * r
    for x in range(size):
        for y in range(size):
            source = x * size + y
            for nx, ny in (((x + 1) % size, y), (x, (y + 1) % size)):
                target = nx * size + ny
                difference = np.zeros(size * size)
                difference[source], difference[target] = 1., -1.
                precision += np.outer(difference, difference)
    return precision


@pytest.mark.parametrize("size,r", [(3, .02), (4, .7), (5, 2.)])
def test_fft_covariance_matches_inverse_dense_precision_with_correct_volume_normalization(size, r):
    precision = dense_precision(size, r)
    covariance = np.linalg.inv(precision)
    displacement = data.exact_covariance(size, r)
    for x in range(size):
        for y in range(size):
            for a in range(size):
                for b in range(size):
                    assert covariance[x * size + y, a * size + b] == pytest.approx(
                        displacement[(a - x) % size, (b - y) % size], rel=2e-13, abs=2e-14)
    assert data.lattice_spectrum(size, r)[0, 0] == r
    assert displacement.sum() == pytest.approx(1 / r)


def test_filtered_white_basis_is_exact_precision_inverse_square_root():
    size, r = 3, .4
    white = np.eye(size * size).reshape(size * size, size, size)
    original = white.copy()
    linear_map = data.filter_white_noise(white, r).reshape(size * size, size * size)
    np.testing.assert_allclose(linear_map.T @ linear_map,
                               np.linalg.inv(dense_precision(size, r)), rtol=3e-14, atol=2e-15)
    np.testing.assert_array_equal(white, original)


def test_constant_white_mode_is_retained_not_centered_or_rescaled():
    white = np.full((2, 6, 6), 2.)
    sampled = data.filter_white_noise(white, .25)
    np.testing.assert_allclose(sampled, 4., atol=1e-14, rtol=0)
    assert np.mean(sampled) != 0


def test_empirical_dense_covariance_mean_mode_and_action_match_exact_iid_distribution():
    size, r, count = 3, .5, 40000
    sampled = data.sample_gaussian_fields(r, size=size, samples=count, seed=1492)
    assert sampled.shape == (count, size, size, 1)
    assert sampled.dtype == np.float64
    flat = sampled[..., 0].reshape(count, size * size)
    exact = np.linalg.inv(dense_precision(size, r))
    np.testing.assert_allclose(np.cov(flat, rowvar=False, ddof=0), exact, atol=.012, rtol=.035)
    mean_fields = flat.mean(axis=1)
    assert np.var(mean_fields) == pytest.approx(1 / (size * size * r), rel=.025)
    diagnostics = data.gaussian_diagnostics(sampled.reshape(4, count // 4, size, size, 1), r)
    assert abs(diagnostics["mean_field"]["mean_z_score"]) < 4
    assert abs(diagnostics["mean_field"]["second_moment_z_score"]) < 4
    action = diagnostics["twice_action_chi_squared"]
    assert action["degrees_of_freedom"] == 9
    assert abs(action["mean_z_score"]) < 4
    assert action["empirical_variance_ddof1"] == pytest.approx(18., rel=.04)
    np.testing.assert_allclose(diagnostics["empirical_structure_factor"],
                               diagnostics["exact_structure_factor"], rtol=.03, atol=.003)


def test_seed_streams_are_reproducible_distinct_and_keyed_by_r_not_grid_position():
    current = options()
    seeds = [data.group_seed(current, 3, cell, group)
             for cell in range(3) for group in range(4)]
    assert len(set(seeds)) == 12
    assert seeds == [data.group_seed(current, 3, cell, group)
                     for cell in range(3) for group in range(4)]
    reordered = replace(current, r_values=(.8, .1, .3))
    assert data.group_seed(current, 3, 0, 1) == data.group_seed(reordered, 3, 1, 1)
    assert data.group_seed(current, 4, 0, 1) != data.group_seed(current, 3, 0, 1)
    a = data.sample_gaussian_fields(.3, size=3, samples=8, seed=seeds[0])
    b = data.sample_gaussian_fields(.3, size=3, samples=8, seed=seeds[0])
    c = data.sample_gaussian_fields(.3, size=3, samples=8, seed=seeds[1])
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_default_plan_is_small_and_fixed_without_randomness_or_filesystem(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Planning must not allocate spectra or sample.")
    monkeypatch.setattr(data, "lattice_spectrum", forbidden)
    monkeypatch.setattr(data, "sample_gaussian_fields", forbidden)
    a = data.generation_plan(data.GaussianScalarOptions())
    b = data.generation_plan(data.GaussianScalarOptions())
    assert a == b
    assert a["schema"] == "euclidean_gaussian_fft_v1"
    assert a["options"]["sizes"] == (36,)
    assert a["options"]["chains"] == a["options"]["groups"] == 4
    assert a["options"]["samples"] == 128
    np.testing.assert_array_equal(a["options"]["r_values"], np.geomspace(.02, 2., 12))
    assert a["test_cell_indices"] == [1, 4, 7, 10]
    assert a["configurations"] == 12 * 4 * 128
    assert a["projected_output_bytes"] < 1024 ** 3
    assert a["projected_working_bytes"] < 256 * 1024 ** 2
    assert a["cpu_threads_requested"] == 1
    assert not list(tmp_path.iterdir())
    json.dumps(a, allow_nan=False)


@pytest.mark.parametrize("changes", [
    {"sizes": ()}, {"sizes": (2,)}, {"sizes": (3, 3)}, {"dimension": 1},
    {"r_values": (0., .2)}, {"r_values": (-.1, .2)}, {"r_values": (float("nan"), .2)},
    {"r_values": (.2, .2)}, {"r_values": (.2,)}, {"groups": 0}, {"groups": True},
    {"samples": 3}, {"samples": 5}, {"seed": -1}, {"test_every": 1},
    {"test_offset": 3}, {"max_output_bytes": 1024 ** 3 + 1},
    {"max_output_bytes": 1}, {"max_memory_bytes": 1},
])
def test_bad_parameters_and_budget_fail_before_creating_or_sampling(tmp_path, monkeypatch, changes):
    def forbidden(*args, **kwargs):
        raise AssertionError("Rejected plan must not sample")
    monkeypatch.setattr(data, "sample_gaussian_fields", forbidden)
    target = tmp_path / "not_created"
    with pytest.raises(ValueError):
        data.generate_dataset(target, options(**changes))
    assert not target.exists()


def test_generated_shards_separate_public_references_from_configurations(tmp_path):
    root = tmp_path / "new"
    manifest = data.generate_dataset(root, options())
    assert manifest["status"] == "complete"
    assert manifest["sample_semantics"] == data.SAMPLE_SEMANTICS
    assert manifest["options"]["chains_field_semantics"] == data.SAMPLE_SEMANTICS
    assert len(manifest["shards"]) == 3
    for cell, shard in enumerate(manifest["shards"]):
        assert shard["cell_index"] == cell
        assert shard["split"] == ("test" if cell == 1 else "train")
        with np.load(root / shard["path"], allow_pickle=False) as archive:
            assert archive.files == ["configurations"]
            assert archive["configurations"].shape == (4, 8, 3, 3, 1)
            assert archive["configurations"].dtype == np.float64
        assert data.sha256_file(root / shard["path"]) == shard["sha256"]
        sidecar_path = root / shard["sidecar"]
        assert data.sha256_file(sidecar_path) == shard["sidecar_sha256"]
        sidecar = json.loads(sidecar_path.read_text())
        assert sidecar["data_sha256"] == shard["sha256"]
        assert sidecar["u"] == sidecar["h"] == 0.
        diagnostic_path = sidecar_path.parent / sidecar["diagnostics"]
        assert data.sha256_file(diagnostic_path) == sidecar["diagnostics_sha256"]
        diagnostics = json.loads(diagnostic_path.read_text())
        assert diagnostics["independent_draws"] == 32
        assert diagnostics["mean_field"]["exact_variance"] == pytest.approx(1 / (9 * sidecar["r"]))
        assert diagnostics["twice_action_chi_squared"]["degrees_of_freedom"] == 9
        assert str(diagnostic_path.relative_to(root)) in manifest["public_files"]
    assert all(Path(path).suffix != ".npz" for path in manifest["public_files"])
    status = json.loads((root / "status.json").read_text())
    assert status["workflow_complete"] is True and status["returncode"] == 0
    assert status["cpu_seconds"] >= 0 and status["wall_seconds"] >= 0
    json.dumps(manifest, allow_nan=False)


def test_never_overwrites_existing_directory_or_resumes_partial_dataset(tmp_path):
    target = tmp_path / "existing"
    target.mkdir()
    sentinel = target / "user.txt"
    sentinel.write_text("preserve")
    before = hashlib.sha256(sentinel.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        data.generate_dataset(target, options())
    assert sentinel.read_text() == "preserve"
    assert hashlib.sha256(sentinel.read_bytes()).hexdigest() == before
    assert sorted(path.name for path in target.iterdir()) == ["user.txt"]


def test_loader_compatibility_and_no_label_or_diagnostic_member_read(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from experiments.phi4_qcr import load_dataset
    root = tmp_path / "size9"
    data.generate_dataset(root, options(sizes=(9,)))
    original_load = np.load
    accessed = []

    class TrackedArchive:
        def __init__(self, archive):
            self.archive = archive
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.archive.close()
        def __getitem__(self, key):
            accessed.append(key)
            return self.archive[key]

    monkeypatch.setattr(np, "load", lambda *args, **kwargs: TrackedArchive(original_load(*args, **kwargs)))
    loaded = load_dataset(root)
    assert loaded.configurations.shape == (12, 8, 9, 9, 1)
    assert loaded.configurations.dtype == torch.float64
    assert set(loaded.train_indices).isdisjoint(set(loaded.test_indices))
    assert len(loaded.train_indices) == 8 and len(loaded.test_indices) == 4
    assert loaded.scale > 0
    assert accessed == ["configurations"] * 3


def test_cli_defaults_to_plan_only_even_with_output_argument(tmp_path):
    from scripts.prepare_gaussian_scalar import main
    root = tmp_path / "not_created"
    result = main(["--output", str(root)])
    assert result["training_updates"] == 0
    assert not root.exists()


def test_cli_execute_requires_output():
    from scripts.prepare_gaussian_scalar import main
    with pytest.raises(SystemExit):
        main(["--execute"])
