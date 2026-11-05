"""KB launch contracts using JSON only: no physical tensors or remote access."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.rsmi_separate_qcr import build_parser, validate_recipe
from recurrent_rg.config import RunConfig, ScaleConfig
from recurrent_rg.learning_rate import LearningRatePolicy
from scripts import run_ising_kb_highdim as runner


@pytest.fixture(autouse=True)
def forbid_tensor_loading(monkeypatch):
    import torch
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail(
        "Runner contract tests must never load a physical or model tensor"))


def normalized(value):
    return json.loads(json.dumps(value))


@pytest.fixture
def audit():
    """Complete lightweight JSON receipt; no configurations are constructed."""
    k_values = [round(.2 + .03 * index, 2) for index in range(17)]
    b_values = [round(-.01 + .001 * index, 3) for index in range(21)]
    statistic = dict(mean=0., se=.01, lag1=0., ess_estimate=64.)
    rows = [dict(K=k, B=b, replica=replica,
                 seed=9100000 + 1000 * i + 2 * j + replica,
                 test=(i + abs(j - 10)) % 3 == 1,
                 staggered_m2=.001, bond_per_site=dict(statistic),
                 m=dict(statistic), m2=dict(statistic))
            for i, k in enumerate(k_values)
            for j, b in enumerate(b_values)
            for replica in (0, 1)]
    return dict(K=k_values, B=b_values, grid="positive-uniform-B", sites=36,
        draws=64, replicas=2, seed=91, equilibration_sweeps=512, sweeps_between=16,
        train_shape=[474, 64, 36, 36, 1], test_shape=[240, 64, 36, 36, 1],
        sampler="SW at K>=0; checkerboard gauge SW with staggered cluster field at K<0; output physical spins",
        split="(abs(K_index-0)+abs(B_index-10))%3==1; entire sign-related cells and replicas kept in same split",
        elapsed_seconds=28., ensembles=rows)


def test_validates_complete_grid_receipt_without_changing_input(audit):
    before = deepcopy(audit)
    summary = runner.validate_audit(audit)
    assert isinstance(summary, dict)
    assert summary
    assert "ensembles" not in summary
    json.dumps(summary, allow_nan=False)
    assert audit == before


@pytest.mark.parametrize("key,value", [
    ("sites", 32), ("draws", 128), ("replicas", 1), ("seed", 92),
    ("train_shape", [473, 64, 36, 36, 1]),
    ("test_shape", [240, 64, 36, 36, 2]),
])
def test_audit_rejects_wrong_physical_sampling_identity(audit, key, value):
    audit[key] = value
    with pytest.raises(ValueError):
        runner.validate_audit(audit)


@pytest.mark.parametrize("mutation", ["missing_cell", "duplicate_cell", "replica_id",
    "duplicate_seed", "wrong_seed", "split_replica_leak", "unknown_coupling", "untyped_split"])
def test_audit_rejects_incomplete_or_leaking_chain_receipts(audit, mutation):
    rows = audit["ensembles"]
    if mutation == "missing_cell":
        rows.pop()
    elif mutation == "duplicate_cell":
        rows[-1] = deepcopy(rows[0])
    elif mutation == "replica_id":
        rows[1]["replica"] = rows[0]["replica"]
    elif mutation == "duplicate_seed":
        rows[-1]["seed"] = rows[0]["seed"]
    elif mutation == "wrong_seed":
        rows[-1]["seed"] += 1000000
    elif mutation == "split_replica_leak":
        rows[0]["test"] = not rows[1]["test"]
    elif mutation == "unknown_coupling":
        rows[0]["B"] = .5
    else:
        rows[0]["test"] = "true"
    with pytest.raises(ValueError):
        runner.validate_audit(audit)


@pytest.mark.parametrize("axis", ["K", "B"])
def test_audit_rejects_wrong_grid_axes_even_when_grid_lengths_match(audit, axis):
    audit[axis][0] += .00001
    with pytest.raises(ValueError):
        runner.validate_audit(audit)


def test_audit_rejects_sign_asymmetric_split_even_with_preserved_counts(audit):
    # Swap two whole cells between splits: counts and per-cell replica grouping
    # remain correct, while the -B/+B partners no longer share the same split.
    for row in audit["ensembles"]:
        if row["K"] == .2 and row["B"] in (-.01, -.009):
            row["test"] = not row["test"]
    assert sum(row["test"] for row in audit["ensembles"]) == 240
    with pytest.raises(ValueError):
        runner.validate_audit(audit)


def test_audit_rejects_different_split_even_when_counts_and_sign_grouping_still_match(audit):
    # Both complete +/-B pairs move together, preserving chain counts and all
    # symmetry grouping while violating the recorded deterministic split rule.
    for row in audit["ensembles"]:
        if row["K"] == .2 and abs(row["B"]) in (.01, .009):
            row["test"] = not row["test"]
    assert sum(row["test"] for row in audit["ensembles"]) == 240
    with pytest.raises(ValueError):
        runner.validate_audit(audit)


@pytest.mark.parametrize("dimension", [5, 10])
def test_new_arms_use_field_sensitive_default_qcr_without_reading_cache(tmp_path, dimension):
    cache = tmp_path / "nonexistent-physical-cache.pt"
    config = runner.build_config(cache, dimension)
    parsed = RunConfig.from_dict(config)
    parsed.validate()
    assert normalized(config) == normalized(parsed.to_dict())
    args = build_parser().parse_args(["--config", "not-loaded.json", "--output", "not-created"])
    validate_recipe(args, parsed)

    assert config["schema_version"] == 5
    assert config["data"]["cache"] == str(cache)
    assert config["data"]["physical_model"] == "ising2d"
    assert list(config["data"]["coupling_names"]) == ["K", "B"]
    assert config["data"]["spin_flip_augment"] is False
    assert config["data"]["spin_flip_symmetric"] is False
    assert config["data"]["train_pool_replicas"] == 1
    assert config["data"]["test_pool_replicas"] == 1
    expected_model = dict(dimension=2, theory_dim=dimension, hidden=96, width=3,
        block_factor=3, kernel_layout="nonoverlapping", categorical_states=0,
        kernel_sampling="bernoulli", kernel_initialization="random",
        kernel_temperature=1., fixed_kernel=False, binary_lookup=True,
        affine_reconstruction=True, reconstruction_hidden=0)
    assert all(config["model"][key] == value for key, value in expected_model.items())

    loss = config["loss"]
    assert loss["closure_scale_weights"] is None
    assert loss["coordinate_reduction"] == "sum"
    assert loss["closure_metric"] == "euclidean"
    assert loss["kernel_noncollapse_mode"] == "variance_floor_ensemble"
    assert loss["kernel_variance_ratio"] == 1.
    assert loss["kernel_sign_weight"] == 1.
    assert loss["kernel_neutral_weight"] == 0.
    assert loss["rollout_weight"] == 5.
    assert loss["weights"] == dict(anchor=2., coarse_consistency=5.,
        separation=10., commutator=5., kernel_noncollapse=2.)
    for name in ("isotropy_weight", "tangent_weight", "orientation_weight",
                 "balanced_reconstruction_weight"):
        assert loss[name] == 0.

    training = config["training"]
    assert training["epochs"] == 12000
    assert training["max_rg_steps"] == 2
    assert (training["system_batch"], training["sample_batch"]) == (16, 64)
    assert training["seed"] == 91
    assert training["learning_rate"] == .001
    assert training["log_every"] == 200
    assert training["checkpoint_every"] == 1000
    assert training["gradient_mode"] == "qc_r_separate"
    assert list(training["train_modules"]) == ["Q", "C", "R"]
    assert training["lr_schedule"] == "multistep"
    assert list(training["lr_milestones"]) == [6000, 9000]
    assert training["lr_gamma"] == .3
    assert ScaleConfig((36, 36), training["max_rg_steps"],
                       config["model"]["block_factor"]).shapes() == [(36, 36), (12, 12), (4, 4)]
    assert not cache.exists()


def test_multistep_changes_after_completed_milestones(tmp_path):
    cfg = RunConfig.from_dict(runner.build_config(tmp_path / "unread.pt", 5))
    policy = LearningRatePolicy(cfg.training)
    assert [policy.rate_for_update(step) for step in (1, 6000, 6001, 9000, 9001, 12000)] == pytest.approx(
        [.001, .001, .0003, .0003, .00009, .00009])


@pytest.mark.parametrize("dimension", [0, 1, 2, 4, 6, 100, True, 5., "5", None])
def test_build_config_rejects_dimensions_outside_exact_authorized_pair(tmp_path, dimension):
    with pytest.raises(ValueError):
        runner.build_config(tmp_path / "unread.pt", dimension)


@pytest.mark.parametrize("dimension", [5, 10])
def test_smoke_uses_two_updates_with_same_scientific_recipe(tmp_path, dimension):
    cache = tmp_path / "unread.pt"
    formal = normalized(runner.build_config(cache, dimension))
    smoke = normalized(runner.build_config(cache, dimension, smoke=True))
    RunConfig.from_dict(smoke).validate()
    assert smoke["training"]["epochs"] == 2
    assert smoke["data"] == formal["data"]
    assert smoke["model"] == formal["model"]
    assert smoke["loss"] == formal["loss"]
    formal_training, smoke_training = deepcopy(formal["training"]), deepcopy(smoke["training"])
    for name in ("epochs", "log_every", "checkpoint_every"):
        formal_training.pop(name)
        smoke_training.pop(name)
    assert smoke_training == formal_training


def test_pair_comparison_accepts_only_output_dimension_and_does_not_mutate_inputs(tmp_path):
    left = runner.build_config(tmp_path / "unread.pt", 5)
    right = runner.build_config(tmp_path / "unread.pt", 10)
    before = deepcopy((left, right))
    assert runner.compare_configs(left, right) == ["model.theory_dim"]
    assert (left, right) == before


@pytest.mark.parametrize("section,key,value", [
    ("data", "spin_flip_augment", True),
    ("data", "cache", "/different-cache.pt"),
    ("model", "hidden", 48),
    ("model", "block_factor", 2),
    ("loss", "closure_scale_weights", [1.8, .2]),
    ("training", "seed", 92),
    ("training", "system_batch", 8),
    ("training", "epochs", 11999),
    ("training", "lr_gamma", .5),
])
def test_pair_comparison_rejects_other_differences(tmp_path, section, key, value):
    left = runner.build_config(tmp_path / "unread.pt", 5)
    right = runner.build_config(tmp_path / "unread.pt", 10)
    right[section][key] = value
    with pytest.raises(ValueError):
        runner.compare_configs(left, right)


def test_pair_comparison_rejects_identical_arms(tmp_path):
    config = runner.build_config(tmp_path / "unread.pt", 5)
    with pytest.raises(ValueError):
        runner.compare_configs(config, deepcopy(config))


@pytest.mark.parametrize("returncode,elapsed,expected", [
    (124, 900., True), (124, 901., True), (124, 899.999, True),
    (137, 900., True), (137, 1000., True), (137, 899.999, False),
    (0, 1000., False), (1, 1000., False), (2, 1000., False),
    (-9, 1000., False), (-15, 1000., False), (130, 1000., False), (75, 1000., False),
])
def test_only_expected_report_timeout_exit_codes_and_elapsed_time_are_nonfatal(
        returncode, elapsed, expected):
    assert runner.report_timeout(returncode, elapsed) is expected


def test_native_gpu_environment_sets_order_in_child_copy_only(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
    environment = runner.native_cuda_environment()
    assert environment["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert "CUDA_VISIBLE_DEVICES" not in environment
    assert __import__("os").environ["CUDA_DEVICE_ORDER"] == "FASTEST_FIRST"


@pytest.mark.parametrize("mask", ["", "0", "2", "0,1,2", runner.GPU_UUID])
def test_inherited_gpu_mask_is_rejected_even_when_empty(monkeypatch, mask):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES"):
        runner.native_cuda_environment()


@pytest.mark.parametrize("torch_uuid", [runner.GPU_UUID, runner.GPU_UUID.removeprefix("GPU-").upper()])
def test_metadata_verification_binds_torch_ordinal_to_physical_uuid_without_allocations(
        tmp_path, monkeypatch, torch_uuid):
    import torch
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    inspected = []

    def properties(device):
        inspected.append(device)
        return SimpleNamespace(uuid=torch_uuid, name="mock selected GPU")

    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    monkeypatch.setattr(torch, "empty", lambda *a, **k: pytest.fail("Metadata verification allocated a tensor"))
    result = runner.verify_cuda_device(tmp_path)
    assert inspected == [runner.GPU]
    assert result["status"] == "complete"
    assert result["device"] == "cuda:2"
    assert result["gpu_uuid"] == runner.GPU_UUID
    assert result["torch_uuid"] == torch_uuid
    assert result["metadata_only"] is True
    assert result["training_updates"] == result["tensor_allocations"] == 0
    assert runner.read(tmp_path / "cuda_verification.json") == result


@pytest.mark.parametrize("properties", [SimpleNamespace(name="missing UUID"),
    SimpleNamespace(uuid="GPU-different", name="wrong GPU")])
def test_device_verification_rejects_missing_or_wrong_uuid_without_publishing_receipt(
        tmp_path, monkeypatch, properties):
    import torch
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: properties)
    with pytest.raises(RuntimeError, match="UUID"):
        runner.verify_cuda_device(tmp_path)
    assert not (tmp_path / "cuda_verification.json").exists()


def test_device_verification_requires_order_before_any_cuda_inspection(tmp_path, monkeypatch):
    import torch
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *a, **k: pytest.fail(
        "CUDA metadata was inspected before checking its ordering"))
    with pytest.raises(RuntimeError, match="ordering"):
        runner.verify_cuda_device(tmp_path)


def completed_training(root, *, smoke=False):
    """Small JSON and fake endpoint bytes; the endpoint is only statted/hashed."""
    config = normalized(runner.build_config(root / "unread-physical.pt", 5, smoke=smoke))
    steps = config["training"]["epochs"]
    run = root / "smoke/dim5/run" if smoke else root / "dim5/run"
    run.mkdir(parents=True)
    configs = root / "configs"
    configs.mkdir(exist_ok=True)
    config_path = configs / ("smoke_dim5.json" if smoke else "dim5.json")
    runner.save(config_path, config)
    policy = LearningRatePolicy(RunConfig.from_dict(config).training)
    log_every = config["training"]["log_every"]
    epochs = sorted({1, steps, *range(log_every, steps + 1, log_every)})
    history = [dict(epoch=epoch, learning_rate=policy.rate_for_update(epoch), total=1.) for epoch in epochs]
    manifest = dict(config=config, training_driver="rsmi_separate_qcr", precision="torch.float64",
        estimator="histogram", environment_mode="full_shell", buffer=1, independent_c=True,
        labels_used_in_training=False, rsmi_weight=2., execution_device=runner.GPU, source=None,
        data_sha256=runner.DATA_SHA, execution_optimizations=dict(
            low_ram_execution=dict(enabled=True, source_dtype_unchanged=True),
            mi_statistics_cache={split: dict(hit=True, cache_version=1) for split in ("train", "test")}))
    summary = dict(config=config, actual_epochs=steps, requested_epochs=steps,
        final_report_skipped=True, stopped_by_wall_limit=False, stopped_by_training_limit=False,
        history=history, lr_scheduler_state=policy.checkpoint_state(steps), timing=dict(
            enabled=True, updates=steps, cuda_max_memory_allocated_bytes=2 * 1024**3,
            cuda_max_memory_reserved_bytes=3 * 1024**3))
    runner.save(run / "manifest.json", manifest)
    runner.save(run / "summary.json", summary)
    (run / f"training_step_{steps}.pt").write_bytes(b"fake endpoint; never tensor-loaded")
    arm = dict(name="dim5", run=str(run), config_path=str(config_path), smoke_config_path=str(config_path))
    return arm, run


@pytest.mark.parametrize("smoke", [False, True])
def test_training_validation_accepts_only_complete_profiled_endpoint(tmp_path, smoke):
    arm, _ = completed_training(tmp_path, smoke=smoke)
    result = runner.validate_training(tmp_path, {}, arm, smoke=smoke)
    assert result["status"] == "complete"
    assert result["actual_updates"] == (2 if smoke else 12000)
    assert result["labels_used_in_training"] is False


@pytest.mark.parametrize("key,value", [
    ("enabled", False), ("updates", 11999),
    ("cuda_max_memory_allocated_bytes", 14 * 1024**3 + 1),
    ("cuda_max_memory_reserved_bytes", 14 * 1024**3 + 1),
    ("cuda_max_memory_allocated_bytes", 0),
    ("cuda_max_memory_reserved_bytes", None),
    ("cuda_max_memory_allocated_bytes", 2.5 * 1024**3),
])
def test_training_validation_rejects_missing_or_excess_allocator_lifetime_profile(tmp_path, key, value):
    arm, run = completed_training(tmp_path)
    summary = runner.read(run / "summary.json")
    summary["timing"][key] = value
    runner.save(run / "summary.json", summary)
    with pytest.raises(RuntimeError):
        runner.validate_training(tmp_path, {}, arm, smoke=False)


@pytest.mark.parametrize("key", ["enabled", "source_dtype_unchanged"])
def test_training_validation_rejects_disabled_low_ram_policy(tmp_path, key):
    arm, run = completed_training(tmp_path)
    manifest = runner.read(run / "manifest.json")
    manifest["execution_optimizations"]["low_ram_execution"][key] = False
    runner.save(run / "manifest.json", manifest)
    with pytest.raises(RuntimeError):
        runner.validate_training(tmp_path, {}, arm, smoke=False)


def test_training_validation_rejects_incorrect_logged_learning_rate_even_with_correct_scheduler_state(tmp_path):
    arm, run = completed_training(tmp_path)
    summary = runner.read(run / "summary.json")
    summary["history"][-1]["learning_rate"] = .001
    runner.save(run / "summary.json", summary)
    with pytest.raises(RuntimeError):
        runner.validate_training(tmp_path, {}, arm, smoke=False)
