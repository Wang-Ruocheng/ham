"""Receipt/admission contract tests, not a physical or model experiment."""
import json
from pathlib import Path

import pytest

from scripts import run_ising3d_no_rsmi as runner


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def baseline(tmp_path, monkeypatch):
    source = tmp_path / "source"
    root = tmp_path / "baseline"
    original = Path(__file__).resolve().parents[1]
    config = json.loads((original / "configs/ising3d_b2_g1_seed91.json").read_text())
    cache = root / "data/ensembles.pt"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"fixture only, never loaded as torch data")
    fingerprint = runner.sha(cache)
    monkeypatch.setattr(runner, "DATA_SHA256", fingerprint)
    config["data"]["cache"] = str(cache)
    fingerprints = {}
    for relative in runner.MODEL_FILES:
        file = source / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("# immutable model fixture\n")
        fingerprints[relative] = runner.sha(file)
    initial = {key: {module: module + "-initial" for module in ("Q", "C", "R")}
               for key in runner.INITIAL_KEYS}
    manifest = dict(config=config, rsmi_weight=2., independent_c=True, labels_used_in_training=False,
                    execution_device="cuda:0", data_sha256=fingerprint, **initial)
    dump(root / "job.json", dict(status="complete", workflow_complete=True,
         stages=[dict(name="train", returncode=0), dict(name="report", returncode=0)]))
    dump(root / "plan.json", dict(source_fingerprints=fingerprints))
    dump(root / "run/manifest.json", manifest)
    dump(root / "run/summary.json", dict(actual_epochs=6000))
    dump(root / "report/report.json", dict(status="complete"))
    dump(root / "config.json", config)
    dump(root / "data/evaluation_metadata.json", dict(cache_sha256=fingerprint, lattice_shape=[16, 16, 16]))
    return root, source, initial


def test_pairing_is_fresh_model_same_cache_and_objectives_except_mi(baseline):
    root, source, initial = baseline
    config, plan = runner.paired_plan(root, source)
    assert config == runner.read(root / "config.json")
    assert plan["config_changes"] == []
    assert plan["initial_module_sha256"] == initial
    assert plan["fixed_training_updates"] == 6000
    assert plan["data_reused_read_only"] and plan["fresh_random_model"]
    assert not any(plan[key] for key in ("physical_sampling", "MI_computed", "MI_cache_read",
                   "critic_created", "baseline_checkpoint_loaded", "optimizer_warm_start"))


def test_pairing_rejects_incomplete_baseline_and_changed_science(baseline):
    root, source, _ = baseline
    dump(root / "job.json", dict(status="running", workflow_complete=False, stages=[]))
    with pytest.raises(ValueError, match="completed"):
        runner.paired_plan(root, source)
    dump(root / "job.json", dict(status="complete", workflow_complete=True, stages=[]))
    (source / runner.MODEL_FILES[0]).write_text("# different model\n")
    with pytest.raises(ValueError, match="source changed"):
        runner.paired_plan(root, source)


def test_commands_explicitly_authorize_zero_mi_without_sampling(tmp_path):
    dump(tmp_path / "plan.json", dict(baseline="/completed/baseline"))
    stages = runner.commands(tmp_path)
    assert [name for name, _, _ in stages] == ["gpu_preflight", "train", "report", "comparison"]
    for name, command, gpu in stages[:2]:
        assert gpu and "--rsmi-weight" in command and command[command.index("--rsmi-weight") + 1] == "0"
        assert "--allow-nondefault-recipe" in command and "--skip-final-report" in command
        assert "--mi-cache-dir" not in command
    assert "--epochs" in stages[0][1] and "--epochs" not in stages[1][1]
    assert "--skip-mi-diagnostic" in stages[2][1]
    assert stages[3][2] is False


def test_training_receipt_enforces_all_initial_module_hashes(tmp_path, monkeypatch):
    initial = {key: {module: module + "-initial" for module in ("Q", "C", "R")}
               for key in runner.INITIAL_KEYS}
    monkeypatch.setattr(runner, "DATA_SHA256", "fixture-data")
    dump(tmp_path / "plan.json", dict(initial_module_sha256=initial))
    manifest = dict(rsmi_weight=0., rsmi_enabled=False, estimator=None, mi_evaluated=False, critic_parameters=0,
                    data_sha256="fixture-data", independent_c=True, labels_used_in_training=False, **initial)
    dump(tmp_path / "gpu_preflight/manifest.json", manifest)
    dump(tmp_path / "gpu_preflight/summary.json", dict(actual_epochs=5))
    assert runner.validate_training_receipt(tmp_path, "gpu_preflight", 5)["actual_epochs"] == 5
    manifest[runner.INITIAL_KEYS[-1]]["C"] = "wrong initial C"
    dump(tmp_path / "gpu_preflight/manifest.json", manifest)
    with pytest.raises(RuntimeError, match="fingerprints"):
        runner.validate_training_receipt(tmp_path, "gpu_preflight", 5)


def test_existing_output_and_launch_receipt_never_overwritten(tmp_path, monkeypatch):
    with pytest.raises(FileExistsError):
        runner.prepare(tmp_path, tmp_path / "baseline")
    monkeypatch.setattr(runner.socket, "gethostname", lambda: "thinklab-105-225")
    dump(tmp_path / "job.json", dict(status="prepared"))
    dump(tmp_path / "logs/launch.json", dict(parent_pid=42))
    with pytest.raises(FileExistsError, match="Existing launch"):
        runner.launch(tmp_path)
