"""2D spin-glass experiment recipe, preflight and read-only lifecycle regressions."""
import json
from pathlib import Path
import shutil

import pytest

from experiments.rsmi_separate_qcr import build_parser, validate_recipe
from recurrent_rg.config import RunConfig
from scripts import run_spin_glass2d as runner


REPO = Path(__file__).resolve().parents[1]


def config_for(root, smoke=False):
    config = json.loads((REPO / runner.CONFIG_PATH).read_text())
    config["data"]["cache"] = str(root / "data/ensembles.pt")
    if smoke:
        config["training"].update(epochs=2, max_rg_steps=1, sample_batch=8, system_batch=4)
        config["evaluation"].update(max_rg_steps=1, samples=8, systems=4)
    return config


def test_2d_config_uses_complete_locked_uniform_recipe(tmp_path):
    raw = config_for(tmp_path)
    cfg = RunConfig.from_dict(raw)
    cfg.validate()
    args = build_parser().parse_args(["--config", "config.json", "--output", "run"])
    validate_recipe(args, cfg)
    assert cfg.schema_version == 5 and cfg.model.dimension == 2 and cfg.model.theory_dim == 1
    assert cfg.model.block_factor == 3 and cfg.model.kernel_layout == "nonoverlapping"
    assert cfg.model.hidden == 96 and cfg.model.width == 3
    assert cfg.model.binary_lookup and not cfg.model.fixed_kernel
    assert cfg.model.kernel_initialization == "random" and cfg.model.kernel_sampling == "bernoulli"
    assert cfg.model.affine_reconstruction and cfg.model.reconstruction_hidden == 0
    assert not cfg.data.coupling_names and not cfg.data.spin_flip_augment
    assert cfg.loss.closure_scale_weights is None
    assert cfg.loss.weights.anchor * 10 == 20 and cfg.loss.weights.coarse_consistency == 5
    assert cfg.loss.weights.separation == 10 and cfg.loss.weights.commutator == 5
    assert cfg.loss.rollout_weight == 5 and cfg.loss.weights.kernel_noncollapse == 2
    assert cfg.loss.kernel_noncollapse_mode == "variance_floor_ensemble"
    assert cfg.loss.kernel_variance_ratio == 1 and cfg.loss.kernel_sign_weight == 1
    assert (args.rsmi_weight, args.variance_weight, args.color_weight) == (2., 2., 1.)
    assert args.estimator == "histogram" and args.environment == "full_shell"
    assert args.independent_c and args.variance_per_ensemble and not args.mi_initialize
    assert args.q_coupling_weight == 0 and args.q_future_weight == 0
    assert cfg.training.max_rg_steps == 2 and cfg.training.epochs == 6000
    assert cfg.training.system_batch == 16 and cfg.training.sample_batch == 64
    assert cfg.training.learning_rate == .001 and cfg.training.seed == 91
    assert cfg.training.threads == 1 and cfg.training.gradient_mode == "qc_r_separate"
    assert set(cfg.training.train_modules) == {"Q", "C", "R"}


def test_physical_preflight_and_exact_stage_commands(tmp_path):
    plan = runner.build_plan(tmp_path, config_for(tmp_path), {}, False)
    assert plan["physical"]["shape"] == [36, 36]
    assert len(plan["physical"]["temperatures"]) == 20
    assert plan["physical"]["temperatures"][0] == .5
    assert plan["physical"]["temperatures"][-1] == 3.5
    assert plan["physical"]["burn_in"] == 2048
    assert plan["physical_preflight"]["total_single_spin_visits"] == 1_433_272_320
    assert plan["physical_preflight"]["total_single_spin_visits"] < runner.MAX_SPIN_VISITS
    assert plan["physical_preflight"]["test_disorders"] == [1, 4, 7]
    assert plan["physical_preflight"]["projected_output_bytes"] < runner.MAX_OUTPUT_BYTES
    assert plan["physical_preflight"]["projected_working_bytes"] < 512 * 1024**2
    stages = runner.build_stages(tmp_path, "cuda:2", plan)
    assert [row[0] for row in stages] == ["physical_sampling", "prepare_overlap", "train", "report"]
    sampling = stages[0][1]
    assert sampling[sampling.index("--dimension") + 1] == "2"
    assert sampling[sampling.index("--max-spin-proposals") + 1] == "1500000000"
    training = stages[2][1]
    assert training[training.index("--max-training-seconds") + 1] == "1800"
    assert "experiments.rsmi_separate_qcr" in training
    assert not any("allow-" in flag or "resume" in flag for _, command, _ in stages for flag in command)
    assert "scripts.report_spin_glass2d" in stages[-1][1]
    assert plan["scientific_result"] and not plan["execution_smoke"]
    assert plan["budget"]["max_gpu_stage_seconds"] == 2500
    assert plan["budget"]["CPU_core_seconds"] == 7200


def test_smoke_geometry_and_non_scientific_qualification(tmp_path):
    config = config_for(tmp_path, True)
    plan = runner.build_plan(tmp_path, config, {}, True)
    assert plan["physical"]["shape"] == [12, 12]
    assert plan["physical"]["disorders"] == 3 and plan["physical"]["samples"] == 8
    assert plan["physical"]["temperatures"] == [1., 3.]
    assert config["training"]["max_rg_steps"] == 1 and config["training"]["epochs"] == 2
    assert plan["execution_smoke"] and not plan["scientific_result"]
    assert plan["physical_preflight"]["total_single_spin_visits"] == 27648


def test_invalid_extra_scale_rejected_before_sampling(tmp_path):
    config = config_for(tmp_path)
    config["training"]["max_rg_steps"] = 3
    with pytest.raises(ValueError, match="divisible"):
        runner.build_plan(tmp_path, config, {})


def test_formal_cpu_rejected_without_creating_root(tmp_path):
    target = tmp_path / "formal"
    with pytest.raises(ValueError, match="CPU"):
        runner.run(target, "cpu")
    assert not target.exists()


def test_cuda_wrong_host_rejected_before_sampling(monkeypatch, tmp_path):
    monkeypatch.setattr(runner.socket, "gethostname", lambda: "wrong-host")
    target = tmp_path / "wrong_host"
    with pytest.raises(ValueError, match="pinned"):
        runner.run(target, "cuda:2")
    assert not target.exists()


def test_two_gpu_checks_are_separated_and_failed_check_resets(monkeypatch, tmp_path):
    snapshots = iter([True, False, True, True])
    sleeps = []
    monkeypatch.setattr(runner, "gpu_snapshot", lambda: {"ready": next(snapshots)})
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: sleeps.append(seconds))
    job = {}
    runner.resource_gate(tmp_path, job)
    assert len(job["resource_checks"]) == 4 and sleeps == [10, 10, 10]


def test_freeze_includes_2d_config_and_python_dependencies(tmp_path):
    source, code = tmp_path / "source", tmp_path / "code"
    for package in ("recurrent_rg", "experiments", "scripts"):
        (source / package).mkdir(parents=True)
        (source / package / "fixture.py").write_text("FIXTURE = True\n")
    (source / "AGENTS.md").write_text("fixture\n")
    (source / "configs").mkdir()
    shutil.copy2(REPO / runner.CONFIG_PATH, source / runner.CONFIG_PATH)
    fingerprints = runner.freeze_code(source, code)
    assert runner.CONFIG_PATH in fingerprints and "scripts/fixture.py" in fingerprints
    assert (code / "fingerprints.json").exists()
    runner.verify_code(code, fingerprints)
    (code / "scripts/fixture.py").write_text("FIXTURE = False\n")
    with pytest.raises(RuntimeError, match="fingerprint"):
        runner.verify_code(code, fingerprints)


def mock_lifecycle(monkeypatch, actual_epochs=2, failed_stage=None):
    """No physical sampling, model inference or training is run in these tests."""
    calls = []

    def freeze(source, target):
        (target / "configs").mkdir(parents=True)
        shutil.copy2(source / runner.CONFIG_PATH, target / runner.CONFIG_PATH)
        return {runner.CONFIG_PATH: runner.sha(target / runner.CONFIG_PATH)}

    class Process:
        pid = 12345

        def __init__(self, command, cwd, stdout, stderr, env):
            self.command = command
            calls.append(command)
            self.root = Path(cwd).parent
            if "experiments.rsmi_separate_qcr" in command:
                (self.root / "run").mkdir()
                runner.write_json(self.root / "run/summary.json", {"actual_epochs": actual_epochs})
            if "scripts.report_spin_glass2d" in command:
                (self.root / "report").mkdir()
                runner.write_json(self.root / "report/report.json", {"status": "complete"})
                (self.root / "report/encodings.npz").write_bytes(b"safe public array fixture")
                (self.root / "report/private.npz").write_bytes(b"must not be listed")
                (self.root / "report/model.pt").write_bytes(b"must not be listed")
            assert env["OMP_NUM_THREADS"] == env["MKL_NUM_THREADS"] == env["OPENBLAS_NUM_THREADS"] == "1"

        def wait(self):
            return 7 if failed_stage and failed_stage in self.command else 0

    monkeypatch.setattr(runner, "freeze_code", freeze)
    monkeypatch.setattr(runner.subprocess, "Popen", Process)
    return calls


def test_partial_training_is_budget_hold_not_resumed(monkeypatch, tmp_path):
    calls = mock_lifecycle(monkeypatch, actual_epochs=1)
    target = tmp_path / "partial"
    job = runner.run(target, "cpu", True)
    assert job["status"] == "budget_hold" and not job["workflow_complete"]
    assert job["requires_user_action"] and len(calls) == 3
    assert job["stages"][-1]["returncode"] == 0
    assert (target / "run/summary.json").exists()
    with pytest.raises(FileExistsError):
        runner.run(target, "cpu", True)
    assert len(calls) == 3


def test_complete_cpu_execution_smoke_records_no_gpu_and_safe_public_files(monkeypatch, tmp_path):
    calls = mock_lifecycle(monkeypatch)
    job = runner.run(tmp_path / "complete", "cpu", True)
    assert job["status"] == "complete" and job["workflow_complete"]
    assert job["execution_smoke"] and not job["scientific_result"]
    assert len(calls) == 4 and all(stage["returncode"] == 0 for stage in job["stages"])
    assert job["GPU_stage_seconds"] == 0 and not any(row["gpu_stage"] for row in job["stages"])
    assert "report/encodings.npz" in job["public_files"]
    assert "report/private.npz" not in job["public_files"]
    assert not any(name.endswith(".pt") or "/disorders/" in name for name in job["public_files"])
    assert job["thermal_nu"] is None


def test_nonzero_stage_preserves_actual_returncode_and_stops(monkeypatch, tmp_path):
    calls = mock_lifecycle(monkeypatch, failed_stage="scripts.generate_spin_glass")
    target = tmp_path / "failed"
    job = runner.run(target, "cpu", True)
    assert job["status"] == "failed" and not job["workflow_complete"]
    assert job["requires_user_action"] and len(calls) == 1
    assert job["stages"][0]["returncode"] == 7 and job["stages"][0]["status"] == "failed"
    assert (target / "logs/physical_sampling.log").exists()
