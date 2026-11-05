"""Bounded preview contract tests: mocked fields/model only, no SSH or training."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from experiments import phi4_hopping_qcr as training
from recurrent_rg import phi4_aligned_qcr as core
from recurrent_rg.data import phi4_hopping_loader as loader
from scripts import capture_phi4_hopping_checkpoint_flow as capture


class _MeanQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1., dtype=torch.float64))

    def forward(self, field):
        return field.flatten(1).mean(1, keepdim=True) * self.weight


class _MockModel(nn.Module):
    """Tiny mock; deliberately does not execute the scientific Q/C networks."""
    def __init__(self, scale, **_kwargs):
        super().__init__()
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float64))
        self.Q = _MeanQ()
        self.C = nn.Linear(1, 1).double()
        self.R = nn.Linear(1, 1).double()

    def coarse_grain(self, field, *, generator):
        levels = [field]
        for _ in range(3):
            noise = torch.randn(field.shape, generator=generator, dtype=field.dtype)
            levels.append(levels[-1] * .8 + .001 * noise)
        return levels

    def advance(self, g):
        return g + self.R(g)


@pytest.fixture
def saved_preview(tmp_path, monkeypatch):
    previous_threads = torch.get_num_threads()
    root = tmp_path / "experiment"
    (root / "training").mkdir(parents=True)
    code = Path(training.__file__).resolve().parents[1]
    pins = {"experiments/phi4_hopping_qcr.py": capture.sha(training.__file__)}
    plan = dict(code_root=str(code), source_sha256=pins)
    (root / "plan.json").write_text(json.dumps(plan))
    (root / "job.json").write_text(json.dumps(dict(plan_sha256=capture.sha(root / "plan.json"))))
    manifest = dict(recipe=training.RECIPE, theory_dim=1, scale=1.,
                    model=training.model_policy(2, 3), lattice_sizes=[128, 64, 32, 16],
                    labels_used_in_training=False, source_sha256=pins,
                    data_manifest_sha256="synthetic-mocked-dataset")
    (root / "training/manifest.json").write_text(json.dumps(manifest))
    (root / "training/status.json").write_text(json.dumps(dict(
        latest_checkpoint_step=1000, actual_steps=1100, status="training")))
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(91)
        model = _MockModel(1.)
        with torch.no_grad():
            model.R.weight.fill_(-.2)
            model.R.bias.fill_(.05)
    checkpoint = root / "training/training_step_1000.pt"
    torch.save(dict(recipe=training.RECIPE, step=1000, manifest=manifest,
                    model=model.state_dict(), optimizers={"never": "restore"},
                    batch_rng="never-restore", field_rng="never-restore"), checkpoint)
    # All physical fields and model calls below are mocks, not actual diagnostics.
    values = torch.linspace(-.5, .8, 60, dtype=torch.float64)
    dataset = loader.HoppingFieldDataset(
        values[:, None, None, None, None].expand(60, 8, 4, 4, 1).clone(),
        np.array(["train"] * 60), np.repeat(np.arange(15), 4),
        np.repeat(np.column_stack((np.linspace(.32, 1.04, 15), np.ones(15), np.zeros(15))), 4, axis=0),
        "synthetic-mocked-dataset", [], 75, "mocked, not physical evaluation")
    monkeypatch.setattr(core, "AlignedPhi4QCR", _MockModel)
    monkeypatch.setattr(loader, "load_dataset", lambda *a, **k: dataset)
    monkeypatch.setattr(capture.time, "sleep", lambda _: None)
    monkeypatch.setattr(capture.os, "cpu_count", lambda: 24)
    monkeypatch.setattr(capture.os, "getloadavg", lambda: (1., 1., 1.))
    monkeypatch.setattr(capture.os, "getpriority", lambda *args: 10)
    original_read = Path.read_text
    def read_text(path, *args, **kwargs):
        if str(path) == "/proc/meminfo":
            return "MemAvailable: 120000000 kB\n"
        if str(path) == "/proc/pressure/io":
            return "some avg10=0.00\nfull avg10=0.00 avg60=0.00\n"
        return original_read(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", read_text)
    yield root, checkpoint
    torch.set_num_threads(previous_threads)


def test_mocked_capture_preserves_model_checkpoint_and_rng(saved_preview):
    root, checkpoint = saved_preview
    before = capture.binding(checkpoint)
    rng = torch.get_rng_state().clone()
    result = capture.capture(root, step=1000, expected_sha256=before[2])
    assert capture.binding(checkpoint) == before
    assert torch.equal(rng, torch.get_rng_state())
    assert result["physical_chains"] == 15
    assert result["selected_ensemble_indices"] == list(range(0, 60, 4))
    assert result["samples_per_chain"] == 8 and result["C_draws_per_chain"] == 1
    assert result["new_optimizer_updates"] == result["new_physical_sampling"] == 0
    assert not result["optimizer_state_restored"] and not result["training_rng_restored"]
    assert not result["cuda_initialized"] and not result["final_report"]
    assert result["model_unchanged"] and result["source_unchanged"] and result["checkpoint_unchanged"]
    assert result["thermal_nu"] is None
    assert np.asarray(result["direct"]).shape == (15, 4)
    assert np.asarray(result["one_step"]).shape == (15, 3)
    np.testing.assert_allclose(result["flow_R"], .8 * np.asarray(result["flow_g"]) + .05)
    np.testing.assert_allclose(result["flow_displacement"],
                               np.asarray(result["flow_R"]) - result["flow_g"])
    np.testing.assert_allclose(result["full_DR"], .8)
    assert not (root / "report").exists()


def test_unconfirmed_checkpoint_rejected_before_load(saved_preview):
    root, _ = saved_preview
    with pytest.raises(ValueError, match="completely saved"):
        capture.choose_checkpoint(root, step=2000)


def test_unstable_checkpoint_rejected(saved_preview, monkeypatch):
    root, _ = saved_preview
    calls = iter([(10, 1, "a"), (11, 1, "b")])
    monkeypatch.setattr(capture, "binding", lambda _: next(calls))
    with pytest.raises(RuntimeError, match="not stable"):
        capture.choose_checkpoint(root)


def test_wrong_sha_rejected(saved_preview):
    root, _ = saved_preview
    with pytest.raises(RuntimeError, match="immutable fingerprint"):
        capture.choose_checkpoint(root, expected_sha256="0" * 64)


def test_wrong_checkpoint_manifest_rejected(saved_preview):
    root, checkpoint = saved_preview
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved["manifest"]["scale"] = 2.
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError, match="manifest/geometry"):
        capture.capture(root)


def test_source_mismatch_rejected(saved_preview):
    root, _ = saved_preview
    plan = json.loads((root / "plan.json").read_text())
    plan["source_sha256"]["experiments/phi4_hopping_qcr.py"] = "0" * 64
    (root / "plan.json").write_text(json.dumps(plan))
    (root / "job.json").write_text(json.dumps(dict(plan_sha256=capture.sha(root / "plan.json"))))
    with pytest.raises(RuntimeError, match="Frozen source/evidence"):
        capture.capture(root)


def test_crossing_interpolation_and_exact_endpoint():
    assert capture.crossing_candidates([-1., 0., 1.], [-1., 0., 1.]) == [0.]
    assert capture.crossing_candidates([0., 1., 2.], [-2., -1., 0.]) == [2.]
    assert capture.crossing_candidates([0., 1.], [-.25, .75]) == [.25]
