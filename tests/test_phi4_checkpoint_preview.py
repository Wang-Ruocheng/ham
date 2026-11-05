"""Synthetic saved states only: no training, real data, GPU or SSH."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments import phi4_qcr as training
from recurrent_rg.phi4_qcr import Phi4QCR
from scripts import preview_phi4_checkpoint_flow as preview


@pytest.fixture
def saved_state(tmp_path, monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(13)
    code = Path(training.__file__).resolve().parents[1]
    root = tmp_path / "experiment"
    (root / "run").mkdir(parents=True)
    plan = {"code_root": str(code), "source_sha256": {
        "experiments/phi4_qcr.py": training.sha256_file(training.__file__)}}
    (root / "plan.json").write_text(json.dumps(plan))
    (root / "job.json").write_text(json.dumps({"plan_sha256": training.sha256_file(root / "plan.json")}))
    model = Phi4QCR(1.)
    manifest = dict(scale=1., theory_dim=1, recipe=training.RECIPE,
                    data_manifest_sha256="synthetic", lattice_sizes=[9, 3, 1])
    torch.save(dict(step=1000, manifest=manifest, model=model.state_dict()), root / "run" / "training_step_1000.pt")
    data = training.FieldDataset(torch.randn(8, 4, 9, 9, 1, dtype=torch.float64),
        np.array(["train"] * 4 + ["test"] * 4), np.repeat([0, 1], 4),
        np.array([[-1., 1., 0.]] * 4 + [[0., 1., 0.]] * 4), "synthetic", [], 5, "synthetic")
    monkeypatch.setattr(training, "load_dataset", lambda path: data)
    yield root, data
    torch.set_num_threads(previous)


def test_cpu_capture_keeps_saved_state_and_separates_mean_predictions(saved_state):
    root, dataset = saved_state
    sha = training.sha256_file(root / "run" / "training_step_1000.pt")
    result = preview.capture(root, step=1000, samples=4)
    assert result["checkpoint_sha256"] == sha
    assert training.sha256_file(root / "run" / "training_step_1000.pt") == sha
    assert result["checkpoint_unchanged"] and result["model_unchanged"] and result["source_unchanged"]
    assert result["cuda_initialized"] is False and result["new_optimizer_updates"] == 0
    assert result["physical_chains"] == 8 and result["samples_per_chain"] == 4
    assert result["final_report"] is False and result["thermal_nu"] is None
    g, r, drift = [np.asarray(result[key]) for key in ("flow_g", "flow_R", "flow_displacement")]
    np.testing.assert_allclose(r - g, drift)
    assert np.all(np.isfinite(result["full_DR"]))
    for cell in result["cells"]:
        assert "mean_R" in cell and "R_of_mean" in cell
    assert not (root / "report").exists()


def test_wrong_checkpoint_identity_is_rejected(saved_state):
    root, _ = saved_state
    checkpoint = root / "run" / "training_step_1000.pt"
    value = torch.load(checkpoint, map_location="cpu", weights_only=False)
    value["step"] = 500
    torch.save(value, checkpoint)
    with pytest.raises(ValueError, match="step"):
        preview.capture(root, step=1000, samples=4)


def test_source_change_is_rejected_before_inference(saved_state):
    root, _ = saved_state
    plan = json.loads((root / "plan.json").read_text())
    plan["source_sha256"]["experiments/phi4_qcr.py"] = "0" * 64
    (root / "plan.json").write_text(json.dumps(plan))
    (root / "job.json").write_text(json.dumps({
        "plan_sha256": training.sha256_file(root / "plan.json")}))
    with pytest.raises(RuntimeError, match="Frozen source"):
        preview.capture(root, step=1000, samples=4)
