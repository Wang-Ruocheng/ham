"""Two-update synthetic execution regression, never a physical experiment."""
import json
import os
from pathlib import Path
import subprocess
import sys

from recurrent_rg.data.spin_glass_pipeline import GenerationOptions, generate_dataset
from scripts.prepare_spin_glass_qcr import prepare_overlap_cache


def test_cpu_default_recipe_two_update_execution(tmp_path):
    source, data = tmp_path / "physical", tmp_path / "data"
    generate_dataset(source, GenerationOptions(shape=(16,), temperatures=(.7, 2.),
        disorders=3, replicas=2, samples=8, burn_in=4, spacing=1, seed=77))
    prepare_overlap_cache(source, data)
    repo = Path(__file__).resolve().parents[1]
    config = json.loads((repo / "configs/spin_glass1d_overlap_g1_seed91.json").read_text())
    config["data"]["cache"] = str(data / "ensembles.pt")
    config["training"].update(epochs=2, max_rg_steps=2, sample_batch=8, system_batch=4,
                              log_every=1, checkpoint_every=2)
    config["evaluation"].update(max_rg_steps=2, samples=8, systems=2)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    command = [sys.executable, "-m", "experiments.rsmi_separate_qcr", "--config", str(path),
               "--output", str(tmp_path / "run"), "--device", "cpu", "--skip-final-report"]
    result = subprocess.run(command, cwd=repo, capture_output=True, text=True, timeout=60,
        env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"))
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((tmp_path / "run/summary.json").read_text())
    manifest = json.loads((tmp_path / "run/manifest.json").read_text())
    assert summary["actual_epochs"] == 2 and not summary["stopped_by_training_limit"]
    assert manifest["labels_used_in_training"] is False
    assert manifest["training_driver"] == "rsmi_separate_qcr"
    assert manifest["config"]["loss"]["closure_scale_weights"] is None
    assert (tmp_path / "run/checkpoint_last.pt").exists()
