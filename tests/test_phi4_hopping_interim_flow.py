import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.plot_phi4_hopping_interim_flow import load_capture, plot


def capture():
    direct = np.asarray([[-.7, -.6, -.5, -.4], [0., .1, .2, .3], [.7, .8, .9, 1.]])
    grid = np.linspace(-.8, 1., 101)
    drift = .15 * grid * (1. - grid)
    return {"kind": "synthetic_interim_flow_test", "checkpoint_step": 1000,
            "direct": direct.tolist(), "one_step": direct[:, :3].tolist(),
            "labels": [[.32, 1, 0], [.68, 1, 0], [1.04, 1, 0]],
            "splits": ["train", "test", "train"], "cell_ids": [0, 7, 14],
            "lattice_sizes": [128, 64, 32, 16], "flow_g": grid.tolist(),
            "flow_R": (grid + drift).tolist(), "flow_displacement": drift.tolist(),
            "per_input_scale_sampled_ranges": [[-.7, .7], [-.6, .8], [-.5, .9]],
            "sampled_input_range": [-.7, .9], "thermal_nu": None,
            "interpolated_zeros": [
                {"g": 0., "full_DR": 1.15, "residual": 0., "classification": "repelling",
                 "inside_sampled_input_range": True},
                {"g": 1., "full_DR": .85, "residual": 0., "classification": "attracting",
                 "inside_sampled_input_range": False}]}


def save(tmp_path, data):
    source = tmp_path / "capture.json"
    source.write_text(json.dumps(data))
    return source


def test_load_valid_public_capture(tmp_path):
    data = capture()
    assert load_capture(save(tmp_path, data)) == data


@pytest.mark.parametrize("field,value", [
    ("flow_displacement", [0.] * 101),
    ("sampled_input_range", [-.8, .9]),
    ("per_input_scale_sampled_ranges", [[-.8, .7], [-.6, .8], [-.5, .9]]),
    ("direct", [[0., 1., 2.], [1., 2., 3.]]),
    ("flow_g", list(range(100, -1, -1))),
    ("lattice_sizes", [108, 36, 12, 4]),
    ("thermal_nu", 1.),
    ("checkpoint_step", 0),
])
def test_reject_inconsistent_capture(tmp_path, field, value):
    data = capture()
    data[field] = value
    with pytest.raises(ValueError):
        load_capture(save(tmp_path, data))


def test_reject_wrong_zero_range_classification(tmp_path):
    data = capture()
    data["interpolated_zeros"][0]["inside_sampled_input_range"] = False
    with pytest.raises(ValueError, match="classification"):
        load_capture(save(tmp_path, data))


def test_plot_never_overwrites(tmp_path):
    destination = tmp_path / "already.png"
    destination.write_bytes(b"preserved")
    with pytest.raises(FileExistsError):
        plot(tmp_path / "missing.json", destination)
    assert destination.read_bytes() == b"preserved"


def test_render_small_synthetic_capture(tmp_path):
    pytest.importorskip("matplotlib")
    source = save(tmp_path, capture())
    try:
        destination = plot(source, tmp_path / "synthetic_flow.png")
    except FileNotFoundError as exc:
        if "font" in str(exc).lower():
            pytest.skip(str(exc))
        raise
    assert destination.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert destination.stat().st_size > 10_000


def test_plot_module_does_not_import_model_dependencies():
    source = Path(importlib.util.find_spec("scripts.plot_phi4_hopping_interim_flow").origin).read_text()
    assert "import torch" not in source
    assert "torch.load" not in source
    assert "recurrent_rg" not in source
