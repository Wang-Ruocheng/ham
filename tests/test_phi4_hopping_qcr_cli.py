"""Shared CLI dispatch tests: no data reading, sampling or optimizer updates."""
import json

import pytest

from recurrent_rg.cli import main


def config(tmp_path):
    path = tmp_path / "new.json"
    path.write_text(json.dumps(dict(recipe="phi4_hopping_raw_patch_joint625_v1")))
    return path


def test_shared_train_entry_dispatches_only_explicit_continuous_recipe(tmp_path, monkeypatch):
    import experiments.phi4_hopping_qcr as trainer
    calls = []
    monkeypatch.setattr(trainer, "main", lambda args: calls.append(args))
    path = config(tmp_path)
    main(["train", "--config", str(path), "--output", "NEW", "--device", "cuda:2"])
    assert calls == [["--config", str(path), "--output", "NEW", "--execute", "--device", "cuda:2"]]


def test_continuous_config_device_is_not_overridden_by_implicit_cli_default(tmp_path, monkeypatch):
    import experiments.phi4_hopping_qcr as trainer
    calls = []
    monkeypatch.setattr(trainer, "main", lambda args: calls.append(args))
    path = config(tmp_path)
    main(["train", "--config", str(path), "--output", "NEW"])
    assert calls == [["--config", str(path), "--output", "NEW", "--execute"]]


@pytest.mark.parametrize("device_flag", ["--device=cpu", "--dev=cpu"])
def test_explicit_cpu_device_overrides_continuous_config(tmp_path, monkeypatch, device_flag):
    import experiments.phi4_hopping_qcr as trainer
    calls = []
    monkeypatch.setattr(trainer, "main", lambda args: calls.append(args))
    path = config(tmp_path)
    main(["train", "--config", str(path), "--output", "NEW", device_flag])
    assert calls == [["--config", str(path), "--output", "NEW", "--execute", "--device", "cpu"]]


@pytest.mark.parametrize("flags", [
    ["--resume", "checkpoint.pt"], ["--initialize-from", "checkpoint.pt"],
    ["--allow-legacy-training"], ["--reference-execution"],
    ["--benchmark-warmup", "10"], ["--mi-cache-dir", "binary_cache"],
])
def test_shared_continuous_entry_rejects_unsupported_modes(tmp_path, monkeypatch, flags):
    import experiments.phi4_hopping_qcr as trainer
    monkeypatch.setattr(trainer, "main", lambda args: pytest.fail("unsupported dispatch executed"))
    with pytest.raises(SystemExit) as error:
        main(["train", "--config", str(config(tmp_path)), "--output", "NEW", *flags])
    assert error.value.code == 2


def test_discrete_configs_keep_existing_trainer_route(tmp_path, monkeypatch):
    import experiments.rsmi_separate_qcr as discrete
    path = tmp_path / "discrete.json"
    path.write_text(json.dumps(dict(schema_version=5)))
    calls = []
    monkeypatch.setattr(discrete, "main", lambda args: calls.append(args))
    main(["train", "--config", str(path), "--output", "NEW"])
    assert calls == [["--config", str(path), "--output", "NEW"]]
