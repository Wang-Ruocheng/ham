from pathlib import Path

import pytest

from scripts import run_phi4_mean_bias_full_qcr as runner


def test_single_change_commands_and_no_physical_sampling(tmp_path):
    stages=runner.commands(tmp_path)
    assert [stage[2] for stage in stages] == [True,False,False,False]
    train=stages[0][1]
    for flag,value in [("--steps","3000"),("--seed","91"),("--system-batch","8"),
                       ("--sample-batch","64"),("--information-pairs","128"),
                       ("--q-initialization","train_mean_bias"),
                       ("--expected-random-construction-sha256",runner.INITIAL_SHA)]:
        assert train[train.index(flag)+1] == value
    flattened=" ".join(token for _,cmd,_ in stages for token in cmd)
    assert "prepare_phi4" not in flattened
    assert "--resume" not in flattened
    assert "--source-checkpoint" not in flattened
    assert "--sampling-quality" in flattened


def test_prepare_never_overwrites_existing_output(tmp_path,monkeypatch):
    monkeypatch.setattr(runner.guard,"validate_host",lambda:None)
    with pytest.raises(FileExistsError):
        runner.prepare(tmp_path)


def test_prepare_rejects_old_endpoint_before_creating_output(tmp_path,monkeypatch):
    monkeypatch.setattr(runner.guard,"validate_host",lambda:None)
    monkeypatch.setattr(runner,"immutable_inputs",lambda:{})
    def fake_read(path):
        return {"status":"complete","actual_steps":12,"initial_state_sha256":"wrong"}
    monkeypatch.setattr(runner,"read",fake_read)
    target=tmp_path/"fresh"
    with pytest.raises(RuntimeError,match="endpoint/data"):
        runner.prepare(target)
    assert not target.exists()


def test_no_duplicate_execution(tmp_path,monkeypatch):
    monkeypatch.setattr(runner.guard,"validate_host",lambda:None)
    monkeypatch.setattr(runner,"read",lambda path:{"status":"complete","stages":[{}]})
    with pytest.raises(RuntimeError,match="duplicate"):
        runner.execute(tmp_path)
    assert not (tmp_path/"execution.lock").exists()


def test_source_inventory_must_remain_frozen(monkeypatch):
    monkeypatch.setattr(runner,"source_hashes",lambda path:{"model.py":"new"})
    with pytest.raises(RuntimeError,match="new source"):
        runner.unchanged({"code_root":"x","source_sha256":{"model.py":"old"}})


def test_immutable_source_metadata_cannot_drift(monkeypatch):
    monkeypatch.setattr(runner,"source_hashes",lambda path:{"model.py":"same"})
    monkeypatch.setattr(runner,"immutable_inputs",lambda:{"data/manifest.json":"wrong"})
    with pytest.raises(RuntimeError,match="original experiment"):
        runner.unchanged({"code_root":"x","source_sha256":{"model.py":"same"},
                          "immutable_input_sha256":{"data/manifest.json":"right"}})
