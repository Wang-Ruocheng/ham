"""Warning-waiver controller tests: tiny bytes/JSON only, no physical or GPU work."""
import copy
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts import continue_phi4_hybrid_exploratory as runner


PROCESS_CHECK = runner.old_processes


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.legacy, "validate_host", lambda: None)
    monkeypatch.setattr(runner, "old_processes", lambda root: [])
    old_code, new_code = tmp_path / "old_code", tmp_path / "new_code"
    expected = {}
    for index, relative in enumerate(runner.EXPECTED_CRITICAL_SHA):
        for code in (old_code, new_code):
            path = code / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# immutable implementation {index}\n")
        expected[relative] = runner.sha(old_code / relative)
    extra = new_code / "scripts/continue_phi4_hybrid_exploratory.py"
    extra.write_text("# independent new orchestration\n")
    monkeypatch.setattr(runner, "EXPECTED_CRITICAL_SHA", expected)
    monkeypatch.setattr(runner, "current_code_root", lambda: new_code)
    root = tmp_path / "held"
    data = root / "data"
    data.mkdir(parents=True)
    options = dict(sizes=[36], dimension=2, chains=4, samples=128, u_values=[.2], h=0.,
        r_values=[-.5 + index*.8/11 for index in range(12)], cluster_probability=.5,
        test_every=3, test_offset=1)
    physical = dict(schema="euclidean_phi4_hmc_embedded_wolff_v1", options=options)
    runner.save(data / "plan.json", physical)
    shards = []
    for cell, r in enumerate(options["r_values"]):
        directory = data / "conditions" / f"c{cell:03d}"
        directory.mkdir(parents=True)
        payload = directory / "data.npz"
        payload.write_bytes(f"tiny opaque physical shard {cell}".encode())
        split = "test" if cell % 3 == 1 else "train"
        sidecar = dict(status="complete", cell_index=cell, dimension=2, size=36,
            r=r, u=.2, h=0., split=split, data_sha256=runner.sha(payload),
            chains=[dict(production_acceptance=.9, max_absolute_delta_h=.2) for _ in range(4)])
        runner.save(directory / "sidecar.json", sidecar)
        shards.append(dict(cell_index=cell, size=36, split=split,
            path=str(payload.relative_to(data)), sha256=runner.sha(payload),
            sidecar=str((directory / "sidecar.json").relative_to(data)),
            sidecar_sha256=runner.sha(directory / "sidecar.json")))
    runner.save(data / "manifest.json", dict(status="complete", schema=physical["schema"], options=options,
        plan_sha256=runner.sha(data / "plan.json"), shards=shards, warning_count=38))
    manifest_sha = runner.sha(data / "manifest.json")
    runner.save(data / "status.json", dict(status="complete", workflow_complete=True, returncode=0,
                manifest_sha256=manifest_sha))
    runner.save(root / "plan.json", dict(output=str(root), host=runner.HOST, gpu_index=1,
        gpu_uuid=runner.GPU_UUID, code_root=str(old_code), source_sha256=runner.common.source_hashes(old_code), physical=physical))
    runner.save(root / "job.json", dict(status="sampling_quality_hold", workflow_complete=False,
        parent_pid=10432, stages=[dict(name=name, status="complete", gpu=False, returncode=0)
            for name in ("sampler_preflight", "physical_sampling")], gpu_stage_wall_seconds=0.,
        cpu_seconds=233.764403, sampling_quality_issue_count=2))
    runner.save(root / "execution.json", dict(status="sampling_quality_hold", workflow_complete=False,
        parent_pid=10432, plan_sha256=runner.sha(root / "plan.json"), gpu_stage_wall_seconds=0., cpu_seconds=233.764403))
    runner.save(root / "launch.json", dict(parent_pid=10432, plan_sha256=runner.sha(root / "plan.json")))
    runner.save(root / "sampling_validity.json", dict(status="complete", severe=[]))
    runner.save(root / "sampling_quality.json", dict(status="sampling_quality_hold", issues=[{"kind": "ESS"}, {"kind": "same warning"}],
        data_manifest_sha256=manifest_sha, training_started=False, GPU_profile_started=False,
        source_warning_count=38, conditions=[{"all_warnings": "kept verbatim"}]))
    return root


@pytest.fixture
def prepared(source):
    child = source / "exploratory_training_v1"
    runner.prepare(source, child)
    return source, child


def test_prepare_preserves_original_hold_and_embeds_unabridged_waiver(prepared):
    source, child = prepared
    plan, auth = runner.read(child / "plan.json"), runner.read(child / "authorization.json")
    assert plan["scientific_qualification"] == runner.QUALIFICATION
    assert plan["source_quality_passed"] is False
    assert auth["user_authorization"] == "允许保留质量警告继续探索性训练"
    assert auth["source_quality"] == runner.read(source / "sampling_quality.json")
    assert auth["source_sampling_quality_sha256"] == runner.sha(source / "sampling_quality.json")
    assert runner.read(source / "job.json")["status"] == "sampling_quality_hold"
    assert plan["source_validation"]["configurations"] == 6144
    assert plan["source_validation"]["physical_data_loaded"] is False
    assert plan["resource_policy"]["source_cpu_seconds"] == 233.764403
    assert plan["resource_policy"]["remaining_cpu_seconds"] <= 7200-233.764403
    assert plan["resource_policy"]["additional_physical_force_site_evaluations"] == 0
    assert plan["training"]["warm_start"] is False
    assert len(plan["source_validation"]["shards_sha256"]) == 12
    runner.sources_unchanged(plan, full=True)


def test_child_must_be_fresh_proper_descendant_and_not_under_data(source, tmp_path):
    for output in (source, source.parent, tmp_path / "outside", source / "data/new_child"):
        with pytest.raises((ValueError, FileExistsError)):
            runner.prepare(source, output)
    child = source / "new_child"
    runner.prepare(source, child)
    with pytest.raises(FileExistsError):
        runner.prepare(source, child)


@pytest.mark.parametrize("mutation", ["status", "extra_stage", "run", "profile", "severe", "acceptance", "payload", "cpu"])
def test_waiver_rejects_incompatible_or_severely_invalid_source(source, mutation):
    if mutation in ("status", "extra_stage", "cpu"):
        path = source / "job.json"
        value = runner.read(path)
        if mutation == "status":
            value["status"] = "failed"
        elif mutation == "extra_stage":
            value["stages"].append({"name": "training", "returncode": 0, "gpu": True})
        else:
            value["cpu_seconds"] = 100
        runner.save(path, value)
    elif mutation in ("run", "profile"):
        (source / ("run" if mutation == "run" else "profile_run")).mkdir()
    elif mutation == "severe":
        runner.save(source / "sampling_validity.json", dict(status="invalid_sampling_hold", severe=[{"rate": 0}]))
    elif mutation == "payload":
        (source / "data/conditions/c000/data.npz").write_bytes(b"changed opaque payload")
    else:
        path = source / "data/conditions/c000/sidecar.json"
        sidecar = runner.read(path)
        sidecar["chains"][0]["production_acceptance"] = 0
        runner.save(path, sidecar)
        manifest_path = source / "data/manifest.json"
        manifest = runner.read(manifest_path)
        manifest["shards"][0]["sidecar_sha256"] = runner.sha(path)
        runner.save(manifest_path, manifest)
        for target in (source / "data/status.json", source / "sampling_quality.json"):
            value = runner.read(target)
            value["manifest_sha256" if target.name == "status.json" else "data_manifest_sha256"] = runner.sha(manifest_path)
            runner.save(target, value)
    with pytest.raises((RuntimeError, ValueError)):
        runner.prepare(source, source / "rejected")
    assert not (source / "rejected").exists()


def test_all_old_frozen_sources_must_match_new_snapshot(source, monkeypatch):
    old_plan = runner.read(source / "plan.json")
    relative = "scripts/additional_old_helper.py"
    old = Path(old_plan["code_root"]) / relative
    old.write_text("# existing old helper\n")
    new = runner.current_code_root() / relative
    new.write_text("# changed helper, model remains unchanged\n")
    old_plan["source_sha256"] = runner.common.source_hashes(Path(old_plan["code_root"]))
    runner.save(source / "plan.json", old_plan)
    for target in ("execution.json", "launch.json"):
        value = runner.read(source / target)
        value["plan_sha256"] = runner.sha(source / "plan.json")
        runner.save(source / target, value)
    with pytest.raises(RuntimeError, match="original frozen source"):
        runner.prepare(source, source / "different_implementation")


def test_real_process_identity_blocks_source_not_unrelated_stale_pid(source, monkeypatch):
    listing = ("10432 1 python -m unrelated.worker --output /elsewhere\n"
               "567 1 arbitrary unbalanced 'shell text\n"
               f"890 1 python -m scripts.run_phi4_u02_hybrid_qcr --execute --output {source}\n")
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *args, **kwargs: listing)
    assert [row["pid"] for row in PROCESS_CHECK(source)] == [890]
    monkeypatch.setattr(runner, "old_processes", PROCESS_CHECK)
    with pytest.raises(RuntimeError, match="actual parent/children"):
        runner.prepare(source, source / "blocked_by_live_source")


def test_metadata_and_shard_mutations_cannot_be_hidden_by_waiver(prepared):
    source, child = prepared
    plan = runner.read(child / "plan.json")
    payload = source / "data/conditions/c000/data.npz"
    payload.write_bytes(b"mutated after waiver")
    with pytest.raises(RuntimeError, match="physical source shard"):
        runner.sources_unchanged(plan, full=True)
    value = runner.read(source / "job.json")
    value["status"] = "complete"
    runner.save(source / "job.json", value)
    with pytest.raises(RuntimeError, match="metadata"):
        runner.sources_unchanged(plan)


def test_commands_use_only_immutable_existing_data_and_fresh_seeds(prepared):
    source, child = prepared
    plan = runner.read(child / "plan.json")
    stages = runner.commands(plan)
    assert [row[0] for row in stages] == ["training_preflight", "training", "report", "phi4_hybrid_report", "exploratory_qualification_report"]
    for index, seed in ((0, "191"), (1, "91")):
        command = stages[index][1]
        assert command[command.index("--data")+1] == str(source / "data")
        assert command[command.index("--seed")+1] == seed
        assert not {"--resume", "--source"}.intersection(command)
    assert stages[1][1][stages[1][1].index("--steps")+1] == "3000"
    assert not any("prepare_phi4" in item or "sample_hybrid" in item for _, command, _ in stages for item in command)


def install_fake_children(prepared, monkeypatch, *, fail=None):
    source, child = prepared
    monkeypatch.setattr(runner.hybrid, "admission", lambda *args: None)
    monkeypatch.setattr(runner.os, "getpriority", lambda *args: 10)
    monkeypatch.setattr(runner.os, "nice", Mock())
    monkeypatch.setattr(runner.common, "profile_projection", lambda *args:
                        dict(budget_passed=True, projected_formal_gpu_seconds_conservative=900.))
    calls = []
    def popen(command, **kwargs):
        module = command[command.index("-m")+1]
        output = Path(command[command.index("--output")+1])
        output.mkdir()
        if module == "experiments.phi4_qcr":
            steps = int(command[command.index("--steps")+1])
            name = "training_preflight" if steps == 20 else "training"
            runner.save(output / "summary.json", dict(actual_steps=steps, device={"uuid": runner.GPU_UUID},
                peak_cuda_bytes=1024, optimizer_steps={key: {"min": steps, "max": steps} for key in ("Q", "C", "R")},
                labels_used_in_training=False, frozen_evaluation_verified=True,
                data_manifest_sha256=runner.sha(source / "data/manifest.json")))
            runner.save(output / "history_progress.json", {})
        else:
            name = "report" if module == "scripts.report_phi4_qcr" else \
                "phi4_hybrid_report" if module == "scripts.report_phi4_hybrid_qcr" else "exploratory_qualification_report"
            runner.save(output / "status.json", dict(status="complete", workflow_complete=True))
        calls.append((name, command, kwargs))
        return Mock(pid=300+len(calls), wait=lambda: int(name == fail))
    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    return calls


def test_success_preserves_source_hold_and_records_shared_budget(prepared, monkeypatch):
    source, child = prepared
    plan = runner.read(child / "plan.json")
    before = runner.metadata_fingerprints(source, child)
    calls = install_fake_children(prepared, monkeypatch)
    runner.execute(source, child)
    job = runner.read(child / "job.json")
    execution = runner.read(child / "execution.json")
    assert job["status"] == "complete" and job["workflow_complete"]
    assert job["scientific_qualification"] == runner.QUALIFICATION and not job["source_quality_passed"]
    assert job["actual_training_steps"] == 3000 and len(calls) == 5
    assert all(stage["returncode"] == 0 for stage in job["stages"])
    assert execution["source_cpu_seconds"] == 233.764403
    assert job["cumulative_cpu_seconds"] >= 233.764403
    assert runner.metadata_fingerprints(source, child) == before
    assert runner.read(source / "job.json")["status"] == "sampling_quality_hold"
    assert not any(key.endswith(".pt") or key.startswith("profile_run/") or key.startswith("data/") for key in job["public_files"])
    assert "authorization.json" in job["public_files"]
    for name, command, kwargs in calls:
        if name in ("training", "training_preflight"):
            assert "CUDA_VISIBLE_DEVICES" not in kwargs["env"]
        else:
            assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
    runner.os.nice.assert_not_called()
    runner.sources_unchanged(plan, full=True)


def test_failure_never_retries_or_restarts_original_parent(prepared, monkeypatch):
    source, child = prepared
    calls = install_fake_children(prepared, monkeypatch, fail="training")
    with pytest.raises(RuntimeError, match="no retry/restart"):
        runner.execute(source, child)
    assert [row[0] for row in calls] == ["training_preflight", "training"]
    assert runner.read(source / "job.json")["status"] == "sampling_quality_hold"
    assert runner.read(child / "job.json")["status"] == "failed"


def test_original_tree_plus_child_directory_budget_blocks_work(prepared, monkeypatch):
    source, child = prepared
    calls = install_fake_children(prepared, monkeypatch)
    monkeypatch.setattr(runner, "directory_bytes", lambda path: runner.DIRECTORY_BYTES+1)
    runner.execute(source, child)
    assert not calls
    assert runner.read(child / "job.json")["status"] == "budget_hold"


def test_exclusive_child_launch_does_not_resume_original_parent(prepared, monkeypatch):
    source, child = prepared
    parent = Mock(return_value=Mock(pid=898))
    monkeypatch.setattr(runner.subprocess, "Popen", parent)
    runner.launch(source, child)
    command = parent.call_args.args[0]
    assert runner.MODULE in command
    assert "--source-root" in command and "--execute" in command
    assert "scripts.run_phi4_u02_hybrid_qcr" not in command
    with pytest.raises(FileExistsError):
        runner.launch(source, child)
    assert parent.call_count == 1
