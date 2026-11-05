"""新物理留出父程序的本地合同测试；不生成构型、不加载模型。"""
import json
from pathlib import Path

import pytest

from scripts import run_phi4_ru_validation_followup as runner


def test_fixed_minimal_physical_design(tmp_path):
    root = tmp_path / "uncreated"
    plan = runner.build_plan(root, tmp_path / "old", tmp_path / "validation")
    assert not root.exists()
    assert len(plan["subsets"]) == 13
    assert plan["independent_physical_chains"] == 156
    assert plan["unique_size_parameter_conditions"] == 24
    assert plan["total_hmc_trajectories"] == 156 * 2048
    assert plan["total_force_site_evaluations"] == (120 * 36**2 + 36 * 108**2) * 2048 * 13
    assert plan["budget_projection"]["volume_weighted_equivalent_L36_chains"] == 444
    assert plan["budget_projection"]["budget_passed"]
    assert plan["all_new_data_evaluation_only"]
    assert not plan["labels_used_in_R_optimization"]
    for row in plan["subsets"]:
        options = row["physical"]["options"]
        assert options["chains"] == 4
        assert options["samples"] == 128
        assert options["burn_in"] == 1024 and options["spacing"] == 8
        assert options["step_size"] == .08 and options["leapfrog_steps"] == 12
        assert row["physical"]["total_force_site_evaluations"] <= 7_000_000_000
        assert row["r_values"][1] == .5 * (row["r_values"][0] + row["r_values"][2])
    assert {s["seed"] for s in plan["subsets"] if s["size"] == 36} == {2026103001, 2026103002}
    assert all(s["seed"] == 2026103003 for s in plan["subsets"] if s["size"] == 108)
    assert plan["subsets"][0]["size"] == 108


def test_child_commands_reproduce_exact_fixed_options(tmp_path, monkeypatch):
    from scripts import prepare_phi4_ru as cli
    seen = []
    monkeypatch.setattr(cli, "generate_dataset", lambda out, options:
        seen.append((out, options)) or dict(shards=[], warning_count=0))
    plan = runner.build_plan(tmp_path / "new", tmp_path / "old", tmp_path / "validation")
    commands = runner.stage_commands(plan)
    for (_, command), expected in zip(commands[:13], plan["subsets"]):
        cli.main(command[command.index("scripts.prepare_phi4_ru") + 1:])
        from recurrent_rg.data.phi4_ru_pipeline import generation_plan
        actual = json.loads(json.dumps(generation_plan(seen[-1][1])))
        assert actual == json.loads(json.dumps(expected["physical"]))
    assert commands[-1][0] == "frozen_evaluation"
    assert "experiments.phi4_ru_followup_evaluate" in commands[-1][1]
    assert "--validation" in commands[-1][1]
    assert "--resume" not in " ".join(c for _, command in commands for c in command)


def make_upstream(tmp_path):
    root = tmp_path / "validation"
    root.mkdir()
    runner.save(root / "status.json", dict(status="complete", workflow_complete=True,
        returncode=0, error=None))
    runner.save(root / "manifest.json", dict(optimizer_updates=dict(Q=0, C=0)))
    runner.save(root / "summary.json", dict(selected_using_validation_only=True))
    (root / "R_parameters.npz").write_bytes(b"fixture-public-parameters")
    runner.save(root / "public_files.json", {p.name: runner.sha(p) for p in root.iterdir()})
    return root


def test_upstream_completion_and_hash_gate(tmp_path):
    root = make_upstream(tmp_path)
    pins = runner.verify_public(root)
    assert len(pins) == 5
    runner.save(root / "summary.json", dict(changed=True))
    with pytest.raises(RuntimeError, match="fingerprint"):
        runner.verify_public(root)


def test_incomplete_upstream_blocks_sampling(tmp_path):
    root = make_upstream(tmp_path)
    runner.save(root / "status.json", dict(status="running", workflow_complete=False,
        returncode=None, error=None))
    with pytest.raises(ValueError, match="complete"):
        runner.verify_public(root)


def test_public_paths_cannot_escape(tmp_path):
    with pytest.raises(ValueError):
        runner.inside(tmp_path, "../other")
    with pytest.raises(ValueError):
        runner.inside(tmp_path, "/etc/passwd")


def test_aggregate_rejects_nonzero_child_status(tmp_path):
    plan = runner.build_plan(tmp_path / "new", tmp_path / "old", tmp_path / "validation")
    directory = Path(plan["output"]) / plan["subsets"][0]["relative_output"]
    directory.mkdir(parents=True)
    runner.save(directory / "manifest.json", {})
    runner.save(directory / "status.json", dict(status="failed", workflow_complete=False,
        returncode=1, error="sampling exception"))
    with pytest.raises(RuntimeError, match="Incomplete"):
        runner.aggregate_data(plan)


def test_resource_cpu_only_thresholds(tmp_path, monkeypatch):
    plan = runner.build_plan(tmp_path / "new", tmp_path / "old", tmp_path / "validation")
    read_text = Path.read_text
    metrics = dict(gib=36., load=30., io=19.9)
    def fake_read(path, *args, **kwargs):
        if str(path) == "/proc/meminfo":
            return f"MemAvailable: {int(metrics['gib'] * 1024**2)} kB\n"
        if str(path) == "/proc/pressure/io":
            return f"full avg10={metrics['io']} avg60=0 avg300=0 total=0\n"
        return read_text(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", fake_read)
    monkeypatch.setattr(runner.os, "cpu_count", lambda: 24)
    monkeypatch.setattr(runner.os, "getloadavg", lambda: (metrics["load"], 0, 0))
    assert runner.resource_snapshot(plan)["host_ready"]
    metrics["io"] = 20.
    assert not runner.resource_snapshot(plan)["host_ready"]
    metrics.update(io=0., gib=35.99)
    assert not runner.resource_snapshot(plan)["host_ready"]


def test_proc_child_cpu_counts_completed_descendants(monkeypatch):
    read_text = Path.read_text
    def fake_read(path, *args, **kwargs):
        if str(path) == "/proc/123/stat":
            fields = ["S"] + ["0"] * 49
            for i, value in zip((11, 12, 13, 14), (10, 20, 30, 40)):
                fields[i] = str(value)
            return "123 (child name) " + " ".join(fields)
        return read_text(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", fake_read)
    monkeypatch.setattr(runner.os, "sysconf", lambda name: 100)
    assert runner.live_child_cpu(123) == 1.


def test_full_inventory_keeps_two_new_chain_sets_and_sizes_distinct(tmp_path):
    plan = runner.build_plan(tmp_path / "new", tmp_path / "old", tmp_path / "validation")
    for row in plan["subsets"]:
        directory = Path(plan["output"]) / row["relative_output"]
        directory.mkdir(parents=True)
        shards = []
        for cell, r in enumerate(row["r_values"]):
            child = directory / f"condition{cell}"
            child.mkdir()
            (child / "data.npz").write_bytes(b"fixture-physical-data")
            from recurrent_rg.data.phi4_hybrid_pipeline import chain_seed
            from recurrent_rg.data.phi4_ru_pipeline import Phi4RUOptions
            options = Phi4RUOptions(**dict(row["physical"]["options"],
                sizes=tuple(row["physical"]["options"]["sizes"]),
                r_values=tuple(row["r_values"]), u_values=(row["u"],)))
            runner.save(child / "sidecar.json", dict(r=r, u=row["u"], h=0.,
                size=row["size"], independent_chains=4, samples_per_chain=128,
                chains=[dict(production_acceptance=.9,
                    seed=chain_seed(options, row["size"], cell, chain)) for chain in range(4)]))
            shards.append(dict(cell_index=cell, path=f"condition{cell}/data.npz",
                sha256=runner.sha(child / "data.npz"), sidecar=f"condition{cell}/sidecar.json",
                sidecar_sha256=runner.sha(child / "sidecar.json"), split="test" if cell == 1 else "train"))
        runner.save(directory / "manifest.json", dict(options=row["physical"]["options"],
            warning_count=1, shards=shards))
        runner.save(directory / "status.json", dict(status="complete", workflow_complete=True,
            returncode=0, error=None))
    actual = runner.aggregate_data(plan)
    assert actual["complete"] and len(actual["datasets"]) == 13
    assert len(actual["chains"]) == 156
    assert len({r["physical_chain_seed"] for r in actual["chains"]}) == 156
    assert len({r["condition_id"] for r in actual["chains"]}) == 24
    assert len([r for r in actual["chains"] if r["size"] == 36]) == 120
    assert len([r for r in actual["chains"] if r["size"] == 108]) == 36
    assert all(r["split"] == "evaluation" for r in actual["chains"])
    assert all(Path(d["root"]).is_absolute() for d in actual["datasets"])


def execution_fixture(tmp_path, monkeypatch):
    root = tmp_path / "new"
    root.mkdir()
    upstream = make_upstream(root)
    source = tmp_path / "source"
    (source / "run").mkdir(parents=True)
    (source / "data").mkdir()
    (source / "run/training_step_3000.pt").write_bytes(b"fixture-original-checkpoint")
    for path in ("run/manifest.json", "run/summary.json", "run/status.json", "data/manifest.json"):
        runner.save(source / path, {})
    monkeypatch.setattr(runner.socket, "gethostname", lambda: runner.HOST)
    monkeypatch.setattr(runner.os, "getpriority", lambda *args: 10)
    for name in runner.THREAD_VARIABLES:
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(runner, "verify_frozen", lambda *args: None)
    monkeypatch.setattr(runner, "verify_source_and_models", lambda *args: {})
    monkeypatch.setattr(runner, "resource_snapshot", lambda plan: dict(host_ready=True))
    return root, source, upstream


def test_child_nonzero_preserves_accurate_stage_and_never_retries(tmp_path, monkeypatch):
    root, source, upstream = execution_fixture(tmp_path, monkeypatch)
    calls = []
    class Child:
        pid = 123
        def __init__(self, command, **kwargs):
            calls.append(command)
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def poll(self): return 17
        def wait(self): return 17
    monkeypatch.setattr(runner.subprocess, "Popen", Child)
    with pytest.raises(RuntimeError, match="returncode 17"):
        runner.run(root, source, upstream)
    job = runner.read(root / "job.json")
    assert job["status"] == "failed" and not job["workflow_complete"]
    assert len(calls) == 1 and len(job["stages"]) == 1
    assert job["stages"][0]["returncode"] == 17
    assert (root / job["stages"][0]["log"]).exists()
    with pytest.raises(FileExistsError):
        runner.run(root, source, upstream)
    assert len(calls) == 1


def test_projection_hold_does_not_launch_sampling(tmp_path, monkeypatch):
    root, source, upstream = execution_fixture(tmp_path, monkeypatch)
    original = runner.build_plan
    def insufficient(*args, **kwargs):
        plan = original(*args, **kwargs)
        plan["budget_projection"]["budget_passed"] = False
        return plan
    monkeypatch.setattr(runner, "build_plan", insufficient)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs:
                        pytest.fail("No child is allowed when conservative budget fails"))
    with pytest.raises(runner.BudgetHold):
        runner.run(root, source, upstream)
    status = runner.read(root / "status.json")
    assert status["status"] == "budget_hold" and status["requires_user_action"]
