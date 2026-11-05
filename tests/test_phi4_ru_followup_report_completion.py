import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments import phi4_ru_followup_report_completion as report


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def inventory(root):
    save(root / "public_files.json", {str(p.relative_to(root)): report.base.sha(p)
        for p in root.rglob("*") if p.is_file() and p.name != "public_files.json"})


def failed_fixture(tmp_path):
    root = tmp_path / report.FAILED_ROOT_NAME
    subsets = [dict(name=f"subset_{index}") for index in range(13)]
    plan = dict(output=str(root), subsets=subsets, resource_policy=dict(load_gate_user_override=True))
    save(root / "plan.json", plan)
    stages = [dict(name=name, status="complete", returncode=0) for name in
              ["fine_sampler_pilot"] + [row["name"] for row in subsets]]
    stages.append(dict(name="frozen_evaluation", status="failed", returncode=1))
    job = dict(status="failed", workflow_complete=False, returncode=1, current="frozen_evaluation",
        error="RuntimeError: " + report.ERROR, stages=stages,
        cumulative_cpu_seconds=report.PREVIOUS_CPU_SECONDS,
        plan_sha256=report.base.sha(root / "plan.json"))
    save(root / "job.json", job)
    save(root / "status.json", {k: v for k, v in job.items() if k != "stages"})
    save(root / "data_manifest.json", dict(complete=True, independent_physical_chains=156,
        unique_size_parameter_conditions=24, chains=[{}] * 156))
    save(root / "evaluation/manifest.json", {})
    save(root / "evaluation/status.json", dict(status="failed", workflow_complete=False,
                                              returncode=1, error=report.ERROR))
    save(root / "evaluation/groups.json", [])
    np.savez(root / "evaluation/evaluation_arrays.npz", placeholder=np.array([1]))
    inventory(root)
    return root


def cached_fixture(tmp_path):
    root = tmp_path / "cached"
    (root / "evaluation").mkdir(parents=True)
    rng = np.random.default_rng(91)
    labels, chains, identities, groups = [], [], [], []
    for size, condition_count, replicas in ((36, 15, 2), (108, 9, 1)):
        for condition in range(condition_count):
            r, u = -1.3 + condition * .01, .5
            groups.append(dict(size=size, r=r, u=u, physical_chains=4*replicas,
                               models={key: {} for key in report.MODEL_KEYS}))
            for replica in range(replicas):
                subset = f"L{size}_c{condition}_rep{replica}"
                path = f"data/{subset}/sidecar.json"
                save(root / path, dict(chains=[dict(chain_index=index, seed=100+index,
                    production_acceptance=.9, warnings=["有限链热化警告"]) for index in range(4)]))
                for index in range(4):
                    identities.append(json.dumps([subset, condition, index]))
                    labels.append([size, r, u])
                    chains.append(dict(subset=subset, generator_cell_index=condition,
                        generator_chain_index=index, size=size, r=r, u=u, sidecar_path=path,
                        physical_chain_seed=100+index))
    direct = rng.normal(size=(4, 156, 3, 2))
    arrays = dict(direct=direct, feature_target=rng.normal(size=(4, 156, 3, 105)),
        moments=np.ones((156, 4)), labels=np.asarray(labels), identities=np.asarray(identities))
    for key in report.MODEL_KEYS:
        arrays[f"{key}_one_step"] = direct[:, :, 1:] + .01
        arrays[f"{key}_rollout"] = np.concatenate((direct[:, :, :1], direct[:, :, 1:] + .01), axis=2)
        arrays[f"{key}_at_chain_mean"] = direct.mean(0)[:, 1:] + .02
    np.savez(root / "evaluation/evaluation_arrays.npz", **arrays)
    save(root / "evaluation/groups.json", groups)
    return root, dict(chains=chains), arrays


def test_failed_contract_preserves_exact_serialization_failure(tmp_path):
    root = failed_fixture(tmp_path)
    pins, job, _, _ = report.verify_failed_root(root)
    assert job["cumulative_cpu_seconds"] == report.PREVIOUS_CPU_SECONDS
    assert str(root / "evaluation/evaluation_arrays.npz") in pins
    job["stages"][3]["returncode"] = 1
    save(root / "job.json", job)
    inventory(root)
    with pytest.raises(RuntimeError, match="report-only"):
        report.verify_failed_root(root)


def test_public_inventory_rejects_raw_data_before_hash(tmp_path):
    root = failed_fixture(tmp_path)
    files = report.base.read(root / "public_files.json")
    files["data/conditions/data.npz"] = "notread"
    save(root / "public_files.json", files)
    with pytest.raises(ValueError, match="Forbidden raw"):
        report.verify_failed_root(root)


def test_public_inventory_detects_tampering(tmp_path):
    root = failed_fixture(tmp_path)
    save(root / "evaluation/groups.json", ["changed"])
    with pytest.raises(RuntimeError, match="fingerprint"):
        report.verify_failed_root(root)


def test_decoder_only_exact_torch_column_normalization(tmp_path):
    weight = torch.arange(210, dtype=torch.float64).reshape(105, 2) / 111
    bias = torch.arange(105, dtype=torch.float64)
    state = {"Q.decoder.weight": weight, "Q.decoder.bias": bias}
    path = tmp_path / "source.pt"
    torch.save(dict(step=3000, manifest={"frozen": True}, model=state), path)
    decoder, actual_bias = report.decoder_parameters(path, {"frozen": True}, report.tensor_state_sha(state))
    np.testing.assert_array_equal(decoder, torch.nn.functional.normalize(weight, p=2, dim=0, eps=1e-15).numpy())
    np.testing.assert_array_equal(actual_bias, bias.numpy())
    with pytest.raises(RuntimeError, match="tensor fingerprint"):
        report.decoder_parameters(path, {"frozen": True}, "different")


def test_cached_loader_reuses_bootstrap_and_never_predicts(tmp_path, monkeypatch):
    root, data, saved = cached_fixture(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError("No inference or bootstrap is allowed")
    monkeypatch.setattr(report.evaluator, "predict", forbidden)
    monkeypatch.setattr(report.evaluator, "grouped_summary", forbidden)
    arrays, predictions, groups, warnings = report.load_cached(root, data)
    np.testing.assert_array_equal(arrays["direct"], saved["direct"])
    assert len(groups) == 24 and len(warnings) == 156
    assert set(predictions) == set(report.MODEL_KEYS)
    assert warnings[0]["warnings"] == ["有限链热化警告"]
    data["chains"][0]["generator_chain_index"] = 1
    with pytest.raises(ValueError, match="identity/order"):
        report.load_cached(root, data)


def test_output_must_be_fresh_before_any_input_read(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError, match="never overwrites"):
        report.build_plan(tmp_path / "source", tmp_path / "data", tmp_path / "validation", output)


def test_cached_completion_end_to_end_no_model_or_resampling(tmp_path, monkeypatch):
    old, data, saved = cached_fixture(tmp_path)
    save(old / "data_manifest.json", data)
    for name in ("job", "plan", "status"):
        save(old / f"{name}.json", dict(status="failed"))
    save(old / "evaluation/status.json", dict(status="failed", error=report.ERROR))
    source = tmp_path / "source"
    save(source / "run/manifest.json", {})
    train = np.arange(136*3*2, dtype=float).reshape(136, 3, 2)
    np.savez(source / "run/evaluation_arrays.npz", train_direct=train)
    output = tmp_path / "completed"
    pins = {str(old / "evaluation/evaluation_arrays.npz"): report.base.sha(old / "evaluation/evaluation_arrays.npz")}
    plan = dict(output=str(output), data_root=str(old), source_root=str(source),
                source_state_sha256="source", fixed_R_sha256={key: key for key in report.MODEL_KEYS}, input_sha256=pins)
    monkeypatch.setattr(report, "resource_contract", lambda: {})
    monkeypatch.setattr(report, "build_plan", lambda *a: plan)
    monkeypatch.setattr(report, "previous_live_processes", lambda *a: [])
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda *a: None)
    decoder, bias = np.zeros((105, 2)), np.zeros(105)
    monkeypatch.setattr(report, "decoder_parameters", lambda *a: (decoder, bias))
    def forbidden(*args, **kwargs):
        raise AssertionError("No inference or bootstrap is allowed")
    for name in ("predict", "grouped_summary", "restore_maps", "load_new_encodings"):
        monkeypatch.setattr(report.evaluator, name, forbidden)
    def plot(output, *args):
        (output / "physical_comparison.png").write_bytes(b"saved plot")
    monkeypatch.setattr(report.evaluator, "write_plots", plot)
    result = report.run(source, old, tmp_path / "validation", output)
    json.dumps(result, allow_nan=False)
    assert result["workflow_complete"] is True and result["model_forward_calls"] == 0
    assert len(result["latent_and_native_feature_metrics"]) == 6
    assert report.base.read(old / "status.json")["status"] == "failed"
    assert report.base.read(output / "status.json")["status"] == "complete"
    assert report.base.sha(output / "evaluation_arrays.npz") == pins[str(old / "evaluation/evaluation_arrays.npz")]
    assert result["cumulative_cpu_seconds"] >= report.PREVIOUS_CPU_SECONDS
    assert result["bootstrap_recomputed"] is False
    assert report.base.read(output / "completion_receipt.json")["inputs_unchanged"] is True


def test_active_original_process_is_never_interrupted(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "resource_contract", lambda: {})
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda *a: None)
    output = tmp_path / "new"
    monkeypatch.setattr(report, "build_plan", lambda *a: dict(output=str(output), data_root=str(tmp_path / "old")))
    monkeypatch.setattr(report, "previous_live_processes", lambda *a: [dict(pid=123)])
    with pytest.raises(RuntimeError, match="still has live"):
        report.run("source", "old", "validation", output)
    assert not output.exists()
