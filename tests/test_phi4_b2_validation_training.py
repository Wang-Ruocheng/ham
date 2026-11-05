"""CPU smoke checks of scientific/RNG contracts, never scientific results.

The small in-memory fixture declares a synthetic source checkpoint identity
so the immutable-source loading path can be exercised. It is not physical
data, a 3000-step training run, or an equilibrium/fixed-point certificate.
"""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments import phi4_b2_validation as training
from experiments.phi4_qcr import FieldDataset
from recurrent_rg.phi4_b2_qcr import continuous_objectives_b2
from recurrent_rg.phi4_b2_validation import ValidationModel
from scripts import report_phi4_b2_validation as report


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def rng(seed=17):
    return torch.Generator().manual_seed(seed)


def read(path):
    return json.loads(path.read_text())


def synthetic_dataset():
    return FieldDataset(
        configurations=torch.randn(6, 4, 32, 32, 1, dtype=torch.double, generator=rng(904)),
        splits=np.asarray(["train"] * 4 + ["test"] * 2),
        cell_ids=np.asarray([0, 0, 1, 1, 2, 2]),
        labels=np.column_stack(([-.5, -.5, .3, .3, -.2, -.2], [.2] * 6, [0.] * 6)),
        manifest_sha256="5" * 64,
        receipts=[], warning_count=35,
        qualification="Synthetic fixture only; not physical sampling",
        sampler_schema="synthetic_validation_cpu_test")


def source_initial(dataset):
    torch.manual_seed(91)
    baseline = ValidationModel(dataset.scale, variant="baseline")
    constructor = training.common_fingerprints(baseline)
    training.initialize_decoder(baseline, dataset, torch.device("cpu"), 91, lambda: None)
    initial = training.common_fingerprints(baseline)
    public = SimpleNamespace(manifest={"random_construction_module_sha256": constructor,
                                       "initial_module_sha256": initial})
    return baseline, public


def test_fresh_joint_exact_source_initialization_and_common_rng_without_test_labels():
    dataset = synthetic_dataset()
    baseline, public = source_initial(dataset)
    expected_rng = torch.get_rng_state().clone()

    class TrainingOnly:
        configurations = dataset.configurations
        train_indices = dataset.train_indices
        scale = dataset.scale

        @property
        def labels(self):
            raise AssertionError("Physical labels entered joint initialization")

        @property
        def test_indices(self):
            raise AssertionError("TEST entered joint initialization")

    models, receipt, initial = training.fresh_joint(
        TrainingOnly(), torch.device("cpu"), 91, public, lambda: None)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert initial == training.common_fingerprints(baseline)
    assert receipt["labels_used"] is receipt["test_data_used"] is False
    assert receipt["C_updates"] == 0
    assert set(models) == {"cubic_decoder", "odd_C"}
    for name, model in models.items():
        assert training.common_fingerprints(model) == initial
        for key, value in baseline.state_dict().items():
            assert torch.equal(model.state_dict()[key], value), f"{name}/{key}"
        assert all(p.grad is None and p.requires_grad for p in model.parameters())
    assert torch.equal(models["odd_C"].Q.decoder.bias, baseline.Q.decoder.bias)


@pytest.mark.parametrize("key", ("random_construction_module_sha256", "initial_module_sha256"))
def test_fresh_joint_rejects_wrong_historical_initialization(key):
    dataset = synthetic_dataset()
    _, public = source_initial(dataset)
    public.manifest[key]["Q"] = "f" * 64
    with pytest.raises(RuntimeError, match="pinned baseline"):
        training.fresh_joint(dataset, torch.device("cpu"), 91, public, lambda: None)


def test_tail_two_adam_steps_match_full_historical_objective_and_rng_exactly():
    torch.manual_seed(91)
    actual = ValidationModel(scale=1.3, hidden=8)
    reference = copy.deepcopy(actual)
    for model in (actual, reference):
        model.Q.requires_grad_(False)
        model.C.requires_grad_(False)
    actual_optimizer = torch.optim.Adam(actual.R.parameters(), lr=.001)
    reference_optimizer = torch.optim.Adam(reference.R.parameters(), lr=.001)
    actual_rng, reference_rng = rng(293), rng(293)
    initial = training.module_fingerprints(actual)
    for index in range(2):
        batch = torch.randn(2, 4, 8, 8, 1, dtype=torch.double, generator=rng(index))
        expected_loss = continuous_objectives_b2(reference, batch, reference_rng, 8)[2]
        expected_loss.backward()
        expected_gradients = [p.grad.clone() for p in reference.R.parameters()]
        reference_optimizer.step()
        reference_optimizer.zero_grad(set_to_none=True)
        metrics = training.tail_step(actual, batch, actual_optimizer, actual_rng, 8)
        assert metrics["r_loss"] == expected_loss.item()
        assert metrics["closure_all_start_anchor"] == expected_loss.item()
        assert torch.equal(actual_rng.get_state(), reference_rng.get_state())
        for parameter, expected, reference_parameter in zip(
                actual.R.parameters(), expected_gradients, reference.R.parameters()):
            assert torch.equal(parameter.grad, expected)
            assert torch.equal(parameter, reference_parameter)
        assert all(p.grad is None for module in (actual.Q, actual.C) for p in module.parameters())
    final = training.module_fingerprints(actual)
    assert all(initial[key] == final[key] for key in ("Q", "C"))
    assert initial["R"] != final["R"]
    assert all(state["step"].item() == 2 for state in actual_optimizer.state.values())


def test_tail_does_not_evaluate_information_proxy(monkeypatch):
    import recurrent_rg.phi4_b2_qcr as core

    def forbidden(*args, **kwargs):
        raise AssertionError("MI is not an R-only training objective")

    monkeypatch.setattr(core, "smoothed_information_b2", forbidden)
    model = ValidationModel(scale=1.3, hidden=8)
    model.Q.requires_grad_(False)
    model.C.requires_grad_(False)
    batch = torch.randn(2, 4, 8, 8, 1, dtype=torch.double, generator=rng())
    loss, metrics = training.r_tail_objective(model, batch, rng(), 8)
    assert torch.isfinite(loss)
    assert all("information" not in key for key in metrics)


@pytest.fixture
def tiny_source(tmp_path):
    """Two actual CPU updates, synthetic source identity, no formal result claim."""
    root = tmp_path / "source"
    root.mkdir()
    dataset = synthetic_dataset()
    baseline, public = source_initial(dataset)
    options = training.Options(source=str(root), output=str(tmp_path / "unused"),
        stage="joint", device="cpu", steps=2, system_batch=2, sample_batch=4, information_pairs=4)
    loop = tmp_path / "baseline_loop"
    loop.mkdir()
    _, _, _, schedule, _, _ = training.optimize(
        {"baseline": baseline}, dataset, torch.device("cpu"), options,
        loop, lambda: None, lambda *a, **k: None, profile=True)
    baseline.eval().requires_grad_(False)
    endpoint = training.evaluate(baseline, dataset, torch.device("cpu"), lambda: None)
    training.save_flow(baseline, endpoint, torch.device("cpu"))
    final = training.module_fingerprints(baseline)
    public.manifest.update(data_manifest_sha256=dataset.manifest_sha256,
                           paired_schedule_sha256=schedule)
    public.summary = {"initial_module_sha256": public.manifest["initial_module_sha256"],
        "arms": {"all_start": {"final_module_sha256": final}}}
    public.endpoints = {"all_start": endpoint}
    # Source checkpoint fields are shape/provenance contract fixtures, not a
    # claim that this synthetic source performed 3000 scientific updates.
    checkpoint = root / "synthetic_source_checkpoint.pt"
    torch.save({"step": 3000, "closure_objective": "all_start_anchor",
                "model": baseline.state_dict()}, checkpoint)
    for name, value in (("manifest.json", public.manifest), ("summary.json", public.summary),
                        ("history_progress.json", {"synthetic_actual_steps": 2}),
                        ("status.json", {"synthetic_fixture": True})):
        training.write_json(root / name, value)
    np.savez_compressed(root / "evaluation_arrays.npz", **endpoint)
    public.fingerprints = {name: training.sha256_file(root / name)
                          for name in report.INPUT_FILES}
    pins = {str(root / name): digest for name, digest in public.fingerprints.items()}
    pins[str(checkpoint)] = training.sha256_file(checkpoint)
    return root, public, dataset, pins, checkpoint, schedule


def test_frozen_tail_rejects_changed_checkpoint_and_freezes_every_qc_parameter(tiny_source, tmp_path):
    _, public, dataset, _, checkpoint, _ = tiny_source
    model = training.frozen_tail(dataset, torch.device("cpu"), checkpoint, public)
    assert all(not p.requires_grad and p.grad is None for module in (model.Q, model.C)
               for p in module.parameters())
    assert all(p.requires_grad for p in model.R.parameters())
    assert not model.Q.training and not model.C.training
    payload = training.load_cpu_tensors(checkpoint)
    payload["model"]["R.0.bias"] += .1
    bad = tmp_path / "changed.pt"
    torch.save(payload, bad)
    with pytest.raises(RuntimeError, match="fingerprints"):
        training.frozen_tail(dataset, torch.device("cpu"), bad, public)


@pytest.mark.parametrize("stage", ("joint", "r_tail"))
def test_two_step_cpu_run_public_artifact_schema_and_endpoint_pairing(tiny_source, tmp_path, monkeypatch, stage):
    root, public, dataset, pins, checkpoint, schedule = tiny_source
    monkeypatch.setattr(training, "source_context", lambda source: (
        root, public, dataset, pins.copy(), checkpoint))
    output = tmp_path / stage
    result = training.run(training.Options(source=str(root), output=str(output),
        stage=stage, device="cpu", steps=2, system_batch=2, sample_batch=4, information_pairs=4))
    manifest, summary, status, history = [read(output / name) for name in (
        "manifest.json", "summary.json", "status.json", "history_progress.json")]
    assert result == summary
    assert summary["actual_steps"] == status["actual_steps"] == 2
    assert status["status"] == "complete" and status["workflow_complete"] is True
    assert status["returncode"] == 0 and not status["requires_user_action"]
    assert manifest["paired_schedule_sha256"] == summary["paired_schedule_sha256"] == schedule
    assert history["latest_step"] == history["latest_checkpoint_step"] == 2
    assert summary["source_inputs_unchanged"] is True
    assert summary["labels_used_in_training"] is False
    assert manifest["fixed_encoding_cache_used"] is False
    assert manifest["stochastic_raw_physical_batches"] is True
    assert manifest["tail_MI_computed"] is False
    for key, filename in (("manifest_sha256", "manifest.json"), ("summary_sha256", "summary.json"),
                          ("history_progress_sha256", "history_progress.json"),
                          ("evaluation_arrays_sha256", "evaluation_arrays.npz")):
        assert status[key] == training.sha256_file(output / filename)
    names = report.STAGE_ARMS[stage]
    assert set(manifest["arms"]) == set(summary["arms"]) == set(names)
    with np.load(output / "evaluation_arrays.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    for name in names:
        local = {key[len(name) + 1:]: value for key, value in arrays.items() if key.startswith(name + "_")}
        report._validate_arrays(local, name)
        metrics = report._metrics(local)
        assert len(metrics["latent_closure"]["rows"]) == 12
        assert len(metrics["native_statistic_prediction"]["rows"]) == 12
        assert len(metrics["q_reconstruction"]["rows"]) == 8
        assert summary["arms"][name]["critical_r"] is None
        assert summary["arms"][name]["thermal_nu"] is None
        assert summary["arms"][name]["root_search_performed"] is False
        for module in ("Q", "C", "R"):
            expected = 0 if stage == "r_tail" and module in ("Q", "C") else 2
            assert summary["arms"][name][f"{module}_updates"] == expected
            counts = summary["arms"][name]["optimizer_steps"][module]
            assert counts["min"] == counts["max"] == expected
        payload = training.load_cpu_tensors(output / name / "training_step_2.pt")
        assert payload["step"] == 2 and payload["variant"] == name
    if stage == "joint":
        assert summary["cubic_C_target_source_equal"] is True
    else:
        assert summary["Q_C_frozen_verified"] is summary["r_tail_QC_source_equal"] is True
    assert all(training.sha256_file(path) == digest for path, digest in pins.items())
    # The formal report must not silently relabel this smoke result as 3000
    # updates. Only schemas/array calculations are exercised above.
    with pytest.raises(ValueError, match="3000-step"):
        report.load_stage(output, stage, public)


@pytest.mark.parametrize("options", [dict(stage="profile", steps=2, seed=191),
                                     dict(stage="profile", steps=20, seed=91),
                                     dict(device="cuda:1"), dict(sample_batch=3),
                                     dict(device="cuda:2", expected_gpu_uuid=None)])
def test_options_prevent_wrong_profile_or_resource_contract(tmp_path, options):
    with pytest.raises(ValueError):
        training.Options(**{**dict(source="unused", output=str(tmp_path / "out"), device="cpu"),
                            **options}).validate()


@pytest.mark.parametrize("variant", ("baseline", "cubic_decoder", "odd_C"))
def test_augmented_evaluation_observes_existing_fields_only_with_identical_outputs_and_rng(monkeypatch, variant):
    dataset=synthetic_dataset()
    torch.manual_seed(91)
    model=ValidationModel(dataset.scale,hidden=8,variant=variant)
    before=training.state_fingerprint(model)
    global_rng=torch.get_rng_state().clone()
    calls={key:0 for key in ("C","Q","R")}
    recorded=[]
    hooks=[]
    for key in calls:
        def count(module,args,result,key=key):
            calls[key]+=1
            if key=="C":recorded.append(result.detach().clone())
        hooks.append(getattr(model,key).register_forward_hook(count))
    existing_hooks=set(model.C._forward_hooks)
    random_states=[]
    original_randn=torch.randn

    def track_random_draws(*args,**kwargs):
        result=original_randn(*args,**kwargs)
        if kwargs.get("generator") is not None:
            random_states.append(kwargs["generator"].get_state().clone())
        return result

    monkeypatch.setattr(torch,"randn",track_random_draws)
    original=training.evaluate(model,dataset,torch.device("cpu"),lambda:None)
    reference_calls=calls.copy()
    reference_states=list(random_states)
    reference_fields=list(recorded)
    for key in calls:calls[key]=0
    recorded.clear();random_states.clear()
    augmented=training.evaluate_validation(model,dataset,torch.device("cpu"),lambda:None)
    assert calls==reference_calls
    assert calls["C"]==3*len(dataset.configurations)
    assert len(random_states)==len(reference_states)==calls["C"]
    assert all(torch.equal(a,b) for a,b in zip(random_states,reference_states))
    assert all(torch.equal(a,b) for a,b in zip(recorded,reference_fields))
    assert set(model.C._forward_hooks)==existing_hooks
    assert torch.equal(torch.get_rng_state(),global_rng)
    assert training.state_fingerprint(model)==before
    assert all(p.grad is None for p in model.parameters())
    assert all(np.array_equal(value,augmented[key]) for key,value in original.items())
    expected_new={f"{split}_{key}" for split in ("train","test")
                  for key in ("field_mean","field_rms","field_mirror_l1")}
    assert set(augmented)-set(original)==expected_new
    for split in ("train","test"):
        indices=np.flatnonzero(dataset.splits==split)
        for local,index in enumerate(indices):
            fields=[dataset.configurations[index]]+reference_fields[3*index:3*index+3]
            expected=[]
            for value in fields:
                flat=value.numpy().reshape(-1)
                logits=-.5*((flat[:,None]/dataset.scale-np.arange(-2,3))/.5)**2
                weights=np.exp(logits-logits.max(axis=1,keepdims=True))
                distribution=(weights/weights.sum(axis=1,keepdims=True)).mean(axis=0)
                expected.append([flat.mean(),np.sqrt(np.mean(flat**2)),
                                 np.abs(distribution-distribution[::-1]).sum()])
            expected=np.asarray(expected)
            for column,name in enumerate(("field_mean","field_rms","field_mirror_l1")):
                actual=augmented[f"{split}_{name}"]
                assert actual.shape==(len(indices),4) and actual.dtype==np.float64
                np.testing.assert_allclose(actual[local],expected[:,column],rtol=1e-12,atol=2e-14)
            assert np.all(augmented[f"{split}_field_mirror_l1"]>=0)
            assert np.all(augmented[f"{split}_field_mirror_l1"]<=2+1e-14)
    for handle in hooks:handle.remove()


def test_augmented_evaluation_hook_is_removed_after_failure(monkeypatch):
    dataset=synthetic_dataset()
    model=ValidationModel(dataset.scale,hidden=8)
    existing=model.C.register_forward_hook(lambda *args:None)
    before=set(model.C._forward_hooks)

    def failed_evaluation(model,dataset,device,budget):
        model.C(dataset.configurations[:1],generator=rng())
        raise RuntimeError("Synthetic evaluation failure")

    monkeypatch.setattr(training,"evaluate",failed_evaluation)
    with pytest.raises(RuntimeError,match="Synthetic evaluation failure"):
        training.evaluate_validation(model,dataset,torch.device("cpu"),lambda:None)
    assert set(model.C._forward_hooks)==before
    existing.remove()


def test_augmented_evaluation_rejects_wrong_existing_call_count_and_cleans_hook(monkeypatch):
    dataset=synthetic_dataset()
    model=ValidationModel(dataset.scale,hidden=8)
    before=set(model.C._forward_hooks)
    monkeypatch.setattr(training,"evaluate",lambda *args:{})
    with pytest.raises(RuntimeError,match="existing C calls"):
        training.evaluate_validation(model,dataset,torch.device("cpu"),lambda:None)
    assert set(model.C._forward_hooks)==before
