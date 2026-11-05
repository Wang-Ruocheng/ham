"""Synthetic execution checks, not a physical or critical-exponent result."""
from dataclasses import replace
import json

import torch

from experiments import rsmi_separate_qcr as driver
from recurrent_rg.config import DataConfig, LossConfig, ModelConfig, RunConfig, TrainingConfig, EvaluationConfig


def test_reverse_weights_preserve_qc_initialization_rng_and_routing(tmp_path):
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(7101)
    payload = dict(schema_version=1)
    for split, group in (("train", 0), ("test", 100)):
        spins = (2 * torch.randint(2, (2, 4, 108, 108, 1), generator=generator) - 1).to(torch.int8)
        payload[split] = dict(configurations=spins, couplings=None,
                              group_ids=torch.tensor([group, group]))
    cache = tmp_path / "synthetic.pt"
    torch.save(payload, cache)
    paired = dict(schema_version=1, data_sha256=driver.file_fingerprint(cache))
    for split, offset in (("train", 0), ("test", 100)):
        paired[split] = dict(condition_ids=[offset, offset], chain_ids=[offset, offset + 1], pairs=[[0, 1]])
    pairs = tmp_path / "pairs.json"
    pairs.write_text(json.dumps(paired))
    config = RunConfig(data=DataConfig(cache=str(cache), physical_model="synthetic"),
        model=ModelConfig(dimension=2, theory_dim=1, hidden=5, block_factor=3, affine_reconstruction=True),
        loss=LossConfig(closure_objective='one_step_and_rollout'),
        training=TrainingConfig(seed=91, max_rg_steps=2, epochs=2, system_batch=2, sample_batch=4,
            threads=1, checkpoint_every=1, log_every=1, gradient_mode="qc_r_separate"),
        evaluation=EvaluationConfig(max_rg_steps=2, samples=4, systems=2))
    states = {}
    for name, weights in (("equal", None), ("late", (.2, 1.8)), ("early", (1.8, .2))):
        current = replace(config, loss=replace(config.loss, closure_scale_weights=weights))
        path, output = tmp_path / (name + ".json"), tmp_path / name
        path.write_text(json.dumps(current.to_dict()))
        driver.main(["--config", str(path), "--output", str(output), "--skip-final-report",
                     "--q-future-control", "--q-future-pairs", str(pairs)])
        saved = torch.load(output / "checkpoint_last.pt", map_location="cpu", weights_only=True)
        assert saved["epoch"] == 2
        metadata = saved["metadata"]
        assert metadata["labels_used_in_training"] is False
        assert metadata["auxiliary_future_parameters"] == 0
        assert metadata["q_future_physics"]["weight"] == 0
        if weights is not None:
            contract = metadata["closure_scale_weighting"]
            assert contract["weights"] == list(weights)
            assert contract["application"] == ["one_step_commutator", "microscopic_start_rollout"]
            torch.testing.assert_close(torch.tensor(contract["normalized_weights"]),
                                       torch.tensor(weights) / sum(weights))
        assert "future_physics_predictor_state" not in saved
        states[name] = saved
    reference = states["equal"]
    # The historical unspecified-weight protocol did not record initial SHAs;
    # do not fabricate that metadata just to make a three-arm assertion.
    assert states["early"]["metadata"]["initial_module_sha256"] == states["late"]["metadata"]["initial_module_sha256"]
    for name in ("late", "early"):
        candidate = states[name]
        for key, value in reference["model"].items():
            if key.startswith(("theory_encoder.", "rg_kernel.")):
                torch.testing.assert_close(value, candidate["model"][key], rtol=0, atol=1e-13)
        for key in ("torch_rng_state", "batch_generator_state", "rsmi_generator_state"):
            assert torch.equal(reference[key], candidate[key])
        assert any(not torch.equal(value, candidate["model"][key]) for key, value in reference["model"].items()
                   if key.startswith("theory_rg."))
    assert any(not torch.equal(value, states["late"]["model"][key])
               for key, value in states["early"]["model"].items() if key.startswith("theory_rg."))
