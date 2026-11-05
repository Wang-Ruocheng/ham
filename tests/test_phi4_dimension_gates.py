import pytest

from scripts import run_phi4_fixed_c_dimension as runner


def closure_summary(a=2., b=1., c=2.):
    scores = {}
    for fold in range(4):
        for seed in (91, 92, 93):
            for arm, loss in (("A_shared96", a), ("B_separate96", b), ("C_shared136", c)):
                scores[f"fold{fold}_{arm}_seed{seed}_selected"] = {
                    "validation": {"chain_mean_loss": loss}}
    return {"closure_scores": scores}


def test_representation_gate_requires_both_comparators(monkeypatch):
    monkeypatch.setattr(runner, "completed", lambda _: (closure_summary(), {"public": "pin"}))
    evidence, pins = runner.representation_gate()
    assert pins == {"public": "pin"}
    assert all(row["passed"] and row["improved_pairs"] == 12 for row in evidence.values())
    # Beating only the wider shared map cannot establish the proposed signal.
    monkeypatch.setattr(runner, "completed", lambda _: (closure_summary(a=.5), {}))
    with pytest.raises(ValueError, match="gate"):
        runner.representation_gate()


def test_replication_gate_uses_validation_common_features_only(monkeypatch):
    metrics = {}
    for seed in (91, 92, 93):
        for dimension, rmse in ((1, 2.), (2, 1.)):
            metrics[f"dg{dimension}_seed{seed}_selected"] = {
                "validation": [{"prediction": {"chain_mean_rmse": rmse}} for _ in range(3)]}
    monkeypatch.setattr(runner, "completed", lambda _: ({"feature_metrics": metrics}, {}))
    evidence, _ = runner.replication_gate()
    assert evidence["passed"] and evidence["median_ratio"] == .25
    for seed in (91, 92, 93):
        for row in metrics[f"dg2_seed{seed}_selected"]["validation"]:
            row["prediction"]["chain_mean_rmse"] = 2.
    with pytest.raises(ValueError, match="gate"):
        runner.replication_gate()
