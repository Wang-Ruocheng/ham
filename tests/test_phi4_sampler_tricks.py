"""Independent checks of the finite diagnostic; no L36 physical pilot is run."""
import json
import math
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from scripts import diagnose_phi4_sampler_tricks as diagnostic


def test_ordinary_rhat_matches_between_and_within_variance_formula():
    # Chain means .5,2.5: W=.5, B=4; var_plus=2.25, Rhat=sqrt(4.5).
    x = np.array([[0., 1.], [2., 3.]])
    assert diagnostic._rhat(x) == pytest.approx(math.sqrt(4.5))
    assert diagnostic._rhat(np.ones((4, 8))) is None


def test_rank_split_rhat_detects_location_and_scale_nonmixing():
    rng = np.random.default_rng(616)
    base = rng.normal(size=(4, 512))
    equal = diagnostic.rank_split_rhat(base)
    assert .98 < equal["maximum"] < 1.02
    offset = base + np.array([0., 0., 4., 4.])[:, None]
    location = diagnostic.rank_split_rhat(offset)
    assert location["rank_normalized_split"] > 1.2
    scale = base * np.array([1., 1., 10., 10.])[:, None]
    spread = diagnostic.rank_split_rhat(scale)
    assert spread["folded_rank_normalized_split"] > 1.2
    assert spread["maximum"] == max(spread["rank_normalized_split"], spread["folded_rank_normalized_split"])


def test_rank_split_rhat_is_invariant_to_common_positive_affine_coordinate_change():
    rng = np.random.default_rng(511)
    values = rng.normal(size=(4, 64))
    first = diagnostic.rank_split_rhat(values)
    second = diagnostic.rank_split_rhat(17 * values + 200)
    for key in first:
        assert second[key] == pytest.approx(first[key], abs=1e-14)


def test_rank_split_uses_average_ties_and_reports_constants_unavailable():
    values = np.tile(np.repeat([-1., 0., 1., 2.], 4), (4, 1))
    result = diagnostic.rank_split_rhat(values)
    assert result["maximum"] is not None
    assert np.isfinite(result["maximum"])
    constant = diagnostic.rank_split_rhat(np.ones((4, 16)))
    assert constant == dict(rank_normalized_split=None, folded_rank_normalized_split=None, maximum=None)


@pytest.mark.parametrize("values", [np.ones((1, 16)), np.ones((4, 7)), np.ones(12),
                                    np.full((4, 16), np.nan), np.full((4, 16), np.inf)])
def test_rank_split_rejects_unavailable_or_nonfinite_data(values):
    with pytest.raises(ValueError):
        diagnostic.rank_split_rhat(values)


@pytest.fixture
def immutable_sources(monkeypatch):
    # Parent/source inventory checks are exercised without mutating any real source.
    values = {"/snapshot/diagnostic.py": "fixed-diagnostic-sha", "/snapshot/sampler.py": "fixed-sampler-sha"}
    monkeypatch.setattr(diagnostic, "source_fingerprints", lambda: values.copy())
    return values


def test_plan_prespecifies_methods_scopes_and_work_before_execution(tmp_path, immutable_sources):
    plan = diagnostic.make_plan()
    assert plan["size"] == 36 and plan["dimension"] == 2
    assert plan["u"] == .2 and plan["h"] == 0
    assert plan["samples"] == 256 and plan["burn_in"] == 128 and plan["spacing"] == 1
    assert plan["starts"] == ["hot", "hot", "cold_positive", "cold_negative"]
    assert plan["r_values"] == [-.5, .3, -.33158]
    assert plan["r_roles"][:2] == ["TRAIN端点", "TRAIN端点"]
    assert plan["r_roles"][2] == "预定压力验证"
    assert [row["name"] for row in plan["methods"]] == ["hmc_long_warmup", "hmc_fine_step", "hmc_wolff"]
    assert plan["cpu_budget_seconds"] == 600 and plan["wall_budget_seconds"] == 900
    assert plan["formal_sampling_started"] is False and plan["training_updates"] == 0
    assert plan["total_hmc_trajectories"] == 9 * 56 + 36 * 384
    assert plan["total_force_site_evaluations"] == (9 * 56 * 13 + 12 * 384 * (13 + 25 + 13)) * 1296
    assert plan["total_force_site_evaluations"] == 313_061_760
    assert plan["total_force_site_evaluations"] + plan["original_held_preflight_force_site_evaluations"] + plan["unchanged_production_force_site_evaluations"] <= 2_000_000_000
    assert plan["production_authorized_by_this_diagnostic"] is False
    assert plan["cluster_probability"] == .5
    assert plan["maximum_cluster_updates"] == 12 * 384
    assert plan["expected_cluster_updates"] == 6 * 384
    assert not any(tmp_path.iterdir())


def test_default_cli_is_plan_only_not_sampling(capsys, immutable_sources, monkeypatch):
    sampler = Mock(side_effect=AssertionError("No default sampling authorization"))
    monkeypatch.setattr(diagnostic, "execute", sampler)
    diagnostic.main([])
    value = json.loads(capsys.readouterr().out)
    assert value["training_updates"] == 0
    sampler.assert_not_called()


def fake_sampler_factory(root, calls):
    def sample(r, u, **kwargs):
        assert (root / "plan.json").is_file(), "Plan must be frozen before any RNG/data call"
        calls.append((r, u, kwargs.copy()))
        # Tiny diagnostic arrays, not L36 fields. Logical HMC call lengths remain
        # visible in metadata so dispatch/seed/work accounting can be checked.
        length = 16
        rng = np.random.default_rng(kwargs["seed"])
        raw = rng.normal(size=(length, len(diagnostic.OBSERVABLES)))
        rows = {name: dict(mean=float(raw[:, i].mean()), std=float(raw[:, i].std()),
                          ess=12., lag1=.1, warning=None)
                for i, name in enumerate(diagnostic.OBSERVABLES)}
        work = kwargs["burn_in"] + kwargs["samples"] * kwargs["spacing"]
        return dict(configurations=np.zeros((length, 3, 3, 1)), raw_observables=raw,
            accepted=np.ones(work, dtype=np.bool_), delta_h=np.zeros(work),
            seed=kwargs["seed"], initial=kwargs["initial"], diagnostics=rows,
            production_acceptance=.9, burn_in_acceptance=.8,
            saved_magnetization_sign_changes=5, hmc_trajectories=work,
            force_evaluations=work * (kwargs["leapfrog_steps"] + 1),
            max_absolute_delta_h=.1, warnings=[], cpu_seconds=.001, wall_seconds=.001)
    return sample


def install_tiny_execution(root, monkeypatch, immutable_sources):
    import recurrent_rg.data.phi4_hybrid as hybrid
    hmc_calls, cluster_calls = [], []
    monkeypatch.setattr(diagnostic, "sample_hmc_chain", fake_sampler_factory(root, hmc_calls))
    monkeypatch.setattr(hybrid, "sample_hybrid_chain", fake_sampler_factory(root, cluster_calls))
    # Figure rendering has separate tests; this checks that it is invoked only
    # after all predeclared chains and public-array serialization complete.
    def plot(output, report, *, font=None):
        assert len(report["groups"]) == 9
        assert (output / "observations.npz").is_file()
        (output / "sampling_comparison.png").write_bytes(b"tiny plot fixture")
    monkeypatch.setattr(diagnostic, "plot_diagnostic", plot)
    return hmc_calls, cluster_calls


def test_tiny_execution_preserves_scope_fixed_seeds_and_never_saves_fields(tmp_path, monkeypatch, immutable_sources):
    root = tmp_path / "new_diagnostic"
    historical = tmp_path / "historical_preflight.json"
    historical.write_text('{"status":"invalid_sampling_hold","original":true}\n')
    original_bytes = historical.read_bytes()
    hmc, cluster = install_tiny_execution(root, monkeypatch, immutable_sources)
    report = diagnostic.execute(root)
    status = json.loads((root / "status.json").read_text())
    assert status["status"] == "complete" and status["workflow_complete"] is True
    assert status["completed_chains"] == 36
    assert len(hmc) == 9 + 24 and len(cluster) == 12
    assert [row[2]["seed"] for row in hmc[:9]] == list(range(202610270200, 202610270209))
    assert report["formal_sampling_started"] is False and report["training_updates"] == 0
    assert report["equilibrium_certified"] is False
    assert historical.read_bytes() == original_bytes
    assert not (root / "data").exists() and not (root / "run").exists()
    assert not any(path.suffix == ".pt" for path in root.rglob("*"))
    with np.load(root / "observations.npz", allow_pickle=False) as saved:
        assert not any("configurations" in key for key in saved.files)
        assert all(f"{method['name']}_r{r}" in saved.files for method in diagnostic.METHODS for r in range(3))
    for r in diagnostic.R_VALUES:
        for chain, initial in enumerate(diagnostic.STARTS):
            expected_seed = diagnostic.seed_for(r, chain)
            matches = [row for row in hmc[9:] + cluster if row[0] == r and row[2]["seed"] == expected_seed]
            assert len(matches) == 3
            assert all(row[2]["initial"] == initial for row in matches)
    assert all(group["equilibrium_certified"] is False for group in report["groups"])


def test_existing_output_is_never_overwritten_or_resumed(tmp_path, monkeypatch, immutable_sources):
    root = tmp_path / "existing"
    root.mkdir()
    marker = root / "original.json"
    marker.write_text("preserve this existing diagnostic\n")
    sampler = Mock(side_effect=AssertionError("No retry permitted"))
    monkeypatch.setattr(diagnostic, "sample_hmc_chain", sampler)
    with pytest.raises(FileExistsError):
        diagnostic.execute(root)
    assert marker.read_text() == "preserve this existing diagnostic\n"
    sampler.assert_not_called()


def test_midrun_source_change_halts_without_modifying_original_or_sampling_again(tmp_path, monkeypatch, immutable_sources):
    root = tmp_path / "source_failure"
    import recurrent_rg.data.phi4_hybrid as hybrid
    calls = []
    ordinary = fake_sampler_factory(root, calls)
    def changes_after_first(*args, **kwargs):
        result = ordinary(*args, **kwargs)
        immutable_sources["/snapshot/sampler.py"] = "unexpected-new-sha"
        return result
    monkeypatch.setattr(diagnostic, "sample_hmc_chain", changes_after_first)
    monkeypatch.setattr(hybrid, "sample_hybrid_chain", Mock(side_effect=AssertionError("No continuation")))
    with pytest.raises(RuntimeError, match="fingerprint"):
        diagnostic.execute(root)
    assert len(calls) == 1
    status = json.loads((root / "status.json").read_text())
    assert status["status"] == "failed" and status["requires_user_action"]
    assert not status["workflow_complete"]


def test_fixed_budget_stops_without_automatic_continuation(tmp_path, monkeypatch, immutable_sources):
    root = tmp_path / "budget_stop"
    ticks = iter([0., 601., 602.])
    monkeypatch.setattr(diagnostic.time, "process_time", lambda: next(ticks))
    sampler = Mock(side_effect=AssertionError("Budget hold must precede sampling"))
    monkeypatch.setattr(diagnostic, "sample_hmc_chain", sampler)
    with pytest.raises(TimeoutError, match="budget"):
        diagnostic.execute(root)
    status = json.loads((root / "status.json").read_text())
    assert status["status"] == "budget_hold"
    assert status["returncode"] == 1  # execute propagates the diagnostic TimeoutError.
    assert status["requires_user_action"] and not status["workflow_complete"]
    sampler.assert_not_called()


def test_post_plot_source_mutation_cannot_receive_complete_status(tmp_path, monkeypatch, immutable_sources):
    root = tmp_path / "post_plot_source_change"
    install_tiny_execution(root, monkeypatch, immutable_sources)
    def altered(output, report, *, font=None):
        (output / "sampling_comparison.png").write_bytes(b"tiny plot fixture")
        immutable_sources["/snapshot/diagnostic.py"] = "changed-during-plot"
    monkeypatch.setattr(diagnostic, "plot_diagnostic", altered)
    with pytest.raises(RuntimeError, match="fingerprint"):
        diagnostic.execute(root)
    status = json.loads((root / "status.json").read_text())
    assert status["status"] == "failed" and status["workflow_complete"] is False
    assert (root / "partial_observations.npz").is_file()


def test_even_observable_warning_is_not_erased_by_many_m_sign_changes():
    rng = np.random.default_rng(414)
    observations = rng.normal(size=(4, 32, len(diagnostic.OBSERVABLES)))
    rows = []
    for chain in range(4):
        stats = {name: dict(ess=100., mean=float(observations[chain, :, i].mean()), warning=None)
                 for i, name in enumerate(diagnostic.OBSERVABLES)}
        stats["action_density"]["ess"] = 4.
        rows.append(dict(diagnostics=stats, cpu_seconds=1., production_acceptance=.99,
                         saved_magnetization_sign_changes=1000))
    result = diagnostic.group_summary(rows, observations)
    assert result["any_even_observable_warning"]
    assert result["equilibrium_certified"] is False
    assert result["observables"]["action_density"]["minimum_ess"] == 4.
