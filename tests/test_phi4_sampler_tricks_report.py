"""Tiny synthetic diagnostics: no physical sampling, models, or real reports."""
from __future__ import annotations

import json

import numpy as np
import pytest

from scripts import report_phi4_sampler_tricks as report


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _fixture(tmp_path):
    root = tmp_path / "diagnostic"
    root.mkdir()
    methods = [dict(name=name, label=label) for name, label in report.METHOD_LABELS.items()]
    r_values = [-.5, .3, -.33158]
    starts = ["hot", "hot", "cold_positive", "cold_negative"]
    names = ["action_density", "magnetization", "absolute_magnetization"]
    samples = 16
    plan = dict(schema="phi4_sampler_tricks_diagnostic_v1", methods=methods, r_values=r_values,
        starts=starts, samples=samples, spacing=1, burn_in=8, training_updates=0,
        formal_sampling_started=False, diagnostic_rhat_warning_threshold=1.05,
        minimum_chain_ess_warning=32, minimum_acceptance_warning=.6)
    arrays, chains, groups = {}, [], []
    for column, r in enumerate(r_values):
        for method_index, method in enumerate(methods):
            raw = np.empty((4, samples, len(names)))
            for chain in range(4):
                m = (1. if chain % 2 == 0 else -1.) * (1. + .1 * method_index + np.arange(samples) * .001)
                raw[chain, :, 0] = .5 + .01 * chain + .001 * np.arange(samples)
                raw[chain, :, 1] = m
                raw[chain, :, 2] = np.abs(m)
                diagnostics = {name: dict(mean=float(raw[chain, :, index].mean()),
                    ess=float(8 + chain), lag1=.5, iat_saved_intervals=1.) for index, name in enumerate(names)}
                chains.append(dict(method=method["name"], r=r, chain_index=chain, initial=starts[chain],
                    raw_observable_names=names, cpu_seconds=2. + chain, production_acceptance=.9,
                    saved_magnetization_sign_changes=0, diagnostics=diagnostics,
                    warnings=["保存的有限链诊断警告"], equilibrium_certified=False))
            arrays[f"{method['name']}_r{column}"] = raw
            arrays[f"{method['name']}_r{column}_chain0_accepted"] = np.ones(samples, dtype=bool)
            observables = {name: dict(chain_means=raw[:, :, index].mean(axis=1).tolist(),
                ess_per_chain=[float(8 + chain) for chain in range(4)],
                rank_split_rhat=dict(rank_normalized_split=1.1, folded_rank_normalized_split=1.2, maximum=1.2),
                hot_minus_cold=.3, hot_cold_three_se_warning=True, warning=True)
                for index, name in enumerate(names)}
            groups.append(dict(method=method["name"], r=r, observables=observables,
                hmc_acceptance_range=[.9, .9], sign_changes_per_chain=[0, 0, 0, 0],
                total_sampling_cpu_seconds=14., energy_ess_per_cpu_second=38 / 14,
                absolute_m_ess_per_cpu_second=38 / 14, any_even_observable_warning=True))
    _write(root / "plan.json", plan)
    _write(root / "report.json", dict(status="complete", groups=groups, spacing=1,
        equilibrium_certified=False, formal_sampling_started=False, training_updates=0,
        limitations=["保存的原始诊断限制"]))
    _write(root / "chains.json", chains)
    np.savez(root / "observations.npz", **arrays)
    _refresh_status(root)
    # Forbidden-looking unrelated payloads must never be loaded by this report.
    (root / "physical_configurations.npz").write_bytes(b"not observations")
    (root / "checkpoint.pt").write_bytes(b"not a model")
    return root, arrays


def _refresh_status(root):
    _write(root / "status.json", dict(status="complete", workflow_complete=True, returncode=0,
        error=None, requires_user_action=False, training_updates=0, formal_sampling_started=False,
        equilibrium_certified=False, completed_chains=36,
        artifacts_sha256={name: report.sha256(root / name) for name in report.INPUTS if name != "status.json"}))


def _source_hashes(root):
    return {name: report.sha256(root / name) for name in report.INPUTS}


def test_public_loader_preserves_nine_groups_and_all_independent_chains(tmp_path):
    root, _ = _fixture(tmp_path)
    public = report.load_public(root)
    assert len(public["arrays"]) == 9
    assert len(public["chains"]) == len(public["chain_map"]) == 36
    assert len(public["group_map"]) == 9
    assert all(value.shape == (4, 16, 3) for value in public["arrays"].values())
    comparison = report.comparison_table(public)
    assert not comparison["equilibrium_certified"]
    assert comparison["samples_per_chain"] == 16
    assert comparison["saved_chain_receipts"] == public["chains"]
    assert comparison["saved_group_summaries"] == public["report"]["groups"]
    assert all(row["any_even_observable_warning"] for row in comparison["rows"])
    first = comparison["rows"][0]["observables"]["action_density"]
    assert first["saved_group_diagnostics"]["rank_split_rhat"]["maximum"] == 1.2
    assert first["ess_per_cpu_second"] == pytest.approx(38 / 14)
    assert [row["ess_per_cpu_second"] for row in first["per_chain"]] == pytest.approx([8 / 2, 9 / 3, 10 / 4, 11 / 5])
    assert all(row["saved_chain_warnings"] for row in first["per_chain"])


def test_read_only_report_chinese_labels_and_no_overwrite(tmp_path, monkeypatch):
    root, _ = _fixture(tmp_path)
    before = _source_hashes(root)
    original_load = np.load
    def guarded_load(path, *args, **kwargs):
        assert str(path).endswith("/observations.npz")
        return original_load(path, *args, **kwargs)
    monkeypatch.setattr(np, "load", guarded_load)
    import matplotlib.figure
    import matplotlib.text
    original_savefig = matplotlib.figure.Figure.savefig
    saved_text = []
    def inspect_figure(self, *args, **kwargs):
        saved_text.extend(item.get_text() for item in self.findobj(matplotlib.text.Text))
        return original_savefig(self, *args, **kwargs)
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", inspect_figure)
    provenance, comparison = report.write_report(root)
    assert _source_hashes(root) == before
    assert provenance["public_sources_unchanged"]
    assert not any(provenance[key] for key in ("sampler_executed", "physical_configurations_loaded",
        "model_loaded", "GPU_used", "equilibrium_certified"))
    assert comparison["metrics_reestimated"] is False
    assert set(provenance["public_files"]) == {"comparison.json", "m_traces.png", "mixing_diagnostics.png", "report.json", "status.json"}
    for name, digest in provenance["outputs_sha256"].items():
        assert report.sha256(root / "analysis" / name) == digest
    text = "\n".join(saved_text)
    assert "四条独立链分别展示，未作链间平均" in text
    assert "链1：热初态" in text and "链4：负冷初态" in text
    assert "预设警告参考" in text and "每点为一条独立链" in text
    assert "不认证" not in text or "认证平衡" in text
    assert "Free Gaussian" not in text and "Training" not in text and "Monte Carlo" not in text
    with pytest.raises(FileExistsError):
        report.write_report(root)


@pytest.mark.parametrize("filename", ["plan.json", "report.json", "chains.json", "observations.npz"])
def test_completed_artifact_fingerprint_disagreement_rejected(tmp_path, filename):
    root, _ = _fixture(tmp_path)
    path = root / filename
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA256 disagrees"):
        report.load_public(root)


@pytest.mark.parametrize("field,value", [("status", "running"), ("workflow_complete", False),
                                       ("returncode", 1), ("requires_user_action", True)])
def test_incomplete_or_failed_execution_cannot_receive_final_plot(tmp_path, field, value):
    root, _ = _fixture(tmp_path)
    status = json.loads((root / "status.json").read_text())
    status[field] = value
    _write(root / "status.json", status)
    with pytest.raises(ValueError, match="successfully completed"):
        report.load_public(root)


def test_source_diagnostic_cannot_claim_equilibrium_or_training(tmp_path):
    root, _ = _fixture(tmp_path)
    status = json.loads((root / "status.json").read_text())
    status["equilibrium_certified"] = True
    _write(root / "status.json", status)
    with pytest.raises(ValueError, match="certified equilibrium"):
        report.load_public(root)
    status["equilibrium_certified"] = False
    status["training_updates"] = 1
    _write(root / "status.json", status)
    with pytest.raises(ValueError, match="zero training"):
        report.load_public(root)


def test_missing_or_duplicate_chain_identity_rejected(tmp_path):
    root, _ = _fixture(tmp_path)
    chains = json.loads((root / "chains.json").read_text())
    chains[-1] = chains[-2]
    _write(root / "chains.json", chains)
    _refresh_status(root)
    with pytest.raises(ValueError, match="Chain identities"):
        report.load_public(root)


def test_group_means_and_ess_must_match_public_arrays_and_chain_receipts(tmp_path):
    root, _ = _fixture(tmp_path)
    path = root / "report.json"
    value = json.loads(path.read_text())
    value["groups"][0]["observables"]["magnetization"]["chain_means"][0] += 1
    _write(path, value)
    _refresh_status(root)
    with pytest.raises(ValueError, match="chain means disagree"):
        report.load_public(root)
    value["groups"][0]["observables"]["magnetization"]["chain_means"][0] -= 1
    value["groups"][0]["observables"]["magnetization"]["ess_per_chain"][0] += 1
    _write(path, value)
    _refresh_status(root)
    with pytest.raises(ValueError, match="ESS receipts disagree"):
        report.load_public(root)


def test_observation_shape_and_nonfinite_values_rejected(tmp_path):
    root, arrays = _fixture(tmp_path)
    key = "hmc_long_warmup_r0"
    original = arrays[key].copy()
    arrays[key] = original[:3]
    np.savez(root / "observations.npz", **arrays)
    _refresh_status(root)
    with pytest.raises(ValueError, match="four independent chains"):
        report.load_public(root)
    arrays[key] = original + np.nan
    np.savez(root / "observations.npz", **arrays)
    _refresh_status(root)
    with pytest.raises(ValueError, match="finite numeric"):
        report.load_public(root)


def test_outputs_never_replace_diagnostic_source(tmp_path):
    root, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="contain or replace"):
        report.write_report(root, root)
    with pytest.raises(ValueError, match="contain or replace"):
        report.write_report(root, tmp_path)


def test_cli_accepts_explicit_new_output_without_running_sampler(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(report, "write_report", lambda root, output, font=None:
                        (calls.append((root, output, font)) or {"status": "complete"}, {}))
    assert report.main(["--root", str(tmp_path), "--output", str(tmp_path / "new")]) == 0
    assert calls[0][:2] == (tmp_path, tmp_path / "new")

