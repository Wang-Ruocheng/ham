import hashlib
import json

import pytest

from scripts.compare_phi4_hopping_ising_losses import canonical_history, metrics, run


def phi_row(step=200):
    return dict(step=step, q_reconstruction=.02, q_subset_micro=.003, q_subset_coarse=.004,
                c_information=.3, c_variance=.01, c_uniform=.02,
                closure_start_0_target_1=.1, closure_start_1_target_2=.2, closure_start_0_target_2=.3,
                closure_one_step=.15, closure_rollout=.2, q_total=.28, c_total=-.56, r_total=1.75)


def ising_row(step=200):
    return dict(epoch=step, train_separation=.02, train_anchor=.03, train_coarse_consistency=.004,
                train_rsmi=-.3, train_kernel_noncollapse=.01, train_kernel_sign=.02,
                train_rollout_level_1=.1, train_rollout_level_2=.3, train_commutator=.15,
                train_rollout=.2, train_total=1.47, train_qcr_total=2.07)


def test_mapping_and_exact_closure_equivalence():
    phi = canonical_history([phi_row()], "phi4")["points"][0]
    ising = canonical_history([ising_row()], "ising")["points"][0]
    assert phi["weighted"] == pytest.approx(ising["weighted"])
    assert phi["totals"] == pytest.approx(dict(Q=.28, C=-.56, R=1.75))
    assert phi["totals"]["R"] == pytest.approx(5 * .15 + 5 * .2)


@pytest.mark.parametrize("kind,key", [("phi4", "q_reconstruction"), ("ising", "train_rsmi")])
def test_nonfinite_rejected(kind, key):
    row = phi_row() if kind == "phi4" else ising_row()
    row[key] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        canonical_history([row], kind)


def test_bad_derived_e12_rejected():
    row = ising_row()
    row["train_commutator"] = .01
    with pytest.raises(ValueError, match="Negative squared-error"):
        canonical_history([row], "ising")


def test_total_validation():
    row = phi_row()
    row["q_total"] += .1
    with pytest.raises(ValueError, match="Inconsistent objective"):
        canonical_history([row], "phi4")


def test_no_interpolation_and_prefix_only():
    series = canonical_history([phi_row(1), phi_row(200), phi_row(3000), phi_row(3200)], "phi4")
    assert [point["step"] for point in series["points"]] == [1, 200, 3000]
    assert metrics(series)["exact_logged_steps"]["1000"] is None
    with pytest.raises(ValueError, match="Exact 200-step"):
        metrics(canonical_history([phi_row(1)], "phi4"))


def test_only_tiny_rounding_is_clipped():
    row = phi_row()
    row.update(c_variance=-1e-14, c_total=-.58 - 2e-14)
    series = canonical_history([row], "phi4")
    assert series["rounding_clips"][0]["component"] == "c_variance"


def test_no_input_writes_and_fresh_output(tmp_path, monkeypatch):
    paths = [tmp_path / name for name in ("phi.json", "phi_manifest.json", "ising.json", "ising_manifest.json")]
    values = [[phi_row()], {}, {"history": [ising_row()]}, {}]
    for path, value in zip(paths, values):
        path.write_text(json.dumps(value))
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    monkeypatch.setattr("scripts.compare_phi4_hopping_ising_losses.plot_comparison", lambda *_: [])
    report = run(*paths, tmp_path / "out")
    assert report["status"] == "complete"
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths} == before
    assert (tmp_path / "out/comparison.json").exists()
    with pytest.raises(FileExistsError):
        run(*paths, tmp_path / "out")


def test_per_scale_reconstruction_kept_and_ratios():
    rows = [phi_row(200), phi_row(1000), phi_row(3000)]
    for index, row in enumerate(rows):
        row.update({f"reconstruction_level_{level}": (index + 1) * (level + 1) * .01 for level in range(3)})
    report = metrics(canonical_history(rows, "phi4"))
    assert report["reconstruction_by_level_final_over_1000"] == pytest.approx([1.5, 1.5, 1.5])


def test_signed_C_total_is_not_reported_as_a_descent_ratio():
    report = metrics(canonical_history([phi_row(200), phi_row(3000)], "phi4"))
    endpoint = report["exact_logged_steps"]["3000"]
    assert endpoint["total_ratio_to_200"]["C"] is None
    assert endpoint["signed_C_total_delta_from_200"] == 0.


def test_zero_variance_has_an_explicit_zero_tick():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scripts.compare_phi4_hopping_ising_losses import _scale
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0., 1e-4])
    _scale(ax, [[0., 1e-4]])
    assert ax.get_ylim()[0] == 0.
    assert 0. in ax.get_yticks()
    plt.close(fig)
