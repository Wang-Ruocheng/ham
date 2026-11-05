import json

import numpy as np
import pytest
import torch

from recurrent_rg.phi4_r_calibration import FrozenResidualMap
from scripts import report_phi4_scalar_r_validation as report


class Affine:
    def __init__(self, slope, intercept):
        self.slope, self.intercept = slope, intercept

    def forward_and_derivative(self, points):
        return self.slope * points + self.intercept, np.full((len(points), 1, 1), self.slope)


@pytest.mark.parametrize("slope,expected", [(.4, "attracting"), (1.5, "repelling"),
                                           (-1.8, "repelling")])
def test_affine_roots_use_full_discrete_DR_and_modulus(slope, expected):
    roots, receipt = report.find_roots(Affine(slope, (1 - slope) * .2), [-1., 1.])
    assert len(roots) == 1
    assert roots[0]["g"] == pytest.approx(.2, abs=1e-9)
    assert roots[0]["classification"] == expected
    assert roots[0]["full_DR"] == slope
    assert roots[0]["thermal_nu"] is None
    assert not roots[0]["physical_critical_point_certified"]
    assert receipt["full_DR_includes_identity"]
    json.dumps(dict(roots=roots, receipt=receipt), allow_nan=False)


def test_no_root_and_identity_continuum_are_not_invented_isolated_roots():
    roots, receipt = report.find_roots(Affine(1., .2), [-1., 1.])
    assert roots == [] and not receipt["nonisolated_or_flat_identity"]
    roots, receipt = report.find_roots(Affine(1., 0.), [-1., 1.])
    assert roots == [] and receipt["nonisolated_or_flat_identity"]


def test_bounds_ignore_target_scale_and_use_TRAIN_only():
    values = np.array([[[[-.5], [.2], [100.]], [[.4], [.7], [-999.]]]])
    bounds, observed = report.train_bounds(values)
    np.testing.assert_allclose(observed, [-.5, .7])
    np.testing.assert_allclose(bounds, [-.62, .82])
    with pytest.raises(ValueError, match="非退化"):
        report.train_bounds(np.zeros((2, 3, 1)))


def test_numpy_restore_matches_torch_and_full_derivative_no_grad_mutation(tmp_path):
    torch.manual_seed(913)
    original = FrozenResidualMap(1, 5, center=[.2], scale=[.03]).eval().requires_grad_(False)
    state = {key: value.detach().numpy().copy() for key, value in original.state_dict().items()}
    expected = report.state_fingerprint(state)
    key = "selected_seed91"
    path = tmp_path / "R_parameters.npz"
    np.savez(path, **{f"{key}.{name}": value for name, value in state.items()})
    restored = report.FrozenScalarMap.restore(path, key, expected)
    points = np.linspace(-.1, .5, 73)[:, None]
    output, full_dr = restored.forward_and_derivative(points)
    x = torch.tensor(points, requires_grad=True, dtype=torch.float64)
    y = original.advance(x)
    dr = torch.autograd.grad(y.sum(), x)[0]
    np.testing.assert_allclose(output, y.detach().numpy(), rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(full_dr[:, :, 0], dr.numpy(), rtol=1e-12, atol=1e-13)
    restored.assert_unchanged()
    assert all(parameter.grad is None for parameter in original.parameters())
    assert all(not value.flags.writeable for value in restored.state.values())
    state["net.0.bias"][0] += .1
    with pytest.raises(ValueError, match="指纹"):
        report.FrozenScalarMap(state, expected)


def test_cached_prediction_contract_checks_every_split_and_rejects_mismatch():
    torch.manual_seed(2)
    original = FrozenResidualMap(1, 3)
    state = {key: value.detach().numpy().copy() for key, value in original.state_dict().items()}
    model = report.FrozenScalarMap(state, report.state_fingerprint(state))
    key, arrays = "selected_seed91", {}
    rng = np.random.default_rng(7)
    for split in ("fit", "validation", "test"):
        direct = rng.normal(size=(4, 8, 3, 1))
        arrays[f"{split}_direct"] = direct
        predictions, _ = model.forward_and_derivative(direct[..., :2, :].reshape(-1, 1))
        arrays[f"eval_{key}_{split}_one_step"] = predictions.reshape(4, 8, 2, 1)
    assert report.check_cached_predictions(model, arrays, key) == 0.
    arrays[f"eval_{key}_test_one_step"][0, 0, 0, 0] += .01
    with pytest.raises(ValueError, match="不一致"):
        report.check_cached_predictions(model, arrays, key)


def test_public_input_pin_and_completion_gate(tmp_path):
    report.write_json(tmp_path / "manifest.json", dict(theory_dim=1))
    report.write_json(tmp_path / "status.json", dict(status="complete", workflow_complete=True,
                                                     returncode=0, error=None))
    report.write_json(tmp_path / "public_files.json",
                      {"manifest.json": report.sha256(tmp_path / "manifest.json")})
    pins = {}
    report.verified_public(tmp_path, ("manifest.json",), pins)
    assert len(pins) == 3
    report.write_json(tmp_path / "manifest.json", dict(theory_dim=2))
    with pytest.raises(ValueError, match="SHA"):
        report.verified_public(tmp_path, ("manifest.json",), {})


def test_completed_report_uses_original_bounds_and_sha_pins_status(tmp_path):
    cache, validation, output = [tmp_path / name for name in ("cache", "validation", "report")]
    cache.mkdir()
    validation.mkdir()
    rng = np.random.default_rng(51)
    arrays, cache_arrays = {}, {}
    for split, cells in (("fit", np.arange(6)), ("validation", np.arange(6, 8)),
                         ("test", np.arange(8, 13))):
        ids = np.repeat(cells, 4)
        arrays[f"{split}_cell_ids"] = ids
        arrays[f"{split}_labels"] = np.column_stack([-.5 + ids * .01, np.ones(len(ids))])
        # Deliberately wider than immutable source bounds: a 64-field repeat
        # must not silently redefine the original 128-field TRAIN interval.
        arrays[f"{split}_direct"] = rng.uniform(-1.3, 1.3, size=(4, len(ids), 3, 1))
    cache_arrays["eval_train_direct"] = np.concatenate(
        [arrays[f"{split}_direct"] for split in ("fit", "validation")], axis=1)
    grid = np.linspace(-1.2, 1.2, 512)[:, None]
    cache_arrays.update(flow_g=grid, source_flow_R=grid.copy(), source_flow_DR=np.ones((512, 1, 1)))
    np.savez(cache / "cache.npz", **cache_arrays)
    cache_manifest = dict(train_input_bounds=[-1., 1.], flow_padding_fraction=.1,
        source_state_sha256="a" * 64, source_module_sha256={key: key * 64 for key in ("Q", "C", "R")})
    report.write_json(cache / "manifest.json", cache_manifest)
    summary = dict(all_seeds_selected_before_TEST_and_labels_loaded=True,
        selected_parameter_keys=[f"selected_seed{s}" for s in report.SEEDS], final_R_sha256={})
    parameters, records = {}, {}
    for seed in report.SEEDS:
        torch.manual_seed(seed)
        model = FrozenResidualMap(1, 3, center=[0.], scale=[.3]).eval()
        state = {name: value.detach().numpy().copy() for name, value in model.state_dict().items()}
        digest = report.state_fingerprint(state)
        frozen = report.FrozenScalarMap(state, digest)
        for endpoint in ("adam6000", "selected"):
            key = f"{endpoint}_seed{seed}"
            summary["final_R_sha256"][key] = digest
            parameters.update({f"{key}.{name}": value for name, value in state.items()})
            for split in ("fit", "validation", "test"):
                direct = arrays[f"{split}_direct"]
                values, _ = frozen.forward_and_derivative(direct[..., :2, :].reshape(-1, 1))
                arrays[f"eval_{key}_{split}_one_step"] = values.reshape(direct.shape[:-2] + (2, 1))
        records[str(seed)] = dict(history={stage: [dict(step=0, fit_loss=.2,
            validation=dict(chain_mean_loss=.3)), dict(step=1, fit_loss=.1,
            validation=dict(chain_mean_loss=.2))] for stage in ("adam", "lbfgs")})
    report.write_json(validation / "manifest.json", dict(cache_manifest=cache_manifest,
        source_state_sha256=cache_manifest["source_state_sha256"],
        source_module_sha256=cache_manifest["source_module_sha256"], theory_dim=1))
    report.write_json(validation / "summary.json", summary)
    report.write_json(validation / "history.json", records)
    np.savez(validation / "R_parameters.npz", **parameters)
    np.savez(validation / "evaluation_arrays.npz", **arrays)
    for root in (cache, validation):
        report.write_json(root / "status.json", dict(status="complete", workflow_complete=True,
            returncode=0, error=None))
        report.write_json(root / "public_files.json", {path.name: report.sha256(path)
            for path in root.iterdir() if path.is_file()})
    result = report.run(validation, cache, output)
    assert result["original_TRAIN_input_interval"] == [-1., 1.]
    assert result["search_interval"] == [-1.2, 1.2]
    assert result["frozen_parameters_unchanged"]
    assert result["parameter_gradients_created"] is False
    assert result["Q_or_C_loaded"] is False
    inventory = json.loads((output / "public_files.json").read_text())
    assert inventory["status.json"] == report.sha256(output / "status.json")
    assert all((output / name).stat().st_size > 1000 for name in
               ("R_vs_g.png", "coupling_by_scale.png", "loss.png"))
    with pytest.raises(FileExistsError):
        report.run(validation, cache, output)
