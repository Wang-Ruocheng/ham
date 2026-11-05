import json

import numpy as np
import pytest

from experiments.phi4_ru_followup_evaluate import (
    evaluation_seed, grouped_summary, path_metrics, physical_metrics, restore_maps,
)


def test_seed_streams_independent_of_labels_and_repeat():
    identity = ["new_dataset", 0, 0]
    seeds = {evaluation_seed(identity, r, s) for r in range(4) for s in ("subset", "C")}
    assert len(seeds) == 8
    assert evaluation_seed(identity, 0, "C") != evaluation_seed(["other", 0, 0], 0, "C")
    with pytest.raises(ValueError):
        evaluation_seed(identity, 4, "C")


def test_binder_uses_pooled_moments_and_actual_physical_volume():
    moments = np.array([[.1, .2, .5, .8], [-.1, .8, 1., 2.]])
    pooled = moments.mean(0)
    values = physical_metrics(pooled, 108)
    assert values[1] == pytest.approx(1-pooled[3]/(3*pooled[2]**2))
    assert values[2] == pytest.approx(108**2 * pooled[2])
    assert values[3] == pytest.approx(108**2 * (pooled[2]-pooled[1]**2))
    assert values[1] != pytest.approx(physical_metrics(moments, 108)[:,1].mean())


def test_metrics_reference_target_physical_size_and_no_averaging_before_square():
    direct = np.zeros((4, 4, 3, 2))
    direct[:,:,1] = 1
    direct[:,:,2] = 2
    one = direct[:,:,1:].copy()
    # Zero-mean repeat error must remain in the raw objective metric.
    one[::2] += .1
    one[1::2] -= .1
    prediction = {"one_step":one,"rollout":direct.copy()}
    labels = np.array([[108,-1,.5]]*4)
    features = np.zeros((4,4,3,105))
    metrics = path_metrics(direct,prediction,labels,{36:2.,12:4.,4:8.},
                           np.zeros((105,2)),np.zeros(105),features)
    assert [m["target_reference_lattice"] for m in metrics] == [36,12,12]
    assert metrics[0]["rmse"] == pytest.approx(.1)
    assert metrics[0]["chain_mean_rmse"] == pytest.approx(0.,abs=1e-15)
    assert metrics[0]["nrmse"] == pytest.approx(.05)
    assert metrics[1]["nrmse"] == pytest.approx(.025)
    assert all(type(row["target_reference_lattice"]) is int for row in metrics)
    # Exercise the actual failing boundary, not just numerical equality of sizes.
    assert json.loads(json.dumps(metrics, allow_nan=False)) == metrics


def test_bootstrap_keeps_models_scales_and_chains_paired():
    rng=np.random.default_rng(12)
    direct=rng.normal(size=(4,8,3,2))
    labels=np.array([[36,-1,.5]]*4+[[108,-1,.5]]*4)
    moments=np.array([[.1,.4,.5,.8]]*8)
    one=direct[:,:,1:].copy()+.2
    arrays=dict(labels=labels,direct=direct,moments=moments)
    predictions={"a":dict(one_step=one),"b":dict(one_step=one.copy())}
    rows=grouped_summary(arrays,predictions,bootstrap=100)
    assert len(rows)==2
    for row in rows:
        assert row["physical_chains"]==4
        assert row["models"]["a"]==row["models"]["b"]
        np.testing.assert_allclose(row["models"]["a"]["mean_residual"],.2)


def test_restored_states_match_fixed_selection_and_reject_tampering(tmp_path):
    import json
    import torch
    from experiments.phi4_qcr import state_fingerprint
    from recurrent_rg.phi4_r_calibration import FrozenResidualMap
    arrays, expected = {}, {}
    for arm in ("selected", "adam6000"):
        for seed in (91, 92, 93):
            torch.manual_seed(seed + (100 if arm == "selected" else 0))
            model = FrozenResidualMap(2,96)
            key=f"{arm}_seed{seed}"
            expected[key]=state_fingerprint(model)
            arrays.update({f"{key}.{name}":v.detach().numpy().copy() for name,v in model.state_dict().items()})
    (tmp_path/"summary.json").write_text(json.dumps({"final_R_sha256":expected}))
    np.savez(tmp_path/"R_parameters.npz",**arrays)
    restored=restore_maps(tmp_path)
    assert {k:state_fingerprint(m) for k,m in restored.items()}==expected
    arrays["selected_seed91.net.0.bias"][0] += .01
    np.savez(tmp_path/"R_parameters.npz",**arrays)
    with pytest.raises(RuntimeError,match="fingerprint mismatch"):
        restore_maps(tmp_path)
