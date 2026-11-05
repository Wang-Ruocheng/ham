"""Tiny random-field implementation checks, not physical experiment results."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments import phi4_hopping_qcr as train
from recurrent_rg.phi4_aligned_qcr import AlignedPhi4QCR
from test_phi4_hopping_qcr_training import data_root


def test_explicit_new_config_keeps_old_policy_and_counts(tmp_path):
    path=Path(__file__).resolve().parents[1]/"configs/phi4_hopping_b2_L128_g1_seed91.json"
    options=train.TrainingOptions(**train.load_config(path))
    plan=train.training_plan(options)
    assert (options.block_size,options.depth)==(2,3)
    assert plan["model"]["patch_chunk_size"]==262144
    assert train.MODEL_POLICY["patch_chunk_size"]==4096
    assert plan["options"]["steps"]==3000
    assert plan["options"]["system_batch"]==32
    assert plan["options"]["sample_batch"]==64
    assert plan["implementation_chunks"]["C_blocks"]==65536


def test_four_scale_frozen_evaluation_has_all_six_pairs():
    torch.set_num_threads(1)
    model=AlignedPhi4QCR(1.,hidden=4,block_size=2,steps=3,patch_chunk_size=64)
    direct=torch.randn(2,4,24,24,1,dtype=torch.float64)
    dataset=SimpleNamespace(configurations=direct,train_indices=np.array([0]),
        labels=np.array([[.5,1.,0.],[.8,1.,0.]]),splits=np.array(["train","test"]),cell_ids=np.array([0,1]))
    before=train.state_fingerprint(model)
    arrays=train.evaluate_model(model,dataset,torch.device("cpu"))
    assert train.state_fingerprint(model)==before
    assert arrays["direct_g"].shape==(2,4,1)
    assert arrays["one_step_g"].shape==(2,3,1)
    assert arrays["all_start_g"].shape==(2,6,1)
    assert arrays["all_start_pairs"].tolist()==[[0,1],[0,2],[1,2],[0,3],[1,3],[2,3]]
    for column,(source,target) in enumerate(arrays["all_start_pairs"]):
        if target==source+1:
            np.testing.assert_allclose(arrays["all_start_g"][:,column],arrays["one_step_g"][:,source])
        if source==0:
            np.testing.assert_allclose(arrays["all_start_g"][:,column],arrays["rollout_g"][:,target])


def test_tiny_b2_full_workflow_one_update_and_chinese_report(data_root,tmp_path):
    # Rebind the synthetic 9x9 fixture to a legal, tiny 24→12→6→3 geometry.
    plan=json.loads((data_root/"plan.json").read_text())
    manifest=json.loads((data_root/"manifest.json").read_text())
    for value in (plan,manifest):value["options"]["sizes"]=[24]
    train.write_json(data_root/"plan.json",plan)
    for cell,shard in enumerate(manifest["shards"]):
        shard["size"]=24
        path=data_root/shard["path"]
        np.savez(path,configurations=np.random.default_rng(102+cell).normal(size=(4,4,24,24,1)))
        shard["sha256"]=train.sha256_file(path)
        sidepath=data_root/shard["sidecar"]
        side=json.loads(sidepath.read_text());side.update(size=24,data_sha256=shard["sha256"])
        train.write_json(sidepath,side);shard["sidecar_sha256"]=train.sha256_file(sidepath)
    manifest["plan_sha256"]=train.sha256_file(data_root/"plan.json")
    train.write_json(data_root/"manifest.json",manifest)
    status=json.loads((data_root/"status.json").read_text())
    status["manifest_sha256"]=train.sha256_file(data_root/"manifest.json")
    train.write_json(data_root/"status.json",status)
    output=tmp_path/"fresh_b2"
    summary=train.run_training(train.TrainingOptions(data=str(data_root),output=str(output),
        device="cpu",steps=1,system_batch=2,sample_batch=4,information_pairs=4,
        checkpoint_every=1,log_every=1,max_training_seconds=300.,block_size=2,depth=3))
    assert summary["actual_steps"]==1 and summary["lattice_sizes"]==[24,12,6,3]
    assert all(clock["min"]==clock["max"]==1 for clock in summary["optimizer_steps"].values())
    assert len(summary["closure"]["test"]["all_start"])==6
    status=json.loads((output/"status.json").read_text())
    assert status["status"]=="complete" and status["returncode"]==0
    for name in ("R_vs_g.png","coupling_by_scale.png","loss_curves.png"):
        assert (output/"report"/name).is_file()
