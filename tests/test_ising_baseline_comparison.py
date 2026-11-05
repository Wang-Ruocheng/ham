"""Small synthetic integration checks; no physical cache or accelerator."""
import copy
import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments import ising_baseline_comparison as worker
from recurrent_rg.baselines.readout import fit_affine_readout, fit_train_pca


@pytest.fixture(autouse=True)
def small_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fields(n=5, m=64, length=27):
    return torch.randint(2, (n,m,length,length), generator=worker.generator(9),
                         dtype=torch.int8)*2-1


@pytest.mark.parametrize("method", worker.METHODS)
def test_native_C_collection_readout_R_end_to_end(method):
    x = fields()
    before = x.clone()
    table = torch.full((512,), .7, dtype=torch.float64)
    model = worker.construct_C(method, seed=91, device="cpu", source_table=table)
    history, elapsed = worker.native_C_training(x, method, model, updates=2,
        seed=91, device="cpu")
    assert elapsed >= 0
    if model is not None:
        assert all(p.grad is None and not p.requires_grad for p in model.parameters())
        c_before = worker.tensor_sha(model)
    stats, _ = worker.collect_statistics(x, method, model, draws=2, seed=100,
        device="cpu", source_table=table)
    assert stats["f512"].shape == (5,2,3,512)
    assert stats["p16"].shape == (5,2,3,16)
    np.testing.assert_allclose(stats["f512"].sum(-1), 1, atol=1e-12)
    np.testing.assert_allclose(stats["p16"].sum(-1), 1, atol=1e-12)
    # Three microscopic copies are deterministic estimator repeats, not new ensembles.
    np.testing.assert_array_equal(stats["f512"][:,0,0],stats["f512"][:,1,0])
    pca = fit_train_pca(stats["f512"][:3],dimension=5)
    g = pca.transform(stats["f512"])
    decoder = fit_affine_readout(g[:3],stats["p16"][:3])
    flow, r_history, _ = worker.train_R(g[:3],updates=3,seed=91,device="cpu")
    arrays, metrics = worker.evaluate(flow,g,decoder,stats,3,device="cpu")
    assert arrays["rollout"].shape == g.shape
    np.testing.assert_array_equal(arrays["rollout"][:,:,0],g[:,:,0])
    assert metrics["train"]["physical_chain_count"] == 3
    assert metrics["test"]["physical_chain_count"] == 2
    assert metrics["test"]["latent"]["rollout_rmse"] >= 0
    assert metrics["test"]["motif16"]["no_probability_clipping"]
    if model is not None:
        assert worker.tensor_sha(model) == c_before
        assert all(p.grad is None for p in model.parameters())
    torch.testing.assert_close(x,before)


def test_closure_uniform_scale_sum_coordinates_and_live_rollout_gradient():
    class Twice(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.factor = torch.nn.Parameter(torch.tensor(2., dtype=torch.float64))
        def forward(self,x):
            return self.factor*x
    flow = Twice()
    data = torch.ones(2,3,3,5,dtype=torch.float64)
    total, one, recurrent = worker.closure_objective(flow,data)
    # One-step residual is 1 in each of five dims; recurrent residuals 1 and 3.
    assert float(one) == 5
    assert float(recurrent) == 25
    assert float(total) == 150
    total.backward()
    assert flow.factor.grad is not None and float(flow.factor.grad) == 375


def test_training_boundary_has_no_label_inputs_and_is_repeatable():
    assert "labels" not in inspect.signature(worker.native_C_training).parameters
    assert "labels" not in inspect.signature(worker.train_R).parameters
    x = fields(n=2)
    a = worker.construct_C("rsmine",seed=91,device="cpu")
    b = worker.construct_C("rsmine",seed=91,device="cpu")
    worker.native_C_training(x,"rsmine",a,updates=2,seed=91,device="cpu")
    worker.native_C_training(x,"rsmine",b,updates=2,seed=91,device="cpu")
    assert worker.tensor_sha(a) == worker.tensor_sha(b)


def test_source_C_probability_table_matches_actual_architecture_without_mutation(monkeypatch):
    import recurrent_rg.checkpoints as checkpoints
    from recurrent_rg.config import ModelConfig
    from recurrent_rg.baselines.statistics import coarsen_tensor
    torch.manual_seed(19)
    model = checkpoints.build_model(ModelConfig(dimension=2,block_factor=3,
        hidden=7,theory_dim=5,affine_reconstruction=True),2).double()
    before = worker.tensor_sha(model)
    monkeypatch.setattr(checkpoints,"load_run",lambda checkpoint:SimpleNamespace(model=model))
    table, recorded = worker.source_C_table("synthetic_checkpoint")
    assert table.shape == (512,) and recorded == before == worker.tensor_sha(model)
    assert all(p.grad is None for p in model.parameters())
    # Exact pattern enumeration and actual model block logits give the same probabilities.
    x = fields(n=2,m=4,length=9).double()
    blocks = model.geometry.extract_blocks(x[...,None]).reshape(2,4,3,3,9)
    with torch.no_grad():
        logits = model.rg_kernel.field_head.logits(model.rg_kernel.local_encoder(blocks)).squeeze(-1)
        expected = torch.sigmoid(2*logits)
    codes = sum((x[...,r::3,c::3]>0).long()*(1<<bit)
                for bit,(r,c) in enumerate((r,c) for r in range(3) for c in range(3)))
    torch.testing.assert_close(table[codes],expected)
    got = coarsen_tensor(x,"qcr_c_pca",probabilities=table,generator=worker.generator(8))
    assert got.shape == (2,4,3,3)


def test_same_R_initialization_and_train_fit_test_apply_only():
    a = worker.build_r(5,96,seed=91)
    b = worker.build_r(5,96,seed=91)
    assert worker.tensor_sha(a) == worker.tensor_sha(b)
    frequency = np.random.default_rng(2).dirichlet(np.ones(512),size=(6,2,3))
    pca = fit_train_pca(frequency[:4])
    before = copy.deepcopy(pca.to_dict())
    pca.transform(frequency[4:])
    assert pca.to_dict() == before
    with pytest.raises(ValueError,match="TRAIN"):
        fit_train_pca(frequency[4:],split="test")


def test_full_5D_root_retains_all_negative_and_complex_spectrum():
    class Linear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.matrix = torch.nn.Parameter(torch.tensor(
                [[.5,-.2,0,0,0],[.2,.5,0,0,0],[0,0,-2,0,0],
                 [0,0,0,.25,0],[0,0,0,0,1.5]],dtype=torch.float64),requires_grad=False)
        def forward(self,x):
            return x@self.matrix.T
    data = np.random.default_rng(5).normal(size=(5,2,3,5))
    roots = worker.find_roots(Linear(),data,maximum_starts=4)
    assert len(roots) == 1
    assert roots[0]["expanding"] == 2
    assert roots[0]["contracting"] == 3
    assert roots[0]["thermal_nu"] is None
    assert not roots[0]["physical_support_certified"]
    spectrum = roots[0]["full_DR_eigenvalues"]
    assert len(spectrum) == 5 and any(e["real"] < 0 for e in spectrum)
    assert any(abs(e["imag"]) > 0 for e in spectrum)


def test_manifest_explicit_adaptations_and_no_default_recipe_override():
    plan = dict(seed=91,data_sha256="abc",source_data="cache.pt",C_draws=3,
                source_checkpoint="reference.pt",source_checkpoint_sha256="def")
    for method in worker.METHODS:
        receipt = worker.manifest(plan,method)
        assert receipt["labels_used_in_training"] is False
        assert receipt["common_readout"]["target_scale_weights"] == [1,1]
        assert receipt["common_readout"]["coordinate_reduction"] == "sum"
        assert "NOT original-paper" in receipt["scientific_claim"]
        assert receipt["thermal_nu"] is None


def test_actual_budget_hold_does_not_call_it_training_failure(tmp_path):
    worker.save(tmp_path/"job.json",dict(gpu_elapsed_seconds=10,cpu_seconds=3))
    plan = dict(device="cpu",resource_policy=dict(gpu_seconds=20,cpu_seconds=15,
                                                   gpu_peak_bytes=1024))
    worker.budget_guard(plan,tmp_path,elapsed=5,used_cpu=2)
    with pytest.raises(worker.BudgetHold):
        worker.budget_guard(plan,tmp_path,elapsed=11,used_cpu=2)


def test_saved_reference_adapter_does_not_invent_motif_metrics():
    original = dict(closure=dict(test=dict(rollout=dict(rmse=[0,.1,.2],nrmse=[0,.3,.4]))))
    before = copy.deepcopy(original)
    adapted = worker.reference_indicators(original)
    assert adapted["test"]["latent"]["rollout_nrmse"] == .4
    assert adapted["test"]["motif16"] == {}
    assert original == before


def test_profile_startup_is_charged_once_without_dropping_real_updates():
    history = [dict(step=1,elapsed_seconds=42.),dict(step=40,elapsed_seconds=42.195)]
    timing = worker.project_updates_with_single_startup(42.195,history,40,12000)
    assert timing["target_updates"] == 12000
    assert timing["startup_charged_once_per_arm"]
    assert timing["scientific_protocol_changed"] is False
    assert timing["steady_seconds_per_update"] == pytest.approx(.005)
    assert timing["projected_seconds"] == pytest.approx(42+11999*.005)
    assert timing["projected_seconds"] >= 42.195


@pytest.mark.parametrize("first,last,total,measured,target",[
    (3,2,4,40,12000),(1,4,3,40,12000),(-1,2,3,40,12000),
    (1,2,float("nan"),40,12000),(1,2,3,1,12000),(1,2,3,40,20)])
def test_profile_rejects_ambiguous_or_invalid_timing(first,last,total,measured,target):
    history=[dict(step=1,elapsed_seconds=first),dict(step=measured,elapsed_seconds=last)]
    with pytest.raises(ValueError):
        worker.project_updates_with_single_startup(total,history,measured,target)


def test_profile_steady_update_cost_is_not_artificially_reduced():
    history=[dict(step=1,elapsed_seconds=.01),dict(step=40,elapsed_seconds=.4)]
    timing=worker.project_updates_with_single_startup(.4,history,40,12000)
    assert timing["projected_seconds"] == pytest.approx(120.)
    # Synchronization outside the final logged update also enters the forecast.
    charged=worker.project_updates_with_single_startup(.41,history,40,12000)
    assert charged["projected_seconds"] > timing["projected_seconds"]


def test_formal_geometry_is_strict():
    with pytest.raises(ValueError,match="shape"):
        worker.validate_data_geometry(fields(),fields())


@pytest.mark.parametrize("column_labels", [False, True])
def test_entire_arm_and_comparison_public_reports_only_synthetic(tmp_path,monkeypatch,column_labels):
    """Real mmap cache loading through all five tiny arms and Chinese reports.

    Both legitimate scalar label shapes and both cache schemas are covered;
    only the production geometry/iteration counts and bounded root search are
    replaced. No real physical cache, reference checkpoint or GPU is opened.
    """
    from scripts.report_ising_zh import select_chinese_font
    try:
        select_chinese_font()
    except RuntimeError:
        pytest.skip("No preinstalled Chinese font")
    tx,vx = fields(n=6),fields(n=4)
    labels = (torch.tensor([.2,.2,.3,.3,.5,.5],dtype=torch.float64),
              torch.tensor([.25,.25,.45,.45],dtype=torch.float64))
    if column_labels:
        labels = tuple(k[:,None] for k in labels)
    cache = tmp_path/"synthetic.pt"
    payload = (dict(schema_version=1,
        train=dict(configurations=tx[...,None],couplings=labels[0]),
        test=dict(configurations=vx[...,None],couplings=labels[1])) if column_labels else
        (tx[...,None],labels[0],vx[...,None],labels[1]))
    torch.save(payload,cache)
    cache_before = cache.read_bytes()
    # Only the strict production geometry check is replaced for the synthetic test.
    monkeypatch.setattr(worker,"validate_data_geometry",lambda *args:None)
    monkeypatch.setattr(worker,"C_UPDATES",2)
    monkeypatch.setattr(worker,"R_UPDATES",2)
    monkeypatch.setattr(worker,"find_roots",lambda *args,**kwargs:[])
    monkeypatch.setattr(worker,"source_C_table",lambda checkpoint:
        (torch.full((512,),.7,dtype=torch.float64),"synthetic_source_sha"))
    worker.save(tmp_path/"job.json",dict(gpu_elapsed_seconds=0,cpu_seconds=0))
    reference = tmp_path/"reference.json"
    worker.save(reference,dict(closure=dict(test=dict(rollout=dict(rmse=[0,.1,.2],nrmse=[0,.3,.4])))))
    plan = dict(seed=91,data_sha256="synthetic",source_data=str(cache),C_draws=3,
                source_checkpoint="reference.pt",source_checkpoint_sha256="fake",
                source_metrics=str(reference),device="cpu",resource_policy=dict(
                    gpu_seconds=1000,cpu_seconds=1000,gpu_peak_bytes=1<<30))
    for method in worker.METHODS:
        status = worker.run_arm(plan,tmp_path,method)
        assert status["status"] == "complete"
        assert status["R_step"] == 2
        assert status["C_step"] == (2 if method in ("nmcrg","rsmine") else 0)
        folder = tmp_path/"arms"/method
        assert all((folder/name).is_file() for name in status["public_files"])
        assert "common_R.pt" not in status["public_files"]
        saved = worker.read(folder/"metrics.json")
        assert saved["test"]["physical_chain_count"] == 4
        assert not saved["fixed_points"]
        assert (folder/"history_progress.json").is_file() and (folder/"history.json").is_file()
        with np.load(folder/"evaluation_arrays.npz",allow_pickle=False) as data:
            np.testing.assert_array_equal(data["labels"][:6],labels[0].reshape(-1).numpy())
            assert data["g"].shape == (10,3,3,5)
            assert data["r_of_C_mean"].shape == (10,2,5)
    assert cache.read_bytes() == cache_before
    complete = worker.compare(plan,tmp_path)
    assert complete["status"] == "complete"
    assert all((tmp_path/"comparison"/name).is_file() for name in complete["public_files"])
    with pytest.raises(FileExistsError):
        worker.run_arm(plan,tmp_path,"majority")


@pytest.mark.parametrize("container", ["torch", "numpy"])
def test_evaluation_label_adapter_accepts_both_shapes_and_does_not_alias(container):
    labels = np.array([.2,.4,.6],dtype=np.float64)
    if container == "torch":
        labels = torch.tensor(labels,requires_grad=True)
    a = worker.evaluation_K_labels(labels,3)
    b = worker.evaluation_K_labels(labels.reshape(3,1),3)
    np.testing.assert_array_equal(a,b)
    assert a.shape == (3,) and a.dtype == np.float64
    a[0] = 99
    assert float(labels[0]) == pytest.approx(.2)
    if container == "torch":
        assert labels.grad is None


@pytest.mark.parametrize("labels,count", [
    (None,3), (np.ones((3,2)),3), (np.ones((3,1,1)),3), (np.ones(2),3),
    (np.array([True,False,True]),3), (np.ones(3,dtype=complex),3),
    (np.array([".1",".2",".3"]),3), (np.array([1,2,3],dtype=object),3),
    (np.array([1,np.nan,3]),3), (np.array([1,np.inf,3]),3),
    (torch.ones(3,dtype=torch.bool),3), (torch.ones(3,dtype=torch.complex128),3),
    (torch.tensor([1.,float("nan"),3.]),3), (torch.ones(2,2),2)])
def test_evaluation_label_adapter_rejects_ambiguous_or_invalid_metadata(labels,count):
    with pytest.raises(ValueError,match="Evaluation K labels"):
        worker.evaluation_K_labels(labels,count)
