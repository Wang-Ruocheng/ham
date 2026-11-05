"""Fixed-endpoint, CPU-only R controls for the completed phi4 r/u scan.

This is an explicitly requested diagnostic, not a fresh default QCR run or an
exact continuation. Q/C and physical fields are never loaded. All controls use
the same saved TRAIN g and original-coordinate all-start loss. TEST is read for
provenance but used only after the fixed 6000 updates. No fixed-point search.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from torch import nn

from experiments.phi4_qcr import optimizer_step_counts, sha256_file, state_fingerprint, write_json
from scripts.report_phi4_ru_qcr import load_public_run


RECIPE = "phi4_ru_frozen_cached_g_R_diagnostic_v1"
ARMS = ("raw96", "scaled96", "scaled256", "scale_specific96")
LABELS = dict(source="原联合训练端点", raw96="冻结表示：原坐标 R96",
              scaled96="冻结表示：归一化 R96", scaled256="冻结表示：归一化 R256",
              scale_specific96="尺度分别建模：两个 R96（非自治）")
LIMITATIONS = [
    "固定端点缓存：每条物理链128构型、一个固定C轨迹；不是原随机64构型且Q/C变化的训练分布。",
    "Q/C完全未加载且更新数为0；没有新采样、物理模型推理、MI或根搜索。",
    "原R仅用保存的预测作为基线；新R都是fresh Adam随机重拟合，不是原R优化器续训。",
    "所有loss在原g坐标计算；内部归一化只使用TRAIN输入尺度0/1，不改变坐标或各方向loss权重。",
    "原坐标96与归一化96初始物理函数精确配对；256容量臂为独立随机初始化，容量与优化影响未完全分离。",
    "两个尺度专用映射拥有更多参数且显式使用尺度索引；只是闭合诊断，不能称统一自治RG。",
    "单训练seed、有限物理链和固定C随机轨迹；改善只证明缓存闭合拟合改善，不认证临界点或物理RG。",
    "TEST只在固定6000步终点评价，不选checkpoint、不用TEST调参，不自动延长或挑选种子。",
    "TEST在此前审计中已查看过；这是固定条件留出诊断，不是从未接触的独立最终测试。",
    "已有物理采样警告全部保留；本诊断不增加独立物理样本，也不证明平衡。",
]


class BudgetHold(RuntimeError):
    pass


@dataclass(frozen=True)
class Options:
    source: str
    output: str
    steps: int = 6000
    seed: int = 91
    learning_rate: float = .001
    cpu_budget_seconds: float = 900.
    output_budget_bytes: int = 100 * 1024 ** 2

    def validate(self):
        if self.steps != 6000 or self.seed != 91 or self.learning_rate != .001:
            raise ValueError("Diagnostic fixes 6000 updates, seed91 and Adam .001")
        if not 0 < self.cpu_budget_seconds <= 900 or not 0 < self.output_budget_bytes <= 100 * 1024 ** 2:
            raise ValueError("Diagnostic budgets must be <=900 CPU seconds and <=100 MiB")


class ResidualMap(nn.Module):
    def __init__(self, hidden: int, center: np.ndarray, scale: np.ndarray):
        super().__init__()
        center, scale = np.asarray(center, dtype=np.float64), np.asarray(scale, dtype=np.float64)
        if (center.shape != (2,) or scale.shape != (2,) or not np.isfinite(center).all()
                or not np.isfinite(scale).all() or np.any(scale <= 0)):
            raise ValueError("Two finite center/positive scale coordinates required")
        self.hidden = hidden
        self.register_buffer("center", torch.tensor(center, dtype=torch.float64))
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float64))
        self.net = nn.Sequential(nn.Linear(2, hidden), nn.Tanh(), nn.Linear(hidden, hidden),
                                 nn.Tanh(), nn.Linear(hidden, 2)).double()

    def advance(self, g, transition=0):
        if transition not in (0, 1):
            raise ValueError("Only the two observed transitions are in scope")
        return g + self.scale * self.net((g - self.center) / self.scale)


class ScaleSpecificMap(nn.Module):
    def __init__(self, first: ResidualMap, second: ResidualMap):
        super().__init__()
        self.maps = nn.ModuleList([first, second])

    def advance(self, g, transition=0):
        if transition not in (0, 1):
            raise ValueError("Only two scale-specific maps are defined")
        return self.maps[transition].advance(g, transition)


def conditioned_copy(raw: ResidualMap, center, scale):
    """Exact change of internal units, preserving the original physical map."""
    if not torch.equal(raw.center, torch.zeros(2, dtype=torch.float64)) or not torch.equal(raw.scale, torch.ones(2, dtype=torch.float64)):
        raise ValueError("Expected original-coordinate residual map")
    result = ResidualMap(raw.hidden, center, scale)
    result.net.load_state_dict(raw.net.state_dict())
    with torch.no_grad():
        result.net[0].weight.copy_(raw.net[0].weight * result.scale[None, :])
        result.net[0].bias.copy_(raw.net[0].bias + raw.net[0].weight @ result.center)
        result.net[-1].weight.copy_(raw.net[-1].weight / result.scale[:, None])
        result.net[-1].bias.copy_(raw.net[-1].bias / result.scale)
    return result


def closure_loss(model, direct):
    if direct.ndim != 3 or direct.shape[1:] != (3, 2) or direct.requires_grad:
        raise ValueError("Expected frozen [physical chain,3,2] g")
    p01 = model.advance(direct[:, 0], 0)
    p12 = model.advance(direct[:, 1], 1)
    p02 = model.advance(p01, 1)
    terms = {"e01": (p01 - direct[:, 1]).square().sum(-1).mean(),
             "e12": (p12 - direct[:, 2]).square().sum(-1).mean(),
             "e02": (p02 - direct[:, 2]).square().sum(-1).mean()}
    return 5 * terms["e01"] + 2.5 * terms["e12"] + 2.5 * terms["e02"], terms


@torch.no_grad()
def predict(model, direct):
    g = torch.as_tensor(direct, dtype=torch.float64)
    first = model.advance(g[:, 0], 0)
    return dict(one_step=torch.stack([first, model.advance(g[:, 1], 1)], dim=1).numpy(),
                rollout=torch.stack([g[:, 0], first, model.advance(first, 1)], dim=1).numpy())


def evaluate_metrics(train, test, predicted):
    rows = []
    for split, g in (("train", train), ("test", test)):
        for path, start, target, prediction in (
            ("0_to_1", 0, 1, predicted[split]["one_step"][:, 0]),
            ("1_to_2", 1, 2, predicted[split]["one_step"][:, 1]),
            ("0_to_2", 0, 2, predicted[split]["rollout"][:, 2])):
            error = prediction - g[:, target]
            rmse = float(np.sqrt(np.mean(error ** 2)))
            std = float(np.sqrt(np.mean(np.var(train[:, target], axis=0, ddof=0))))
            identity = float(np.sqrt(np.mean((g[:, start] - g[:, target]) ** 2)))
            rows.append(dict(split=split, path=path, rmse=rmse,
                nrmse=rmse / std if std > 0 else None, train_target_std=std,
                relative_displacement_error=rmse / identity if identity > 0 else None,
                identity_rmse=identity, chains=len(g),
                train_constant_rmse=float(np.sqrt(np.mean((train[:, target].mean(axis=0) - g[:, target]) ** 2))),
                coordinate_rmse=np.sqrt(np.mean(error ** 2, axis=0)).tolist(),
                mean_error=error.mean(axis=0).tolist()))
    return rows


def _normalizer(train):
    inputs = train[:, :2].reshape(-1, 2)
    mean, scale = inputs.mean(axis=0), inputs.std(axis=0, ddof=0)
    if np.any(scale < 1e-10):
        raise ValueError("Rank-poor coordinate scale; no automatic whitening repair")
    return mean, scale


def _directory_bytes(path):
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _decoder_from_saved_train(arrays):
    """Recover the already fixed affine decoder, without refitting targets."""
    g = arrays["train_direct"].reshape(-1, 2)
    saved = arrays["train_feature_reconstruction"].reshape(-1, 105)
    design = np.column_stack([g, np.ones(len(g))])
    coef = np.linalg.lstsq(design, saved, rcond=None)[0]
    residual = float(np.max(np.abs(design @ coef - saved)))
    if residual > 1e-10:
        raise ValueError("Cannot exactly recover the saved affine decoder")
    return coef, residual


def _native_metrics(source, predicted, coef):
    rows = []
    for split in ("train", "test"):
        targets = source[f"{split}_feature_target"]
        for path, start, target, g in (
            ("0_to_1", 0, 1, predicted[split]["one_step"][:, 0]),
            ("1_to_2", 1, 2, predicted[split]["one_step"][:, 1]),
            ("0_to_2", 0, 2, predicted[split]["rollout"][:, 2])):
            f = g @ coef[:2] + coef[2]
            rmse = float(np.sqrt(np.mean((f - targets[:, target]) ** 2)))
            std = float(np.sqrt(np.mean(np.var(source["train_feature_target"][:, target], axis=0, ddof=0))))
            rows.append(dict(split=split, path=path, rmse=rmse, nrmse=rmse / std,
                fixed_train_feature_target_std=std,
                raw_feature_identity_rmse=float(np.sqrt(np.mean((targets[:, start] - targets[:, target]) ** 2)))))
    return rows


def _paired_condition_bootstrap(source, saved, *, repeats=1000):
    """Descriptive TEST-condition bootstrap; no selection or optimizer feedback."""
    ids = source["test_cell_ids"]
    cells = np.unique(ids)
    rng = np.random.default_rng(20261030)
    sampled = rng.integers(len(cells), size=(repeats, len(cells)))
    result = {}
    for name in ARMS:
        rows = []
        for path, target, key, index in (("0_to_1", 1, "one_step", 0),
                ("1_to_2", 2, "one_step", 1), ("0_to_2", 2, "rollout", 2)):
            truth = source["test_direct"][:, target]
            base = np.mean((saved[f"source_test_{key}"][:, index] - truth) ** 2, axis=1)
            other = np.mean((saved[f"{name}_test_{key}"][:, index] - truth) ** 2, axis=1)
            bm = np.array([base[ids == cell].mean() for cell in cells])
            om = np.array([other[ids == cell].mean() for cell in cells])
            delta = np.sqrt(om[sampled].mean(axis=1)) - np.sqrt(bm[sampled].mean(axis=1))
            rows.append(dict(path=path, rmse_difference_vs_source=float(np.sqrt(om.mean()) - np.sqrt(bm.mean())),
                descriptive_95_percentile_interval=np.quantile(delta, [.025, .975]).tolist()))
        result[name] = rows
    return dict(unit="whole TEST parameter condition; all four chains retained together", repeats=repeats,
        seed=20261030, paired=True, comparisons=result,
        qualification="11 held-out grid cells, conditional on one Q/C/R seed and fixed C draw; no multiplicity correction or population generalization guarantee")


def _collision_checks(source, models):
    """Describe the two previously identified TRAIN pairs; not a new search."""
    ids, direct = source["train_cell_ids"], source["train_direct"]
    result = []
    for first, second in (((23, 0), (44, 1)), ((14, 0), (24, 1))):
        inputs, targets = [], []
        for cell, k in (first, second):
            inputs.append(direct[ids == cell, k].mean(axis=0))
            targets.append(direct[ids == cell, k + 1].mean(axis=0))
        inputs, targets = np.asarray(inputs), np.asarray(targets)
        item = dict(a=dict(cell=first[0], input_scale=first[1]), b=dict(cell=second[0], input_scale=second[1]),
            input_g=inputs.tolist(), target_g=targets.tolist(), input_distance=float(np.linalg.norm(inputs[0]-inputs[1])),
            target_distance=float(np.linalg.norm(targets[0]-targets[1])), models={})
        source_predictions = []
        for cell,k in (first,second):
            index = int(np.flatnonzero(source["grid_cell_ids"] == cell)[0])
            source_predictions.append(source["grid_R_of_mean"][index,k])
        item["models"]["source"] = dict(predictions=np.asarray(source_predictions).tolist(),
            prediction_distance=float(np.linalg.norm(source_predictions[0]-source_predictions[1])))
        for name, model in models.items():
            values, norms = [], []
            for value, (_, k) in zip(inputs, (first, second)):
                x = torch.tensor(value[None], dtype=torch.float64, requires_grad=True)
                y = model.advance(x, k)
                jac = torch.stack([torch.autograd.grad(y[0,i], x, retain_graph=i==0)[0][0] for i in range(2)])
                values.append(y.detach().numpy()[0])
                norms.append(float(torch.linalg.svdvals(jac)[0]))
            values = np.asarray(values)
            item["models"][name] = dict(predictions=values.tolist(), prediction_distance=float(np.linalg.norm(values[0]-values[1])),
                endpoint_error_norms=np.linalg.norm(values-targets,axis=1).tolist(), full_map_lipschitz_at_two_inputs=norms)
        result.append(item)
    return dict(preselected_in_previous_audit=True, pairs=result,
        qualification="Near-input pairs only; not proof of identical latent states or impossibility of a more nonlinear autonomous map")


def run(options):
    options.validate()
    source, output = Path(options.source).resolve(), Path(options.output).resolve()
    if output == source or source in output.parents:
        raise ValueError("New output must be outside the immutable source directory")
    output.mkdir(parents=True, exist_ok=False)
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    actual, history, optimizers = {name: 0 for name in ARMS}, [], {}
    def status(state, **extra):
        write_json(output / "status.json", dict(recipe=RECIPE, status=state,
            workflow_complete=state == "complete", actual_steps_by_arm=actual.copy(),
            cpu_seconds=time.process_time() - start_cpu, wall_seconds=time.perf_counter() - start_wall,
            requires_user_action=state in ("failed", "budget_hold"),
            returncode=0 if state == "complete" else 3 if state == "budget_hold" else 1 if state == "failed" else None,
            **extra))
    def budget():
        if time.process_time() - start_cpu > options.cpu_budget_seconds:
            raise BudgetHold("Fixed CPU budget reached; no automatic continuation")
        if _directory_bytes(output) > options.output_budget_bytes:
            raise BudgetHold("Fixed directory budget reached")
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        status("running", current="source_validation")
        public = load_public_run(source)
        a = public.arrays
        fingerprints = {str(source / key): value for key, value in public.fingerprints.items()}
        inventory = source.parent / "public_files.json"
        trusted = json.loads(inventory.read_text())
        for path, digest in fingerprints.items():
            relative = str(Path(path).relative_to(source.parent))
            if trusted.get(relative) != digest:
                raise ValueError("Local public inventory does not match input SHA")
        fingerprints[str(inventory)] = sha256_file(inventory)
        own_sources = [Path(__file__), Path(__file__).parents[1] / "scripts/report_phi4_ru_frozen_r.py"]
        source_hashes = {str(p): sha256_file(p) for p in own_sources}
        train, test = a["train_direct"].copy(), a["test_direct"].copy()
        direct = torch.tensor(train, dtype=torch.float64)
        center, scale = _normalizer(train)
        torch.manual_seed(options.seed)
        raw = ResidualMap(96, np.zeros(2), np.ones(2))
        normalized = conditioned_copy(raw, center, scale)
        torch.manual_seed(options.seed)
        wide = ResidualMap(256, center, scale)
        models = dict(raw96=raw, scaled96=normalized, scaled256=wide,
            scale_specific96=ScaleSpecificMap(copy.deepcopy(normalized), copy.deepcopy(normalized)))
        with torch.no_grad():
            pair_error = float((raw.advance(direct.reshape(-1, 2)) - normalized.advance(direct.reshape(-1, 2))).abs().max())
        if pair_error > 1e-12:
            raise ValueError("Initial physical function pairing failed")
        initial = {name: state_fingerprint(m) for name, m in models.items()}
        optimizers = {name: torch.optim.Adam(m.parameters(), lr=options.learning_rate) for name, m in models.items()}
        manifest = dict(schema_version=5, recipe=RECIPE, options=asdict(options), input_sha256=fingerprints,
            source_sha256=source_hashes, device="cpu", threads=1, interop_threads=1,
            Q_loaded=False, C_loaded=False, Q_updates=0, C_updates=0, GPU_used=False,
            physical_fields_loaded=False, physical_sampling=False, source_R_weights_loaded=False,
            original_R_is_saved_prediction_baseline=True, labels_used_by_optimizer=False,
            cached_endpoint_g=True, exact_original_training_distribution=False,
            source_QCR_steps=3000, requested_new_R_steps_per_arm=options.steps,
            batch="full TRAIN:136 physical chains; no TEST optimization",
            optimizer="fresh Adam .001 fixed; no scheduler, weight decay or checkpoint selection",
            loss="5*e01+2.5*e12+2.5*e02; sum coordinates then mean chains; original g units",
            normalization_fit_scope="TRAIN input scales 0 and 1 only", center=center.tolist(), scale=scale.tolist(),
            paired_raw96_scaled96_initial_prediction_max_error=pair_error,
            initial_R_sha256=initial, parameter_counts={n: sum(p.numel() for p in m.parameters()) for n,m in models.items()},
            scientific_control_type={n: "nonautonomous_scale_specific" if n == "scale_specific96" else "autonomous" for n in models},
            evaluation_plan="all fixed endpoints; held-out condition metrics; paired condition bootstrap1000; same saved grid; no root search",
            default_recipe_modified=False, source_sampling_warning_count=public.summary["data_warning_count"],
            limitations=LIMITATIONS)
        write_json(output / "manifest.json", manifest)
        for step in range(options.steps + 1):
            budget()
            if step:
                for name, model in models.items():
                    loss, _ = closure_loss(model, direct)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite {name} objective")
                    optimizers[name].zero_grad(set_to_none=True)
                    loss.backward()
                    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                        raise FloatingPointError(f"Invalid {name} gradient")
                    optimizers[name].step()
                    actual[name] = step
            if step % 100 == 0 or step == options.steps:
                with torch.no_grad():
                    row = dict(step=step, cpu_seconds=time.process_time() - start_cpu, arms={})
                    for name, model in models.items():
                        loss, terms = closure_loss(model, direct)
                        row["arms"][name] = dict(loss=float(loss), **{k:float(v) for k,v in terms.items()})
                    history.append(row)
                write_json(output / "history_progress.json", dict(history=history, actual_steps_by_arm=actual.copy()))
                status("running", current="fixed_g_R_optimization")
                print(json.dumps(row), flush=True)
        status("running", current="fixed_endpoint_evaluation")
        saved = {key: a[key].copy() for key in ("train_direct", "test_direct", "train_labels", "test_labels",
            "train_cell_ids", "test_cell_ids", "flow_g", "flow_in_train_hull")}
        metrics, native = {}, {}
        decoder, decoder_error = _decoder_from_saved_train(a)
        all_predictions = {"source": {split:{key:a[f"{split}_{key}"] for key in ("one_step", "rollout")} for split in ("train","test")}}
        all_predictions.update({name:{"train":predict(m,train), "test":predict(m,test)} for name,m in models.items()})
        for name, predictions in all_predictions.items():
            metrics[name] = evaluate_metrics(train, test, predictions)
            native[name] = _native_metrics(a, predictions, decoder)
            for split, values in predictions.items():
                saved.update({f"{name}_{split}_{key}":value for key,value in values.items()})
        saved["source_flow_R"] = a["flow_R"].copy()
        grid = torch.tensor(a["flow_g"], dtype=torch.float64)
        with torch.no_grad():
            for name, model in models.items():
                if name == "scale_specific96":
                    for k in (0,1):
                        saved[f"{name}_flow_R{k}"] = model.advance(grid,k).numpy()
                else:
                    saved[f"{name}_flow_R"] = model.advance(grid).numpy()
        bootstrap = _paired_condition_bootstrap(a, saved)
        collisions = _collision_checks(a, models)
        counts = optimizer_step_counts(optimizers, options.steps)
        if any(row["min"] != options.steps or row["max"] != options.steps for row in counts.values()):
            raise RuntimeError("Not all Adam parameters reached the fixed endpoint")
        parameters = {f"{name}.{key}":value.detach().numpy() for name,m in models.items() for key,value in m.state_dict().items()}
        np.savez_compressed(output / "R_parameters.npz", **parameters)
        np.savez_compressed(output / "evaluation_arrays.npz", **saved)
        write_json(output / "metrics.json", dict(latent=metrics, native_features=native, paired_condition_bootstrap=bootstrap,
            preselected_collision_checks=collisions))
        from scripts.report_phi4_ru_frozen_r import write_plots
        write_plots(output, saved, metrics, history, LABELS)
        budget()
        if any(sha256_file(path) != digest for path,digest in {**fingerprints, **source_hashes}.items()):
            raise RuntimeError("A source artifact or script changed during execution")
        summary = dict(schema_version=5, recipe=RECIPE, status="complete", workflow_complete=True, returncode=0,
            actual_steps_by_arm=actual.copy(), optimizer_steps=counts, source_fingerprints_unchanged=True,
            Q_loaded=False, C_loaded=False, Q_updates=0, C_updates=0, source_R_weights_loaded=False,
            parameters_initial_sha256=initial, parameters_final_sha256={n:state_fingerprint(m) for n,m in models.items()},
            recovered_fixed_decoder_max_residual=decoder_error, labels_used_by_optimizer=False,
            metrics=metrics, native_feature_metrics=native, paired_condition_bootstrap=bootstrap,
            preselected_collision_checks=collisions,
            thermal_nu=None, critical_r=None, new_root_search=False, physical_fixed_point_certified=False,
            cpu_seconds=time.process_time()-start_cpu, wall_seconds=time.perf_counter()-start_wall,
            directory_bytes=_directory_bytes(output), GPU_seconds=0, threads=1, limitations=LIMITATIONS)
        write_json(output / "summary.json", summary)
        public_hashes = {p.name:sha256_file(p) for p in output.iterdir() if p.is_file() and p.name != "status.json"}
        write_json(output / "public_files.json", public_hashes)
        status("complete", current="complete", error=None, summary_sha256=sha256_file(output / "summary.json"))
        return summary
    except BaseException as error:
        held = isinstance(error, BudgetHold)
        status("budget_hold" if held else "failed", current="stopped_without_restart", error=str(error), traceback=traceback.format_exc())
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    options = Options(**vars(parser.parse_args(argv)))
    result = run(options)
    print(json.dumps(dict(status=result["status"], output=options.output, cpu_seconds=result["cpu_seconds"])), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
