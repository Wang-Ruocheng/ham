"""Fixed longer numerical R-only check after all short L-BFGS runs hit max_iter.

Restart from the same immutable scaled96/6000 states, not selected checkpoints.
No Q/C load, new objective, new samples, labels, or TEST access in optimization.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from experiments import phi4_ru_r_optimizer_probe as probe
from experiments.phi4_qcr import sha256_file, state_fingerprint, write_json
from experiments.phi4_ru_r_calibration import endpoint_metrics, predict
from scripts.phi4_ru_empirical_flow_diagnostic import chain_and_cell_means, draw_flow
from scripts.report_phi4_qcr import _plotting


def run(cache, calibration, short_probe, output):
    cache, calibration, short_probe, output = map(lambda p: Path(p).resolve(),
        (cache, calibration, short_probe, output))
    if any(output == p or p in output.parents for p in (cache, calibration, short_probe)):
        raise ValueError("Output must be separate from immutable inputs")
    output.mkdir(parents=True, exist_ok=False)
    started, cpu = time.perf_counter(), time.process_time()
    hashes, models, records, current = {}, {}, {}, "verify_inputs"
    def status(state, **extra):
        write_json(output / "status.json", dict(status=state,
            workflow_complete=state == "complete", returncode=0 if state == "complete" else None,
            current=current, completed_models=list(records), cpu_seconds=time.process_time()-cpu,
            wall_seconds=time.perf_counter()-started, **extra))
    def guard():
        if time.process_time()-cpu > 600.:
            raise probe.BudgetHold("Fixed 600 CPU-second refinement budget reached")
        if sum(p.stat().st_size for p in output.iterdir() if p.is_file()) > 100*1024**2:
            raise probe.BudgetHold("Fixed 100 MiB refinement output budget reached")
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        status("running")
        probe.verify_public(cache, hashes)
        manifest = probe.verify_public(calibration, hashes)
        probe.verify_public(short_probe, hashes)
        previous = json.loads((short_probe / "summary.json").read_text())
        if any(previous["records"][f"lbfgs_seed{s}"]["stopping"] != "max_iter" for s in probe.SEEDS):
            raise ValueError("Longer check requires all short runs to have hit the predeclared iteration cap")
        source = json.loads((calibration / "summary.json").read_text())
        direct = probe.load_fit_only(cache)
        endpoints = probe.restore_endpoints(calibration, source, direct)
        code_files = (Path(__file__), Path(probe.__file__),
            Path(__file__).parents[1] / "recurrent_rg/phi4_r_calibration.py",
            Path(__file__).parents[1] / "experiments/phi4_ru_r_calibration.py",
            Path(__file__).parents[1] / "scripts/phi4_ru_empirical_flow_diagnostic.py")
        code_sha = {str(p.resolve()): sha256_file(p) for p in code_files}
        write_json(output / "manifest.json", dict(schema_version=5,
            recipe="phi4_ru_same_endpoints_fixed1000_lbfgs_v1",
            trigger="All three TRAIN-only short L-BFGS solves reached max_iter100",
            initial_source="same immutable scaled96/6000 endpoints for seeds91/92/93",
            initial_R_sha256={str(s):state_fingerprint(m) for s,m in endpoints.items()},
            loss="5*e01+2.5*e12+2.5*e02 in original g; full FIT TRAIN mean",
            optimizer=dict(type="fresh L-BFGS", lr=1., max_iter=1000, max_eval=4000,
                history_size=20, tolerance_grad=1e-9, tolerance_change=1e-12,
                line_search_fn="strong_wolfe"),
            endpoint_selection="all three fixed solver endpoints, no TEST selection",
            input_sha256=hashes, source_sha256=code_sha, CPU_threads=1,
            cpu_budget_seconds=600, output_budget_bytes=100*1024**2,
            target_stds=manifest["target_stds"], Q_updates=0, C_updates=0,
            Q_loaded=False, C_loaded=False, labels_used_by_optimizer=False,
            test_arrays_loaded_before_optimization=False, new_root_search=False,
            new_physical_sampling=False, GPU_used=False))
        for seed in probe.SEEDS:
            key = f"lbfgs1000_seed{seed}"
            current = key
            status("running")
            def progress(row):
                write_json(output / "history_progress.json", dict(current=current,
                    completed=records, current_progress=row))
            model, record = probe.optimizer_arm(endpoints[seed], direct, "lbfgs", guard=guard,
                lbfgs_max_iter=1000, lbfgs_max_eval=4000, on_progress=progress)
            if record["initial_R_sha256"] != source["final_R_sha256"][f"scaled96_seed{seed}"]:
                raise RuntimeError("Paired original endpoint mismatch")
            records[key], models[key] = record, model
            write_json(output / "history_progress.json", dict(completed=records))
            print(json.dumps(dict(model=key, **{name:record[name] for name in
                ("gradient_calls", "optimizer_iterations", "endpoint_train_loss", "stopping")})), flush=True)
        current = "evaluate_fixed_endpoints"
        status("running")
        with np.load(cache / "cache.npz", allow_pickle=False) as archive:
            arrays = {f"{split}_{name}": archive[f"{split}_{name}"].copy()
                for split in ("train", "test") for name in ("labels", "cell_ids")}
            arrays.update({f"{split}_direct": archive[f"eval_{split}_direct"].copy()
                           for split in ("train", "test")})
        metrics, directions, states = {}, {}, {}
        for key, model in models.items():
            guard()
            states[key] = state_fingerprint(model)
            metrics[key], directions[key] = {}, {}
            for split in ("train", "test"):
                g = arrays[f"{split}_direct"]
                prediction = predict(model, g)
                metrics[key][split] = endpoint_metrics(g, prediction, manifest["target_stds"])
                directions[key][split] = probe.direction_metrics(g, prediction, arrays[f"{split}_cell_ids"])
                arrays.update({f"eval_{key}_{split}_{kind}": value for kind, value in prediction.items()})
            if state_fingerprint(model) != states[key]:
                raise RuntimeError("Endpoint changed during evaluation")
        np.savez_compressed(output / "evaluation_arrays.npz", **arrays)
        np.savez_compressed(output / "R_parameters.npz", **{
            f"{key}.{name}": value.detach().numpy() for key, model in models.items()
            for name, value in model.state_dict().items()})
        plt = _plotting(None)
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
        for ax, (key, record) in zip(axes, records.items()):
            ax.plot([r["gradient_calls"] for r in record["history"]],
                    [r["loss"] for r in record["history"]], color="#467c9f", linewidth=1.1)
            ax.set(title=f"R 端点种子 {key[-2:]}", xlabel="梯度计算次数", ylabel="训练闭合目标", yscale="log")
        fig.savefig(output / "loss.png", dpi=180)
        plt.close(fig)
        g = np.concatenate((arrays["train_direct"], arrays["test_direct"]), axis=1)
        ids = np.concatenate((arrays["train_cell_ids"], arrays["test_cell_ids"]))
        _, _, means, _ = chain_and_cell_means(g, ids)
        prediction = np.concatenate([arrays[f"eval_lbfgs1000_seed93_{split}_one_step"]
                                     for split in ("train", "test")], axis=1)
        _, _, pmean, _ = chain_and_cell_means(prediction, ids)
        draw_flow(output, g, means, pmean, model="scaled96_seed93", plt=plt)
        # Q/C is fixed, so the coupling-to-g map is unchanged and referenced
        # by fingerprint rather than recomputing an identical map.
        for name, digest in {**hashes, **code_sha}.items():
            if sha256_file(name) != digest:
                raise RuntimeError(f"Immutable input changed: {name}")
        guard()
        result = dict(status="complete", workflow_complete=True, returncode=0,
            records=records, latent_metrics=metrics, directional_metrics=directions,
            final_R_sha256=states, input_and_code_sha256_unchanged=True,
            Q_updates=0, C_updates=0, Q_loaded=False, C_loaded=False, GPU_seconds=0,
            cpu_seconds=time.process_time()-cpu, wall_seconds=time.perf_counter()-started,
            labels_used_by_optimizer=False, test_arrays_loaded_only_after_all_solves=True,
            root_search=False, critical_r=None, thermal_nu=None,
            coupling_by_scale_reference=str(calibration / "coupling_by_scale.png"),
            coupling_reference_sha256=sha256_file(calibration / "coupling_by_scale.png"),
            limits="Three R restarts share the same frozen Q/C and existing correlated chains. Existing TEST already inspected. No physical boundary or intrinsic-dimension certification.")
        write_json(output / "summary.json", result)
        write_json(output / "public_files.json", {p.name:sha256_file(p) for p in output.iterdir()
                   if p.is_file() and p.name not in ("status.json", "public_files.json")})
        current = "complete"
        status("complete", error=None)
    except BaseException as error:
        status("budget_hold" if isinstance(error, probe.BudgetHold) else "failed",
               error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "calibration", "short-probe", "output"):
        parser.add_argument("--" + name, required=True)
    run(**vars(parser.parse_args()))
