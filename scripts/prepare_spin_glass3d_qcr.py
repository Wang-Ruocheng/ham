"""New-only three-dimensional EA ICM+PT overlap adapter, with J held out whole.

Only the predeclared replica pair 0/1 becomes one QCR ensemble per (J,T).
The four coupled real replicas are not four independent disorder samples.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re

import numpy as np
import torch

from recurrent_rg.data.ensembles import EnsembleDataset, load_ensembles
from recurrent_rg.data.spin_glass_pipeline import iter_spin_glass, _sha, _json, _bond_sha


SAMPLER_SCHEMA = "isoenergetic_cluster_parallel_tempering_3d_v1"
QUALIFICATION = (
    "三维零场 EA ±J，PT 全温度与预定低温 ICM 的耦合真实副本组；目标为乘积玻尔兹曼分布，"
    "有限采样未认证平衡或副本独立性。每个无序实现仅一组全随机初态，不构成独立冷热对照。"
    "仅预先指定副本 0/1 的同温、同时刻重叠场训练，不把全部副本对当作独立系综。"
    "训练／测试划分单位是完整无序实现 J；温度和副本不是额外独立无序样本。"
    "随机键与温度仅用于数据生成和事后评价，不监督 Q、C 或 R。"
)


def _sampler_receipt(source):
    status = json.loads((source / "status.json").read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    plan = json.loads((source / "plan.json").read_text())
    if (status.get("status") != "complete" or status.get("workflow_complete") is not True
            or manifest.get("status") != "complete"):
        raise ValueError("3D ICM+PT physical source must be complete")
    sampler = manifest.get("sampler")
    if (not isinstance(sampler, dict) or sampler.get("schema") != SAMPLER_SCHEMA
            or sampler.get("algorithm") != "icm_pt"):
        raise ValueError("Require the explicit new 3D low-temperature ICM+PT receipt")
    if plan.get("sampler") != sampler:
        raise ValueError("Sampler differs between frozen plan and manifest")
    if re.fullmatch(r"[0-9a-f]{64}", str(sampler.get("kernel_source_sha256", ""))) is None:
        raise ValueError("Missing ICM kernel-source SHA256 receipt")
    options = manifest.get("options", {})
    if plan.get("options") != options:
        raise ValueError("Physical options differ between frozen plan and manifest")
    if (options.get("distribution") != "bimodal" or options.get("bond_scale") != 1
            or options.get("bond_mean") != 0 or options.get("positive_probability") != .5):
        raise ValueError("This experiment requires symmetric unit +/-J bimodal disorder")
    replicas = options.get("replicas")
    shape = options.get("shape", [])
    if (len(shape) != 3 or len(set(shape)) != 1 or type(replicas) is not int
            or replicas < 2 or replicas % 2 or sampler.get("replicas") != replicas):
        raise ValueError("Require cubic 3D geometry and a consistent even replica count")
    if (sampler.get("initial") != "hot" or sampler.get("real_replicas_coupled") is not True
            or options.get("field") != 0):
        raise ValueError("Require zero-field all-hot jointly ICM-updated real replicas")
    if sampler.get("saved_overlap_signs_unfolded") is not True:
        raise ValueError("Raw saved overlap signs must be unfolded after ICM gauge construction")
    threshold = sampler.get("icm_max_temperature")
    if not isinstance(threshold, (int, float)) or not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Missing finite positive predeclared ICM temperature threshold")
    if status.get("equilibrium_certified") is True or manifest.get("equilibrium_certified") is True:
        raise ValueError("Finite-chain source cannot assert certified equilibrium")
    return manifest, sampler, replicas


def _plaquette_frustration(bonds):
    signs = np.sign(bonds)
    planes = []
    for a in range(3):
        for b in range(a + 1, 3):
            planes.append(signs[a] * np.roll(signs[b], -1, axis=a)
                * np.roll(signs[a], -1, axis=b) * signs[b])
    values = np.stack(planes)
    return float(np.mean(values < 0)), float(np.mean(values == 0))


def prepare_overlap_cache(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    manifest, sampler, replicas = _sampler_receipt(source)
    xs, groups, rows = ({split: [] for split in ("train", "test")} for _ in range(3))
    geometry, lattice_shape = None, None
    for record in iter_spin_glass(source):
        if record.configurations.ndim != 6 or record.configurations.shape[1] != replicas:
            raise ValueError("Require declared 3D ICM-coupled replica fields")
        q = record.overlap_view(0, 1)
        if q.shape[1] < 4 or q.shape[1] % 2:
            raise ValueError("Half-Q needs an even saved sample count >= 4")
        if geometry is None:
            geometry, lattice_shape = q.shape[1:], list(q.shape[2:-1])
        if q.shape[1:] != geometry:
            raise ValueError("All overlap cells must have identical sample/lattice geometry")
        frustrated, zero = _plaquette_frustration(record.bonds)
        physical = dict(dimension=3, lattice_shape=lattice_shape,
            zero_bond_count=int(np.count_nonzero(record.bonds == 0)),
            frustrated_plaquette_fraction=frustrated, zero_plaquette_fraction=zero,
            plaquette_planes=[[0, 1], [0, 2], [1, 2]], real_replica_count=replicas,
            real_replicas_coupled_by_icm=True, replica_pairs_are_not_independent=True,
            initialization="one group, all independent random starts", equilibrium_certified=False)
        for t, temperature in enumerate(record.temperatures):
            xs[record.split].append(torch.from_numpy(q[t].copy()))
            groups[record.split].append([record.disorder_id, t])
            rows[record.split].append(dict(disorder_id=record.disorder_id,
                temperature_index=t, temperature=float(temperature),
                bond_sha256=_bond_sha(record.bonds), overlap_replicas=[0, 1],
                samples=q.shape[1], sites=int(np.prod(lattice_shape)), **physical))
    if not all(xs.values()):
        raise ValueError("Both whole-disorder splits must be nonempty")
    datasets = {split: EnsembleDataset(torch.stack(xs[split]), couplings=None,
        group_ids=torch.tensor(groups[split], dtype=torch.long)).validate() for split in xs}
    if {g[0] for g in groups["train"]} & {g[0] for g in groups["test"]}:
        raise ValueError("Whole-disorder leakage")
    output.mkdir(parents=True, exist_ok=False)
    torch.save(dict(schema_version=1, **{split: asdict(ds) for split, ds in datasets.items()}), output / "ensembles.pt")
    load_ensembles(output / "ensembles.pt")
    metadata = dict(schema_version=1, source_dataset=str(source),
        source_manifest_sha256=_sha(source / "manifest.json"),
        cache_sha256=_sha(output / "ensembles.pt"), adapter_code_sha256=_sha(Path(__file__)),
        dimension=3, lattice_shape=lattice_shape, **rows,
        representation="q_x=s_x^0*s_x^1; same J, temperature, saved time; predeclared real-replica pair 0/1 only",
        sampler=sampler, real_replica_count=replicas, real_replicas_coupled_by_icm=True,
        replica_pairs_are_not_independent=True, overlap_pair=[0, 1],
        training_ensemble_count_per_disorder_and_temperature=1,
        initialization="one group, all independent random starts", equilibrium_certified=False,
        source_thermal_independence_certified=False, couplings_in_training_cache=False,
        labels_used_in_training=False, split_unit="entire quenched disorder, all temperatures and replicas",
        grouping="[disorder_id,temperature_index] identifies one overlap ensemble, not independent disorder samples",
        source_sampling_sha256=manifest.get("sampling_sha256"), qualification=QUALIFICATION)
    _json(output / "evaluation_metadata.json", metadata)
    _json(output / "status.json", dict(status="complete", workflow_complete=True, returncode=0,
        cache_sha256=metadata["cache_sha256"], labels_used_in_training=False, error=None))
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    metadata = prepare_overlap_cache(args.source, args.output)
    print(json.dumps(dict(status="complete", train_ensembles=len(metadata["train"]),
        test_ensembles=len(metadata["test"]), overlap_pair=[0, 1], labels_used_in_training=False), ensure_ascii=False))
    return metadata


if __name__ == "__main__":
    main()
