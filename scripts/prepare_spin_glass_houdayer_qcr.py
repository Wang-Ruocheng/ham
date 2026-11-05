"""Explicit Houdayer-coupled replica 0/1 overlap adapter for new QCR runs.

The complete group contains four (or another declared even number of) real
replicas coupled by cluster moves. They target a product Boltzmann measure,
but finite-chain independence/equilibration is NOT certified. Exactly one
predeclared overlap pair, 0/1, becomes one ensemble for each (J,T) cell. Other
replica pairs are not invented as independent training ensembles.
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


QUALIFICATION = (
    "Houdayer 簇更新耦合的真实副本组；目标为相同 J、T 下的乘积玻尔兹曼分布，"
    "但本次有限链未认证平衡或副本独立性。全部真实副本从独立随机初态启动。"
    "仅预先指定副本 0/1 的同温、同时刻重叠场用于训练；不把全部副本对当作额外独立系综。"
    "完整无序实现 J 划分训练／测试，温度和副本对不是独立无序样本。"
    "本数据生成、适配步骤不训练模型，不使用 J、T 标签监督模型。"
)


def _sampler_receipt(source):
    status = json.loads((source / "status.json").read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    if status.get("status") != "complete" or not status.get("workflow_complete"):
        raise ValueError("Houdayer physical data must be complete")
    if manifest.get("status") != "complete":
        raise ValueError("Houdayer manifest must be complete")
    sampler = manifest.get("sampler")
    if not isinstance(sampler, dict) or sampler.get("algorithm") != "houdayer_pt":
        raise ValueError("Require explicit Houdayer+PT sampler receipt; ordinary PT is not accepted")
    if sampler.get("schema") != "houdayer_parallel_tempering_v1":
        raise ValueError("Unsupported Houdayer sampler receipt schema")
    plan = json.loads((source / "plan.json").read_text())
    if plan.get("sampler") != sampler:
        raise ValueError("Houdayer sampler receipt differs between frozen plan and manifest")
    kernel_sha = sampler.get("kernel_source_sha256", "")
    if not isinstance(kernel_sha, str) or re.fullmatch(r"[0-9a-f]{64}", kernel_sha) is None:
        raise ValueError("Missing Houdayer kernel-source SHA256 receipt")
    if sampler.get("initial") != "hot" or sampler.get("real_replicas_coupled") is not True:
        raise ValueError("Require declared all-random initial states and Houdayer-coupled real replicas")
    replicas = manifest.get("options", {}).get("replicas")
    if type(replicas) is not int or replicas < 2 or replicas % 2 or sampler.get("replicas") != replicas:
        raise ValueError("Require a consistent even real-replica group count >= 2")
    if manifest.get("options", {}).get("field") != 0:
        raise ValueError("This adapter requires the declared zero-field target")
    if status.get("equilibrium_certified") is True or manifest.get("equilibrium_certified") is True:
        raise ValueError("Finite-chain Houdayer source must not assert certified equilibrium")
    return manifest, sampler, replicas


def prepare_overlap_cache(source, output):
    """New-only adaptation retaining all saved times, not six pair replicas."""
    source, output = Path(source).resolve(), Path(output).resolve()
    manifest, sampler, replicas = _sampler_receipt(source)
    xs = {split: [] for split in ("train", "test")}
    groups, rows = {split: [] for split in xs}, {split: [] for split in xs}
    geometry, lattice_shape = None, None
    for record in iter_spin_glass(source):
        if record.configurations.ndim != 5 or record.configurations.shape[1] != replicas:
            raise ValueError("Require 2D configurations with the declared Houdayer-coupled real replicas")
        q = record.overlap_view(0, 1)
        if q.shape[1] < 4 or q.shape[1] % 2:
            raise ValueError("Training-aligned half-Q needs an even saved sample count >= 4")
        if geometry is None:
            geometry, lattice_shape = q.shape[1:], list(q.shape[2:-1])
        if q.shape[1:] != geometry:
            raise ValueError("All overlap ensembles must have identical sample/lattice geometry")
        if len(lattice_shape) != 2 or lattice_shape[0] != lattice_shape[1]:
            raise ValueError("Require square 2D overlap fields")
        bx, by = np.sign(record.bonds)
        plaquettes = bx * np.roll(by, -1, axis=0) * np.roll(bx, -1, axis=1) * by
        physical = dict(dimension=2, lattice_shape=lattice_shape,
            zero_bond_count=int(np.count_nonzero(record.bonds == 0)),
            frustrated_plaquette_fraction=float(np.mean(plaquettes < 0)),
            zero_plaquette_fraction=float(np.mean(plaquettes == 0)),
            real_replica_count=replicas, real_replicas_coupled_by_houdayer=True,
            replica_pairs_are_not_independent=True, initialization="all independent random starts",
            equilibrium_certified=False)
        for t, temperature in enumerate(record.temperatures):
            xs[record.split].append(torch.from_numpy(q[t].copy()))
            groups[record.split].append([record.disorder_id, t])
            rows[record.split].append(dict(disorder_id=record.disorder_id,
                temperature_index=t, temperature=float(temperature),
                bond_sha256=_bond_sha(record.bonds), overlap_replicas=[0, 1],
                samples=q.shape[1], sites=int(np.prod(lattice_shape)), **physical))
    datasets = {}
    for split in xs:
        if not xs[split]:
            raise ValueError("Both disorder-held-out splits must be nonempty")
        datasets[split] = EnsembleDataset(torch.stack(xs[split]), couplings=None,
            group_ids=torch.tensor(groups[split], dtype=torch.long)).validate()
    train_ids = {row["disorder_id"] for row in rows["train"]}
    test_ids = {row["disorder_id"] for row in rows["test"]}
    if train_ids & test_ids:
        raise ValueError("Whole-disorder leakage in Houdayer adapter")
    output.mkdir(parents=True, exist_ok=False)
    torch.save(dict(schema_version=1, **{split: asdict(data) for split, data in datasets.items()}),
        output / "ensembles.pt")
    load_ensembles(output / "ensembles.pt")
    metadata = dict(schema_version=1, source_dataset=str(source),
        source_manifest_sha256=_sha(source / "manifest.json"),
        cache_sha256=_sha(output / "ensembles.pt"), adapter_code_sha256=_sha(Path(__file__)),
        dimension=2, lattice_shape=lattice_shape, **rows,
        representation="q_x=s_x^0*s_x^1; same J, temperature, saved time; predeclared real-replica pair 0/1 only",
        sampler=sampler, real_replica_count=replicas, real_replicas_coupled_by_houdayer=True,
        replica_pairs_are_not_independent=True, overlap_pair=[0, 1],
        training_ensemble_count_per_disorder_and_temperature=1,
        initialization="all independent random starts", equilibrium_certified=False,
        source_thermal_independence_certified=False,
        couplings_in_training_cache=False, labels_used_in_training=False,
        grouping="Whole disorder held out; [disorder_id,temperature_index] identifies one overlap ensemble",
        source_sampling_sha256=manifest.get("sampling_sha256"), qualification=QUALIFICATION)
    _json(output / "evaluation_metadata.json", metadata)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    metadata = prepare_overlap_cache(args.source, args.output)
    print(json.dumps(dict(train_ensembles=len(metadata["train"]),
        test_ensembles=len(metadata["test"]), real_replica_count=metadata["real_replica_count"],
        overlap_pair=[0, 1], labels_used_in_training=False, equilibrium_certified=False),
        ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
