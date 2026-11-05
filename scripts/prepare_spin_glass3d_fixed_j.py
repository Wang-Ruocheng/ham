"""Select one predeclared J from an immutable overlap cache; no sampling.

Hold out whole temperature indices t % 5 == 2, not random configurations.
This is within-J temperature generalization, not unseen-disorder evaluation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from recurrent_rg.data.ensembles import EnsembleDataset, load_ensembles
from scripts.run_spin_glass2d import sha, write_json


SPLIT = "whole_temperature_within_fixed_disorder"
QUALIFICATION = (
    "单一预先指定 J 的条件化探索；完整温度索引 t%5==2 留出，其余训练。"
    "不是跨无序泛化；温度点不是独立无序样本。复用原副本0/1重叠场及全部保存时刻，"
    "不重新采样、不改变J、不把ICM耦合副本或相关保存时刻算成独立物理链。"
    "原数据未认证充分热化或副本独立性；单尺寸神经不动点不认证相变或普适指数。"
)


def select_fixed_disorder(source_cache, source_metadata, disorder_id=0):
    source_cache = Path(source_cache).resolve()
    metadata = json.loads(Path(source_metadata).read_text())
    if sha(source_cache) != metadata["cache_sha256"]:
        raise ValueError("Source cache fingerprint mismatch")
    if type(disorder_id) is not int or disorder_id < 0:
        raise ValueError("Require a predeclared nonnegative integer disorder ID")
    if (metadata.get("dimension") != 3 or metadata.get("overlap_pair") != [0, 1]
            or metadata.get("labels_used_in_training") is not False
            or metadata.get("equilibrium_certified") is not False):
        raise ValueError("Require label-free, unqualified 3D pair-01 source")
    cells = []
    for split, ds in zip(("train", "test"), load_ensembles(source_cache)):
        rows = metadata[split]
        if len(rows) != len(ds.configurations) or ds.couplings is not None or ds.group_ids is None:
            raise ValueError("Source metadata or label boundary mismatch")
        for group, field, row in zip(ds.group_ids.tolist(), ds.configurations, rows):
            if group != [row["disorder_id"], row["temperature_index"]]:
                raise ValueError("Source group identity mismatch")
            if group[0] == disorder_id:
                if row["samples"] != field.shape[0] or row.get("overlap_replicas") != [0, 1]:
                    raise ValueError("Source sample count or overlap pair mismatch")
                cells.append((group[1], field.clone(), dict(row)))
    cells.sort(key=lambda cell: cell[0])
    if not cells:
        raise ValueError("Requested fixed disorder is absent")
    if [cell[0] for cell in cells] != list(range(20)):
        raise ValueError("Require the complete predeclared 20-temperature grid exactly once")
    bonds = {cell[2]["bond_sha256"] for cell in cells}
    if len(bonds) != 1:
        raise ValueError("Fixed-J cells have conflicting bond fingerprints")
    datasets, rows = {}, {}
    for split in ("train", "test"):
        chosen = [cell for cell in cells if (cell[0] % 5 == 2) == (split == "test")]
        datasets[split] = EnsembleDataset(torch.stack([cell[1] for cell in chosen]), couplings=None,
            group_ids=torch.tensor([[disorder_id, cell[0]] for cell in chosen], dtype=torch.long)).validate()
        rows[split] = [cell[2] for cell in chosen]
    result = {key: value for key, value in metadata.items()
              if key not in ("train", "test", "cache_sha256")}
    result.update(**rows, split_unit=SPLIT, fixed_disorder_id=disorder_id,
        fixed_bond_sha256=next(iter(bonds)), independent_disorder_count=1,
        source_cache=str(source_cache), source_cache_sha256=metadata["cache_sha256"],
        source_metadata_sha256=sha(source_metadata),
        selection="J ID fixed before this training; never selected by learned roots, losses, Tc or nu",
        split_rule="temperature_index % 5 == 2 is test; complete cells retained",
        train_temperature_indices=[cell[0] for cell in cells if cell[0] % 5 != 2],
        test_temperature_indices=[2, 7, 12, 17],
        grouping="[fixed_disorder_id,temperature_index]; same J in both splits, disjoint whole temperatures",
        labels_used_in_training=False, couplings_in_training_cache=False,
        physical_sampling_updates=0, qualification=QUALIFICATION)
    return datasets, result


def prepare(source_root, output, disorder_id=0):
    raw_output = Path(output)
    if raw_output.exists() or raw_output.is_symlink():
        raise FileExistsError(raw_output)
    source_root, output = Path(source_root).resolve(), raw_output.resolve()
    source_job = json.loads((source_root / "job.json").read_text())
    if source_job.get("status") != "complete" or source_job.get("workflow_complete") is not True:
        raise ValueError("Source experiment must be complete")
    source_config = json.loads((source_root / "config.json").read_text())
    datasets, metadata = select_fixed_disorder(source_config["data"]["cache"],
        source_root / "data/evaluation_metadata.json", disorder_id)
    output.mkdir(parents=True)
    torch.save(dict(schema_version=1, **{split: asdict(ds) for split, ds in datasets.items()}),
               output / "ensembles.pt")
    load_ensembles(output / "ensembles.pt")
    metadata["cache_sha256"] = sha(output / "ensembles.pt")
    write_json(output / "evaluation_metadata.json", metadata)
    write_json(output / "status.json", dict(status="complete", workflow_complete=True, returncode=0,
        fixed_disorder_id=disorder_id, physical_sampling_updates=0,
        cache_sha256=metadata["cache_sha256"], equilibrium_certified=False))
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--disorder-id", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = prepare(args.source_root, args.output, args.disorder_id)
    print(json.dumps(dict(status="complete", fixed_disorder_id=result["fixed_disorder_id"],
        train_ensembles=len(result["train"]), test_ensembles=len(result["test"]),
        physical_sampling_updates=0, cache_sha256=result["cache_sha256"])))


if __name__ == "__main__":
    main()
