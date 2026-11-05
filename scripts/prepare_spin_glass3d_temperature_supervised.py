"""Explicit TRAIN-temperature Q supervision on unchanged whole-J overlap data.

This is an authorized supervised ablation, not the unsupervised default.  Raw
temperature is the sole target; neither J identity nor a coarse-level target is
introduced.  Existing fields, group ordering, splits and sampling limitations
are retained without physical sampling.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from recurrent_rg.data.ensembles import EnsembleDataset, load_ensembles
from scripts.report_spin_glass3d import validate_metadata
from scripts.run_spin_glass2d import sha, write_json


SPLIT_CONTRACT_KIND = "whole_disorder_temperature_supervision"
SUPERVISION = dict(weight=10., scale=0, target="raw_temperature_T",
    estimator="Q(full microscopic batch)", fitting_split="TRAIN only",
    test_labels_evaluation_only=True, coarse_labels_used=False,
    disorder_identity_used_in_loss=False, gradients="Q only; no C/R supervision")
QUALIFICATION = (
    "显式温度监督消融：仅 TRAIN 原始温度 T 以权重10监督微观尺度 Q，TEST 标签仅用于评价。"
    "完整保留原多 J、多温度的副本0/1重叠场、保存时刻及 whole-J 划分；随机键编号不进入损失。"
    "粗尺度不强制拟合同一微观温度；C 的完整壳层经验 RSMI、方差和保色项及 R 均匀尺度闭合保持不变。"
    "复用采样未认证热化或 ICM 耦合副本独立性，不把温度、保存时刻或副本对算成独立无序实现。"
    "神经不动点、插值温度和候选指数不认证真实相变或普适热指数。"
)


def temperature_supervised_datasets(source_cache, source_metadata):
    """Return unchanged field/group tensors and explicit raw-T target columns."""
    source_cache, source_metadata = Path(source_cache).resolve(), Path(source_metadata).resolve()
    before = {str(path): sha(path) for path in (source_cache, source_metadata)}
    metadata = json.loads(source_metadata.read_text())
    if before[str(source_cache)] != metadata.get("cache_sha256"):
        raise ValueError("Source cache fingerprint mismatch")
    originals = list(zip(("train", "test"), load_ensembles(source_cache)))
    if metadata.get("split_unit") == "whole_temperature_within_fixed_disorder":
        raise ValueError("Require the original whole-disorder multi-J source, not a fixed-J split")
    validate_metadata(originals, metadata)
    datasets = {}
    bonds, ladder = {}, {}
    for split, dataset in originals:
        rows = metadata[split]
        if (dataset.group_ids.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
                or torch.any(dataset.group_ids < 0)):
            raise ValueError("Source group IDs must be nonnegative integers")
        labels = torch.tensor([[row["temperature"]] for row in rows], dtype=torch.float64)
        datasets[split] = EnsembleDataset(dataset.configurations, labels, dataset.group_ids).validate()
        for row in rows:
            key, fingerprint = str(row["disorder_id"]), row["bond_sha256"]
            if key in bonds and bonds[key] != fingerprint:
                raise ValueError("Conflicting disorder bond fingerprint")
            bonds[key] = fingerprint
            ti, temperature = row["temperature_index"], row["temperature"]
            if ti in ladder and ladder[ti] != temperature:
                raise ValueError("Temperature-index ladder differs between disorders")
            ladder[ti] = temperature
    if len(set(bonds.values())) != len(bonds):
        raise ValueError("Different disorder IDs cannot reuse an identical bond fingerprint")
    contract = dict(kind=SPLIT_CONTRACT_KIND,
        group_columns=["disorder_id", "temperature_index"], supervision_columns=["T"],
        bond_sha256_by_disorder=bonds)
    result = {key: value for key, value in metadata.items() if key != "cache_sha256"}
    result.update(source_cache=str(source_cache), source_cache_sha256=before[str(source_cache)],
        source_metadata_sha256=before[str(source_metadata)],
        original_qualification=metadata.get("qualification"),
        labels_used_in_training=True, couplings_in_training_cache=True,
        temperature_supervision=dict(SUPERVISION), split_contract=contract,
        physical_sampling_updates=0, source_fields_and_groups_unchanged=True,
        qualification=QUALIFICATION)
    if {str(path): sha(path) for path in (source_cache, source_metadata)} != before:
        raise RuntimeError("Read-only source fingerprints changed")
    return datasets, result


def prepare(source_root, output):
    raw_output = Path(output)
    if raw_output.exists() or raw_output.is_symlink():
        raise FileExistsError(raw_output)
    source_root, output = Path(source_root).resolve(), raw_output.resolve()
    source_job_path, source_config_path = source_root / "job.json", source_root / "config.json"
    source_job = json.loads(source_job_path.read_text())
    if source_job.get("status") != "complete" or source_job.get("workflow_complete") is not True:
        raise ValueError("Source experiment must be complete")
    config = json.loads(source_config_path.read_text())
    cache = Path(config["data"]["cache"])
    if not cache.is_absolute():
        cache = source_root / cache
    source_metadata = source_root / "data/evaluation_metadata.json"
    sampling_path = source_root / "physical_data/sampling.json"
    protected = [source_job_path, source_config_path, cache, source_metadata]
    if sampling_path.is_file():
        protected.append(sampling_path)
    before = {str(path.resolve()): sha(path) for path in protected}
    datasets, metadata = temperature_supervised_datasets(cache, source_metadata)
    if sampling_path.is_file():
        sampling = json.loads(sampling_path.read_text())
        expected = metadata.get("source_sampling_sha256")
        if expected is not None and expected != before[str(sampling_path.resolve())]:
            raise ValueError("Original sampling-warning receipt fingerprint mismatch")
        metadata["source_sampling_warning_receipt"] = dict(path=str(sampling_path),
            sha256=before[str(sampling_path.resolve())], warning_count=sampling.get("warning_count"),
            warnings_by_disorder=[dict(disorder_id=row["disorder_id"], split=row.get("split"),
                warnings=row.get("warnings", [])) for row in sampling.get("disorders", [])],
            equilibrium_certified=sampling.get("equilibrium_certified", False),
            qualification=sampling.get("qualification"))
    else:
        metadata["source_sampling_warning_receipt"] = dict(available=False,
            note="Original metadata and sampling SHA retained; no new thermalization claim")
    if {str(path.resolve()): sha(path) for path in protected} != before:
        raise RuntimeError("Read-only source fingerprint changed before output")
    output.mkdir(parents=True, exist_ok=False)
    torch.save(dict(schema_version=1, split_contract=metadata["split_contract"],
        **{split: asdict(dataset) for split, dataset in datasets.items()}), output / "ensembles.pt")
    actual = load_ensembles(output / "ensembles.pt")
    for (split, expected), restored in zip(datasets.items(), actual):
        if (not torch.equal(expected.configurations, restored.configurations)
                or not torch.equal(expected.group_ids, restored.group_ids)
                or not torch.equal(expected.couplings, restored.couplings)):
            raise RuntimeError(f"Supervised cache round-trip changed {split}")
    metadata.update(cache_sha256=sha(output / "ensembles.pt"),
        adapter_code_sha256=sha(Path(__file__)), source_files_sha256=before,
        source_files_unchanged=True)
    if {str(path.resolve()): sha(path) for path in protected} != before:
        raise RuntimeError("Read-only source fingerprints changed during adapter execution")
    write_json(output / "evaluation_metadata.json", metadata)
    write_json(output / "status.json", dict(status="complete", workflow_complete=True,
        returncode=0, physical_sampling_updates=0, labels_used_in_training=True,
        supervision_weight=10., supervision_scale=0, source_files_unchanged=True,
        cache_sha256=metadata["cache_sha256"], equilibrium_certified=False))
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    result = prepare(args.source_root, args.output)
    print(json.dumps(dict(status="complete", train_ensembles=len(result["train"]),
        test_ensembles=len(result["test"]), physical_sampling_updates=0,
        labels_used_in_training=True, cache_sha256=result["cache_sha256"])))
    return result


if __name__ == "__main__":
    main()
