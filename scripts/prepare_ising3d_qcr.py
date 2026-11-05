"""将新三维 Ising 数据转为无耦合监督标签的 QCR cache；K 只留在公开评价 JSON。"""
from dataclasses import asdict
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from recurrent_rg.data.ensembles import EnsembleDataset, load_ensembles
from recurrent_rg.ising3d import (SCHEMA, QUALIFICATION, Wolff3DOptions, chain_seed,
    energy_3d, sha256_file, write_json)


def prepare_cache(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    status = json.loads((source / "status.json").read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    plan = json.loads((source / "plan.json").read_text())
    if status.get("status") != "complete" or status.get("workflow_complete") is not True:
        raise ValueError("Physical source must be complete")
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "complete" or manifest.get("dimension") != 3:
        raise ValueError("Require completed new 3D Wolff source")
    if status.get("manifest_sha256") != sha256_file(source / "manifest.json"):
        raise ValueError("Physical manifest fingerprint mismatch")
    if manifest.get("plan_sha256") != sha256_file(source / "plan.json") or manifest.get("sampling_sha256") != sha256_file(source / "sampling.json"):
        raise ValueError("Physical plan/sampling fingerprint mismatch")
    if plan.get("options") != manifest.get("options") or manifest.get("equilibrium_certified") is not False:
        raise ValueError("Physical options mismatch or unsupported equilibrium claim")
    if plan.get("kernel_source_sha256") != manifest.get("kernel_source_sha256"):
        raise ValueError("Physical kernel fingerprint receipt mismatch")
    options = manifest["options"]
    physical_options = Wolff3DOptions(**options).validate()
    xs, groups, rows = ({split: [] for split in ("train", "test")} for _ in range(3))
    seen = set()
    size, samples, chains = options["size"], options["samples"], options["chains"]
    for shard in manifest["shards"]:
        index = shard["coupling_index"]
        if index in seen or index < 0 or index >= len(options["couplings"]):
            raise ValueError("Duplicate or out-of-range coupling index")
        seen.add(index)
        split = "test" if index % options["test_every"] == options["test_offset"] else "train"
        if shard["split"] != split or shard["K"] != options["couplings"][index]:
            raise ValueError("Whole-K split or coupling metadata mismatch")
        path, sidepath = source / shard["path"], source / shard["sidecar"]
        if not path.resolve().is_relative_to(source) or not sidepath.resolve().is_relative_to(source):
            raise ValueError("Physical shard escapes source root")
        if sha256_file(path) != shard["sha256"] or sha256_file(sidepath) != shard["sidecar_sha256"]:
            raise ValueError("Physical shard or sidecar fingerprint mismatch")
        sidecar = json.loads(sidepath.read_text())
        if sidecar.get("raw_diagnostics_before_global_z2") is not True or sidecar.get("equilibrium_certified") is not False:
            raise ValueError("Missing raw pre-augmentation diagnostics qualification")
        if sidecar.get("global_z2_augmentation") != options["augment_global_z2"] or len(sidecar.get("chains", [])) != chains:
            raise ValueError("Global symmetry augmentation or physical chain count mismatch")
        if (sidecar.get("coupling_index") != index or sidecar.get("split") != split
                or sidecar.get("K") != shard["K"] or sidecar.get("dimension") != 3
                or sidecar.get("size") != size or sidecar.get("status") != "complete"
                or sidecar.get("data_sha256") != shard["sha256"]):
            raise ValueError("Physical sidecar metadata mismatch")
        with np.load(path, allow_pickle=False) as arrays:
            configurations = arrays["configurations"]
            raw = arrays["raw_observables"]
            flips = arrays["augmentation_flips"]
            cluster_sizes = arrays["cluster_sizes"]
            if configurations.shape != (chains, samples, size, size, size) or configurations.dtype != np.int8:
                raise ValueError("Unexpected 3D source spin geometry or dtype")
            if raw.shape != (chains, samples, 4) or flips.shape != (chains, samples):
                raise ValueError("Raw diagnostic/augmentation geometry mismatch")
            expected_clusters = options["burn_in_clusters"] + samples * options["clusters_between"]
            if (cluster_sizes.shape != (chains, expected_clusters) or cluster_sizes.dtype != np.int64
                    or not np.all((cluster_sizes >= 1) & (cluster_sizes <= size ** 3))):
                raise ValueError("Physical cluster-size work receipt mismatch")
            if not np.isfinite(raw).all() or not np.all((flips == -1) | (flips == 1)):
                raise ValueError("Invalid raw diagnostics or global Z2 signs")
            if not options["augment_global_z2"] and not np.all(flips == 1):
                raise ValueError("Unexpected global symmetry flips in unaugmented source")
            if not np.all((configurations == -1) | (configurations == 1)):
                raise ValueError("Source spins are not binary")
            if not np.array_equal(energy_3d(configurations) / size ** 3, raw[..., 0]):
                raise ValueError("Physical energy consistency check failed")
            raw_m = configurations.mean(axis=(-3, -2, -1)) * flips
            if not np.array_equal(raw_m, raw[..., 1]) or not np.array_equal(raw_m ** 2, raw[..., 2]) or not np.array_equal(abs(raw_m), raw[..., 3]):
                raise ValueError("Raw pre-augmentation magnetization mismatch")
            for chain in range(chains):
                xs[split].append(torch.from_numpy(configurations[chain].copy()).unsqueeze(-1))
                groups[split].append(index)
                row = sidecar["chains"][chain]
                if (row.get("chain_index") != chain or row.get("cluster_proposals") != expected_clusters
                        or row.get("cluster_visited_sites") != int(cluster_sizes[chain].sum())
                        or row.get("diagnostics_before_global_z2") is not True):
                    raise ValueError("Per-chain work/diagnostic receipt mismatch")
                expected_initial = "hot" if chain < chains // 2 else "cold"
                expected_sign = 1 if (chain - chains // 2) % 2 == 0 else -1
                if (row.get("initial") != expected_initial or row.get("cold_sign") != expected_sign
                        or row.get("thermal_seed") != chain_seed(physical_options, index, chain, "thermal")
                        or row.get("augmentation_seed") != chain_seed(physical_options, index, chain, "global_z2")):
                    raise ValueError("Physical initialization or seed namespace mismatch")
                rows[split].append(dict(coupling_index=index, K=float(shard["K"]), chain_index=chain,
                    initial=row["initial"], cold_sign=row["cold_sign"], samples=samples,
                    size=size, dimension=3, sites=size ** 3, thermal_seed=row["thermal_seed"],
                    augmentation_seed=row["augmentation_seed"], source_data_sha256=shard["sha256"],
                    raw_diagnostics_before_global_z2=True, independent_physical_chain=True,
                    equilibrium_certified=False))
    if seen != set(range(len(options["couplings"]))):
        raise ValueError("Physical source has missing coupling conditions")
    datasets = {split: EnsembleDataset(torch.stack(xs[split]), couplings=None,
        group_ids=torch.tensor(groups[split], dtype=torch.long)).validate() for split in xs}
    if set(groups["train"]) & set(groups["test"]):
        raise ValueError("Whole-K train/test leakage")
    output.mkdir(parents=True, exist_ok=False)
    torch.save(dict(schema_version=1, **{split: asdict(dataset) for split, dataset in datasets.items()}), output / "ensembles.pt")
    load_ensembles(output / "ensembles.pt")
    metadata = dict(schema_version=1, physical_schema=SCHEMA, dimension=3, lattice_shape=[size] * 3,
        representation="s_x (physical Ising spins), not replica-overlap fields", **rows,
        source_dataset=str(source), source_manifest_sha256=sha256_file(source / "manifest.json"),
        cache_sha256=sha256_file(output / "ensembles.pt"), adapter_code_sha256=sha256_file(Path(__file__)),
        source_sampling_sha256=manifest["sampling_sha256"], couplings_in_training_cache=False,
        labels_used_in_training=False, split_unit="entire K index including all physical chains",
        independent_physical_chains_per_K=chains, equilibrium_certified=False, qualification=QUALIFICATION)
    write_json(output / "evaluation_metadata.json", metadata)
    write_json(output / "status.json", dict(status="complete", workflow_complete=True, returncode=0,
        cache_sha256=metadata["cache_sha256"], labels_used_in_training=False, error=None))
    return metadata


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    metadata = prepare_cache(args.source, args.output)
    print(json.dumps(dict(status="complete", train_ensembles=len(metadata["train"]),
        test_ensembles=len(metadata["test"]), labels_used_in_training=False), ensure_ascii=False))
    return metadata


if __name__ == "__main__":
    main()
