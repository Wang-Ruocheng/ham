"""Explicit overlap-to-QCR adapter; temperature/bonds stay in an evaluation sidecar."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from recurrent_rg.data.ensembles import EnsembleDataset, load_ensembles
from recurrent_rg.data.spin_glass_pipeline import iter_spin_glass, _sha, _json, _bond_sha


def prepare_overlap_cache(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    status = json.loads((source / "status.json").read_text())
    if not status.get("workflow_complete") or status.get("status") != "complete":
        raise ValueError("Physical spin-glass data must be complete")
    xs, groups, rows = {s: [] for s in ("train", "test")}, {}, {}
    for split in xs:
        groups[split], rows[split] = [], []
    geometry = None
    dimension = None
    lattice_shape = None
    for record in iter_spin_glass(source):
        record_dimension = record.configurations.ndim - 3
        if record_dimension not in (1, 2):
            raise ValueError("This learning adapter supports spatially one- and two-dimensional EA only")
        if record.configurations.shape[1] != 2:
            raise ValueError("Use exactly two real replicas: one predeclared overlap pair per disorder")
        q = record.overlap_view(0, 1)
        if q.shape[1] < 4 or q.shape[1] % 2:
            raise ValueError("Training-aligned half-Q needs an even sample count >=4")
        if geometry is None:
            geometry = q.shape[1:]
            dimension = record_dimension
            lattice_shape = list(q.shape[2:-1])
        if q.shape[1:] != geometry:
            raise ValueError("All overlap ensembles must have identical sample/lattice geometry")
        if record_dimension != dimension:
            raise ValueError("All overlap ensembles must have the same spatial dimension")
        physical_metadata = dict(dimension=dimension, lattice_shape=lattice_shape,
            zero_bond_count=int(np.count_nonzero(record.bonds == 0)))
        if dimension == 1:
            physical_metadata["ring_flux"] = int(np.prod(np.sign(record.bonds)))
        else:
            # Each elementary square has a gauge-invariant product of its
            # four edge signs. Unlike 1D, local frustration cannot be removed.
            bx, by = np.sign(record.bonds)
            plaquettes = bx * np.roll(by, -1, axis=0) * np.roll(bx, -1, axis=1) * by
            physical_metadata["frustrated_plaquette_fraction"] = float(np.mean(plaquettes < 0))
            physical_metadata["zero_plaquette_fraction"] = float(np.mean(plaquettes == 0))
        for t, temperature in enumerate(record.temperatures):
            xs[record.split].append(torch.from_numpy(q[t].copy()))
            groups[record.split].append([record.disorder_id, t])
            rows[record.split].append(dict(disorder_id=record.disorder_id,
                temperature_index=t, temperature=float(temperature),
                bond_sha256=_bond_sha(record.bonds), overlap_replicas=[0, 1],
                **physical_metadata, samples=q.shape[1], sites=int(np.prod(lattice_shape))))
    datasets = {}
    for split in xs:
        if not xs[split]:
            raise ValueError("Both disorder-held-out splits must be nonempty")
        datasets[split] = EnsembleDataset(torch.stack(xs[split]), couplings=None,
            group_ids=torch.tensor(groups[split], dtype=torch.long)).validate()
    train_ids = {r["disorder_id"] for r in rows["train"]}
    test_ids = {r["disorder_id"] for r in rows["test"]}
    if train_ids & test_ids:
        raise ValueError("Disorder leakage in training adapter")
    output.mkdir(parents=True, exist_ok=False)
    torch.save(dict(schema_version=1, **{s: asdict(d) for s, d in datasets.items()}),
               output / "ensembles.pt")
    # Round-trip through the actual training loader; no duplicate T labels are
    # invented to work around its deliberately stricter historical convention.
    load_ensembles(output / "ensembles.pt")
    metadata = dict(schema_version=1, source_dataset=str(source),
        dimension=dimension, lattice_shape=lattice_shape,
        source_manifest_sha256=_sha(source / "manifest.json"),
        cache_sha256=_sha(output / "ensembles.pt"), **rows,
        representation="q_x=s_x^a*s_x^b; same J, temperature, saved time; real replicas 0/1",
        couplings_in_training_cache=False, labels_used_in_training=False,
        grouping="Whole disorder held out; [disorder_id,temperature_index] identifies an ensemble",
        qualification="Same-J temperatures are not independent disorder samples; no equilibrium certification")
    _json(output / "evaluation_metadata.json", metadata)
    return metadata


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    metadata = prepare_overlap_cache(args.source, args.output)
    print(json.dumps(dict(train_ensembles=len(metadata["train"]),
        test_ensembles=len(metadata["test"]), labels_used_in_training=False)))


if __name__ == "__main__":
    main()
