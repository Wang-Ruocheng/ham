"""Read-only CPU diagnostic, executed through SSH stdin without remote files.

No model imports, inference, sampling, coupling-label reads or source-data hash.
Only five-bin aggregate histograms are returned, never configurations.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import resource
import socket
import sys
import time

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

DATA = Path("/mnt/nas-new/home/yange/ham/data/ising2d_positiveK_L108_b3_replicas12_seed91.pt")
EXPECTED_BYTES = 9172950952
EXPECTED_SIDECAR_SHA = "0d2d7591b37aa27757bde1d1d964059d38b64ffb6374df1d3d7724993e801e2a"


def histograms(spins):
    """spins has shape [configuration, x, y], all entries exactly -1 or +1."""
    values = np.asarray(spins)
    if values.ndim != 3 or not np.isin(values, (-1, 1)).all():
        raise ValueError("Expected signed binary configurations, no silent recoding")
    values = values.astype(np.int8)
    field = (np.roll(values, 1, axis=-2) + np.roll(values, -1, axis=-2)
             + np.roll(values, 1, axis=-1) + np.roll(values, -1, axis=-1))
    t = values * field
    return np.bincount(((t.ravel() + 4) // 2).astype(np.int64), minlength=5)


def snapshot():
    mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(mem["MemAvailable"].split()[0]) * 1024
    full = next(line for line in Path("/proc/pressure/io").read_text().splitlines() if line.startswith("full "))
    io = float(next(part.split("=")[1] for part in full.split() if part.startswith("avg10=")))
    load = os.getloadavg()[0]
    cpus = os.cpu_count()
    return dict(mem_available_bytes=available, load1=load, cpus=cpus, io_full_avg10=io,
                ready=available >= 36 * 1024**3 and load <= 1.25 * cpus and io < 20)


def main():
    start_wall, start_cpu = time.monotonic(), time.process_time()
    result = dict(status="running", host=socket.gethostname(), source_data=str(DATA),
                  source_data_rehashed=False, labels_accessed=False, models_loaded=False,
                  model_inference=False, QCR_updates=0, GPU_used=False, MI_accessed=False,
                  remote_files_written=False, threads=1, nice=os.getpriority(os.PRIO_PROCESS, 0),
                  split=[], ensemble_index=[], group_index=[], replica_index=[], counts=[])
    if result["host"] != "thinklab-105-225" or result["nice"] != 19:
        raise ValueError("Host or priority mismatch")
    resource.setrlimit(resource.RLIMIT_CPU, (300, 305))
    wait_used = 0.

    def ready():
        nonlocal wait_used
        while True:
            snap = snapshot()
            if snap["ready"]:
                return snap
            print(json.dumps(dict(event="resource_wait", resources=snap, wait_used=wait_used)), file=sys.stderr, flush=True)
            if wait_used >= 180 or time.monotonic() - start_wall >= 590:
                raise RuntimeError("RESOURCE_HOLD: no further data reads")
            before = time.monotonic()
            time.sleep(10)
            wait_used += time.monotonic() - before

    try:
        result["resource_before"] = ready()
        before = DATA.stat()
        if before.st_size != EXPECTED_BYTES:
            raise ValueError("Data size mismatch")
        sidecar_sha = hashlib.sha256(DATA.with_suffix(".json").read_bytes()).hexdigest()
        if sidecar_sha != EXPECTED_SIDECAR_SHA:
            raise ValueError("Data sidecar fingerprint mismatch")
        result.update(source_bytes=before.st_size, source_mtime_ns=before.st_mtime_ns,
                      sidecar_sha256=sidecar_sha,
                      source_sha256_from_existing_receipt="f0e319a771d220e948c70e97fda8bbd373d46fdbb280211f0cbd29ce9d36de54")
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        payload = torch.load(str(DATA), map_location="cpu", weights_only=True, mmap=True)
        if isinstance(payload, (tuple, list)) and len(payload) == 4:
            configurations = (payload[0], payload[2])
        elif isinstance(payload, dict) and payload.get("schema_version") == 1:
            configurations = (payload["train"]["configurations"], payload["test"]["configurations"])
        else:
            raise ValueError("Unsupported original cache; no fallbacks")
        for data, n in zip(configurations, (516, 252)):
            if tuple(data.shape) != (n, 128, 108, 108, 1):
                raise ValueError("Configuration shape mismatch")
        collection_cpu = time.process_time()
        for split, data in enumerate(configurations):
            for index in range(len(data)):
                completed = len(result["counts"])
                if completed % 32 == 0:
                    ready()
                if time.process_time() - start_cpu > 295 or time.monotonic() - start_wall > 590:
                    raise RuntimeError("BUDGET_HOLD: do not reduce fixed sample count")
                rows = [histograms(data[index, sl, ..., 0].numpy()).tolist()
                        for sl in (slice(0, 64), slice(64, 128))]
                if any(sum(row) != 64 * 108**2 for row in rows):
                    raise ValueError("Histogram count mismatch")
                result["counts"].append(rows)
                result["split"].append(split)
                result["ensemble_index"].append(index)
                result["group_index"].append(index // 12)
                result["replica_index"].append(index % 12)
                if completed == 2:
                    per = (time.process_time() - collection_cpu) / 3
                    projected = time.process_time() - start_cpu + per * 765 * 2 + 30
                    result["preflight"] = dict(ensembles=3, cpu_seconds_per_ensemble=per,
                                               projected_cpu_seconds_2x_margin=projected, passed=projected <= 300)
                    if projected > 300:
                        raise RuntimeError("BUDGET_HOLD: fixed diagnostic projection exceeds 300 CPU seconds")
                if (completed + 1) % 96 == 0:
                    print(json.dumps(dict(event="progress", completed=completed + 1,
                                          cpu_seconds=time.process_time() - start_cpu)), file=sys.stderr, flush=True)
        after = DATA.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Source data metadata changed during diagnostic")
        result.update(status="complete", workflow_complete=True, resource_after=snapshot(),
                      source_metadata_unchanged=True, bins=[-4, -2, 0, 2, 4])
    except Exception as error:
        result.update(status="resource_hold" if "RESOURCE_HOLD" in str(error) else
                      "budget_hold" if "BUDGET_HOLD" in str(error) else "failed",
                      workflow_complete=False, error=f"{type(error).__name__}: {error}")
    result.update(completed_ensembles=len(result["counts"]), cpu_seconds=time.process_time() - start_cpu,
                  wall_seconds=time.monotonic() - start_wall, resource_wait_seconds=wait_used,
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
