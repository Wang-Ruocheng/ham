"""One-shot detached launcher; refuses all existing output or launch receipts."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys


BASE = Path("/mnt/nas-new/home/yange/ham")
SNAPSHOT = BASE / "run_code/phi4_hopping_r_finefit_seed91_20261101_v1"
SOURCE = BASE / "results/phi4_hopping_b3_g1_kappa032_104_seed91_gpu3_vm225_20261031"
OUTPUT = BASE / "results/phi4_hopping_kappa032_104_frozenQC_Rfinefit_seed91_vm225_20261101_v1"
LOG = BASE / "logs/phi4_hopping_r_finefit_seed91_20261101_v1.log"
RECEIPT = LOG.with_suffix(".launch.json")
EXPECTED = {
    "experiments/phi4_hopping_r_finefit.py": "5a76c54f11dee410cbd1b05008147f022c47c77692aee530b09cf313e312040f",
    "recurrent_rg/phi4_r_calibration.py": "9a305ee67f18b94470c82ed8b5327e7a0c7245ea057193a18da8f657b5af7173",
}
SOURCE_EXPECTED = {
    "training/summary.json": "6a86cc4f6e8c08efd30e67cc5a83b1a85b3acc6b5b4ea2b02004e4fadfb022d3",
    "training/manifest.json": "5e0e863296aca5a89204362479d78678a213bedb3d24efbe955cf904ecb65523",
    "training/evaluation_arrays.npz": "37e7f914417badd61b0c6392a5737fd02e86c400033ff5377c616e53dc654a43",
    "public_files.json": "c9a04eae5cd09b48082e0011dfcf6989afb828e4e01c2f1bd9354feb8bf96ce7",
}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    if socket.gethostname() != "thinklab-105-225" or Path.cwd().resolve() != SNAPSHOT:
        raise RuntimeError("Unexpected host or code snapshot")
    if OUTPUT.exists() or LOG.exists() or RECEIPT.exists():
        raise RuntimeError("Existing output/log/receipt: refusing duplicate launch")
    code = {str(p.relative_to(SNAPSHOT)): sha(p) for p in SNAPSHOT.rglob("*.py")}
    if any(code.get(name) != value for name, value in EXPECTED.items()):
        raise RuntimeError("Frozen code fingerprint mismatch")
    if any(sha(SOURCE / name) != value for name, value in SOURCE_EXPECTED.items()):
        raise RuntimeError("Formal source public fingerprint mismatch")
    job = json.loads((SOURCE / "job.json").read_text())
    if not (job["status"] == "complete" and job["workflow_complete"] and job["returncode"] == 0):
        raise RuntimeError("Formal source incomplete")
    mem = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
                   if line.startswith("MemAvailable:"))) / 1024**2
    load = os.getloadavg()
    io = Path("/proc/pressure/io").read_text()
    full_avg10 = float(next(line.split("avg10=")[1].split()[0] for line in io.splitlines()
                           if line.startswith("full ")))
    if mem < 3 or load[0] > 1.25 * os.cpu_count() or full_avg10 >= 20:
        raise RuntimeError("CPU/memory/I/O resource gate not met; no child started")
    command = [str(BASE / ".env/bin/python"), "-u", "-m", "experiments.phi4_hopping_r_finefit",
               "--source-root", str(SOURCE), "--output", str(OUTPUT),
               "--font", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "--execute"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)
    with RECEIPT.open("x", encoding="utf-8") as receipt, LOG.open("xb") as log:
        child = subprocess.Popen(command, cwd=SNAPSHOT, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                 close_fds=True)
        value = dict(host=socket.gethostname(), started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     pid=child.pid, command=command, cwd=str(SNAPSHOT), source=str(SOURCE), output=str(OUTPUT),
                     code_sha256=code, source_public_sha256=SOURCE_EXPECTED,
                     cpu_budget_seconds=600, output_budget_bytes=64 * 1024**2, gpu_used=False,
                     resource_gate=dict(mem_available_gib=mem, load=load, io_full_avg10=full_avg10))
        json.dump(value, receipt, indent=2)
        receipt.write("\n")
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
