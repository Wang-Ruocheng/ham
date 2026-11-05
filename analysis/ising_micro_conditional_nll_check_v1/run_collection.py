"""Invoke a read-only stdin diagnostic; retain only public count statistics locally."""
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results/ising_micro_conditional_nll_check_matched9000_seed20261017"


def main():
    if OUT.exists():
        raise FileExistsError("Do not overwrite a diagnostic run")
    protocol = (HERE / "protocol.json").read_bytes()
    code = (HERE / "collect_histograms.py").read_bytes()
    command = ["ssh", "-F", "/private/tmp/sjtu-vm.SiBJPz/config", "-o", "BatchMode=yes",
               "-o", "ConnectTimeout=15", "225",
               "cd /mnt/nas-new/home/yange/ham && nice -n 19 .env/bin/python -"]
    OUT.mkdir(parents=True)
    (OUT / "protocol.json").write_bytes(protocol)
    receipt = dict(command=command, collector_sha256=hashlib.sha256(code).hexdigest(),
                   protocol_sha256=hashlib.sha256(protocol).hexdigest(), remote_files_written=False)
    with (OUT / "collection.log").open("w") as progress:
        process = subprocess.run(command, input=code, stdout=subprocess.PIPE, stderr=progress, timeout=640)
    receipt["ssh_returncode"] = process.returncode
    if process.returncode == 0:
        payload = json.loads(process.stdout)
        (OUT / "histograms.json").write_bytes(process.stdout)
        receipt.update(histograms_sha256=hashlib.sha256(process.stdout).hexdigest(),
                       diagnostic_status=payload["status"], completed_ensembles=payload["completed_ensembles"])
    (OUT / "transport_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
