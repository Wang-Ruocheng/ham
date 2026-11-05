"""Supervise only this new child's session, with a complete 1200-second cap."""
import argparse
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from experiments.ising_kb_supervised_short import validate_config
from experiments.phi4_hopping_cnn_qcr import GPU_UUID
from scripts.run_phi4_cnn_refined_trial import write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--receipt-prefix',type=Path,required=True);a=p.parse_args();validate_config(a.config)
    prefix=a.receipt_prefix.resolve()
    if a.output.exists() or any(Path(str(prefix)+s).exists() for s in ('.launch.json','.execution.json','.log')):
        raise FileExistsError('Fresh independent output and receipts required')
    if os.uname().nodename!='thinklab-105-225' or os.getpriority(os.PRIO_PROCESS,0)<10:
        raise RuntimeError('Wrong host/priority')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=GPU_UUID,CUDA_DEVICE_ORDER='PCI_BUS_ID',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
        PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
    command=[sys.executable,'-m','recurrent_rg','train','--config',str(a.config.resolve()),'--output',str(a.output.resolve())]
    prefix.parent.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    with Path(str(prefix)+'.log').open('x') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        write(Path(str(prefix)+'.launch.json'),dict(command=command,cwd=str(Path.cwd()),pid=child.pid,
            supervisor_pid=os.getpid(),host=os.uname().nodename,started_unix=time.time(),hard_seconds=1200,
            config_sha256=hashlib.sha256(a.config.read_bytes()).hexdigest(),gpu_uuid=GPU_UUID,nice=os.getpriority(os.PRIO_PROCESS,0)))
        timeout=False
        try:code=child.wait(timeout=max(0,1200-(time.monotonic()-started)))
        except subprocess.TimeoutExpired:
            timeout=True;os.killpg(child.pid,signal.SIGTERM)
            try:code=child.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
    write(Path(str(prefix)+'.execution.json'),dict(returncode=code,budget_deadline_reached=timeout,
        elapsed_seconds=time.monotonic()-started,child_pid=child.pid,finished_unix=time.time()))
    return 3 if timeout else code


if __name__=='__main__':raise SystemExit(main())
