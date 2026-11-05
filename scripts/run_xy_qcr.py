"""Own-session supervisor for the authorized fresh XY pilot; no remote retries."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from experiments.xy_qcr import validate_config
from experiments.phi4_hopping_cnn_qcr import GPU_UUID
from experiments.phi4_hopping_qcr import write_json,sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--receipt-prefix',required=True,type=Path);a=p.parse_args()
    c=validate_config(a.config);prefix=a.receipt_prefix.resolve()
    if a.output.exists() or any(Path(str(prefix)+s).exists() for s in ('.launch.json','.execution.json','.log')):raise FileExistsError('Fresh output required')
    if os.uname().nodename!='thinklab-105-225' or os.getpriority(os.PRIO_PROCESS,0)<10:raise RuntimeError('Wrong host/priority')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=GPU_UUID,CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
    command=[sys.executable,'-m','recurrent_rg','train','--config',str(a.config.resolve()),'--output',str(a.output.resolve())]
    started=time.monotonic();prefix.parent.mkdir(parents=True,exist_ok=True)
    with Path(str(prefix)+'.log').open('x') as f:
        child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        write_json(Path(str(prefix)+'.launch.json'),dict(command=command,cwd=str(Path.cwd()),pid=child.pid,supervisor_pid=os.getpid(),gpu_uuid=GPU_UUID,config_sha256=sha256_file(a.config),hard_seconds=c['hard_seconds'],started_unix=time.time()))
        timeout=False
        try:code=child.wait(timeout=c['hard_seconds'])
        except subprocess.TimeoutExpired:
            timeout=True;os.killpg(child.pid,signal.SIGTERM)
            try:code=child.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
    write_json(Path(str(prefix)+'.execution.json'),dict(returncode=code,budget_deadline_reached=timeout,elapsed_seconds=time.monotonic()-started,child_pid=child.pid,finished_unix=time.time()))
    return 3 if timeout else code


if __name__=='__main__':raise SystemExit(main())
