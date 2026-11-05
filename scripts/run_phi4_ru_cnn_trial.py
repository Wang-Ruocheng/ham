"""Two-hour supervisor for the explicit 6000-update ru CNN trial only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts.run_phi4_cnn_refined_trial import write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--receipt-prefix',type=Path,required=True)
    args=p.parse_args()
    config=json.loads(args.config.read_text())
    if config.get('recipe')!='phi4_ru_b3_cnn_g2_Rlr003_v1' or config.get('hard_seconds')!=7200:
        raise ValueError('Only the explicit ru CNN two-hour preset is supported')
    prefix=args.receipt_prefix.resolve()
    if args.output.exists() or any(Path(str(prefix)+s).exists() for s in ('.launch.json','.execution.json','.log')):
        raise FileExistsError('Fresh output and receipt paths required')
    if os.uname().nodename!='thinklab-105-225' or os.getpriority(os.PRIO_PROCESS,0)<10:
        raise RuntimeError('Wrong host or priority')
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='GPU-a167decd-ce49-6c8f-6bcf-136fde596683',CUDA_DEVICE_ORDER='PCI_BUS_ID',
               OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
               PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
    command=[sys.executable,'-m','recurrent_rg','train','--config',str(args.config.resolve()),'--output',str(args.output.resolve())]
    prefix.parent.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    with Path(str(prefix)+'.log').open('x') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        write(Path(str(prefix)+'.launch.json'),dict(command=command,cwd=str(Path.cwd()),pid=child.pid,
            supervisor_pid=os.getpid(),host=os.uname().nodename,started_unix=time.time(),hard_seconds=7200,
            config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),gpu_uuid=env['CUDA_VISIBLE_DEVICES'],
            nice=os.getpriority(os.PRIO_PROCESS,0)))
        timeout=False
        try:code=child.wait(timeout=max(0,7200-(time.monotonic()-started)))
        except subprocess.TimeoutExpired:
            timeout=True
            os.killpg(child.pid,signal.SIGTERM)
            try:code=child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);code=child.wait()
    write(Path(str(prefix)+'.execution.json'),dict(returncode=code,budget_deadline_reached=timeout,
        elapsed_seconds=time.monotonic()-started,child_pid=child.pid,finished_unix=time.time()))
    return 3 if timeout else code


if __name__=='__main__':
    raise SystemExit(main())
