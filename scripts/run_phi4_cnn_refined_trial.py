"""External 20-minute supervisor for one explicit fresh continuous CNN trial.

Only the process group created here can be terminated at the declared deadline.
Launch/execution receipts remain outside the scientific output directory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def write(path, value):
    temporary=path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    os.replace(temporary,path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--config',type=Path)
    mode.add_argument('--source',type=Path,help='Explicit pinned R-only continuation, not a new joint run')
    parser.add_argument('--receipt-prefix',type=Path,required=True)
    args=parser.parse_args()
    prefix=args.receipt_prefix.resolve()
    if args.output.exists() or any(Path(str(prefix)+s).exists() for s in ('.launch.json','.execution.json','.log')):
        raise FileExistsError('Fresh output and receipts required')
    if os.uname().nodename!='thinklab-105-225' or os.getpriority(os.PRIO_PROCESS,0)<10:
        raise RuntimeError('Wrong host or priority')
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='GPU-a167decd-ce49-6c8f-6bcf-136fde596683',
               CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1',
               PYTHONDONTWRITEBYTECODE='1')
    if args.source is not None:
        command=[sys.executable,'-m','experiments.rsmi_separate_qcr','--phi4-cnn-r-only',
                 '--source',str(args.source.resolve()),'--output',str(args.output.resolve()),'--device','cuda:0']
        pinned_file=args.source/'latest.pt'
    else:
        command=[sys.executable,'-m','recurrent_rg','train','--config',str(args.config.resolve()),'--output',str(args.output.resolve())]
        pinned_file=args.config
    prefix.parent.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    with Path(str(prefix)+'.log').open('x') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        launch=dict(command=command,cwd=str(Path.cwd()),pid=child.pid,supervisor_pid=os.getpid(),
                    host=os.uname().nodename,started_unix=time.time(),hard_seconds=1200,
                    input_path=str(pinned_file.resolve()), input_sha256=hashlib.sha256(pinned_file.read_bytes()).hexdigest(),
                    gpu_uuid=env['CUDA_VISIBLE_DEVICES'],nice=os.getpriority(os.PRIO_PROCESS,0))
        write(Path(str(prefix)+'.launch.json'),launch)
        timeout=False
        try:
            code=child.wait(timeout=max(0,1200-(time.monotonic()-started)))
        except subprocess.TimeoutExpired:
            timeout=True
            os.killpg(child.pid,signal.SIGTERM)
            try:
                code=child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL)
                code=child.wait()
    write(Path(str(prefix)+'.execution.json'),dict(returncode=code,budget_deadline_reached=timeout,
          elapsed_seconds=time.monotonic()-started,child_pid=child.pid,finished_unix=time.time()))
    return code if not timeout else 3


if __name__=='__main__':
    raise SystemExit(main())
