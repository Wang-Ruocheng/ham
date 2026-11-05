"""用户授权的无监督改进序列第一阶段；独占输出、GPU等待、顺序执行。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from scripts.run_ising_response_loss_ablation import fingerprint, save_json, utc_now


CODE_FILES = (
    'experiments/prepare_ising_physical_response.py',
    'experiments/ising_physical_response_ablation.py',
    'scripts/report_ising_physical_response.py',
    'scripts/run_ising_physical_response_sequence.py',
    'experiments/ising_response_loss_ablation.py',
    'scripts/diagnose_ising_mcrg.py', 'scripts/diagnose_ising_dimg_response.py',
    'scripts/report_ising_dimg_ablation.py', 'scripts/report_ising_zh.py',
    'recurrent_rg/checkpoints.py', 'recurrent_rg/model_nd.py', 'recurrent_rg/losses.py',
)


def gpu_idle(device):
    index = int(device.split(':')[1])
    uuid = subprocess.check_output(['nvidia-smi',f'--id={index}',
        '--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid',
        '--format=csv,noheader,nounits'],text=True)
    return not any(uuid in row for row in processes.splitlines())


def plan(args):
    source, output = args.source_root.resolve(), args.output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError('拒绝覆盖、重复启动或恢复已有实验根')
    if output == source or output in source.parents or source in output.parents:
        raise ValueError('新实验根须与历史源分离')
    if args.device not in ('cuda:2','cuda:3') or args.updates < 1:
        raise ValueError('仅使用已检查的225 GPU2/3及正预算，不干预GPU0/1')
    source_job = json.loads((source/'job.json').read_text())
    if source_job.get('status') != 'complete' or source_job.get('workflow_complete') is not True:
        raise ValueError('dimg消融源尚未完整完成')
    project=Path(__file__).resolve().parents[1]
    commands=[]
    source_hashes={}
    for dim in (1,3):
        run=source/f'dim{dim}/run'
        for name in ('training_step_12000.pt','manifest.json','summary.json'):
            path=run/name; source_hashes[str(path)]=fingerprint(path)
        folder=output/f'phase1/dim{dim}'
        prepare=[sys.executable,'-u','-m','experiments.prepare_ising_physical_response',
            '--source-run',str(run),'--output',str(folder/'cache'),'--device',args.device,
            '--c-repeats','5','--seed','20261016',
            '--expected-checkpoint-sha256',source_hashes[str(run/'training_step_12000.pt')]]
        train=[sys.executable,'-u','-m','experiments.ising_physical_response_ablation',
            '--source-run',str(run),'--cache-path',str(folder/'cache/response_cache.pt'),
            '--output',str(folder/'training'),'--device',args.device,
            '--updates',str(args.updates),'--learning-rate','.001']
        report=[sys.executable,'-u','-m','scripts.report_ising_physical_response',
            '--run',str(folder),'--output',str(folder/'report'),'--updates',str(args.updates)]
        if args.smoke:
            prepare.append('--smoke'); train.append('--smoke')
        stages=[('prepare',prepare,True),('training',train,True)]
        if not args.smoke: stages.append(('report',report,False))
        commands.extend(dict(name=f'dim{dim}_{name}',dim=dim,command=command,uses_gpu=gpu)
                        for name,command,gpu in stages)
    return dict(schema_version=1,created_at=utc_now(),source_root=str(source),output=str(output),
        device=args.device,updates=3 if args.smoke else args.updates,smoke=args.smoke,
        code_sha256={str(project/name):fingerprint(project/name) for name in CODE_FILES},
        source_sha256=source_hashes,commands=commands,
        source_epoch=12000,labels_used_in_training=False,Kc_or_exponent_target=False,
        ordered_experiments=[
            dict(stage=1,name='冻结Q/C的R物理响应对照',dimensions=[1,3],arms=['control','response'],
                 updates_per_arm=3 if args.smoke else args.updates,c_draws=1 if args.smoke else 5),
            dict(stage=2,name='Q的自然耦合条件似然表示消融',status='pending_stage1_review',
                 rule='第一阶段完成后分析响应拟合与谱，再实现独立目录的新架构消融；不能把已有临界TEST诊断回灌训练。'),
            dict(stage=3,name='独立初始尺寸与相同末尺度匹配',status='pending_stage2_review',
                 rule='使用独立物理系综验证尺度效应；不把当前108→54→27链伪装独立尺寸。')],
        budget_rule='不以理论指数接近程度挑根、快照、训练长度或响应权重；阶段间先分析，不盲目扩预算。',
        immutable_sources=True,physical_sampling=False,MI_estimation=False)


def execute(args):
    specification=plan(args)
    root=Path(specification['output']); root.mkdir(parents=True,exist_ok=False)
    (root/'logs').mkdir(); (root/'phase1').mkdir()
    save_json(root/'plan.json',specification)
    job=dict(status='running',workflow_complete=False,parent_pid=os.getpid(),
        started_at=utc_now(),current=None,stages=[],phase1_complete=False,
        pending_experiments=[2,3],requires_user_action=False)
    path=root/'job.json'; save_json(path,job)
    try:
        for entry in specification['commands']:
            for source,digest in {**specification['source_sha256'],**specification['code_sha256']}.items():
                if fingerprint(source)!=digest:raise ValueError(f'来源或代码指纹变化：{source}')
            if entry['uses_gpu']:
                consecutive=0
                while consecutive<2:
                    idle=gpu_idle(args.device)
                    consecutive=consecutive+1 if idle else 0
                    job['current']=dict(name='wait_gpu',next_stage=entry['name'],device=args.device,
                                        consecutive_idle_checks=consecutive)
                    save_json(path,job)
                    if consecutive<2:time.sleep(10)
            log=root/'logs'/f'{entry["name"]}.log'
            started=time.monotonic()
            with log.open('x') as stream:
                child=subprocess.Popen(entry['command'],stdout=stream,stderr=subprocess.STDOUT)
                stage={**entry,'pid':child.pid,'status':'running','log':str(log),'started_at':utc_now()}
                job['current']=stage; save_json(path,job)
                code=child.wait()
            stage.update(returncode=code,status='complete' if code==0 else 'failed',
                         finished_at=utc_now(),elapsed_seconds=time.monotonic()-started)
            job['stages'].append(stage);job['current']=None;save_json(path,job)
            if code:raise RuntimeError(f'{entry["name"]} returncode={code}；保留结果，不自动修复或重跑')
        reports={}
        for dim in (1,3):
            train_job=json.loads((root/f'phase1/dim{dim}/training/training_job.json').read_text())
            if train_job.get('status')!='complete':raise ValueError('训练完成标志缺失')
            if not args.smoke:
                receipt=json.loads((root/f'phase1/dim{dim}/report/report.json').read_text())
                if receipt.get('status')!='complete' or receipt.get('report_complete') is not True:
                    raise ValueError('正式报告不完整')
                reports[f'dim{dim}']=json.loads((root/f'phase1/dim{dim}/report/comparison.json').read_text())
        if not args.smoke:
            save_json(root/'phase1/comparison.json',dict(status='complete',reports=reports,
                rule='先检验留出物理响应拟合是否改善，再解释完整谱；不按nu接近1决定结果。'))
        job.update(status='smoke_complete' if args.smoke else 'phase1_complete',current=None,
            phase1_complete=not args.smoke,awaiting_scientific_review=not args.smoke,finished_at=utc_now(),
            next_experiment=None if args.smoke else 2,requires_user_action=False)
        save_json(path,job)
    except BaseException as error:
        job.update(status='failed',requires_user_action=True,error=f'{type(error).__name__}: {error}',finished_at=utc_now())
        save_json(path,job);traceback.print_exc();return 1
    print(json.dumps(job,ensure_ascii=False),flush=True)
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:2')
    parser.add_argument('--updates',type=int,default=6000)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--detach',action='store_true')
    args=parser.parse_args()
    specification=plan(args)
    if args.dry_run:print(json.dumps(specification,ensure_ascii=False,indent=2));return 0
    if args.detach:
        logs=Path.cwd()/'logs';logs.mkdir(exist_ok=True)
        receipt=logs/f'{args.output.name}.launch.json';log=logs/f'{args.output.name}.log'
        command=['nice','-n','10',sys.executable,'-u','-m','scripts.run_ising_physical_response_sequence',
            '--source-root',str(args.source_root.resolve()),'--output',str(args.output.resolve()),
            '--device',args.device,'--updates',str(args.updates)]
        if args.smoke:command.append('--smoke')
        with receipt.open('x') as stream:
            json.dump(dict(status='launching',command=command,created_at=utc_now()),stream,ensure_ascii=False)
        environment={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MPLBACKEND':'Agg'}
        with log.open('x') as stream:
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                start_new_session=True,env=environment)
        result=dict(status='launched',parent_pid=child.pid,output=str(args.output.resolve()),
            log=str(log),command=command,launched_at=utc_now())
        save_json(receipt,result);print(json.dumps(result,ensure_ascii=False,indent=2));return 0
    return execute(args)


if __name__=='__main__':raise SystemExit(main())
