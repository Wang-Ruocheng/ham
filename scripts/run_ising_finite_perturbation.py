"""有限物理扰动 × 可靠方向均衡：独立、固定预算的无标签 R-only 实验。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone


NEW_FILES = ('scripts/ising_finite_perturbation_data.py',
    'experiments/prepare_ising_finite_perturbation.py',
    'experiments/ising_finite_perturbation_ablation.py',
    'scripts/report_ising_finite_perturbation.py',
    'scripts/run_ising_finite_perturbation.py')
DEPENDENCIES = ('experiments/ising_physical_response_ablation.py',
    'experiments/prepare_ising_physical_response.py',
    'experiments/ising_response_loss_ablation.py',
    'experiments/ising_coupling_representation.py',
    'scripts/report_ising_coupling_representation.py',
    'scripts/diagnose_ising_mcrg.py', 'scripts/diagnose_ising_dimg_response.py',
    'scripts/prepare_ising2d_qcr.py', 'scripts/prepare_ising_matched_sizes.py',
    'scripts/report_ising_zh.py', 'recurrent_rg/model_nd.py',
    'recurrent_rg/checkpoints.py', 'recurrent_rg/binary_pooling.py',
    'recurrent_rg/coupling_decoder.py', 'recurrent_rg/geometry.py',
    'recurrent_rg/adapters.py', 'recurrent_rg/losses.py', 'recurrent_rg/config.py',
    'recurrent_rg/representation.py','scripts/report_ising_dimg_ablation.py')
ARMS = ('A', 'B', 'C', 'D', 'E')


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    pending = path.with_name(path.name + '.writing')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    pending.replace(path)


def fingerprint(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'须为普通、非链接来源文件：{path}')
    if path.stat().st_size > 32 * 2**20:
        raise ValueError(f'禁止在来源清单中读取大型物理数据或缓存：{path}')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def resources(device='cpu'):
    lines = Path('/proc/meminfo').read_text().splitlines()
    memory = int(next(x.split()[1] for x in lines if x.startswith('MemAvailable:'))) * 1024
    full = next(x for x in Path('/proc/pressure/io').read_text().splitlines() if x.startswith('full '))
    pressure = float(next(x.split('=')[1] for x in full.split() if x.startswith('avg10=')))
    cores, load = os.cpu_count() or 1, os.getloadavg()[0]
    result = dict(checked_at=now(), available_memory_bytes=memory, load1=load,
                  cpu_count=cores, io_full_avg10=pressure,
                  ready=memory >= 36*2**30 and load <= 1.25*cores and pressure < 20.)
    if device.startswith('cuda:'):
        index = int(device.split(':')[1])
        if index not in (2, 3):
            raise ValueError('只允许明确指定GPU2或3，不使用GPU0/1')
        query = subprocess.run(['nvidia-smi', f'--id={index}',
            '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'],
            check=True, capture_output=True, text=True).stdout.strip().split(',')
        uuid, free = query[0].strip(), int(query[1].strip())
        active = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
            '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True).stdout
        processes = [line for line in active.splitlines() if line.split(',')[0].strip() == uuid]
        result.update(gpu=index, gpu_uuid=uuid, gpu_free_mib=free, gpu_processes=processes)
        result['ready'] = result['ready'] and free >= 12*1024 and not processes
    return result


def child_environment():
    return {**os.environ, 'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1',
        'OPENBLAS_NUM_THREADS':'1', 'NUMBA_NUM_THREADS':'1', 'MPLBACKEND':'Agg'}


def validate_source(source_root, previous_root):
    source, previous = Path(source_root).resolve(), Path(previous_root).resolve()
    old, job = read(previous/'job.json'), read(source/'job.json')
    if (old.get('status') != 'complete' or old.get('workflow_complete') is not True
            or old.get('smoke_only') is not False or old.get('requires_user_action')):
        raise ValueError('需要已完成正式第三阶段；不能恢复旧任务或用smoke代替')
    if job.get('status') != 'complete' or job.get('workflow_complete') is not True:
        raise ValueError('第二阶段未完成')
    arm = 'dim3_control'
    run = source/arm/'run'
    summary, manifest = read(run/'summary.json'), read(run/'manifest.json')
    cfg = summary['config']
    row = job['training'][arm]
    if (row.get('actual_epochs') != 12000 or row.get('labels_used_in_training') is not False
            or summary.get('actual_epochs') != 12000 or cfg.get('schema_version') != 5
            or cfg['model'].get('theory_dim') != 3 or cfg['model'].get('block_factor') != 2
            or cfg['model'].get('width') != 3 or cfg['model'].get('hidden') != 96
            or cfg['model'].get('fixed_kernel') is not False
            or cfg['model'].get('kernel_initialization') != 'random'
            or cfg['model'].get('kernel_layout') != 'nonoverlapping'
            or cfg['training'].get('max_rg_steps') != 2
            or manifest.get('labels_used_in_training') is not False):
        raise ValueError('来源必须为明确历史b2、随机可训练C、无监督dim3原表示12000终点')
    checkpoint = run/'training_step_12000.pt'
    if fingerprint(checkpoint) != row['checkpoint_sha256']:
        raise ValueError('来源checkpoint指纹不匹配')
    return run, row['checkpoint_sha256']


def validate_execution_receipt(path,device,checkpoint_sha):
    if path is None:raise ValueError('正式实验需独立执行检查和GPU预算回执，不能仅靠旧Wolff速度估计')
    path=Path(path).resolve();d=read(path)
    budget_fields=('projected_total_gpu_seconds_with_margin','actual_preflight_gpu_seconds',
        'projected_collection_gpu_seconds_with_margin','projected_training_gpu_seconds_with_margin')
    if any(k not in d or not isinstance(d[k],(int,float)) or not math.isfinite(d[k]) or d[k]<0
           for k in budget_fields):raise ValueError('GPU预算回执必须为完整有限非负值')
    if (d.get('status')!='complete' or d.get('workflow_complete') is not True
            or d.get('source_checkpoint_sha256')!=checkpoint_sha or d.get('device')!=device
            or d.get('projected_total_gpu_seconds_with_margin',float('inf'))>3600
            or d.get('smoke_complete') is not True):
        raise ValueError('来源/设备/烟测/GPU1卡时预算执行回执未通过')
    expected=d['actual_preflight_gpu_seconds']+d['projected_collection_gpu_seconds_with_margin']+d['projected_training_gpu_seconds_with_margin']
    if abs(expected-d['projected_total_gpu_seconds_with_margin'])>1e-6 or len(d.get('evidence_sha256',{}))<6:
        raise ValueError('GPU预算总量或执行证据清单不完整')
    for p,sha in d.get('evidence_sha256',{}).items():
        if fingerprint(p)!=sha:raise ValueError(f'执行预检回执来源变化：{p}')
    return d


def execution_budget_totals(collection, training, smoke):
    """Include all already-consumed GPU checks, not just future optimizer steps."""
    projections = [float(collection['projected_collection_gpu_seconds_with_margin']),
                   float(training['projected_training_gpu_seconds_with_margin'])]
    benchmark_values = [float(d.get('actual_gpu_seconds', d.get('elapsed_gpu_budget_seconds',
        d.get('elapsed_gpu_card_seconds', d.get('elapsed_seconds', 0.)))))
        for d in (collection, training)]
    smoke_seconds = float(smoke.get('gpu_card_seconds', 0.))
    values = (*projections, *benchmark_values, smoke_seconds)
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('预算回执必须为有限非负卡时')
    return sum(projections), sum(benchmark_values) + smoke_seconds


def consumed_gpu_seconds(receipt):
    for key in ('gpu_card_seconds','actual_gpu_seconds','elapsed_gpu_budget_seconds',
                'elapsed_gpu_card_seconds'):
        if key in receipt:
            value=float(receipt[key])
            if not math.isfinite(value) or value<0:raise ValueError('历史预检GPU耗时须有限非负')
            return value
    raise ValueError('历史预检缺少实际GPU耗时，不能静默按零计算')


def remaining_gpu_projection(plan, kind):
    evidence=plan.get('execution_preflight') or {}
    training=float(evidence.get('projected_training_gpu_seconds_with_margin',0.))
    collection=float(evidence.get('projected_collection_gpu_seconds_with_margin',0.))
    return training+collection if kind=='collection' else training if kind=='training' else 0.


def build_plan(source_root, previous_root, output, *, device='cuda:2', seed=20261017, smoke=False,
               execution_receipt=None):
    from scripts.ising_finite_perturbation_data import condition_specs
    if device not in ('cpu', 'cuda:2', 'cuda:3'):
        raise ValueError('明确使用cpu或GPU2/3，禁止自动迁移')
    if type(seed) is not int or seed != 20261017:
        raise ValueError('物理采样主种子固定20261017')
    raw = [Path(p) for p in (source_root, previous_root, output)]
    if any(p.is_symlink() for p in raw):
        raise ValueError('来源和新输出不允许符号链接')
    source, previous, root = [p.resolve() for p in raw]
    if root.exists():
        raise FileExistsError(f'拒绝覆盖、重复启动或隐式续跑：{root}')
    for protected in (source, previous):
        if root == protected or root in protected.parents or protected in root.parents:
            raise ValueError('新根必须与全部历史来源分离')
    run, checkpoint_sha = validate_source(source, previous)
    execution=None if smoke else validate_execution_receipt(execution_receipt,device,checkpoint_sha)
    project = Path(__file__).resolve().parents[1]
    pins = {str(project/p):fingerprint(project/p) for p in (*NEW_FILES, *DEPENDENCIES)}
    for p in (source/'job.json', source/'plan.json', source/'dim3_control/report/metrics.json',
              source/'dim3_control/report/report.json', source/'dim3_control/report/loss_comparison.png',
              run/'training_step_12000.pt', run/'manifest.json', run/'summary.json', previous/'job.json',
              previous/'report/comparison.json'):
        pins[str(p)] = fingerprint(p)
    cache_sidecar=Path(read(run/'summary.json')['config']['data']['cache']).with_suffix('.json')
    pins[str(cache_sidecar)]=fingerprint(cache_sidecar)
    # Only the small public third-stage statistics, never its physical data.pt.
    for logical in ('L54', 'L108'):
        for filename in ('operator_statistics.npz', 'diagnostic.json', 'status.json'):
            p = previous/f'arms/dim3_control/{logical}/collection'/filename
            pins[str(p)] = fingerprint(p)
    if execution_receipt is not None:
        pins[str(Path(execution_receipt).resolve())]=fingerprint(execution_receipt)
        if execution:
            pins.update(execution['evidence_sha256'])
    conditions = condition_specs(smoke)
    if len(conditions) != 68 or [r['id'] for r in conditions] != [f'c{i:03d}' for i in range(68)]:
        raise ValueError('须为冻结的68条件，不允许补点/删点/选seed')
    command0 = [sys.executable, '-u', '-m']
    smoke_args = ['--smoke'] if smoke else []
    stages = [dict(name='calibration_and_budget', kind='calibration', device='cpu',
        command=command0+['scripts.ising_finite_perturbation_data', 'calibrate',
            '--output', str(root/'data/calibration'), '--seed', str(seed)]+smoke_args,
        marker=str(root/'data/calibration/status.json'))]
    for row in conditions:
        stages.append(dict(name='sample_'+row['id'], kind='sampling', device='cpu',
            command=command0+['scripts.ising_finite_perturbation_data','sample',
                '--calibration', str(root/'data/calibration'), '--condition-index', str(row['index']),
                '--output', str(root/'data/conditions'/row['id']), '--seed', str(seed)]+smoke_args,
            marker=str(root/'data/conditions'/row['id']/'status.json')))
    stages += [dict(name='frozen_collection', kind='collection', device=device,
        command=command0+['experiments.prepare_ising_finite_perturbation', '--source-run', str(run),
            '--data-root', str(root/'data'), '--output', str(root/'cache'),
            '--expected-checkpoint-sha256', checkpoint_sha,'--device',device,'--seed',str(seed)]+smoke_args,
        marker=str(root/'cache/status.json')),
        dict(name='identifiability_and_five_arms', kind='training', device=device,
            command=command0+['experiments.ising_finite_perturbation_ablation', '--source-run',str(run),
                '--cache-root',str(root/'cache'),'--output',str(root/'training'),
                '--device',device,'--updates','6000','--seed','91017']+smoke_args,
            marker=str(root/'training/status.json')),
        dict(name='report',kind='report',device='cpu', command=command0+[
            'scripts.report_ising_finite_perturbation','--root',str(root),
            '--previous-root',str(previous),'--source-root',str(source),
            '--output',str(root/'report')]+smoke_args, marker=str(root/'report/status.json'))]
    return dict(schema_version=1, experiment='finite_physical_perturbation_R_five_arm_v1',
        authorization='用户2026-10-17明确批准固定实验设计；新独立根，不恢复前三阶段',
        created_at=now(), output=str(root), source_root=str(source), source_run=str(run),
        previous_root=str(previous), source_checkpoint_sha256=checkpoint_sha, code_and_source_sha256=pins,
        execution_receipt=str(Path(execution_receipt).resolve()) if execution_receipt is not None else None,
        execution_preflight=execution,
        smoke_only=smoke, device=device, physical_seed=seed, optimizer_seed=91017,
        source_QCR_updates=12000, extra_R_updates_per_arm=2 if smoke else 6000,
        arms=list(ARMS), new_QC_updates=0, MI_estimates=0,
        conditions=conditions, stages=stages,
        budget=dict(conditions=68, production_chains=136 if smoke else 816,
            production_configurations=1088 if smoke else 104448,
            pilot_chains=136, formal_cluster_proposals_including_pilots=26181632,
            formal_local_full_sweeps_including_pilots=818176,
            formal_local_site_proposals=6701294592,
            cpu_core_hours=24., gpu_card_hours=1., output_directory_gib=10.,
            max_gpu_memory_gib=8., benchmark_required=True,
            cap_policy='启动生产前按保守实测核对预算；超出则不启动，不缩短固定采样/训练来凑完成。'),
        resources=dict(nice=10, compute_threads_per_child=1, sequential_children=True,
            min_available_memory_gib=36, max_load_per_core=1.25, max_io_full_avg10=20.,
            gpu_idle_checks=2, gpu_idle_separation_seconds=10, gpu_min_free_gib=12,
            never_touch_other_processes=True),
        recipe=dict(historical_block_factor=2, model_schema_version=5, Q_motif_width=3,
            Q_hidden=96, dtype='float64', frozen=['Q','C'], optimized=['R'],
            gradient_routing='R参数独立；缓存detach；不加载Q/C进入优化器。'),
        supervision=dict(K_labels=False, perturbation_labels=False, Kc_labels=False,
            exponent_labels=False, theoretical_multiplier_labels=False,
            generation_metadata_only=True, physical_even_operator_prior=True),
        scientific_gate='增强域可靠三方向支撑不足即科学停止，不以伪逆补齐、不加样/改阈值/换seed；热模未识别仍null。',
        failure_policy='失败保留，不重跑、不覆盖、不停止其他进程；仅更新本新根状态。')


def verify_pins(plan):
    for path, sha in plan['code_and_source_sha256'].items():
        if fingerprint(path) != sha:
            raise ValueError(f'冻结来源或代码指纹变化：{path}')


def cpu_seconds():
    a, b = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return a.ru_utime+a.ru_stime+b.ru_utime+b.ru_stime


def public_manifest(root):
    def public_hash(path):
        if path.is_symlink() or path.stat().st_size>512*2**20:
            raise ValueError('公共结果须为不超过512MiB的普通JSON/PNG/NPZ；禁止物理PT')
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(2**20),b''):h.update(chunk)
        return h.hexdigest()
    return {str(p.relative_to(root)):public_hash(p) for p in root.rglob('*')
        if p.is_file() and p.suffix in ('.json','.png','.npz')
        and p.name != 'job.json' and 'logs' not in p.parts}


def execute(args):
    plan = build_plan(args.source_root,args.previous_root,args.output,
        device=args.device,seed=args.seed,smoke=args.smoke,execution_receipt=args.execution_receipt)
    root = Path(plan['output']); root.mkdir(parents=True,exist_ok=False); (root/'logs').mkdir()
    save(root/'plan.json',plan)
    if os.getpriority(os.PRIO_PROCESS,0) < 10:
        os.setpriority(os.PRIO_PROCESS,0,10)
    job = dict(status='running',workflow_complete=False,parent_pid=os.getpid(),started_at=now(),
        current=None,stages=[],requires_user_action=False,smoke_only=args.smoke,
        historical_outputs_modified=False,new_QC_updates=0,MI_estimates=0)
    save(root/'job.json',job)
    start_cpu = cpu_seconds()
    gpu_seconds = float((plan.get('execution_preflight') or {}).get('actual_preflight_gpu_seconds',0.))
    scientific_stop = False
    try:
        for stage in plan['stages']:
            remaining_gpu=remaining_gpu_projection(plan,stage['kind'])
            if not args.smoke and (cpu_seconds()-start_cpu >= 24*3600 or gpu_seconds >= 3600
                    or gpu_seconds+remaining_gpu>3600):
                job.update(status='budget_hold',requires_user_action=True,
                    reason='累计资源或剩余阶段保守投影触及批准预算；不缩短固定终点、不继续下一阶段。',
                    gpu_card_seconds=gpu_seconds,projected_remaining_gpu_seconds_with_margin=remaining_gpu,
                    ended_at=now())
                save(root/'job.json',job); return 0
            checks=[]
            while True:
                snap=resources(stage['device']); checks.append(snap)
                job['current']=dict(name='wait_resources',next_stage=stage['name'],resources=snap)
                save(root/'job.json',job)
                if snap['ready'] and (not stage['device'].startswith('cuda:')
                        or len(checks)>1 and checks[-2]['ready']):
                    break
                time.sleep(10)
            verify_pins(plan)
            entry={**stage,'status':'running','started_at':now(),
                'log':str(root/'logs'/(stage['name']+'.log')),'resource_checks':checks[-2:]}
            wall=time.monotonic(); cpu=cpu_seconds()
            with Path(entry['log']).open('x') as stream:
                child=subprocess.Popen(entry['command'],stdin=subprocess.DEVNULL,stdout=stream,
                    stderr=subprocess.STDOUT,env=child_environment())
                entry['pid']=child.pid; job['current']=entry; save(root/'job.json',job)
                code=child.wait()
            elapsed=time.monotonic()-wall
            if stage['device'].startswith('cuda:'): gpu_seconds+=elapsed
            entry.update(returncode=code,ended_at=now(),elapsed_seconds=elapsed,
                cpu_seconds=cpu_seconds()-cpu,status='complete' if code==0 else 'failed')
            job['stages'].append(entry); job['current']=None
            job.update(cpu_seconds=cpu_seconds()-start_cpu,gpu_card_seconds=gpu_seconds)
            save(root/'job.json',job)
            if code: raise RuntimeError(f"{stage['name']}退出码{code}；禁止自动修复或重跑")
            marker=read(stage['marker'])
            if stage['kind']=='sampling':
                sidecar=read(Path(stage['marker']).parent/'sidecar.json')
                quality=sidecar.get('quality',{})
                if quality.get('status')=='warning':
                    warnings=job.setdefault('sampling_quality_warnings',[])
                    warnings.append(dict(condition_id=sidecar['condition_id'],
                        count=len(quality.get('warnings',[])),sidecar=str(Path(stage['marker']).parent/'sidecar.json')))
                    save(root/'job.json',job)
            if stage['kind']=='calibration':
                cal=read(root/'data/calibration/calibration.json')
                gate=cal.get('budget_gate',marker.get('budget_gate',{}))
                if gate.get('passed') is not True:
                    job.update(status='budget_hold',workflow_complete=False,requires_user_action=True,
                        ended_at=now(),reason='正式前吞吐预检未通过批准预算，不生成后续物理数据。',
                        budget_gate=gate,public_files=public_manifest(root))
                    save(root/'job.json',job); return 0
            if marker.get('status')=='scientific_stop' and stage['kind']=='training':
                scientific_stop=True
            elif marker.get('status')!='complete' or not marker.get('workflow_complete'):
                raise ValueError(f"{stage['name']}缺少完成标志：{marker.get('status')}")
        verify_pins(plan)
        job.update(status='scientific_stop' if scientific_stop else 'complete',workflow_complete=True,
            experimental_workflow_complete=True,scientific_gate_passed=not scientific_stop,
            actual_extra_R_updates_per_arm=0 if scientific_stop else (2 if args.smoke else 6000),
            current=None,ended_at=now(),public_files=public_manifest(root),
            cpu_seconds=cpu_seconds()-start_cpu,gpu_card_seconds=gpu_seconds,thermal_nu=None)
        save(root/'job.json',job)
        print(json.dumps(dict(status=job['status'],output=str(root),smoke=args.smoke)),flush=True)
        return 0
    except BaseException as error:
        job.update(status='failed',workflow_complete=False,requires_user_action=True,
            error=f'{type(error).__name__}: {error}',ended_at=now(),
            active_children_preserved=job.get('current'))
        save(root/'job.json',job); traceback.print_exc(); return 1


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-root','previous-root','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',default='cuda:2',choices=['cpu','cuda:2','cuda:3'])
    parser.add_argument('--seed',type=int,default=20261017)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--execution-receipt',type=Path)
    parser.add_argument('--certify-execution',action='store_true')
    parser.add_argument('--collection-benchmark',type=Path)
    parser.add_argument('--training-benchmark',type=Path)
    parser.add_argument('--smoke-root',type=Path)
    parser.add_argument('--prior-preflight-receipt',type=Path,action='append',default=[])
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--detach',action='store_true')
    args=parser.parse_args(argv)
    if args.certify_execution:
        if any(p is None for p in (args.execution_receipt,args.collection_benchmark,args.training_benchmark,args.smoke_root)):
            raise ValueError('执行回执须提供两个独立benchmark和已完成smoke根')
        if args.execution_receipt.exists():raise FileExistsError('执行回执拒绝覆盖')
        run,sha=validate_source(args.source_root,args.previous_root)
        evidence={}
        benchmarks=[]
        modules=('experiments/prepare_ising_finite_perturbation.py','experiments/ising_finite_perturbation_ablation.py')
        for folder,module in zip((args.collection_benchmark,args.training_benchmark),modules):
            for filename in ('benchmark.json','status.json'):
                p=folder/filename;evidence[str(p.resolve())]=fingerprint(p)
            d=read(folder/'benchmark.json');s=read(folder/'status.json')
            if s.get('status')!='complete' or s.get('workflow_complete') is not True or d.get('device')!=args.device:
                raise ValueError('benchmark未完成或实际设备不符')
            if d.get('source_checkpoint_sha256')!=sha:raise ValueError('benchmark源权重不匹配')
            module_path=Path(__file__).resolve().parents[1]/module
            if d.get('benchmark_module_sha256')!=fingerprint(module_path):
                raise ValueError('benchmark不是当前冻结实现的执行证据')
            benchmarks.append(d)
        p=args.smoke_root/'job.json';smoke=read(p);evidence[str(p.resolve())]=fingerprint(p)
        if smoke.get('status')!='complete' or smoke.get('workflow_complete') is not True or smoke.get('smoke_only') is not True:
            raise ValueError('端到端smoke未完成，不能启动正式实验')
        p=args.smoke_root/'plan.json';smoke_plan=read(p);evidence[str(p.resolve())]=fingerprint(p)
        if smoke_plan.get('source_checkpoint_sha256')!=sha:raise ValueError('smoke源模型不符')
        verify_pins(smoke_plan)
        identities=[d['device_identity'] for d in benchmarks]
        uuids=[str(x.get('physical_gpu_uuid',x.get('physical_uuid',''))).lower().removeprefix('gpu-') for x in identities]
        if args.device.startswith('cuda:') and (not uuids[0] or uuids[0]!=uuids[1]):
            raise ValueError('两个GPU预检的实际物理UUID不一致')
        collection,training=benchmarks
        projected,actual=execution_budget_totals(collection,training,smoke)
        prior_seconds=0.
        for p in args.prior_preflight_receipt:
            p=p.resolve()
            if str(p) in evidence:raise ValueError('同一预检不能重复计入')
            evidence[str(p)]=fingerprint(p)
            prior_seconds+=consumed_gpu_seconds(read(p))
        actual+=prior_seconds
        if projected+actual>3600:raise ValueError('保守GPU吞吐估计超过1卡时；正式生产不启动')
        receipt=dict(schema_version=1,status='complete',workflow_complete=True,created_at=now(),
            device=args.device,source_checkpoint_sha256=sha,source_run=str(run),
            projected_total_gpu_seconds_with_margin=projected+actual,actual_preflight_gpu_seconds=actual,
            prior_preflight_gpu_seconds=prior_seconds,
            projected_collection_gpu_seconds_with_margin=collection['projected_collection_gpu_seconds_with_margin'],
            projected_training_gpu_seconds_with_margin=training['projected_training_gpu_seconds_with_margin'],
            smoke_complete=True,smoke_root=str(args.smoke_root.resolve()),evidence_sha256=evidence,
            non_scientific_execution_checks=True)
        with args.execution_receipt.open('x') as stream:json.dump(receipt,stream,ensure_ascii=False,indent=2)
        print(json.dumps(receipt,ensure_ascii=False,indent=2));return 0
    plan=build_plan(args.source_root,args.previous_root,args.output,
        device=args.device,seed=args.seed,smoke=args.smoke,execution_receipt=args.execution_receipt)
    if args.dry_run:
        print(json.dumps(plan,ensure_ascii=False,indent=2)); return 0
    if args.detach:
        logs=Path.cwd()/'logs'; logs.mkdir(exist_ok=True)
        receipt=logs/(args.output.name+'.launch.json'); logfile=logs/(args.output.name+'.log')
        command=['nice','-n','10',sys.executable,'-u','-m','scripts.run_ising_finite_perturbation',
            '--source-root',str(args.source_root.resolve()),'--previous-root',str(args.previous_root.resolve()),
            '--output',str(args.output.resolve()),'--device',args.device,'--seed',str(args.seed)]
        if args.smoke: command.append('--smoke')
        if args.execution_receipt:command+=['--execution-receipt',str(args.execution_receipt.resolve())]
        with receipt.open('x') as stream:
            json.dump(dict(status='launching',created_at=now(),command=command),stream)
        with logfile.open('x') as stream:
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                start_new_session=True,env=child_environment())
        result=dict(status='launched',parent_pid=child.pid,output=str(args.output.resolve()),
            command=command,log=str(logfile),launched_at=now())
        save(receipt,result); print(json.dumps(result,ensure_ascii=False,indent=2)); return 0
    return execute(args)


if __name__=='__main__':
    raise SystemExit(main())
