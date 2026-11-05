"""独立冻结诊断；不修改正式实验，不训练Q/C/R，不访问物理标签。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np
import torch

sys.path.insert(0, str(Path.cwd()))
from recurrent_rg.checkpoints import load_run, isolated_rng
from recurrent_rg.data.statistics import plaquette_distribution
from experiments.prepare_ising_physical_response import (
    q_feature_table, head_value_jacobian, validate_derivatives, mmap_configurations,
)
from scripts.diagnose_ising_mcrg import c_probability_table, c_draw, covariance, operator_values


BASE = Path('/mnt/nas-new/home/yange/ham')
SOURCES = (
    BASE/'results/ising2d_positiveK_QCR_block2_latest12000_seed91_vm225_gpu0/run/training_step_9000.pt',
    BASE/'results/ising2d_positiveK_QCR_block2_Ksupervised_continue12000_seed91_vm225_gpu0/segments/epoch_9000/training_step_9000.pt',
)
DATA = BASE/'data/ising2d_positiveK_L108_b3_replicas12_seed91.pt'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_sha(model):
    h = hashlib.sha256()
    for name, v in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(str(v.dtype).encode())
        h.update(str(tuple(v.shape)).encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def resource_snapshot():
    mem = dict(x.split(':', 1) for x in Path('/proc/meminfo').read_text().splitlines())
    available = int(mem['MemAvailable'].split()[0])*1024
    full = next(x for x in Path('/proc/pressure/io').read_text().splitlines() if x.startswith('full '))
    io = float(next(x.split('=')[1] for x in full.split() if x.startswith('avg10=')))
    return dict(mem_available_bytes=available, load1=os.getloadavg()[0], io_full_avg10=io,
                ready=available >= 36*1024**3 and os.getloadavg()[0] <= 1.25*os.cpu_count() and io < 20)


def statistics_from_motifs(spins, table):
    """由同一3x3整数直方图精确边缘化2x2及前五个偶算符，避免重复扫描。"""
    index = torch.zeros_like(spins, dtype=torch.int64)
    for power, (a, b) in zip(table['powers'], table['offsets']):
        index += (torch.roll(spins, (-a, -b), (-2, -1)) > 0).long()*power
    flat = index.flatten(1)
    offset = torch.arange(len(spins))[:, None]*512
    counts = torch.bincount((flat+offset).flatten(), minlength=len(spins)*512).reshape(len(spins), 512)
    p3 = counts.double()/flat.shape[1]
    bins = torch.arange(512)
    bits = ((bins[:,None]//(2**torch.arange(9)))%2)
    marginal_index=8*bits[:,4]+4*bits[:,7]+2*bits[:,5]+bits[:,8]
    count2=torch.zeros((len(spins),16),dtype=torch.int64)
    count2.scatter_add_(1,marginal_index[None,:].expand(len(spins),-1),counts)
    p2=count2.double()/flat.shape[1]
    pattern=(2*bits-1).double()
    p=pattern
    operator_table=torch.stack((p[:,4]*(p[:,7]+p[:,5]),
        p[:,4]*(p[:,8]+p[:,6]),p[:,4]*p[:,7]*p[:,5]*p[:,8],
        p[:,0]*(p[:,6]+p[:,2]),p[:,0]*p[:,6]*p[:,1]*p[:,7]+p[:,0]*p[:,3]*p[:,2]*p[:,5]),-1)
    first_five=counts.double()@operator_table
    distance3=(spins*(torch.roll(spins,-3,-2)+torch.roll(spins,-3,-1))).sum((-2,-1))
    score=torch.cat((first_five,distance3[:,None]),-1)
    return p3,p2,score


def frequencies3(spins, table):
    return statistics_from_motifs(spins,table)[0]


def frequencies2(spins):
    b = (spins > 0).long()
    index = 8*b+4*torch.roll(b, -1, -2)+2*torch.roll(b, -1, -1)+torch.roll(b, (-1, -1), (-2, -1))
    flat = index.flatten(1)
    offset = torch.arange(len(spins))[:, None]*16
    return torch.bincount((flat+offset).flatten(), minlength=len(spins)*16).reshape(len(spins), 16).double()/flat.shape[1]


def longrange(spins):
    side = spins.shape[-1]
    result = []
    for distance in (4, 8):
        result.append((spins*(torch.roll(spins, -distance, -2)+torch.roll(spins, -distance, -1))).mean((-2, -1))/2)
    q1 = torch.exp(2j*torch.pi*torch.arange(side, dtype=torch.float64)/side)
    row, col = spins.sum(-1).to(torch.complex128), spins.sum(-2).to(torch.complex128)
    for n in (1, 2):
        q = q1**n
        result.append(((row@q).abs().square()+(col@q).abs().square())/(2*side**4))
    phase_plus = q1[:, None]*q1[None, :]
    phase_minus = q1[:, None]*q1.conj()[None, :]
    s = spins.to(torch.complex128)
    result.append(((s*phase_plus).sum((-2, -1)).abs().square()+(s*phase_minus).sum((-2, -1)).abs().square())/(2*side**4))
    return torch.stack(result, -1)


def collect_one(spins, split, index, models, tables, probabilities, draws, spotcheck=False):
    g = np.empty((2, draws, 3, 2)); gfull = np.empty((2, draws, 3)); rfull = np.empty_like(gfull)
    obs = np.empty((draws, 3, 2, 5)); motif = np.empty((draws, 3, 2, 16))
    target = np.empty((draws, 3, 16, 6)); target_half = np.empty((draws, 3, 2, 16, 6))
    prediction = np.empty((2, draws, 3, 16, 6)); qresp = np.empty((2, draws, 3, 1, 6))
    score_cov = np.empty((draws, 3, 6, 6)); score_std = np.empty((draws, 3, 6))
    checks = []
    micro = None
    for draw in range(draws):
        gen = torch.Generator().manual_seed(2026101701+split*1_000_000_000+index*10_000+draw)
        levels = [spins]
        for _ in range(2):
            previous = levels[-1]
            uniform = torch.rand((128, previous.shape[-1]//2, previous.shape[-1]//2), dtype=torch.float64, generator=gen)
            a, b = (c_draw(previous, probability, uniform) for probability in probabilities)
            if not torch.equal(a, b):
                raise ValueError('共同uniform下两C产生不同构型，不能以共用target隔离Q；停止而不改协议')
            levels.append(a)
        for scale, level in enumerate(levels):
            if scale == 0 and micro is not None:
                p3, p2, score, measured = micro
            else:
                with torch.no_grad():
                    p3,p2,score = statistics_from_motifs(level,tables[0])
                    measured = longrange(level)
                if scale == 0:
                    micro = p3, p2, score, measured
            centered = score-score.mean(0)
            sd = centered.square().mean(0).sqrt().clamp_min(1e-12)
            normalized = centered/sd
            score_std[draw, scale] = sd.numpy()
            score_cov[draw, scale] = covariance(normalized, ddof=0).numpy()
            target[draw, scale] = covariance(p2, normalized, ddof=0).numpy()
            for half, sl in enumerate((slice(0,64), slice(64,128))):
                obs[draw, scale, half] = measured[sl].mean(0).numpy()
                motif[draw, scale, half] = p2[sl].mean(0).numpy()
                target_half[draw, scale, half] = covariance(p2[sl], normalized[sl], ddof=0).numpy()
            if spotcheck and draw == 0:
                with torch.no_grad():
                    reference = plaquette_distribution(level[:, None, :, :, None])
                if not torch.equal(reference, p2):
                    raise ValueError('2x2 pattern顺序与原decoder训练目标不一致')
                if not torch.equal(operator_values(level)[:,:6],score):
                    raise ValueError('优化的motif算符与原六偶算符定义不一致')
            for m, (model, table) in enumerate(zip(models, tables)):
                features = p3@table['values']
                value, jac = head_value_jacobian(model.theory_encoder.head, features.mean(0))
                response = jac@covariance(features, normalized, ddof=0)
                with torch.no_grad():
                    W = model.theory_encoder.distribution_decoder.weight
                    prediction[m,draw,scale] = (W@response).numpy()
                    gfull[m,draw,scale] = value.item()
                    rfull[m,draw,scale] = model.advance_theory(value).item()
                    qresp[m,draw,scale] = response.numpy()
                    for h, sl in enumerate((slice(0,64),slice(64,128))):
                        g[m,draw,scale,h] = model.theory_encoder.head(features[sl].mean(0)).item()
                if spotcheck and draw == 0:
                    checks.append(dict(model=m,scale=scale,**validate_derivatives(model.theory_encoder.head,features,score,jac,jac@covariance(features,score,ddof=0))))
    return dict(g=g,g_full=gfull,r_full=rfull,observables=obs,motif_means=motif,
                response_target=target,response_target_halves=target_half,response_prediction=prediction,
                q_response=qresp,score_cov=score_cov,score_std=score_std), checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',required=True); parser.add_argument('--protocol',required=True)
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True,exist_ok=False)
    protocol = json.loads(Path(args.protocol).read_text()); save_json(out/'protocol.json', protocol)
    os.setpriority(os.PRIO_PROCESS,0,19)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    start_wall=time.monotonic(); start_cpu=time.process_time()
    status=dict(status='running',workflow_complete=False,stage='validation',completed_ensembles=0,total_ensembles=768,QCR_updates=0,MI_estimates=0,physical_sampling=0)
    save_json(out/'status.json',status)
    try:
        snap=resource_snapshot()
        if not snap['ready']:
            status.update(status='resource_hold',resource=snap);save_json(out/'status.json',status);return
        actual=[sha(p) for p in SOURCES]
        if actual!=protocol['source_checkpoint_sha256']:
            raise ValueError('源checkpoint SHA不符')
        sidecar=DATA.with_suffix('.json')
        if sha(sidecar)!=protocol['data_sidecar_sha256']:
            raise ValueError('原物理数据既有侧车SHA不符；不重hash原数据')
        models=[];configs=[];source_meta=[]
        for m,p in enumerate(SOURCES):
            with isolated_rng('cpu',20261017):loaded=load_run(p)
            if loaded.payload.get('epoch')!=9000:raise ValueError('不是实际9000步checkpoint')
            cfg=loaded.config
            if cfg['model']['theory_dim']!=1 or cfg['model']['block_factor']!=2 or not cfg['model']['affine_reconstruction']:
                raise ValueError('历史模型结构不符')
            if loaded.payload.get('metadata',{}).get('data_sha256')!=protocol['data_sha256_from_existing_receipt']:
                raise ValueError('历史物理来源指纹不符')
            model=loaded.model.cpu().double().eval()
            for param in model.parameters():param.requires_grad_(False);param.grad=None
            models.append(model);configs.append(cfg)
            source_meta.append(dict(name=protocol['models'][m],checkpoint=str(p),checkpoint_sha256=actual[m],actual_epoch=9000,model_sha256_before=state_sha(model)))
        for cfg in configs:cfg['training'].pop('epochs',None)
        if configs[0]!=configs[1]:raise ValueError('两个历史配置除计划训练步数外不同')
        tables=[q_feature_table(m) for m in models]
        probabilities=[c_probability_table(m)[1] for m in models]
        c_difference=float((probabilities[0]-probabilities[1]).abs().max())
        if c_difference>1e-12:raise ValueError('两个C概率表不等价，停止Q隔离比较')
        data=mmap_configurations(DATA)
        receipt=dict(models=source_meta,data_path=str(DATA),data_stat_bytes=DATA.stat().st_size,
                     data_storage_sha256_from_receipt=protocol['data_sha256_from_existing_receipt'],data_rehashed=False,
                     labels_accessed=False,MI_opened=False,old_reports_modified=False,ongoing_experiment_modified=False,
                     C_probability_max_abs_difference=c_difference,resource_before=snap,device='cpu',threads=1,nice=os.getpriority(os.PRIO_PROCESS,0),axes=dict(g='model,physical_ensemble,C_draw,scale,half',response_prediction='model,physical_ensemble,C_draw,scale,16bins,6scores'))
        save_json(out/'receipt.json',receipt)
        records=[];identities=[];checks=[];benchmark_cpu=None
        collection_start_cpu=time.process_time()
        for split_name, split_id in (('train',0),('test',1)):
            for index in range(len(data[split_name])):
                elapsed_cpu=time.process_time()-start_cpu
                if elapsed_cpu>1200 or time.monotonic()-start_wall>1800:
                    status.update(status='budget_hold',reason='独立诊断预算触顶；不减少预先固定样本或更改模型');save_json(out/'status.json',status);return
                if len(records)%32==0 and not resource_snapshot()['ready']:
                    status.update(status='resource_hold',resource=resource_snapshot());save_json(out/'status.json',status);return
                before=time.process_time()
                spins=data[split_name][index,...,0].double()
                record,check=collect_one(spins,split_id,index,models,tables,probabilities,protocol['draws'],spotcheck=len(records)==0)
                records.append(record);checks+=check
                identities.append((split_id,index,index//12,index%12))
                if len(records)==3:
                    benchmark_cpu=(time.process_time()-collection_start_cpu)/3
                    projection=(time.process_time()-start_cpu)+benchmark_cpu*(768-3)*2+60
                    receipt['preflight']=dict(ensembles=3,seconds_per_ensemble=benchmark_cpu,setup_cpu_accounted_once=True,projected_cpu_seconds_with_2x_margin=projection,passed=projection<=1200)
                    save_json(out/'receipt.json',receipt)
                    if projection>1200:
                        status.update(status='budget_hold',reason='固定诊断CPU预算预检未通过；不缩减样本凑结果',preflight=receipt['preflight']);save_json(out/'status.json',status);return
                status.update(stage='collection',completed_ensembles=len(records),current_split=split_name,current_index=index,cpu_seconds=time.process_time()-start_cpu,last_ensemble_cpu_seconds=time.process_time()-before)
                if len(records)%12==0 or len(records)<=3:
                    save_json(out/'status.json',status);print(json.dumps(status,ensure_ascii=False),flush=True)
        arrays={}
        model_axes={'g','g_full','r_full','response_prediction','q_response'}
        for key in records[0]:arrays[key]=np.stack([rec[key] for rec in records],axis=1 if key in model_axes else 0)
        ids=np.array(identities,dtype=np.int64)
        arrays.update(split=ids[:,0],ensemble_index=ids[:,1],group_index=ids[:,2],replica_index=ids[:,3],model_names=np.array(protocol['models']))
        if any(not np.isfinite(v).all() for v in arrays.values() if np.issubdtype(v.dtype,np.number)):
            raise FloatingPointError('采集统计存在NaN/Inf')
        for m,model in enumerate(models):
            source_meta[m]['model_sha256_after']=state_sha(model)
            source_meta[m]['all_parameter_grad_none']=all(p.grad is None for p in model.parameters())
            if source_meta[m]['model_sha256_after']!=source_meta[m]['model_sha256_before'] or not source_meta[m]['all_parameter_grad_none']:
                raise ValueError('冻结模型状态或梯度不符')
        if [sha(p) for p in SOURCES]!=actual:raise ValueError('源checkpoint在诊断中变化')
        np.savez_compressed(out/'collected.npz',**arrays)
        receipt.update(models=source_meta,derivative_checks=checks,collected_sha256=sha(out/'collected.npz'),cpu_seconds=time.process_time()-start_cpu,wall_seconds=time.monotonic()-start_wall,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        save_json(out/'receipt.json',receipt)
        status.update(status='complete',workflow_complete=True,stage='collected',cpu_seconds=receipt['cpu_seconds'],wall_seconds=receipt['wall_seconds'])
        save_json(out/'status.json',status);print(json.dumps(status,ensure_ascii=False),flush=True)
    except BaseException as error:
        status.update(status='failed',workflow_complete=False,error=f'{type(error).__name__}: {error}')
        save_json(out/'status.json',status);traceback.print_exc();raise


if __name__=='__main__':main()
