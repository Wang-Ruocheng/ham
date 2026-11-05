"""Fresh kappa-h physical data and a strict configuration-only loader.

No historical dataset/schema is reinterpreted. Labels are attached to evaluation
metadata only; training batch selection uses configurations and split indices.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
from . import phi4 as base
from . import phi4_kappa_h as kernel
from . import phi4_hybrid_pipeline as historical

ARRAY_MEMBERS = historical.ARRAY_MEMBERS + ('field_accepted','field_log_acceptance')
WORK_FIELDS = historical.WORK_FIELDS

def source_pins():
    from . import phi4_hopping, phi4_hybrid
    modules=(base,kernel,historical,phi4_hopping,phi4_hybrid)
    return {str(Path(m.__file__).resolve()):base.sha256_file(m.__file__) for m in modules} | {
        str(Path(__file__).resolve()):base.sha256_file(__file__)}

def generation_plan(options):
    o=json.loads(json.dumps(options))
    if o['dimension']!=2 or len(o['sizes'])!=1 or o['sizes'][0]<9 or o['sizes'][0]%9:
        raise ValueError('Require two b3 scales in 2D')
    for name in ('chains','samples','burn_in','spacing','leapfrog_steps','seed','test_every','test_offset'):
        base._integer(o[name],name,0 if name in ('burn_in','seed','test_offset') else 1)
    if o['chains']!=4 or o['samples']<4 or o['samples']%2 or o['test_offset']>=o['test_every']:
        raise ValueError('Require four chains, even samples, valid split')
    if base._finite(o['step_size'],'step_size')<=0 or o['cluster_probability']!=.5:
        raise ValueError('Require positive HMC step and fixed lazy probability')
    for name in ('kappa_values','h_values'):
        if len(set(o[name]))!=len(o[name]) or not o[name]:raise ValueError('Invalid grid')
    for k in o['kappa_values']:
        for h in o['h_values']:kernel.gradient_parameters(k,o['quartic_lambda'],2,h)
    n=len(o['kappa_values'])*len(o['h_values']);v=o['sizes'][0]**2
    cycles=o['burn_in']+o['samples']*o['spacing'];total=n*o['chains']*cycles
    projected=n*o['chains']*(o['samples']*(v+8)*8+cycles*67+65536)+n*262144
    if total>400000 or total*(o['leapfrog_steps']+1)*v>13_000_000_000 or projected>1024**3:
        raise ValueError('Independent physical work envelope exceeded')
    return dict(schema=kernel.SCHEMA,options=o,conditions=n,configurations=n*o['chains']*o['samples'],
        total_hmc_trajectories=total,total_force_site_evaluations=total*(o['leapfrog_steps']+1)*v,
        maximum_cluster_neighbor_visits=total*4*v,projected_output_bytes=projected,
        train_cell_indices=[i for i in range(n) if i%o['test_every']!=o['test_offset']],
        test_cell_indices=[i for i in range(n) if i%o['test_every']==o['test_offset']],
        action=kernel.ACTION,source_pins=source_pins(),training_member='configurations',
        evaluation_label_columns=['kappa','h','quartic_lambda'],labels_used_in_training=False,
        symmetry_augmentation=False,equilibrium_certified=False,qualification=kernel.QUALIFICATION)

def generate(root,options):
    root=Path(root).resolve();plan=generation_plan(options);o=plan['options']
    if root.exists():raise FileExistsError('No overwrite/resume')
    root.mkdir(parents=True);(root/'conditions').mkdir()
    write=lambda p,x:base._write_json(p,x)
    write(root/'plan.json',plan)
    shards=[];qualities=[];warnings=0;totals={k:0 for k in WORK_FIELDS};start=time.monotonic()
    current=None
    try:
        for cell in range(plan['conditions']):
            kappa=o['kappa_values'][cell//len(o['h_values'])];h=o['h_values'][cell%len(o['h_values'])]
            split='test' if cell in plan['test_cell_indices'] else 'train'
            current=f'c{cell:03d}';directory=root/'conditions'/current;directory.mkdir()
            values=[];rows=[]
            for chain,initial in enumerate(('hot','hot','cold_positive','cold_negative')):
                write(root/'status.json',dict(status='running',workflow_complete=False,current=current,
                    chain_index=chain,completed_conditions=len(shards),returncode=None,error=None,requires_user_action=False))
                seed=int(np.random.SeedSequence([o['seed'],o['sizes'][0],cell,chain,7001]).generate_state(1)[0])
                result=kernel.sample_chain(kappa,o['quartic_lambda'],h=h,size=o['sizes'][0],
                    dimension=2,samples=o['samples'],burn_in=o['burn_in'],spacing=o['spacing'],
                    step_size=o['step_size'],leapfrog_steps=o['leapfrog_steps'],seed=seed,initial=initial)
                if result['production_acceptance']<=0 or any(not np.isfinite(result[k]).all() for k in ARRAY_MEMBERS):
                    raise FloatingPointError('Nonfinite or zero HMC acceptance; no retry')
                values.append({k:result.pop(k) for k in ARRAY_MEMBERS})
                rows.append(dict(chain_index=chain,**result))
                for key in WORK_FIELDS:totals[key]+=int(result[key])
                warnings+=len(result['warnings'])
                print(json.dumps(dict(event='chain_complete',cell=cell,chain=chain,
                    production_acceptance=result['production_acceptance'],
                    cluster_field_acceptance=result['cluster_field_acceptance'],warnings=len(result['warnings']))),flush=True)
            stacked={k:np.stack([v[k] for v in values]) for k in ARRAY_MEMBERS}
            path=directory/'data.npz';np.savez(path,**stacked)
            summary=base._condition_summary(stacked['raw_observables'],rows,o['sizes'][0],2)
            for name in ('magnetization','mean_field_squared'):
                hot=np.array([r['diagnostics'][name]['mean'] for r in rows[:2]])
                cold=np.array([r['diagnostics'][name]['mean'] for r in rows[2:]])
                se=float(np.sqrt(np.var(hot,ddof=1)/2+np.var(cold,ddof=1)/2))
                diff=float(hot.mean()-cold.mean())
                summary['hot_cold_comparison'][name]=dict(hot_chain_means=hot.tolist(),
                    cold_chain_means=cold.tolist(),hot_minus_cold=diff,between_chain_standard_error=se,
                    descriptive_three_se_warning=abs(diff)>max(3*se,1e-14))
            hotcold=[k for k,v in summary['hot_cold_comparison'].items() if v['descriptive_three_se_warning']]
            warnings+=len(hotcold)
            quality=dict(cell_index=cell,kappa=kappa,h=h,split=split,hot_cold_warning_observables=hotcold,
                production_acceptance=[r['production_acceptance'] for r in rows],
                cluster_field_acceptance=[r['cluster_field_acceptance'] for r in rows],
                minimum_chain_ess={k:min(float(r['diagnostics'][k]['ess'] or 0) for r in rows) for k in base.OBSERVABLES},
                chain_warnings=[r['warnings'] for r in rows],equilibrium_certified=False)
            qualities.append(quality)
            sidecar=dict(schema=kernel.SCHEMA,status='complete',cell_index=cell,size=o['sizes'][0],dimension=2,
                kappa=kappa,h=h,quartic_lambda=o['quartic_lambda'],split=split,chains=rows,summary=summary,
                data_sha256=base.sha256_file(path),source_pins=plan['source_pins'],
                training_member='configurations',raw_observable_names=list(base.OBSERVABLES),
                labels_used_in_training=False,symmetry_augmentation=False,equilibrium_certified=False)
            write(directory/'sidecar.json',sidecar)
            write(directory/'status.json',dict(status='complete',workflow_complete=True,returncode=0,error=None))
            shards.append(dict(cell_index=cell,size=o['sizes'][0],split=split,path=str(path.relative_to(root)),
                sha256=sidecar['data_sha256'],sidecar=str((directory/'sidecar.json').relative_to(root)),
                sidecar_sha256=base.sha256_file(directory/'sidecar.json')))
            del values,stacked
        if totals['hmc_trajectories']!=plan['total_hmc_trajectories'] or totals['force_site_evaluations']!=plan['total_force_site_evaluations']:
            raise RuntimeError('Work count mismatch')
        if source_pins()!=plan['source_pins']:raise RuntimeError('Frozen sampler changed')
        write(root/'quality_summary.json',dict(status='complete',warning_count=warnings,conditions=qualities,
            warnings_retained_without_resampling=True,equilibrium_certified=False))
        manifest=dict(schema=kernel.SCHEMA,status='complete',options=o,shards=shards,
            plan_sha256=base.sha256_file(root/'plan.json'),source_pins=plan['source_pins'],
            quality_summary_sha256=base.sha256_file(root/'quality_summary.json'),warning_count=warnings,
            actual_work=totals,labels_used_in_training=False,symmetry_augmentation=False,
            evaluation_label_columns=['kappa','h','quartic_lambda'],physical_field_coordinate='phi',
            sample_semantics='independent_correlated_hybrid_markov_chains',qualification=kernel.QUALIFICATION)
        write(root/'manifest.json',manifest)
        write(root/'status.json',dict(status='complete',workflow_complete=True,returncode=0,error=None,
            requires_user_action=False,current=None,completed_conditions=len(shards),warning_count=warnings,
            wall_seconds=time.monotonic()-start))
        return manifest
    except Exception as exc:
        write(root/'status.json',dict(status='failed',workflow_complete=False,returncode=1,current=current,
            error=f'{type(exc).__name__}: {exc}',requires_user_action=True,completed_conditions=len(shards)))
        raise

def sampling_validity(root,plan=None):
    root=Path(root)/'data'
    status=json.loads((root/'status.json').read_text());m=json.loads((root/'manifest.json').read_text())
    q=json.loads((root/'quality_summary.json').read_text())
    expected=generation_plan(m['options'])
    if (status['status']!='complete' or status['returncode']!=0 or len(m['shards'])!=expected['conditions']
            or m['schema']!=kernel.SCHEMA or base.sha256_file(root/'plan.json')!=m['plan_sha256']
            or base.sha256_file(root/'quality_summary.json')!=m['quality_summary_sha256']
            or any(min(r['production_acceptance'])<=0 for r in q['conditions'])):
        raise RuntimeError('Physical completion/quality receipt mismatch')
    if plan and m['options']!=plan['physical']['options']:raise RuntimeError('Frozen options changed')
    return dict(status='complete',warning_count=m['warning_count'],equilibrium_certified=False,
        retained_sampling_warnings=True)

def load_dataset(root):
    import torch
    from experiments.phi4_qcr import FieldDataset,_inside
    root=Path(root).resolve();m=json.loads((root/'manifest.json').read_text());o=m['options']
    p=generation_plan(o)
    if (m['status']!='complete' or m['schema']!=kernel.SCHEMA or m['physical_field_coordinate']!='phi'
            or m['evaluation_label_columns']!=['kappa','h','quartic_lambda']
            or base.sha256_file(root/'plan.json')!=m['plan_sha256']):
        raise ValueError('Wrong physical semantics or plan')
    values=[];splits=[];cells=[];labels=[];receipts=[];seen=set()
    for shard in m['shards']:
        cell=shard['cell_index']
        if cell in seen or not 0<=cell<p['conditions']:raise ValueError('Duplicate/invalid cell')
        seen.add(cell);path=_inside(root,shard['path']);sidepath=_inside(root,shard['sidecar'])
        if base.sha256_file(path)!=shard['sha256'] or base.sha256_file(sidepath)!=shard['sidecar_sha256']:
            raise ValueError('Physical shard hash mismatch')
        side=json.loads(sidepath.read_text());split='test' if cell in p['test_cell_indices'] else 'train'
        label=[o['kappa_values'][cell//len(o['h_values'])],o['h_values'][cell%len(o['h_values'])],o['quartic_lambda']]
        if (side['schema']!=kernel.SCHEMA or side['status']!='complete' or side['cell_index']!=cell
                or side['size']!=o['sizes'][0] or shard['size']!=o['sizes'][0] or side['dimension']!=2
                or side['split']!=split or shard['split']!=split or side['data_sha256']!=shard['sha256']
                or [side[k] for k in ('kappa','h','quartic_lambda')]!=label):
            raise ValueError('Sidecar identity or split mismatch')
        with np.load(path,allow_pickle=False) as archive:fields=archive['configurations']
        if fields.shape!=(o['chains'],o['samples'],o['sizes'][0],o['sizes'][0],1) or fields.dtype!=np.float64 or not np.isfinite(fields).all():
            raise ValueError('Invalid configuration array')
        values.append(torch.from_numpy(fields));splits.extend([split]*o['chains']);cells.extend([cell]*o['chains'])
        labels.extend([label]*o['chains']);receipts.append(shard)
    if seen!=set(range(p['conditions'])) or set(splits)!={'train','test'}:raise ValueError('Incomplete grid/split')
    dataset=FieldDataset(torch.cat(values),np.asarray(splits),np.asarray(cells),np.asarray(labels),
        base.sha256_file(root/'manifest.json'),receipts,m['warning_count'],kernel.QUALIFICATION,
        kernel.SCHEMA,m['sample_semantics'])
    if not math.isfinite(dataset.scale) or dataset.scale<=0:raise ValueError('Invalid TRAIN RMS')
    return dataset

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    args=parser.parse_args()
    from experiments.phi4_kappa_h_cnn import physical_options
    generate(args.output,physical_options())
