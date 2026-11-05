"""读取有限扰动缓存与已保存 R 评价数组；不加载模型、不生成物理样本。"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.report_ising_zh import chinese_figures, select_chinese_font


LABELS={'initial':'源模型','A':'甲：旧域原约束','B':'乙：旧域方向均衡',
        'C':'丙：新域原约束','D':'丁：新域方向均衡','E':'戊：新域仅闭合'}
CAUTION=('物理链为独立重复，C抽样不是新增物理系综；区间条件于固定Q/C和单一训练种子。'
    '有限耦合扰动含物理先验，K与扰动系数仅用于生成及事后评价，不是训练标签。'
    '实际编码点的完整DR谱不等于不动点热谱；所有热指数保持空值。')


def clean(value):
    if isinstance(value,dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [clean(v) for v in value]
    if isinstance(value,np.ndarray): return clean(value.tolist())
    if isinstance(value,(np.bool_,)):return bool(value)
    if isinstance(value,(np.integer,)):return int(value)
    if isinstance(value,(np.floating,float)):return float(value) if np.isfinite(value) else None
    if isinstance(value,(complex,np.complexfloating)):return dict(real=float(value.real),imag=float(value.imag))
    return value


def read(path):return json.loads(Path(path).read_text())


def save(path,value):
    Path(path).write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()


def interval(values,seed=20261017,repeats=1000):
    values=np.asarray(values,dtype=float)
    if not len(values):return None
    idx=np.random.default_rng(seed).integers(len(values),size=(repeats,len(values)))
    return np.percentile(values[idx].mean(1),[2.5,50,97.5],axis=0)


def standard_response(vin,vout,std):
    inverse=np.divide(1.,std,out=np.zeros_like(std,dtype=float),where=std>1e-12)
    return vin*inverse[...,None,:],vout*inverse[...,None,:]


def relative_response(dr,vin,vout,std):
    x,y=standard_response(vin,vout,std)
    pred=dr[...,:x.shape[-3],:,:]@x
    residual=pred-y
    den=np.linalg.norm(y,axis=(-2,-1))
    ratios=np.divide(np.linalg.norm(residual,axis=(-2,-1)),den,
        out=np.full_like(den,np.nan),where=den>1e-12)
    best=np.empty_like(ratios);ranks=np.empty_like(ratios,dtype=int);conditions=np.empty_like(ratios)
    for ix in np.ndindex(x.shape[:-2]):
        a,b=x[ix],y[ix]
        u,s,vt=np.linalg.svd(a,full_matrices=False)
        keep=s>max(s[0]*1e-8,1e-15)
        ranks[ix]=keep.sum();conditions[ix]=s[0]/s[keep][-1] if keep.any() else np.nan
        fitted=(b@vt[keep].T/s[keep])@u[:,keep].T if keep.any() else np.zeros((a.shape[0],a.shape[0]))
        best[ix]=np.linalg.norm(fitted@a-b)/max(np.linalg.norm(b),1e-12)
    return dict(R_relative_per_replica=ratios.mean(1),best_J_relative_per_replica=best.mean(1),
        rank=ranks,condition=conditions,signed_residual=residual)


def find_array(arrays,arm,logical,key,required=True):
    variants=(f'{arm}_{logical}_{key}',f'{arm}_{logical.lower()}_{key}',
              f'{arm}__{logical}__{key}',f'{arm}__{logical.lower()}__{key}')
    for candidate in variants:
        if candidate in arrays:return arrays[candidate]
    if required:raise ValueError(f'已保存评价数组缺少{arm}/{logical}/{key}；不能临时模型推理补齐')
    return None


def group_rows(direct,rg,dr,vin,vout,std,condition,replica,train_mask,metrics,logical,
               *,smoke=False,repeats=1000,all_predictions=None):
    levels=direct.shape[2]
    lattice=[108,54,27] if logical=='L108' else [54,27]
    if levels!=len(lattice):raise ValueError('保存数组尺度不匹配')
    response=relative_response(dr,vin,vout,std)
    rows=[]
    for cid in np.unique(condition):
        mask=condition==cid
        for split,sub in [('train',mask&train_mask),('test',mask&~train_mask)]:
            if not sub.any():continue
            for transition in range(levels-1):
                diff=rg[sub,:,transition]-direct[sub,:,transition+1]
                per_rep_mse=np.mean(diff**2,axis=(1,2))
                norm=metrics['train_representation'][transition+1 if logical=='L108' else 2]
                scale=float(norm['pooled_coordinate_std'])
                r=response['R_relative_per_replica'][sub,transition]
                b=response['best_J_relative_per_replica'][sub,transition]
                rows.append(dict(condition_index=int(cid),split=split,physical_replicas=int(sub.sum()),
                    physical_replica_indices=replica[sub],path=f'{lattice[transition]}→{lattice[transition+1]}',
                    transition=transition,rmse=float(np.sqrt(per_rep_mse.mean())),
                    nrmse=float(np.sqrt(per_rep_mse.mean())/scale),
                    fixed_original_TRAIN_target_std=scale,
                    original_TRAIN_target_scale=transition+1 if logical=='L108' else 2,
                    existing_R_relative_error_mean=float(r.mean()),best_J_relative_error_mean=float(b.mean()),
                    existing_R_relative_error_percentiles=interval(r,repeats=repeats),
                    best_J_relative_error_percentiles=interval(b,repeats=repeats),
                    per_replica_mse=per_rep_mse,per_replica_R_relative=r,per_replica_best_J_relative=b,
                    R_mean_g_caution='R(mean g)与mean R(g)另外由保存数组列出，不把此闭合RMSE当均值漂移。',
                    thermal_nu=None))
            if levels==3:
                if all_predictions is None:
                    raise ValueError('缺少保存的两步R预测；不得临时推理补算')
                diff=all_predictions[sub,:,0,1]-direct[sub,:,2]
                mse=np.mean(diff**2,axis=(1,2))
                scale=float(metrics['train_representation'][2]['pooled_coordinate_std'])
                rows.append(dict(condition_index=int(cid),split=split,physical_replicas=int(sub.sum()),
                    physical_replica_indices=replica[sub],path='108→27',transition=None,
                    rmse=float(np.sqrt(mse.mean())),nrmse=float(np.sqrt(mse.mean())/scale),
                    fixed_original_TRAIN_target_std=scale,original_TRAIN_target_scale=2,
                    per_replica_mse=mse,thermal_nu=None))
    return rows,response


def paired_comparisons(arms,*,repeats=1000):
    """同一条件同一物理链跨模型配对；不是跨尺寸配对。"""
    results=[]
    for left,right in (('C','A'),('D','B'),('B','A'),('D','C'),('C','E'),('D','E')):
        if left not in arms or right not in arms:continue
        reference={(r['initial_size'],r['condition_index'],r['path'],r['split']):r for r in arms[right]}
        for row in arms[left]:
            key=(row['initial_size'],row['condition_index'],row['path'],row['split'])
            if row['split']!='test' or key not in reference:continue
            ref=reference[key]
            if not np.array_equal(row['physical_replica_indices'],ref['physical_replica_indices']):
                raise ValueError('跨模型物理链配对不一致')
            mse=np.asarray(row['per_replica_mse'])-ref['per_replica_mse']
            record=dict(left=left,right=right,initial_size=key[0],condition_index=key[1],path=key[2],
                mean_paired_mse_difference=float(mse.mean()),
                paired_mse_difference_percentiles=interval(mse,repeats=repeats),
                interval_caution='本条件/路径点态区间，不覆盖所有条件、模型与路径的联合族。')
            if 'per_replica_R_relative' in row:
                a=np.asarray(row['per_replica_R_relative']);b=np.asarray(ref['per_replica_R_relative'])
                record.update(mean_paired_response_norm_difference=float((a-b).mean()),
                    paired_response_norm_difference_percentiles=interval(a-b,repeats=repeats),
                    relative_response_norm_improvement=float(1-a.mean()/b.mean()) if b.mean()>1e-12 else None,
                    response_metric='逐C相对Frobenius范数先平均每物理链；不是MSE。')
            results.append(record)
    return results


def finite_interventions(panels,metadata,*,repeats=1000):
    """真实独立正负分布之间的有限位移；不做同批构型重加权。"""
    mapping={int(r.get('condition_index',r.get('index',-1))):r for r in metadata}
    rows=[]
    for arm,sizes in panels.items():
        for logical,p in sizes.items():
            ids=list(np.unique(p['condition']))
            for plus in ids:
                a=mapping[int(plus)];v=np.asarray(a.get('unit_delta6',a.get('condition',{}).get('unit_delta6',[])))
                nonzero=np.flatnonzero(np.abs(v)>1e-12)
                if a.get('role')!='heldout_domain' or not len(nonzero) or v[nonzero[0]]<0:continue
                candidates=[int(k) for k in ids if mapping[int(k)].get('role')=='heldout_domain'
                    and np.allclose(mapping[int(k)].get('unit_delta6',mapping[int(k)].get('condition',{}).get('unit_delta6',[])),-v)]
                if len(candidates)!=1:raise ValueError('预定正负扰动条件配对不唯一')
                minus=candidates[0];maskp=p['condition']==plus;maskm=p['condition']==minus
                gp=p['direct'][maskp].mean(1);gm=p['direct'][maskm].mean(1)
                rp=p['r_of_g'][maskp].mean(1);rm=p['r_of_g'][maskm].mean(1)
                gen=np.random.default_rng(20261017+int(plus)*100+int(minus))
                ip=gen.integers(len(gp),size=(repeats,len(gp)))
                im=gen.integers(len(gm),size=(repeats,len(gm)))
                for transition in range(gp.shape[1]-1):
                    target=gp[:,transition+1].mean(0)-gm[:,transition+1].mean(0)
                    prediction=rp[:,transition].mean(0)-rm[:,transition].mean(0)
                    boot=(rp[ip,transition].mean(1)-rm[im,transition].mean(1)
                          -gp[ip,transition+1].mean(1)+gm[im,transition+1].mean(1))
                    denom=np.linalg.norm(target)
                    rows.append(dict(arm=arm,initial_size=logical,plus_condition=int(plus),
                        minus_condition=int(minus),family=a['family'],transition=transition,
                        delta_input_mean=gp[:,transition].mean(0)-gm[:,transition].mean(0),
                        delta_output_mean=target,delta_mean_R_prediction=prediction,
                        signed_residual=prediction-target,
                        residual_l2=float(np.linalg.norm(prediction-target)),target_delta_l2=float(denom),
                        relative_residual=float(np.linalg.norm(prediction-target)/denom) if denom>1e-12 else None,
                        signed_residual_percentiles=np.percentile(boot,[2.5,50,97.5],axis=0),
                        bootstrap='正负条件分别重采物理链；同条件跨模型使用相同索引；C先平均。',
                        mean_order='预测为正负条件mean R(g)之差，不冒充R(mean g)之差。',thermal_nu=None))
    return rows


def shared_scales(panels,metadata,arrays,*,repeats=1000):
    mapping={int(r.get('condition_index',r.get('index',-1))):r for r in metadata}
    rows=[]
    for arm,sizes in panels.items():
        large,small=sizes['L108'],sizes['L54']
        for cid in np.unique(small['condition']):
            m=mapping[int(cid)];v=m.get('unit_delta6',m.get('condition',{}).get('unit_delta6'))
            matches=[int(k) for k in np.unique(large['condition'])
                if mapping[int(k)].get('role')=='heldout_domain'
                and mapping[int(k)].get('K0')==m.get('K0')
                and np.allclose(mapping[int(k)].get('unit_delta6',mapping[int(k)].get('condition',{}).get('unit_delta6')),v)]
            if len(matches)!=1:raise ValueError('跨独立尺寸同Hamiltonian条件不唯一')
            other=matches[0];lm=large['condition']==other;sm=small['condition']==cid
            gen=np.random.default_rng(20261017+int(cid))
            li=gen.integers(lm.sum(),size=(repeats,lm.sum()));si=gen.integers(sm.sum(),size=(repeats,sm.sum()))
            for ls,ss,lattice in ((1,0,54),(2,1,27)):
                left=large['direct'][lm,:,ls].mean(1);right=small['direct'][sm,:,ss].mean(1)
                boot=left[li].mean(1)-right[si].mean(1)
                record=dict(arm=arm,large_condition=other,small_condition=int(cid),lattice=lattice,
                    delta_g_larger_minus_smaller=left.mean(0)-right.mean(0),
                    delta_g_pointwise95=np.percentile(boot,[2.5,50,97.5],axis=0),
                    delta_g_coordinate_family95=np.percentile(boot,[2.5/3,50,100-2.5/3],axis=0),
                    caution='两个尺寸物理链独立重采；相同K和末尺度不保证全分布相同；g差不能单独归因Q。')
                if 'L108_physical_even_density' in arrays and 'L54_physical_even_density' in arrays:
                    pl=arrays['L108_physical_even_density'][lm,:,ls].mean(1)
                    ps=arrays['L54_physical_even_density'][sm,:,ss].mean(1)
                    pb=pl[li].mean(1)-ps[si].mean(1)
                    record.update(delta_physical_even_density=pl.mean(0)-ps.mean(0),
                        delta_physical_even_coordinate_family95=np.percentile(pb,[2.5/6,50,100-2.5/6],axis=0),
                        physical_evidence_shared_by_all_R_arms=True)
                rows.append(record)
    return rows


def _histories(root):
    rows={}
    for arm in ('A','B','C','D','E'):
        for path in (root/'training'/arm/'history_progress.json',root/'training/arms'/arm/'history_progress.json'):
            if path.exists():rows[arm]=read(path);break
    return rows


def _metadata(root):
    path=root/'cache/metadata.json'
    if not path.exists():raise ValueError('缺少生成元数据，不能以训练缓存反推耦合标签')
    d=read(path)
    if isinstance(d,list):return d
    return d.get('conditions',d.get('condition_metadata',[]))


def pointwise_R_plot_arrays(panel,max_points=180):
    """按保存顺序固定抽稀真实逐 C 点；绝不以 C 均值点替代 R(g)。"""
    if type(max_points) is not int or max_points<1:
        raise ValueError('绘图抽稀上限必须为正整数')
    direct=np.asarray(panel['direct']);advanced=np.asarray(panel['r_of_g'])
    if direct.shape!=advanced.shape or direct.ndim<2 or direct.shape[-1]!=3 or not direct.size:
        raise ValueError('绘图需要配对保存的三维 direct 与 R(g) 点')
    flat=direct.reshape(-1,3);rflat=advanced.reshape(-1,3)
    stride=max(1,(len(flat)+max_points-1)//max_points)
    return flat[::stride],rflat[::stride]


@contextmanager
def chinese_render_guard(font=None):
    """创建、布局和真实保存全程使用已有中文字体；缺字立即失败。

    旧 chinese_figures 的逐字覆盖检查及 savefig 钩子仍保留。这里只把同一
    字体提前设置到布局阶段，避免 tight_layout 先以 DejaVu Sans 绘制中文。
    addfont 仅在进程内注册已存在的文件，不安装、下载或修改任何字体文件。
    """
    from matplotlib import font_manager
    font_path=select_chinese_font(font)
    font_manager.fontManager.addfont(str(font_path))
    family=font_manager.FontProperties(fname=str(font_path)).get_name()
    with matplotlib.rc_context({'font.family':[family],'font.sans-serif':[family]}):
        with warnings.catch_warnings():
            warnings.filterwarnings('error',message=r'Glyph .* missing from .*font',category=UserWarning)
            with chinese_figures(font_path):
                yield font_path


def plot_report(output,report,arrays,panels,histories,metadata):
    with chinese_render_guard() as font_path:
        print(f'中文绘图字体：{font_path}；布局和保存缺字均立即失败。',flush=True)
        prefix='小格点执行检查，非科学结果：' if report.get('smoke') else ''
        fig,ax=plt.subplots(1,2,figsize=(12,4.5))
        if report['scientific_stop']:
            ax[0].text(.5,.5,'增强域可靠三维支撑不足\n未开展正式 R 更新',ha='center',va='center',transform=ax[0].transAxes)
            ax[0].set_title('可识别性门控：科学停止')
        else:
            for arm,rows in report['arms'].items():
                selected=[r for r in rows if r['split']=='test']
                if selected:
                    ax[0].scatter([LABELS[arm]],[np.mean([r['nrmse'] for r in selected])],label=LABELS[arm])
            ax[0].set_title('留出闭合：条件×路径等权描述统计')
        gate=report['identifiability']
        records=gate.get('conditions',gate.get('rows',gate.get('groups',[])))
        if isinstance(records,dict):records=list(records.values())
        for i,row in enumerate(records):
            if not isinstance(row,dict):continue
            vals=row.get('singular_values',row.get('singular_values_mean'))
            if vals is not None:ax[1].plot(np.arange(1,len(vals)+1),vals,alpha=.3)
        ax[1].set_title('训练支撑奇异谱：不得用白化创造秩')
        ax[1].set_xlabel('奇异方向编号');ax[1].set_ylabel('奇异值')
        ax[0].set_ylabel('原训练标尺归一闭合误差')
        fig.suptitle(prefix+'有限物理扰动与方向均衡：不认证热指数')
        fig.tight_layout();fig.savefig(output/'comparison.png',dpi=160);plt.close(fig)

        fig,ax=plt.subplots(figsize=(9,4.5))
        if not histories:
            ax.text(.5,.5,'门控未通过：没有新增训练损失曲线\n原训练曲线另存为历史参考',ha='center',va='center',transform=ax.transAxes)
        for arm,rows in histories.items():
            if isinstance(rows,dict):rows=rows.get('history',[])
            values=[];steps=[]
            for row in rows:
                value=row.get('train_total',row.get('loss',row.get('total')))
                if value is None and isinstance(row.get('train'),dict):value=row['train'].get('total')
                if value is not None:
                    values.append(value);steps.append(row.get('step',row.get('extra_R_updates',len(steps))))
            if values:ax.plot(steps,values,label=LABELS[arm])
        ax.set_xlabel('实际额外 R 更新次数');ax.set_ylabel('训练损失（各臂目标不同）')
        ax.set_title(prefix+'仅 R 更新的损失；不更新 Q 或 C')
        if histories:ax.legend()
        fig.tight_layout();fig.savefig(output/'loss_comparison.png',dpi=160);plt.close(fig)

        names=list(panels)
        fig1,axes1=plt.subplots(2,3,figsize=(13,8));fig2,axes2=plt.subplots(2,3,figsize=(13,8))
        for index,arm in enumerate(names):
            if index>=6:break
            p=panels[arm]['L108'];g,rg=pointwise_R_plot_arrays(p)
            a,b=axes1.flat[index],axes2.flat[index]
            for component in range(3):
                a.scatter(g[...,component].ravel(),rg[...,component].ravel(),s=5,alpha=.25,label=f'分量{component+1}')
            lo=min(g.min(),rg.min());hi=max(g.max(),rg.max());a.plot([lo,hi],[lo,hi],'k:')
            a.set_title(LABELS[arm]);a.set_xlabel('实际编码 g 分量');a.set_ylabel('完整 R(g) 分量')
            flat=g;move=rg-g
            b.quiver(flat[:,0],flat[:,1],move[:,0],move[:,1],angles='xy',scale_units='xy',scale=1,width=.002)
            ends=flat+move
            for coordinate,setlim in ((0,b.set_xlim),(1,b.set_ylim)):
                low=min(flat[:,coordinate].min(),ends[:,coordinate].min());high=max(flat[:,coordinate].max(),ends[:,coordinate].max())
                pad=max((high-low)*.08,1e-4);setlim(low-pad,high+pad)
            b.set_title(LABELS[arm]);b.set_xlabel('编码 g1');b.set_ylabel('编码 g2')
        for i in range(min(len(names),6),6):axes1.flat[i].set_visible(False);axes2.flat[i].set_visible(False)
        fig1.suptitle(prefix+'逐 C 抽样实际点的完整 R(g)，不是根或临界点')
        fig2.suptitle(prefix+'真实位移 R(g)−g：逐 C 抽样点前两分量投影；第三分量保留数值')
        fig1.tight_layout();fig2.tight_layout()
        fig1.savefig(output/'R_vs_g.png',dpi=160);fig2.savefig(output/'R_displacement.png',dpi=160)
        plt.close(fig1);plt.close(fig2)

        fig,axes=plt.subplots(1,3,figsize=(12,4))
        initial=panels['initial']['L108'];g=initial['direct'].mean(1)
        cid=initial['condition'];look={int(r.get('index',r.get('condition_index',-1))):r for r in metadata}
        zeros=[c for c in np.unique(cid) if look.get(int(c),{}).get('family')=='zero']
        zeros.sort(key=lambda c:look[int(c)]['K0'])
        for component,axis in enumerate(axes):
            for level in range(g.shape[1]):
                if zeros:
                    axis.plot([look[int(c)]['K0'] for c in zeros],
                        [g[cid==c,level,component].mean() for c in zeros],marker='o',label=f'尺度{level}')
            axis.set_xlabel('物理基准 K（仅评价）');axis.set_ylabel(f'编码 g{component+1}')
            axis.set_title('冻结 Q 的零扰动映射');axis.legend()
        fig.suptitle(prefix+'K 到编码的映射：只连接已采集点，无平滑')
        fig.tight_layout();fig.savefig(output/'coupling_by_scale.png',dpi=160);plt.close(fig)


def report(root,previous_root,source_root,output,*,smoke=False):
    root,previous,source,output=map(lambda x:Path(x).resolve(),(root,previous_root,source_root,output))
    if output.exists() or output.is_symlink():raise FileExistsError('正式报告拒绝覆盖或隐式重跑')
    output.mkdir(parents=True,exist_ok=False)
    save(output/'status.json',dict(status='running',workflow_complete=False,no_model_loading=True))
    try:
        status=read(root/'training/status.json')
        if status.get('status') not in ('complete','scientific_stop') or not status.get('workflow_complete'):
            raise ValueError('训练或科学门控尚未完成')
        metrics=read(source/'dim3_control/report/metrics.json')
        ident=read(root/'training/identifiability.json')
        receipt=read(root/'cache/receipt.json')
        if (receipt.get('status')!='complete' or receipt.get('workflow_complete') is not True
                or sha256(root/'cache/collected_statistics.npz')!=receipt.get('public_statistics_sha256')
                or sha256(root/'cache/metadata.json')!=receipt.get('metadata_sha256')):
            raise ValueError('冻结采集统计/生成映射指纹不符，不重推理或自动修复')
        arrays=dict(np.load(root/'training/final_statistics.npz',allow_pickle=False))
        cache_arrays=dict(np.load(root/'cache/collected_statistics.npz',allow_pickle=False))
        # Cache arrays may carry dimensions, but numerical physical labels live only in metadata.json.
        for key,value in cache_arrays.items():
            if key not in arrays:arrays[key]=value
        metadata=_metadata(root)
        names=['initial']+([a for a in ('A','B','C','D','E')] if status['status']=='complete' else [])
        result=dict(status='complete',workflow_complete=True,report_complete=True,smoke=smoke,
            scientific_stop=status['status']=='scientific_stop',identifiability=ident,
            new_QC_updates=0,MI_estimates=0,no_model_loading=True,no_new_inference=True,
            thermal_mode_certified=False,thermal_nu=None,caution=CAUTION,
            original_source_metrics=metrics,
            previous_independent_size_reference=read(previous/'report/comparison.json')['arms']['dim3_control'],
            arms={},source_rows={},finite_intervention_validation=[],histories=_histories(root))
        panels={}
        repeats=100 if smoke else 1000
        for arm in names:
            panels[arm]={};allrows=[]
            for logical in ('L108','L54'):
                def get(key,required=True):return find_array(arrays,arm,logical,key,required)
                def common(key):
                    v=get(key,False)
                    if v is not None:return v
                    for candidate in (f'{logical}_{key}',f'{logical.lower()}_{key}'):
                        if candidate in arrays:return arrays[candidate]
                    return find_array(arrays,'initial',logical,key)
                direct=get('direct');rg=get('r_of_g');dr=get('full_DR')
                if not np.array_equal(direct,cache_arrays[f'{logical}_direct']):
                    raise ValueError('训练评价编码与冻结采集缓存不一致')
                vin,vout,std=common('vin'),common('vout'),common('score_std')
                condition=common('condition_index').astype(int)
                replica=common('replica_index').astype(int)
                train=common('train_mask').astype(bool)
                all_predictions=get('all_start_predictions')
                rows,response=group_rows(direct,rg,dr,vin,vout,std,condition,replica,train,metrics,
                    logical,smoke=smoke,repeats=repeats,all_predictions=all_predictions)
                for row in rows:row.update(initial_size=logical)
                allrows.extend(rows)
                eig=np.linalg.eigvals(dr)
                panels[arm][logical]=dict(direct=direct,r_of_g=rg,full_DR=dr,
                    condition=condition,replica=replica,train=train)
                key=f'{arm}_{logical}_'
                # Preserve every saved mean-order, horizon, derivative and spectrum array,
                # not merely the coordinates visible in a two-dimensional figure.
                kept={k[len(key):]:v for k,v in arrays.items() if k.startswith(key)}
                kept.update(full_DR_eigenvalues_real=eig.real,full_DR_eigenvalues_imag=eig.imag)
                np.savez_compressed(output/f'{key}evaluation.npz',**kept)
            if arm=='initial':result['source_rows']=allrows
            else:result['arms'][arm]=allrows
        result['paired_arm_comparisons']=paired_comparisons(result['arms'],repeats=repeats)
        result['finite_intervention_validation']=finite_interventions(panels,metadata,repeats=repeats)
        result['independent_size_shared_scale_validation']=shared_scales(panels,metadata,arrays,repeats=repeats)
        result['finite_intervention_caution']='有限割线与同批有限分布点闭合相关，不当两份独立证据；正负物理链独立。'
        result['sampling_quality_warnings']=[dict(condition_index=r['condition_index'],
            family=r['family'],size=r['size'],quality=r.get('quality')) for r in metadata
            if r.get('quality',{}).get('status')=='warning']
        plot_report(output,result,arrays,panels,result['histories'],metadata)
        old=source/'dim3_control/report/loss_comparison.png'
        shutil.copy2(old,output/'source_training_loss_reference.png')
        save(output/'comparison.json',result)
        save(output/'report.json',dict(status='complete',workflow_complete=True,report_complete=True,
            scientific_stop=result['scientific_stop'],smoke=smoke,thermal_nu=None,
            no_model_loading=True,no_new_inference=True,new_QC_updates=0,MI_estimates=0,
            caution=CAUTION,files=sorted(p.name for p in output.iterdir())))
        save(output/'status.json',dict(status='complete',workflow_complete=True,report_complete=True,
            scientific_stop=result['scientific_stop'],smoke=smoke,thermal_nu=None))
        return result
    except BaseException as error:
        save(output/'status.json',dict(status='failed',workflow_complete=False,requires_user_action=True,
            error=f'{type(error).__name__}: {error}',no_model_loading=True))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','previous-root','source-root','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    report(**vars(p.parse_args()))


if __name__=='__main__':main()
