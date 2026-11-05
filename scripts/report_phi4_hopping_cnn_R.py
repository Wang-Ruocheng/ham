"""JSON/NPZ-only Chinese report of the fresh R on frozen scalar CNN-Q."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scripts import report_phi4_hopping_qonly_balance as common

PAIRS=np.asarray([(s,t) for t in range(1,4) for s in range(t)])
INPUTS=('plan.json','history.json','metrics.json','roots.json','summary.json','evaluation_arrays.npz')
COLORS=('#0072B2','#D55E00','#009E73','#CC79A7')
STABILITY={'attractor':'吸引点','repeller':'排斥点','near_neutral':'近中性点'}


def load_run(root):
    root=Path(root);pins={n:common._sha(root/n) for n in INPUTS}
    plan,history,metrics,roots,summary=[json.loads((root/n).read_text()) for n in INPUTS[:5]]
    if (summary.get('status')!='complete' or summary.get('actual_steps')!=3000 or summary.get('Q_updates')!=0
        or summary.get('C_updates')!=0 or summary.get('labels_used_in_training') is not False
        or summary.get('test_used_in_training') is not False or summary.get('thermal_nu') is not None
        or summary.get('physical_fixed_points_certified') is not False):raise ValueError('Invalid R-only summary')
    if [r['step'] for r in history]!=list(range(0,3001,100)):raise ValueError('Incomplete fixed endpoint history')
    with np.load(root/'evaluation_arrays.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    for key,v in a.items():
        if v.dtype.kind in 'fc' and not np.isfinite(v).all():raise ValueError('Nonfinite '+key)
    for key,shape in [('direct_g',(60,4,1)),('direct_full_g',(60,4,1)),('labels',(60,3))]:common._numeric(a[key],key,shape)
    tr=a['splits']=='train'
    if tr.sum()!=40 or (a['splits']=='test').sum()!=20:raise ValueError('Invalid split')
    if not np.array_equal(a['all_start_pairs'],PAIRS):raise ValueError('Wrong all-start pair order')
    for cell in np.unique(a['cell_ids']):
        mask=a['cell_ids']==cell
        if mask.sum()!=4 or len(set(a['splits'][mask]))!=1 or not np.all(a['labels'][mask]==a['labels'][mask][0]):raise ValueError('Condition split overlap')
    if set(map(tuple,a['labels'][tr])) & set(map(tuple,a['labels'][~tr])):raise ValueError('Label split overlap')
    for key,value in [('train_target_std',a['direct_g'][tr].std(0,ddof=0)[:,0]),('train_target_mean',a['direct_g'][tr].mean(0)[:,0])]:
        if not np.allclose(a[key],value,rtol=1e-10,atol=1e-12):raise ValueError('TRAIN normalization changed')
    if not (np.diff(a['flow_grid'][:,0])>0).all():raise ValueError('Unordered flow grid')
    for model in ('initial','final'):
        for suffix,shape in [('all_start',(60,6,1)),('one_step',(60,3,1)),('flow_next',a['flow_grid'].shape)]:common._numeric(a[model+'_'+suffix],suffix,shape)
        for k,i in enumerate((0,2,5)):
            if not np.allclose(a[model+'_all_start'][:,i],a[model+'_one_step'][:,k],rtol=1e-10,atol=1e-12):raise ValueError('Prediction pair mismatch')
        for split in ('train','test'):
            mask=a['splits']==split
            e=np.square(a[model+'_all_start'][mask]-a['direct_g'][mask][:,PAIRS[:,1]]).mean((0,2))
            rows=metrics[model][split]['pairs']
            if len(rows)!=6:raise ValueError('Incomplete metrics')
            for i,(_,t) in enumerate(PAIRS):
                if not np.allclose([rows[i]['rmse'],rows[i]['nrmse']],[np.sqrt(e[i]),np.sqrt(e[i])/a['train_target_std'][t]],rtol=1e-9,atol=1e-12):raise ValueError('Metrics disagree')
            expected=sum(10/(3*t)*v for (_,t),v in zip(PAIRS,e))
            if not np.isclose(expected,metrics[model][split]['equal_target_objective'],rtol=1e-9,atol=1e-12):raise ValueError('Objective differs')
    return plan,history,metrics,roots,summary,a,pins


def root_mappings(roots,a):
    """Posthoc inverse of saved condition means; never call this physical Kc."""
    rows=common._groups(a['direct_g'][:,:,0],a);kappa=np.array([r['kappa'] for r in rows]);g=np.array([r['mean'] for r in rows]);result=[]
    for original in roots['roots']:
        row=dict(original);maps=[]
        for k in range(4):
            values=g[:,k];d=np.diff(values);monotone=bool(np.all(d>0) or np.all(d<0))
            value=None;bracket=None
            if monotone and values.min()<=row['g']<=values.max():
                order=np.argsort(values);xx=values[order];yy=kappa[order]
                value=float(np.interp(row['g'],xx,yy));i=int(np.clip(np.searchsorted(xx,row['g'])-1,0,len(xx)-2))
                bracket=sorted([float(yy[i]),float(yy[i+1])])
            maps.append(dict(scale=k,condition_mean_monotone=monotone,posthoc_interpolated_kappa=value,bracketing_kappa=bracket,physical_Kc_certified=False))
        row['posthoc_coordinate_mappings']=maps;result.append(row)
    return result


def render_report(output):
    root=Path(output).resolve();plan,history,metrics,roots,summary,a,pins=load_run(root)
    dest=root/'report'
    if dest.exists():raise FileExistsError('No report overwrite')
    dest.mkdir();plt=common._plot_setup();plots=[];mapped=root_mappings(roots,a)
    def save(fig,name):common._save(fig,dest/name,plt);plots.append(name)
    grid=a['flow_grid'][:,0];nxt=a['final_flow_next'][:,0];train=a['splits']=='train'
    low,high=float(a['direct_g'][train,:3].min()),float(a['direct_g'][train,:3].max())
    fig,axes=plt.subplots(1,3,figsize=(18,5.6),constrained_layout=True)
    for ax in axes[:2]:
        ax.plot(grid,grid,'k--',lw=1,label='恒等映射');ax.plot(grid,nxt,color='#CC79A7',lw=2,label='训练后的 R(g)')
        for k in range(3):
            for split,marker in [('train','o'),('test','x')]:
                mask=a['splits']==split;ax.scatter(a['direct_g'][mask,k,0],a['direct_g'][mask,k+1,0],s=16,marker=marker,alpha=.6,color=COLORS[k],label=f'k={k}→{k+1}，'+('训练' if split=='train' else '留出'))
        for lo,hi in [(grid.min(),low),(high,grid.max())]:
            if hi>lo:ax.axvspan(lo,hi,color='gray',alpha=.16)
        for i,r in enumerate(mapped):
            ax.scatter([r['g']],[r['g']],marker='D',s=45,facecolors='black' if r['inside_train_input_envelope'] else 'white',edgecolors='black',zorder=10)
            ax.annotate(f"{i+1}：{STABILITY[r['numerical_stability']]}",(r['g'],r['g']),xytext=(6,7),textcoords='offset points',fontsize=8)
        ax.set(xlabel='CNN-Q 的输入 g',ylabel='输出 g');ax.grid(alpha=.2)
    axes[0].set(title='完整 R(g) 与相邻尺度数据',xlim=(grid.min(),grid.max()));axes[0].legend(fontsize=7,ncol=2)
    micro=a['direct_g'][train,0,0];pad=.1*np.ptp(micro);axes[1].set(title='微观输入区放大：检查同一 g 的目标冲突',xlim=(micro.min()-pad,micro.max()+pad))
    ax=axes[2];indices=np.linspace(0,len(grid)-1,35).astype(int)
    ax.quiver(grid[indices],np.zeros(len(indices)),nxt[indices]-grid[indices],np.zeros(len(indices)),angles='xy',scale_units='xy',scale=1,color='#CC79A7',width=.005)
    for k in range(3):
        for split,marker in [('train','|'),('test','x')]:
            mask=a['splits']==split;v=a['direct_g'][mask,k,0];ax.scatter(v,np.full_like(v,-.12-.1*k),marker=marker,s=25,color=COLORS[k],alpha=.65)
        ax.text(grid.min(),-.12-.1*k,f'k={k}',fontsize=8,va='bottom')
    for i,r in enumerate(mapped):
        ax.scatter([r['g']],[0],marker='D',s=45,facecolors='black' if r['inside_train_input_envelope'] else 'white',edgecolors='black',zorder=10)
        ax.annotate(str(i+1),(r['g'],0),xytext=(0,10),textcoords='offset points')
    ax.set(xlim=(min(grid.min(),nxt[indices].min()),max(grid.max(),nxt[indices].max())),ylim=(-.4,.13),yticks=[],xlabel='CNN-Q 坐标 g',title='真实位移 R(g)−g；下方为各尺度数据覆盖');ax.grid(alpha=.2)
    fig.suptitle('冻结一维 CNN-Q 与 C，随机 R 单独训练 3000 步\n灰色：TRAIN 输入包络之外；空心菱形：外推不动点；包络内也不保证充分支撑；数值不动点不等于物理临界点')
    save(fig,'R_flow.png')
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    rows=common._groups(a['direct_g'][:,:,0],a)
    for k,ax in enumerate(axes.flat):
        x=[r['kappa'] for r in rows];y=np.array([r['mean'][k] for r in rows]);sd=np.array([r['chain_sd'][k] for r in rows]);ax.plot(x,y,color=COLORS[k]);ax.fill_between(x,y-sd,y+sd,color=COLORS[k],alpha=.15)
        for r in rows:ax.scatter([r['kappa']],[r['mean'][k]],color=COLORS[k],marker='o' if r['split']=='train' else 'x')
        ax.set(title=f'冻结 Q：k={k}，边长 {128//2**k}',xlabel='κ（仅事后评价）',ylabel='g');ax.grid(alpha=.2)
    fig.suptitle('原 CNN-Q 输出原样复用，不重新计算；圆点训练、叉号留出；带宽为四链标准差，非置信区间')
    save(fig,'Q_g_kappa.png')
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for index,ax in zip([0,2,5,3],axes.flat):
        s,t=PAIRS[index];truth=common._groups(a['direct_g'][:,t,0],a);pred=common._groups(a['final_all_start'][:,index,0],a)
        for rr,color,label in [(truth,'black',f'Q 在 k={t} 的直接输出'),(pred,'#CC79A7',f'R 迭代 {t-s} 次')]:
            x=[r['kappa'] for r in rr];y=np.array([r['mean'] for r in rr]);sd=np.array([r['chain_sd'] for r in rr]);ax.plot(x,y,color=color,label=label);ax.fill_between(x,y-sd,y+sd,color=color,alpha=.1)
            for r in rr:ax.scatter([r['kappa']],[r['mean']],color=color,marker='o' if r['split']=='train' else 'x',s=20)
        ax.set(title=f'k={s}→{t}',xlabel='κ（仅事后评价）',ylabel='目标尺度 g');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('闭合预测与冻结 Q 的直接输出；四链均值±链间标准差')
    save(fig,'closure_predictions.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),constrained_layout=True);steps=[r['step'] for r in history]
    axes[0].plot(steps,[r['equal_target_objective'] for r in history]);axes[0].set(title='训练集 all-start 闭合损失',xlabel='R 更新次数',ylabel='总损失',yscale='log')
    for i,(s,t) in enumerate(PAIRS):axes[1].plot(steps,[r['pairs'][i]['nrmse'] for r in history],label=f'{s}→{t}')
    axes[1].set(title='训练集各预测跨度的 NRMSE',xlabel='R 更新次数',ylabel='原 CNN-Q TRAIN 目标标准差归一化',yscale='log');axes[1].legend(ncol=2)
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('只记录 TRAIN 曲线；固定 3000 步终点；未按 TEST 或不动点选模')
    save(fig,'loss_curves.png')
    if pins!={n:common._sha(root/n) for n in INPUTS}:raise RuntimeError('Report input changed')
    report=dict(status='complete',source_sha256=pins,metrics=metrics,numerical_roots=mapped,plots=plots,Q_updates=0,C_updates=0,R_updates=3000,
        critical_kappa=None,thermal_nu=None,limitations=[
            '固定 CNN-Q、C 及60条链的表示缓存，非重新抽样构型的联合 QCR 续训。',
            '标签仅事后画图；各尺度 κ 逆映射仅为数值坐标定位，不认证临界 κ。',
            '网格变号搜索可能漏掉切触根和同一网格内多个根；包络内不等于数据密集支撑。',
            '数值稳定性按完整 R 的导数绝对值判断，不将残差网络导数当作 R 导数。',
            '单数据、单随机种子、有限尺寸、保留75项采样警告；未认证热化或普适指数。'])
    (dest/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return ['report/'+n for n in [*plots,'report.json']]
