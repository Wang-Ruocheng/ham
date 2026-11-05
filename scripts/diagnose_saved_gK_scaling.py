"""仅从保存的标量 g(K) 和公共 R 数值诊断局部放大、窗口偏差与漂移。"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.report_ising_zh import chinese_figures


KC = .5*np.log(1+np.sqrt(2))
WIDTHS = (.02, .03, .04, .06)
COLORS = ('#0072B2', '#D55E00', '#009E73')
PURPLE = '#9C27B0'


def serialize(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def crossings(x, difference):
    roots, plateaus = [], []
    for j in range(len(x)-1):
        a, b = difference[j:j+2]
        if a == 0 and b == 0:
            plateaus.append([x[j], x[j+1]])
        elif a == 0:
            roots.append(x[j])
        elif a*b < 0 or b == 0:
            roots.append(x[j] - a*(x[j+1]-x[j])/(b-a))
    return dict(K_branches=sorted(set(roots)), K_plateaus=plateaus)


def inverse_curves(knots, means, value):
    return [crossings(knots, means[:, k]-value) for k in range(3)]


def horizontal_collapse(knots, means, multiplier, critical_index):
    """有限窗口曲线几何拟合；不是训练，也不是热模选择。"""
    rows=[]
    for width in WIDTHS:
        mask=abs(knots-KC)<=width
        x=knots[mask]
        for step in range(2):
            for anchored in (False,True):
                offset=(means[critical_index,step+1]-means[critical_index,step]) if anchored else 0.
                target=means[mask,step+1]
                def error(a):
                    query=KC+a*(x-KC)
                    if query.min()<knots[0] or query.max()>knots[-1]:
                        return float('inf')
                    predicted=np.interp(query,knots,means[:,step])+offset
                    return float(np.sqrt(np.mean((predicted-target)**2)))
                grid=np.linspace(.5,3.,5001)
                costs=np.array([error(a) for a in grid])
                best=int(np.argmin(costs))
                left,right=grid[max(0,best-1)],grid[min(len(grid)-1,best+1)]
                phi=(np.sqrt(5)-1)/2
                for _ in range(80):
                    a=right-phi*(right-left); b=left+phi*(right-left)
                    if error(a)<error(b): right=b
                    else: left=a
                optimal=.5*(left+right)
                rows.append(dict(half_width=width,transition=step,K_count=int(mask.sum()),
                    offset_mode='固定为各尺度Kc编码之差' if anchored else '不允许纵向平移',
                    anchored=anchored,offset=offset,best_horizontal_multiplier=optimal,
                    best_rmse=error(optimal),rmse_at_2=error(2.),
                    rmse_at_saved_R_multiplier=error(multiplier),
                    search_interval=[.5,3.],search_grid_points=5001,
                    caution='只比较有限窗口曲线形状；折线插值、尺度漂移与离散K网格使其不等于固定点导数。'))
    return rows


def prepare(folder, name, epochs):
    path = folder/'encodings.npz'
    with np.load(path, allow_pickle=False) as z:
        labels = np.r_[z['train_labels'], z['test_labels']]
        values = np.r_[z['train'], z['test']][..., 0]
    knots = np.unique(labels)
    groups = np.stack([values[labels == k] for k in knots])
    if groups.shape != (64, 12, 3) or not np.isfinite(groups).all():
        raise ValueError('需要64个K、每K12独立系综、三个保存的标量尺度')
    report = json.loads((folder/'critical_exponents.json').read_text())
    root = next(r for r in report['roots'] if r['root_id'] == 'F2')
    if 'jacobian' in root:
        multiplier = root['jacobian'][0][0]
    else:
        multiplier = root['multiplier']
    return dict(name=name, epochs=epochs, folder=str(folder.resolve()), K=knots,
        grouped_encodings=groups, mean=groups.mean(1), sd=groups.std(1, ddof=1),
        root=root, root_g=float(root['g'][0]), root_multiplier=float(multiplier),
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def analyze(arm, *, bootstraps=1000):
    knots, means = arm['K'], arm['mean']
    critical_index = int(np.argmin(abs(knots-KC)))
    if abs(knots[critical_index]-KC) > 1e-12:
        raise ValueError('没有保存精确Kc数据点')
    rng = np.random.default_rng(20261016)
    selections = rng.integers(0, 12, (bootstraps, len(knots), 12))
    boot_means = arm['grouped_encodings'][np.arange(len(knots))[None, :, None], selections].mean(2)
    fits = []
    for width in WIDTHS:
        inside = abs(knots-KC) <= width
        u = (knots[inside]-KC)/width
        for degree in (1, 2, 3):
            design = np.polynomial.polynomial.polyvander(u, degree)
            inverse = np.linalg.pinv(design)
            coefficients = inverse @ means[inside]
            slopes = coefficients[1]/width
            ratios = slopes[1:]/slopes[:-1]
            boot_slopes = np.einsum('n,bns->bs', inverse[1]/width, boot_means[:, inside])
            boot_ratios = boot_slopes[:, 1:]/boot_slopes[:, :-1]
            fits.append(dict(half_width=width, degree=degree, K_count=int(inside.sum()),
                K_knots=knots[inside], fitted_slopes=slopes, slope_ratios=ratios,
                nominal_inverse_y=np.log(2)/np.log(ratios),
                fit_rmse=np.sqrt(((design @ coefficients-means[inside])**2).mean(0)),
                ratio_conditional_bootstrap_percentiles=np.percentile(boot_ratios, [2.5,50,97.5], axis=0),
                thermal_nu_certified=False))
    matched=[]
    for width in WIDTHS:
        pair=[]
        for w in (width,2*width):
            mask=abs(knots-KC)<=w
            coefficients=np.polynomial.polynomial.polyfit((knots[mask]-KC)/w,means[mask],3)
            pair.append(coefficients[1]/w)
        matched.append(dict(output_half_width=width,input_half_width=2*width,
            slope_ratios=pair[0][1:]/pair[1][:-1],
            caution='仅检验横向缩放2倍假设下的一致性；窗口也乘2，不是独立测得的临界倍率。'))
    value = arm['root_g']
    branches = inverse_curves(knots, means, value)
    at_root = []
    for step in range(2):
        for initial_k in branches[step]['K_branches']:
            observed = np.interp(initial_k, knots, means[:, step+1])
            at_root.append(dict(transition=step, input_g=value, original_K=initial_k,
                observed_next_mean_g=observed, observed_displacement=observed-value,
                R_at_input_g=value,
                caution='R_at_input_g读取保存的数值根；数据输出仅作原K折线插值。'))
    pair_crossings = []
    for first, second in ((0,1),(1,2),(0,2)):
        record = crossings(knots, means[:, second]-means[:, first])
        pair_crossings.append(dict(scales=[first,second], **record,
            near_Kc_branches=[dict(K=k,g=np.interp(k,knots,means[:, first]))
                for k in record['K_branches'] if abs(k-KC)<=.06]))
    clean = {k:v for k,v in arm.items() if k != 'grouped_encodings'}
    return dict(**clean, critical_index=critical_index,
        g_at_Kc=means[critical_index], fits=fits,
        saved_root_inverse_K_by_scale=branches, data_mapping_at_saved_R_root=at_root,
        pairwise_curve_crossings=pair_crossings,
        horizontal_collapse=horizontal_collapse(knots,means,arm['root_multiplier'],critical_index),
        matched_windows_for_a2_hypothesis=matched)


def saved_R_samples(folder, knots, means):
    report = json.loads((folder/'critical_exponents.json').read_text())
    jensen = report['jensen']
    if not np.allclose(jensen['K'], knots, rtol=0, atol=1e-12):
        raise ValueError('保存R值与K曲线网格不一致')
    inputs = means.ravel()
    outputs = np.asarray(jensen['R_of_mean'])[..., 0].ravel()
    order = np.argsort(inputs)
    return inputs[order], outputs[order]


def plot(unsupervised, folder, output):
    arm = unsupervised
    knots, means, ic = arm['K'], arm['mean'], arm['critical_index']
    fig, grid = plt.subplots(2,2,figsize=(15.5,11.6))
    axes=grid.ravel()
    fig.subplots_adjust(left=.065,right=.985,bottom=.14,top=.85,wspace=.26,hspace=.38)
    for k, (size,color) in enumerate(zip((108,54,27),COLORS)):
        axes[0].plot(knots,means[:,k],'-o',color=color,markersize=3.5,lw=1.8,
                     label=f'k={k}，尺寸{size}×{size}')
        axes[0].scatter(KC,means[ic,k],c=PURPLE,marker='D',s=52,edgecolors='white',zorder=5)
    axes[0].axvline(KC,c=PURPLE,ls=':',lw=1.2)
    axes[0].axhline(arm['root_g'],c='#555555',ls='--',lw=1,label='保存的 R 根坐标')
    axes[0].set(xlim=(KC-.03,KC+.03),xlabel='原始微观 K',ylabel='平均编码 g',
                title='临界附近：曲线靠拢，但不完全重合')
    axes[0].legend(fontsize=9)
    axes[1].plot(means[:,0],means[:,1],'-o',color=COLORS[0],markersize=3.5,
                 label='数据映射：108→54')
    axes[1].plot(means[:,1],means[:,2],'-o',color=COLORS[1],markersize=3.5,
                 label='数据映射：54→27')
    rx,ry=saved_R_samples(folder,knots,means)
    axes[1].plot(rx,ry,c=COLORS[2],lw=1.8,label='保存的 R（作用于均值）')
    low,high=.17,.27
    axes[1].plot([low,high],[low,high],'--',c='#555555',lw=1,label='恒等映射')
    axes[1].scatter(arm['root_g'],arm['root_g'],s=100,marker='*',c=PURPLE,zorder=6,
                    label='保存的 R 不动点')
    outputs=[v['observed_next_mean_g'] for v in arm['data_mapping_at_saved_R_root']]
    axes[1].plot([arm['root_g']]*2,outputs,c='#555555',lw=2)
    axes[1].scatter([arm['root_g']]*2,outputs,c=COLORS[:2],s=40,zorder=6)
    axes[1].set(xlim=(low,high),ylim=(low,high),xlabel='当前尺度平均 g',
                ylabel='下一尺度平均 g',title='同一个输入 g：两步数据映射并不相同')
    axes[1].legend(fontsize=8)
    cubic=[f for f in arm['fits'] if f['degree']==3]
    for step,color in enumerate(COLORS[:2]):
        y=np.array([f['slope_ratios'][step] for f in cubic])
        ci=np.array([f['ratio_conditional_bootstrap_percentiles'] for f in cubic])[:,:,step]
        axes[2].errorbar(WIDTHS,y,yerr=np.maximum(np.array([y-ci[:,0],ci[:,2]-y]),0),
            color=color,marker='o',capsize=3,label=('108→54' if step==0 else '54→27')+'斜率比')
    axes[2].axhline(2,c='#555555',ls='--',lw=1,label='理论热倍率：仅作参照')
    axes[2].axhline(arm['root_multiplier'],c=PURPLE,ls=':',lw=1.5,
        label=f'保存的 R 根倍率：{arm["root_multiplier"]:.4f}')
    axes[2].set(xlabel='以 Kc 为中心的拟合窗口半宽',ylabel='相邻尺度曲线的斜率比',
                title='三次局部拟合：读数随窗口改变',xticks=WIDTHS)
    axes[2].legend(fontsize=8)
    for step,color in enumerate(COLORS[:2]):
        records=[r for r in arm['horizontal_collapse'] if r['anchored'] and r['transition']==step]
        axes[3].plot([r['half_width'] for r in records],
            [r['best_horizontal_multiplier'] for r in records],'-o',c=color,
            label=('108→54' if step==0 else '54→27')+'曲线横向缩放')
    axes[3].axhline(2,c='#555555',ls='--',lw=1,label='理论热倍率：仅作参照')
    axes[3].axhline(arm['root_multiplier'],c=PURPLE,ls=':',lw=1.5,
        label=f'保存的 R 根倍率：{arm["root_multiplier"]:.4f}')
    axes[3].set(xlabel='曲线比较窗口半宽',ylabel='最佳横向缩放倍率',
        title='整段曲线形状：横向缩放更接近2',xticks=WIDTHS)
    axes[3].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.18)
        ax.spines[['top','right']].set_visible(False)
        ax.ticklabel_format(axis='y',style='plain',useOffset=False)
    fig.suptitle('g(K) 诊断：局部斜率、尺度漂移与整段曲线缩放',fontsize=18,y=.975)
    fig.text(.5,.918,'完全无监督，一维表示，12000次更新；只使用保存的编码与 R 数值，不运行模型。',
             ha='center',fontsize=12)
    fig.text(.065,.092,'上排：每 K 的12系综均值，折线插值；紫菱形是 Kc 输入编码。下排：窗口含6、8、10、16个实际 K。',fontsize=10)
    fig.text(.065,.061,'左下：局部三次拟合，误差线仅覆盖系综采样。右下：比较 g 在横向缩放后的形状，纵向基线差固定为实际 Kc 编码差。',fontsize=10)
    fig.text(.065,.030,'两种倍率不是同一估计；都不能直接认证热指数。有限网格、窗口偏差与非共同固定点仍须验证。',fontsize=10)
    buffer=io.BytesIO()
    try:
        fig.savefig(buffer,format='png',dpi=180,facecolor='white')
        with output.open('xb') as stream: stream.write(buffer.getvalue())
    finally:
        plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--unsupervised',type=Path,required=True)
    p.add_argument('--supervised',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError('拒绝覆盖已有诊断')
    unsup=analyze(prepare(args.unsupervised,'完全无监督',12000))
    sup=analyze(prepare(args.supervised,'微观K监督',9000))
    payload=dict(status='complete',model_inference=False,model_loading=False,training=False,
        physical_sampling=False,critical_reference_K=KC,bootstrap_seed=20261016,bootstrap_draws=1000,
        bootstrap='每K分别重采12个独立物理系综，三个尺度配对；条件于单训练模型与保存C抽样。',
        limitations=['窗口斜率比不是固定点热本征值；未按理论选择窗口。',
            '监督9000与无监督12000不是同预算因果对照。',
            '仅一个初始格点尺寸、两个粗粒化步和单训练seed。',
            '多项式拟合和折线插值均为保存数据的离线诊断，不生成新物理数据或修改模型。'],
        arms=[unsup,sup])
    args.output.mkdir(parents=True,exist_ok=False)
    with chinese_figures(): plot(unsup,args.unsupervised,args.output/'diagnostic.png')
    with (args.output/'diagnostic.json').open('x') as stream:
        json.dump(payload,stream,ensure_ascii=False,indent=2,default=serialize,allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status='complete',output=str(args.output.resolve()),
        slopes=[dict(name=a['name'],main=next(f for f in a['fits']
            if f['degree']==3 and f['half_width']==.03),
            at_root=a['data_mapping_at_saved_R_root']) for a in (unsup,sup)]),
        ensure_ascii=False,default=serialize))


if __name__=='__main__': main()
