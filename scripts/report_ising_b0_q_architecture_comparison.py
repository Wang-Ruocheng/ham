"""Compare saved public results of matched CNN/local-MLP scalar supervision."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.report_phi4_hopping_qcr import _setup_plot


def load(root):
    def read(name): return json.loads((root/name).read_text())
    job = read('job.json')
    assert job['status'] == 'complete' and job['workflow_complete'] and job['returncode'] == 0
    with np.load(root/'evaluation_arrays.npz', allow_pickle=False) as z:
        arrays = {key: z[key] for key in z.files}
    return dict(manifest=read('manifest.json'), summary=read('summary.json'),
                report=read('report/report.json'), history=read('history.json'), arrays=arrays)


def render(cnn_root, mlp_root, iid_reference, output):
    runs = [load(cnn_root), load(mlp_root)]
    assert all(r['summary']['actual_steps'] == 550 for r in runs)
    left, right = [r['manifest']['config'] for r in runs]
    left = json.loads(json.dumps(left)); left['model']['q_encoder'] = 'local_motif'
    assert left == right
    assert runs[0]['manifest']['data_sha256'] == runs[1]['manifest']['data_sha256']
    for key in ('labels', 'splits'):
        assert np.array_equal(runs[0]['arrays'][key], runs[1]['arrays'][key])
    output.mkdir(exist_ok=False)
    names = ['逐构型 CNN', '早期局部 MLP']
    plt = _setup_plot(None)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    records = []
    for col, (name, run) in enumerate(zip(names, runs)):
        a = run['arrays']; g = a['direct_g'][..., 0]; K = a['labels'][:, 0]
        train = a['splits'] == 'train'; knots = np.unique(K); low = K <= .32
        mean = np.array([g[K == v].mean(0) for v in knots])
        std = np.array([g[K == v].std(0, ddof=1) for v in knots])
        ax = axes[0, col]
        for k in range(3):
            ax.plot(knots, mean[:, k], color=f'C{k}', label=f'k={k}，L={36//3**k}')
            ax.fill_between(knots, mean[:, k]-std[:, k], mean[:, k]+std[:, k], color=f'C{k}', alpha=.12)
            ax.scatter(K[~train], g[~train, k], s=12, marker='x', color=f'C{k}')
        ax.set(xlabel='原始耦合 K', ylabel='编码 g', title=name+'：各尺度直接编码')
        ax.legend(fontsize=8)
        ax = axes[1, col]; grid = a['flow_grid'][:, 0]; nxt = a['flow_next'][:, 0]
        ax.plot(grid, nxt, color='C4', label='R(g)'); ax.plot(grid, grid, 'k--', lw=1, label='恒等映射')
        for k in range(2):
            for mask, marker, label in [(train, 'o', '训练'), (~train, 'x', '留出')]:
                ax.scatter(g[mask, k], g[mask, k+1], s=10, marker=marker, color=f'C{k}', label=f'k={k}→{k+1}，{label}')
        for r in run['report']['roots']['roots']:
            label = {'repeller':'排斥点', 'attractor':'吸引点', 'neutral':'近中性点'}[r['numerical_stability']]
            ax.scatter(r['g'], r['g'], marker='D', s=36, facecolors='black' if r['inside_train_input_envelope'] else 'white', edgecolors='black')
            ax.annotate(label, (r['g'], r['g']), xytext=(3,7), textcoords='offset points', fontsize=8)
        lo, hi = g[train, :-1].min(), g[train, :-1].max()
        ax.axvspan(grid.min(), lo, color='.8', alpha=.3); ax.axvspan(hi, grid.max(), color='.8', alpha=.3)
        ax.set(xlabel='输入 g', ylabel='输出 g', title=name+'：共享 R 与直接编码目标'); ax.legend(fontsize=7)
        rows=[]
        for i, row in enumerate(run['summary']['closure']['test']['all_start']):
            ref=runs[0]['summary']['closure']['test']['all_start'][i]['train_target_std']
            rows.append(dict(source=row['source_scale'],target=row['target_scale'],rmse=row['rmse'],
                             nrmse_CNN_train_target_std=row['rmse']/ref,nrmse_own_train_target_std=row['nrmse']))
        records.append(dict(architecture=name,low_K_interval=[float(K[low].min()),float(K[low].max())],
            low_K_mean_g=g[low].mean(0).tolist(),test_supervision=run['summary']['supervision']['test']['direct_full_g'],
            test_closure=rows,roots=run['report']['roots']['roots']))
    for ax in axes.flat: ax.grid(alpha=.2)
    for row in range(2):
        low=min(axes[row,c].get_ylim()[0] for c in range(2));high=max(axes[row,c].get_ylim()[1] for c in range(2))
        for c in range(2):axes[row,c].set_ylim(low,high)
    fig.suptitle('同数据、同监督强度、同 550 步：只改变 Q 架构\n上排阴影为三条物理链标准差；下排灰色为训练输入包络外')
    fig.savefig(output/'Q_R_comparison.png',dpi=170);plt.close(fig)
    old_iid=json.loads(iid_reference.read_text())['iid']
    new_iid=json.loads((mlp_root/'iid_size_control.json').read_text())['rows']
    fig,ax=plt.subplots(figsize=(8,4.7),constrained_layout=True)
    for name,rows,record in zip(names,[old_iid,new_iid],records):
        ax.errorbar([r['L'] for r in rows],[np.mean(r['g']) for r in rows],yerr=[np.std(r['g'],ddof=1) for r in rows],marker='o',capsize=4,label=name)
        record['iid_size_means']={str(r['L']):float(np.mean(r['g'])) for r in rows}
        record['iid_L4_minus_L36']=record['iid_size_means']['4']-record['iid_size_means']['36']
    ax.set(xlabel='格点边长 L',ylabel='冻结 Q 输出 g',title='独立随机自旋：相同随机样本、每尺寸 4 组×256 构型',xticks=[4,6,8,12,18,36])
    ax.legend();ax.grid(alpha=.2);fig.savefig(output/'iid_size_comparison.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for name,run in zip(names,runs):
        h=run['history']
        for ax,key,title in zip(axes,['q_supervision','c_total','r_total'],['微观 K 监督误差（未加权）','C 总目标','R 闭环总目标']):
            ax.plot([r['step'] for r in h],[r[key] for r in h],label=name)
            ax.set(xlabel='联合更新次数',ylabel='目标值',title=title);ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0].set_yscale('log');axes[2].set_yscale('log')
    fig.savefig(output/'loss_comparison.png',dpi=170);plt.close(fig)
    result=dict(runs=records,paired_initialization=runs[1]['manifest']['paired_initialization'],
                paired_final=runs[1]['summary']['paired_final'],
                limitations=['单个训练种子，固定550步；架构及Q初始参数变化，未进行超参数搜索。',
                             'NRMSE同时保存各自及共同CNN训练目标标准差口径。',
                             '不动点为数值候选；K反插值不是已认证临界点。'])
    (output/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--cnn',type=Path,required=True);p.add_argument('--mlp',type=Path,required=True)
    p.add_argument('--iid-reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(render(a.cnn,a.mlp,a.iid_reference,a.output),ensure_ascii=False,indent=2))
