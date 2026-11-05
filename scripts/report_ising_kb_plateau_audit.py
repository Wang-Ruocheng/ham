"""Array-only presentation correction: backward root matching and readable labels.

Never opens models, computes new roots, or modifies downloaded formal artifacts.
"""
import argparse
import hashlib
import json
from pathlib import Path
import collections
import numpy as np
from experiments.ising_kb_r_plateau_continue import root_stability
from scripts.report_phi4_hopping_qcr import _setup_plot
from scripts.plot_saved_kb_parameter_flow import cell_means
from scripts.report_ising_kb_r_plateau_continue import plot_stability


def render(root,out):
    from matplotlib.colors import Normalize,TwoSlopeNorm
    root=Path(root);out=Path(out);out.mkdir(exist_ok=False)
    summary=json.loads((root/'summary.json').read_text())
    old=json.loads((root/'root_stability.json').read_text())
    with np.load(root/'evaluation_arrays.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    st=root_stability(old['snapshots'],a['flow_bounds'],summary['source_roots'])
    assert st['snapshots']==old['snapshots'] and st['final_root_tracks']==old['final_root_tracks']
    (out/'root_stability_corrected.json').write_text(json.dumps(st,ensure_ascii=False,indent=2)+'\n')
    plt=_setup_plot(None);plot_stability(st,out,plt)
    roots=summary['roots'];means,labels,_=cell_means(a);grid=a['flow_g'];drift=a['flow_R']-grid
    np.testing.assert_allclose(drift,a['flow_displacement'],rtol=1e-12,atol=1e-12)
    spacing=min(np.min(np.diff(np.unique(grid[:,d]))) for d in range(2))
    arrows=drift/np.maximum(np.linalg.norm(drift,axis=1,keepdims=True),1e-15)*.68*spacing
    norms=[Normalize(labels[:,0].min(),labels[:,0].max()),TwoSlopeNorm(vmin=labels[:,1].min(),vcenter=0,vmax=labels[:,1].max())]
    offsets=[(-14,13),(12,-19),(9,12),(-18,15),(-22,-20),(12,13),(-24,8),(11,-14),(-24,-16)]
    for k in range(3):
        fig,axes=plt.subplots(1,2,figsize=(13,5.7),layout='constrained')
        for d,ax in enumerate(axes):
            ax.quiver(*grid.T,*arrows.T,angles='xy',scale_units='xy',scale=1,color='#7c8792',alpha=.66,width=.0023,zorder=1)
            sc=ax.scatter(*means[:,k].T,c=labels[:,d],s=19,cmap='viridis' if d==0 else 'coolwarm',norm=norms[d],edgecolors='#30343b',linewidths=.25,zorder=3)
            fig.colorbar(sc,ax=ax,shrink=.75,pad=.02,label='原始最近邻耦合 K' if d==0 else '原始外场 B')
            for i,r in enumerate(roots):
                color='black' if r['within_train_hull'] else '#b65a00'
                ax.scatter(*r['g'],marker='*',s=150,facecolors='white',edgecolors=color,zorder=5)
                ax.annotate(str(i+1),r['g'],xytext=offsets[i] if len(roots)==9 else (7,8),textcoords='offset points',fontsize=10,color=color,
                    bbox=dict(facecolor='white',edgecolor='none',alpha=.9),arrowprops=dict(arrowstyle='-',color=color,lw=.6),zorder=6)
            ax.set(xlabel='$g_1$',ylabel='$g_2$',aspect='equal',xlim=a['flow_bounds'][0],ylim=a['flow_bounds'][1],title='按真实 K 着色' if d==0 else '按真实 B 着色');ax.grid(alpha=.12)
        fig.suptitle(f'R 训练至损失平台：累计 112000 次 R-only 更新；k={k}\n灰色等长箭头：R(g)−g；星号 3 为最终中央排斥点',fontsize=13)
        fig.supxlabel('散点为同参数两条物理链的平均编码；包含训练与留出。橙色星号为训练输入凸包外的数值根。\n数据覆盖外的流场为外推；粗尺度颜色仍表示原始参数，数值根未获物理认证。',fontsize=9)
        fig.savefig(out/f'R_flow_by_KB_k{k}.png',dpi=180);plt.close(fig)
    central=[r['point'] for r in st['central_track']]
    result=dict(source_formal_directory=str(root.resolve()),presentation_correction='Replace the greedy source-to-late central label by the already-computed final-root backward assignments; all root positions, spectra, losses and checkpoints unchanged.',
        source_file_sha256={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ['root_stability.json','summary.json','evaluation_arrays.npz','history.json']},
        new_R_updates=summary['new_R_updates'],cumulative_R_only_updates=summary['actual_extra_R_updates'],
        train_relative_change=summary['train_final']/summary['train_initial']-1,
        test_relative_change=summary['after_closure']['test']['weighted_all_start_objective']/summary['before_closure']['test']['weighted_all_start_objective']-1,
        central_final=central[-1],central_all_late_repelling=all(r is not None and r['classification']=='repelling' for r in central),
        central_coordinate_min=np.min([r['g'] for r in central if r],axis=0).tolist(),central_coordinate_max=np.max([r['g'] for r in central if r],axis=0).tolist(),
        root_counts=[dict(step=s['extra_R_updates'],count=len(s['roots']),classes=dict(collections.Counter(r['classification'] for r in s['roots']))) for s in st['snapshots']],
        final_outside_train_hull=sum(not r['within_train_hull'] for r in roots),thermal_nu=None)
    (out/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (out/'receipt.json').write_text(json.dumps(dict(array_only=True,model_loaded=False,new_root_search=False,formal_artifacts_modified=False,
        files=[dict(path=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(out.iterdir()) if p.is_file()]),indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    print(json.dumps(render(a.source,a.output),ensure_ascii=False))
