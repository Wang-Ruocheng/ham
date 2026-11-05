"""Array-only paired report for the frozen-Q/C R diagnostic."""
import json
from pathlib import Path
import numpy as np
from scripts.report_phi4_hopping_qcr import _setup_plot
from scripts.plot_saved_kb_parameter_flow import cell_means
from scripts.report_ising_kb_free_decoder_matched import decoder_diagnostics


def plot_flow(a,roots,out,plt,title):
    from matplotlib.colors import Normalize,TwoSlopeNorm
    means,labels,counts=cell_means(a);grid=a['flow_g'];drift=a['flow_R']-grid
    np.testing.assert_allclose(drift,a['flow_displacement'],rtol=1e-12,atol=1e-12)
    spacing=min(np.min(np.diff(np.unique(grid[:,d]))) for d in range(2))
    arrows=drift/np.maximum(np.linalg.norm(drift,axis=1,keepdims=True),1e-15)*.68*spacing
    norms=[Normalize(labels[:,0].min(),labels[:,0].max()),TwoSlopeNorm(vmin=labels[:,1].min(),vcenter=0,vmax=labels[:,1].max())]
    names=dict(attracting='吸引点',repelling='排斥点',saddle='鞍点',neutral_or_unresolved='近中性或未定')
    for k in range(3):
        fig,axes=plt.subplots(1,2,figsize=(12.8,7.6),layout='constrained')
        for d,ax in enumerate(axes):
            ax.quiver(*grid.T,*arrows.T,angles='xy',scale_units='xy',scale=1,color='#7c8792',alpha=.66,width=.0023,zorder=1)
            sc=ax.scatter(*means[:,k].T,c=labels[:,d],s=23,cmap='viridis' if d==0 else 'coolwarm',norm=norms[d],edgecolors='#30343b',linewidths=.28,zorder=3)
            fig.colorbar(sc,ax=ax,shrink=.82,label='原始最近邻耦合 K' if d==0 else '原始外场 B')
            for i,r in enumerate(roots,1):
                ax.scatter(*r['g'],marker='*',s=170,facecolors='white',edgecolors='black',zorder=5)
                ax.annotate(f"{names[r['classification']]}{i}"+('（外推）' if not r['within_train_hull'] else ''),r['g'],
                    xytext=(5,8),textcoords='offset points',fontsize=8,bbox=dict(facecolor='white',edgecolor='none',alpha=.85))
            ax.set(xlabel='$g_1$',ylabel='$g_2$',aspect='equal',xlim=a['flow_bounds'][0],ylim=a['flow_bounds'][1],title='按真实 K 着色' if d==0 else '按真实 B 着色')
            ax.grid(alpha=.12)
        fig.suptitle(f'{title}；k={k}\n箭头为 R(g)−g 的方向（等长）')
        fig.supxlabel('散点为同参数两条物理链的平均编码；训练与留出均展示。覆盖外箭头为外推。\n星号为数值不动点，未认证为物理临界点；粗尺度颜色仍表示原始参数。',fontsize=9)
        fig.savefig(out/f'R_flow_by_KB_k{k}.png',dpi=190);plt.close(fig)


def render(root):
    root=Path(root);m,s,h,r=[json.loads((root/(n+'.json')).read_text()) for n in ('manifest','summary','history','roots')]
    if m['labels_used_in_training'] or m['test_used_in_training'] or s['additional_Q_updates'] or s['additional_C_updates'] or s['frozen_before_sha256']!=s['frozen_after_sha256']:
        raise ValueError('R-only invariant violated')
    with np.load(root/'evaluation_arrays.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
    old={k:v.copy() for k,v in a.items()}
    for k in ('one_step_g','rollout_g','all_start_g','flow_g','flow_R','flow_displacement'):old[k]=a['source_'+k]
    plt=_setup_plot(None);out=root/'report';out.mkdir(exist_ok=False)
    before=out/'before';before.mkdir()
    plot_flow(a,r['roots'],out,plt,f"冻结局部 MLP Q/C 后，R 额外更新 {s['actual_extra_R_updates']} 次")
    plot_flow(old,s['source_roots'],before,plt,'局部 MLP Q 原联合训练终点：1250 次更新')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for key,label in [('train','固定训练轨迹'),('best','训练历史最小值')]:
        vals=[v['train']['total'] if key=='train' else v['best_train_closure'] for v in h]
        axes[0].plot([v['extra_R_updates'] for v in h],vals,label=label)
    axes[0].set(xlabel='R 额外更新次数',ylabel='加权闭环损失',yscale='log',title='仅训练集参与停止判断');axes[0].legend();axes[0].grid(alpha=.2)
    x=np.arange(3)
    for off,key,label in [(-.18,'before_closure','R 续训前'),(.18,'after_closure','R 续训后')]:
        vals=s[key]['test']['all_start'];axes[1].bar(x+off,[v['nrmse'] for v in vals],width=.36,label=label)
    axes[1].set(xticks=x,xticklabels=['36→12','36→4','12→4'],ylabel='NRMSE（原训练目标标准差）',title='固定留出轨迹：同坐标配对比较');axes[1].legend()
    fig.savefig(out/'R_loss_and_heldout.png',dpi=180);plt.close(fig)
    means,labels,_=cell_means(a);fig,axes=plt.subplots(2,3,figsize=(13,7.5),constrained_layout=True)
    for d in range(2):
        for k in range(3):
            ax=axes[d,k];sc=ax.scatter(*labels.T,c=means[:,k,d],cmap='coolwarm',s=35,marker='s',vmin=means[...,d].min(),vmax=means[...,d].max())
            ax.set(xlabel='最近邻耦合 K',ylabel='外场 B',title=f'$g_{d+1}$：k={k}，L={36//3**k}');fig.colorbar(sc,ax=ax,label='同参数两链平均编码')
    fig.suptitle('冻结的 Q/C：参数到编码的关系与原实验完全相同')
    fig.savefig(out/'Q_g_KB.png',dpi=160);plt.close(fig)
    result=dict(status='complete',Q_C_unchanged=True,reason=s['reason'],source_joint_updates=1250,
        extra_R_updates=s['actual_extra_R_updates'],before_closure=s['before_closure'],after_closure=s['after_closure'],
        before_roots=s['source_roots'],after_roots=r['roots'],root_search=s['root_search'],before_decoder=decoder_diagnostics(old),after_decoder=decoder_diagnostics(a),
        thermal_nu=None,qualifications=['本轮 R 训练不读标签；冻结来源的 Q 曾接受微观 K/B 监督。',
            '固定一次已保存 C 轨迹的全训练集拟合，未做独立 C 重抽验证。',
            '同坐标误差可配对比较；数值根及稳定性不等于物理临界点认证。',
            '有界网格及原根邻域补充搜索可能遗漏根；不按根或留出结果选模。'])
    (out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return result
