"""Array-only Chinese XY report; no critical-point inference from a scalar root."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from scripts.report_phi4_hopping_qcr import _setup_plot,_plot_loss,_plot_all_start_closure


def render(root):
    root=Path(root);out=root/'report';out.mkdir(exist_ok=False)
    a=dict(np.load(root/'evaluation_arrays.npz',allow_pickle=False));g=a['direct_g'][...,0]
    summary=json.loads((root/'summary.json').read_text());roots=json.loads((root/'roots.json').read_text())
    data=json.loads((root/'data/metadata.json').read_text());history=json.loads((root/'history.json').read_text())
    plt=_setup_plot(None);T=np.unique(a['labels'][:,0]);colors=['#277da8','#f39c37','#20856f']
    fig,axes=plt.subplots(1,2,figsize=(13,4.7),layout='constrained')
    for k in range(3):
        mean=np.array([g[a['labels'][:,0]==t,k].mean() for t in T])
        se=np.array([g[a['labels'][:,0]==t,k].std(ddof=1)/2 for t in T])
        axes[0].errorbar(T,mean,yerr=se,fmt='o-',ms=4,color=colors[k],label=f'k={k}')
    axes[0].axvline(.89294,c='gray',ls=':',label='文献参考，非拟合结果')
    axes[0].set(xlabel='原始温度 T/J',ylabel='g（四条物理链均值及标准误）',title='不同尺度的温度—潜变量关系');axes[0].legend(fontsize=9);axes[0].grid(alpha=.2)
    grid=a['flow_grid'][:,0];nxt=a['flow_next'][:,0]
    axes[1].plot(grid,nxt,label='R(g)',c='#a3609e');axes[1].plot(grid,grid,'k--',label='恒等映射')
    for k in range(2):axes[1].scatter(g[:,k],g[:,k+1],s=18,alpha=.6,c=colors[k],label=f'直接编码 k={k}→{k+1}')
    ix=np.linspace(0,len(grid)-1,25,dtype=int);axes[1].quiver(grid[ix],grid[ix],np.zeros(len(ix)),nxt[ix]-grid[ix],angles='xy',scale_units='xy',scale=1,color='gray',width=.003)
    for r in roots['roots']:
        axes[1].plot(r['g'],r['g'],'r*',ms=10);axes[1].annotate('数值根'+('（外推）' if not r['inside_train_input_envelope'] else ''),(r['g'],r['g']),fontsize=8)
    axes[1].set(xlabel='输入 g',ylabel='输出 g′',title='学习的流与直接编码目标；箭头为 R(g)−g');axes[1].legend(fontsize=9);axes[1].grid(alpha=.2)
    fig.suptitle('二维 XY：单标量探索不等于恢复完整 BKT 双变量流');fig.savefig(out/'Q_and_R.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    rows=data['chains'];temps=np.array([r['temperature'] for r in rows])
    for name,ax,label in [('helicity_modulus_over_J',axes[0],'刚度 Υ/J'),('vortex_density',axes[1],'每格点涡旋密度')]:
        vals=np.array([r['diagnostics'][name] if name=='helicity_modulus_over_J' else r['diagnostics'][name]['mean'] for r in rows])
        mean=[vals[temps==t].mean() for t in T];se=[vals[temps==t].std(ddof=1)/2 for t in T]
        ax.errorbar(T,mean,yerr=se,fmt='o-',ms=4);ax.axvline(.89294,c='gray',ls=':');ax.set(xlabel='T/J',ylabel=label);ax.grid(alpha=.2)
    axes[0].plot(T,2*T/np.pi,'k--',label='2T/(πJ)，热力学极限跃变参考');axes[0].legend(fontsize=8);axes[0].set_title('有限尺寸刚度，不以单尺寸交点估计 Tc')
    axes[1].set_title('总涡旋密度，不等于游离涡旋密度')
    for t in (.50,.78,.89,.95,1.15,1.6):
        corr=np.mean([r['diagnostics']['correlation_mean'] for r in rows if r['temperature']==t],0);dist=np.arange(1,len(corr)+1)
        axes[2].plot(dist[corr>0],corr[corr>0],'.-',label=f'T/J={t:g}')
    axes[2].set(xscale='log',yscale='log',xlabel='间距 r',ylabel='G(r)',title='同一 L=54 的关联函数；不作指数拟合');axes[2].legend(fontsize=8);axes[2].grid(alpha=.2)
    fig.savefig(out/'physical_diagnostics.png',dpi=170);plt.close(fig)
    _plot_loss(history,out,plt,uniform_label='均匀角度保持');_plot_all_start_closure(summary['closure'],out,plt)
    correlations={split:[float(spearmanr(a['labels'][a['splits']==split,0],g[a['splits']==split,k]).statistic) for k in range(3)] for split in ('train','test')}
    drift={split:[float(np.sqrt(np.mean((g[a['splits']==split,k+1]-g[a['splits']==split,k])**2))) for k in range(2)] for split in ('train','test')}
    result=dict(status='complete',temperature_g_spearman=correlations,direct_scale_drift_rmse=drift,roots=roots,
                closure=summary['closure'],sampling_warnings=data['warnings'],information_diagnostic=summary['information_diagnostic'],
                reference_T_over_J=.89294,reference_K_bare=1/.89294,reference_is_training_label=False,
                BKT_certified=False,critical_temperature_estimate=None,thermal_nu=None,
                limitations=['One L and one training seed; scalar latent cannot certify vortex-fugacity/stiffness flow.',
                    'Circular smoothed MI proxy; evaluate paired versus mismatched baseline.',
                    'No spontaneous magnetization at positive T in thermodynamic limit; critical low-T fixed line, not isolated repeller.',
                    'No BKT or equilibration certification from scalar roots/finite-chain observations alone.'])
    (out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    return result
