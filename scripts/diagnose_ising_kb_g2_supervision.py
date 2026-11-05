"""Read public frozen-Q arrays and summarize the microscopic B-supervision error."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scripts.report_phi4_hopping_qcr import _setup_plot


def render(source,manifest,output):
    source=Path(source);manifest=Path(manifest);out=Path(output);out.mkdir(exist_ok=False)
    with np.load(source/'evaluation_arrays.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    m=json.loads(manifest.read_text());scale=m['label_scaler']
    target=(a['labels']-scale['mean'])/scale['std'];pred=a['direct_full_g'][:,0]
    error=(pred-target)**2;K=a['labels'][:,0];B=a['labels'][:,1]
    def metrics(mask):
        return dict(chains=int(mask.sum()),mse_per_coordinate=error[mask].mean(0).tolist(),
            supervision_weighted_per_coordinate=(10*error[mask].mean(0)).tolist(),
            physical_B_rmse=float(np.sqrt(error[mask,1].mean())*scale['std'][1]),
            g2_standard_deviation=float(pred[mask,1].std()),target_B_standard_deviation=float(target[mask,1].std()))
    rows=[]
    for value in np.unique(K):
        v=K==value
        rows.append(dict(K=float(value),g1_mean=float(pred[v,0].mean()),
            g2_to_standardized_B_slope=float(np.cov(pred[v,1],target[v,1],ddof=0)[0,1]/target[v,1].var()),
            **{split:metrics(v&(a['splits']==split)) for split in ['train','test']}))
    bands=[]
    for name,mask in [('K=0.20–0.35',K<.36),('K=0.38–0.50',(K>=.36)&(K<.52)),('K=0.53–0.68',K>=.52),('g1<-1',pred[:,0]<-1)]:
        bands.append(dict(region=name,**{split:metrics(mask&(a['splits']==split)) for split in ['train','test']}))
    result=dict(scale=0,prediction='Q on full saved microscopic ensemble; same supervision convention as training',
        error_order='coordinate squared error for each physical chain, then mean; no averaging chains before error',
        supervision_weight=10,label_scaler=scale,per_K=rows,bands=bands,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source/'evaluation_arrays.npz',manifest]},
        qualification='Post-training diagnostics, not an update-by-update loss trace. No training, model loading or physical-data sampling.')
    (out/'diagnostics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    plt=_setup_plot(None);fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for split,label,style in [('train','训练集','o-'),('test','留出集','s--')]:
        axes[0].plot([r['K'] for r in rows],[r[split]['mse_per_coordinate'][1] for r in rows],style,label=label,markersize=4)
    axes[0].set(xlabel='原始最近邻耦合 K',ylabel=r'$g_2$ 监督均方误差（未乘权重 10）',title='低 K 区域的外场监督误差更大');axes[0].legend();axes[0].grid(alpha=.2)
    for value in [.2,.35,.44,.68]:
        mask=np.isclose(K,value,rtol=0,atol=1e-12);bs=np.unique(B[mask]);xs=[];ys=[]
        for b in bs:
            v=mask&(B==b);assert v.sum()==2
            xs.append(float(target[v,1].mean()));ys.append(float(pred[v,1].mean()))
        axes[1].plot(xs,ys,'o-',markersize=3,label=f'K={value:.2f}')
    lo,hi=float(target[:,1].min()),float(target[:,1].max())
    axes[1].plot([lo,hi],[lo,hi],'k--',label='理想监督目标')
    axes[1].set(xlabel=r'标准化真实外场 $\widetilde B$',ylabel=r'微观编码 $g_2$',title='低 K 时，外场编码幅度受到压缩');axes[1].legend(fontsize=9);axes[1].grid(alpha=.2)
    fig.suptitle('冻结的局部 MLP Q：微观尺度 k=0 的监督诊断',fontsize=14)
    fig.supxlabel('左：逐物理链误差后平均；右：每个 (K,B) 的两条物理链平均编码。\n只读取已保存的公开数组；本次 R-only 续训未改变 Q。',fontsize=9)
    fig.savefig(out/'g2_supervision_by_K.png',dpi=180);plt.close(fig)
    return bands


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    print(json.dumps(render(a.source,a.manifest,a.output),ensure_ascii=False))
