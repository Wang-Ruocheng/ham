"""Read-only physical-chain diagnostics for a fixed, nonrectangular r/u design.

No model loading or optimization. C draws are averaged before chain bootstrap.
Parameter finite differences are evaluation only; no rank-driven resampling.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.run_dual_ising_current import write_json, sha256


def responses(grouped, coordinates, train, seed=91606, bootstrap=500):
    # [cells, physical chains, scales, features]; fixed TRAIN standardization.
    coordinates=np.asarray(coordinates);means=grouped.mean(1)
    scale=np.maximum(means[train].std(0),1e-3)
    z=grouped/scale[None,None]
    axes=[np.unique(coordinates[:,j]) for j in range(2)]
    ranges=np.array([np.ptp(x) for x in axes])
    if np.any(ranges<=0):raise ValueError('Two varying design axes required')
    lookup={tuple(x):i for i,x in enumerate(coordinates)};rng=np.random.default_rng(seed)
    rows=[]
    for i in range(1,len(axes[0])-1):
        for j in range(1,len(axes[1])-1):
            at=[(axes[0][i-1],axes[1][j]),(axes[0][i+1],axes[1][j]),
                (axes[0][i],axes[1][j-1]),(axes[0][i],axes[1][j+1])]
            ids=[lookup[x] for x in at]
            span=np.array([axes[0][i+1]-axes[0][i-1],axes[1][j+1]-axes[1][j-1]])/ranges
            draw=rng.integers(0,grouped.shape[1],(bootstrap,4,grouped.shape[1]))
            for k in range(grouped.shape[2]):
                mm=z[ids,:,k].mean(1)
                mat=np.stack([(mm[1]-mm[0])/span[0],(mm[3]-mm[2])/span[1]],axis=-1)
                singular=np.linalg.svd(mat,compute_uv=False)
                b=np.stack([z[idx,:,k][draw[:,a]].mean(1) for a,idx in enumerate(ids)],axis=1)
                bm=np.stack([(b[:,1]-b[:,0])/span[0],(b[:,3]-b[:,2])/span[1]],axis=-1)
                delta=bm-mat
                noise=np.sqrt(np.maximum(np.linalg.eigvalsh(np.einsum('bdi,bdj->bij',delta,delta))[:,-1],0))
                n95=float(np.quantile(noise,.95))
                rows.append(dict(coordinates=[float(axes[0][i]),float(axes[1][j])],scale=k,
                    sigma=singular.tolist(),singular_ratio=float(singular[1]/max(singular[0],1e-30)),
                    noise95=n95,weak_over_noise95=float(singular[1]/max(n95,1e-30)),
                    all_neighbors_TRAIN=bool(np.all(train[ids]))))
    summaries=[]
    for k in range(grouped.shape[2]):
        rr=[x for x in rows if x['scale']==k]
        summaries.append(dict(scale=k,count=len(rr),median_singular_ratio=float(np.median([x['singular_ratio'] for x in rr])),
            weak_above_noise95=sum(x['weak_over_noise95']>1 for x in rr),
            weak_above_twice_noise95=sum(x['weak_over_noise95']>2 for x in rr)))
    return dict(summary=summaries,rows=rows,bootstrap=bootstrap,standardization=scale.tolist(),
        parameter_units='full span of each design axis; unequal secant spacing divided explicitly',
        chain_units='independent physical chains; C draws averaged first',
        qualification='Descriptive finite-chain response, not a certified local rank or equilibrium test; fixed TRAIN feature scales; no training/size/normalization uncertainty.')


def physical(data):
    manifest=json.loads((data/'manifest.json').read_text())
    design=json.loads((data/'sampling_design.json').read_text())
    groups=[];binders=[];train=[];labels=[];pins={}
    for shard in manifest['shards']:
        p=data/shard['path']; side=data/shard['sidecar']
        if sha256(p)!=shard['sha256'] or sha256(side)!=shard['sidecar_sha256']:
            raise ValueError('New data fingerprint mismatch')
        metadata=json.loads(side.read_text());pins[str(side.relative_to(data))]=sha256(side)
        with np.load(p,allow_pickle=False) as a:x=a['configurations'][...,0]
        phi2=np.mean(x*x,axis=(1,2,3));phi4=np.mean(x**4,axis=(1,2,3))
        nn=np.mean(x*(np.roll(x,1,2)+np.roll(x,1,3))*.5,axis=(1,2,3))
        groups.append(np.stack([phi2,phi4,nn],-1)[:,None,:])
        binders.append(metadata['summary']['binder_1_minus_m4_over_3m2sq'])
        train.append(shard['split']=='train');labels.append([metadata['r'],metadata['u']])
    coordinates=[row['design_coordinates'] for row in design['cells']]
    if labels != [row['parameters'] for row in design['cells']]:raise ValueError('Design/data ordering mismatch')
    out=data.parent/'physical_diagnostics';out.mkdir(exist_ok=False)
    diagnostic=responses(np.array(groups),coordinates,np.array(train))
    diagnostic.update(status='complete',features=['mean_phi2','mean_phi4','mean_nearest_neighbor_product'],
        manifest_sha256=sha256(data/'manifest.json'),source_sidecar_sha256=pins,labels=labels,
        binder=binders,labels_used_for_training=False,gate='numerical failures only; weak response is reported without tuning')
    write_json(out/'report.json',diagnostic)
    np.savez_compressed(out/'chain_statistics.npz',features=np.array(groups),parameters=np.array(labels),
                        design_coordinates=np.array(coordinates),train=np.array(train))
    from scripts.report_phi4_hopping_qcr import _setup_plot
    plt=_setup_plot(None);fig,ax=plt.subplots(figsize=(7,5),layout='constrained')
    labels=np.array(labels);sc=ax.scatter(*labels.T,c=binders,cmap='viridis',s=55,vmin=0,vmax=2/3)
    ax.plot([x['r_center'] for x in design['centers']],[x['u'] for x in design['centers']],ls='--',color='gray',label='旧 TRAIN 数据估计的过渡带中心')
    ax.set(xlabel='$r$',ylabel='$u$',title='新带状采样：微观 Binder 诊断（单尺寸，不认证临界线）');ax.legend(fontsize=8)
    fig.colorbar(sc,ax=ax,label='Binder 累积量');fig.savefig(out/'sampling_region.png',dpi=170);plt.close(fig)


def grouped_run(root, design=None):
    a=dict(np.load(root/'evaluation_arrays.npz',allow_pickle=False))
    f=dict(np.load(root/'reconstruction_arrays.npz',allow_pickle=False))
    cell_ids=np.unique(a['cell_ids']);indices=[np.flatnonzero(a['cell_ids']==c) for c in cell_ids]
    labels=np.array([a['labels'][i[0]] for i in indices])
    train=np.array([a['splits'][i[0]]=='train' for i in indices])
    g=np.stack([a['direct_draws'][:,i].mean(0) for i in indices])
    descriptors=np.stack([f['targets'][:,i].mean(0) for i in indices])
    if design:
        coordinates=np.array([design['cells'][int(c)]['design_coordinates'] for c in cell_ids])
    else:coordinates=labels
    return dict(Q=responses(g,coordinates,train),descriptor=responses(descriptors,coordinates,train),
        coordinate_policy='(r-r0(u),u)' if design else '(r,u)',
        evaluation_arrays_sha256=sha256(root/'evaluation_arrays.npz'),
        reconstruction_arrays_sha256=sha256(root/'reconstruction_arrays.npz'))


def compare(data, run, reference):
    out=run/'band_diagnostics';out.mkdir(exist_ok=False)
    design=json.loads((data/'sampling_design.json').read_text())
    rows=dict(band=grouped_run(run,design),wide=grouped_run(reference))
    for name,p in [('band',run),('wide',reference)]:
        rows[name]['training_summary']=json.loads((p/'summary.json').read_text())
        rows[name]['roots']=json.loads((p/'roots.json').read_text())
    write_json(out/'comparison.json',dict(status='complete',results=rows,
        interpretation='One seed, changed distribution. Latent axes and TRAIN normalizers differ across independently trained runs; absolute latent RMSE/eigenvalues are not coordinate-aligned comparisons. PCA is not local rank; no physical root assignment.',
        C_draws_averaged_before_independent_chain_bootstrap=True,thermal_nu=None))
    from scripts.report_phi4_hopping_qcr import _setup_plot
    plt=_setup_plot(None);fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,key,title in zip(axes,['descriptor','Q'],['有限局域描述','Q 编码']):
        for name,offset,label in [('wide',-.16,'原宽窗口'),('band',.16,'新带状窗口')]:
            rr=rows[name][key]['summary'];ax.bar(np.arange(3)+offset,[r['weak_above_twice_noise95']/r['count'] for r in rr],width=.30,label=label)
        ax.set(xticks=[0,1,2],xlabel='粗粒化层数 k',ylabel='第二响应强度 > 2×噪声的中心比例',ylim=(0,1),title=title);ax.legend()
    fig.suptitle('独立物理链 bootstrap 描述性比较；不认证秩或平衡')
    fig.savefig(out/'response_comparison.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--run',type=Path);p.add_argument('--reference',type=Path);a=p.parse_args()
    if a.run:compare(a.data,a.run,a.reference)
    else:physical(a.data)
