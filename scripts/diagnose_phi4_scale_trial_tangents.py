"""Offline weak-response audit of a completed scale-weight trial.

Only public NPZ/JSON are read. Central differences span one r/u grid increment;
the output is descriptive, not a derivative, rank or fixed-point certificate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.diagnose_phi4_ru_local_window_saved import local_differences


def run(root,output):
    root,output=Path(root).resolve(),Path(output).resolve()
    if output.exists():raise ValueError('Fresh audit output required')
    job=json.loads((root/'job.json').read_text());assert job['workflow_complete'] and job['returncode']==0
    sha=hashlib.sha256((root/'evaluation_arrays.npz').read_bytes()).hexdigest()
    assert sha==job['public_files']['evaluation_arrays.npz']
    with np.load(root/'evaluation_arrays.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
    arms=('source','A_original','B_scale_normalized')
    groups=[np.flatnonzero(a['cell_ids']==c) for c in np.unique(a['cell_ids'])]
    labels=np.array([a['labels'][ix[0],:2] for ix in groups])
    lookup={tuple(np.round(x,8)):i for i,x in enumerate(labels)}
    grouped={key:np.array([a[key][ix] for ix in groups]) for key in ['target']+[arm+tail for arm in arms for tail in ('_g','_decoded')]}
    means={k:v.mean(1) for k,v in grouped.items()}
    even=(grouped['target']+grouped['target'][...,::-1])/2
    noise={ (r['cell'],r['scale']):r for r in local_differences(even,labels,bootstrap=500)}
    def jac(x,k,ix):return np.stack([(x[ix[1],k]-x[ix[0],k])/2,(x[ix[3],k]-x[ix[2],k])/2],axis=-1)
    neighbors={}
    for c,(r,u) in enumerate(labels):
        keys=[(r-.15,u),(r+.15,u),(r,u-.25),(r,u+.25)]
        if all(tuple(np.round(x,8)) in lookup for x in keys):neighbors[c]=[lookup[tuple(np.round(x,8))] for x in keys]
    rows=[]
    for arm in arms:
        for k in range(3):
            for c,ix in neighbors.items():
                j=jac(means[arm+'_g'],k,ix);s=np.linalg.svd(j,compute_uv=False)
                t=jac(means['target'],k,ix);d=jac(means[arm+'_decoded'],k,ix)
                row=dict(arm=arm,scale=k,cell=c,sigma_ratio=float(s[1]/s[0]),
                         even_weak_over_noise95=noise[c,k]['weak_sigma_over_noise95'])
                for sym in (False,True):
                    target=(t+t[::-1])/2 if sym else t;decoded=(d+d[::-1])/2 if sym else d
                    _,sv,vh=np.linalg.svd(target,full_matrices=False)
                    row['even_weak_error' if sym else 'weak_error']=float(np.linalg.norm((decoded-target)@vh[-1])/sv[-1])
                rows.append(row)
    summary={}
    for arm in arms:
        summary[arm]=[]
        for k in range(3):
            x=[r for r in rows if r['arm']==arm and r['scale']==k];selected=[r for r in x if r['even_weak_over_noise95']>2]
            summary[arm].append(dict(scale=k,median_sigma_ratio=float(np.median([r['sigma_ratio'] for r in x])),
                median_weak_error=float(np.median([r['weak_error'] for r in x])),
                median_even_weak_error=float(np.median([r['even_weak_error'] for r in x])),
                exploratory_above_twice_noise_count=len(selected),
                median_even_weak_error_above_twice_noise=float(np.median([r['even_weak_error'] for r in selected])) if selected else None))
    result=dict(input_sha256=sha,summary=summary,rows=rows,training_or_inference=False,
                caveats=['Finite-grid differences and four-chain bootstrap, not reliable physical-rank certification.',
                         'Global PCA components can arise from curvature; they are not independent local coupling directions.',
                         'Even projection is diagnostic only; no changes to training, C, Q, or stored data.',
                         'At coarse scales C is not architecturally sign-equivariant; even projection alone does not certify symmetry.'])
    output.mkdir(parents=True);(output/'tangent_audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.root,args.output)
