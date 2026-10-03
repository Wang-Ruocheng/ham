"""Compare completed closure ablations, including non-extrapolated inverse labels."""
import argparse
import json
from pathlib import Path
import numpy as np


def inverse_labels(g, labels, target, alternate=False):
    ks,bs=np.unique(labels[:,0]),np.unique(labels[:,1])
    means=np.array([[g[np.all(labels==[k,b],axis=1)].mean(0) for b in bs] for k in ks])
    hits=[]
    for i in range(len(ks)-1):
        for j in range(len(bs)-1):
            corners=[(i,j),(i+1,j),(i+1,j+1),(i,j+1)]
            for indices in ([(0,1,3),(1,2,3)] if alternate else [(0,1,2),(0,2,3)]):
                ids=[corners[t] for t in indices]
                vertices=np.array([means[a,b] for a,b in ids])
                physical=np.array([[ks[a],bs[b]] for a,b in ids])
                matrix=np.vstack([vertices.T,np.ones(3)])
                if abs(np.linalg.det(matrix))<1e-14:
                    continue
                weights=np.linalg.solve(matrix,np.r_[target,1])
                if weights.min()>=-1e-9:
                    hits.append(dict(KB=(weights@physical).tolist(),vertices=physical.tolist()))
    return hits


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--runs',nargs='+',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=[]
    for run in args.runs:
        folder=run/'full_report'
        metrics=json.loads((folder/'metrics.json').read_text())
        roots=json.loads((folder/'fixed_points.json').read_text())['roots']
        z=np.load(folder/'encodings.npz')
        g=np.concatenate([z['train'],z['test']])[:,0]
        labels=np.concatenate([z['train_labels'],z['test_labels']])
        for root in roots:
            root['inverse_KB']=inverse_labels(g,labels,root['g'])
            root['inverse_KB_alternate']=inverse_labels(g,labels,root['g'],True)
        result.append(dict(run=str(run),representation=metrics['representation'],
            closure=metrics['closure'],roots=roots,
            final_loss=json.loads((run/'history_progress.json').read_text())[-1]))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    for r in result:
        print(r['run'], 'weak fraction',r['representation'][0]['weak_fraction'],
              'NRMSE',r['closure']['rollout']['nrmse'])
        for root in r['roots']:
            print(root['stability'],root['g'],root['eigenvalues'],'inverse',root['inverse_KB'])


if __name__=='__main__':
    main()
