"""Compare one common QCR configuration on TFIM and 2D Ising; no training."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.compare_closure_refinements import inverse_labels


def project_to_curve(g,labels,target):
    labels=labels.reshape(-1)
    knots=np.unique(labels)
    means=np.array([g[labels==v].mean(0) for v in knots])
    best=None
    for a,b,x,y in zip(means[:-1],means[1:],knots[:-1],knots[1:]):
        delta=b-a
        t=float(np.clip(np.dot(target-a,delta)/max(np.dot(delta,delta),1e-20),0,1))
        residual=float(np.linalg.norm(a+t*delta-target))
        if best is None or residual<best['residual']:
            best=dict(coupling=float(x+t*(y-x)),residual=residual,segment=[float(x),float(y)])
    return best


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tfim',type=Path,required=True)
    parser.add_argument('--ising',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    configs=[json.loads((p/'manifest.json').read_text())['config'] for p in [args.tfim,args.ising]]
    assert configs[0]['loss']==configs[1]['loss'],'Loss configurations differ'
    models=[dict(c['model']) for c in configs]
    for m in models:m.pop('dimension')
    assert models[0]==models[1],'Model settings differ beyond spatial dimension'
    output=dict(shared_loss=configs[0]['loss'],shared_model=models[0],systems={})
    for name,path,cfg in zip(['tfim','ising'],[args.tfim,args.ising],configs):
        folder=path/(f'step{cfg["training"]["epochs"]}' if name=='tfim' else 'full_report')
        metrics=json.loads((folder/('report.json' if name=='tfim' else 'metrics.json')).read_text())
        roots=json.loads((folder/'fixed_points.json').read_text())['roots']
        if name=='tfim':
            z=np.load(folder/'trajectories.npz')
            g=np.concatenate([z['train_direct'],z['direct']])[:,0]
            labels=np.concatenate([z['train_couplings'],z['couplings']])
            for root in roots:root['microscopic_projection']=project_to_curve(g,labels,np.array(root['g']))
            weak=metrics['weak_fraction'][0]
        else:
            z=np.load(folder/'encodings.npz')
            g=np.concatenate([z['train'],z['test']])[:,0]
            labels=np.concatenate([z['train_labels'],z['test_labels']])
            for root in roots:
                root['inverse_KB']=inverse_labels(g,labels,root['g'])
                root['inverse_KB_alternate']=inverse_labels(g,labels,root['g'],True)
            weak=metrics['representation'][0]['weak_fraction']
        history=json.loads((path/'history_progress.json').read_text())
        output['systems'][name]=dict(run=str(path),weak_fraction=weak,closure=metrics['closure'],
            kernel=metrics['kernel'],roots=roots,final_train=history[-1]['train_total'],final_test=history[-1]['test_total'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2)+'\n')
    for name,r in output['systems'].items():
        print(name,'weak fraction',r['weak_fraction'],'rollout',r['closure']['rollout']['nrmse'])
        for root in r['roots']:print(root)


if __name__=='__main__':main()
