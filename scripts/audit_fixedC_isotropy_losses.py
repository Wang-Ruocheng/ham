"""Read-only regional closure audit for the fixed-C covariance20 experiment."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recurrent_rg.checkpoints import load_run
from experiments.ising2d_q_only import statistics,encode
from experiments.ising2d_qr_loss_ablation import components


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(2);report=json.loads((args.run/'report.json').read_text())
    model=load_run(args.run/'iso_both'/'qcr_model.pt').model
    tx,ty,vx,vy=torch.load(report['cache'],weights_only=True)
    tr=statistics(model,tx);te=statistics(model,vx)
    output={}
    with torch.no_grad():
        gtrain=.5*(encode(model,tr[0][:,:2].mean(1))+encode(model,tr[0][:,2:].mean(1)))
        g=.5*(encode(model,te[0][:,:2].mean(1))+encode(model,te[0][:,2:].mean(1)))
        split_a=encode(model,te[0][:,:2].mean(1));split_b=encode(model,te[0][:,2:].mean(1))
        split=(split_a-split_b).square().sum(-1).mean(0)
        output['test_split_weighted']=dict(micro=float(20*split[0]),coarse=float(5*split[1:].mean()))
        one=model.advance_theory(g[:,:-1]);r1=model.advance_theory(g[:,0]);roll=torch.stack((r1,model.advance_theory(r1)),1)
        sd=gtrain[:,1:].var(0,unbiased=False).mean(-1).sqrt()
        kc=.5*np.log(1+np.sqrt(2))
        masks={'all':torch.ones(len(vy),dtype=torch.bool),'B0':vy[:,1]==0,
               'near_critical':((vy[:,0]-kc).abs()<=.06)&(vy[:,1].abs()<=.005)}
        output['regions']={}
        for name,mask in masks.items():
            target=g[mask,1:];source=g[mask,:-1]
            row={'ensembles':int(mask.sum()),'normalization':'all-training target per-coordinate SD, per scale'}
            for kind,pred,base in [('step',one[mask],source),('rollout',roll[mask],g[mask,0:1].expand_as(target))]:
                err=pred-target;raw=err.square().sum(-1).mean(0)
                identity=(target-base).square().sum(-1).mean(0)
                row[kind]=dict(weighted_loss=float(5*raw.mean()),weighted_loss_by_k=(5*raw).tolist(),nrmse=(raw.sqrt()/2**.5/sd).tolist(),error_over_actual_displacement=(raw/identity).sqrt().tolist())
            output['regions'][name]=row
        # Full-ensemble rather than midpoint encoding for physical input reference.
        z=np.load(args.run/'iso_both'/'encodings.npz');labels=np.concatenate((z['train_labels'],z['test_labels']));cloud=np.concatenate((z['train'],z['test']))[:,0]
        ks=np.unique(labels[:,0]);curve=np.array([cloud[(labels[:,0]==k)&(labels[:,1]==0)].mean(0) for k in ks]);gc=np.array([np.interp(kc,ks,curve[:,i]) for i in range(2)])
        delta=model.advance_theory(torch.from_numpy(gc)).numpy()-gc
        output['critical_input_reference']=dict(K=kc,B=0,g=gc.tolist(),R_minus_g=delta.tolist(),displacement_norm=float(np.linalg.norm(delta)),note='Microscopic interpolated critical input, not an assumed RG fixed point.')
    total=report['arms']['iso_both']['history'][-1]['test']['loss']
    output['test_contribution_percent']={k:100*v/total for k,v in report['arms']['iso_both']['history'][-1]['test']['components'].items()}
    (args.run/'loss_regional_audit.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))


if __name__=='__main__':main()
