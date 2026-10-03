"""Loss and freeze audit for QC followed by R, compared with joint training."""
import argparse
import json
from pathlib import Path
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--qc',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--joint',type=Path,required=True)
    args=p.parse_args();out=args.run/'staged_report';out.mkdir(exist_ok=True)
    hs=[json.loads((d/'history_progress.json').read_text()) for d in [args.qc,args.run,args.joint]]
    a=torch.load(args.qc/'qcr_model.pt',weights_only=True)
    b=torch.load(args.run/'qcr_model.pt',weights_only=True)
    frozen={name:all(torch.equal(a[k],b[k]) for k in a if k.startswith(prefix))
            for name,prefix in [('Q','theory_encoder.'),('C','rg_kernel.')]}
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,h,title in zip(axes,hs[:2],['Stage 1: QC only (no R losses)','Stage 2: R only (frozen QC)']):
        for split,color in [('train','#2166ac'),('test','#b2182b')]:
            ax.plot([r['epoch'] for r in h],[r[split+'_total'] for r in h],label=split,color=color)
        ax.set(xlabel='Updates within stage',ylabel='Stage objective',title=title,yscale='log');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Different objectives: the two stage totals must not be compared directly')
    fig.savefig(out/'stage_losses.png',dpi=180);plt.close(fig)
    def common(row):
        weights=dict(anchor=2,coarse_consistency=5,kernel_noncollapse=2,separation=10,commutator=5,rollout=5)
        return {split:{'weighted_components':{k:w*row[f'{split}_{k}'] for k,w in weights.items()},
                       'total':sum(w*row[f'{split}_{k}'] for k,w in weights.items())}
                for split in ['train','test']}
    reports={name:json.loads((root/'full_report/metrics.json').read_text()) for name,root in [('staged',args.run),('joint',args.joint)]}
    manifests=[json.loads((root/'manifest.json').read_text()) for root in [args.qc,args.run,args.joint]]
    report=dict(qc_run=str(args.qc),r_run=str(args.run),joint_run=str(args.joint),QC_bitwise_unchanged_in_stage2=frozen,
                data_sha256_equal=len({m['data_sha256'] for m in manifests})==1,
                stage1_final=hs[0][-1],stage2_final=hs[1][-1],
                common_objective=dict(staged=common(hs[1][-1]),joint=common(hs[2][-1])),diagnostics=reports,
                configurations={name:m['config'] for name,m in zip(['QC','R','joint'],manifests)},
                caveat='For the 6000+6000 staged protocol versus 12000 joint updates, each module has a different update budget. Check configurations and data_sha256_equal: lattice size, block factor or data may also differ. Same formula does not mean same coarse targets when C differs. No claim that differences are solely caused by R feedback.')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(frozen=frozen,common_objective=report['common_objective']),indent=2))


if __name__=='__main__':main()
