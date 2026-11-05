"""Compare recipe, data, every logged metric, and final Q/C/R tensors."""
import argparse
import json
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    manifests=[json.loads((r/'manifest.json').read_text()) for r in (args.reference,args.run)]
    keys=('config','data_sha256','training_driver','estimator','rsmi_weight',
          'rsmi_scales','buffer','independent_c','data_driven_initialization',
          'gradient_QC','gradient_R','kernel_auxiliaries')
    changed=[k for k in keys if manifests[0].get(k)!=manifests[1].get(k)]
    if manifests[0].get('environment_mode','full_shell')!=manifests[1].get('environment_mode','full_shell'):
        changed.append('environment_mode')
    histories=[json.loads((r/'history_progress.json').read_text()) for r in (args.reference,args.run)]
    states=[torch.load(r/'qcr_model.pt',map_location='cpu',weights_only=True) for r in (args.reference,args.run)]
    same_keys=states[0].keys()==states[1].keys()
    exact=same_keys and all(torch.equal(states[0][k],states[1][k]) for k in states[0])
    max_difference=(max(float((states[0][k]-states[1][k]).abs().max()) for k in states[0])
                    if same_keys else None)
    result=dict(reference=str(args.reference.resolve()),run=str(args.run.resolve()),
                changed_contract_fields=changed,history_exact_match=histories[0]==histories[1],
                parameter_exact_match=exact,max_parameter_absolute_difference=max_difference,
                logged_steps=len(histories[1]),final_step=histories[1][-1]['epoch'])
    result['exact_reproduction']=not changed and result['history_exact_match'] and exact
    target=args.run/'report';target.mkdir(exist_ok=True)
    (target/'reproduction_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if not result['exact_reproduction']:
        raise SystemExit('Not an exact reproduction; inspect the recorded differences.')


if __name__=='__main__':main()
