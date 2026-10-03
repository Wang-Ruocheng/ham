"""Compare the paired MI-only and MI+variance runs, without retraining."""
import json
from pathlib import Path
import torch
from recurrent_rg.checkpoints import load_run
from recurrent_rg.config import RunConfig
from recurrent_rg.runner import prepared_data


def main():
    torch.set_num_threads(2)
    root=Path('results/ising1d_g1_exactMI_variance2_2400_seed91/qr')
    baseline=Path('results/ising1d_g1_exactEmpiricalMISeparate_w2_2400_seed91/qr')
    run=load_run(root,2400);old=load_run(baseline,2400)
    state=run.model.state_dict();other=old.model.state_dict()
    result={'max_parameter_difference':max(float((v-other[k]).abs().max()) for k,v in state.items())}
    cfg=RunConfig.from_dict(run.config);train,test=prepared_data(cfg)
    for split,data in [('train',train),('test',test)]:
        x=data.configurations.double()
        if cfg.data.spin_flip_augment:x=torch.cat([x,-x],1)
        with torch.no_grad(),torch.random.fork_rng(devices=[]):
            torch.manual_seed(12345)
            levels=run.model.coarse_grain(x)
        means=torch.stack([z.mean(tuple(range(1,z.ndim))) for z in levels],1)
        batch_var=1-means.mean(0).square()
        each_var=1-means.square()
        ratio=batch_var[1:]/batch_var[:-1].clamp_min(1e-12)
        result[split]={'aggregate_variance_by_scale':batch_var.tolist(),
            'aggregate_variance_ratio':ratio.tolist(),
            'ensemble_fraction_below_floor':((each_var[:,1:]<.25*each_var[:,:-1]) & (each_var[:,:-1]>1e-12)).double().mean(0).tolist(),
            'note':'Full cached split with paired spin flips; diagnostic differs from training minibatches.'}
    history=json.loads((root/'history_progress.json').read_text())
    result['max_logged_C_loss']={s:max(r[f'{s}_kernel_noncollapse'] for r in history) for s in ['train','test']}
    (root/'report'/'variance_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
