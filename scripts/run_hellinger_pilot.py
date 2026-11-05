"""Paired Hellinger geometry pilots, default C/routing and historical datasets."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from recurrent_rg.config import RunConfig
from recurrent_rg.runner import file_fingerprint

def main():
    root=Path('results/hellinger_geometry_pilot_seed91').resolve()
    root.mkdir(exist_ok=False)
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONPATH='.')
    cases=[('ising2d',1200,'ising2d_signedK_B0_g1_block3_locked_recipe_h96','--signed-ising'),
           ('spin1',400,'spin1_signedK_g1_block3_locked_recipe_h96','--signed-spin1')]
    for name,epochs,stem,flag in cases:
        base=Path('results')/(stem+'_3000_seed91')
        old=json.loads((base/'manifest.json').read_text())
        cfg=RunConfig.load(Path('configs')/(stem+'_seed91.json')).to_dict()
        assert file_fingerprint(cfg['data']['cache'])==old['data_sha256']
        cfg['schema_version']=5;cfg['loss']['geometry_distance']='hellinger';cfg['training']['epochs']=epochs
        assert cfg['loss']['geometry_tail_mix']==cfg['loss']['geometry_tail_add_weight']==0
        config=root/(name+'_config.json');config.write_text(json.dumps(cfg,indent=2)+'\n')
        RunConfig.load(config).loss.objective_kwargs()
        run=root/name
        protocol=dict(baseline=str(base.resolve()),baseline_checkpoint=epochs,data_sha256=old['data_sha256'],
                      change='Hellinger distance and Hellinger half-split noise heuristic; no tail, C and R targets/routing unchanged',
                      limitations='Historical source code may differ; paired C parameters checked after training. Short pilot, not proof of convergence.')
        (root/(name+'_protocol.json')).write_text(json.dumps(protocol,indent=2)+'\n')
        commands=[
            [sys.executable,'-u','-m','recurrent_rg','train','--config',str(config),'--output',str(run)],
            [sys.executable,'scripts/report_scalar_joint.py','--run',str(run)],
            [sys.executable,'scripts/report_mi_training.py','--run',str(run)],
            [sys.executable,'scripts/plot_dual_ising_scalar.py','--run',str(run),flag],
            [sys.executable,'scripts/report_scalar_joint.py','--run',str(base),'--checkpoint',str(epochs)],
            [sys.executable,'scripts/plot_dual_ising_scalar.py','--run',str(base),'--checkpoint',str(epochs),flag]]
        start=time.time();status=dict(status='running',case=name)
        try:
            for stage,cmd in enumerate(commands):
                status.update(stage=stage);(root/(name+'_status.json')).write_text(json.dumps(status)+'\n')
                with (root/f'{name}_stage{stage}.log').open('w') as log:
                    subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
            import torch
            from recurrent_rg.checkpoints import load_run
            torch.set_num_threads(2)
            current=load_run(run).model;historical=load_run(base,epochs).model
            c_diff=max(float((a-b).abs().max()) for a,b in zip(current.rg_kernel.parameters(),historical.rg_kernel.parameters()))
            status.update(status='complete',elapsed_seconds=time.time()-start,C_max_parameter_difference=c_diff)
        except Exception as exc:
            status.update(status='failed',error=str(exc))
        (root/(name+'_status.json')).write_text(json.dumps(status,indent=2)+'\n')
        print(json.dumps(status),flush=True)
        if status['status']=='failed':raise RuntimeError(status)

if __name__=='__main__':main()
