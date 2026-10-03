"""Plot every regularly saved scalar Potts checkpoint, without selecting a winner."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recurrent_rg.checkpoints import load_run
from recurrent_rg.data.ensembles import load_ensembles
from recurrent_rg.evaluation import collect_trajectories,scalar_flow


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(2)
    files=sorted(args.run.glob('training_step_*.pt'),key=lambda f:int(f.stem.split('_')[-1]))
    if not files:raise ValueError('No scheduled checkpoints')
    cfg=json.loads((args.run/'manifest.json').read_text())['config']
    train,test=load_ensembles(cfg['data']['cache'])
    x=torch.cat((train.configurations,test.configurations))
    K=torch.cat((train.couplings,test.couplings)).numpy().ravel();knots=np.unique(K)
    rows=(len(files)+2)//3;fig,axes=plt.subplots(rows,3,figsize=(15,4.2*rows),squeeze=False,layout='constrained')
    reports=[]
    for file,ax in zip(files,axes.flat):
        step=int(file.stem.split('_')[-1]);model=load_run(file).model
        a=collect_trajectories(model,x,samples=cfg['evaluation']['samples'])
        g=a['direct'][:,:,0];means=np.array([g[K==v,0].mean() for v in knots])
        lo,hi=g[:,:-1].min(),g[:,:-1].max();width=hi-lo+1
        wide=scalar_flow(model,float(lo-2*width),float(hi+2*width),points=6001)
        lower=float(lo-max(.4,.4*(hi-lo)));upper=float(hi+max(.4,.4*(hi-lo)))
        for r in wide['roots']:
            z=r['g'][0];hits=[]
            for ka,kb,ga,gb in zip(knots[:-1],knots[1:],means[:-1],means[1:]):
                if min(ga,gb)<=z<=max(ga,gb) and ga!=gb:hits.append(float(ka+(kb-ka)*(z-ga)/(gb-ga)))
            r['inverse_microscopic_K']=hits;r['inside_observed_inputs']=bool(lo<=z<=hi)
            # Display nearby roots, but do not let very remote extrapolations
            # squash the data-supported flow; all searched roots remain in JSON.
            if lo-width<=z<=hi+width:lower=min(lower,z-.1);upper=max(upper,z+.1)
        flow=scalar_flow(model,lower,upper,points=1501)
        ax.plot(flow['grid'],flow['mapped'],label='R(g)');ax.plot(flow['grid'],flow['grid'],'k--',label='Identity')
        ax.axvspan(lo,hi,color='green',alpha=.10,label='Observed inputs')
        gc=float(np.interp(np.log1p(np.sqrt(3)),knots,means));ax.axvline(gc,color='purple',ls=':',label='Q(D) at Kc')
        for r in flow['roots']:
            z=r['g'][0];ax.scatter(z,z,color='green' if r['stability']=='stable' else 'red')
        kinds=[r['stability'] for r in flow['roots']]
        ax.set(xlabel='g (checkpoint-specific)',ylabel='R(g)',title=f'{step} updates: '+', '.join(kinds))
        ax.grid(alpha=.2);ax.legend(fontsize=7)
        reports.append(dict(updates=step,observed_range=[float(lo),float(hi)],g_at_Kc=gc,
            search_range=wide['search_range'],roots=wide['roots']))
        print(f'Audited {step}',flush=True)
    for ax in list(axes.flat)[len(files):]:ax.set_visible(False)
    out=args.run/'full_report';out.mkdir(exist_ok=True)
    fig.savefig(out/'R_checkpoints.png',dpi=150);plt.close(fig)
    (out/'checkpoint_audit.json').write_text(json.dumps(dict(checkpoints=reports,
        note='All scheduled checkpoints; no best-checkpoint selection. Bounded sign-change searches may miss tangencies.'),indent=2)+'\n')


if __name__=='__main__':main()
