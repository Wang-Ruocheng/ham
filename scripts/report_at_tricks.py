"""Matched-cache Q prediction and local-environment MI ablation audit."""
import json
from pathlib import Path
import math
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recurrent_rg.checkpoints import load_run
from recurrent_rg.data.ensembles import load_ensembles
from recurrent_rg.evaluation import collect_trajectories, closure_metrics, scalar_flow
from recurrent_rg.empirical_mi import categorical_local_pair_mi_statistics, categorical_lattice_mi_statistics, lattice_mi_statistics
from recurrent_rg.rsmi import categorical_kernel_probabilities


def main():
    torch.set_num_threads(2)
    base=Path('results');out=base/'ashkin_teller_tricks_comparison';out.mkdir(exist_ok=True)
    paths=dict(fixed_baseline='ashkin_teller2d_g2_middle_dense_fixed_decimation_1200_seed91',
               predictive_Q='ashkin_teller2d_g2_middle_dense_fixed_predictQ1_1200_seed91',
               full_MI='ashkin_teller2d_g1_middle_dense_mi_variance2ratio1_color1_1200_seed91',
               local_MI='ashkin_teller2d_g1_middle_dense_localPairMI_color1_1200_seed91')
    report={};fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    for index,(name,path) in enumerate(paths.items()):
        folder=base/path;run=load_run(folder);m=run.model.eval();tr,te=load_ensembles(run.config['data']['cache'])
        train=collect_trajectories(m,tr.configurations,samples=128,max_rg_steps=2)
        test=collect_trajectories(m,te.configurations,samples=128,max_rg_steps=2)
        K=torch.cat([tr.couplings,te.couplings])[:,0].numpy();knots=np.unique(K)
        d=np.concatenate([train['direct'],test['direct']]);means=np.stack([d[K==v].mean(0) for v in knots])
        rec=dict(closure=closure_metrics(test,reference_direct=train['direct']))
        dim=d.shape[-1]
        rec['input_covariance_eigenvalues']=np.linalg.eigvalsh(np.cov(d[:,:-1].reshape(-1,dim).T,bias=True).reshape(dim,dim)).tolist()
        mask=(te.couplings[:,0].numpy()>=.17)&(te.couplings[:,0].numpy()<=.245)
        err=test['one_step'][mask]-test['direct'][mask,1:]
        motion=test['direct'][mask,1:]-test['direct'][mask,:-1]
        rec['transition_error_over_motion']=(np.sqrt((err**2).mean((0,2)))/np.sqrt((motion**2).mean((0,2)))).tolist()
        ax=axes.ravel()[index]
        if dim==2:
            for k in range(3):ax.plot(*means[:,k].T,'o-',ms=3,label=f'k={k}')
            rec['roots']=json.loads((folder/'report/fixed_points.json').read_text())['roots']
            ax.set(xlabel='g1',ylabel='g2',title=name)
        else:
            lo,hi=d[:,:-1].min(),d[:,:-1].max();pad=max(.3,.3*(hi-lo))
            f=scalar_flow(m,float(lo-pad),float(hi+pad),points=6001)
            rec['roots']=f['roots']
            for r in rec['roots']:
                z=r['g'][0];r['inverse_K']=[float(a+(b-a)*(z-u)/(v-u)) for a,b,u,v in zip(knots[:-1],knots[1:],means[:-1,0,0],means[1:,0,0]) if u!=v and min(u,v)<=z<=max(u,v)]
            ax.plot(f['grid'],f['mapped'],label='R(g)');ax.plot(f['grid'],f['grid'],'k--',label='identity');ax.set(xlabel='g',ylabel='R(g)',title=name)
            codes=(torch.arange(256)[:,None]//(4**torch.arange(4)))%4
            pat=torch.nn.functional.one_hot(codes,4).flatten(1).double()[None]
            with torch.no_grad():table=categorical_kernel_probabilities(m.rg_kernel,pat)[0]
            ids=[int(np.argmin(abs(te.couplings[:,0].numpy()-v))) for v in [.15,.20,.21,.28]]
            x=te.configurations[ids]
            actual=(categorical_local_pair_mi_statistics(x) if name=='local_MI' else categorical_lattice_mi_statistics(x))
            shuffled=(categorical_local_pair_mi_statistics(x,shuffle_visible_seed=551) if name=='local_MI' else lattice_mi_statistics(x,states=4,shuffle_visible_seed=551))
            with torch.no_grad():
                rec['MI']=dict(K=te.couplings[ids,0].tolist(),actual_bits=(actual.information(table)/math.log(2)).tolist(),shuffled_bits=(shuffled.information(table)/math.log(2)).tolist(),statistics=actual.diagnostics)
            rec['iid_product_reference']={}
            signs=np.array([[-1,-1,1],[-1,1,-1],[1,-1,-1],[1,1,1]])
            for label,p0 in [('product_plus',[.5,0,0,.5]),('product_minus',[0,.5,.5,0]),('disordered',[.25]*4)]:
                p=np.array(p0);rows=[]
                for k in range(5):
                    rows.append((p@signs).tolist());p=np.prod(p[codes.numpy()],axis=1)@table.numpy()
                rec['iid_product_reference'][label]=rows
        ax.legend(fontsize=8);ax.grid(alpha=.2);report[name]=rec
    fig.savefig(out/'comparison.png',dpi=180)
    report['caveat']='Two separate single-seed ablations, not a combined model: predictive Q uses fixed C dim2 and shared step400 checkpoint; local MI uses trainable C dim1 fresh initialization. Cross-family raw totals are not comparable. All report trajectories use 128 configurations.'
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:{'closure':v['closure'],'roots':v['roots'],'transition_error_over_motion':v['transition_error_over_motion']} for k,v in report.items() if isinstance(v,dict)},indent=2))


if __name__=='__main__':main()
