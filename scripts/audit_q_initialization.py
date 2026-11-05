"""Read-only seed audit of original Q initialization; no training or labels in model."""
import inspect
import json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from recurrent_rg.checkpoints import load_run,build_model
from recurrent_rg.data.ensembles import load_ensembles
from experiments.q_only_cached_sweep import sample_counts,encode


def main():
    torch.set_num_threads(2)
    source=Path('results/spin1_Qonly_loss_candidates_exact_2400_seed91')
    out=source/'initialization_audit';out.mkdir(exist_ok=True)
    loaded=load_run(source/'baseline.pt');_,ds=load_ensembles(loaded.config['data']['cache'])
    k=ds.couplings.numpy().reshape(len(ds.configurations),-1)[:,0];unique=np.unique(k)
    hist=torch.from_numpy(np.stack([np.asarray(sample_counts(x[None],loaded.model.geometry,3).mean(0)).ravel() for x in ds.configurations]))
    spins=ds.configurations.argmax(-1).double()-1
    rho0=(spins==0).double().mean((1,2,3)).numpy();mag=spins.mean((1,2,3)).numpy()
    codes=(torch.arange(3**9)[:,None]//(3**torch.arange(9)))%3
    patterns=torch.nn.functional.one_hot(codes,3).flatten(1).double()
    stats={};curves={}
    with torch.no_grad():
        for seed in range(91,123):
            torch.manual_seed(seed);q=build_model(loaded.model_config,2).theory_encoder
            g=encode(q,patterns,hist).numpy().ravel()
            if seed==91:np.testing.assert_allclose(g,np.load(source/'feature_audit/random_features.npz')['g_test'].ravel(),atol=1e-10,rtol=1e-10)
            curve=np.array([g[k==v].mean() for v in unique]);delta=np.diff(curve)
            stats[str(seed)]=dict(abs_spearman_K=float(abs(spearmanr(k,g).statistic)),abs_spearman_absK=float(abs(spearmanr(abs(k),g).statistic)),corr_vacancy=float(np.corrcoef(rho0,g)[0,1]),corr_magnetization=float(np.corrcoef(mag,g)[0,1]),reverse_variation_fraction=float(min(np.maximum(delta,0).sum(),np.maximum(-delta,0).sum())/max(abs(delta).sum(),1e-12)),g_std=float(g.std()),g_mean=float(g.mean()))
            curves[str(seed)]=curve
    report=dict(protocol='32 untrained full-model seeds 91..122; original builder and RNG order; same microscopic test ensembles; diagnostic labels only; reversal fraction is min(total positive increments,total negative increments)/total variation; >0.1 is a descriptive substantial-fold threshold, not a theorem',torch_linear_reset_source=inspect.getsource(torch.nn.Linear.reset_parameters),results=stats)
    report['summary']=dict(substantial_fold_count=sum(r['reverse_variation_fraction']>.1 for r in stats.values()),absK_more_correlated_count=sum(r['abs_spearman_absK']>r['abs_spearman_K'] for r in stats.values()),median_abs_rhoK=float(np.median([r['abs_spearman_K'] for r in stats.values()])),median_abs_rho_absK=float(np.median([r['abs_spearman_absK'] for r in stats.values()])))
    (out/'metrics.json').write_text(json.dumps(report,indent=2));np.savez(out/'curves.npz',K=unique,**curves)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    for ax,seed in zip(axes.flat,range(91,97)):
        curve=curves[str(seed)];ax.plot(unique,(curve-curve.mean())/curve.std(),'-o',markersize=3)
        ax.set(title=f'Untrained Q, seed {seed}',xlabel='K (evaluation only)',ylabel='standardized g')
    fig.tight_layout();fig.savefig(out/'untrained_g_K.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(summary=report['summary'],seed91=stats['91'],seed_range=[91,122]),indent=2))
    print(report['torch_linear_reset_source'])


if __name__=='__main__':main()
