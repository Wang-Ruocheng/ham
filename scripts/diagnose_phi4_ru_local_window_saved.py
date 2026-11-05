"""Read-only scientific diagnosis from public, SHA-pinned JSON/NPZ artifacts.

No checkpoint loading, network inference, optimizer update or physical sampling.
Optional fetch copies only the 45 explicitly whitelisted public sidecar JSONs.
PCA axes are fitted on TRAIN condition means; no fit changes the trained model.
Finite differences use one r/u grid increment as the two parameter units.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path):
    return json.loads(path.read_text())


def fetch_sidecars(final):
    public = load_json(final / 'job.json')['public_files']
    remote = '/mnt/nas-new/home/yange/ham/results/phi4_ru_cnn_local_window_seed91_gpu2_vm225_20261103_v1'
    names = [f'data/conditions/L36_c{i:04d}/sidecar.json' for i in range(45)]
    for name in names:
        assert name in public
        destination = final / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            subprocess.run(['scp', '-F', '/Users/ruochengwang/.ssh/sjtu-vm-225.conf',
                            '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                            f'225:{remote}/{name}', str(destination)], check=True, timeout=30)
        assert digest(destination) == public[name], name


def pca_diagnostic(grouped, reconstructed, train_cells):
    """Group shape [condition,4 chains,3 scales,feature]."""
    output, components = [], []
    means = grouped.mean(1)
    for k in range(3):
        center = means[train_cells, k].mean(0)
        _, s, vh = np.linalg.svd(means[train_cells, k] - center, full_matrices=False)
        variance = s**2 / train_cells.sum()
        fluctuations = (grouped[train_cells, :, k] - means[train_cells, None, k]) @ vh.T
        mean_noise = (fluctuations**2).sum(1).mean(0) / 12
        row = dict(total_between_condition_variance=float(variance.sum()),
                   variance_fractions=(variance / variance.sum())[:6].tolist(),
                   between_variance_to_chain_mean_noise=(variance / np.maximum(mean_noise, 1e-30))[:6].tolist())
        if reconstructed is not None:
            row['reconstruction'] = {}
            for name, mask in [('train', train_cells), ('test', ~train_cells)]:
                target = grouped[mask, :, k] - center
                residual = reconstructed[mask, :, k] - grouped[mask, :, k]
                projection = target @ vh[:3].T
                error = residual @ vh[:3].T
                row['reconstruction'][name] = dict(
                    relative_mse_to_train_mean=float((residual**2).sum(-1).mean() / (target**2).sum(-1).mean()),
                    pc_r2_against_train_mean=(1-(error**2).mean((0, 1)) / (projection**2).mean((0, 1))).tolist(),
                    squared_l2_error=float((residual**2).sum(-1).mean()))
        output.append(row)
        components.append(vh)
    return output, components


def local_differences(grouped, labels, bootstrap=500):
    """Chain bootstrap is descriptive, not a rank or equilibrium certificate.

    Independent conditions are resampled independently. Each chain's whole
    descriptor is resampled as one block. Fixed Q/C and one C realization;
    uncertainty excludes training seeds, finite size and sampler bias.
    """
    rng = np.random.default_rng(910311)
    lookup = {tuple(np.round(label, 8)): i for i, label in enumerate(labels)}
    means = grouped.mean(1)
    rows = []
    for c, (r, u) in enumerate(labels):
        neighbors = [(r-.15, u), (r+.15, u), (r, u-.25), (r, u+.25)]
        if not all(tuple(np.round(x, 8)) in lookup for x in neighbors):
            continue
        indices = [lookup[tuple(np.round(x, 8))] for x in neighbors]
        draw = rng.integers(0, 4, (bootstrap, 4, 4))
        for k in range(3):
            j = np.column_stack([(means[indices[1], k] - means[indices[0], k])/2,
                                 (means[indices[3], k] - means[indices[2], k])/2])
            _, singular, right = np.linalg.svd(j, full_matrices=False)
            bmeans = np.stack([grouped[index, :, k][draw[:, q]].mean(1)
                               for q, index in enumerate(indices)], axis=1)
            bj = np.stack([(bmeans[:, 1]-bmeans[:, 0])/2,
                           (bmeans[:, 3]-bmeans[:, 2])/2], axis=-1)
            delta = bj-j
            # Largest singular value of a D x 2 perturbation without D x D covariance.
            gram = np.einsum('bdi,bdj->bij', delta, delta)
            noise95 = np.sqrt(np.maximum(np.linalg.eigvalsh(gram)[:, -1], 0))
            row = dict(cell=int(c), r=float(r), u=float(u), scale=k,
                       singular_ratio=float(singular[1]/singular[0]),
                       sigma=singular.tolist(), noise95=float(np.quantile(noise95, .95)),
                       weak_sigma_over_noise95=float(singular[1]/np.quantile(noise95, .95)),
                       weak_parameter_direction=right[1].tolist())
            if j.shape == (2, 2):
                bdets = np.linalg.det(bj)
                row.update(determinant=float(np.linalg.det(j)),
                           determinant_bootstrap_interval=np.quantile(bdets, [.025, .975]).tolist())
            rows.append(row)
    return rows


def main(final, output, fetch=False):
    final, output = Path(final).resolve(), Path(output).resolve()
    if fetch:
        fetch_sidecars(final)
    assert not output.exists(), 'Use a fresh diagnostic output, preserve prior reports'
    public = load_json(final/'job.json')['public_files']
    run = final/'run'
    status = load_json(run/'status.json')
    assert status['status'] == 'complete' and status['workflow_complete']
    names = ['evaluation_arrays.npz', 'manifest.json', 'history.json', 'train_probes.json', 'roots.json']
    pins = {f'run/{name}': digest(run/name) for name in names}
    for name in names:
        assert pins[f'run/{name}'] == status['public_files'][name], name
    for name in ['data/manifest.json', 'data/quality_summary.json'] + [f'data/conditions/L36_c{i:04d}/sidecar.json' for i in range(45)]:
        pins[name] = digest(final/name)
        assert pins[name] == public[name], name
    with np.load(run/'evaluation_arrays.npz', allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    cells = np.unique(a['cell_ids'])
    assert len(cells) == 45
    groups = [np.flatnonzero(a['cell_ids'] == c) for c in cells]
    assert all(len(g) == 4 for g in groups)
    labels = np.array([a['labels'][g[0], :2] for g in groups])
    train = np.array([a['splits'][g[0]] == 'train' for g in groups])
    get = lambda key: np.stack([a[key][g] for g in groups])
    p, g, decoded = map(get, ['target_features', 'direct_g', 'reconstructed_features'])
    assert p.shape == decoded.shape == (45, 4, 3, 625)
    assert g.shape == (45, 4, 3, 2)
    assert np.allclose(p.sum(-1), 1.) and np.isfinite(p).all()
    pd, pcs = pca_diagnostic(p, decoded, train)
    gd, _ = pca_diagnostic(g, None, train)
    differences = {key: local_differences(value, labels) for key, value in [('descriptor', p), ('Q', g)]}
    differences_summary = {}
    for key, rows in differences.items():
        differences_summary[key] = []
        for k in range(3):
            selected = [r for r in rows if r['scale'] == k]
            summary = dict(median_singular_ratio=float(np.median([r['singular_ratio'] for r in selected])),
                           weak_above_noise95=sum(r['weak_sigma_over_noise95'] > 1 for r in selected),
                           weak_above_twice_noise95=sum(r['weak_sigma_over_noise95'] > 2 for r in selected),
                           count=len(selected))
            if key == 'Q':
                summary.update(positive_orientation_ci=sum(r['determinant_bootstrap_interval'][0] > 0 for r in selected),
                               negative_orientation_ci=sum(r['determinant_bootstrap_interval'][1] < 0 for r in selected))
            differences_summary[key].append(summary)
    # Exact algebraic even/odd decomposition of the saved soft-bin descriptor.
    # At h=0 the microscopic Gibbs measure has a global sign symmetry. This is
    # a diagnostic only: it does not augment data or impose symmetry on Q/C.
    symmetry = {}
    for name, sign in [('even', 1), ('odd', -1)]:
        component = (p+sign*p[..., ::-1])/2
        reconstructed_component = (decoded+sign*decoded[..., ::-1])/2
        component_pca, _ = pca_diagnostic(component, reconstructed_component, train)
        diff = local_differences(component, labels)
        symmetry[name] = dict(pca=component_pca, local_finite_differences=diff,
            local_weak_above_noise95=[sum(row['weak_sigma_over_noise95']>1 for row in diff if row['scale']==k) for k in range(3)],
            local_weak_above_twice_noise95=[sum(row['weak_sigma_over_noise95']>2 for row in diff if row['scale']==k) for k in range(3)])
    symmetry['interpretation'] = ('At k=0 this removes finite sign-occupancy imbalance without new samples. '
        'At k>0 C is not architecturally Z2-equivariant; even/odd decomposition alone does not certify its symmetry.')
    wide_run = final.parents[1]/'phi4_ru_cnn_nonlinear_seed91_vm225_20261103_monitor'/'final'/'run'
    wide_comparison = None
    if wide_run.exists():
        wide_sha = digest(wide_run/'evaluation_arrays.npz')
        assert wide_sha == load_json(wide_run/'status.json')['public_files']['evaluation_arrays.npz']
        with np.load(wide_run/'evaluation_arrays.npz', allow_pickle=False) as z:
            wg = [np.flatnonzero(z['cell_ids']==c) for c in np.unique(z['cell_ids'])]
            wt = np.array([z['splits'][ix[0]]=='train' for ix in wg])
            wp = np.stack([z['target_features'][ix] for ix in wg])
            wr = np.stack([z['reconstructed_features'][ix] for ix in wg])
        wdiag, _ = pca_diagnostic(wp, wr, wt)
        wide_comparison = dict(path=str(wide_run), evaluation_sha256=wide_sha,
                               descriptor_pca=wdiag,
                               caveat='Different data range, C and latent coordinates; not a paired causal ablation or raw latent loss ranking.')
    quality = load_json(final/'data/quality_summary.json')
    physics = []
    for c in cells:
        side = load_json(final/f'data/conditions/L36_c{c:04d}/sidecar.json')
        obs = {name: np.array([chain['diagnostics'][name]['mean'] for chain in side['chains']])
               for name in side['raw_observable_names']}
        means = {name: float(value.mean()) for name, value in obs.items()}
        binder = 1-means['magnetization_fourth']/(3*means['magnetization_squared']**2)
        physics.append(dict(cell=int(c), r=side['r'], u=side['u'], means=means,
                            binder=binder, chi_abs=float(side['size']**2*(means['magnetization_squared']-means['absolute_magnetization']**2)),
                            chain_mean_se={name:float(value.std(ddof=1)/2) for name,value in obs.items()}))
    acceptance = np.concatenate([v['production_acceptance'] for v in quality['conditions']])
    ess = [v for c in quality['conditions'] for v in c['minimum_chain_ess'].values()]
    sampling = dict(warnings=quality['warning_count'],
                    conditions_with_chain_warning=sum(c['chain_warning_count'] > 0 for c in quality['conditions']),
                    conditions_with_hot_cold_warning=sum(bool(c['hot_cold_warning_observables']) for c in quality['conditions']),
                    acceptance_range=[float(acceptance.min()), float(acceptance.max())],
                    minimum_ess=float(min(ess)), equilibrium_certified=False)
    closure = []
    pred = get('all_start_g')
    for j,(s,t) in enumerate(a['all_start_pairs']):
        norm = g[train,:,t].reshape(-1,2).std(0,ddof=0)
        # Match original report: one scalar target std across coordinates.
        scalar = float(np.sqrt((norm**2).mean()))
        for split,mask in [('train',train),('test',~train)]:
            residual = pred[mask,:,j] - g[mask,:,t]
            rmse = float(np.sqrt((residual**2).mean()))
            mean_rmse = float(np.sqrt((residual.mean(1)**2).mean()))
            closure.append(dict(start=int(s),target=int(t),split=split,rmse=rmse,
                                target_std=scalar,nrmse=rmse/scalar,
                                condition_mean_rmse=mean_rmse,
                                systematic_fraction=float(mean_rmse**2/rmse**2)))
    history, probes = load_json(run/'history.json'), load_json(run/'train_probes.json')
    # Check actual source identities; no model modules imported or executed.
    manifest = load_json(run/'manifest.json')
    source_matches = {}
    for filename, sha in manifest['source_sha256'].items():
        if '/snapshots/' in filename:
            relative = filename.split('/snapshots/')[1].split('/',1)[1]
            local = Path(__file__).resolve().parents[1]/relative
            source_matches[relative] = local.is_file() and digest(local) == sha
    result = dict(scope='saved_public_arrays_only_no_model_inference_or_training',
                  input_sha256=pins, source_files_match_frozen_run=source_matches,
                  sampling=sampling, descriptor_pca=pd, latent_pca=gd,
                  local_finite_difference_summary=differences_summary,
                  local_finite_differences=differences, physical_observables=physics,
                  symmetry_decomposition=symmetry, wide_window_comparison=wide_comparison,
                  closure=closure,
                  full_vs_two_half_Q_rmse=np.sqrt(((get('direct_full_g')-g)**2).mean((0,1))).tolist(),
                  fixed_probe_closure=[dict(step=q['step'],value=q['weighted_closure']) for q in probes],
                  caveats=['PCA variance is not physical coupling dimension; a curved line can have multiple PCs.',
                           'Singular ratios depend on latent/descriptor metrics and chosen parameter units.',
                           'Four-chain bootstrap is exploratory; no reliable rank or equilibrium certification.',
                           'Finite differences span grid increments, not infinitesimal response.',
                           'Final reconstruction diagnostic includes encoder AND decoder; cannot uniquely assign encoder causality.',
                           'Fixed probe inputs are fixed but Q/C and R targets change during training.',
                           'No causal attribution to C without a matched frozen-C/Q control.'])
    output.mkdir(parents=True)
    (output/'diagnosis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    installed={f.name for f in font_manager.fontManager.ttflist}
    font=next(x for x in ['Arial Unicode MS','Heiti TC','Noto Sans CJK SC'] if x in installed)
    plt.rcParams.update({'font.family':font,'axes.unicode_minus':False,'font.size':11})
    fig,axes=plt.subplots(2,3,figsize=(15,9),layout='constrained')
    colors=plt.cm.viridis(np.linspace(0,1,5))
    means=g.mean(1)
    for k in range(3):
        ax=axes[0,k]
        for ui,u in enumerate(np.unique(labels[:,1])):
            ix=np.flatnonzero(labels[:,1]==u)
            ax.plot(means[ix,k,0],means[ix,k,1],'-o',ms=4,color=colors[ui],label=f'u={u:.2f}')
        for r in np.unique(labels[:,0]):
            ix=np.flatnonzero(labels[:,0]==r)
            ax.plot(means[ix,k,0],means[ix,k,1],color='gray',alpha=.35,lw=.8)
        ax.set(xlabel=r'$g_1$',ylabel=r'$g_2$',title=f'k={k}：参数网格经 Q 映射\n彩线固定 u，灰线固定 r')
        ax.set_aspect('equal',adjustable='datalim')
    axes[0,0].legend(fontsize=8)
    ax=axes[1,0]
    for j,color in enumerate(['#0072B2','#D55E00']):
        ax.bar(np.arange(3)+(j-.5)*.32,[pd[k]['reconstruction']['test']['pc_r2_against_train_mean'][j] for k in range(3)],.3,color=color,label=f'描述量第{j+1}主成分')
    ax.axhline(0,color='gray',lw=.7)
    ax.set(xticks=[0,1,2],xticklabels=['k=0','k=1','k=2'],ylabel='相对训练均值基线的解释率',title='Q＋解码器：留出重构\n负值表示不如常数基线')
    ax.legend(fontsize=9)
    ax=axes[1,1]
    for key,color,title in [('descriptor','#0072B2','输入描述量'),('Q','#D55E00','Q 输出')]:
        ax.plot(range(3),[x['median_singular_ratio'] for x in differences_summary[key]],'-o',color=color,label=title)
    ax.set(xticks=range(3),xlabel='尺度 k',ylabel=r'局部有限差分 $\sigma_2/\sigma_1$ 中位数',title='两个参数方向的区分度\n仅为指定坐标度量下的诊断')
    ax.legend(fontsize=9)
    ax=axes[1,2]
    steps=np.array([q['step'] for q in probes])
    ax.semilogy(steps,[q['weighted_closure'] for q in probes],'-o',ms=3,label='固定系综探针的闭环损失')
    ax.set(xlabel='训练更新次数',ylabel='加权闭环损失',title='损失后期回升\nQ/C 改变，目标也随训练移动')
    ax.legend(fontsize=9)
    fig.suptitle('局部 φ⁴ 实验诊断：原始参数二维 ≠ 潜空间已保留两个方向',fontsize=16)
    fig.savefig(output/'Q_and_R_diagnosis.png',dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14.5,4.2),layout='constrained')
    fields=[('absolute_magnetization','平均绝对磁化强度'),('binder','Binder 累积量'),('chi_abs','绝对磁化波动量')]
    for ax,(key,title) in zip(axes,fields):
        values=np.array([v[key] if key in v else v['means'][key] for v in physics]).reshape(9,5)
        mesh=ax.pcolormesh(np.unique(labels[:,1]),np.unique(labels[:,0]),values,shading='nearest',cmap='viridis')
        ax.set(xlabel='四次项耦合 u',ylabel='质量参数 r',title=title)
        fig.colorbar(mesh,ax=ax)
    fig.suptitle('微观 L=36：已有有序—无序变化；单尺寸与四链不认证临界线',fontsize=14)
    fig.savefig(output/'physical_observables.png',dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), layout='constrained')
    variance = np.array([row['total_between_condition_variance'] for row in pd])
    axes[0].plot(range(3), variance/variance[0], '-o', label='本次局部窗口')
    if wide_comparison:
        wvar = np.array([row['total_between_condition_variance'] for row in wdiag])
        axes[0].plot(range(3), wvar/wvar[0], '-s', label='此前宽窗口（非配对因果对照）')
    axes[0].set(xlabel='尺度 k', ylabel='描述量条件间方差／各自微观方差',
                xticks=[0,1,2], title='等绝对误差权重不等于等相对信息权重')
    axes[0].legend(fontsize=9)
    for j, (name, title) in enumerate([('even','符号翻转偶分量'),('odd','符号翻转奇分量')]):
        axes[1].bar(np.arange(3)+(j-.5)*.32,symmetry[name]['local_weak_above_twice_noise95'],.3,label=title)
    axes[1].set(xlabel='尺度 k',ylabel='弱有限差分超过两倍噪声界的网格数',xticks=[0,1,2],
                ylim=(0,21),title='21 个内点：四链探索性重采样\n不是可靠秩或热化认证')
    axes[1].legend(fontsize=9)
    fig.savefig(output/'scale_and_symmetry_diagnosis.png',dpi=170)
    plt.close(fig)
    assert all(digest(final/name)==value for name,value in pins.items())
    print(json.dumps({k:result[k] for k in ['sampling','descriptor_pca','latent_pca','local_finite_difference_summary','closure']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--final',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--fetch-sidecars',action='store_true')
    args=parser.parse_args()
    main(args.final,args.output,args.fetch_sidecars)
