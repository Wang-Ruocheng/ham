"""仅用已保存公共数组核对 Ising 观测量坐标与耦合坐标；不加载模型。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


KC = 0.4406867935
SIZES = np.array([108, 54, 27])


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def relative_error(predicted, observed):
    return np.linalg.norm(predicted - observed, axis=(-2, -1)) / np.maximum(
        np.linalg.norm(observed, axis=(-2, -1)), 1e-30)


def audit_arm(path):
    with np.load(path, allow_pickle=False) as source:
        a = {key: source[key] for key in source.files}
    ik = int(np.argmin(abs(a['K'] - KC)))
    if abs(a['K'][ik] - KC) > 1e-9:
        raise ValueError('缺少事先保存的 Kc 诊断点')
    d = a['g'].shape[-1]
    covariance = a['covariance'][ik]
    cross = a['cross_covariance'][ik]
    susceptibility = covariance[..., 0, 0] / SIZES**2
    energy, responses, holdout = [], [], []
    for step in range(2):
        # Average within-ensemble covariance estimates, NEVER pool samples
        # or add between-ensemble covariance; the five C draws are not replicas.
        vi = covariance[:, :, step, 0, 0].mean()
        vo = covariance[:, :, step + 1, 0, 0].mean()
        ax = cross[:, :, step, 0, 0].mean()
        chi_in, chi_out = vi / SIZES[step]**2, vo / SIZES[step + 1]**2
        coupling = ax / vo
        observable = 4 * ax / vi
        energy.append(dict(transition=step, coupling_one_operator=coupling,
            energy_density_response=observable, susceptibility_in=chi_in,
            susceptibility_out=chi_out,
            susceptibility_ratio=chi_out / chi_in,
            algebra_identity_error=abs(observable - coupling * chi_out / chi_in),
            caution='单算符投影不是认证热模；协方差为各独立系综内估计的平均。'))

        vin = a['q_response_in'][ik, :, :, step, :, :6]
        vout = a['q_response_out'][ik, :, :, step, :, :6]
        matrix = a['full_DR'][ik, :, :, step]
        invstd = np.divide(1., a['score_std'][ik, :, :, step, :6],
            out=np.zeros_like(a['score_std'][ik, :, :, step, :6]),
            where=a['score_std'][ik, :, :, step, :6] > 1e-12)
        x, y = vin * invstd[..., None, :], vout * invstd[..., None, :]
        predicted = (matrix @ vin) * invstd[..., None, :]
        energy_gain = np.sum(vin[..., 0] * vout[..., 0], axis=-1) / np.sum(vin[..., 0]**2, axis=-1)
        responses.append(dict(transition=step,
            existing_R_relative_error_mean=relative_error(predicted, y).mean(),
            observed_energy_direction_projection_mean=energy_gain.mean(),
            caution='能量方向投影不是不动点本征值。'))

        # Leave out an ORIGINAL ensemble, not a C draw. The common local
        # matrix is a diagnostic approximation across slightly different g.
        x, y = x.mean(1), y.mean(1)
        errors = []
        for held in range(len(x)):
            train = np.arange(len(x)) != held
            xt = x[train].transpose(1, 0, 2).reshape(d, -1)
            yt = y[train].transpose(1, 0, 2).reshape(d, -1)
            local = yt @ np.linalg.pinv(xt, rcond=1e-6)
            errors.append(float(relative_error(local @ x[held], y[held])))
        holdout.append(dict(transition=step, relative_error_mean=np.mean(errors),
            relative_error_per_ensemble=errors, rcond=1e-6,
            caution='其他11系综拟合共同局部矩阵，留1系综评估；不是训练R或固定点谱。'))

    # For step zero only, both G0=dg0/dtheta0 and G1=dg1/dtheta1
    # were saved. There is no dg2/dtheta2 array for a second-step pullback.
    fine = covariance[:, :, 0, :6, :6]
    coarse = covariance[:, :, 1, :6, :6]
    cross0 = cross[:, :, 0, :6, :6]
    coupling_t = np.linalg.pinv(coarse, rcond=1e-6) @ cross0
    g1 = a['q_response_in'][ik, :, :, 1, :, :6]
    observed = a['q_response_out'][ik, :, :, 0, :, :6]
    invstd0 = 1 / a['score_std'][ik, :, :, 0, :6]
    predicted = g1 @ coupling_t
    pullback_error = relative_error(predicted * invstd0[..., None, :],
                                    observed * invstd0[..., None, :])
    moment_map = 4 * cross0 @ np.linalg.pinv(fine, rcond=1e-6)
    chi0, chi1 = fine / SIZES[0]**2, coarse / SIZES[1]**2
    converted = chi1 @ coupling_t @ np.linalg.pinv(chi0, rcond=1e-6)

    # Descriptive regression only: three nearby K values, never a global
    # invertibility, thermal identity, or fixed-K cross-validation claim.
    g = a['g'].mean(2)
    mean_energy = a['means'].mean(2)[..., 0] / SIZES**2
    energy_regression = []
    for scale in range(3):
        target = g[:, :, scale].reshape(-1, d)
        e = mean_energy[:, :, scale].reshape(-1)
        design = np.column_stack([np.ones(len(e)), e])
        prediction = design @ np.linalg.lstsq(design, target, rcond=None)[0]
        r2 = 1 - np.sum((prediction-target)**2) / np.sum((target-target.mean(0))**2)
        energy_regression.append(dict(scale=scale, three_nearby_K_in_sample_R2=r2))
    return dict(theory_dim=d, source=str(path.resolve()), source_sha256=digest(path),
        K=float(a['K'][ik]), saved_K_values=a['K'],
        energy_susceptibility_mean=susceptibility.mean((0, 1)),
        one_operator_coordinate_identity=energy,
        existing_Q_R_response=responses, leave_one_ensemble_out=holdout,
        first_step_six_operator_chain=dict(
            relative_error_mean=pullback_error.mean(),
            relative_error_per_ensemble=pullback_error.mean(1),
            covariance_coordinate_identity_max_abs_error=abs(moment_map-converted).max(),
            coupling_matrices=coupling_t,
            moment_matrices=moment_map,
            second_step_available=False,
            caution='G1*T 预测 dg1/dtheta0；这是有限基响应一致性，不是全分布闭合或热指数认证。'),
        descriptive_energy_regression=energy_regression)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('拒绝覆盖诊断记录')
    payload = dict(status='complete', model_loading=False, model_inference=False,
        training=False, physical_sampling=False, checkpoint_access=False,
        source='已下载公共 operator_statistics.npz；无远程操作',
        coordinate_identity='M=chi_out*T*inverse(chi_in); chi=Cov(S,S)/N',
        limitations=['单训练seed、有限格点108/54/27、两次粗粒化、仅B0。',
            '观测量倍率、物理算符截断耦合谱、神经R根谱是不同对象。',
            'LOEO是同K的局部响应诊断，不能认证固定点或临界指数。',
            '对数示例只说明可能机制，不是本数据的拟合、修正公式或实证结论。'],
        logarithmic_susceptibility_illustration=[dict(L=int(L),
            assumed_observable_gain=2*np.log(L/2)/np.log(L),
            nominal_inverse_y=np.log(2)/np.log(2*np.log(L/2)/np.log(L)))
            for L in SIZES[:2]],
        arms=[audit_arm(args.root/f'dim{d}/response/operator_statistics.npz')
              for d in (1, 2, 3)])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=json_value, allow_nan=False)
        handle.write('\n')
    print(json.dumps(dict(status='complete', output=str(args.output.resolve()),
        arms=[dict(dimg=r['theory_dim'],
            energy=r['one_operator_coordinate_identity'],
            chain_error=r['first_step_six_operator_chain']['relative_error_mean'],
            LOEO=[v['relative_error_mean'] for v in r['leave_one_ensemble_out']])
            for r in payload['arms']]), ensure_ascii=False, default=json_value))


if __name__ == '__main__':
    main()
