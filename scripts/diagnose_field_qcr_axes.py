"""Resolve rollout errors along strong/weak latent axes using saved trajectories."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    arrays = np.load(args.run/'field_flow_data.npz')
    direct, rollout, test = arrays['direct'], arrays['rollout'], arrays['is_test']
    diagnostics = json.loads((args.run/'field_diagnostics.json').read_text())
    result = {'axis_order': ['weak (small eigenvalue)', 'strong (large eigenvalue)'], 'test_rollout_by_scale': []}
    for k in range(1, 4):
        eig, axes = np.linalg.eigh(np.cov(direct[~test, k].T, bias=True))
        errors = (rollout[test, k]-direct[test, k])@axes
        rmse = np.sqrt(np.mean(errors**2, axis=0))
        result['test_rollout_by_scale'].append({'scale': k, 'pc_rmse': rmse.tolist(),
                                               'pc_rmse_over_train_pc_sd': (rmse/np.sqrt(eig)).tolist()})
    center = direct[:, 0].mean(0)
    eig, axes = np.linalg.eigh(np.cov(direct[:, 0].T, bias=True))
    result['microscopic_pc_standardized_support'] = [((direct[:, 0]-center)@axes/np.sqrt(eig)).min(0).tolist(),
                                                    ((direct[:, 0]-center)@axes/np.sqrt(eig)).max(0).tolist()]
    result['roots_in_microscopic_standardized_pc_frame'] = [((np.array(r['g'])-center)@axes/np.sqrt(eig)).tolist()
                                                            for r in diagnostics['fixed_points']]
    result['caution'] = 'Axis-normalized errors expose weak-direction errors hidden by pooled-coordinate normalization; not physical critical exponents.'
    (args.run/'axis_diagnostics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
