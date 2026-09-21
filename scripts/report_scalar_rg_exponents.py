"""Audit scalar RG slopes; the thermal interpretation remains conditional."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recurrent_rg.model_nd import RecurrentRGNetwork1D, RecurrentRGNetwork2D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads((args.run / 'summary.json').read_text())['config']
    reports = json.loads((args.run / 'scalar_report.json').read_text())
    assert cfg['theory_dim'] == 1
    constructor = RecurrentRGNetwork2D if cfg.get('dimension') == 2 else RecurrentRGNetwork1D
    output = {'length_scale_factor': 2, 'interpretation':
              'nu_eff assumes a regular thermal coordinate and physically faithful RG; not validated by closure alone.',
              'checkpoints': {}}
    for step, report in reports.items():
        model = constructor(levels=cfg['levels'], hidden=cfg['hidden'], theory_dim=1).double().eval()
        path = args.run / f'training_step_{step}.pt'
        state = torch.load(path if path.exists() else args.run / 'qcr_model.pt', weights_only=True)
        mismatch = model.load_state_dict(state.get('model', state), strict=False)
        assert not mismatch.unexpected_keys
        assert all(k.startswith('action_network.') for k in mismatch.missing_keys)

        def residual(g):
            with torch.no_grad():
                x = torch.tensor([[g]], dtype=torch.float64)
                return (model.advance_theory(x) - x).item()

        roots = []
        for saved in report['roots']:
            root = saved['g']
            x = torch.tensor([[root]], dtype=torch.float64, requires_grad=True)
            slope = torch.autograd.grad(model.advance_theory(x).sum(), x)[0].item()
            eps = 1e-5
            finite_difference = 1 + (residual(root + eps) - residual(root - eps)) / (2 * eps)
            assert abs(slope - finite_difference) < 1e-6
            assert abs(residual(root)) < 1e-8
            row = saved | {'fixed_point_residual': residual(root), 'finite_difference_slope': finite_difference,
                           'y': math.log(abs(slope)) / math.log(2) if slope else None,
                           'nu_eff_if_thermal': math.log(2) / math.log(slope) if slope > 1 else None}
            # All piecewise-linear inverse intersections, without assuming monotonicity.
            coupling = np.asarray(report['coupling'])
            g0 = np.asarray(report['g_mean'])[:, 0]
            intersections = []
            for a, b, ga, gb in zip(coupling[:-1], coupling[1:], g0[:-1], g0[1:]):
                if (ga-root)*(gb-root) < 0:
                    intersections.append(float(brentq(lambda k: np.interp(k, [a,b], [ga,gb])-root, a,b)))
            row['microscopic_coupling_intersections_not_physical_fixed_points'] = intersections
            roots.append(row)
        output['checkpoints'][step] = {'roots': roots, 'g_at_Kc': report.get('g_at_Kc')}
    path = args.run / 'critical_exponent_audit.json'
    path.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
