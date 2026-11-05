"""Frozen range-only follow-up to the 6000-step r-u CNN experiment.

The literature saddle motivates coverage only, never a label/loss/root target.
No historical preset, sampler, trained model or output is replaced.
"""
from dataclasses import asdict
import re

RECIPE = 'phi4_ru_b3_cnn_g2_Rlr003_saddle_window_v1'
ROOT = '/mnt/nas-new/home/yange/ham/results/phi4_ru_cnn_saddle_seed91_gpu3_vm225_20261103_v1'
SNAPSHOT = '/mnt/nas-new/home/yange/ham/snapshots/phi4_ru_cnn_saddle_seed91_20261103_v1'
R_VALUES = tuple(round(-2.8 + 2.3*i/8, 8) for i in range(9))
U_VALUES = (.5, 1.125, 1.75, 2.375, 3.)


def sampling_options():
    from recurrent_rg.data.phi4_ru_pipeline import Phi4RUOptions
    return Phi4RUOptions(r_values=R_VALUES, u_values=U_VALUES)


def physical_options():
    import json
    return json.loads(json.dumps(asdict(sampling_options())))


def preset(manifest_sha):
    from experiments.phi4_ru_cnn_qcr import preset as previous
    if not isinstance(manifest_sha, str) or not re.fullmatch('[0-9a-f]{64}', manifest_sha):
        raise ValueError('A completed fresh data manifest must be SHA256 pinned')
    config = previous()
    config.update(recipe=RECIPE, data_root=ROOT+'/data', data_manifest_sha256=manifest_sha,
                  physical_options=physical_options())
    return config
