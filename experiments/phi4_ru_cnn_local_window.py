"""Range-only ablation around a literature WF-like reference, not a target root.

Retain the preceding explicit continuous nonlinear-decoder recipe. The physical
grid is narrower; all counts, seed policy, model, losses and updates are fixed.
This is not a change to the repository's default QCR recipe.
"""
from dataclasses import asdict
import json
import re

from experiments import phi4_ru_cnn_nonlinear as previous

RECIPE = 'phi4_ru_b3_cnn_g2_Rlr003_nonlinear_local_window_v1'
ROOT = '/mnt/nas-new/home/yange/ham/results/phi4_ru_cnn_local_window_seed91_gpu2_vm225_20261103_v1'
SNAPSHOT = '/mnt/nas-new/home/yange/ham/snapshots/phi4_ru_cnn_local_window_seed91_20261103_v1'
R_VALUES = tuple(round(-2.57 + 1.2*i/8, 8) for i in range(9))
U_VALUES = (1.63, 1.88, 2.13, 2.38, 2.63)
GPU_INDEX, GPU_UUID = previous.GPU_INDEX, previous.GPU_UUID
REFERENCE_RU = (-1.97, 2.13)
resource_state = previous.resource_state
resource_snapshot = previous.resource_snapshot


def sampling_options():
    from recurrent_rg.data.phi4_ru_pipeline import Phi4RUOptions
    return Phi4RUOptions(r_values=R_VALUES, u_values=U_VALUES)


def physical_options():
    return json.loads(json.dumps(asdict(sampling_options())))


def preset(manifest_sha):
    if not isinstance(manifest_sha, str) or not re.fullmatch('[0-9a-f]{64}', manifest_sha):
        raise ValueError('Require the completed new physical manifest SHA256')
    config = previous.preset()
    config.update(recipe=RECIPE, data_root=ROOT+'/data', data_manifest_sha256=manifest_sha,
                  physical_options=physical_options())
    return config
