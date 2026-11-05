"""Explicit parameter-scan follow-up to the continuous nonlinear CNN recipe."""
import re
from experiments import phi4_ru_cnn_nonlinear as previous

RECIPE = 'phi4_kappa_h_b3_cnn_g2_nonlinear_seed91_v1'
ROOT = '/mnt/nas-new/home/yange/ham/results/phi4_kappa_h_cnn_seed91_gpu2_vm225_20261103_v1'
SNAPSHOT = '/mnt/nas-new/home/yange/ham/snapshots/phi4_kappa_h_cnn_seed91_20261103_v1'
KAPPA_VALUES = (.56,.59,.62,.65,.68,.71,.74,.77,.80)
H_VALUES = (-.01,-.002,0.,.002,.01)
GPU_INDEX, GPU_UUID = previous.GPU_INDEX, previous.GPU_UUID
resource_state, resource_snapshot = previous.resource_state, previous.resource_snapshot

def physical_options():
    return dict(kappa_values=list(KAPPA_VALUES), h_values=list(H_VALUES), quartic_lambda=1.,
        sizes=[36],dimension=2,chains=4,samples=128,burn_in=1024,spacing=8,
        step_size=.04,leapfrog_steps=24,seed=20261103,test_every=4,test_offset=1,
        cluster_probability=.5)

def preset(manifest_sha):
    if not isinstance(manifest_sha,str) or not re.fullmatch('[0-9a-f]{64}',manifest_sha):
        raise ValueError('Require completed fresh manifest SHA256')
    config=previous.preset()
    config.update(recipe=RECIPE,data_root=ROOT+'/data',data_manifest_sha256=manifest_sha,
        physical_options=physical_options())
    return config
