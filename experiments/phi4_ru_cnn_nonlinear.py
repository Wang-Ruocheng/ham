"""Explicit fresh joint-QCR decoder ablation; historical/default models unchanged.

Only the shared Q decoder changes relative to the completed saddle-window run.
The residual decoder is the exact B-arm implementation previously tested with
frozen C. No directional reweighting, supervision or extra R-only stage is used.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess

import torch

from experiments.phi4_ru_qonly_decoder_balance import QArm

RECIPE = 'phi4_ru_b3_cnn_g2_Rlr003_nonlinear_decoder_v1'
ROOT = '/mnt/nas-new/home/yange/ham/results/phi4_ru_cnn_nonlinear_seed91_gpu2_vm225_20261103_v1'
SNAPSHOT = '/mnt/nas-new/home/yange/ham/snapshots/phi4_ru_cnn_nonlinear_seed91_20261103_v1'
DATA_SHA = '586f9865c6afc206bf8c8a712ccde5cf84371775f3b336099e07e9fbe88db205'
GPU_INDEX = 2
GPU_UUID = 'GPU-713b07d0-118c-f939-c0a8-565390215e01'
DECODER = dict(kind='affine_plus_centered_residual', hidden=[32,32], activation='tanh',
               residual_seed=9103, residual_last_weight_initialization='zero',
               affine_column_norm=1., residual_layer_spectral_norm_max=2.,
               output='625 soft-plaquette descriptor components; unconstrained reconstruction',
               direction_balancing=False)


def preset():
    from experiments.phi4_ru_cnn_saddle import preset as source
    config = source(DATA_SHA)
    config.update(recipe=RECIPE, decoder=dict(DECODER), gpu_index=GPU_INDEX,
                  gpu_uuid=GPU_UUID)
    return config


class JointNonlinearQ(QArm):
    def __init__(self, source):
        super().__init__(source, nonlinear=True)
        self.patch_chunk_size = source.patch_chunk_size
        self.frame_chunk = source.frame_chunk

    @torch.no_grad()
    def normalize_decoder_(self):
        self.normalize_()


def install_decoder(model):
    """Call after the unchanged TRAIN-only affine-bias initialization.

The added random parameters have a private stream, leaving C/R initialization,
physical batch RNG and C sampling RNG unchanged. Initial predictions reproduce
the initialized affine model to floating-point precision.
"""
    if isinstance(model.Q, JointNonlinearQ):
        raise ValueError('Decoder already installed')
    devices = [model.scale.device] if model.scale.device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(DECODER['residual_seed'])
        model.Q = JointNonlinearQ(model.Q)
    return model


def resource_state():
    mem = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                   if x.startswith('MemAvailable:')))*1024
    io = float(next(x.split('avg10=')[1].split()[0] for x in Path('/proc/pressure/io').read_text().splitlines()
                    if x.startswith('full ')))
    gpu = subprocess.check_output(['nvidia-smi','-i',str(GPU_INDEX),
        '--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'], text=True, timeout=10).strip().split(',')
    processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader'], text=True, timeout=10)
    if gpu[0].strip() != GPU_UUID:
        raise RuntimeError('Physical GPU2 identity mismatch; no migration')
    row = dict(gpu_index=GPU_INDEX,gpu_uuid=GPU_UUID,free_gpu_mib=int(gpu[1]),
               compute_process_present=GPU_UUID in processes,mem_available_bytes=mem,
               load1=os.getloadavg()[0],cpu_count=os.cpu_count(),io_full_avg10=io)
    row['ready'] = (not row['compute_process_present'] and row['free_gpu_mib']>=12288
                    and mem>=36*1024**3 and row['load1']<=1.25*row['cpu_count'] and io<20.)
    return row


def resource_snapshot():
    from experiments.phi4_hopping_qcr import BudgetHold
    row = resource_state()
    if not row['ready']:
        raise BudgetHold('Resources changed after queue admission; no shared launch or migration')
    return row
