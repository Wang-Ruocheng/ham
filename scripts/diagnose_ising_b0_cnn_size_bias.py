"""CPU-only frozen-Q/C size controls; no optimizer, model edit or physical MC."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from recurrent_rg.checkpoints import load_run, qcr_state_dict, isolated_rng
from recurrent_rg.config import RunConfig
from recurrent_rg.data.statistics import plaquette_distribution
from recurrent_rg.encoding import half_subset_theory
from recurrent_rg.runner import prepared_data
from recurrent_rg.rsmi import kernel_probabilities
from experiments.ising_kb_frozen_r import validate_loaded_state


def fingerprint(model):
    h = hashlib.sha256()
    for key, value in sorted(qcr_state_dict(model).items()):
        h.update(key.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def descriptors(x):
    s = x[..., 0]
    m = s.mean((-2, -1))
    nn = .5 * (s * s.roll(1, -1) + s * s.roll(1, -2)).mean((-2, -1))
    p = plaquette_distribution(x).flatten(1)
    return dict(nn=nn.mean(1).tolist(), abs_m=m.abs().mean(1).tolist(),
                mean_spin=m.mean(1).tolist(), mean_m2=m.square().mean(1).tolist(),
                motif_distance_uniform=(p - 1/16).norm(dim=1).tolist())


def encode_features(q, value):
    return q.head(torch.cat((value.mean(1), value.var(1, unbiased=False)), -1))[:, 0]


def frame_features(q, x):
    e, n, side, _, _ = x.shape
    frames = x.reshape(e*n, side, side, 1).permute(0, 3, 1, 2)
    return torch.cat([q.frame_features(v) for v in frames.split(64)]).reshape(e, n, 16)


@torch.no_grad()
def run(source):
    started = time.monotonic()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    job = json.loads((source/'job.json').read_text())
    assert job['status'] == 'complete' and job['workflow_complete']
    loaded = load_run(source, checkpoint=550)
    model = loaded.model.eval(); q = model.theory_encoder
    for p in model.parameters(): p.requires_grad_(False); p.grad = None
    validation = validate_loaded_state(model, loaded.payload)
    before = fingerprint(model)
    output = dict(source=str(source), checkpoint_sha256=hashlib.sha256(loaded.checkpoint.read_bytes()).hexdigest(),
                  validation=validation, device='cpu', threads=1, training_updates=0,
                  physical_MC_updates=0, iid_control_ensembles=4, iid_control_frames=256,
                  control_seed=2026110407, iid=[], physical=[])
    gen = torch.Generator().manual_seed(2026110407)
    controls = {}; features = {}
    for side in (4, 6, 8, 12, 18, 36):
        x = (2*torch.randint(2, (4, 256, side, side, 1), generator=gen)-1).double()
        f = frame_features(q, x)
        controls[side] = x; features[side] = f
        output['iid'].append(dict(L=side, g=encode_features(q, f).tolist(), **descriptors(x),
                                 frame_feature_variance_mean=f.var(1, unbiased=False).mean(1).tolist()))
    # Same 4x4 central spins, same spatial pooling area; only the context differs.
    # The large-lattice context is independent iid material, not periodic copies.
    big = controls[36]; central = big[:, :, 16:20, 16:20].contiguous()
    spatial_pieces = []
    frames = big.reshape(-1, 36, 36, 1).permute(0, 3, 1, 2)
    for part in frames.split(64):
        maps = q.spatial(part/q.scale)[:, :, 16:20, 16:20]
        pools = torch.cat((maps.mean((-2, -1)), maps.square().mean((-2, -1))), -1)
        spatial_pieces.append(q.frame_head(pools))
    context = torch.cat(spatial_pieces).reshape(4, 256, 16)
    output['paired_context_control'] = dict(
        periodic_4=q(central)[:, 0].tolist(),
        independent_context_pool4=encode_features(q, context).tolist(),
        independent_context_pool36=encode_features(q, features[36]).tolist(),
        description='Same central iid spins; pool4 with L4 periodic reuse vs independent L36 surroundings; pool36 changes area too')
    # Diagnostic intervention on frozen features, not a proposed trained encoder.
    f4, f36 = features[4], features[36]
    output['feature_intervention'] = dict(
        mean4_variance36=q.head(torch.cat((f4.mean(1), f36.var(1, unbiased=False)), -1))[:, 0].tolist(),
        mean36_variance4=q.head(torch.cat((f36.mean(1), f4.var(1, unbiased=False)), -1))[:, 0].tolist())
    # Reuse existing physical data only. Three low-K values, all three chains;
    # repeat only the learned stochastic C, with independent diagnostic RNGs.
    cfg = RunConfig.from_dict(loaded.config)
    train, test = prepared_data(cfg)
    labels = torch.cat((train.couplings, test.couplings)).reshape(-1)
    raw = torch.cat((train.configurations, test.configurations))
    knots = labels.unique()
    selected = [float(knots[(knots-v).abs().argmin()]) for v in (.2, .26, .32)]
    mask = torch.zeros_like(labels, dtype=torch.bool)
    for v in selected: mask |= labels == v
    raw = raw[mask]; selected_labels = labels[mask]
    for seed in (4041, 4042, 4043):
        with isolated_rng('cpu', seed): levels = model.coarse_grain(raw)
        g = half_subset_theory(model, levels)[..., 0]
        for k, x in enumerate(levels):
            output['physical'].append(dict(C_seed=seed, k=k, L=x.shape[2],
                K=selected_labels.tolist(), g=g[:, k].tolist(), **descriptors(x)))
    with isolated_rng('cpu', 4044): levels = model.coarse_grain(controls[36])
    output['iid_through_C'] = [dict(k=k, L=x.shape[2], g=q(x)[:, 0].tolist(), **descriptors(x))
                               for k, x in enumerate(levels)]
    patterns = ((torch.arange(512)[:, None] >> torch.arange(8, -1, -1)) & 1).double()*2-1
    probabilities = kernel_probabilities(model.rg_kernel, patterns[None])[0, :, 1]
    # kernel_probabilities returns P(output spin = +1).
    output['C_iid_bias'] = dict(mean_plus_probability=float(probabilities.mean()),
                               spin_flip_pair_error_max=float((probabilities + probabilities.flip(0)-1).abs().max()))
    after = fingerprint(model)
    assert before == after and all(p.grad is None for p in model.parameters())
    assert not torch.cuda.is_initialized()
    output.update(qcr_sha256_before=before, qcr_sha256_after=after,
                  weights_unchanged=True, elapsed_seconds=time.monotonic()-started)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.source), ensure_ascii=False, allow_nan=False))
