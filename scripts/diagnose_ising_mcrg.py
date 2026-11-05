"""Read-only operator-response diagnostic of the saved unsupervised b2 endpoint.

This never trains, estimates MI, samples a physical ensemble or changes a source
artifact. Existing physical configurations are mapped through the saved C with
independent diagnostic uniforms. Coupling labels are evaluation metadata only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from recurrent_rg.binary_pooling import binary_pattern_table
from recurrent_rg.checkpoints import load_run


KC = float(np.log1p(np.sqrt(2.0)) / 2.0)
EXPECTED_CHECKPOINT_SHA256 = 'fae117fed7abb86d1aec212e780a7fc2f0ce34a81524d873bcb6cf4ddc9b19f7'
EXPECTED_DATA_SHA256 = 'f0e319a771d220e948c70e97fda8bbd373d46fdbb280211f0cbd29ce9d36de54'
OPERATOR_NAMES = ('nn_axial', 'diagonal', 'plaquette', 'axial_distance2',
                  'rectangle_2x1', 'axial_distance3', 'magnetization')
OPERATOR_DEFINITIONS = (
    'sum_x s(x)[s(x+e1)+s(x+e2)]',
    'sum_x s(x)[s(x+e1+e2)+s(x+e1-e2)]',
    'sum_x s(x)s(x+e1)s(x+e2)s(x+e1+e2)',
    'sum_x s(x)[s(x+2e1)+s(x+2e2)]',
    'sum_x s(x)s(x+2e1)s(x+e2)s(x+2e1+e2) + rotated term',
    'sum_x s(x)[s(x+3e1)+s(x+3e2)]',
    'sum_x s(x)',
)
ORIENTATION_NAMES = ('nn_x', 'nn_y', 'diagonal_plus', 'diagonal_minus',
                     'axial_distance2_x', 'axial_distance2_y', 'plaquette')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def model_sha256(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')


def operator_values(spins):
    """Seven extensive translationally summed operators, one row per sample.

    No division by lattice area or operator count: otherwise the response matrix
    has a spurious inter-scale volume factor. Periodic boundaries are retained.
    """
    if spins.ndim == 4 and spins.shape[-1] == 1:
        spins = spins[..., 0]
    if spins.ndim != 3 or spins.shape[-1] != spins.shape[-2]:
        raise ValueError('Expected [samples,L,L] binary spins')
    if not torch.isfinite(spins).all() or not torch.all(spins.abs() == 1):
        raise ValueError('Selected configurations must be finite binary spins')
    s = spins.to(torch.float64)
    shift = lambda a, b: torch.roll(s, shifts=(-a, -b), dims=(-2, -1))
    values = [s * (shift(1, 0) + shift(0, 1)),
              s * (shift(1, 1) + shift(1, -1)),
              s * shift(1, 0) * shift(0, 1) * shift(1, 1),
              s * (shift(2, 0) + shift(0, 2)),
              s * shift(2, 0) * shift(0, 1) * shift(2, 1)
              + s * shift(1, 0) * shift(0, 2) * shift(1, 2),
              s * (shift(3, 0) + shift(0, 3)), s]
    return torch.stack([value.sum((-2, -1)) for value in values], dim=-1)


def covariance(left, right=None, *, ddof=1):
    """Rows are paired configurations; result rows index left operators."""
    if right is None:
        right = left
    if left.ndim != 2 or right.ndim != 2 or len(left) != len(right) or len(left) <= ddof:
        raise ValueError('Covariance requires paired 2D arrays and positive degrees of freedom')
    return (left - left.mean(0)).T @ (right - right.mean(0)) / (len(left) - ddof)


def orientation_operator_values(spins):
    """Directional basis, so isotropic grouping cannot conceal C anisotropy."""
    s = spins.to(torch.float64)
    shift = lambda a, b: torch.roll(s, shifts=(-a, -b), dims=(-2, -1))
    fields = [s * shift(1, 0), s * shift(0, 1), s * shift(1, 1),
              s * shift(1, -1), s * shift(2, 0), s * shift(0, 2),
              s * shift(1, 0) * shift(0, 1) * shift(1, 1)]
    return torch.stack([field.sum((-2, -1)) for field in fields], dim=-1)


def d4_probability_errors(patterns, probabilities):
    fields = patterns.reshape(-1, 2, 2)
    errors = []
    for reflected in (False, True):
        base = fields.flip(-1) if reflected else fields
        for rotation in range(4):
            transformed = torch.rot90(base, rotation, (-2, -1)).reshape(-1, 4)
            index = ((transformed > 0).long() * (2 ** torch.arange(4))).sum(-1)
            errors.append(probabilities[index] - probabilities)
    return torch.stack(errors)


def c_probability_table(model):
    """Exact probability table of the saved 2x2 local C; no symmetry imposed."""
    kernel = model.rg_kernel
    if kernel.kernel_layout != 'nonoverlapping' or kernel.geometry.factors != (2, 2):
        raise ValueError('Require nonoverlapping b2 C')
    if kernel.field_head.sampling_mode != 'bernoulli':
        raise ValueError('Require the saved probabilistic Bernoulli C')
    reference = next(kernel.parameters())
    _, patterns = binary_pattern_table(4, reference)
    with torch.no_grad():
        probabilities = torch.sigmoid(2 * kernel.field_head.logits(kernel.local_encoder(patterns))).flatten()
    return patterns, probabilities


def c_draw(spins, probabilities, uniform):
    """Match the original Bernoulli probability, with explicit supplied noise.

    A diagnostic draw uses u >= 1-p(+|s), exactly the hard threshold convention
    of BinaryFieldHead.sample_logits (apart from zero-probability ties).
    """
    if spins.ndim != 3 or any(size % 2 for size in spins.shape[-2:]):
        raise ValueError('Every C input side must be divisible by 2')
    indices = torch.zeros_like(spins[:, ::2, ::2], dtype=torch.long)
    for bit, (a, b) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        indices += (spins[:, a::2, b::2] > 0).long() * (2 ** bit)
    if tuple(uniform.shape) != tuple(indices.shape):
        raise ValueError('Uniform shape does not match output lattice')
    return torch.where(uniform >= 1 - probabilities[indices], 1.0, -1.0).to(spins.dtype)


def local_feature_means(model, spins):
    """Exact per-configuration pooled Q features, evaluating each binary motif once."""
    encoder = model.theory_encoder
    if tuple(encoder.geometry.widths) != (3, 3):
        raise ValueError('This diagnostic preserves Q motif width 3')
    reference = next(encoder.parameters())
    powers, patterns = binary_pattern_table(9, reference)
    indices = torch.zeros_like(spins, dtype=torch.long)
    for power, (a, b) in zip(powers, encoder.geometry.patch_offsets()):
        indices += (torch.roll(spins, shifts=(-a, -b), dims=(-2, -1)) > 0).long() * power
    flat = indices.flatten(1)
    offsets = torch.arange(len(spins))[:, None] * len(patterns)
    counts = torch.bincount((flat + offsets).flatten(), minlength=len(spins) * len(patterns))
    frequencies = counts.reshape(len(spins), len(patterns)).to(torch.float64) / flat.shape[1]
    with torch.no_grad():
        return frequencies @ encoder.local_mlp(patterns)


def head_jacobian(model, pooled):
    pooled = pooled.detach().requires_grad_(True)
    with torch.enable_grad():
        value = model.theory_encoder.head(pooled)
        jacobian = torch.stack([torch.autograd.grad(value[j], pooled, retain_graph=True)[0]
                                for j in range(value.numel())])
    return value.detach(), jacobian.detach()


def full_r_jacobian(model, g):
    g = g.detach().requires_grad_(True)
    with torch.enable_grad():
        out = model.advance_theory(g)
        jacobian = torch.stack([torch.autograd.grad(out[j], g, retain_graph=True)[0]
                                for j in range(out.numel())])
    return jacobian.detach()


def weighted_q_response(model, features, score, *, t_values=(.001, .003)):
    """Derivative of Q under empirical exponential tilting, including nonlinear head.

    Scores are extensive operators of the SAME paired configurations. Covariance
    uses 1/N, not 1/(N-1), so it is exactly the empirical reweighting derivative.
    FD epsilon = t / population score standard deviation, C draws held fixed.
    """
    g, head_jac = head_jacobian(model, features.mean(0))
    response = head_jac @ covariance(features, score, ddof=0)
    centered = score - score.mean(0)
    std = centered.square().mean(0).sqrt()
    finite_difference = torch.empty((len(t_values), len(g), score.shape[1]), dtype=torch.float64)
    for j in range(score.shape[1]):
        if std[j] <= 0:
            finite_difference[:, :, j] = 0
            continue
        for it, t in enumerate(t_values):
            eps = t / std[j]
            weights_plus = torch.softmax(eps * centered[:, j], dim=0)
            weights_minus = torch.softmax(-eps * centered[:, j], dim=0)
            with torch.no_grad():
                plus = model.theory_encoder.head(weights_plus @ features)
                minus = model.theory_encoder.head(weights_minus @ features)
            finite_difference[it, :, j] = (plus - minus) / (2 * eps)
    return g, response, finite_difference, std


def select_points(labels, k_count=3):
    if not isinstance(k_count, int) or k_count < 1 or k_count % 2 != 1:
        raise ValueError('k-count must be a positive odd integer')
    unique = np.unique(np.asarray(labels, dtype=float))
    center = int(np.argmin(abs(unique - KC)))
    if abs(unique[center] - KC) > 1e-9:
        raise ValueError('Original physical cache must contain the exact Kc evaluation cell')
    half = k_count // 2
    if center < half or center + half >= len(unique):
        raise ValueError('Not enough neighboring K cells')
    return unique[center - half:center + half + 1]


def mmap_dataset(path):
    """Read metadata and selected tensor views, never validate/scan the 9GB cache."""
    payload = torch.load(str(Path(path)), map_location='cpu', weights_only=True, mmap=True)
    if isinstance(payload, (tuple, list)) and len(payload) == 4:
        tx, ty, vx, vy = payload
        return ((tx, ty, 'train'), (vx, vy, 'test'))
    if isinstance(payload, dict) and payload.get('schema_version') == 1:
        return tuple((payload[split]['configurations'], payload[split]['couplings'], split)
                     for split in ('train', 'test'))
    raise ValueError('Require an existing supported physical cache; no physical sampling fallback')


def validate_source_provenance(source_run, data, metadata, summary):
    """Small pre-existing receipts only: no repeat scan or hash of physical cache."""
    manifest_path = source_run / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if Path(metadata['config']['data']['cache']).resolve() != data.resolve():
        raise ValueError('Explicit physical cache differs from the source configuration')
    if (metadata.get('data_sha256') != EXPECTED_DATA_SHA256
            or manifest.get('data_sha256') != metadata['data_sha256']
            or json.loads(json.dumps(manifest.get('config'))) != json.loads(json.dumps(metadata.get('config')))
            or manifest.get('labels_used_in_training') is not False
            or manifest.get('q_supervision') is not None):
        raise ValueError('Source manifest, fingerprint or unsupervised provenance mismatch')
    if summary.get('actual_epochs') != 12000:
        raise ValueError('Source must have actual 12000 completed updates')
    receipt_path = source_run.parent / 'job.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        training = receipt.get('training', {})
        if (receipt.get('status') != 'complete' or receipt.get('workflow_complete') is not True
                or training.get('actual_epochs') != 12000
                or training.get('data_sha256') != EXPECTED_DATA_SHA256
                or training.get('manifest_sha256') != sha256_file(manifest_path)):
            raise ValueError('Historical source workflow is not verifiably complete')


def diagnose(source_run, data, output, *, k_count=3, c_repeats=5, seed=12345,
             expected_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256, smoke=False):
    source_run, data, output = map(lambda p: Path(p).resolve(), (source_run, data, output))
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Refusing overwrite: {output}')
    if source_run == output or source_run in output.parents or output in source_run.parents:
        raise ValueError('Diagnostic output must be separate from historical source')
    if type(c_repeats) is not int or c_repeats < 1:
        raise ValueError('Require a positive predeclared number of C draws')
    torch.set_num_threads(1)
    checkpoint = source_run / 'training_step_12000.pt'
    fingerprint = sha256_file(checkpoint)
    if fingerprint != expected_checkpoint_sha256:
        raise ValueError('Fixed unsupervised source checkpoint fingerprint mismatch')
    loaded = load_run(source_run, 12000)
    model, metadata = loaded.model.double().eval(), loaded.payload.get('metadata', {})
    cfg = loaded.config
    if (loaded.payload.get('epoch') != 12000 or metadata.get('labels_used_in_training') is not False
            or metadata.get('q_supervision') is not None or cfg['model']['dimension'] != 2
            or cfg['model']['theory_dim'] != 1 or cfg['model']['block_factor'] != 2
            or cfg['training']['max_rg_steps'] != 2):
        raise ValueError('Require the completed fully unsupervised scalar b2 depth-two source')
    summary = json.loads((source_run / 'summary.json').read_text())
    validate_source_provenance(source_run, data, metadata, summary)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    before_hash = model_sha256(model)
    dataset = mmap_dataset(data)
    label_values = np.concatenate([labels.detach().cpu().numpy().reshape(-1) for _, labels, _ in dataset])
    k_values = select_points(label_values, k_count)
    selected = []
    for k in k_values:
        rows = [(configurations, i, split) for configurations, labels, split in dataset
                for i in np.flatnonzero(labels.detach().cpu().numpy().reshape(-1) == k)]
        if len(rows) != 12:
            raise ValueError('Require 12 independent physical ensembles at every selected K')
        selected.append(rows[:1] if smoke else rows)
    patterns, probabilities = c_probability_table(model)
    probability_symmetry_error = probabilities + probabilities.flip(0) - 1
    probability_d4_error = d4_probability_errors(patterns, probabilities)
    nrep = 1 if smoke else 12
    shape = (len(k_values), nrep, c_repeats)
    arrays = dict(
        operators=np.empty(shape + (3, 128, 7)), means=np.empty(shape + (3, 7)),
        orientation_operators=np.empty(shape + (3, 128, 7)),
        covariance=np.empty(shape + (3, 7, 7)), cross_covariance=np.empty(shape + (2, 7, 7)),
        g=np.empty(shape + (3, 1)), full_DR=np.empty(shape + (3, 1, 1)),
        q_response_micro=np.empty(shape + (3, 1, 7)),
        q_response_in=np.empty(shape + (2, 1, 7)), q_response_out=np.empty(shape + (2, 1, 7)),
        q_response_predicted=np.empty(shape + (2, 1, 7)), score_std=np.empty(shape + (3, 7)),
        q_fd_response_in=np.empty(shape + (2, 2, 1, 7)),
        q_fd_response_out=np.empty(shape + (2, 2, 1, 7)),
        symmetry_sample_mismatch=np.empty(shape + (3,)),
        symmetry_magnetization_sum=np.empty(shape + (3,)),
    )
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    write_json(output / 'status.json', dict(status='running', source_run=str(source_run),
        K=k_values.tolist(), c_repeats=c_repeats, completed_ensembles=0))
    try:
        for ik, rows in enumerate(selected):
            for replica, (source, source_index, split) in enumerate(rows):
                x0 = source[source_index].detach().to(torch.float64).clone()
                if tuple(x0.shape) != (128, 108, 108, 1):
                    raise ValueError('Selected source ensemble must have [128,108,108,1] shape')
                x0 = x0[..., 0]
                op0 = operator_values(x0)
                oriented0 = orientation_operator_values(x0)
                feature0 = local_feature_means(model, x0)
                for draw in range(c_repeats):
                    generator = torch.Generator(device='cpu')
                    draw_seed = seed + ik * 1_000_000 + replica * 10_000 + draw
                    generator.manual_seed(draw_seed)
                    levels, flipped = [x0], [-x0]
                    for step in range(2):
                        uniform = torch.rand((128, levels[-1].shape[-2] // 2,
                                              levels[-1].shape[-1] // 2),
                                             dtype=torch.float64, generator=generator)
                        levels.append(c_draw(levels[-1], probabilities, uniform))
                        flipped.append(c_draw(flipped[-1], probabilities, 1 - uniform))
                    ops = [op0] + [operator_values(level) for level in levels[1:]]
                    oriented = [oriented0] + [orientation_operator_values(level) for level in levels[1:]]
                    features = [feature0] + [local_feature_means(model, level) for level in levels[1:]]
                    ix = (ik, replica, draw)
                    arrays['operators'][ix] = torch.stack(ops).numpy()
                    arrays['orientation_operators'][ix] = torch.stack(oriented).numpy()
                    arrays['means'][ix] = torch.stack([op.mean(0) for op in ops]).numpy()
                    arrays['covariance'][ix] = torch.stack([covariance(op) for op in ops]).numpy()
                    arrays['cross_covariance'][ix] = torch.stack(
                        [covariance(ops[k + 1], ops[k]) for k in range(2)]).numpy()
                    for scale in range(3):
                        g, response_micro, _, std = weighted_q_response(model, features[scale], op0)
                        arrays['g'][ix + (scale,)] = g.numpy()
                        arrays['full_DR'][ix + (scale,)] = full_r_jacobian(model, g).numpy()
                        arrays['q_response_micro'][ix + (scale,)] = response_micro.numpy()
                        arrays['score_std'][ix + (scale,)] = ops[scale].std(0, correction=0).numpy()
                        arrays['symmetry_sample_mismatch'][ix + (scale,)] = float(
                            (levels[scale] != -flipped[scale]).to(torch.float64).mean())
                        arrays['symmetry_magnetization_sum'][ix + (scale,)] = float(
                            (levels[scale] + flipped[scale]).mean())
                    for step in range(2):
                        _, vin, fin, _ = weighted_q_response(model, features[step], ops[step])
                        _, vout, fout, _ = weighted_q_response(model, features[step + 1], ops[step])
                        arrays['q_response_in'][ix + (step,)] = vin.numpy()
                        arrays['q_response_out'][ix + (step,)] = vout.numpy()
                        arrays['q_response_predicted'][ix + (step,)] = arrays['full_DR'][ix + (step,)] @ vin.numpy()
                        arrays['q_fd_response_in'][ix + (step,)] = fin.numpy()
                        arrays['q_fd_response_out'][ix + (step,)] = fout.numpy()
                completed = ik * nrep + replica + 1
                write_json(output / 'status.json', dict(status='running', source_run=str(source_run),
                    K=k_values.tolist(), c_repeats=c_repeats, completed_ensembles=completed,
                    elapsed_seconds=time.monotonic() - start))
                print(json.dumps(dict(completed_ensembles=completed, total_ensembles=len(k_values) * nrep,
                    K=float(k_values[ik]), replica=replica, elapsed_seconds=time.monotonic() - start)), flush=True)
        after_hash = model_sha256(model)
        if before_hash != after_hash:
            raise RuntimeError('Source model state mutated during read-only diagnostic')
        arrays.update(K=k_values, operator_names=np.array(OPERATOR_NAMES),
            c_draw_seeds=np.array([[[seed + ik * 1_000_000 + replica * 10_000 + draw
                for draw in range(c_repeats)] for replica in range(nrep)] for ik in range(len(k_values))]),
            source_indices=np.array([[row[1] for row in rows] for rows in selected]),
            replica_index=np.array([[row[1] for row in rows] for rows in selected]),
            split_index=np.array([[0 if row[2] == 'train' else 1 for row in rows] for rows in selected]),
            source_splits=np.array([[row[2] for row in rows] for rows in selected]),
            c_patterns=patterns.numpy(), c_p_plus=probabilities.numpy(),
            c_spin_flip_probability_error=probability_symmetry_error.numpy(),
            c_d4_probability_error=probability_d4_error.numpy())
        np.savez_compressed(output / 'operator_statistics.npz', **arrays)
        report = dict(status='complete', source_run=str(source_run), source_checkpoint=str(checkpoint),
            source_epoch=12000, source_checkpoint_sha256=fingerprint, source_data=str(data),
            source_data_file_bytes=data.stat().st_size,
            source_data_sha256_from_checkpoint=metadata.get('data_sha256'),
            source_data_path_matches_checkpoint=True, source_manifest_matches_checkpoint=True,
            source_data_rehashed=False, source_data_loading='torch.load mmap=True, metadata plus selected cells only',
            model_sha256_before=before_hash, model_sha256_after=after_hash,
            source_labels_used_in_training=False, labels_used_here='post-hoc evaluation cell selection only',
            no_training=True, no_MI=True, no_new_physical_sampling=True,
            no_QC_parameter_changes=True, device='cpu', threads=1,
            smoke=smoke, independent_replicas=nrep,
            K=k_values.tolist(), critical_reference_K=KC,
            scales=[108, 54, 27], operator_names=OPERATOR_NAMES,
            operator_definitions=OPERATOR_DEFINITIONS,
            orientation_operator_names=ORIENTATION_NAMES,
            operator_normalization='extensive sum, no division by sites or coordination count',
            even_operator_indices=list(range(6)), odd_operator_indices=[6], nested_even_basis_sizes=[1, 3, 5, 6],
            covariance_definition='unbiased within each 128-configuration ensemble (N-1); no replica pooling',
            cross_covariance_definition='A[gamma,beta]=Cov(S_coarse_gamma,S_fine_beta); B=Cov(S_coarse,S_coarse); solve B T=A',
            q_response_definition='head Jacobian at mean pooled local features times population (1/N) covariance of features with extensive source score',
            q_response_micro_definition='same microscopic S0 perturbation through all three scales',
            q_response_transition_definition='same current-scale Sk perturbation for paired input and output',
            q_fd_definition='fixed C draw, exponential weights exp(+-epsilon*(S-mean S)), epsilon=t/population_std(S), t=.001,.003',
            q_fd_t_values=[.001, .003], c_repeats=c_repeats, seed=seed,
            c_repeats_interpretation='conditional repeated C noise on the SAME 128 physical configurations; not independent physical ensembles',
            symmetry_definition='paired sigma and -sigma with antithetic 1-u through both C steps, diagnostic only; not augmented or pooled in main statistics',
            symmetry_probability_max_abs_error=float(probability_symmetry_error.abs().max()),
            d4_probability_max_abs_error=float(probability_d4_error.abs().max()),
            d4_probability_rms_error=float(probability_d4_error.square().mean().sqrt()),
            d4_transform_order='unreflected rotations 0/90/180/270, then reflect last axis and those rotations',
            array_axes='K, independent_replica, C_draw, then named scale/transition/sample/operator axes',
            arrays={key:list(value.shape) for key, value in arrays.items()},
            limitation='Single trained seed, 12 physical replicas per K, 128 saved configurations per replica, finite L and two C steps. Operator truncation and covariance conditioning require separate analysis. No claimed thermal eigenvalue in this collection script.',
            elapsed_seconds=time.monotonic() - start)
        write_json(output / 'diagnostic.json', report)
        write_json(output / 'status.json', dict(status='complete', workflow_complete=True,
            completed_ensembles=len(k_values) * nrep, elapsed_seconds=report['elapsed_seconds']))
        return report
    except Exception as error:
        write_json(output / 'status.json', dict(status='failed', workflow_complete=False,
            error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic() - start))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--k-count', type=int, default=3)
    parser.add_argument('--c-repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--smoke', action='store_true', help='One physical replica per selected cell; never formal evidence')
    args = parser.parse_args()
    diagnose(**vars(args))


if __name__ == '__main__':
    main()
