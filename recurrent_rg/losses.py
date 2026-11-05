"""Shared QCR objective. Labels and physical sampling never enter this module."""
import torch
import torch.nn.functional as F
from .config import QCRWeights, validate_closure_scale_weights, validate_closure_objective
from .metrics import mean_pairwise_square
from .encoding import split_theory_views

def binary_kernel_noncollapse(levels: list[torch.Tensor]) -> torch.Tensor:
    """Keep both binary coarse states populated across a balanced batch.

    Every coarse tensor has a straight-through hard forward value, so a
    constant +1/-1 kernel gives loss one while gradients still flow through
    the relaxed binary head. The microscopic level is intentionally excluded.
    """
    coarse_means = torch.stack([level.mean() for level in levels[1:]])
    return coarse_means.square().mean()


def distribution_geometry_loss(theory, patterns_a, patterns_b, *, scale=4.0, noise_multiplier=2.0,
                               mode='absolute', relative_floor=.05, max_inverse_weight=100.,
                               neighbors=4, local_fraction=.5, tail_mix=0., tail_fraction=.1,
                               tail_min_reliability=.5, tail_add_weight=0., distance_metric='euclidean', bounds='interval'):
    """Match distances to fixed-unit, split-noise intervals in pattern space.

    Targets are detached (including from C). A half-split difference estimates
    the noise of the mean distribution: E||pa-pb||^2/4. This is a heuristic
    noise allowance, not a confidence interval or a distribution-equality test.
    No batch standardization, fixed spread, labels, or trainable target scale.
    """
    if scale <= 0 or noise_multiplier < 0:
        raise ValueError('Need positive geometry scale and nonnegative noise multiplier')
    if bounds not in ('interval', 'lower_only'):
        raise ValueError('Unknown geometry bounds')
    if theory.shape[0] < 2:
        raise ValueError('Need at least two ensembles')
    if mode not in ('absolute', 'relative_local'):
        raise ValueError('Unknown geometry mode')
    if distance_metric not in ('euclidean','hellinger'):
        raise ValueError('Unknown geometry distance')
    if not 0 <= tail_mix <= 1 or not 0 < tail_fraction <= 1 or not 0 <= tail_min_reliability <= 1:
        raise ValueError('Invalid geometry tail settings')
    if not 0 <= tail_add_weight < float('inf') or (tail_mix and tail_add_weight):
        raise ValueError('Invalid or conflicting additive geometry tail weight')
    if (tail_mix or tail_add_weight) and mode != 'relative_local':
        raise ValueError('Geometry tail requires relative_local')
    if relative_floor <= 0 or max_inverse_weight <= 0 or neighbors < 1 or not 0 <= local_fraction <= 1:
        raise ValueError('Invalid relative geometry settings')
    with torch.no_grad():
        mean = .5 * (patterns_a + patterns_b)
        noise = .25 * (patterns_a - patterns_b).square().sum(dim=-1)
        i, j = torch.triu_indices(len(theory), len(theory), offset=1, device=theory.device)
        if distance_metric == 'hellinger':
            for p in (patterns_a,patterns_b):
                if not torch.isfinite(p).all() or (p < 0).any() or not torch.allclose(p.sum(-1),torch.ones_like(p.sum(-1)),atol=1e-5,rtol=1e-5):
                    raise ValueError('Hellinger requires normalized nonnegative histograms')
            # Euclidean embedding phi(p)=sqrt(p)/sqrt(2), with H in [0,1].
            # Split noise is H(pa,pb)^2/4: a local delta-method heuristic
            # for the pooled histogram, NOT an unbiased estimate or CI.
            phi=mean.sqrt()/(2**.5)
            noise=.125*(patterns_a.sqrt()-patterns_b.sqrt()).square().sum(-1)
            distance=(phi[i]-phi[j]).norm(dim=-1)
        else:
            distance = (mean[i] - mean[j]).norm(dim=-1)
        allowance = noise_multiplier * (noise[i] + noise[j]).sqrt()
        lower = scale * (distance - allowance).clamp_min(0)
        upper = scale * (distance + allowance)
    latent = torch.pdist(theory)
    penalties = F.relu(lower-latent).square()
    if bounds == 'interval':
        penalties = penalties + F.relu(latent-upper).square()
    loss = penalties.mean()
    extra = {}
    if mode == 'relative_local':
        with torch.no_grad():
            target = scale*distance
            uncertainty = scale*allowance
            denominator = target.square()+uncertainty.square()+relative_floor**2
            inverse = denominator.reciprocal().clamp(max=max_inverse_weight)
            reliability = target.square()/denominator
            # Select neighbors from data, never from the learned coordinates.
            # Unresolved pairs cannot consume all the local-neighbor slots.
            distances = theory.new_full((len(theory),len(theory)), float('inf'))
            resolved_distance = target.masked_fill(lower <= 0, float('inf'))
            distances[i,j] = resolved_distance; distances[j,i] = resolved_distance
            selected = distances.topk(min(neighbors,len(theory)-1),largest=False,dim=1).indices
            directed = torch.zeros_like(distances,dtype=torch.bool)
            directed.scatter_(1,selected,True)
            directed &= torch.isfinite(distances)
            local = (directed | directed.T)[i,j]
            local_weights = reliability*local
        scaled = penalties*inverse
        global_loss = (reliability*scaled).sum()/reliability.sum().clamp_min(1e-12)
        local_loss = (local_weights*scaled).sum()/local_weights.sum().clamp_min(1e-12)
        # No reliable local pairs: retain the global noise-aware objective.
        mix = local_fraction if bool(local.any()) else 0.
        loss = (1-mix)*global_loss+mix*local_loss
        extra = {'relative_global':global_loss,'relative_local':local_loss,
                 'local_pair_fraction':local.to(theory.dtype).mean(),
                 'mean_reliability':reliability.mean()}
        if tail_mix or tail_add_weight:
            import math
            # Opt-in ablation only. Retain the noise interval and detached
            # reliability; never mine unresolved pairs or use physical labels.
            eligible = torch.where((lower > 0) & (reliability >= tail_min_reliability))[0]
            base_loss = loss
            if len(eligible):
                count = max(1, math.ceil(tail_fraction * len(eligible)))
                chosen = eligible[scaled[eligible].detach().topk(count).indices]
                tail_loss = (reliability[chosen]*scaled[chosen]).sum()/reliability[chosen].sum().clamp_min(1e-12)
                loss = (1-tail_mix)*base_loss+(tail_mix+tail_add_weight)*tail_loss
            else:
                # With no eligible pairs there is no additive penalty.
                # Preserve the historical mixture fallback exactly.
                tail_loss = base_loss if tail_mix else base_loss*0
            extra.update(geometry_base=base_loss,geometry_tail=tail_loss,
                         tail_eligible_fraction=theory.new_tensor(len(eligible)/len(latent)))
    return loss, {
        'resolved_fraction': (lower > 0).to(theory.dtype).mean(),
        'target_distance': (scale * distance).mean(),
        'noise_allowance': (scale * allowance).mean(),
        'latent_distance': latent.mean(),
        **extra,
    }


def latent_squared_error(pred, target, coordinate_reduction='sum'):
    if coordinate_reduction == 'sum':
        return (pred-target).square().sum(dim=-1).mean()
    if coordinate_reduction != 'mean':
        raise ValueError('coordinate_reduction must be mean or sum')
    return F.mse_loss(pred, target)


def noise_aware_precision(first, second):
    """Detached, trace-normalized metric; amplify reliable weak axes by <=10x.

    Half-split differences estimate noise of the mean encoder output. This
    heuristic does not enforce full rank, and leaves unresolved axes unboosted.
    """
    with torch.no_grad():
        g = .5*(first+second)
        centered = g-g.mean(0)
        covariance = centered.T@centered/len(g)
        noise = (first-second).T@(first-second)/(4*len(g))
        eig, vectors = torch.linalg.eigh(covariance)
        projected_noise = (vectors.T@noise@vectors).diagonal().clamp_min(0)
        reliability = ((eig-projected_noise).clamp_min(0)/(eig+1e-6)).clamp(0,1)
        boost = (eig.mean()/(eig+1e-4)).clamp(1,10)
        weights = 1+reliability*(boost-1)
        weights = weights/weights.mean()
        return (vectors*weights)@vectors.T


def metric_error(pred, target, precision, reduction='sum'):
    error = pred-target
    value = torch.einsum('...i,ij,...j->...',error,precision,error).mean()
    return value if reduction == 'sum' else value/pred.shape[-1]


def local_tangent_loss(source, predicted, target, pa, pb, neighbors=4, noise_multiplier=2.):
    """Data-selected resolved neighbors; bounded relative pair-residual loss."""
    with torch.no_grad():
        mean = .5*(pa+pb)
        noise = .25*(pa-pb).square().sum(-1)
        distance = torch.cdist(mean,mean)
        allowance = noise_multiplier*(noise[:,None]+noise[None,:]).sqrt()
        resolved = distance > allowance
        resolved.fill_diagonal_(False)
        candidates = distance.masked_fill(~resolved,float('inf'))
        ids = candidates.topk(min(neighbors,len(source)-1),largest=False).indices
        mask = torch.zeros_like(resolved).scatter_(1,ids,True)&resolved
        i,j = torch.where(torch.triu(mask|mask.T,diagonal=1))
        reliability = (distance[i,j]-allowance[i,j]).clamp_min(0)/(distance[i,j]+1e-6)
        inverse = ((source[i]-source[j]).square().sum(-1)+.05**2).reciprocal().clamp(max=100)
    residual = (predicted[i]-predicted[j])-(target[i]-target[j])
    return (residual.square().sum(-1)*inverse*reliability).sum()/reliability.sum().clamp_min(1e-12)


def closure_scale_mean(terms, closure_scale_weights=None):
    """Normalized target-scale mean; a uniform choice preserves legacy math.

    Rescale weights before tensor conversion so even large finite entries do
    not overflow in lower-precision objectives. No labels or batch statistics
    enter the fixed scale weights.
    """
    values = torch.stack(terms)
    scale_weights = validate_closure_scale_weights(closure_scale_weights, len(terms))
    if scale_weights is None or len(set(scale_weights)) == 1:
        return values.mean()
    largest = max(scale_weights)
    weights = values.new_tensor([value/largest for value in scale_weights])
    return (values * (weights/weights.sum())).sum()


def one_step_closure_loss(predicted, target, coordinate_reduction='sum',
                          closure_scale_weights=None):
    """One-step closure weighted by target k, retaining legacy raw metrics."""
    scale_weights = validate_closure_scale_weights(closure_scale_weights, target.shape[1])
    unweighted = latent_squared_error(predicted, target, coordinate_reduction)
    errors = [latent_squared_error(predicted[:, k], target[:, k], coordinate_reduction)
              for k in range(target.shape[1])]
    # The vectorized legacy reduction may differ at roundoff from a mean of
    # per-scale means. Keep it exactly for None and every uniform weighting.
    loss = (unweighted if scale_weights is None or len(set(scale_weights)) == 1
            else closure_scale_mean(errors, scale_weights))
    return loss, errors, unweighted


def multistep_closure_loss(model, direct, coordinate_reduction='sum',
                           closure_scale_weights=None):
    """Autonomous rollout from g0; retain gradients through every R call."""
    predicted = direct[:, 0]
    errors = []
    for k in range(1, direct.shape[1]):
        predicted = model.advance_theory(predicted)
        errors.append(latent_squared_error(predicted, direct[:, k], coordinate_reduction))
    return closure_scale_mean(errors, closure_scale_weights), errors


def all_start_anchor_closure_loss(model, direct, coordinate_reduction='sum',
                                  closure_scale_weights=None):
    """Match every earlier start to each directly encoded target scale.

    Average starts within each target, then average target scales uniformly.
    Every recurrent R call stays in the graph. The supplied encodings and C
    trajectory are reused; no physical labels, additional sampling or Q calls
    enter this loss. Gradient routing to R alone is the trainer's responsibility.

    Return the normalized loss, per-target means and raw (start, target) errors.
    """
    if direct.ndim != 3 or min(direct.shape) < 1 or direct.shape[1] < 2:
        raise ValueError('all_start_anchor needs nonempty [batch, scales>=2, coordinates] encodings')
    if coordinate_reduction not in ('sum', 'mean'):
        raise ValueError('coordinate_reduction must be mean or sum')
    steps = direct.shape[1] - 1
    scale_weights = validate_closure_scale_weights(closure_scale_weights, steps)
    if scale_weights is not None and len(set(scale_weights)) != 1:
        raise ValueError('all_start_anchor requires uniform closure_scale_weights')
    pairs = {}
    for start in range(steps):
        recurrent = direct[:, start]
        for target in range(start + 1, steps + 1):
            recurrent = model.advance_theory(recurrent)
            pairs[start, target] = latent_squared_error(
                recurrent, direct[:, target], coordinate_reduction)
    targets = [torch.stack([pairs[start, target] for start in range(target)]).mean()
               for target in range(1, steps + 1)]
    return torch.stack(targets).mean(), targets, pairs


def binary_kernel_sign_loss(kernel):
    """Exact Bernoulli NLL on all-minus/all-plus blocks; no sampling or labels.

    Fixes the orientation of uniform states, not mixed blocks or Z2 equivariance.
    The recurrent C is shared, so this regularizer is counted once per batch.
    """
    from .adapters import BinaryFieldHead
    head = getattr(kernel, 'field_head', None)
    if not isinstance(head, BinaryFieldHead) or head.sampling_mode != 'bernoulli' or kernel.input_channels != 1:
        raise ValueError('Kernel sign loss requires a learnable single-channel Bernoulli C')
    first = kernel.local_encoder[0]
    signs = first.weight.new_tensor([-1., 1.])
    patterns = signs[:, None].expand(2, first.in_features)
    logits = head.logits(kernel.local_encoder(patterns)).flatten()
    return F.softplus(-2*signs*logits).mean()


def categorical_kernel_preserve_loss(kernel):
    """Exact categorical NLL retaining each uniform block's color, counted once."""
    q = kernel.input_channels
    first = kernel.local_encoder[0]
    patterns = torch.eye(q, dtype=first.weight.dtype, device=first.weight.device).repeat(1,first.in_features//q)
    logits = kernel.field_head.logits(kernel.local_encoder(patterns))
    return F.cross_entropy(logits, torch.arange(q,device=logits.device))


def categorical_kernel_variance_floor(levels, ratio=.25, *, per_ensemble=False):
    """Trace-covariance floor: trace Cov(onehot) = 1-||mean(onehot)||^2.

    Hinge on the norm, retaining an STE derivative at a constant output.
    Does not force all colors to have equal occupancy.
    """
    terms = []
    for source, target in zip(levels[:-1], levels[1:]):
        dims = tuple(range(1 if per_ensemble else 0,source.ndim-1))
        variance = (1-source.detach().mean(dims).square().sum(-1)).clamp(0,1)
        bound = (1-ratio*variance).sqrt()
        target_dims=tuple(range(1 if per_ensemble else 0,target.ndim-1))
        terms.append(F.relu(target.mean(target_dims).norm(dim=-1)-bound).square().mean())
    return torch.stack(terms).mean()


def local_orientation_loss(model, direct, radius=0.02):
    """Penalize reversed local displacements at every observed R-input scale.

    Deterministic circle quadrature (16 directions) in 2D, +/- in 1D;
    higher dimensions use axes and pair diagonals. No training RNG consumed.
    Radius is a fraction of detached per-scale RMS latent spread. Source
    locations remain differentiable, so joint training can adjust Q/C too.
    This coordinate-dependent prior does not enforce contraction or symmetry.
    """
    import math
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError('Orientation radius must be finite and positive')
    d = direct.shape[-1]
    if d == 2:
        angle = torch.arange(16, device=direct.device, dtype=direct.dtype)*(2*math.pi/16)
        directions = torch.stack((angle.cos(), angle.sin()), -1)
    else:
        eye = torch.eye(d, device=direct.device, dtype=direct.dtype)
        vectors = [eye, -eye]
        for i in range(d):
            for j in range(i+1, d):
                for sign in (-1, 1):
                    v = (eye[i]+sign*eye[j])/math.sqrt(2)
                    vectors.extend((v[None], -v[None]))
        directions = torch.cat(vectors)
    source = direct[:, :-1]
    spread = source.detach().var(dim=0, unbiased=False).sum(-1).sqrt().clamp_min(1e-3)
    delta = radius*spread[None, :, None, None]*directions[None, None]
    change = model.advance_theory(source[:, :, None]+delta)-model.advance_theory(source)[:, :, None]
    cosine = (delta*change).sum(-1)/(delta.norm(dim=-1)*change.norm(dim=-1)+1e-12)
    penalties = F.relu(-cosine).square()
    return penalties.mean(), penalties.mean(dim=(0, 2))


def binary_kernel_variance_floor(levels, ratio=0.25, *, per_ensemble=False):
    """Allow biased binary marginals; penalize near-constant batch outputs.

    Require 1-m_out^2 >= ratio*(1-m_in^2), with a detached input
    reference. A hinge on |m_out| retains an STE gradient at constant outputs.
    This is a soft anti-collapse heuristic, not a magnetization constraint.
    per_ensemble=True checks each ensemble before averaging, preventing
    cancellation across different ensembles. False preserves historical runs.
    """
    if not 0 < ratio <= 1:
        raise ValueError('variance floor ratio must be in (0, 1]')
    terms = []
    for source, target in zip(levels[:-1], levels[1:]):
        axes=tuple(range(1,source.ndim)) if per_ensemble else tuple(range(source.ndim))
        target_axes=tuple(range(1,target.ndim)) if per_ensemble else tuple(range(target.ndim))
        variance = (1-source.detach().mean(axes).square()).clamp(0, 1)
        bound = (1-ratio*variance).sqrt()
        terms.append(F.relu(target.mean(target_axes).abs()-bound).square().mean())
    return torch.stack(terms).mean()


def data_geometry_objective(model, levels, first, second, pattern_fn, weights,
                            geometry_scale, noise_multiplier, rollout_weight=5.0,
                            kernel_noncollapse_mode='balance', kernel_variance_ratio=0.25,
                            coordinate_reduction='sum', geometry_options=None,
                            balanced_reconstruction_weight=0., closure_scale_weights=None,
                            closure_objective='one_step_and_rollout'):
    """Equal-scale distribution geometry replaces anchor spread + hinge floor."""
    midpoint = levels[0].shape[1] // 2
    pa = [pattern_fn(level[:, :midpoint]) for level in levels]
    pb = [pattern_fn(level[:, midpoint:]) for level in levels]
    kernel = (binary_kernel_variance_floor(levels, kernel_variance_ratio,
                  per_ensemble=kernel_noncollapse_mode=='variance_floor_ensemble')
              if kernel_noncollapse_mode in ('variance_floor','variance_floor_ensemble')
              else binary_kernel_noncollapse(levels))
    if levels[0].shape[-1] > 1:
        if kernel_noncollapse_mode not in ('variance_floor','variance_floor_ensemble'):
            raise ValueError('Categorical fields require variance_floor')
        kernel = categorical_kernel_variance_floor(levels, kernel_variance_ratio,
            per_ensemble=kernel_noncollapse_mode=='variance_floor_ensemble')
    return data_geometry_statistics_objective(model, first, second, pa, pb, kernel,
                  weights, geometry_scale, noise_multiplier, rollout_weight, coordinate_reduction,
                  geometry_options=geometry_options,
                  balanced_reconstruction_weight=balanced_reconstruction_weight,
                  closure_scale_weights=closure_scale_weights,
                  **({'closure_objective': closure_objective}
                     if closure_objective != 'one_step_and_rollout' else {}))


def distribution_reconstruction_loss(decoder, theory, target):
    """JS divergence in nats to detached local-pattern frequencies, Q/decoder only."""
    target = target.detach()
    predicted = decoder(theory).softmax(-1)
    if predicted.shape != target.shape:
        raise ValueError('Distribution reconstruction shape mismatch')
    mixture = .5 * (target + predicted)
    log_mix = mixture.clamp_min(1e-15).log()
    return .5 * ((target * (target.clamp_min(1e-15).log() - log_mix)).sum(-1)
                 + (predicted * (predicted.clamp_min(1e-15).log() - log_mix)).sum(-1)).mean()


def data_geometry_statistics_objective(model, first, second, pattern_first, pattern_second,
                                       kernel, weights, geometry_scale, noise_multiplier,
                                       rollout_weight=5.0, coordinate_reduction='sum', geometry_options=None,
                                       balanced_reconstruction_weight=0., closure_scale_weights=None,
                                       closure_objective='one_step_and_rollout'):
    """Shared objective algebra, also usable with exact fixed-C sufficient statistics."""
    validate_closure_objective(closure_objective, weights, rollout_weight,
        closure_scale_weights=closure_scale_weights)
    direct = .5 * (first + second)
    geometry, extra, balanced = [], {}, []
    if balanced_reconstruction_weight:
        from .balanced_reconstruction import balanced_reconstruction_loss
        metrics_by_scale = getattr(model, 'balanced_reconstruction_metrics', None)
        if metrics_by_scale is None or len(metrics_by_scale) != len(pattern_first):
            raise ValueError('Fit TRAIN-only balanced metrics before evaluating the balanced objective')
        if getattr(model.theory_encoder, 'distribution_reconstruction_mode', '') != 'affine_mse':
            raise ValueError('Balanced objective requires an affine probability decoder')
    for k, (pa, pb) in enumerate(zip(pattern_first, pattern_second)):
        decoder = getattr(model.theory_encoder, 'distribution_decoder', None)
        if getattr(model.theory_encoder, 'distribution_reconstruction_mode', '') == 'ensemble_vae_cross_subset':
            term, metrics = model.theory_encoder.cross_subset_objective(pa, pb, scale_index=k)
        elif decoder is not None:
            if getattr(model.theory_encoder, 'distribution_reconstruction_mode', '') == 'affine_mse':
                from .representation import linear_distribution_reconstruction_loss
                term = linear_distribution_reconstruction_loss(decoder, direct[:, k], .5*(pa+pb), mode='squared',
                    validate=not getattr(model, 'fast_validated_statistics', False))
                metrics = {'reconstruction_mse': term}
            elif getattr(model.theory_encoder, 'distribution_reconstruction_mode', '') == 'probability_mse':
                from .representation import probability_distribution_reconstruction_loss
                term = probability_distribution_reconstruction_loss(decoder, direct[:, k], .5*(pa+pb))
                metrics = {'reconstruction_mse': term}
            else:
                term = distribution_reconstruction_loss(decoder, direct[:, k], .5*(pa+pb))
                metrics = {'reconstruction_js': term}
        else:
            term, metrics = distribution_geometry_loss(direct[:, k], pa, pb,
                            scale=geometry_scale, noise_multiplier=noise_multiplier, **(geometry_options or {}))
        geometry.append(term)
        extra[f'data_geometry_level_{k}'] = term
        for name, value in metrics.items():
            extra[f'{name}_level_{k}'] = value
        extra[f'pairwise_level_{k}'] = mean_pairwise_square(direct[:, k])
        if balanced_reconstruction_weight:
            value = balanced_reconstruction_loss(decoder, direct[:, k], .5*(pa+pb), metrics_by_scale[k])
            balanced.append(value)
            extra[f'balanced_reconstruction_level_{k}'] = value
    # Preserve old subset-consistency and C/R weights, removing only the
    # old unit-variance/covariance/correlation anchor and fixed separation.
    anchor = 10. * latent_squared_error(first[:, 0], second[:, 0], coordinate_reduction)
    consistency = latent_squared_error(first[:, 1:], second[:, 1:], coordinate_reduction)
    separation = torch.stack(geometry).mean()
    predicted = model.advance_theory(direct[:, :-1])
    commutator, step_levels, unweighted_step = one_step_closure_loss(
        predicted, direct[:, 1:], coordinate_reduction, closure_scale_weights)
    rollout, rollout_levels = multistep_closure_loss(
        model, direct, coordinate_reduction, closure_scale_weights)
    extra['unweighted_commutator'] = unweighted_step
    extra['unweighted_rollout'] = torch.stack(rollout_levels).mean()
    extra['weighted_commutator'] = commutator
    extra['weighted_rollout'] = rollout
    extra['rollout'] = rollout
    for k, term in enumerate(step_levels, start=1):
        extra[f'one_step_level_{k}'] = term
    for k, term in enumerate(rollout_levels, start=1):
        extra[f'rollout_level_{k}'] = term
    scale_weights = validate_closure_scale_weights(closure_scale_weights, len(step_levels))
    if scale_weights is not None:
        largest = max(scale_weights)
        total_weight = sum(value/largest for value in scale_weights)
        for k, value in enumerate(scale_weights, start=1):
            extra[f'closure_scale_weight_{k}'] = direct.new_tensor((value/largest)/total_weight)
    if closure_objective == 'one_step_and_rollout':
        # Do not regroup arithmetic on the historical/default execution path.
        total = (weights.anchor * anchor + weights.coarse_consistency * consistency
                 + weights.kernel_noncollapse * kernel + weights.separation * separation
                 + weights.commutator * commutator + rollout_weight * rollout)
    else:
        all_start, target_errors, pair_errors = all_start_anchor_closure_loss(
            model, direct, coordinate_reduction, closure_scale_weights)
        extra['all_start_anchor'] = all_start
        extra['closure_total'] = (weights.commutator + rollout_weight) * all_start
        extra.update({f'all_start_target_{target}': error
                      for target, error in enumerate(target_errors, start=1)})
        extra.update({f'all_start_pair_{start}_{target}': error
                      for (start, target), error in pair_errors.items()})
        total = (weights.anchor * anchor + weights.coarse_consistency * consistency
                 + weights.kernel_noncollapse * kernel + weights.separation * separation
                 + extra['closure_total'])
    if balanced_reconstruction_weight:
        extra['balanced_reconstruction'] = torch.stack(balanced).mean()
        total = total + balanced_reconstruction_weight * extra['balanced_reconstruction']
    return total, {'anchor': anchor, 'coarse_consistency': consistency,
                   'kernel_noncollapse': kernel, 'separation': separation,
                   'commutator': commutator, **extra}


def qcr_objective(
    model,
    microscopic: torch.Tensor,
    *,
    pattern_fn,
    weights: QCRWeights = QCRWeights(anchor=2.0, separation=10.0),
    objective_mode: str = 'data_geometry',
    geometry_scale: float = 4.0,
    noise_multiplier: float = 2.0,
    rollout_weight: float = 5.0,
    closure_scale_weights: tuple[float, ...] | None = None,
    closure_objective: str = 'one_step_and_rollout',
    isotropy_weight: float = 0.0,
    tangent_weight: float = 0.0,
    orientation_weight: float = 0.0,
    orientation_radius: float = 0.02,
    kernel_sign_weight: float = 0.0,
    kernel_neutral_weight: float = 0.0,
    kernel_neutral_max_patterns: int = 65536,
    closure_metric: str = 'euclidean',
    kernel_noncollapse_mode: str = 'balance',
    kernel_variance_ratio: float = 0.25,
    coordinate_reduction: str = 'sum',
    geometry_mode: str = 'absolute',
    geometry_distance: str = 'euclidean',
    geometry_bounds: str = 'interval',
    geometry_relative_floor: float = .05,
    geometry_max_inverse_weight: float = 100.,
    geometry_neighbors: int = 4,
    geometry_local_fraction: float = .5,
    geometry_tail_mix: float = 0.,
    geometry_tail_add_weight: float = 0.,
    geometry_tail_fraction: float = .1,
    geometry_tail_min_reliability: float = .5,
    balanced_reconstruction_weight: float = 0.,
    levels_sink: list[torch.Tensor] | None = None,
    closure_targets_sink: list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Joint all-scale distribution geometry and single/multistep closure."""
    if objective_mode != "data_geometry":
        raise ValueError("Legacy loss moved to archive.qcr_ablations.legacy_objective")
    if coordinate_reduction not in ('mean', 'sum'):
        raise ValueError('coordinate_reduction must be mean or sum')
    validate_closure_scale_weights(closure_scale_weights)
    validate_closure_objective(closure_objective, weights, rollout_weight,
        closure_metric, closure_scale_weights)
    if rollout_weight < 0:
        raise ValueError('rollout_weight must be nonnegative')
    if isotropy_weight < 0:
        raise ValueError('isotropy_weight must be nonnegative')
    import math
    if not math.isfinite(balanced_reconstruction_weight) or balanced_reconstruction_weight < 0:
        raise ValueError('Invalid balanced reconstruction weight')
    if not math.isfinite(kernel_sign_weight) or kernel_sign_weight < 0:
        raise ValueError('Invalid kernel sign weight')
    if not math.isfinite(kernel_neutral_weight) or kernel_neutral_weight < 0:
        raise ValueError('Invalid kernel neutral weight')
    if not isinstance(kernel_neutral_max_patterns,int) or kernel_neutral_max_patterns < 1:
        raise ValueError('Invalid neutral reference enumeration limit')
    if not math.isfinite(orientation_weight) or orientation_weight < 0 or not math.isfinite(orientation_radius) or orientation_radius <= 0:
        raise ValueError('Invalid orientation settings')
    if tangent_weight < 0 or closure_metric not in ('euclidean','noise_aware'):
        raise ValueError('Invalid tangent weight or closure metric')
    if kernel_noncollapse_mode not in ('balance', 'variance_floor', 'variance_floor_ensemble'):
        raise ValueError('Unknown kernel noncollapse mode')
    samples = microscopic.shape[1]
    midpoint = samples // 2
    if midpoint < 1 or samples % 2:
        raise ValueError("QCR batches need an even number of samples")

    levels = model.coarse_grain(microscopic)
    if levels_sink is not None:
        # Opt-in auxiliary objectives reuse this exact C trajectory/RNG draw.
        # They must detach it before constructing their independent Q graph.
        levels_sink.extend(levels)
    # Share the estimator with evaluation; original levels still carry C's
    # variance gradients, while the independent Q route detaches C inputs.
    first, second = split_theory_views(model, levels)
    if closure_targets_sink is not None:
        # Snapshot this batch's exact estimator, before any Q/C/R update.
        # No second Q/C forward, RNG draw, labels, or cross-batch cache.
        closure_targets_sink.append((.5 * (first + second)).detach())
    total, terms = data_geometry_objective(model, levels, first, second, pattern_fn,
                                   weights, geometry_scale, noise_multiplier, rollout_weight,
                                   kernel_noncollapse_mode, kernel_variance_ratio, coordinate_reduction,
                                   geometry_options=dict(mode=geometry_mode,distance_metric=geometry_distance,bounds=geometry_bounds,relative_floor=geometry_relative_floor,
                                       max_inverse_weight=geometry_max_inverse_weight,neighbors=geometry_neighbors,
                                       local_fraction=geometry_local_fraction,tail_mix=geometry_tail_mix,
                                       tail_add_weight=geometry_tail_add_weight,
                                       tail_fraction=geometry_tail_fraction,tail_min_reliability=geometry_tail_min_reliability),
                                   balanced_reconstruction_weight=balanced_reconstruction_weight,
                                   **({'closure_objective': closure_objective}
                                      if closure_objective != 'one_step_and_rollout' else {}),
                                   **({'closure_scale_weights': closure_scale_weights}
                                      if closure_scale_weights is not None else {}))
    if tangent_weight or closure_metric != 'euclidean':
        direct = .5*(first+second)
        predicted = model.advance_theory(direct[:,:-1])
        if tangent_weight:
            tangent_terms = []
            for k in range(len(levels)-1):
                pa = pattern_fn(levels[k][:,:midpoint])
                pb = pattern_fn(levels[k][:,midpoint:])
                tangent_terms.append(local_tangent_loss(direct[:,k],predicted[:,k],direct[:,k+1],
                    pa,pb,geometry_neighbors,noise_multiplier))
            tangent = torch.stack(tangent_terms).mean()
            total = total+tangent_weight*tangent
            terms['tangent'] = tangent
        if closure_metric == 'noise_aware':
            step_terms, rollout_terms = [], []
            recurrent = direct[:,0]
            for k in range(1,len(levels)):
                precision = noise_aware_precision(first[:,k],second[:,k])
                step_terms.append(metric_error(predicted[:,k-1],direct[:,k],precision,coordinate_reduction))
                recurrent = model.advance_theory(recurrent)
                rollout_terms.append(metric_error(recurrent,direct[:,k],precision,coordinate_reduction))
            step = closure_scale_mean(step_terms, closure_scale_weights)
            rollout = closure_scale_mean(rollout_terms, closure_scale_weights)
            total = total+weights.commutator*(step-terms['commutator'])+rollout_weight*(rollout-terms['rollout'])
            terms['euclidean_commutator'] = terms['commutator']
            terms['euclidean_rollout'] = terms['rollout']
            terms['commutator'],terms['rollout'] = step,rollout
            terms['weighted_commutator'], terms['weighted_rollout'] = step, rollout
            terms['unweighted_commutator'] = torch.stack(step_terms).mean()
            terms['unweighted_rollout'] = torch.stack(rollout_terms).mean()
            # Existing rollout_level_k metrics remain the raw Euclidean
            # quantities. Explicit metric-prefixed fields avoid changing them.
            for k, (step_term, rollout_term) in enumerate(zip(step_terms, rollout_terms), start=1):
                terms[f'noise_aware_one_step_level_{k}'] = step_term
                terms[f'noise_aware_rollout_level_{k}'] = rollout_term
    if kernel_neutral_weight:
        from .kernel_regularizers import kernel_neutral_loss
        neutral = kernel_neutral_loss(model.rg_kernel, max_patterns=kernel_neutral_max_patterns)
        total = total+kernel_neutral_weight*neutral
        terms['kernel_neutral'] = neutral
    if kernel_sign_weight:
        from .adapters import CategoricalFieldHead
        sign = (categorical_kernel_preserve_loss(model.rg_kernel)
                if isinstance(getattr(model.rg_kernel,'field_head',None),CategoricalFieldHead)
                else binary_kernel_sign_loss(model.rg_kernel))
        total = total+kernel_sign_weight*sign
        terms['kernel_sign'] = sign
    if orientation_weight:
        orientation, by_scale = local_orientation_loss(model, .5*(first+second), orientation_radius)
        total = total+orientation_weight*orientation
        terms['orientation'] = orientation
        terms.update({f'orientation_level_{k}': value for k, value in enumerate(by_scale)})
    if isotropy_weight:
        g = .5*(first[:,0]+second[:,0])
        centered = g-g.mean(0)
        covariance = centered.T@centered/len(g)
        identity = torch.eye(g.shape[-1], dtype=g.dtype, device=g.device)
        isotropy = (covariance-identity).square().sum()/g.shape[-1]
        total = total+isotropy_weight*isotropy
        terms['isotropy'] = isotropy
    return total, terms
