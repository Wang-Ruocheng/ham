"""Default Q/C/R-independent training: empirical MI + variance + color C loss."""
import argparse
import copy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import time
import torch
from recurrent_rg.checkpoints import build_model, checkpoint_payload, qcr_state_dict, load_run, load_model_state
from recurrent_rg.checkpoints import isolated_rng, restore_checkpoint_rng
from recurrent_rg.config import LossConfig, RunConfig
from recurrent_rg.runner import prepared_data, file_fingerprint, evaluate_run
from recurrent_rg.training import evaluate, separate_qc_r_gradients
from recurrent_rg.losses import qcr_objective
from recurrent_rg.rsmi import EnvironmentCritic, region_offsets, sample_regions, kernel_probabilities, enumerated_infonce
from recurrent_rg.rsmi import binary_chain_joint_histograms, histogram_mutual_information
from recurrent_rg.rsmi import best_binary_pair_kernel, initialize_kernel_from_table
from recurrent_rg.rsmi import categorical_kernel_probabilities
from recurrent_rg.empirical_mi import binary_lattice_mi_statistics,best_deterministic_binary_rule,categorical_lattice_mi_statistics,optimize_categorical_rule
from experiments.potts_frozen_qc_continue import pattern_function
from recurrent_rg.empirical_mi import categorical_local_pair_mi_statistics


ISING_B0_ABLATIONS = (
    'none', 'c_no_mi', 'c_no_variance', 'c_no_color',
    'q_no_reconstruction', 'q_no_subset', 'r_no_rollout')


def validate_ising_b0_ablation(args, cfg=None):
    """Narrow, explicit campaign gate; absent marker preserves historical paths.

    This is not a general recipe override. Exactly the named term(s) may be
    zeroed, while independent gradients, random C and label-free training stay
    locked. Configurations remain authoritative: this function never rewrites
    loss weights or silently activates an ablation.
    """
    name = getattr(args, 'ising_b0_ablation', None)
    if name is None:
        return
    if name not in ISING_B0_ABLATIONS:
        raise ValueError('Unknown ising-b0-ablation')
    conflicts = ('allow_nondefault_recipe', 'source', 'resume_checkpoint',
        'mi_initialize', 'freeze_c_checkpoint', 'affine_pca_init', 'theory_dim',
        'q_coupling_weight', 'q_future_weight', 'q_future_control', 'q_future_pairs',
        'q_future_normalizer', 'q_only_weight_sweep', 'q_only_candidates',
        'q_balanced_reconstruction', 'q_last_layer_audit', 'q_head_structure_audit',
        'q_feature_bottleneck', 'q_original_plateau', 'q_initialization_pilot',
        'q_unfolding', 'r_only_continue', 'r_only_refit', 'r_only_plateau',
        'joint_train_plateau', 'frozen_cache', 'all_start_closure_ablation')
    if args.config is None or any(getattr(args, key, None) for key in conflicts):
        raise ValueError('ising-b0-ablation requires a fresh standalone config and no other ablations or overrides')
    expected = dict(estimator='histogram', environment='full_shell',
        rsmi_weight=0. if name == 'c_no_mi' else 2.,
        variance_weight=0. if name == 'c_no_variance' else 2.,
        color_weight=0. if name == 'c_no_color' else 1., variance_ratio=1.,
        variance_per_ensemble=True, independent_c=True)
    changed = [key for key, value in expected.items() if getattr(args, key) != value]
    if name == 'c_no_mi' and (args.mi_cache_dir or args.mi_test_cpu):
        changed.append('unused_mi_cache_or_placement')
    if changed:
        raise ValueError('ising-b0-ablation must change only its named loss: ' + ', '.join(changed))
    if cfg is None:
        return
    model, training, data = cfg.model, cfg.training, cfg.data
    required_model = dict(dimension=2, theory_dim=5, block_factor=3, width=3,
        categorical_states=0, kernel_layout='nonoverlapping', fixed_kernel=False,
        binary_lookup=True, kernel_initialization='random', kernel_sampling='bernoulli',
        kernel_temperature=1., affine_reconstruction=True, reconstruction_hidden=0,
        theory_head='nonlinear')
    changed = [f'model.{key}' for key, value in required_model.items()
               if getattr(model, key) != value]
    if cfg.schema_version != 5:
        changed.append('schema_version')
    if data.physical_model != 'ising2d' or tuple(data.coupling_names) != ('K',):
        changed.append('B0_K_only_data')
    if set(training.train_modules) != {'Q', 'C', 'R'}:
        changed.append('train_modules')
    # The campaign freezes one complete policy in its plan; both documented
    # constant Adam and the existing deterministic multistep policy are valid.
    # This marker does not opt into adaptive/loss-dependent optimization.
    training.validate_learning_rate()
    if training.learning_rate != .001 or training.lr_schedule not in ('constant', 'multistep'):
        changed.append('Adam_learning_rate_policy')
    # C auxiliaries are CLI-authoritative in this driver; compare the effective
    # objective rather than obsolete serialized pre-override C defaults.
    effective = replace(cfg.loss, kernel_sign_weight=args.color_weight,
        kernel_neutral_weight=0., kernel_noncollapse_mode='variance_floor_ensemble',
        kernel_variance_ratio=1.,
        weights=replace(cfg.loss.weights, kernel_noncollapse=args.variance_weight))
    expected_loss = LossConfig(closure_objective='one_step_and_rollout',
        kernel_sign_weight=args.color_weight,
        kernel_noncollapse_mode='variance_floor_ensemble', kernel_variance_ratio=1.)
    expected_weights = replace(expected_loss.weights, kernel_noncollapse=args.variance_weight,
        separation=0. if name == 'q_no_reconstruction' else 10.,
        anchor=0. if name == 'q_no_subset' else 2.,
        coarse_consistency=0. if name == 'q_no_subset' else 5.)
    expected_loss = replace(expected_loss, weights=expected_weights,
        rollout_weight=0. if name == 'r_no_rollout' else 5.)
    scale_weights = effective.closure_scale_weights
    if scale_weights is not None:
        from recurrent_rg.config import validate_closure_scale_weights
        values = validate_closure_scale_weights(scale_weights, training.max_rg_steps)
        if not all(value == values[0] and value > 0 for value in values):
            changed.append('uniform_closure_scale_weights')
    effective = replace(effective, closure_scale_weights=None)
    actual_loss = effective.objective_kwargs(historical_metadata=True)
    expected_objective = expected_loss.objective_kwargs(historical_metadata=True)
    changed.extend(f'loss.{key}' for key in expected_objective
                   if actual_loss[key] != expected_objective[key])
    if cfg.loss.kernel_distinct_alpha:
        changed.append('retired_kernel_distinct_alpha')
    if cfg.loss.kernel_neutral_weight:
        changed.append('kernel_neutral_weight')
    if cfg.loss.closure_objective != 'one_step_and_rollout':
        changed.append('closure_objective')
    if changed:
        raise ValueError('ising-b0-ablation locked recipe changed: ' + ', '.join(changed))


def ising_b0_ablation_contract(args, cfg):
    """Public provenance, including zero weights rather than claimed zero MI."""
    name = getattr(args, 'ising_b0_ablation', None)
    if name is None:
        return None
    validate_ising_b0_ablation(args, cfg)
    removed = dict(none=[], c_no_mi=['C empirical histogram MI'],
        c_no_variance=['C per-ensemble variance floor'],
        c_no_color=['C uniform-block color-preservation NLL'],
        q_no_reconstruction=['Q shared affine motif reconstruction'],
        q_no_subset=['Q microscopic subset consistency', 'Q coarse subset consistency'],
        r_no_rollout=['R microscopic-start multistep closure'])[name]
    return dict(name=name, removed_terms=removed, baseline=name == 'none',
        labels_used_in_training=False, random_trainable_C=True,
        gradient_routing='Q representation only; C own objective only; R closure only',
        closure_target_scale_weights='uniform; each branch averaged over target scales',
        effective_weights=dict(C_MI=-args.rsmi_weight, C_variance=args.variance_weight,
            C_color=args.color_weight, Q_reconstruction=cfg.loss.weights.separation,
            Q_microscopic_subset=10.*cfg.loss.weights.anchor,
            Q_coarse_subset=cfg.loss.weights.coarse_consistency,
            R_one_step=cfg.loss.weights.commutator, R_multistep=cfg.loss.rollout_weight),
        learning_rate_policy=dict(schedule=cfg.training.lr_schedule,
            base_learning_rate=cfg.training.learning_rate,
            milestones=list(cfg.training.lr_milestones), gamma=cfg.training.lr_gamma),
        MI_evaluated=bool(args.rsmi_weight),
        caution='Named zero weights remove objectives, not evidence that their measured values vanish; B0 data provenance must be fingerprint-verified separately.')


def independent_component_gradients(total,closure,c_objective,q,c,r,critic=()):
    """Each trainable component gets its own objective, on one forward graph."""
    c=list(c)
    cgrads=torch.autograd.grad(c_objective,c,retain_graph=True,allow_unused=True)
    separate_qc_r_gradients(total,closure,list(q)+list(critic),list(r))
    for parameter,gradient in zip(c,cgrads):parameter.grad=gradient


def validate_all_start_closure_ablation(args, cfg=None):
    """Narrow b=2, four-level, label-free paired closure experiment gate."""
    enabled = getattr(args, 'all_start_closure_ablation', False)
    if not enabled:
        return
    conflicts = ('allow_nondefault_recipe', 'source', 'resume_checkpoint',
        'ising_b0_ablation', 'mi_initialize', 'freeze_c_checkpoint',
        'affine_pca_init', 'theory_dim', 'q_coupling_weight', 'q_future_weight',
        'q_future_control', 'q_future_pairs', 'q_future_normalizer',
        'q_only_weight_sweep', 'q_only_candidates', 'q_balanced_reconstruction',
        'q_last_layer_audit', 'q_head_structure_audit', 'q_feature_bottleneck',
        'q_original_plateau', 'q_initialization_pilot', 'q_unfolding',
        'r_only_continue', 'r_only_refit', 'r_only_plateau',
        'joint_train_plateau', 'frozen_cache')
    if args.config is None or any(getattr(args, key, None) for key in conflicts):
        raise ValueError('all-start closure ablation requires fresh standalone paired configs without other ablations')
    expected = dict(estimator='histogram', environment='full_shell', rsmi_weight=2.,
        variance_weight=2., color_weight=1., variance_ratio=1.,
        variance_per_ensemble=True, independent_c=True)
    changed = [key for key, value in expected.items() if getattr(args, key) != value]
    if cfg is None:
        if changed:
            raise ValueError('all-start closure ablation locked recipe changed: ' + ', '.join(changed))
        return
    model_expected = dict(dimension=2, theory_dim=5, hidden=96, width=3,
        block_factor=2, categorical_states=0, kernel_layout='nonoverlapping',
        fixed_kernel=False, binary_lookup=True, kernel_initialization='random',
        kernel_sampling='bernoulli', kernel_temperature=1.,
        affine_reconstruction=True, reconstruction_hidden=0,
        theory_head='nonlinear', local_features=12)
    changed.extend('model.' + key for key, value in model_expected.items()
                   if getattr(cfg.model, key) != value)
    if cfg.schema_version != 5:
        changed.append('schema_version')
    if (cfg.data.physical_model != 'ising2d'
            or tuple(cfg.data.coupling_names) != ('K',)):
        changed.append('B0_K_only_data')
    if (set(cfg.training.train_modules) != {'Q', 'C', 'R'}
            or cfg.training.gradient_mode != 'qc_r_separate'):
        changed.append('independent_QCR_gradients')
    if cfg.training.max_rg_steps != 3 or cfg.evaluation.max_rg_steps != 3:
        changed.append('four_scale_points')
    cfg.training.validate_learning_rate()
    if (cfg.training.learning_rate != .001
            or cfg.training.lr_schedule not in ('constant', 'multistep')):
        changed.append('Adam_learning_rate_policy')
    effective = replace(cfg.loss, closure_objective='one_step_and_rollout',
        closure_scale_weights=None, kernel_sign_weight=args.color_weight,
        kernel_neutral_weight=0., kernel_noncollapse_mode='variance_floor_ensemble',
        kernel_variance_ratio=1.,
        weights=replace(cfg.loss.weights, kernel_noncollapse=args.variance_weight))
    expected_loss = LossConfig(closure_objective='one_step_and_rollout', kernel_sign_weight=1.,
        kernel_noncollapse_mode='variance_floor_ensemble', kernel_variance_ratio=1.)
    actual = effective.objective_kwargs(historical_metadata=True)
    changed.extend('loss.' + key for key, value in
        expected_loss.objective_kwargs(historical_metadata=True).items()
        if actual[key] != value)
    cfg.loss.objective_kwargs()  # Validate the selected closure formula itself.
    from recurrent_rg.config import validate_closure_scale_weights
    values = validate_closure_scale_weights(cfg.loss.closure_scale_weights, 3)
    if values is not None and len(set(values)) != 1:
        changed.append('uniform_closure_scale_weights')
    if cfg.loss.kernel_distinct_alpha or cfg.loss.kernel_neutral_weight:
        changed.append('retired_or_extra_C_objectives')
    if changed:
        raise ValueError('all-start closure ablation locked recipe changed: ' + ', '.join(changed))


def evaluate_information(model,critic,x,*,flip,pairs=64,repeats=8):
    gen=torch.Generator().manual_seed(818)
    values=[];shuffled=[]
    with torch.no_grad():
        for _ in range(repeats):
            v,e=sample_regions(x,min(32,len(x)),pairs,gen,buffer=1,flip=flip)
            probabilities=kernel_probabilities(model.rg_kernel,v)
            scores=critic(e)
            values.append(enumerated_infonce(probabilities,scores))
            perm=torch.stack([torch.randperm(pairs,generator=gen) for _ in range(len(e))])
            shuffled.append(enumerated_infonce(probabilities,scores.gather(1,perm[:,:,None].expand_as(scores))))
    return float(torch.cat(values).mean()),float(torch.cat(shuffled).mean())


def microscopic_coupling_loss(encoder, batch, labels):
    """Explicit ablation only; labels never become targets for coarse scales."""
    predicted=encoder(batch)
    target=labels.detach().to(predicted).reshape(len(predicted),-1)
    if target.shape!=predicted.shape:raise ValueError('Coupling target shape mismatch')
    return (predicted-target).square().sum(-1).mean()


def validate_ensemble_vae_training(args):
    """Explicit microscopic supervision is compatible with detached VAE Q.

    The existing coupling-loss path changes Q only. Future-observable losses
    and coupled C gradients remain unsupported; zero weight is unchanged.
    """
    if args.q_future_weight or not args.independent_c:
        raise ValueError('Weak-KL ensemble Q requires independent training without future-observable loss')


def closure_scale_contract(cfg):
    """Record the explicit scale-weight ablation without altering gradient routing."""
    values = getattr(cfg.loss, 'closure_scale_weights', None)
    if values is None:
        return None
    cfg.validate()
    values = list(values)
    largest = max(values)
    scaled = [value / largest for value in values]
    mass = sum(scaled)
    if cfg.loss.closure_objective == 'all_start_anchor':
        return dict(weights=values, normalized_weights=[value / mass for value in scaled],
            target_levels=list(range(1, len(values) + 1)),
            application=['all_start_anchor'],
            normalization='Uniform mean over target levels; mean over all earlier starts within each target',
            total_weight=cfg.loss.weights.commutator + cfg.loss.rollout_weight,
            labels_used=False, gradient_routing='R only; no closure gradients to Q or C')
    return dict(weights=values, normalized_weights=[value / mass for value in scaled],
        target_levels=list(range(1, len(values) + 1)),
        application=['one_step_commutator', 'microscopic_start_rollout'],
        normalization='sum(w_k * loss_k) / sum(w_k); original outer closure weights unchanged',
        one_step='R(g[k-1]) versus g[k]', rollout='R^k(g[0]) versus g[k]',
        labels_used=False, gradient_routing='R only; no closure gradients to Q or C')


def closure_objective_contract(cfg):
    """Describe the actual routed R objective, independently of historical logs."""
    mode = cfg.loss.closure_objective
    if mode == 'all_start_anchor':
        coefficient = cfg.loss.weights.commutator + cfg.loss.rollout_weight
        return dict(objective=mode, total_weight=coefficient,
            formula='weight/S * sum_(t=1..S) (1/t) sum_(s=0..t-1) ||R^(t-s)(sg g_s)-sg g_t||^2',
            target_scale_weights='uniform', starts_within_target='uniform over all earlier scales',
            prediction_prediction_penalty=False, labels_used=False,
            gradient_routing='R only; all recurrent R calls retain gradients; Q/C unchanged',
            gradient_R=f'grad({coefficient:g} L_all_start_anchor); no feedback to QC',
            interpretation='Koopman-inspired multi-horizon consistency; nonlinear R, not a finite-dimensional linear Koopman operator')
    return dict(objective=mode,
        one_step_weight=cfg.loss.weights.commutator, rollout_weight=cfg.loss.rollout_weight,
        formula='commutator_weight * mean_t e[t-1,t] + rollout_weight * mean_t e[0,t]',
        target_scale_weights=('configured' if cfg.loss.closure_scale_weights is not None else 'uniform'),
        prediction_prediction_penalty=False, labels_used=False,
        gradient_routing='R only; no feedback to Q/C',
        gradient_R=(f'grad({cfg.loss.weights.commutator:g} L_commutator + '
                    f'{cfg.loss.rollout_weight:g} L_rollout); no feedback to QC'))


def validate_future_options(args, cfg=None):
    """Explicit opt-in only; weight zero retains the original trainer exactly."""
    weight=args.q_future_weight
    if not math.isfinite(weight) or weight<0:
        raise ValueError('q-future-weight must be finite and nonnegative')
    if args.q_future_control and weight:
        raise ValueError('Paired control must have zero q-future-weight and no auxiliary predictor')
    if not weight and not args.q_future_control:
        if args.q_future_pairs is not None or args.q_future_normalizer is not None:
            raise ValueError('Future files require positive q-future-weight or explicit paired control')
        return
    conflicts=('source','q_coupling_weight','allow_nondefault_recipe','mi_initialize',
        'freeze_c_checkpoint','affine_pca_init','q_only_weight_sweep','q_only_candidates',
        'q_balanced_reconstruction','q_last_layer_audit','q_head_structure_audit',
        'q_feature_bottleneck','q_original_plateau','q_initialization_pilot','q_unfolding',
        'r_only_continue','r_only_refit','r_only_plateau','joint_train_plateau','frozen_cache')
    if (any(getattr(args,key) for key in conflicts) or not args.independent_c or
            args.estimator!='histogram' or args.environment!='full_shell' or args.config is None):
        raise ValueError('Future physics supports standalone default independent QCR only; no labels/other ablations')
    if args.q_future_hidden!=32 or args.q_future_pairs is None or (weight and args.q_future_normalizer is None):
        raise ValueError('Future physics requires hidden32, explicit paired-chain identities and frozen TRAIN normalizer')
    if cfg is not None:
        w=cfg.loss.weights
        if (cfg.schema_version!=5 or cfg.model.dimension!=2 or cfg.model.categorical_states or
                cfg.model.block_factor!=3 or cfg.model.kernel_sampling!='bernoulli' or
                cfg.model.width!=3 or cfg.model.theory_head!='nonlinear' or not cfg.model.binary_lookup or
                not cfg.model.affine_reconstruction or cfg.training.max_rg_steps!=2 or
                cfg.loss.balanced_reconstruction_weight or cfg.loss.isotropy_weight or
                cfg.loss.tangent_weight or cfg.loss.orientation_weight or
                cfg.loss.closure_metric!='euclidean' or cfg.loss.coordinate_reduction!='sum' or
                (w.anchor,w.coarse_consistency,w.separation,w.commutator,cfg.loss.rollout_weight)!=(2.,5.,10.,5.,5.) or
                cfg.data.train_pool_replicas!=1 or cfg.data.test_pool_replicas!=1 or
                cfg.training.system_batch<2 or cfg.training.system_batch%2 or cfg.evaluation.systems<2):
            raise ValueError('Future physics requires schema5 binary 2D b3, two RG steps, affine Q and even system_batch')


def validate_future_data(configurations):
    """The explicit b3 physical adapter has a stricter lattice domain than QCR."""
    if (configurations.ndim!=5 or configurations.shape[-1]!=1 or
            configurations.shape[-2]!=configurations.shape[-3]):
        raise ValueError('Future physics requires square single-channel binary 2D configurations')
    side=configurations.shape[-2]
    if side%9 or side//9<12 or (side//9)%6:
        raise ValueError('Future b3 two-step observable lattice must be L>=12 divisible by 6')


def future_module_sha(module):
    """Canonical tensor-only state identity shared with calibration receipts."""
    digest=hashlib.sha256()
    for name,tensor in sorted(module.state_dict().items()):
        if not isinstance(tensor,torch.Tensor):continue
        digest.update(name.encode());digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def qcr_module_sha256(model):
    """Read-only identities; hashing must not advance any sampling RNG."""
    return dict(Q=future_module_sha(model.theory_encoder),
                C=future_module_sha(model.rg_kernel),
                R=future_module_sha(model.theory_rg))


_INITIAL_MODULE_METADATA_FIELDS = (
    'initial_module_sha256', 'initial_module_sha256_scope',
    'random_construction_module_sha256', 'pre_affine_initialization_module_sha256',
    'training_start_module_sha256', 'module_initialization_provenance',
)


def record_initial_module_provenance(metadata, model, *, random_construction_sha256=None,
                                     pre_affine_sha256=None, resume=None):
    """Distinguish constructor, pre-calibration, and actual optimizer-start states.

    The legacy ``initial_module_sha256`` alias is deliberately left at its
    historical pre-affine-initialization point: scale-weight runner preflights
    compare it to a fresh constructor. A resumed checkpoint is not a fresh
    initialization, and missing source provenance cannot be reconstructed from
    its already-trained parameters.
    """
    if resume is not None:
        source = resume.get('metadata', {})
        for key in _INITIAL_MODULE_METADATA_FIELDS:
            metadata.pop(key, None)
            if key in source:
                metadata[key] = copy.deepcopy(source[key])
        metadata['resume_start_module_sha256'] = qcr_module_sha256(model)
        metadata['resume_start_provenance'] = dict(
            schema_version=1,
            scope='Loaded checkpoint state after setup, before this resumed segment; not an initial model',
            source_epoch=resume.get('epoch'),
            source_initial_metadata_preserved=[key for key in _INITIAL_MODULE_METADATA_FIELDS
                                               if key in source],
            training_start_initial_state_available='training_start_module_sha256' in source,
            legacy_initial_scope=('Source-declared scope preserved' if 'initial_module_sha256_scope' in source
                                  else 'Unknown legacy scope; do not interpret as first-update initialization'),
            missing_initial_metadata='Unavailable, not reconstructed from trained checkpoint state')
        return
    if random_construction_sha256 is None or pre_affine_sha256 is None:
        raise ValueError('Fresh initialization provenance requires both constructor and pre-affine identities')
    metadata['random_construction_module_sha256'] = copy.deepcopy(random_construction_sha256)
    metadata['pre_affine_initialization_module_sha256'] = copy.deepcopy(pre_affine_sha256)
    metadata['training_start_module_sha256'] = qcr_module_sha256(model)
    metadata['module_initialization_provenance'] = dict(
        schema_version=1,
        random_construction_scope='Immediately after build_model; before warm starts or data-driven initialization',
        pre_affine_initialization_scope='After optional warm start or MI initialization; before affine decoder PCA/bias initialization',
        training_start_scope='After all model initialization/calibration; before the first optimizer update of this fresh run',
        legacy_initial_module_sha256='Pre-affine-initialization alias retained for existing scale-weight preflight compatibility',
        rng_and_parameters='Read-only state hashing; no random draws or parameter changes')
    if 'initial_module_sha256' in metadata:
        metadata['initial_module_sha256_scope'] = 'pre_affine_initialization_module_sha256; legacy alias, not the post-calibration training-start state'


def sync_initial_module_provenance(contract, metadata):
    """Keep summary/checkpoint contracts aligned with the manifest's true scope."""
    for key in _INITIAL_MODULE_METADATA_FIELDS + (
            'resume_start_module_sha256', 'resume_start_provenance'):
        contract.pop(key, None)
        if key in metadata:
            contract[key] = copy.deepcopy(metadata[key])


def validate_future_normalizer_model(contract, cfg, model, configurations):
    """Check the initial random C, before loading any resume state or updating it."""
    if contract is None or not contract['weight']:return
    norm=contract['normalizer_provenance']
    side=configurations.shape[-2]
    normalizer_model=dict(norm['model_config'])
    # Historical fixed-width Q normalizers predate this explicit architecture
    # field. Only its exact historical default may be supplied when absent.
    normalizer_model.setdefault('local_features',12)
    if (normalizer_model!=cfg.to_dict()['model'] or
            norm['initial_C_sha256']!=future_module_sha(model.rg_kernel) or
            norm['scales']!=[side,side//3,side//9] or
            norm['samples']!=configurations.shape[1]):
        raise ValueError('Frozen future normalizer must match this initial random C/model/geometry/sample count')


def load_future_inputs(args, data_sha256, sizes):
    """Pair identities are explicit provenance, never inferred from K tensors."""
    if not args.q_future_weight and not args.q_future_control:
        return None,None
    raw=json.loads(args.q_future_pairs.read_text())
    pair_sha=file_fingerprint(args.q_future_pairs)
    if raw.get('schema_version')!=1 or raw.get('data_sha256')!=data_sha256:
        raise ValueError('Future pair identities must bind to this exact data SHA')
    # The public pair index is identity-only: physical labels are not accepted
    # as metadata fields, let alone read to infer a partner.
    if set(raw)-{'schema_version','data_sha256','train','test','provenance'}:
        raise ValueError('Future pairing JSON accepts identities/provenance only, not physical labels')
    result={}
    for split,n in sizes.items():
        item=raw[split]
        if set(item)!={'condition_ids','chain_ids','pairs'}:
            raise ValueError('Future split pairing accepts only condition_ids, chain_ids and pairs')
        for name in ('condition_ids','chain_ids'):
            if (not isinstance(item.get(name),list) or len(item[name])!=n or
                    any(type(v) is not int for v in item[name])):
                raise ValueError('Future identities must be explicit integer IDs for every chain')
        condition=torch.tensor(item['condition_ids'],dtype=torch.long)
        chain=torch.tensor(item['chain_ids'],dtype=torch.long)
        pairs=item.get('pairs')
        if (not isinstance(pairs,list) or len(pairs)*2!=n or
                any(not isinstance(p,list) or len(p)!=2 or any(type(i) is not int for i in p) for p in pairs)):
            raise ValueError('Future pairs must be a complete disjoint partition')
        pairs=torch.tensor(pairs,dtype=torch.long)
        if sorted(pairs.flatten().tolist())!=list(range(n)) or len(torch.unique(chain))!=n:
            raise ValueError('Future pairs/chains overlap, omit chains or leave the split')
        if not torch.equal(condition[pairs[:,0]],condition[pairs[:,1]]):
            raise ValueError('Independent future partners must have the same condition identity')
        result[split]=dict(pairs=pairs,condition_ids=condition,chain_ids=chain)
    if set(result['train']['chain_ids'].tolist()) & set(result['test']['chain_ids'].tolist()):
        raise ValueError('Physical chain identities must be disjoint across TRAIN/TEST')
    from recurrent_rg.future_physics import BINARY2D_OBSERVABLE_NAMES
    contract=dict(schema_version=1,weight=args.q_future_weight,hidden=args.q_future_hidden,
        mode='paired_control' if args.q_future_control else 'future_physics',
        paths=[[0,1],[1,2],[0,2]],pairs_sha256=pair_sha,
        data_sha256=data_sha256,labels_used=False,
        sampling='Uniform disjoint physical-chain pairs, both directions; system_batch counts chains',
        observables=list(BINARY2D_OBSERVABLE_NAMES),
        adapter='periodic binary square; correlations at L/6 and L/3, not historical fixed r4/r8',
        normalizer_sha256=None,target_mean=None,target_std=None,std_floor=.01,
        fitting_split=None,gradient='Original independent Q/C/R objectives; no auxiliary predictor')
    if args.q_future_weight:
        norm=json.loads(args.q_future_normalizer.read_text())
        if (norm.get('schema_version')!=1 or norm.get('data_sha256')!=data_sha256 or
                norm.get('pairs_sha256')!=pair_sha or norm.get('fitting_split')!='train'):
            raise ValueError('Future normalizer must be frozen TRAIN-only and bound to data/pair SHA')
        mean=torch.as_tensor(norm['target_mean'],dtype=torch.float64)
        std=torch.as_tensor(norm['target_std'],dtype=torch.float64)
        if (mean.shape!=(3,5) or std.shape!=(3,5) or not torch.isfinite(mean).all() or
                not torch.isfinite(std).all() or (std<0).any()):
            raise ValueError('Future fixed TRAIN mean/std must be finite [3,5] with nonnegative std')
        unfloored=torch.as_tensor(norm.get('target_std_unfloored',[]),dtype=torch.float64)
        if (norm.get('std_floor')!=.01 or unfloored.shape!=(3,5) or
                not torch.isfinite(unfloored).all() or (unfloored<0).any() or
                not torch.equal(std,unfloored.clamp_min(.01)) or
                norm.get('paths')!=contract['paths'] or norm.get('observable_names')!=contract['observables'] or
                norm.get('fitting_ensembles')!=sizes['train'] or norm.get('C_draws')!=5 or
                norm.get('model_pretraining_updates')!=0 or norm.get('QCR_updates')!=0 or
                type(norm.get('calibration_seed')) is not int or
                not isinstance(norm.get('model_config'),dict) or
                not isinstance(norm.get('initial_C_sha256'),str) or len(norm['initial_C_sha256'])!=64 or
                not isinstance(norm.get('scales'),list) or type(norm.get('samples')) is not int):
            raise ValueError('Future normalizer needs the fixed initial-C TRAIN calibration provenance and .01 physical std floor')
        contract.update(normalizer_sha256=file_fingerprint(args.q_future_normalizer),
            target_mean=mean.tolist(),target_std=std.tolist(),
            target_std_unfloored=unfloored.tolist(),
            normalizer_provenance={key:norm[key] for key in ('model_config','initial_C_sha256','scales',
                'samples','fitting_ensembles','C_draws','calibration_seed','model_pretraining_updates','QCR_updates')},
            fitting_split='TRAIN only; frozen initial-random-C normalizer never refreshed',
            gradient='Q and auxiliary predictor only; C targets detached; R closure unchanged')
    return result,contract


def future_batch_ids(identities, systems, generator=None):
    pairs=identities['pairs']
    count=min(systems//2,len(pairs))
    if count<1:
        raise ValueError('Future batch requires at least one independent chain pair')
    selected=(torch.randperm(len(pairs),generator=generator)[:count] if generator is not None else
              torch.linspace(0,len(pairs)-1,count).long())
    return pairs[selected].flatten()


def future_encoding_execution(weight, execution):
    """Memory-only Q execution; the default/CPU/paired-control path is intact."""
    if weight and execution.get('enabled'):
        return dict(encoding_batch_chunk=4,
                    validated_binary_inputs=bool(execution.get('trusted_internal_binary_checks')))
    return {}


def future_batch_loss(predictor, model, levels, identities, indices, weight, **encoding_execution):
    from recurrent_rg.future_physics import paired_future_physics_loss
    partner=torch.arange(len(indices),device=levels[0].device).reshape(-1,2).flip(1).flatten()
    return paired_future_physics_loss(predictor,model.theory_encoder,levels,partner,
        identities['condition_ids'][indices].to(levels[0].device),
        identities['chain_ids'][indices].to(levels[0].device),weight=weight,**encoding_execution)


def validate_resume_future(resume, contract, device, execution):
    old=resume.get('metadata',{})
    if old.get('q_future_physics')!=contract:
        raise ValueError('Exact resume must preserve future-physics loss/pairing/normalizer contract')
    if contract is None:
        if 'future_physics_predictor_state' in resume:
            raise ValueError('Cannot silently remove saved future-physics predictor')
        return
    auxiliary_expected=bool(contract['weight'])
    if (old.get('execution_device')!=str(device) or old.get('precision')!='torch.float64' or
            old.get('independent_c') is not True or old.get('training_driver')!='rsmi_separate_qcr' or
            ('future_physics_predictor_state' in resume)!=auxiliary_expected):
        raise ValueError('Future exact resume requires auxiliary state and identical device/dtype/routing')
    fields=('version','enabled','reference','shared_binary_Q_tables','trusted_internal_binary_checks','precision','selected_empirical_mi','mi_test_device','future_encoding_batch_chunk','future_validated_binary_inputs')
    if any(old.get('execution_optimizations',{}).get(k)!=execution.get(k) for k in fields):
        raise ValueError('Future exact resume requires identical execution conventions')
    required=('model','optimizer','history','torch_rng_state','batch_generator_state','rsmi_generator_state','lr_scheduler_state')
    if any(k not in resume for k in required):
        raise ValueError('Future exact resume needs Adam, RNG, history and scheduler state')
    if torch.device(device).type=='cuda' and ('cuda_rng_state' not in resume or resume.get('cuda_device_index')!=torch.device(device).index):
        raise ValueError('Future exact CUDA resume needs the identical device and CUDA RNG')
    if torch.device(device).type=='cpu' and ('cuda_rng_state' in resume or 'cuda_device_index' in resume):
        raise ValueError('A future CUDA checkpoint cannot resume exactly on CPU')


def validate_mi_test_cpu(args, device):
    """Explicit execution-only placement; the TRAIN estimator stays unchanged."""
    if args.mi_test_cpu and (torch.device(device).type!='cuda' or
            args.estimator!='histogram' or args.environment!='full_shell'):
        raise ValueError('mi-test-cpu requires explicit CUDA execution and full-shell histogram MI')


def place_mi_histograms(histograms, device, *, test_cpu=False):
    """Do not materialize the TEST histogram on CUDA when explicitly requested."""
    return {split:hist.to('cpu' if test_cpu and split=='test' else device)
            for split,hist in histograms.items()}


def place_mi_histogram_split(histogram, split, device, *, selected=False, test_cpu=False):
    """Prepare/place one split before the next CPU cache is loaded.

    The empirical tensors, dtype, support ordering, sparse reductions and
    selection method are unchanged. The caller drops its CPU object reference
    after this function returns; CPU TEST placement remains explicitly opt-in.
    """
    if selected and split=='train':
        histogram.prepare_selection()
    return histogram.to('cpu' if test_cpu and split=='test' else device)


def validate_resume_low_ram_execution(resume, policy):
    """An exact continuation must not silently switch storage/gather protocol."""
    previous=resume.get('metadata',{}).get('execution_optimizations',{}).get('low_ram_execution')
    if previous is None:
        if policy.get('enabled'):
            raise ValueError('Checkpoint predates the low-RAM execution protocol; use its original '
                             'frozen code for exact continuation, not a silent execution-path change')
        return
    if previous!=policy:
        raise ValueError('Exact resume must preserve the low-RAM execution protocol')


def logged_mi_table(table, split, *, test_cpu=False):
    """Only evaluation may use this detached CPU table; never the TRAIN loss."""
    return table.detach().cpu() if test_cpu and split=='test' else table


def validate_resume_mi_placement(resume, execution):
    if resume.get('metadata',{}).get('execution_optimizations',{}).get('mi_test_device')!=execution.get('mi_test_device'):
        raise ValueError('Exact resume must preserve the TEST empirical-MI execution device')


def benchmark_memory_snapshot(benchmark_warmup, device, phase, *, step=None):
    """Allocator counters only: no synchronization, forward pass, or RNG draw."""
    if benchmark_warmup is None or torch.device(device).type!='cuda' or step not in (None,1):
        return
    print(json.dumps(dict(benchmark_memory=dict(phase=phase,step=step,
        allocated_bytes=torch.cuda.memory_allocated(device),
        reserved_bytes=torch.cuda.memory_reserved(device),
        max_allocated_bytes=torch.cuda.max_memory_allocated(device),
        max_reserved_bytes=torch.cuda.max_memory_reserved(device)))),flush=True)


def validate_cuda_options(args):
    """Allow default, supervised, or explicitly MI-free independent QCR."""
    cuda_ablations=('source','mi_initialize','freeze_c_checkpoint',
        'affine_pca_init','q_only_weight_sweep','q_only_candidates',
        'q_balanced_reconstruction','q_last_layer_audit','q_head_structure_audit',
        'q_feature_bottleneck','q_original_plateau','q_initialization_pilot',
        'q_unfolding','r_only_continue','r_only_refit','r_only_plateau','joint_train_plateau',
        'frozen_cache')
    validate_ising_b0_ablation(args)
    validate_all_start_closure_ablation(args)
    no_mi_ablation=args.rsmi_weight==0 and args.allow_nondefault_recipe
    if (any(getattr(args,key) for key in cuda_ablations)
            or (args.allow_nondefault_recipe and not no_mi_ablation)
            or args.estimator!='histogram' or not args.independent_c):
        raise ValueError('CUDA supports default independent QCR, identical-device resume, '
                         'microscopic Q coupling supervision, or explicit zero-RSMI ablation only')
    if not math.isfinite(args.q_coupling_weight) or args.q_coupling_weight<0:
        raise ValueError('Nonnegative finite Q coupling weight required')
    if args.q_coupling_weight and args.config is None:
        raise ValueError('CUDA supervised training requires a standalone configuration')


def add_supervision_contract(contract, weight):
    """Describe explicit Q-label supervision without changing the zero-weight contract."""
    if not weight:
        return
    contract['q_supervision']=dict(weight=weight,scale=0,
        target='raw coupling labels, Q(full microscopic batch)',fitting_split='TRAIN only',
        gradient='Q only; no labels for coarse levels or C/R',
        evaluation='TRAIN/TEST coupling losses are diagnostics; TEST never enters optimization')
    contract['note']=contract['note'].replace('No critic or coupling labels.','No critic.').replace(
        'no critic or coupling labels.','no critic.') + (
        ' Raw TRAIN coupling labels explicitly supervise microscopic Q; TEST labels are diagnostic only.')
    if 'Q: representation losses only;' in contract['gradient_QC']:
        contract['gradient_QC']=contract['gradient_QC'].replace('Q: representation losses only;',
            'Q: representation losses + q_coupling_weight * microscopic coupling MSE;')
    else:
        contract['gradient_QC']+='; microscopic coupling MSE adds Q-encoder gradients only'


def validate_resume_supervision(resume, args, cfg, *, device, data_sha256, execution):
    """Reject objective/device changes before restoring any model or Adam state."""
    old=resume['metadata']
    if old.get('data_sha256')!=data_sha256:
        raise ValueError('Resume data changed')
    if not args.q_coupling_weight:
        if old.get('labels_used_in_training',False) or old.get('q_supervision') is not None:
            raise ValueError('Exact resume cannot remove checkpoint Q coupling supervision')
        return
    expected=dict(note='',gradient_QC='')
    add_supervision_contract(expected,args.q_coupling_weight)
    expected['q_supervision']['coupling_names']=list(cfg.data.coupling_names)
    if old.get('labels_used_in_training') is not True or old.get('q_supervision')!=expected['q_supervision']:
        raise ValueError('Exact resume requires identical Q coupling supervision, weight, scale and target labels')
    if (not args.independent_c or old.get('independent_c') is not True
            or old.get('training_driver')!='rsmi_separate_qcr' or old.get('frozen_C')
            or old.get('precision')!='torch.float64' or old.get('execution_device')!=str(device)):
        raise ValueError('Supervised exact resume requires the same device, dtype and independent Q/C/R gradients')
    old_execution=old.get('execution_optimizations',{})
    stable_fields=('version','enabled','reference','shared_binary_Q_tables',
                   'trusted_internal_binary_checks','precision','selected_empirical_mi')
    if any(old_execution.get(key)!=execution.get(key) for key in stable_fields):
        raise ValueError('Supervised exact resume requires identical execution conventions')
    required=('model','optimizer','history','torch_rng_state','batch_generator_state',
              'rsmi_generator_state','lr_scheduler_state')
    if any(key not in resume for key in required):
        raise ValueError('Supervised exact resume requires saved Adam, RNG, history and learning-rate state')
    device=torch.device(device)
    if device.type=='cuda':
        if 'cuda_rng_state' not in resume or resume.get('cuda_device_index')!=device.index:
            raise ValueError('Supervised exact resume requires the same CUDA device and saved CUDA RNG')
    elif 'cuda_rng_state' in resume or 'cuda_device_index' in resume:
        raise ValueError('A CUDA checkpoint cannot resume exactly on CPU')


def final_report(output, *, skip=False):
    """Keep saved training artifacts even when the expensive final report is omitted."""
    started=time.perf_counter()
    if skip:
        print('Final report explicitly skipped; final model and training summary are saved.',flush=True)
    else:
        evaluate_run(output,output=output/'report')
    summary=json.loads((output/'summary.json').read_text())
    summary['final_report_skipped']=skip
    summary['timing']['report_seconds']=0. if skip else time.perf_counter()-started
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


def train_component_losses(row, cfg, args):
    """Actual independently routed objectives; TEST never enters stopping."""
    w=cfg.loss.weights
    return dict(epoch=row['epoch'],
        component_Q=w.anchor*row['train_anchor']+
                    w.coarse_consistency*row['train_coarse_consistency']+
                    w.separation*row['train_separation'],
        component_C=(args.rsmi_weight*row['train_rsmi'] if args.rsmi_weight else 0.)+
                    args.variance_weight*row['train_kernel_noncollapse']+
                    args.color_weight*row.get('train_kernel_sign',0.),
        component_R=row.get('train_closure_total',
                    w.commutator*row['train_commutator']+
                    cfg.loss.rollout_weight*row['train_rollout']))


def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--source',type=Path)
    source.add_argument('--config',type=Path,help='Standalone configuration; initialize a fresh model')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--configuration-cnn-short-trial', action='store_true',
                   help='Explicit Ising B0 scalar CNN trial, fresh initialization and <=20min complete execution')
    p.add_argument('--configuration-cnn-r-only', action='store_true',
                   help='Explicit frozen-Q/C continuation of the scalar Ising CNN trial; online targets and original R Adam')
    p.add_argument('--phi4-cnn-r-only', action='store_true',
                   help='Pinned phi4 CNN R-only continuation; freeze Q/C and retain original R Adam and online targets')
    p.add_argument('--device',default='cpu',help='Explicit execution device; CPU remains the historical default')
    p.add_argument('--benchmark-warmup',type=int,help='Measure real updates; omit this many first updates from steady timing, without adding extra training')
    p.add_argument('--reference-execution',action='store_true',help='Disable execution-only GPU accelerations for a same-recipe comparison')
    p.add_argument('--mi-cache-dir',type=Path,help='GPU CPU-statistics cache directory; defaults to .qcr_mi_cache beside input data')
    p.add_argument('--mi-test-cpu',action='store_true',help='CUDA execution only: retain full TEST empirical-MI statistics on CPU; TRAIN MI and gradients unchanged')
    p.add_argument('--theory-dim',type=int,help='Override latent dimension; all modules are newly initialized unless resuming')
    p.add_argument('--rsmi-weight',type=float,default=2.)
    p.add_argument('--estimator',choices=['infonce','histogram'],default='histogram')
    p.add_argument('--environment',choices=['full_shell','local_pairs'],default='full_shell')
    p.add_argument('--variance-weight',type=float,default=2.)
    p.add_argument('--color-weight',type=float,default=1.,help='Uniform-block label-preservation NLL weight')
    p.add_argument('--variance-ratio',type=float,default=1.)
    p.add_argument('--variance-per-ensemble',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--mi-initialize',action='store_true',help='Initialize trainable C from the best empirical-MI binary rule')
    p.add_argument('--independent-c',action=argparse.BooleanOptionalAction,default=True,help='C: MI/variance/color; Q: representation; R: closure')
    p.add_argument('--allow-nondefault-recipe',action='store_true',help='Explicit opt-in for a user-requested ablation; never enable automatically')
    p.add_argument('--ising-b0-ablation', choices=ISING_B0_ABLATIONS,
                   help='Explicit B0 d_g=5 b=3 campaign; none locks the baseline, other names remove only the named loss')
    p.add_argument('--all-start-closure-ablation', action='store_true',
                   help='Explicit paired b=2 four-level anchor-only closure ablation; no other objective changes')
    p.add_argument('--resume-checkpoint',type=Path,help='Resume this driver with identical objectives and RNG/Adam state')
    p.add_argument('--epochs',type=int,help='Total updates, including updates already completed when resuming')
    p.add_argument('--max-wall-seconds',type=float,help='Optional setup+training wall limit; finish the current step and save actual progress')
    p.add_argument('--max-training-seconds',type=float,help='Optional training-only wall limit after completed setup; save actual progress, never reduce configured epochs')
    p.add_argument('--joint-train-plateau',action='store_true',
                   help='Stop complete default Q/C/R training on separate TRAIN-only objective plateaus; no LR or gradient changes')
    p.add_argument('--freeze-c-checkpoint',type=Path,help='Explicit ablation: warm-start Q/C/R weights and freeze this learned probabilistic C; reset Adam')
    p.add_argument('--affine-pca-init',action='store_true',help='Explicit frozen-C ablation: initialize trainable affine decoder from TRAIN histograms only')
    p.add_argument('--q-coupling-weight',type=float,default=0.,help='Explicit supervised ablation: microscopic Q(D) fits coupling labels only')
    p.add_argument('--q-future-weight',type=float,default=0.,help='Explicit label-free Q future-physical-observable objective; zero preserves original training')
    p.add_argument('--q-future-control',action='store_true',help='Explicit paired-sampler control: zero future weight, no auxiliary predictor or extra Q loss')
    p.add_argument('--q-future-hidden',type=int,default=32,help='Auxiliary future predictor width (fixed32)')
    p.add_argument('--q-future-pairs',type=Path,help='Public independent-chain identity JSON bound to data SHA; no coupling labels')
    p.add_argument('--q-future-normalizer',type=Path,help='Frozen TRAIN-only physical target mean/std JSON bound to data and pairing SHA')
    p.add_argument('--skip-final-report',action='store_true',help='Explicitly skip evaluate_run after saving the final model and training summary')
    p.add_argument('--q-only-weight-sweep',action='store_true',help='Explicit ablation: frozen cached source C, fresh paired Q, separation weights 10/30/100')
    p.add_argument('--q-only-candidates',action='store_true',help='Explicit paired Q representation-objective ablation')
    p.add_argument('--q-balanced-reconstruction',action='store_true',help='Explicit paired Q-only affine plus signal-balanced reconstruction ablation; source C/R frozen')
    p.add_argument('--q-last-layer-audit',action='store_true',help='Explicit ablation: freeze Q backbone/C/R and train only final Q linear layer')
    p.add_argument('--q-head-structure-audit',action='store_true',help='Explicit ablation: linear/whitened/monotone Q heads on frozen features')
    p.add_argument('--q-feature-bottleneck',action='store_true',help='Explicit ablation: frozen features, reconstruction bottleneck without geometry')
    p.add_argument('--q-original-plateau',action='store_true',help='Explicit ablation: resume original full Q with train-only plateau LR schedule')
    p.add_argument('--q-initialization-pilot',action='store_true',help='Explicit paired Q-only initialization ablation')
    p.add_argument('--q-unfolding',action='store_true',help='Explicit Q-only graph-coordinate / linear distribution reconstruction ablation')
    p.add_argument('--r-only-continue',action='store_true',help='Explicit ablation: freeze source Q/C, continue existing R and Adam on saved TRAIN trajectories; --epochs is additional R updates')
    p.add_argument('--r-only-refit',action='store_true',help='Explicit ablation: refit original R to a paired balanced-Q checkpoint, Q/C frozen and fresh R Adam')
    p.add_argument('--r-only-plateau',action='store_true',help='Explicit ablation: freeze final Q/C, preserve online half-subset targets and R Adam, stop on a TRAIN plateau; --epochs is a safety cap on additional R updates')
    p.add_argument('--frozen-cache',type=Path,help='Existing Q-only run containing frozen_levels.pt')
    return p


def validate_recipe(args, cfg):
    """Fail closed: a historical flag must not silently change the default recipe."""
    if cfg.training.r_updates_per_batch > 1 and not args.independent_c:
        raise ValueError('R inner updates require independent C gradients')
    validate_all_start_closure_ablation(args, cfg)
    if getattr(args, 'all_start_closure_ablation', False):
        return
    if getattr(args, 'ising_b0_ablation', None) is not None:
        validate_ising_b0_ablation(args, cfg)
        return
    if args.allow_nondefault_recipe:
        return
    expected=dict(estimator='histogram',environment='full_shell',rsmi_weight=2.,
                  variance_weight=2.,color_weight=1.,variance_ratio=1.,
                  variance_per_ensemble=True,independent_c=True,mi_initialize=False)
    changed=[key for key,value in expected.items() if getattr(args,key)!=value]
    if cfg.model.kernel_initialization!='random':changed.append('kernel_initialization')
    if cfg.model.fixed_kernel:changed.append('fixed_kernel')
    if cfg.model.kernel_sampling not in ('bernoulli','categorical'):changed.append('kernel_sampling')
    modules={'Q','R'} if args.freeze_c_checkpoint else {'Q','C','R'}
    if set(cfg.training.train_modules)!=modules:changed.append('train_modules')
    if changed:
        raise ValueError('Locked default recipe changed: '+', '.join(changed)+
                         '; use --allow-nondefault-recipe only for an explicitly requested ablation')


def validate_rsmi_options(args):
    """Zero MI is an explicit ablation, not a measured zero-information result."""
    if not math.isfinite(args.rsmi_weight) or args.rsmi_weight<0:
        raise ValueError('Nonnegative finite RSMI weight required')
    if args.rsmi_weight:
        return
    named_zero=getattr(args, 'ising_b0_ablation', None)=='c_no_mi'
    if named_zero:
        validate_ising_b0_ablation(args)
    if (not args.allow_nondefault_recipe and not named_zero) or not args.independent_c:
        raise ValueError('Zero RSMI requires --allow-nondefault-recipe and independent C gradients')
    conflicts=('mi_initialize','mi_cache_dir','mi_test_cpu','source','freeze_c_checkpoint',
        'affine_pca_init','q_future_weight','q_future_control','q_only_weight_sweep',
        'q_only_candidates','q_balanced_reconstruction','q_last_layer_audit',
        'q_head_structure_audit','q_feature_bottleneck','q_original_plateau',
        'q_initialization_pilot','q_unfolding','r_only_continue','r_only_refit',
        'r_only_plateau','joint_train_plateau','frozen_cache')
    if any(getattr(args,key) for key in conflicts):
        raise ValueError('Zero RSMI supports standalone independent QCR only; no MI initialization/cache/placement or other component ablations')
    if args.estimator!='histogram' or args.environment!='full_shell':
        raise ValueError('Zero RSMI must leave unused estimator/environment options at their defaults')


def default_representation(cfg, raw_config, *, continuing=False):
    """Fresh runs default to affine reconstruction; preserve explicit/history choices."""
    model=raw_config.get('model',{})
    if continuing or 'affine_reconstruction' in model or model.get('reconstruction_hidden',0):
        return cfg
    return replace(cfg,model=replace(cfg.model,affine_reconstruction=True))


def default_closure(cfg, raw_config, *, continuing=False, historical=False):
    """New default only for fresh standard training, never checkpoint loading.

    Missing fields deserialize as the historical objective. Fresh standard
    training opts into the current policy, like default_representation; exact
    reproduction must explicitly pin the old mode in a separate configuration.
    Explicit named historical ablations and all continuations retain their mode.
    """
    if (continuing or historical
            or 'closure_objective' in raw_config.get('loss', {})):
        return cfg
    return replace(cfg, loss=replace(cfg.loss, closure_objective='all_start_anchor'))


def resume_closure(cfg, raw_config, resume):
    """An omitted request field inherits the checkpoint's actual objective.

    This covers both pre-policy checkpoints (missing means historical) and
    current-default runs originally started from a JSON without the new field.
    Explicit requested modes remain authoritative and face exact-resume checks.
    """
    if resume is None or 'closure_objective' in raw_config.get('loss', {}):
        return cfg
    saved = resume.get('metadata', {}).get('config')
    if not saved:
        raise ValueError('Exact resume requires saved configuration for closure policy')
    mode = RunConfig.from_dict(saved).loss.closure_objective
    return replace(cfg, loss=replace(cfg.loss, closure_objective=mode))


def default_terminal_r(cfg, raw_config, *, historical=False, resume=None):
    """Fresh standard pipelines opt in; historical recipes/checkpoints stay exact."""
    if resume is not None:
        if (resume.get('metadata', {}).get('terminal_R_fit_completed') or
                resume.get('metadata', {}).get('terminal_R_stage')):
            raise ValueError('A terminal R-only checkpoint is not a joint-resume checkpoint; use its preserved joint source')
        if 'terminal_r_fit' not in raw_config.get('training', {}):
            saved = RunConfig.from_dict(resume['metadata']['config']).training
            return replace(cfg, training=replace(cfg.training, terminal_r_fit=saved.terminal_r_fit,
                terminal_r_max_steps=saved.terminal_r_max_steps, terminal_r_cache_repeats=saved.terminal_r_cache_repeats))
        return cfg
    if historical or 'terminal_r_fit' in raw_config.get('training', {}): return cfg
    return replace(cfg, training=replace(cfg.training, terminal_r_fit=True))


def selected_mi_execution(execution, sparse, resume):
    """Fresh optimized runs may select supports; exact old resumes stay old."""
    if not execution['enabled'] or not sparse:
        return False
    if resume is None:
        return True
    return bool(resume.get('metadata',{}).get('execution_optimizations',{}).get(
        'selected_empirical_mi',False))


def validate_training_clock(args):
    if args.max_training_seconds is None:
        return
    if not math.isfinite(args.max_training_seconds) or args.max_training_seconds<=0:
        raise ValueError('max-training-seconds must be finite and positive')
    if args.max_wall_seconds is not None:
        raise ValueError('Choose either max-training-seconds or max-wall-seconds, not both')
    if args.source is not None:
        raise ValueError('Training-only clock requires a standalone default config')


def main(argv=None):
    wall_started=time.perf_counter()
    args=build_parser().parse_args(argv)
    if args.phi4_cnn_r_only:
        from experiments.phi4_cnn_r_only import run
        return run(args)
    if args.configuration_cnn_r_only:
        from experiments.ising_cnn_r_only import run
        return run(args)
    if args.configuration_cnn_short_trial:
        from experiments.ising_cnn_short_trial import run
        return run(args)
    validate_ising_b0_ablation(args)
    validate_all_start_closure_ablation(args)
    validate_training_clock(args)
    validate_rsmi_options(args)
    validate_future_options(args)
    rsmi_enabled=bool(args.rsmi_weight)
    device=torch.device(args.device)
    if device.type not in ('cpu','cuda') or (device.type=='cpu' and device.index is not None):
        raise ValueError('Supported execution devices are cpu and cuda:<index>')
    validate_mi_test_cpu(args,device)
    if device.type=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA was explicitly requested but is unavailable')
        device=torch.device('cuda',0 if device.index is None else device.index)
        torch.cuda.set_device(device)
        validate_cuda_options(args)
    if args.benchmark_warmup is not None and args.benchmark_warmup<0:
        raise ValueError('benchmark-warmup must be nonnegative')
    if args.mi_cache_dir and (device.type!='cuda' or args.reference_execution):
        raise ValueError('mi-cache-dir requires optimized CUDA execution')
    if args.joint_train_plateau:
        joint_conflicts=('r_only_plateau','r_only_refit','r_only_continue',
            'q_only_weight_sweep','q_only_candidates','q_balanced_reconstruction',
            'q_last_layer_audit','q_head_structure_audit','q_feature_bottleneck',
            'q_original_plateau','q_initialization_pilot','q_unfolding',
            'freeze_c_checkpoint','affine_pca_init','q_coupling_weight','frozen_cache')
        if any(getattr(args,key) for key in joint_conflicts):
            raise ValueError('Joint TRAIN plateau cannot be combined with component-only ablations')
    if args.r_only_plateau:
        conflicts=('config','resume_checkpoint','freeze_c_checkpoint','affine_pca_init',
                   'q_coupling_weight','q_only_weight_sweep','q_only_candidates','q_balanced_reconstruction',
                   'q_last_layer_audit','q_head_structure_audit','q_feature_bottleneck','q_original_plateau',
                   'q_initialization_pilot','q_unfolding','r_only_continue','r_only_refit',
                   'theory_dim','frozen_cache','mi_initialize','allow_nondefault_recipe')
        if args.source is None or any(getattr(args,key) for key in conflicts):
            raise ValueError('R-only plateau continuation requires a final source run and no other ablations')
        if args.epochs is not None and args.epochs < 1:
            raise ValueError('Positive update count required')
        loaded_source=load_run(args.source/'checkpoint_last.pt')
        saved_metadata=loaded_source.payload['metadata']
        source_config=RunConfig.from_dict(saved_metadata['source_config']
            if saved_metadata.get('training_driver') == 'frozen_qc_r_plateau' else loaded_source.config)
        validate_recipe(args,source_config)
        from experiments.frozen_qc_r_plateau import run
        return run(args.source,args.output,30000 if args.epochs is None else args.epochs,
                   max_wall_seconds=args.max_wall_seconds)
    if args.q_balanced_reconstruction:
        conflicts=('config','resume_checkpoint','freeze_c_checkpoint','affine_pca_init',
                   'q_coupling_weight','q_only_weight_sweep','q_only_candidates',
                   'q_last_layer_audit','q_head_structure_audit','q_feature_bottleneck',
                   'q_original_plateau','q_initialization_pilot','q_unfolding',
                   'r_only_continue','r_only_refit','theory_dim','frozen_cache','mi_initialize','allow_nondefault_recipe')
        if args.source is None or any(getattr(args, key) for key in conflicts):
            raise ValueError('Balanced Q-only pilot requires --source and no other ablations')
        if args.max_wall_seconds is not None and (not math.isfinite(args.max_wall_seconds) or args.max_wall_seconds <= 0):
            raise ValueError('Positive finite wall limit required')
        if args.epochs is not None and args.epochs < 1:
            raise ValueError('Positive update count required')
        source_config=RunConfig.from_dict(json.loads((args.source/'manifest.json').read_text())['config'])
        validate_recipe(args,source_config)
        from experiments.q_balanced_reconstruction import run
        return run(args.source,args.output,args.epochs or 400,
                   max_wall_seconds=args.max_wall_seconds or 540.)
    if args.r_only_refit:
        conflicts=('config','resume_checkpoint','freeze_c_checkpoint','affine_pca_init',
                   'q_coupling_weight','q_only_weight_sweep','q_only_candidates','q_balanced_reconstruction',
                   'q_last_layer_audit','q_head_structure_audit','q_feature_bottleneck','q_original_plateau',
                   'q_initialization_pilot','q_unfolding','r_only_continue','theory_dim','frozen_cache',
                   'mi_initialize','allow_nondefault_recipe')
        if args.source is None or any(getattr(args,key) for key in conflicts):
            raise ValueError('R-only refit requires a Q-only checkpoint and no other ablations')
        if args.epochs is not None and args.epochs<1:
            raise ValueError('Positive update count required')
        if args.max_wall_seconds is not None and (not math.isfinite(args.max_wall_seconds) or args.max_wall_seconds<=0):
            raise ValueError('Positive finite wall limit required')
        source_config=RunConfig.from_dict(load_run(args.source).config)
        validate_recipe(args,replace(source_config,training=replace(source_config.training,train_modules=('Q','C','R'))))
        from experiments.frozen_balanced_qc_r_refit import run
        return run(args.source,args.output,args.epochs or 6000,
                   max_wall_seconds=args.max_wall_seconds or 540.)
    if args.r_only_continue:
        other_ablations=('freeze_c_checkpoint','affine_pca_init','resume_checkpoint','theory_dim',
            'q_coupling_weight','q_only_weight_sweep','q_only_candidates','q_last_layer_audit',
            'q_head_structure_audit','q_feature_bottleneck','q_original_plateau',
            'q_initialization_pilot','q_unfolding','frozen_cache','mi_initialize','allow_nondefault_recipe')
        if args.source is None or any(getattr(args,key) for key in other_ablations):
            raise ValueError('R-only continuation requires --source and no other ablations or overrides')
        from experiments.frozen_qc_r_continue import run
        return run(args.source,args.output,args.epochs or 6000,max_wall_seconds=args.max_wall_seconds)
    if args.freeze_c_checkpoint and (args.resume_checkpoint or not args.config or not args.independent_c or args.mi_initialize or args.q_coupling_weight):
        raise ValueError('Frozen C requires a standalone QR config, independent gradients, and no resume/kernel search/supervision')
    if args.affine_pca_init and not args.freeze_c_checkpoint:
        raise ValueError('PCA decoder initialization is an explicit frozen-C QR ablation')
    if args.max_wall_seconds is not None and (not math.isfinite(args.max_wall_seconds) or args.max_wall_seconds<=0):
        raise ValueError('Positive finite wall limit required')
    if args.q_unfolding:
        if args.source is None or args.frozen_cache is None or args.q_initialization_pilot or args.q_original_plateau or args.q_feature_bottleneck or args.q_head_structure_audit or args.q_last_layer_audit or args.q_only_candidates or args.q_only_weight_sweep or args.resume_checkpoint or args.q_coupling_weight:
            raise ValueError('Q unfolding ablation requires source/cache and no other ablations')
        from experiments.q_unfolding import run
        return run(args.source,args.frozen_cache,args.output,args.epochs or 1200)
    if args.q_initialization_pilot:
        if args.source is None or args.frozen_cache is None or args.q_original_plateau or args.q_feature_bottleneck or args.q_head_structure_audit or args.q_last_layer_audit or args.q_only_candidates or args.q_only_weight_sweep or args.resume_checkpoint or args.q_coupling_weight:
            raise ValueError('Q initialization pilot requires source/cache and no other ablations')
        from experiments.q_initialization_pilot import run
        return run(args.source,args.frozen_cache,args.output,args.epochs or 400)
    if args.q_original_plateau:
        if args.source is None or args.frozen_cache is None or args.q_feature_bottleneck or args.q_head_structure_audit or args.q_last_layer_audit or args.q_only_candidates or args.q_only_weight_sweep or args.resume_checkpoint or args.q_coupling_weight:
            raise ValueError('Original Q plateau audit requires source/cache and no other ablations')
        from experiments.q_original_plateau import run
        return run(args.source,args.frozen_cache,args.output,args.epochs or 30000)
    if args.q_feature_bottleneck:
        if args.source is None or args.frozen_cache is None or args.q_head_structure_audit or args.q_last_layer_audit or args.q_only_candidates or args.q_only_weight_sweep or args.resume_checkpoint or args.q_coupling_weight:
            raise ValueError('Feature bottleneck requires source/cache and no other ablations')
        from experiments.q_feature_bottleneck import run
        return run(args.source,args.frozen_cache,args.output,args.epochs or 2400)
    if args.q_head_structure_audit:
        if args.source is None or args.frozen_cache is None or args.q_last_layer_audit or args.q_only_candidates or args.q_only_weight_sweep or args.resume_checkpoint or args.q_coupling_weight:
            raise ValueError('Head structure audit requires source/cache and no other ablations')
        from experiments.q_head_structure_audit import run
        return run(args.source,args.frozen_cache,args.output,args.epochs or 1200)
    if args.q_last_layer_audit:
        if args.source is None or args.frozen_cache is None or args.q_only_candidates or args.q_only_weight_sweep or args.resume_checkpoint or args.q_coupling_weight:
            raise ValueError('Last-layer audit requires source/cache and no other training ablations')
        from experiments.q_last_layer_audit import run
        return run(args.source,args.frozen_cache,args.output,args.epochs or 1200)
    if args.q_only_candidates:
        if args.source is None or args.frozen_cache is None or args.q_only_weight_sweep or args.resume_checkpoint or args.q_coupling_weight:
            raise ValueError('Candidate sweep requires source/cache and no other ablations')
        from experiments.q_only_candidates import run
        return run(args.source,args.frozen_cache,args.output,args.epochs or 2400)
    if args.q_only_weight_sweep:
        if args.source is None or args.resume_checkpoint or args.q_coupling_weight:
            raise ValueError('Q-only sweep needs a source, no resume or supervision')
        from experiments.q_only_cached_sweep import run
        return run(args.source,args.output,args.epochs or 240)
    if not math.isfinite(args.q_coupling_weight) or args.q_coupling_weight<0:
        raise ValueError('Nonnegative finite Q coupling weight required')
    if not math.isfinite(args.variance_weight) or args.variance_weight<0:raise ValueError('Nonnegative variance weight required')
    if not math.isfinite(args.color_weight) or args.color_weight<0:raise ValueError('Nonnegative color weight required')
    if not 0<args.variance_ratio<=1:raise ValueError('Variance ratio must be in (0,1]')
    raw_config=(json.loads(args.config.read_text()) if args.config else
                json.loads((args.source/'manifest.json').read_text())['config'])
    resume=torch.load(args.resume_checkpoint,map_location='cpu',weights_only=True) if args.resume_checkpoint else None
    cfg=(RunConfig.load(args.config) if args.config else RunConfig.from_dict(raw_config))
    cfg=default_representation(cfg,raw_config,continuing=bool(args.resume_checkpoint or args.freeze_c_checkpoint))
    cfg=default_closure(cfg,raw_config,
        continuing=bool(args.resume_checkpoint or args.freeze_c_checkpoint),
        historical=bool(args.source or args.ising_b0_ablation is not None
            or args.all_start_closure_ablation or args.allow_nondefault_recipe
            or args.q_coupling_weight or args.q_future_weight or args.q_future_control
            or args.q_future_pairs or args.q_future_normalizer))
    cfg=resume_closure(cfg,raw_config,resume)
    cfg=default_terminal_r(cfg,raw_config,resume=resume,
        historical=bool(args.source or args.freeze_c_checkpoint or args.ising_b0_ablation is not None
            or args.all_start_closure_ablation or args.allow_nondefault_recipe or args.q_coupling_weight
            or args.q_future_weight or args.q_future_control or args.q_future_pairs or args.q_future_normalizer
            or args.benchmark_warmup is not None))
    if cfg.training.terminal_r_fit and (cfg.loss.closure_objective!='all_start_anchor' or cfg.loss.closure_metric!='euclidean'):
        raise ValueError('Terminal R fitting requires all_start_anchor/euclidean; pin terminal_r_fit=false for a legacy-objective reproduction')
    validate_future_options(args,cfg)
    if args.affine_pca_init and not cfg.model.affine_reconstruction:
        raise ValueError('PCA initialization requires affine_reconstruction=true')
    validate_recipe(args,cfg)
    if args.epochs is not None:
        if args.epochs<1:raise ValueError('epochs must be positive')
        cfg=replace(cfg,training=replace(cfg.training,epochs=args.epochs))
    if args.theory_dim is not None:
        if args.theory_dim<1:raise ValueError('theory-dim must be positive')
        cfg=replace(cfg,model=replace(cfg.model,theory_dim=args.theory_dim))
    cfg=replace(cfg,training=replace(cfg.training,gradient_mode='qc_r_separate'),
        loss=replace(cfg.loss,kernel_sign_weight=args.color_weight,kernel_neutral_weight=0.,
            kernel_noncollapse_mode='variance_floor_ensemble' if args.variance_per_ensemble else 'variance_floor',kernel_variance_ratio=args.variance_ratio,
            weights=replace(cfg.loss.weights,kernel_noncollapse=args.variance_weight)))
    categorical=bool(cfg.model.categorical_states)
    if args.environment!='full_shell' and (not categorical or cfg.model.dimension!=2 or args.estimator!='histogram' or args.mi_initialize):
        raise ValueError('Local-pair MI requires 2D categorical histogram training without kernel search')
    if cfg.model.fixed_kernel or cfg.model.kernel_sampling not in ('bernoulli','categorical') or cfg.model.kernel_layout!='nonoverlapping':
        raise ValueError('Requires learnable nonoverlapping finite-alphabet probabilistic C')
    block=cfg.model.block_factor
    if block!=2 and (cfg.model.dimension!=2 or args.estimator!='histogram' or args.environment!='full_shell' or args.mi_initialize):
        raise ValueError('Non-two blocks require 2D full-shell histogram MI without kernel search')
    if rsmi_enabled and (cfg.model.categorical_states or 2)**(block**cfg.model.dimension)>262144:
        raise ValueError('Empirical MI kernel table exceeds 262144 input patterns')
    cfg.validate()
    # Historical fields may remain readable, but retired objectives must fail
    # before allocating statistics, loading data, or creating an output run.
    cfg.loss.objective_kwargs()
    t,e=cfg.training,cfg.evaluation
    from recurrent_rg.learning_rate import LearningRatePolicy
    lr_policy=LearningRatePolicy(t)
    plateau_settings=dict(minimum=4000,window=2000,confirmations=3,check_every=200,
                          rtol=.01,rebound_rtol=.05,loss_floor=1e-3)
    if args.joint_train_plateau:
        conflicts=('freeze_c_checkpoint','affine_pca_init','q_coupling_weight',
                   'allow_nondefault_recipe')
        if (any(getattr(args,key) for key in conflicts) or not args.independent_c or
                args.estimator!='histogram' or set(t.train_modules)!={'Q','C','R'} or
                not cfg.model.affine_reconstruction or cfg.loss.balanced_reconstruction_weight or cfg.loss.isotropy_weight or
                t.log_every!=plateau_settings['check_every'] or
                t.checkpoint_every!=plateau_settings['check_every']):
            raise ValueError('Joint plateau requires the complete default affine Q/C/R recipe and log/checkpoint_every=200')
    exact=args.estimator=='histogram'
    if categorical and not exact:raise ValueError('Categorical experiment requires histogram MI')
    if args.mi_initialize and not exact:raise ValueError('MI initialization requires histogram estimator')
    sparse=rsmi_enabled and exact and cfg.model.dimension in (2,3)
    torch.set_num_threads(t.threads);torch.manual_seed(t.seed)
    m=build_model(cfg.model,t.max_rg_steps).to(device)
    random_construction_sha256=(qcr_module_sha256(m) if not args.resume_checkpoint else None)
    tx,vx=prepared_data(cfg,mmap=not args.reference_execution);fn=pattern_function(cfg)
    if device.type=='cuda' and cfg.loss.balanced_reconstruction_weight:
        raise ValueError('Balanced historical ablation is not yet CUDA-adapted')
    m.detach_kernel_representation=args.independent_c
    from recurrent_rg.execution import configure_cuda_execution, resident_training_data
    from recurrent_rg.execution import reference_binary_pattern_distribution, select_training_batch, low_ram_execution_policy
    execution=configure_cuda_execution(m,device,reference=args.reference_execution,
                                      independent_c=args.independent_c)
    execution['low_ram_execution']=low_ram_execution_policy(reference=args.reference_execution)
    low_ram=execution['low_ram_execution']['enabled']
    if args.mi_test_cpu:execution['mi_test_device']='cpu'
    if args.reference_execution and not categorical:
        fn=reference_binary_pattern_distribution
    data_sha256=file_fingerprint(cfg.data.cache)
    future_identities,future_contract=load_future_inputs(args,data_sha256,
        dict(train=len(tx.configurations),test=len(vx.configurations)))
    future_encoding_kwargs=future_encoding_execution(args.q_future_weight,execution)
    if future_encoding_kwargs:
        future_contract['encoding_execution']=dict(future_encoding_kwargs)
        execution.update(future_encoding_batch_chunk=future_encoding_kwargs['encoding_batch_chunk'],
                         future_validated_binary_inputs=future_encoding_kwargs['validated_binary_inputs'])
    future_predictor=None
    if future_identities is not None:
        for ds in (tx,vx):validate_future_data(ds.configurations)
    validate_future_normalizer_model(future_contract,cfg,m,tx.configurations)
    if args.q_future_weight:
        from recurrent_rg.future_physics import FuturePhysicsPredictor
        # Auxiliary initialization must not consume the Q/C/R sampling stream.
        with isolated_rng('cpu',t.seed+9011):
            future_predictor=FuturePhysicsPredictor(cfg.model.theory_dim,
                hidden=args.q_future_hidden,
                target_mean=torch.tensor(future_contract['target_mean'],dtype=torch.float64),
                target_std=torch.tensor(future_contract['target_std_unfloored'],dtype=torch.float64),
                std_floor=future_contract['std_floor']).double().to(device)
    future_parameters=[] if future_predictor is None else list(future_predictor.parameters())
    frozen_source=None
    if args.freeze_c_checkpoint:
        frozen_source=load_run(args.freeze_c_checkpoint)
        if frozen_source.model_config!=cfg.model or frozen_source.config['training']['max_rg_steps']!=t.max_rg_steps:
            raise ValueError('Frozen-C warm start requires the same Q/C/R architecture and scales')
        old_data=frozen_source.payload.get('metadata',{}).get('data_sha256')
        if old_data is None:
            old_data=file_fingerprint(frozen_source.config['data']['cache'])
        if old_data!=file_fingerprint(cfg.data.cache):
            raise ValueError('Frozen-C warm start data fingerprint differs')
        m.load_state_dict(qcr_state_dict(frozen_source.model),strict=False)
        for parameter in m.rg_kernel.parameters():parameter.requires_grad_(False)
    critic=None
    if rsmi_enabled:
        _,env=region_offsets(cfg.model.dimension,block=block,buffer=1)
        # Extra critic initialization must not change QCR's sampling RNG stream.
        with isolated_rng('cpu',t.seed+1):
            critic=EnvironmentCritic(len(env),64).double().to(device)
    qc=list(m.theory_encoder.parameters())+([] if frozen_source else list(m.rg_kernel.parameters()))
    r=list(m.theory_rg.parameters());cp=[] if exact or not rsmi_enabled else list(critic.parameters())
    opt=torch.optim.Adam(qc+r+future_parameters,lr=t.learning_rate);co=None if exact or not rsmi_enabled else torch.optim.Adam(cp,lr=t.learning_rate)
    statistics=categorical_lattice_mi_statistics if categorical else binary_lattice_mi_statistics if sparse else binary_chain_joint_histograms
    if args.environment=='local_pairs':statistics=categorical_local_pair_mi_statistics
    statistics_kwargs={'block':block} if sparse and args.environment=='full_shell' else {}
    histograms=None
    execution['mi_statistics_cache']={}
    if resume is not None and resume.get('metadata',{}).get('data_sha256')!=data_sha256:
        raise ValueError('Resume data changed')
    if resume is not None:validate_resume_mi_placement(resume,execution)
    if resume is not None:validate_resume_low_ram_execution(resume,execution['low_ram_execution'])
    execution['selected_empirical_mi']=rsmi_enabled and selected_mi_execution(execution,sparse,resume)
    if rsmi_enabled and exact:
        histograms={}
        for split,x in [('train',tx.configurations),('test',vx.configurations)]:
            build=lambda x=x:statistics(x,flip=cfg.data.spin_flip_augment,**statistics_kwargs)
            if execution['enabled']:
                from recurrent_rg.mi_cache import load_or_build_mi_statistics, lattice_statistics_cache_key
                key_data=lattice_statistics_cache_key(cfg,data_sha256,split,x,env,
                    statistics.__name__,statistics_kwargs,environment=args.environment)
                cache_dir=args.mi_cache_dir or Path(cfg.data.cache).parent/'.qcr_mi_cache'
                hist,cache_record=load_or_build_mi_statistics(cache_dir,key_data,build,mmap=low_ram)
                execution['mi_statistics_cache'][split]=cache_record
                print(f'MI statistics {split}: '+('cache hit' if cache_record['hit'] else 'built and cached'),flush=True)
            else:
                hist=build()
            histograms[split]=(place_mi_histogram_split(hist,split,device,
                selected=execution['selected_empirical_mi'],test_cpu=args.mi_test_cpu) if low_ram else hist)
            # In the CUDA low-RAM route, retain only the placed statistics and
            # required CPU support indices, not an extra CPU histogram object.
            del hist
    # Histogram construction/compression stays on CPU; only its existing tensors
    # move once. No change to the empirical MI estimator or sampled ensembles.
    if histograms is not None and not low_ram and execution['selected_empirical_mi']:
        histograms['train'].prepare_selection()
    if histograms is not None and not low_ram:
        histograms=place_mi_histograms(histograms,device,test_cpu=args.mi_test_cpu)
    train_configurations,residency=resident_training_data(tx.configurations,device,
                                                        enabled=execution['enabled'])
    execution['training_data']=residency
    if resume is not None and bool(resume['metadata'].get('joint_train_plateau'))!=args.joint_train_plateau:
        raise ValueError('Exact resume must preserve the joint TRAIN plateau stopping convention')
    initialization=None
    if resume is not None:
        old=resume['metadata']
        validate_resume_future(resume,future_contract,device,execution)
        validate_resume_supervision(resume,args,cfg,device=device,
            data_sha256=data_sha256,execution=execution)
        if device.type=='cuda':
            # Older CUDA checkpoints predate these accelerations. Preserve their
            # original execution via --reference-execution for an exact resume.
            old_reference=old.get('execution_optimizations',{}).get('reference',True)
            if old_reference!=args.reference_execution:
                raise ValueError('Exact CUDA resume must preserve reference-execution; '
                                 'older unoptimized CUDA checkpoints require --reference-execution')
        old_config=RunConfig.from_dict(old['config']).to_dict()
        old_config['training']['epochs']=cfg.training.epochs
        expected_estimator=args.estimator if rsmi_enabled else None
        expected_environment=args.environment if rsmi_enabled else None
        if old_config!=cfg.to_dict() or old['estimator']!=expected_estimator or old.get('environment_mode','full_shell')!=expected_environment or old.get('independent_c',False)!=args.independent_c or old['rsmi_weight']!=args.rsmi_weight:
            raise ValueError('Resume requires identical configuration, MI estimator/weight, and gradient routing')
        if t.epochs<=resume['epoch']:raise ValueError('Resume total epochs must exceed checkpoint epoch')
        initialization=old.get('data_driven_initialization')
        if bool(initialization)!=args.mi_initialize:raise ValueError('Resume initialization flag mismatch')
        load_model_state(m,resume);opt.load_state_dict(resume['optimizer'])
        if t.r_updates_per_batch > 1:
            from recurrent_rg.r_refinement import component_optimizer_steps
            clocks=component_optimizer_steps(m,opt)['R']
            if clocks['min']!=clocks['max'] or clocks['min']!=resume['epoch']*t.r_updates_per_batch:
                raise ValueError('Resume R Adam clocks do not match the saved inner-update schedule')
        if future_predictor is not None:
            future_predictor.load_state_dict(resume['future_physics_predictor_state'],strict=True)
        if co is not None:critic.load_state_dict(resume['critic']);co.load_state_dict(resume['critic_optimizer'])
        lr_policy.validate_resume(resume,opt)
        if co is not None:lr_policy.validate_resume(resume,co)
    elif args.mi_initialize:
        table,initialization=(optimize_categorical_rule(histograms['train'],cfg.model.categorical_states,seed=t.seed) if categorical else
            (best_deterministic_binary_rule if sparse else best_binary_pair_kernel)(histograms['train']))
        initialization.update(initialize_kernel_from_table(m.rg_kernel,table))
        print('Data-driven MI initialization:',initialization,flush=True)
    patterns=None
    if rsmi_enabled:
        visible_sites=block**cfg.model.dimension
        states=cfg.model.categorical_states or 2
        codes=(torch.arange(states**visible_sites,device=device)[:,None]//(states**torch.arange(visible_sites,device=device)))%states
        patterns=(torch.nn.functional.one_hot(codes,states).flatten(1).double() if categorical else (2*codes-1).double())[None]
    probabilities=categorical_kernel_probabilities if categorical else kernel_probabilities
    frozen_information=None
    if frozen_source:
        with torch.no_grad():
            fixed_table=probabilities(m.rg_kernel,patterns)[0]
            frozen_information={split:float(hist.information(fixed_table).mean() if sparse else
                                histogram_mutual_information(hist,fixed_table).mean())
                                for split,hist in histograms.items()}
    gen=torch.Generator().manual_seed(t.seed);rg=torch.Generator().manual_seed(t.seed+2)
    pairs=min(64,tx.configurations.shape[1],vx.configurations.shape[1]);systems=min(16,len(tx.configurations))
    args.output.mkdir(parents=True,exist_ok=False)
    contract=dict(training_driver='rsmi_separate_qcr',rsmi_weight=args.rsmi_weight,rsmi_scales=[0],buffer=1,
        pairs=pairs,systems=systems,critic_hidden=64,critic_parameters=sum(p.numel() for p in cp),
        QCR_parameters=sum(p.numel() for p in m.theory_encoder.parameters())+sum(p.numel() for p in m.rg_kernel.parameters())+sum(p.numel() for p in r),gradient_QC='grad(L_QC - rsmi_weight*I_NCE)',
        gradient_R='grad(5 L_commutator + 5 L_rollout); no feedback to QC',
        gradient_critic='grad(-rsmi_weight*I_NCE)',negatives='Within ensemble only',
        kernel_auxiliaries='All disabled; RSMI replaces C loss',
        labels_used_in_training=bool(args.q_coupling_weight),source=str(args.source) if args.source else None,config_source=str(args.config) if args.config else None,selection='Final checkpoint; one seed',
        note='Extra region batches from the same cache, not extra generated data. Empirical InfoNCE is not exact MI.')
    contract['estimator']=args.estimator
    contract['rsmi_enabled']=rsmi_enabled
    contract['mi_evaluated']=rsmi_enabled
    contract['environment_mode']=args.environment
    contract['data_driven_initialization']=initialization
    contract['independent_c']=args.independent_c
    contract['execution_device']=str(device)
    contract['precision']='torch.float64'
    contract['execution_optimizations']=execution
    contract['learning_rate_policy']=lr_policy.description()
    from recurrent_rg.r_refinement import refinement_contract, refine_r_on_targets, component_optimizer_steps
    contract['R_optimization']=refinement_contract(t)
    contract['terminal_R_fit']=dict(enabled=t.terminal_r_fit, max_extra_R_updates=t.terminal_r_max_steps,
        fixed_TRAIN_cache_repeats=t.terminal_r_cache_repeats, same_objective=True,
        optimizer='Continue R Adam and current LR; Q/C weights, gradients and clocks frozen',
        stopping='TRAIN plateau only; cap is reported separately; resource limits are not extended')
    contract['skip_final_report']=args.skip_final_report
    contract['closure_objective_contract'] = closure_objective_contract(cfg)
    contract['gradient_R'] = contract['closure_objective_contract']['gradient_R']
    if args.all_start_closure_ablation:
        contract['closure_ablation'] = dict(
            objective=cfg.loss.closure_objective, four_scale_points=True,
            block_factor=2, prediction_prediction_penalty=False,
            labels_used=False, gradient_routing='R only; no feedback to Q/C',
            formula=('10/S * sum_t mean_{s<t} ||R^(t-s)(g_s)-g_t||^2'
                     if cfg.loss.closure_objective == 'all_start_anchor' else
                     '5/S * sum_t [||R(g_(t-1))-g_t||^2 + ||R^t(g_0)-g_t||^2]'),
            target_scale_weights='uniform; equal total weight per endpoint')
        if cfg.loss.closure_objective == 'all_start_anchor':
            contract['gradient_R']='grad(10 L_all_start_anchor); no feedback to QC'
    contract['device_note']='CPU histogram construction; model, selected batches, kernel table and MI tensor algebra on execution_device. Reports reload on CPU.'
    if args.mi_test_cpu:
        contract['device_note']+=' Full TEST empirical MI is logged on CPU using the detached current kernel table; TRAIN MI remains on CUDA, unchanged.'
    contract['kernel_auxiliaries']=f'Multiscale {cfg.loss.kernel_noncollapse_mode} weight={args.variance_weight}, ratio={args.variance_ratio}; uniform-color NLL weight={args.color_weight}; neutral disabled; distinct retired'
    if exact:
        contract.update(critic_hidden=None,gradient_critic=None,negatives=None,pairs=None,
            gradient_QC='grad(L_QC - rsmi_weight*I_histogram)',
            note='Exact MI of empirical per-ensemble 4x4 P(V,E), all cached samples/translations, with original flip augmentation. Not exact population MI. No critic or coupling labels.')
    if args.independent_c:
        contract['gradient_QC']='Q: representation losses only; C: -rsmi_weight*I + variance_weight*variance_floor + color_weight*uniform_color_NLL only'
    if sparse:
        contract['empirical_statistics']={split:hist.diagnostics for split,hist in histograms.items()}
        contract['note']='Exact full-shell empirical MI via compressed P(V|E), per-ensemble weights, all cached samples/translations; no critic or coupling labels. Large finite-sample bias is possible.'
        if args.environment=='local_pairs':contract['note']='Mean of four buffered two-site environment MI objectives, not full-shell joint MI. Finite-cache bias must be audited independently.'
    if not rsmi_enabled:
        contract.update(estimator=None,environment_mode=None,rsmi_scales=[],buffer=None,
            critic_hidden=None,critic_parameters=0,gradient_critic=None,negatives=None,pairs=None,
            gradient_QC='Q: representation losses only; C: variance_weight*variance_floor + color_weight*uniform_color_NLL only',
            device_note='No MI statistics/cache, critic, kernel MI table, or MI evaluation; model and selected batches on execution_device. Reports reload on CPU.',
            note='Explicit zero-RSMI ablation. MI is not computed in setup, training, or evaluation; null MI logs are unmeasured, not zero information. No critic or coupling labels.')
    if frozen_source:
        contract['gradient_QC']='Q: representation only; C: frozen learned weights; R: configured closure only'
        contract['frozen_C']=dict(checkpoint=str(frozen_source.checkpoint),
            checkpoint_sha256=file_fingerprint(frozen_source.checkpoint),
            warm_start='Q/C/R weights; Adam reset; affine decoder optionally reinitialized',
            stochastic_sampling='Fresh independent C draws at each training step, not a frozen trajectory cache',
            empirical_MI='Constant diagnostic only; no C updates',labels_used=False)
        contract['trainable_parameters']=sum(p.numel() for p in qc+r)
    campaign_contract=ising_b0_ablation_contract(args,cfg)
    if campaign_contract is not None:
        contract['ising_b0_ablation']=campaign_contract
        contract['gradient_R']=(f'grad({cfg.loss.weights.commutator:g} L_commutator + '
            f'{cfg.loss.rollout_weight:g} L_rollout); no feedback to QC')
    add_supervision_contract(contract,args.q_coupling_weight)
    if future_contract is not None:
        contract['q_future_physics']=future_contract
        contract['auxiliary_future_parameters']=sum(p.numel() for p in future_parameters)
        contract['trainable_parameters']=sum(p.numel() for p in qc+r+future_parameters)
        if future_predictor is not None:
            contract['gradient_QC']+='; Q auxiliary future-physics objective; predictor gradients only from its objective'
        else:
            contract['gradient_QC']+='; paired independent-chain sampler control, no auxiliary loss or parameters'
    scale_contract = closure_scale_contract(cfg)
    pre_affine_sha256=qcr_module_sha256(m) if resume is None else None
    if scale_contract is not None:
        contract['closure_scale_weighting'] = scale_contract
        if resume is None:
            contract['initial_module_sha256'] = dict(pre_affine_sha256)
    metadata=dict(config=cfg.to_dict(),data_sha256=data_sha256,**contract)
    if args.joint_train_plateau:
        metadata['joint_train_plateau']=dict(**plateau_settings,
            stopping_split='TRAIN only; TEST and critical exponents not used',
            panel='Fixed evenly spaced TRAIN ensembles; fixed half-subsets; C sampling seed12345; full TRAIN empirical MI',
            objectives=('Q: affine reconstruction+subset; C: -2 MI+2 variance+color; R: '
                        + contract['closure_objective_contract']['gradient_R']),
            optimizer=('Original constant Adam learning rate; no scheduler, freezing or gradient changes'
                       if lr_policy.kind=='constant' else
                       f'Original Adam parameters/moments with deterministic cumulative {lr_policy.kind} learning rates; no freezing or gradient changes'),
            interpretation='Operational local plateau, not global optimality or exact physical scaling')
        if resume is not None and resume['metadata'].get('joint_train_plateau')!=metadata['joint_train_plateau']:
            raise ValueError('Joint plateau exact resume requires identical stopping settings')
    if cfg.model.q_encoder == 'ensemble_vae':
        from recurrent_rg.ensemble_vae import calibrate_from_train
        validate_ensemble_vae_training(args)
        if resume is None and not frozen_source:
            calibration = calibrate_from_train(m.theory_encoder, tx.configurations)
        else:
            calibration = dict(restored_from_checkpoint=True, labels_used=False)
        metadata['q_reconstruction'] = dict(
            objective='Shared cross-subset normalized histogram CE + .01 KL; equal mean over all scales',
            kl_warmup_updates=600, calibration=calibration,
            architecture=f'{len(m.theory_encoder.center)}-96-96 Gaussian Q; {cfg.model.theory_dim}-64-64-{len(m.theory_encoder.center)} LeakyReLU probability decoder',
            original_subset_weights_disabled=True, TC_weight=0,
            gradient='Q/decoder only; histograms detached; no Q gradient into C',
            encoding='R and reports use half-mean posterior means, never posterior samples',
            sampling='Private update/scale-indexed posterior noise; C/data RNG unchanged',
            initialization='Fresh random Q; TRAIN microscopic histogram centering/scaling and decoder bias only',
            logging='separation=mean CE+warmup*.01 KL; vae_cross_entropy/KL/effective_KL_weight per scale')
        if args.q_coupling_weight:
            metadata['q_reconstruction']['additional_microscopic_supervision'] = metadata['q_supervision']
    if cfg.model.affine_reconstruction:
        pca_initialization=None
        if args.affine_pca_init:
            from recurrent_rg.representation import initialize_affine_decoder_pca_
            with isolated_rng(device,t.seed+9000), torch.no_grad():
                hist=[]
                for x in tx.configurations:
                    hist.extend(fn(level) for level in m.coarse_grain(x[None].to(device=device,dtype=torch.float64)))
                pca_initialization=initialize_affine_decoder_pca_(m.theory_encoder,torch.cat(hist))
                pca_initialization.update(fitting_split='train',all_scales=True,sampling_seed=t.seed+9000)
            print('TRAIN-only affine PCA initialization:',pca_initialization,flush=True)
        elif resume is None and not frozen_source:
            # Only a bias calibration from TRAIN distributions at random C;
            # no kernel/encoder pretraining, labels, or change to training RNG.
            with isolated_rng(device,t.seed+9000), torch.no_grad():
                levels=m.coarse_grain(tx.configurations[:min(32,len(tx.configurations))].to(device=device,dtype=torch.float64))
                center=torch.stack([fn(level).mean(0) for level in levels]).mean(0)
                m.theory_encoder.distribution_decoder.bias.copy_(center)
        metadata['q_reconstruction']=dict(
            objective='Shared affine probability reconstruction sum-bin MSE at all scales; replaces geometric distances',
            weight=cfg.loss.weights.separation,parameters=sum(p.numel() for p in m.theory_encoder.distribution_decoder.parameters()),
            gauge='Each decoder column norm1, projected after each optimizer update; no full-rank or isotropy constraint',
            initialization='Random decoder direction; bias=train-only mean motif distribution at initially random C; preserves sampling RNG',
            gradient='Q and auxiliary decoder only; target detached; default C/R objectives unchanged',
            logging='separation/data_geometry_level_* record affine reconstruction MSE; reconstruction_mse_level_* explicit')
        if pca_initialization:
            metadata['q_reconstruction']['initialization']='TRAIN-only histogram PCA direction and mean at the frozen learned C; no Q pretraining'
            metadata['q_reconstruction']['PCA']=pca_initialization
    if cfg.model.reconstruction_hidden:
        metadata['q_reconstruction']=dict(
            objective='Shared decoder JS to detached same-scale local-pattern histograms; replaces geometry at every scale',
            weight=cfg.loss.weights.separation,hidden=cfg.model.reconstruction_hidden,
            parameters=sum(p.numel() for p in m.theory_encoder.distribution_decoder.parameters()),
            gradient='Q and auxiliary decoder only; C/R objectives unchanged',
            logging='separation/data_geometry_level_* now record reconstruction JS, not distance loss')
        if cfg.model.reconstruction_probability_loss == 'squared':
            if resume is None and not frozen_source:
                # Match affine TRAIN-only bias calibration, in log-probability
                # coordinates. No labels, pretraining, or sampling RNG changes.
                with isolated_rng(device,t.seed+9000), torch.no_grad():
                    levels=m.coarse_grain(tx.configurations[:min(32,len(tx.configurations))].to(device=device,dtype=torch.float64))
                    center=torch.stack([fn(level).mean(0) for level in levels]).mean(0)
                    m.theory_encoder.distribution_decoder[-1].bias.copy_(center.clamp_min(1e-8).log())
            metadata['q_reconstruction'].update(
                objective='Shared nonlinear softmax probability reconstruction sum-bin MSE at all scales',
                hidden_layers=cfg.model.reconstruction_hidden_layers,
                initialization='Random decoder; final bias=log TRAIN-only mean motif probability at initially random C, floor1e-8; preserves sampling RNG',
                logging='separation/data_geometry_level_* and reconstruction_mse_level_* record sum-bin probability MSE')
        if cfg.model.reconstruction_spectral_bound:
            metadata['q_reconstruction'].update(
                gauge='Exact spectral norm cap after initialization and each Q update; no latent rank/isotropy constraint',
                spectral_bound_per_layer=cfg.model.reconstruction_spectral_bound,
                probability_decoder_lipschitz_bound=.5*cfg.model.reconstruction_spectral_bound**(cfg.model.reconstruction_hidden_layers+1))
    if cfg.loss.balanced_reconstruction_weight:
        if not args.independent_c or args.q_coupling_weight or args.affine_pca_init:
            raise ValueError('Balanced full-QCR test requires independent gradients, no labels/PCA initialization')
        metadata['q_balanced_reconstruction']=dict(
            weight=cfg.loss.balanced_reconstruction_weight, additive_to_original_MSE=True,
            max_directions=cfg.model.theory_dim, signal_noise_threshold=5., max_weight_ratio=20.,
            fitting_split='All TRAIN ensembles; no TEST or coupling labels',
            refresh_every=t.checkpoint_every, sampling_seed=9191,
            gradient='Q/affine decoder only; current-C targets and metric detached',
            initialization='No change to model initialization; no PCA weight initialization',
            metric='Same signal-gated observable covariance metric as paired Q-only pilot; '
                   'refreshed periodically because C is now trainable')
    if args.q_coupling_weight:
        for ds in (tx,vx):
            if ds.couplings is None or ds.couplings.reshape(len(ds.configurations),-1).shape[1]!=cfg.model.theory_dim:
                raise ValueError('Supervision requires coupling dimension equal to dim g')
        metadata['q_supervision']['coupling_names']=list(cfg.data.coupling_names)
    if resume is not None:
        metadata['resumed_from']=str(args.resume_checkpoint)
    metric_history=[]
    if cfg.loss.balanced_reconstruction_weight:
        from recurrent_rg.balanced_reconstruction import refresh_training_metrics
        if resume is not None:
            if not hasattr(m,'balanced_reconstruction_metrics'):
                raise ValueError('Balanced exact resume requires saved TRAIN metric state')
        else:
            refresh_training_metrics(m,tx.configurations,fn,samples=t.sample_batch,
                                     fitted_epoch=0)
        metric_history.append(dict(fitted_epoch=m.balanced_metric_fitted_epoch,
            scales=[metric.description() for metric in m.balanced_reconstruction_metrics]))
        print('TRAIN-only balanced metric ranks: '+str([metric.axes.shape[1]
              for metric in m.balanced_reconstruction_metrics]),flush=True)
    record_initial_module_provenance(metadata,m,
        random_construction_sha256=random_construction_sha256,
        pre_affine_sha256=pre_affine_sha256,resume=resume)
    sync_initial_module_provenance(contract,metadata)
    (args.output/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    if metric_history:
        (args.output/'balanced_metric_history.json').write_text(json.dumps(metric_history,indent=2)+'\n')
    evaluation_ids={split:torch.linspace(0,len(ds.configurations)-1,min(e.systems,len(ds.configurations))).long()
                    for split,ds in [('train',tx),('test',vx)]}
    future_eval_ids=None
    if future_identities is not None:
        future_eval_ids={split:future_batch_ids(ids,e.systems) for split,ids in future_identities.items()}
        evaluation_ids=future_eval_ids
    # Preserve one complete ordered panel and one evaluate() call per split;
    # do not change its shape, sample count, C RNG draws or loss reduction.
    evaluation_panels=(None if low_ram else {
        split:ds.configurations[evaluation_ids[split]] for split,ds in [('train',tx),('test',vx)]})
    history=[] if resume is None else resume['history'];started=time.perf_counter()
    plateau_rows=[] if resume is None else resume.get('joint_plateau_rows',[])
    if args.joint_train_plateau and resume is not None and not plateau_rows:
        raise ValueError('Missing joint plateau history in resume checkpoint')
    plateau_diagnostics=None
    if resume is not None:
        gen.set_state(resume['batch_generator_state']);rg.set_state(resume['rsmi_generator_state'])
        restore_checkpoint_rng(resume,device=device)

    def synchronize():
        if device.type=='cuda':torch.cuda.synchronize(device)

    timing_enabled=args.benchmark_warmup is not None
    step_seconds=[];logging_seconds=0.;checkpoint_seconds=0.
    if timing_enabled or args.max_training_seconds is not None:synchronize()
    setup_seconds=time.perf_counter()-wall_started
    training_started=time.perf_counter()
    benchmark_memory_snapshot(args.benchmark_warmup,device,'after_setup')

    def save(step,name):
        state=checkpoint_payload(m,opt,gen,step,metadata=metadata,history=history)
        state['optimizer_steps']=component_optimizer_steps(m,opt)
        state.update(rsmi_generator_state=rg.get_state(),
                     lr_scheduler_state=lr_policy.checkpoint_state(step))
        if args.joint_train_plateau:
            state.update(joint_plateau_rows=plateau_rows,
                         joint_plateau_diagnostics=plateau_diagnostics)
        if co is not None:state.update(critic=critic.state_dict(),critic_optimizer=co.state_dict())
        if future_predictor is not None:
            state['future_physics_predictor_state']=future_predictor.state_dict()
        torch.save(state,args.output/name)

    completed_epoch=0 if resume is None else resume['epoch']
    stopped_by_wall_limit=False
    stopped_by_training_limit=False
    stopped_by_train_plateau=False
    for step in range(1 if resume is None else resume['epoch']+1,t.epochs+1):
        if cfg.model.q_encoder == 'ensemble_vae':
            m.theory_encoder.set_objective_step(step)
        if cfg.loss.balanced_reconstruction_weight and step>1 and (step-1)%t.checkpoint_every==0:
            refresh_training_metrics(m,tx.configurations,fn,samples=t.sample_batch,
                                     fitted_epoch=step-1)
            metric_history.append(dict(fitted_epoch=step-1,
                scales=[metric.description() for metric in m.balanced_reconstruction_metrics]))
            (args.output/'balanced_metric_history.json').write_text(json.dumps(metric_history,indent=2)+'\n')
            print(f'Balanced metric refreshed at {step-1}; TRAIN ranks '+str([
                metric.axes.shape[1] for metric in m.balanced_reconstruction_metrics]),flush=True)
        if timing_enabled:synchronize()
        update_started=time.perf_counter()
        # Constant uses the historical Adam path without even rewriting LR.
        # Scheduling uses cumulative update numbers, not elapsed wall time.
        if lr_policy.kind!='constant':
            lr_policy.apply_update(opt,step)
            if co is not None:lr_policy.apply_update(co,step)
        m.train()
        if critic is not None:critic.train()
        ids=(torch.randperm(len(tx.configurations),generator=gen)[:t.system_batch] if future_identities is None else
             future_batch_ids(future_identities['train'],t.system_batch,gen))
        indices=torch.randperm(tx.configurations.shape[1],generator=gen)[:t.sample_batch]
        selected=select_training_batch(train_configurations,ids,indices,device,reference=not low_ram);half=t.sample_batch//2
        a,b=selected[:,:half],selected[:,half:]
        batch=torch.cat((a,-a,b,-b),1) if cfg.data.spin_flip_augment else selected
        opt.zero_grad(set_to_none=True)
        if co is not None:co.zero_grad(set_to_none=True)
        future_levels=[] if future_predictor is not None else None
        future_kwargs={} if future_levels is None else dict(levels_sink=future_levels)
        closure_targets=[] if t.r_updates_per_batch > 1 else None
        if closure_targets is not None:future_kwargs['closure_targets_sink']=closure_targets
        base,terms=qcr_objective(m,batch,pattern_fn=fn,**cfg.loss.objective_kwargs(),**future_kwargs)
        benchmark_memory_snapshot(args.benchmark_warmup,device,'afterbase',step=step)
        if future_predictor is not None:
            future_predictor.train()
            auxiliary,future_terms=future_batch_loss(future_predictor,m,future_levels,
                future_identities['train'],ids,args.q_future_weight,**future_encoding_kwargs)
            base=base+auxiliary
            terms.update(future_terms)
        benchmark_memory_snapshot(args.benchmark_warmup,device,'afteraux',step=step)
        if args.q_coupling_weight:
            supervised=microscopic_coupling_loss(m.theory_encoder,batch,tx.couplings[ids])
            base=base+args.q_coupling_weight*supervised
        if not rsmi_enabled:
            info=None
        elif frozen_source:
            info=frozen_information['train']
        elif exact:
            selected_hist=torch.randperm(len(tx.configurations),generator=rg)[:systems]
            table=probabilities(m.rg_kernel,patterns)[0]
            info=(histograms['train'].information_for(table,selected_hist).mean()
                  if execution['selected_empirical_mi'] else
                  histograms['train'].information(table)[selected_hist].mean() if sparse else
                  histogram_mutual_information(histograms['train'][selected_hist],table).mean())
        else:
            v,envs=sample_regions(tx.configurations,systems,pairs,rg,buffer=1,flip=cfg.data.spin_flip_augment)
            info=enumerated_infonce(kernel_probabilities(m.rg_kernel,v),critic(envs)).mean()
        benchmark_memory_snapshot(args.benchmark_warmup,device,'afterMI',step=step)
        mi_objective=-args.rsmi_weight*info if rsmi_enabled else 0.
        total=base+mi_objective
        closure=terms.get('closure_total',
            cfg.loss.weights.commutator*terms['commutator']+cfg.loss.rollout_weight*terms['rollout'])
        if not torch.isfinite(total):raise FloatingPointError(f'Nonfinite at {step}')
        if frozen_source:
            separate_qc_r_gradients(total,closure,list(m.theory_encoder.parameters()),r)
        elif args.independent_c:
            independent_component_gradients(total,closure,
                mi_objective+args.variance_weight*terms['kernel_noncollapse']+args.color_weight*terms.get('kernel_sign',0.),
                list(m.theory_encoder.parameters())+future_parameters,m.rg_kernel.parameters(),r,cp)
        else:
            separate_qc_r_gradients(total,closure,qc+cp,r)
        opt.step()
        from recurrent_rg.representation import normalize_affine_decoder_, project_probability_decoder_
        normalize_affine_decoder_(m.theory_encoder)
        project_probability_decoder_(m.theory_encoder)
        if co is not None:co.step()
        r_batch_after=None
        if closure_targets is not None:
            if len(closure_targets)!=1:raise RuntimeError('Missing unique online R target snapshot')
            r_batch_after=refine_r_on_targets(m,opt,closure_targets[0],cfg.loss,t.r_updates_per_batch-1)
        benchmark_memory_snapshot(args.benchmark_warmup,device,'afteropt',step=step)
        if timing_enabled:
            synchronize();step_seconds.append(time.perf_counter()-update_started)
        completed_epoch=step
        stopped_by_wall_limit=args.max_wall_seconds is not None and time.perf_counter()-wall_started>=args.max_wall_seconds
        stopped_by_training_limit=args.max_training_seconds is not None and time.perf_counter()-training_started>=args.max_training_seconds
        if step==1 or step%t.log_every==0 or step==t.epochs or stopped_by_wall_limit or stopped_by_training_limit:
            logging_started=time.perf_counter()
            row=dict(epoch=step,learning_rate=opt.param_groups[0]['lr'])
            if r_batch_after is not None:
                row.update(r_batch_before=float(closure.detach()),r_batch_after=r_batch_after,
                           r_updates_per_batch=t.r_updates_per_batch)
                for name,clock in component_optimizer_steps(m,opt).items():
                    row[name+'_updates_min']=clock['min'];row[name+'_updates_max']=clock['max']
            m.eval()
            if critic is not None:critic.eval()
            for split,full in [('train',tx.configurations),('test',vx.configurations)]:
                x=full[evaluation_ids[split]] if low_ram else evaluation_panels[split]
                eval_levels=[] if future_predictor is not None else None
                eval_kwargs=cfg.loss.objective_kwargs()
                if eval_levels is not None:eval_kwargs['levels_sink']=eval_levels
                metrics=evaluate(m,x,fn,e.samples,spin_flip_augment=cfg.data.spin_flip_augment,objective_kwargs=eval_kwargs)
                if future_predictor is not None:
                    future_predictor.eval()
                    metrics['qcr_total_without_future']=metrics['total']
                    with torch.no_grad():
                        auxiliary,future_terms=future_batch_loss(future_predictor,m,eval_levels,
                            future_identities[split],future_eval_ids[split],args.q_future_weight,**future_encoding_kwargs)
                    metrics.update({key:float(value) for key,value in future_terms.items()})
                    metrics['weighted_future_physics']=float(auxiliary)
                    metrics['total']+=float(auxiliary)
                benchmark_memory_snapshot(args.benchmark_warmup,device,f'afterevaluate_{split}',step=step)
                if args.q_coupling_weight:
                    ds=tx if split=='train' else vx
                    eval_ids=torch.linspace(0,len(ds.configurations)-1,min(e.systems,len(ds.configurations))).long()
                    with torch.no_grad():
                        sup=float(microscopic_coupling_loss(m.theory_encoder,x[:,:e.samples].to(device=device,dtype=torch.float64),ds.couplings[eval_ids]))
                    metrics['coupling_supervision']=sup
                    metrics['weighted_coupling_supervision']=args.q_coupling_weight*sup
                    metrics['total']+=args.q_coupling_weight*sup
                if not rsmi_enabled:
                    info=shuffled=None
                elif frozen_source:
                    info=frozen_information[split];shuffled=0.
                elif exact:
                    with torch.no_grad():
                        table=probabilities(m.rg_kernel,patterns)[0]
                        hist=histograms[split]
                        table=logged_mi_table(table,split,test_cpu=args.mi_test_cpu)
                        if sparse:
                            info=float(hist.information(table).mean())
                            # A zero is not a measured shuffled finite-cache
                            # null. In 3D the full shell is often nearly unique;
                            # the separate formal report computes that control.
                            shuffled=None if cfg.model.dimension==3 else 0.
                        else:
                            info=float(histogram_mutual_information(hist,table).mean())
                            independent=hist.sum(2,keepdim=True)*hist.sum(1,keepdim=True)
                            shuffled=float(histogram_mutual_information(independent,table).mean())
                else:
                    info,shuffled=evaluate_information(m,critic,full,flip=cfg.data.spin_flip_augment,pairs=pairs)
                benchmark_memory_snapshot(args.benchmark_warmup,device,f'afterMIlog_{split}',step=step)
                metrics.update(qcr_total=metrics['total'],rsmi=None if info is None else -info,
                    rsmi_bits=None if info is None else info/math.log(2),
                    shuffled_rsmi_bits=None if shuffled is None else shuffled/math.log(2))
                if rsmi_enabled:metrics['total']-=args.rsmi_weight*info
                row.update({f'{split}_{key}':value for key,value in metrics.items()})
                if low_ram:del x
            history.append(row)
            (args.output/'history_progress.json').write_text(json.dumps(history,indent=2)+'\n')
            information_log=(f'I_{args.estimator}={row["test_rsmi_bits"]:.4f} bits' if rsmi_enabled else 'RSMI disabled; MI not evaluated')
            print(f'epoch {step}/{t.epochs}: test total={row["test_total"]:.5f}, '+information_log+
                  (f', balanced={row["test_balanced_reconstruction"]:.5f}' if cfg.loss.balanced_reconstruction_weight else '')+
                  (f', future NRMSE={row["test_future_physics_nrmse"]:.5f}' if future_predictor is not None else ''),flush=True)
            if args.joint_train_plateau:
                from recurrent_rg.joint_plateau import joint_plateau_reached
                component_row=train_component_losses(row,cfg,args)
                plateau_rows.append(component_row)
                stopped_by_train_plateau,plateau_diagnostics=joint_plateau_reached(plateau_rows,**plateau_settings)
                (args.output/'joint_plateau.json').write_text(json.dumps(dict(
                    history=plateau_rows,diagnostics=plateau_diagnostics),indent=2)+'\n')
                print('TRAIN components: '+str(component_row)+
                      '; plateau='+str(stopped_by_train_plateau),flush=True)
            if timing_enabled:synchronize()
            logging_seconds+=time.perf_counter()-logging_started
        if step%t.checkpoint_every==0:
            checkpoint_started=time.perf_counter();save(step,f'training_step_{step}.pt')
            checkpoint_seconds+=time.perf_counter()-checkpoint_started
        if stopped_by_train_plateau:
            print(f'Separate TRAIN Q/C/R objectives reached a local plateau at {step}; saving final endpoint.',flush=True)
            break
        if stopped_by_wall_limit:
            print(f'Wall limit reached after actual epoch {step}; saving completed training, not restarting.',flush=True)
            break
        if stopped_by_training_limit:
            print(f'Training-only limit reached after actual epoch {step}; saving completed training, not restarting.',flush=True)
            break
    if timing_enabled or args.max_training_seconds is not None:synchronize()
    training_wall_seconds=time.perf_counter()-training_started
    checkpoint_started=time.perf_counter()
    save(completed_epoch,'checkpoint_last.pt');torch.save(qcr_state_dict(m),args.output/'qcr_model.pt')
    checkpoint_seconds+=time.perf_counter()-checkpoint_started
    warmup=0 if args.benchmark_warmup is None else args.benchmark_warmup
    timing=dict(device=str(device),precision='torch.float64',setup_seconds=setup_seconds,
        training_wall_seconds=training_wall_seconds,updates=len(step_seconds),
        warmup_updates=min(warmup,len(step_seconds)),measured_updates=max(0,len(step_seconds)-warmup),
        step_seconds_all=step_seconds,steady_step_seconds=step_seconds[warmup:],
        logging_seconds=logging_seconds,checkpoint_seconds=checkpoint_seconds,
        includes_eval_in_step=False,enabled=timing_enabled)
    if device.type=='cuda':
        timing.update(cuda_max_memory_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      cuda_max_memory_reserved_bytes=torch.cuda.max_memory_reserved(device))
    (args.output/'summary.json').write_text(json.dumps(dict(config=cfg.to_dict(),contract=contract,history=history,
        elapsed_seconds=time.perf_counter()-started,setup_and_training_seconds=time.perf_counter()-wall_started,
        actual_epochs=completed_epoch,requested_epochs=t.epochs,stopped_by_wall_limit=stopped_by_wall_limit,
        optimizer_steps=component_optimizer_steps(m,opt),
        stopped_by_training_limit=stopped_by_training_limit,training_limit_seconds=args.max_training_seconds,
        timing=timing,lr_scheduler_state=lr_policy.checkpoint_state(completed_epoch)),indent=2)+'\n')
    if args.joint_train_plateau:
        summary=json.loads((args.output/'summary.json').read_text())
        summary.update(stopped_by_train_plateau=stopped_by_train_plateau,
            stop_reason='train_plateau' if stopped_by_train_plateau else 'wall_limit' if stopped_by_wall_limit else 'training_limit' if stopped_by_training_limit else 'update_limit',
            joint_plateau_diagnostics=plateau_diagnostics)
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if t.terminal_r_fit:
        from experiments.terminal_r_stage import run_terminal_stage
        def terminal_budget_check():
            if args.max_wall_seconds is not None and time.perf_counter()-wall_started>=args.max_wall_seconds:
                raise RuntimeError('Whole-run wall budget exhausted; terminal R stage incomplete, joint checkpoint retained')
            if args.max_training_seconds is not None and time.perf_counter()-training_started>=args.max_training_seconds:
                raise RuntimeError('Training budget exhausted; terminal R stage incomplete, joint checkpoint retained')
        terminal=run_terminal_stage(m,opt,tx.configurations,cfg,args.output,metadata,gen,completed_epoch,
            budget_check=terminal_budget_check)
        summary=json.loads((args.output/'summary.json').read_text())
        summary.update(terminal_R_fit=terminal,final_checkpoint='r_only/checkpoint_last.pt',
            joint_optimizer_steps=summary['optimizer_steps'],optimizer_steps=component_optimizer_steps(m,opt))
        summary['timing']['terminal_r_seconds']=terminal['total_stage_seconds']
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    final_report(args.output,skip=args.skip_final_report)
    if t.terminal_r_fit:
        from experiments.terminal_r_stage import write_json
        pipeline=json.loads((args.output/'pipeline.json').read_text())
        pipeline.update(status='complete',workflow_complete=True,current='complete',
                        final_report_skipped=args.skip_final_report)
        write_json(args.output/'pipeline.json',pipeline)


if __name__=='__main__':main()
