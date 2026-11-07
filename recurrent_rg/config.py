"""Serializable configuration and backward-compatible objective weights."""
from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path


def validate_closure_scale_weights(weights, max_rg_steps=None):
    """Validate optional target-scale weights without changing their ratios.

    Entries correspond to target scales k=1..max_rg_steps for both the
    one-step and microscopic-start rollout closure objectives.
    """
    import math
    from numbers import Real
    if weights is None:
        return None
    if not isinstance(weights, (list, tuple)) or not weights:
        raise ValueError('closure_scale_weights must be a nonempty list or tuple')
    if any(isinstance(value, bool) or not isinstance(value, Real)
           or not math.isfinite(value) or value < 0 for value in weights):
        raise ValueError('closure_scale_weights must contain finite nonnegative numbers')
    values = tuple(float(value) for value in weights)
    if not any(value > 0 for value in values):
        raise ValueError('closure_scale_weights must have positive total weight')
    if max_rg_steps is not None and len(values) != max_rg_steps:
        raise ValueError('closure_scale_weights length must match max_rg_steps')
    return values


def validate_closure_objective(objective, weights, rollout_weight,
                               closure_metric='euclidean', closure_scale_weights=None):
    """Validate closure selection without relaxing historical objective rules."""
    import math
    from numbers import Real
    if objective not in ('one_step_and_rollout', 'all_start_anchor'):
        raise ValueError('Unknown closure_objective')
    if objective == 'one_step_and_rollout':
        return
    scale_weights = validate_closure_scale_weights(closure_scale_weights)
    if closure_metric != 'euclidean':
        raise ValueError('all_start_anchor requires the euclidean closure metric')
    if scale_weights is not None and len(set(scale_weights)) != 1:
        raise ValueError('all_start_anchor requires uniform closure_scale_weights')
    coefficients = (weights.commutator, rollout_weight)
    if (any(isinstance(value, bool) or not isinstance(value, Real) for value in coefficients)
            or not math.isfinite(weights.commutator) or not math.isfinite(rollout_weight)
            or weights.commutator <= 0 or weights.commutator != rollout_weight):
        raise ValueError('all_start_anchor requires equal positive commutator and rollout weights')


@dataclass(frozen=True)
class QCRWeights:
    """Loss weights for the QCR (Q-C-R) joint training objective.

    The ``anchor`` default differs between 1D (1.0) and 2D (2.0); scripts should
    pass the appropriate value explicitly.
    """
    anchor: float = 1.0
    coarse_consistency: float = 5.0
    kernel_noncollapse: float = 2.0
    separation: float = 2.0
    commutator: float = 5.0


@dataclass(frozen=True)
class ScaleConfig:
    spatial_shape: tuple[int, ...]
    max_rg_steps: int
    block_factor: int | None = None
    min_sites: int = 4

    def __post_init__(self):
        if self.block_factor is None:
            object.__setattr__(self, 'block_factor', 3 if len(self.spatial_shape) == 2 else 2)

    def shapes(self):
        shape = tuple(self.spatial_shape)
        if not shape or any(n < 1 for n in shape) or self.max_rg_steps < 1:
            raise ValueError('Need positive spatial sizes and at least one RG step')
        if self.block_factor < 2 or self.min_sites < 1:
            raise ValueError('Invalid block factor or minimum size')
        result = [shape]
        for _ in range(self.max_rg_steps):
            if any(n % self.block_factor for n in shape):
                raise ValueError('Spatial size is not divisible at every requested scale')
            shape = tuple(n // self.block_factor for n in shape)
            result.append(shape)
        if min(shape) < self.min_sites:
            raise ValueError(f'Final spatial shape {shape} is smaller than {self.min_sites}')
        return result


@dataclass(frozen=True)
class ModelConfig:
    categorical_states: int = 0
    dimension: int = 1
    theory_dim: int = 1
    hidden: int = 60
    width: int = 3
    block_factor: int | None = None
    kernel_layout: str = 'nonoverlapping'
    fixed_kernel: bool = False
    binary_lookup: bool = True
    kernel_initialization: str = 'random'
    kernel_sampling: str = 'bernoulli'
    kernel_temperature: float = 1.0
    reconstruction_hidden: int = 0
    reconstruction_hidden_layers: int = 1
    reconstruction_spectral_bound: float = 0.0
    reconstruction_probability_loss: str = 'js'
    affine_reconstruction: bool = True
    theory_head: str = 'nonlinear'
    # Pooled learned local Q features, independent of the output theory_dim.
    # Historical architectures used 12; omitted configuration fields retain it.
    local_features: int = 12
    q_encoder: str = 'local_motif'

    def __post_init__(self):
        import math
        if type(self.reconstruction_hidden_layers) is not int or self.reconstruction_hidden_layers < 1:
            raise ValueError('reconstruction_hidden_layers must be positive')
        if not math.isfinite(self.reconstruction_spectral_bound) or self.reconstruction_spectral_bound < 0:
            raise ValueError('Invalid reconstruction spectral bound')
        if self.reconstruction_probability_loss not in ('js', 'squared'):
            raise ValueError('Unknown probability reconstruction loss')
        if (self.reconstruction_spectral_bound or self.reconstruction_probability_loss != 'js'
                or self.reconstruction_hidden_layers != 1) and (not self.reconstruction_hidden or self.affine_reconstruction):
            raise ValueError('Nonlinear decoder options require a nonlinear probability decoder')
        if self.q_encoder not in ('local_motif', 'configuration_cnn', 'ensemble_vae'):
            raise ValueError('Unknown Q encoder architecture')
        if self.q_encoder == 'ensemble_vae' and (self.dimension != 2 or self.categorical_states not in (0, 4)
                or self.affine_reconstruction or self.reconstruction_hidden
                or self.theory_head != 'nonlinear'):
            raise ValueError('Ensemble VAE requires binary or four-state 2D fields and its own decoder')
        if self.q_encoder == 'configuration_cnn' and (self.dimension != 2 or self.categorical_states
                or type(self.theory_dim) is not int or self.theory_dim < 1
                or not self.affine_reconstruction or self.theory_head != 'nonlinear'):
            raise ValueError('Configuration CNN requires binary 2D positive-dimensional affine Q')
        if type(self.local_features) is not int or self.local_features < 1:
            raise ValueError('local_features must be a positive integer')
        if self.block_factor is None:
            object.__setattr__(self, 'block_factor', 3 if self.dimension == 2 else 2)


@dataclass(frozen=True)
class DataConfig:
    cache: str
    physical_model: str
    coupling_names: tuple[str, ...] = ()
    spin_flip_augment: bool = False
    spin_flip_symmetric: bool = False
    train_pool_replicas: int = 1
    test_pool_replicas: int = 1


@dataclass(frozen=True)
class LossConfig:
    balanced_reconstruction_weight: float = 0.0
    geometry_scale: float = 4.0
    geometry_mode: str = 'absolute'
    geometry_distance: str = 'euclidean'
    geometry_bounds: str = 'interval'
    geometry_relative_floor: float = .05
    geometry_max_inverse_weight: float = 100.
    geometry_neighbors: int = 4
    geometry_local_fraction: float = .5
    geometry_tail_mix: float = 0.0
    geometry_tail_add_weight: float = 0.0
    geometry_tail_fraction: float = .1
    geometry_tail_min_reliability: float = .5
    noise_multiplier: float = 2.0
    rollout_weight: float = 5.0
    closure_scale_weights: tuple[float, ...] | None = None
    # Fresh runs use uniform-target, all-start anchored consistency. Serialized
    # configurations with this field omitted retain the historical formula in
    # RunConfig.from_dict, independently of their schema version.
    closure_objective: str = 'all_start_anchor'
    isotropy_weight: float = 0.0
    tangent_weight: float = 0.0
    orientation_weight: float = 0.0
    orientation_radius: float = 0.02
    kernel_sign_weight: float = 0.0
    kernel_neutral_weight: float = 0.0
    kernel_neutral_max_patterns: int = 65536
    # Read-only compatibility with historical manifests/checkpoints. These are
    # never forwarded to an active objective; nonzero alpha cannot train.
    kernel_distinct_alpha: float = 0.0
    kernel_distinct_margin: float = 0.8
    closure_metric: str = 'euclidean'
    coordinate_reduction: str = 'sum'
    kernel_noncollapse_mode: str = 'balance'
    kernel_variance_ratio: float = .25
    weights: QCRWeights = field(default_factory=lambda: QCRWeights(anchor=2., separation=10.))

    def objective_kwargs(self, *, historical_metadata=False):
        """Active objective arguments, or validation-only historical metadata.

        Historical nonzero distinct loss remains readable by report loaders,
        but no trainer may silently continue with that retired term omitted.
        """
        import math
        validate_closure_scale_weights(self.closure_scale_weights)
        validate_closure_objective(self.closure_objective, self.weights,
            self.rollout_weight, self.closure_metric, self.closure_scale_weights)
        if not math.isfinite(self.balanced_reconstruction_weight) or self.balanced_reconstruction_weight < 0:
            raise ValueError('Invalid balanced reconstruction weight')
        if self.geometry_bounds not in ('interval', 'lower_only'):
            raise ValueError('Unknown geometry bounds')
        if self.geometry_distance not in ('euclidean', 'hellinger'):
            raise ValueError('Unknown geometry distance')
        if not 0 <= self.geometry_tail_mix <= 1 or not 0 < self.geometry_tail_fraction <= 1 or not 0 <= self.geometry_tail_min_reliability <= 1:
            raise ValueError('Invalid geometry tail settings')
        if not 0 <= self.geometry_tail_add_weight < float('inf') or (self.geometry_tail_mix and self.geometry_tail_add_weight):
            raise ValueError('Invalid or conflicting additive geometry tail weight')
        if (self.geometry_tail_mix or self.geometry_tail_add_weight) and self.geometry_mode != 'relative_local':
            raise ValueError('Geometry tail requires relative_local')
        if not math.isfinite(self.kernel_distinct_alpha) or self.kernel_distinct_alpha < 0 or not math.isfinite(self.kernel_distinct_margin) or not 0 < self.kernel_distinct_margin <= 1:
            raise ValueError('Invalid kernel distinct settings')
        if self.kernel_distinct_alpha and not historical_metadata:
            raise ValueError('kernel_distinct_loss has been retired; historical configurations '
                             'remain readable but cannot be used for active training. '
                             'Use a new explicit configuration with kernel_distinct_alpha=0.')
        if not math.isfinite(self.kernel_sign_weight) or self.kernel_sign_weight < 0:
            raise ValueError('Invalid kernel sign weight')
        if not math.isfinite(self.kernel_neutral_weight) or self.kernel_neutral_weight < 0:
            raise ValueError('Invalid kernel neutral weight')
        if not isinstance(self.kernel_neutral_max_patterns,int) or self.kernel_neutral_max_patterns < 1:
            raise ValueError('Invalid neutral reference enumeration limit')
        if not math.isfinite(self.orientation_weight) or self.orientation_weight < 0 or not math.isfinite(self.orientation_radius) or self.orientation_radius <= 0:
            raise ValueError('Invalid orientation settings')
        if self.tangent_weight < 0 or self.closure_metric not in ('euclidean', 'noise_aware'):
            raise ValueError('Invalid tangent weight or closure metric')
        values = asdict(self)
        values.pop('kernel_distinct_alpha')
        values.pop('kernel_distinct_margin')
        # Preserve legacy objective signatures and frozen execution manifests.
        # The new default is forwarded explicitly; omitted low-level arguments
        # continue to mean the historical objective for direct-call compatibility.
        if self.closure_objective == 'one_step_and_rollout':
            values.pop('closure_objective')
        values['weights'] = self.weights
        return values


@dataclass(frozen=True)
class TrainingConfig:
    max_rg_steps: int = 5
    epochs: int = 2000
    system_batch: int = 32
    sample_batch: int = 128
    learning_rate: float = .001
    seed: int = 91
    threads: int = 2
    train_modules: tuple[str, ...] = ('Q', 'C', 'R')
    checkpoint_every: int = 500
    log_every: int = 40
    gradient_mode: str = 'joint'
    lr_schedule: str = 'constant'
    lr_milestones: tuple[int, ...] = ()
    lr_gamma: float = .5
    lr_min_factor: float = .1
    lr_cosine_updates: int = 0
    # Total R Adam updates per online Q/C batch (including the joint update).
    # Missing in historical configurations means exactly the old 1:1 path.
    r_updates_per_batch: int = 1
    # Current fresh pipelines end with a fixed-Q/C fit; old serialized runs do not.
    terminal_r_fit: bool = True
    terminal_r_max_steps: int = 10000
    terminal_r_cache_repeats: int = 8

    def validate_learning_rate(self):
        from .learning_rate import validate_learning_rate_config
        validate_learning_rate_config(self)


@dataclass(frozen=True)
class EvaluationConfig:
    max_rg_steps: int = 5
    samples: int = 48
    systems: int = 24


@dataclass(frozen=True)
class RunConfig:
    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    schema_version: int = 5

    @classmethod
    def from_dict(cls, values):
        known = {'data', 'model', 'loss', 'training', 'evaluation', 'schema_version'}
        if set(values) - known:
            raise ValueError(f'Unknown configuration keys: {set(values)-known}')
        version = values.get('schema_version', 1)
        if version not in (1, 2, 3, 4, 5):
            raise ValueError('Unsupported configuration version')
        model = dict(values.get('model', {}))
        # Serialized configurations retain their historical representation.
        # Fresh default-driver runs upgrade omitted fields explicitly; loading
        # a checkpoint/config must never silently add a new decoder.
        model.setdefault('affine_reconstruction', False)
        # Old JSON files (including unversioned ones) used stride two.
        # Only new schema-5 configurations adopt the dimension-aware default.
        if version < 5:
            model.setdefault('block_factor', 2)
        if version == 1 or model.get('fixed_kernel', False):
            model.setdefault('kernel_sampling', 'deterministic')
        loss = dict(values.get('loss', {}))
        # Missing is historical, not an instruction to adopt today's default.
        # Schema 5 predates all-start closure too, and report/resume loaders must
        # never reinterpret its saved loss or gradient routing on reload.
        loss.setdefault('closure_objective', 'one_step_and_rollout')
        if isinstance(loss.get('closure_scale_weights'), (list, tuple)):
            loss['closure_scale_weights'] = tuple(loss['closure_scale_weights'])
        # Preserve historical objectives; the sign NLL only applies to
        # learnable probabilistic kernels in new configurations.
        if version < 3 or model.get('fixed_kernel', False) or model.get('kernel_sampling') == 'deterministic':
            loss.setdefault('kernel_sign_weight', 0.0)
        elif version == 3:
            loss.setdefault('kernel_sign_weight', 0.1)
        if 'weights' in loss:
            loss['weights'] = QCRWeights(**(asdict(LossConfig().weights) | loss['weights']))
        training = dict(values.get('training', {}))
        training.setdefault('terminal_r_fit', False)
        if isinstance(training.get('lr_milestones'), (list, tuple)):
            training['lr_milestones'] = tuple(training['lr_milestones'])
        return cls(data=DataConfig(**values['data']), model=ModelConfig(**model), schema_version=version,
                   loss=LossConfig(**loss),
                   training=TrainingConfig(**training),
                   evaluation=EvaluationConfig(**values.get('evaluation', {})))

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        config = cls.from_dict(json.loads(path.read_text()))
        cache = Path(config.data.cache)
        if not cache.is_absolute(): cache = path.parent/cache
        return replace(config, data=replace(config.data, cache=str(cache.resolve())))

    def to_dict(self):
        return asdict(self)

    def validate(self):
        m, t, e = self.model, self.training, self.evaluation
        t.validate_learning_rate()
        if m.q_encoder == 'ensemble_vae' and (self.loss.weights.anchor != 0
                or self.loss.weights.coarse_consistency != 0 or self.loss.weights.separation != 1
                or self.loss.balanced_reconstruction_weight or self.loss.isotropy_weight
                or self.loss.tangent_weight or self.loss.orientation_weight):
            raise ValueError('Ensemble VAE replaces the Q objective with unit-weight cross-subset CE + weak KL')
        if self.loss.balanced_reconstruction_weight and not m.affine_reconstruction:
            raise ValueError('Balanced reconstruction requires the shared affine decoder')
        if not isinstance(m.affine_reconstruction, bool) or (m.affine_reconstruction and m.reconstruction_hidden):
            raise ValueError('Affine and nonlinear distribution decoders are mutually exclusive')
        if m.dimension not in (1, 2, 3) or min(m.theory_dim, m.hidden, m.width) < 1:
            raise ValueError('Current runner supports binary 1D/2D/3D models with positive dimensions')
        if m.dimension == 3:
            if m.width != 2:
                raise ValueError('3D models require the explicit anchored width-2 cube window')
        elif m.width % 2 != 1:
            raise ValueError('Need odd centered patch width in 1D/2D')
        if m.block_factor < 2:
            raise ValueError('Need block_factor >= 2')
        if m.kernel_layout not in ('nonoverlapping', 'overlapping'):
            raise ValueError('Unknown kernel layout')
        if m.kernel_initialization not in ('random', 'positive_decimation'):
            raise ValueError('Unknown kernel initialization')
        if m.kernel_sampling not in ('deterministic', 'bernoulli', 'categorical') or not 0 < m.kernel_temperature < float('inf'):
            raise ValueError('Invalid kernel sampling or relaxation temperature')
        if m.categorical_states:
            expected_sampling = 'deterministic' if m.fixed_kernel else 'categorical'
            if m.categorical_states < 2 or m.dimension != 2 or m.kernel_sampling != expected_sampling or m.kernel_initialization != 'random':
                raise ValueError('Categorical runner requires 2D probabilistic C or fixed deterministic decimation')
            if self.data.spin_flip_augment or self.loss.kernel_noncollapse_mode not in ('variance_floor','variance_floor_ensemble'):
                raise ValueError('Categorical data require no binary flip augmentation and variance_floor')
        elif m.kernel_sampling == 'categorical':
            raise ValueError('Categorical sampling requires categorical_states')
        if m.fixed_kernel and m.kernel_sampling != 'deterministic':
            raise ValueError('Fixed decimation cannot have stochastic sampling')
        if m.kernel_initialization == 'positive_decimation' and (m.fixed_kernel or m.kernel_layout != 'nonoverlapping'):
            raise ValueError('Positive decimation initialization requires a learnable nonoverlapping kernel')
        if min(t.max_rg_steps, e.max_rg_steps, t.epochs, t.threads, t.checkpoint_every, t.log_every) < 1:
            raise ValueError('Steps, epochs, threads and checkpoint interval must be positive')
        if min(t.system_batch, e.systems) < 2 or t.learning_rate <= 0:
            raise ValueError('Need at least two ensembles and a positive learning rate')
        if any(n < 2 or n % 2 for n in (t.sample_batch, e.samples)):
            raise ValueError('Sample counts must be positive even numbers')
        if not t.train_modules or len(set(t.train_modules)) != len(t.train_modules) or set(t.train_modules)-{'Q','C','R'}:
            raise ValueError('train_modules must be a nonempty, unique subset of Q,C,R')
        if t.gradient_mode not in ('joint', 'qc_r_separate'):
            raise ValueError('Unknown gradient mode')
        if type(t.r_updates_per_batch) is not int or not 1 <= t.r_updates_per_batch <= 32:
            raise ValueError('r_updates_per_batch must be an integer in [1, 32]')
        if type(t.terminal_r_fit) is not bool:
            raise ValueError('terminal_r_fit must be boolean')
        if (type(t.terminal_r_max_steps) is not int or not 1 <= t.terminal_r_max_steps <= 100000
                or type(t.terminal_r_cache_repeats) is not int or not 1 <= t.terminal_r_cache_repeats <= 64):
            raise ValueError('Invalid terminal R update/cache limits')
        if t.r_updates_per_batch > 1 and (
                t.gradient_mode != 'qc_r_separate' or 'R' not in t.train_modules
                or self.loss.closure_objective != 'all_start_anchor'):
            raise ValueError('R inner updates require separate gradients and all_start_anchor closure')
        if t.gradient_mode == 'qc_r_separate' and (set(t.train_modules) not in ({'Q','C','R'},{'Q','R'},{'R'}) or self.loss.tangent_weight or self.loss.orientation_weight):
            raise ValueError('Separate gradients require Q/R with optional C, or explicit R-only continuation, and no tangent/orientation terms')
        if self.data.spin_flip_augment and not self.data.spin_flip_symmetric:
            raise ValueError('Spin-flip augmentation requires an explicitly symmetric data measure')
        if min(self.data.train_pool_replicas,self.data.test_pool_replicas) < 1:
            raise ValueError('Pooling factors must be positive')
        l = self.loss
        # Structural validation is also used by historical report readers.
        # Training obtains objective_kwargs() separately and rejects retired
        # nonzero objectives before executing any optimizer updates.
        l.objective_kwargs(historical_metadata=True)
        validate_closure_scale_weights(l.closure_scale_weights, t.max_rg_steps)
        if l.kernel_neutral_weight:
            if m.fixed_kernel or m.kernel_sampling == 'deterministic':
                raise ValueError('Neutral loss requires a learnable probabilistic kernel')
            alphabet = m.categorical_states or 2
            sites = (m.block_factor if m.kernel_layout == 'nonoverlapping' else m.width)**m.dimension
            if alphabet**sites > l.kernel_neutral_max_patterns:
                raise ValueError('Neutral reference exceeds kernel_neutral_max_patterns')
        if l.geometry_mode not in ('absolute','relative_local'):
            raise ValueError('Unknown geometry mode')
        if l.geometry_relative_floor <= 0 or l.geometry_max_inverse_weight <= 0 or not isinstance(l.geometry_neighbors,int) or l.geometry_neighbors < 1 or not 0 <= l.geometry_local_fraction <= 1:
            raise ValueError('Invalid relative geometry settings')
        if l.coordinate_reduction not in ('sum','mean') or l.kernel_noncollapse_mode not in ('balance','variance_floor','variance_floor_ensemble'):
            raise ValueError('Unknown loss convention')
        if l.geometry_scale <= 0 or l.noise_multiplier < 0 or l.rollout_weight < 0 or l.isotropy_weight < 0 or not 0 < l.kernel_variance_ratio <= 1:
            raise ValueError('Invalid loss parameters')
        if any(v < 0 for v in asdict(l.weights).values()):
            raise ValueError('Loss weights must be nonnegative')
