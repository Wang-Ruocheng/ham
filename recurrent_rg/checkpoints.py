"""One model builder and strict Q/C/R checkpoint loading for all consumers."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import torch
from .config import ModelConfig
from .model_nd import RecurrentRGNetwork1D, RecurrentRGNetwork2D, FixedDecimationCoarseGrainer


def attach_distribution_decoder(model, config):
    """Optional Q-only ablation; preserve Q/C/R initialization and sampling RNG."""
    if config.theory_head not in ('nonlinear', 'linear'):
        raise ValueError('Unknown Q theory head')
    if config.theory_head == 'linear':
        first, last = model.theory_encoder.head[0], model.theory_encoder.head[2]
        # Match the old head's value and Jacobian at pooled feature zero.
        # Construct after the original model to preserve all other weights/RNG.
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            head = torch.nn.Linear(first.in_features, last.out_features).double()
            activation = first.bias.tanh()
            head.weight.copy_((last.weight * (1-activation.square())[None]) @ first.weight)
            head.bias.copy_(last.weight @ activation + last.bias)
        model.theory_encoder.head = head
    if not isinstance(config.reconstruction_hidden, int) or config.reconstruction_hidden < 0:
        raise ValueError('reconstruction_hidden must be a nonnegative integer')
    if config.affine_reconstruction:
        if config.reconstruction_hidden:
            raise ValueError('Affine reconstruction cannot also use a nonlinear decoder')
        states = config.categorical_states or 2
        outputs = states ** (4 if config.dimension == 2 else 2)
        with torch.random.fork_rng(devices=[]):
            model.theory_encoder.distribution_decoder = torch.nn.Linear(config.theory_dim, outputs).double()
        model.theory_encoder.distribution_reconstruction_mode = 'affine_mse'
        from .representation import normalize_affine_decoder_
        normalize_affine_decoder_(model.theory_encoder)
    if config.reconstruction_hidden:
        states = config.categorical_states or 2
        outputs = states ** (4 if config.dimension == 2 else 2)
        with torch.random.fork_rng(devices=[]):
            model.theory_encoder.distribution_decoder = torch.nn.Sequential(
                torch.nn.Linear(config.theory_dim, config.reconstruction_hidden),
                torch.nn.Tanh(), torch.nn.Linear(config.reconstruction_hidden, outputs)).double()
    return model


def build_model(config: ModelConfig, max_rg_steps: int):
    if config.dimension not in (1, 2) or max_rg_steps < 1:
        raise ValueError('Unsupported dimension or depth')
    if config.kernel_sampling not in ('deterministic', 'bernoulli', 'categorical') or not 0 < config.kernel_temperature < float('inf'):
        raise ValueError('Invalid kernel sampling or relaxation temperature')
    if config.fixed_kernel and config.kernel_sampling != 'deterministic':
        raise ValueError('Fixed decimation cannot have stochastic sampling')
    if config.categorical_states:
        from .adapters import CategoricalFieldHead
        from .geometry import square_geometry
        from .model_nd import RecurrentRGNetwork
        expected_sampling = 'deterministic' if config.fixed_kernel else 'categorical'
        if config.dimension != 2 or config.kernel_sampling != expected_sampling or config.kernel_initialization != 'random':
            raise ValueError('Unsupported categorical model settings')
        head = CategoricalFieldHead(config.hidden, config.categorical_states, config.kernel_temperature)
        head.sampling_mode = 'categorical'
        model = RecurrentRGNetwork(square_geometry(config.width, config.block_factor), head,
            input_channels=config.categorical_states, levels=max_rg_steps+1,
            theory_dim=config.theory_dim, hidden=config.hidden, kernel_layout=config.kernel_layout).double()
        model.theory_encoder.categorical_states = config.categorical_states if config.binary_lookup else 0
        model.rg_kernel.categorical_states = config.categorical_states if config.binary_lookup else 0
        if config.fixed_kernel:
            # Construct first, then replace: paired Q/R initialization is preserved.
            # Downsample the entire one-hot vector, retaining both AT spins together.
            model.rg_kernel = FixedDecimationCoarseGrainer(model.geometry)
        return attach_distribution_decoder(model, config)
    cls = RecurrentRGNetwork1D if config.dimension == 1 else RecurrentRGNetwork2D
    # Preserve paired Q/R initialization: construct C, then replace if fixed.
    model = cls(levels=max_rg_steps+1, hidden=config.hidden, theory_dim=config.theory_dim,
                width=config.width, block_factor=config.block_factor,
                kernel_layout=config.kernel_layout).double()
    if config.fixed_kernel:
        model.rg_kernel = FixedDecimationCoarseGrainer(model.geometry)
    if config.kernel_initialization == 'positive_decimation':
        if config.fixed_kernel or config.kernel_layout != 'nonoverlapping':
            raise ValueError('Positive decimation initialization requires a learnable nonoverlapping kernel')
        model.rg_kernel.initialize_positive_decimation()
    elif config.kernel_initialization != 'random':
        raise ValueError('Unknown kernel initialization')
    model.theory_encoder.binary_lookup = config.binary_lookup
    model.rg_kernel.binary_lookup = config.binary_lookup
    if not config.fixed_kernel:
        model.rg_kernel.field_head.sampling_mode = config.kernel_sampling
        model.rg_kernel.field_head.relaxation_temperature = config.kernel_temperature
    return attach_distribution_decoder(model, config)


def qcr_state_dict(model):
    return {k: v for k, v in model.state_dict().items() if not k.startswith('action_network.')}


def load_model_state(model, payload):
    state = payload.get('model', payload)
    mismatch = model.load_state_dict(state, strict=False)
    if mismatch.unexpected_keys or any(not k.startswith('action_network.') for k in mismatch.missing_keys):
        raise ValueError(f'Incompatible Q/C/R checkpoint: {mismatch}')
    if 'balanced_reconstruction_metrics' in payload:
        from .balanced_reconstruction import BalancedMetric
        model.balanced_reconstruction_metrics = [BalancedMetric(**item)
            for item in payload['balanced_reconstruction_metrics']]
        model.balanced_metric_fitted_epoch = payload['balanced_metric_fitted_epoch']
    return model


def checkpoint_payload(model, optimizer, generator, epoch, *, metadata=None, history=None):
    payload = dict(epoch=epoch, model=qcr_state_dict(model), optimizer=optimizer.state_dict(),
                batch_generator_state=generator.get_state(), torch_rng_state=torch.get_rng_state(),
                metadata=metadata or {}, history=history or [])
    if hasattr(model, 'balanced_reconstruction_metrics'):
        payload['balanced_reconstruction_metrics'] = [m.state_dict()
            for m in model.balanced_reconstruction_metrics]
        payload['balanced_metric_fitted_epoch'] = model.balanced_metric_fitted_epoch
    return payload


@dataclass
class LoadedRun:
    model: torch.nn.Module
    config: dict
    checkpoint: Path
    payload: dict
    model_config: ModelConfig


def load_run(path, checkpoint=None, *, overrides=None):
    """Load new manifest runs or historical summary/run_config directories.

    Physical labels are never inferred from filenames. Historical missing
    architecture fields are recovered from weight shapes where unambiguous.
    Explicit overrides are useful for old weights without configuration files.
    """
    path = Path(path)
    directory = path if path.is_dir() else path.parent
    if path.is_file() and checkpoint is not None:
        raise ValueError('Specify a checkpoint file OR a run plus checkpoint selector')
    file = path if path.is_file() else directory / (
        f'training_step_{checkpoint}.pt' if isinstance(checkpoint, int)
        else str(checkpoint or 'qcr_model.pt'))
    payload = torch.load(file, map_location='cpu', weights_only=True)
    state = payload.get('model', payload)
    cfg = {}
    for name in ('manifest.json', 'summary.json', 'run_config.json'):
        if (directory/name).exists():
            raw = json.loads((directory/name).read_text())
            cfg = raw.get('config', raw)
            break
    if payload.get('metadata', {}).get('config'):
        cfg = payload['metadata']['config']
    cfg = dict(cfg)
    if 'model' in cfg and 'training' in cfg:
        model_values = dict(cfg['model'])
        if cfg.get('schema_version', 1) < 5:
            model_values.setdefault('block_factor', 2)
        steps = cfg['training']['max_rg_steps']
        if overrides:
            raise ValueError('Do not override versioned run architecture; use a new run')
    else:
        cfg.update(overrides or {})
        width = cfg.get('width', 3)
        patch = state['theory_encoder.local_mlp.0.weight'].shape[1]
        dimension = cfg.get('dimension')
        if dimension is None:
            matches = [d for d in (1,2) if width**d == patch]
            if len(matches) != 1:
                raise ValueError('Ambiguous legacy geometry; supply dimension/width overrides')
            dimension = matches[0]
        fixed = not any(k.startswith('rg_kernel.') for k in state)
        configured_fixed = cfg.get('fixed_kernel', not cfg.get('learnable_kernel', not fixed))
        if configured_fixed != fixed:
            raise ValueError('Kernel configuration conflicts with saved weights')
        layout = cfg.get('kernel_layout')
        factor = cfg.get('block_factor', 2)
        if layout is None:
            if fixed:
                layout = 'nonoverlapping'
            else:
                volume = state['rg_kernel.local_encoder.0.weight'].shape[1]
                candidates = [s for s,n in [('nonoverlapping',factor**dimension),('overlapping',width**dimension)] if volume==n]
                if len(candidates) != 1:
                    raise ValueError('Ambiguous legacy kernel layout; supply an override')
                layout = candidates[0]
        model_values = dict(dimension=dimension, width=width, block_factor=factor,
                            theory_dim=cfg.get('theory_dim',state['theory_rg.0.weight'].shape[1]),
                            hidden=cfg.get('hidden',state['theory_rg.0.weight'].shape[0]),
                            fixed_kernel=fixed, kernel_layout=layout,
                            binary_lookup=cfg.get('binary_lookup',False))
        if 'levels' not in cfg:
            raise ValueError('Legacy weights need an explicit levels override')
        steps = cfg['levels']-1
    # Historical saved runs without the sampling field used hard deterministic C.
    model_values.setdefault('kernel_sampling', 'deterministic')
    model_values.setdefault('affine_reconstruction',
                           'theory_encoder.distribution_decoder.weight' in state)
    if 'model' in cfg:
        cfg['model'] = dict(model_values)
    spec = ModelConfig(**model_values)
    # Loading a report must not consume the caller's training RNG stream.
    with torch.random.fork_rng(devices=[]):
        model = build_model(spec, steps)
    load_model_state(model, payload)
    return LoadedRun(model.eval(), cfg, file, payload, spec)
