"""One model builder and strict Q/C/R checkpoint loading for all consumers."""
from dataclasses import asdict, dataclass
from contextlib import contextmanager
import json
from pathlib import Path
import torch
from .config import ModelConfig
from .model_nd import RecurrentRGNetwork1D, RecurrentRGNetwork2D, RecurrentRGNetwork3D, FixedDecimationCoarseGrainer


def model_device(model):
    return next(model.parameters()).device


@contextmanager
def isolated_rng(device, seed):
    """Use reproducible diagnostic draws without changing training RNGs."""
    device = torch.device(device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('RNG isolation supports CPU or CUDA devices only')
    index = (torch.cuda.current_device() if device.index is None else device.index) if device.type == 'cuda' else None
    with torch.random.fork_rng(devices=[] if index is None else [index]):
        # torch.manual_seed also seeds CUDA; CPU diagnostics must not do that.
        torch.random.default_generator.manual_seed(seed)
        if index is not None:
            with torch.cuda.device(index):
                torch.cuda.manual_seed(seed)
        yield


def restore_checkpoint_rng(payload, *, device='cpu'):
    """Restore an exact same-device continuation; CPU reports never call this."""
    device = torch.device(device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Checkpoint RNG restoration supports CPU or CUDA only')
    if device.type == 'cuda':
        if 'cuda_rng_state' not in payload or 'cuda_device_index' not in payload:
            raise ValueError('Exact CUDA resume requires saved CUDA RNG state')
        index = torch.cuda.current_device() if device.index is None else device.index
        if payload['cuda_device_index'] != index:
            raise ValueError('Exact CUDA resume requires the same CUDA device')
    elif 'cuda_rng_state' in payload or 'cuda_device_index' in payload:
        raise ValueError('A CUDA checkpoint cannot resume exactly on CPU')
    torch.set_rng_state(payload['torch_rng_state'].cpu())
    if device.type == 'cuda':
        torch.cuda.set_rng_state(payload['cuda_rng_state'].cpu(), device=index)


def attach_distribution_decoder(model, config):
    """Optional Q-only ablation; preserve Q/C/R initialization and sampling RNG."""
    if config.q_encoder == 'ensemble_vae':
        from .ensemble_vae import EnsembleVAEQ
        with torch.random.fork_rng(devices=[]):
            model.theory_encoder = EnsembleVAEQ(config.theory_dim, config.categorical_states or 2)
        return model
    if config.q_encoder == 'configuration_cnn':
        from .ising_cnn import IsingConfigurationCNN
        with torch.random.fork_rng(devices=[]):
            model.theory_encoder = IsingConfigurationCNN(model.geometry, theory_dim=config.theory_dim)
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
        outputs = states ** {1: 2, 2: 4, 3: 8}[config.dimension]
        with torch.random.fork_rng(devices=[]):
            model.theory_encoder.distribution_decoder = torch.nn.Linear(config.theory_dim, outputs).double()
        model.theory_encoder.distribution_reconstruction_mode = 'affine_mse'
        from .representation import normalize_affine_decoder_
        normalize_affine_decoder_(model.theory_encoder)
    if config.reconstruction_hidden:
        states = config.categorical_states or 2
        outputs = states ** {1: 2, 2: 4, 3: 8}[config.dimension]
        with torch.random.fork_rng(devices=[]):
            layers = [torch.nn.Linear(config.theory_dim, config.reconstruction_hidden), torch.nn.Tanh()]
            for _ in range(config.reconstruction_hidden_layers-1):
                layers.extend([torch.nn.Linear(config.reconstruction_hidden, config.reconstruction_hidden), torch.nn.Tanh()])
            layers.append(torch.nn.Linear(config.reconstruction_hidden, outputs))
            model.theory_encoder.distribution_decoder = torch.nn.Sequential(*layers).double()
        if config.reconstruction_probability_loss == 'squared':
            model.theory_encoder.distribution_reconstruction_mode = 'probability_mse'
        model.theory_encoder.distribution_decoder_spectral_bound = config.reconstruction_spectral_bound
        from .representation import project_probability_decoder_
        project_probability_decoder_(model.theory_encoder)
    return model


def build_model(config: ModelConfig, max_rg_steps: int):
    if config.dimension not in (1, 2, 3) or max_rg_steps < 1:
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
            theory_dim=config.theory_dim, hidden=config.hidden, local_features=config.local_features,
            kernel_layout=config.kernel_layout).double()
        model.theory_encoder.categorical_states = config.categorical_states if config.binary_lookup else 0
        model.rg_kernel.categorical_states = config.categorical_states if config.binary_lookup else 0
        if config.fixed_kernel:
            # Construct first, then replace: paired Q/R initialization is preserved.
            # Downsample the entire one-hot vector, retaining both AT spins together.
            model.rg_kernel = FixedDecimationCoarseGrainer(model.geometry)
        return attach_distribution_decoder(model, config)
    cls = {1: RecurrentRGNetwork1D, 2: RecurrentRGNetwork2D, 3: RecurrentRGNetwork3D}[config.dimension]
    # Preserve paired Q/R initialization: construct C, then replace if fixed.
    model = cls(levels=max_rg_steps+1, hidden=config.hidden, theory_dim=config.theory_dim,
                width=config.width, block_factor=config.block_factor,
                kernel_layout=config.kernel_layout, local_features=config.local_features).double()
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
    device = model_device(model)
    if device.type == 'cuda':
        payload.update(cuda_rng_state=torch.cuda.get_rng_state(device),
                       cuda_device_index=device.index)
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
    if path.is_dir() and checkpoint is None and (path/'pipeline.json').exists():
        pipeline = json.loads((path/'pipeline.json').read_text())
        if pipeline.get('schema') == 'qcr_joint_then_terminal_r_v1':
            if not pipeline.get('model_complete'):
                raise ValueError('Terminal R pipeline is incomplete; request the preserved joint checkpoint explicitly')
            final = (path/pipeline['final_checkpoint']).resolve()
            if not final.is_relative_to(path.resolve()) or not final.is_file():
                raise ValueError('Invalid terminal pipeline checkpoint path')
            path = final
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
                            local_features=cfg.get('local_features', state['theory_encoder.local_mlp.2.weight'].shape[0]),
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
