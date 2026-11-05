"""Shared QCR optimization and deterministic resume loop."""
import json
from pathlib import Path
import torch
from contextlib import nullcontext
from .losses import qcr_objective
from .data.views import ensemble_views
from .checkpoints import checkpoint_payload, load_model_state, qcr_state_dict, isolated_rng, model_device, restore_checkpoint_rng
from .config import QCRWeights

DTYPE = torch.float64


def separate_qc_r_gradients(total, closure, qc_parameters, r_parameters):
    """Same forward samples; QC minimizes total-closure, R minimizes closure.

    Parameter-selective differentiation is equivalent to detaching both R
    inputs and target encodings for the R update, without detaching recurrent
    R calls from each other. No R-loss gradients are assigned to Q/C.
    """
    qc_grads = torch.autograd.grad(total-closure, qc_parameters, retain_graph=True, allow_unused=True)
    r_grads = torch.autograd.grad(closure, r_parameters, allow_unused=True)
    for parameter, gradient in zip(qc_parameters, qc_grads):
        parameter.grad = gradient
    for parameter, gradient in zip(r_parameters, r_grads):
        parameter.grad = gradient

def evaluate(model, configurations, pattern_fn, samples: int, *, spin_flip_augment=True,
             objective_kwargs=None) -> dict[str, float]:
    """Evaluate QCR objective on a subset."""
    model.eval()
    device = model_device(model)
    subset = ensemble_views(configurations, samples, spin_flip_augment=spin_flip_augment).to(device=device, dtype=DTYPE)
    stochastic = getattr(getattr(model.rg_kernel, 'field_head', None), 'sampling_mode', '') in ('bernoulli','categorical')
    # Common random numbers make logged evaluations comparable
    # without advancing the training sampling RNG.
    with (isolated_rng(device, 12345) if stochastic else nullcontext()), torch.no_grad():
        total, terms = qcr_objective(model, subset, pattern_fn=pattern_fn, **(objective_kwargs or {}))
    return {"total": float(total)} | {key: float(value) for key, value in terms.items()}


def train(model, train_x, test_x, *, pattern_fn, epochs: int, system_batch: int,
          sample_batch: int, eval_samples: int, eval_systems: int,
          learning_rate: float, seed: int, progress_dir=None,
          spin_flip_augment=True, objective_kwargs=None, resume_state=None,
          initial_history=None, train_modules=('Q', 'C', 'R'), checkpoint_every=500,
          checkpoint_metadata=None, record_every=None, gradient_mode='joint') -> list[dict[str, float]]:
    """Joint QCR training."""
    if model_device(model).type != 'cpu':
        raise ValueError('The historical generic trainer remains CPU-only; use the default QCR driver for CUDA training')
    if progress_dir is not None:
        progress_dir = Path(progress_dir)
    if gradient_mode not in ('joint', 'qc_r_separate'):
        raise ValueError('Unknown gradient mode')
    options = objective_kwargs or {}
    if gradient_mode == 'qc_r_separate' and (set(train_modules) not in ({'Q','C','R'},{'Q','R'}) or options.get('tangent_weight',0) or options.get('orientation_weight',0)):
        raise ValueError('Separate gradients require Q/R, optional C, and no tangent/orientation terms')
    if not train_modules or len(set(train_modules)) != len(train_modules) or set(train_modules)-{'Q','C','R'}:
        raise ValueError('train_modules must be a unique nonempty subset of Q,C,R')
    if checkpoint_every < 1 or epochs < 1 or min(system_batch,eval_systems) < 2:
        raise ValueError('Invalid training counts')
    for count, available in ((sample_batch,train_x.shape[1]),(eval_samples,train_x.shape[1]),(eval_samples,test_x.shape[1])):
        if count < 2 or count % 2 or count > available:
            raise ValueError('Sample batch must be even and available in every ensemble')
    if min(len(train_x),len(test_x)) < 2:
        raise ValueError('Geometry loss needs at least two ensembles in both splits')
    modules = {'Q':model.theory_encoder, 'C':model.rg_kernel, 'R':model.theory_rg}
    parameters = []
    for name,module in modules.items():
        for parameter in module.parameters():
            parameter.requires_grad_(name in train_modules)
            if name in train_modules:
                parameters.append(parameter)
    if not parameters:
        raise ValueError('Selected modules contain no trainable parameters')
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    history = list(initial_history or [])
    start_epoch = 0
    if resume_state is not None:
        load_model_state(model, resume_state)
        optimizer.load_state_dict(resume_state['optimizer'])
        generator.set_state(resume_state['batch_generator_state'])
        restore_checkpoint_rng(resume_state, device='cpu')
        start_epoch = int(resume_state['epoch'])
        if epochs <= start_epoch:
            raise ValueError('Total epochs must exceed the resumed checkpoint epoch')
    record_every = max(1, epochs // 50) if record_every is None else record_every
    if not isinstance(record_every, int) or record_every < 1:
        raise ValueError('record_every must be a positive integer')
    train_indices = torch.linspace(0, train_x.shape[0] - 1, min(eval_systems, train_x.shape[0])).long()
    test_indices = torch.linspace(0, test_x.shape[0] - 1, min(eval_systems, test_x.shape[0])).long()
    train_eval = train_x[train_indices]
    test_eval = test_x[test_indices]

    for epoch in range(start_epoch, epochs):
        model.train()
        systems = torch.randperm(train_x.shape[0], generator=generator)[:system_batch]
        half = sample_batch // 2
        samples = torch.randperm(train_x.shape[1], generator=generator)[:2 * half]
        selected = train_x[systems][:, samples].to(dtype=DTYPE)
        first_base, second_base = selected[:, :half], selected[:, half:]
        batch = (torch.cat((first_base, -first_base, second_base, -second_base), dim=1)
                 if spin_flip_augment else selected)

        optimizer.zero_grad(set_to_none=True)
        loss, terms = qcr_objective(model, batch, pattern_fn=pattern_fn, **options)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite QCR loss at update {epoch+1}')
        if gradient_mode == 'qc_r_separate':
            closure = terms.get('closure_total', options.get('weights', QCRWeights()).commutator*terms['commutator'] + options.get('rollout_weight',5.0)*terms['rollout'])
            separate_qc_r_gradients(loss, closure,
                [p for p in list(model.theory_encoder.parameters())+list(model.rg_kernel.parameters()) if p.requires_grad],
                list(model.theory_rg.parameters()))
        else:
            loss.backward()
        optimizer.step()
        from .representation import normalize_affine_decoder_
        normalize_affine_decoder_(model.theory_encoder)

        if epoch == 0 or (epoch + 1) % record_every == 0 or epoch + 1 == epochs:
            train_metrics = evaluate(model, train_eval, pattern_fn, eval_samples, spin_flip_augment=spin_flip_augment, objective_kwargs=objective_kwargs)
            test_metrics = evaluate(model, test_eval, pattern_fn, eval_samples, spin_flip_augment=spin_flip_augment, objective_kwargs=objective_kwargs)
            history.append(
                {"epoch": epoch + 1}
                | {f"train_{k}": v for k, v in train_metrics.items()}
                | {f"test_{k}": v for k, v in test_metrics.items()}
            )
            print(
                f"epoch {epoch + 1}/{epochs}: "
                f"train={train_metrics['total']:.6g} test={test_metrics['total']:.6g}",
                flush=True,
            )
            if progress_dir is not None:
                progress_dir.mkdir(parents=True, exist_ok=True)
                (progress_dir / "history_progress.json").write_text(json.dumps(history, indent=2)+"\n")
                torch.save({k: v for k, v in model.state_dict().items()
                            if not k.startswith("action_network.")}, progress_dir / "qcr_latest.pt")
        if progress_dir is not None and (epoch + 1) % checkpoint_every == 0:
            progress_dir.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint_payload(model, optimizer, generator, epoch+1,
                       metadata=checkpoint_metadata, history=history),
                       progress_dir / f'training_step_{epoch + 1}.pt')
    if progress_dir is not None:
        torch.save(checkpoint_payload(model, optimizer, generator, epochs,
                   metadata=checkpoint_metadata, history=history), progress_dir/'checkpoint_last.pt')
    return history
