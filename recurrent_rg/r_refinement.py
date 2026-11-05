"""Extra R-only inner updates inside each independent-gradient QCR batch.

This changes optimizer work, not the loss. Targets live for one batch only.
The first Q/C/R gradients are computed simultaneously by the caller; only
then do extra R updates reuse that batch's detached half-subset encodings.
"""
import torch

from .losses import all_start_anchor_closure_loss


def refinement_contract(training):
    return dict(
        r_updates_per_batch=training.r_updates_per_batch,
        extra_R_updates_per_batch=training.r_updates_per_batch - 1,
        target_policy='Same pre-update online half-subset-mean Q/C encodings, detached; discarded after each batch',
        sampling='One fresh physical minibatch and stochastic C trajectory per outer batch; no resampling inside R updates',
        gradient_policy='First Q/C/R gradients simultaneous and independent; extra gradients and Adam steps affect R only',
        learning_rate='Unchanged configured Adam; any scheduler uses outer-batch clock',
        loss_changed=False, labels_used=False, separate_R_only_stage=False,
        interpretation='More R optimization per moving target; not equivalent to post-training frozen-Q/C optimization or a convergence guarantee')


def closure_on_targets(model, direct, loss_config):
    if loss_config.closure_objective != 'all_start_anchor' or loss_config.closure_metric != 'euclidean':
        raise ValueError('R refinement supports the current euclidean all_start_anchor objective only')
    loss_config.objective_kwargs()  # Reject invalid/retired objective combinations.
    value = all_start_anchor_closure_loss(
        model, direct.detach(), loss_config.coordinate_reduction,
        loss_config.closure_scale_weights)[0]
    return (loss_config.weights.commutator + loss_config.rollout_weight) * value


def refine_r_on_targets(model, optimizer, direct, loss_config, extra_updates):
    """No Q/C forward or backward; supports R-only or shared Adam instances.

    Clearing ALL optimizer grads to None is essential: zero Q/C gradients
    would still advance their Adam moments/clocks and potentially weights.
    Recurrent R calls are NOT detached. The returned loss is a same-batch
    optimization diagnostic, not held-out error or evidence of convergence.
    """
    if type(extra_updates) is not int or not 0 <= extra_updates <= 31:
        raise ValueError('extra_updates must be an integer in [0, 31]')
    if not extra_updates:
        return None  # Historical path: no RNG, grad, loss, or optimizer changes.
    parameters = list(model.theory_rg.parameters())
    owned = {id(p) for group in optimizer.param_groups for p in group['params']}
    if any(id(p) not in owned or not p.requires_grad for p in parameters):
        raise ValueError('Optimizer must contain every trainable R parameter')
    direct = direct.detach()
    for _ in range(extra_updates):
        optimizer.zero_grad(set_to_none=True)
        loss = closure_on_targets(model, direct, loss_config)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite R inner-update loss')
        gradients = torch.autograd.grad(loss, parameters)
        for parameter, gradient in zip(parameters, gradients):
            if not torch.isfinite(gradient).all():
                raise FloatingPointError('Nonfinite R inner-update gradient')
            parameter.grad = gradient
        optimizer.step()
    with torch.no_grad():
        after = closure_on_targets(model, direct, loss_config)
    if not torch.isfinite(after):
        raise FloatingPointError('Nonfinite R post-update loss')
    return float(after)


def component_optimizer_steps(model, optimizer):
    """Actual parameterwise Adam clocks; absent/frozen states count as zero."""
    result = {}
    for name, module in [('Q', model.theory_encoder), ('C', model.rg_kernel), ('R', model.theory_rg)]:
        counts = [int(optimizer.state.get(p, {}).get('step', 0)) for p in module.parameters()]
        result[name] = dict(min=min(counts), max=max(counts), parameters=len(counts))
    return result
