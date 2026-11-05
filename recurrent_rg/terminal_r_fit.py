"""Terminal fixed-Q/C, fixed-TRAIN-cache optimization of the existing R.

This is a second stage, distinct from per-minibatch R inner updates. Labels,
TEST, roots and physical reference values are never accepted by the fitter.
"""
import hashlib
import math
import time

import torch

from .checkpoints import isolated_rng
from .r_refinement import closure_on_targets


def fingerprint(module):
    h = hashlib.sha256()
    for name, value in module.state_dict().items():
        value = value.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode()); h.update(value.numpy().tobytes())
    return h.hexdigest()


def freeze_qc(model, r_module):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False); p.grad = None
    for p in r_module.parameters(): p.requires_grad_(True)


@torch.no_grad()
def collect_train_targets(model, configurations, encode, *, samples, systems,
                          repeats=8, seed=91915, budget_check=None):
    """encode(batch) returns the SAME two-half-mean convention as joint training.

    All input rows must already be TRAIN physical chains. Repeats are fixed
    configuration subsets/C realizations, not extra independent physical chains.
    Both batch and stochastic-C streams are private; no training RNG is consumed.
    """
    if (type(repeats) is not int or repeats < 1 or samples < 2 or samples % 2
            or samples > configurations.shape[1] or systems < 1 or not len(configurations)):
        raise ValueError('Invalid terminal TRAIN cache geometry')
    device = next(model.parameters()).device
    gen = torch.Generator().manual_seed(seed)
    targets, chains = [], []
    with isolated_rng(device, seed+1):
        for _ in range(repeats):
            for start in range(0, len(configurations), systems):
                if budget_check: budget_check()
                indices = torch.randperm(configurations.shape[1], generator=gen)[:samples]
                batch = configurations[start:start+systems][:, indices].to(device=device, dtype=torch.float64)
                direct = encode(batch).detach().cpu()
                if direct.ndim != 3 or not torch.isfinite(direct).all():
                    raise FloatingPointError('Invalid frozen TRAIN encodings')
                targets.append(direct); chains.extend(range(start, start+len(direct)))
    targets = torch.cat(targets)
    return targets, dict(physical_chains=len(configurations), cached_rows=len(targets),
        subset_and_C_repeats_per_chain=repeats, samples_per_subset=samples, seed=seed,
        chain_index=chains, target_shape=list(targets.shape), labels_used=False,
        independent_physical_chains=len(configurations), Q_C_fixed=True)


def plateau_reached(rows, *, minimum=3000, window=2000, confirmations=3):
    if len(rows) < confirmations: return False
    for end in rows[-confirmations:]:
        if end['extra_R_updates'] < minimum: return False
        old = [r for r in rows if r['extra_R_updates'] <= end['extra_R_updates']-window]
        if not old: return False
        previous = old[-1]['best_train_closure']; current = end['train_loss']
        if (not math.isfinite(current) or
                (previous-end['best_train_closure'])/max(abs(previous), 1e-12) >= .01 or
                current > end['best_train_closure']*1.05 + 1e-15):
            return False
    return True


def fit_r(model, r_module, frozen_modules, optimizer, train_targets, loss_config, *,
          max_steps=10000, check_every=500, minimum=3000, window=2000,
          chunk_size=512, callback=None, budget_check=None):
    """Deterministic full-cache Adam, chunked without changing the mean objective.

    Keep the supplied optimizer/moments/LR; clear ALL optimizer gradients to
    None so shared Adam does not advance Q/C. Checkpoints are final endpoints,
    never selected by TEST or roots. An update cap is not called a plateau.
    """
    for value in (max_steps, check_every, minimum, window, chunk_size):
        if type(value) is not int or value < 1: raise ValueError('Positive integer fit limits required')
    if train_targets.ndim != 3 or not len(train_targets) or not torch.isfinite(train_targets).all():
        raise ValueError('Finite nonempty TRAIN cache required')
    loss_config.objective_kwargs()
    if loss_config.closure_objective != 'all_start_anchor' or loss_config.closure_metric != 'euclidean':
        raise ValueError('Terminal fit currently supports the unchanged Euclidean all-start objective')
    parameters = list(r_module.parameters())
    owned = {id(p) for group in optimizer.param_groups for p in group['params']}
    if any(id(p) not in owned for p in parameters): raise ValueError('Optimizer must own every R parameter')
    before = {n:fingerprint(m) for n,m in frozen_modules.items()}
    frozen_clocks = {id(p):int(optimizer.state.get(p, {}).get('step', 0))
                     for m in frozen_modules.values() for p in m.parameters()}
    freeze_qc(model, r_module)
    device = next(r_module.parameters()).device
    targets = train_targets.detach().to(device)
    chunks = list(targets.split(chunk_size))
    rows = []; best = float('inf'); started = time.monotonic()
    initial_clocks = [int(optimizer.state.get(p, {}).get('step', 0)) for p in parameters]

    def measure(step):
        nonlocal best
        with torch.no_grad():
            value = sum(float(closure_on_targets(model, x, loss_config))*len(x)/len(targets) for x in chunks)
        if not math.isfinite(value): raise FloatingPointError('Nonfinite terminal TRAIN loss')
        best = min(best, value)
        row = dict(extra_R_updates=step, train_loss=value, best_train_closure=best,
                   elapsed_seconds=time.monotonic()-started)
        rows.append(row)
        if callback: callback(row, rows)

    measure(0); reason = 'update_cap'
    for step in range(1, max_steps+1):
        if budget_check: budget_check()
        optimizer.zero_grad(set_to_none=True)
        for x in chunks:
            if budget_check: budget_check()
            loss = closure_on_targets(model, x, loss_config)*len(x)/len(targets)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite terminal R loss')
            loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
            raise FloatingPointError('Invalid terminal R gradient')
        optimizer.step()
        if step % check_every == 0 or step == max_steps:
            measure(step)
            if plateau_reached(rows, minimum=minimum, window=window):
                reason = 'TRAIN_plateau'; break
    after = {n:fingerprint(m) for n,m in frozen_modules.items()}
    if before != after or any(p.grad is not None for m in frozen_modules.values() for p in m.parameters()):
        raise AssertionError('Frozen Q/C changed or received gradients')
    if any(int(optimizer.state.get(p, {}).get('step', 0)) != frozen_clocks[id(p)]
           for m in frozen_modules.values() for p in m.parameters()):
        raise AssertionError('Frozen Q/C Adam clocks advanced')
    clocks = [int(optimizer.state[p]['step']) for p in parameters]
    if clocks != [c+step for c in initial_clocks]: raise AssertionError('Wrong additional R Adam clocks')
    for p in model.parameters(): p.grad = None
    return dict(actual_extra_R_updates=step, stop_reason=reason, train_plateau_reached=reason=='TRAIN_plateau',
        initial_R_Adam_steps=initial_clocks, final_R_Adam_steps=clocks,
        frozen_before_sha256=before, frozen_after_sha256=after, Q_C_unchanged=True,
        initial_train_loss=rows[0]['train_loss'], final_train_loss=rows[-1]['train_loss'],
        elapsed_seconds=time.monotonic()-started, history=rows, labels_used=False,
        objective_unchanged=True, target_sampling='fixed finite TRAIN subsets/C draws; full-cache mean',
        limitation='Operational TRAIN plateau, not global optimality or a physical RG certification')
