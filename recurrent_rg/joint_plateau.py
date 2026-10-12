"""TRAIN-only operational plateau checks for independent Q, C and R losses.

This module uses only the Python standard library. It neither evaluates a
model nor chooses a checkpoint: the caller decides when to stop its trainer.
"""
from bisect import bisect_right
import math
from numbers import Integral


COMPONENTS = ('component_Q', 'component_C', 'component_R')


def joint_plateau_reached(rows, *, minimum=4000, window=2000, confirmations=3,
                          check_every=200, rtol=.01, rebound_rtol=.05,
                          loss_floor=1e-3):
    """Return ``(reached, diagnostics)`` using only each row's TRAIN components.

    Every confirmation must be a consecutive scheduled check with its own
    full lookback window and minimum epoch. Each component's running minimum
    must improve strictly less than ``rtol`` over that window, and its current
    value must lie within ``rebound_rtol * max(abs(best), loss_floor)`` of its
    running minimum. Absolute magnitudes keep the rule valid for negative C.
    Missing, unordered or nonfinite history cannot establish a plateau.
    """
    integer_settings = dict(minimum=minimum, window=window,
                            confirmations=confirmations, check_every=check_every)
    for name, value in integer_settings.items():
        lower = 0 if name == 'minimum' else 1
        if isinstance(value, bool) or not isinstance(value, Integral) or value < lower:
            raise ValueError(f'{name} must be an integer >= {lower}')
    if (not math.isfinite(rtol) or not 0 < rtol < 1 or
            not math.isfinite(rebound_rtol) or rebound_rtol < 0 or
            not math.isfinite(loss_floor) or loss_floor <= 0):
        raise ValueError('Invalid plateau tolerances or loss floor')

    diagnostics = dict(
        reached=False, reason='empty_history', latest_epoch=None, checks=[],
        settings=dict(**integer_settings, rtol=rtol,
                      rebound_rtol=rebound_rtol, loss_floor=loss_floor),
    )
    values, epochs, bests = [], [], []
    running_best = {name: math.inf for name in COMPONENTS}
    for index, row in enumerate(rows):
        try:
            epoch = float(row['epoch'])
            components = {name: float(row[name]) for name in COMPONENTS}
        except (KeyError, TypeError, ValueError, OverflowError):
            diagnostics.update(reason='invalid_history', invalid_row=index)
            return False, diagnostics
        if (not math.isfinite(epoch) or epoch < 0 or not epoch.is_integer() or
                (epochs and epoch <= epochs[-1]) or
                any(not math.isfinite(value) for value in components.values())):
            diagnostics.update(reason='invalid_history', invalid_row=index)
            return False, diagnostics
        epochs.append(int(epoch))
        values.append(components)
        running_best = {name: min(running_best[name], components[name])
                        for name in COMPONENTS}
        bests.append(running_best)

    if not epochs:
        return False, diagnostics
    latest = epochs[-1]
    diagnostics['latest_epoch'] = latest
    if latest % check_every:
        diagnostics['reason'] = 'not_check_epoch'
        return False, diagnostics
    scheduled = [index for index, epoch in enumerate(epochs) if epoch % check_every == 0]
    if len(scheduled) < confirmations:
        diagnostics['reason'] = 'insufficient_confirmations'
        return False, diagnostics
    tail = scheduled[-confirmations:]
    expected = list(range(latest - (confirmations - 1) * check_every,
                          latest + 1, check_every))
    if [epochs[index] for index in tail] != expected:
        diagnostics['reason'] = 'missing_confirmation'
        return False, diagnostics

    all_passed = True
    for end in tail:
        epoch = epochs[end]
        check = dict(epoch=epoch, passed=False, components={})
        diagnostics['checks'].append(check)
        if epoch < minimum:
            check['reason'] = 'minimum_not_reached'
            all_passed = False
            continue
        start = bisect_right(epochs, epoch - window) - 1
        if start < 0:
            check['reason'] = 'insufficient_window'
            all_passed = False
            continue
        check['window_start_epoch'] = epochs[start]
        for name in COMPONENTS:
            before, best, current = bests[start][name], bests[end][name], values[end][name]
            improvement = (before - best) / max(abs(before), loss_floor)
            rebound = current - best
            rebound_limit = rebound_rtol * max(abs(best), loss_floor)
            improved_little = improvement < rtol
            near_best = current <= best + rebound_limit
            check['components'][name] = dict(
                window_start_best=before, best=best, current=current,
                relative_best_improvement=improvement,
                rebound=rebound, rebound_limit=rebound_limit,
                improved_little=improved_little, near_best=near_best,
                passed=improved_little and near_best,
            )
        check['passed'] = all(item['passed'] for item in check['components'].values())
        check['reason'] = 'plateau' if check['passed'] else 'improving_or_rebounded'
        all_passed = all_passed and check['passed']

    diagnostics['reached'] = all_passed
    diagnostics['reason'] = ('plateau' if all_passed else
                             next(check['reason'] for check in diagnostics['checks']
                                  if not check['passed']))
    return all_passed, diagnostics
