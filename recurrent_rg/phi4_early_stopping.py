"""Explicit, TRAIN-only early stopping for the short phi4 follow-up.

This module is inert unless an experiment constructs :class:`EarlyStopping`.
It does not change the standard QCR recipe, evaluate a model, inspect labels,
or select from TEST results. The caller supplies the post-update closure on
one fixed TRAIN probe, including fixed configurations and private C randomness.
The same probe must be used at every observation. Provenance and gradient/RNG
isolation of that evaluation remain the caller's responsibility.

The predeclared selector examines steps 100, 110, ..., 200, stopping after
20 updates without a strictly lower value, or at step 200. Thus it selects
the best *observed* checkpoint, not a retrospectively known minimum over all
100--200 steps. The stopping update and selected checkpoint are distinct.
"""
from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any


class EarlyStopping:
    """Fixed, opt-in selector; ties retain the earlier observed checkpoint.

    ``observe(step, value)`` returns whether training should stop. All allowed
    observations are required in sequence so a missing scheduled probe cannot
    silently change the selection. Invalid inputs and calls after stopping
    raise ``ValueError`` without changing state. Values must be finite,
    nonnegative real scalars; booleans are rejected for both arguments.

    ``receipt()`` returns a detached, JSON-serializable description. It contains
    no key named ``event``, ``step``, or ``train_closure``, leaving those names
    available to the trainer's outer log event.
    """

    min_step = 100
    max_step = 200
    every = 10
    patience = 20
    absolute_min_delta = 0.0
    relative_min_delta = 0.0

    def __init__(self) -> None:
        self.best_step: int | None = None
        self.best_value: float | None = None
        self.stop_reason: str | None = None
        self._observations: list[tuple[int, float]] = []

    @property
    def stopped(self) -> bool:
        return self.stop_reason is not None

    def observe(self, step: int, value: float) -> bool:
        """Accept exactly one next scheduled, post-update TRAIN observation."""
        if isinstance(step, bool) or not isinstance(step, Integral):
            raise ValueError("Early stopping step must be an integer, not a boolean")
        step = int(step)
        if not self.min_step <= step <= self.max_step:
            raise ValueError("Early stopping observations must be within steps 100--200")
        if (step - self.min_step) % self.every:
            raise ValueError("Early stopping observations must occur every 10 updates")
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("TRAIN closure must be a real scalar, not a boolean")
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError("TRAIN closure must be finite and nonnegative")
        if self.stopped:
            raise ValueError("Early stopping has already stopped; construct a new selector")
        expected = (self._observations[-1][0] + self.every
                    if self._observations else self.min_step)
        if step != expected:
            raise ValueError(f"Expected the next scheduled observation at step {expected}")

        # Validation above is deliberately complete before any state changes.
        self._observations.append((step, value))
        if self.best_value is None or value < self.best_value:
            self.best_step = step
            self.best_value = value
        if step >= self.max_step:
            self.stop_reason = "max_step"
        elif step - self.best_step >= self.patience:
            self.stop_reason = "patience"
        return self.stopped

    def receipt(self) -> dict[str, Any]:
        """Return an independent JSON-safe audit receipt without TEST metrics."""
        return {
            "policy": {
                "min_step": self.min_step,
                "max_step": self.max_step,
                "every": self.every,
                "patience_updates": self.patience,
                "absolute_min_delta": self.absolute_min_delta,
                "relative_min_delta": self.relative_min_delta,
                "comparison": "strict_lower",
                "tie_break": "earlier_step",
                "selector_split": "train",
                "metric": "post_update_fixed_probe_closure",
            },
            "best_step": self.best_step,
            "best_value": self.best_value,
            "stopped": self.stopped,
            "stop_reason": self.stop_reason,
            "observation_count": len(self._observations),
            "observations": [
                {"step": step, "value": value} for step, value in self._observations
            ],
        }
