"""Strict, dependency-free helpers for configuration provenance checks."""

from __future__ import annotations

import math
from typing import Any


def _json_config_value(value: Any, path: str, ancestors: set[int]) -> tuple:
    """Validate a JSON-shaped value and make only sequence spelling canonical."""
    value_type = type(value)
    if value is None:
        return ("null",)
    if value_type is bool:
        return ("bool", value)
    if value_type is int:
        return ("int", value)
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError(f"Non-finite configuration number at {path}")
        return ("float", value)
    if value_type is str:
        return ("str", value)
    if value_type not in (dict, list, tuple):
        raise TypeError(
            f"Unsupported configuration type {value_type.__name__} at {path}"
        )

    identity = id(value)
    if identity in ancestors:
        raise ValueError(f"Circular configuration container at {path}")
    ancestors.add(identity)
    try:
        if value_type is dict:
            if any(type(key) is not str for key in value):
                raise TypeError(f"Configuration object keys must be strings at {path}")
            return (
                "object",
                tuple(
                    (key, _json_config_value(value[key], f"{path}[{key!r}]", ancestors))
                    for key in sorted(value)
                ),
            )
        return (
            "array",
            tuple(
                _json_config_value(item, f"{path}[{index}]", ancestors)
                for index, item in enumerate(value)
            ),
        )
    finally:
        ancestors.remove(identity)


def json_config_equal(left: Any, right: Any) -> bool:
    """Compare configurations while accepting JSON's tuple-to-list conversion.

    Dictionary key order is irrelevant; array order, nesting, keys, scalar types,
    and scalar values are significant. In particular, ``True``, ``1``, and
    ``1.0`` are distinct. No string coercion, missing-key defaults, or float
    tolerance is applied.

    Both inputs must contain only ``None``, exact built-in JSON scalar types,
    string-keyed dictionaries, lists, and tuples. Unsupported types/keys raise
    ``TypeError``; non-finite floats and circular containers raise ``ValueError``.
    Validation also applies to unmatched branches, so malformed configurations
    cannot be silently accepted or hidden by an earlier mismatch.
    """
    left_value = _json_config_value(left, "left", set())
    right_value = _json_config_value(right, "right", set())
    return left_value == right_value
