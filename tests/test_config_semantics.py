"""Configuration comparisons must ignore only JSON sequence spelling."""

import json
from decimal import Decimal

import pytest

from recurrent_rg.provenance import json_config_equal


def test_tuple_list_round_trip_is_equal_recursively():
    native = {
        "model": {"widths": (32, 16), "labels": ("Q", "C", "R")},
        "training": [{"scales": ((108, 36), (36, 12))}],
        "enabled": True,
        "learning_rate": 0.001,
        "unused": None,
    }
    serialized = json.loads(json.dumps(native))
    assert native != serialized
    assert json_config_equal(native, serialized)
    assert json_config_equal(serialized, native)


def test_object_key_order_is_ignored():
    assert json_config_equal({"a": 1, "b": [2]}, {"b": (2,), "a": 1})


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ({"seed": 91}, {"seed": 92}),
        ({"lr": 0.001}, {"lr": 0.0010000000000001}),
        ({"a": 1}, {"b": 1}),
        ({"a": None}, {}),
        ([1, 2], [2, 1]),
        ([1, 2], [[1, 2]]),
        ([1, 2], [1, 2, 3]),
        ({"a": [1]}, {"a": {"0": 1}}),
        (True, 1),
        (False, 0),
        (1, 1.0),
        (1, "1"),
        (None, "null"),
    ],
)
def test_actual_configuration_changes_are_not_equal(left, right):
    assert not json_config_equal(left, right)
    assert not json_config_equal(right, left)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_numbers_are_rejected_even_after_unmatched_keys(invalid):
    with pytest.raises(ValueError, match="Non-finite"):
        json_config_equal({"other": 0}, {"unmatched": [invalid]})
    with pytest.raises(ValueError, match="Non-finite"):
        json_config_equal([invalid], [invalid])


@pytest.mark.parametrize("invalid", [{1: "a"}, {True: "a"}, {None: "a"}])
def test_json_string_coercion_of_object_keys_is_not_allowed(invalid):
    with pytest.raises(TypeError, match="keys must be strings"):
        json_config_equal(invalid, invalid)


@pytest.mark.parametrize("invalid", [{1, 2}, b"text", Decimal("1"), object()])
def test_non_json_values_are_rejected(invalid):
    with pytest.raises(TypeError, match="Unsupported configuration type"):
        json_config_equal({"a": invalid}, {"a": invalid})


def test_circular_containers_are_rejected():
    sequence = []
    sequence.append(sequence)
    mapping = {}
    mapping["recursive"] = mapping
    for invalid in (sequence, mapping):
        with pytest.raises(ValueError, match="Circular"):
            json_config_equal(invalid, invalid)


def test_shared_acyclic_containers_are_allowed():
    shared = [1, 2]
    assert json_config_equal([shared, shared], ((1, 2), (1, 2)))


def test_real_run_config_closure_weights_match_their_json_manifest():
    from recurrent_rg.config import RunConfig
    from scripts.run_ising_closure_scale_ablation import make_config

    config = make_config('/not-opened/synthetic-cache.pt')
    config['loss']['closure_scale_weights'] = [1.8, .2]
    checkpoint_config = RunConfig.from_dict(config).to_dict()
    manifest_config = json.loads(json.dumps(checkpoint_config))
    assert isinstance(checkpoint_config['loss']['closure_scale_weights'], tuple)
    assert checkpoint_config != manifest_config  # The original collector falsely rejected this.
    assert json_config_equal(checkpoint_config, manifest_config)
    manifest_config['loss']['closure_scale_weights'] = [.2, 1.8]
    assert not json_config_equal(checkpoint_config, manifest_config)
