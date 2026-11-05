import json
import math

import pytest

from recurrent_rg.phi4_early_stopping import EarlyStopping


def test_fixed_policy_and_initial_receipt():
    selector = EarlyStopping()
    receipt = selector.receipt()
    assert receipt["policy"] == {
        "min_step": 100, "max_step": 200, "every": 10,
        "patience_updates": 20, "absolute_min_delta": 0.,
        "relative_min_delta": 0., "comparison": "strict_lower",
        "tie_break": "earlier_step", "selector_split": "train",
        "metric": "post_update_fixed_probe_closure",
    }
    assert selector.best_step is None
    assert selector.best_value is None
    assert selector.stopped is False
    assert receipt["observations"] == []
    assert receipt["stop_reason"] is None
    assert not {"event", "step", "train_closure"}.intersection(receipt)
    json.dumps(receipt, allow_nan=False)


def test_patience_selects_observed_minimum_and_distinguishes_stopping_update():
    selector = EarlyStopping()
    assert selector.observe(100, .5) is False
    assert selector.observe(110, .2) is False
    assert selector.observe(120, .3) is False
    assert selector.observe(130, .4) is True
    assert selector.best_step == 110
    assert selector.best_value == .2
    assert selector.receipt()["observations"][-1]["step"] == 130
    assert selector.stop_reason == "patience"


def test_strict_improvement_resets_patience_and_tiny_improvement_counts():
    selector = EarlyStopping()
    assert not selector.observe(100, 1.)
    assert not selector.observe(110, 2.)
    assert not selector.observe(120, math.nextafter(1., 0.))
    assert selector.best_step == 120
    assert not selector.observe(130, 1.)
    assert selector.observe(140, 2.)
    assert selector.best_step == 120


def test_ties_keep_earliest_and_zero_is_valid():
    selector = EarlyStopping()
    assert not selector.observe(100, 0)
    assert not selector.observe(110, 0.)
    assert selector.observe(120, -0.)
    assert selector.best_step == 100
    assert selector.best_value == 0.
    assert selector.stop_reason == "patience"


def test_improvement_at_200_is_selected_but_maximum_still_stops():
    selector = EarlyStopping()
    for step in range(100, 200, 10):
        assert selector.observe(step, 201 - step) is False
    assert selector.observe(200, .1) is True
    assert selector.best_step == 200
    assert selector.best_value == .1
    assert selector.stop_reason == "max_step"
    assert selector.receipt()["observation_count"] == 11
    json.dumps(selector.receipt(), allow_nan=False)


def test_maximum_reason_takes_precedence_over_simultaneous_patience():
    selector = EarlyStopping()
    for step in range(100, 190, 10):
        assert not selector.observe(step, 200 - step)
    assert not selector.observe(190, 30)
    assert selector.observe(200, 30)
    assert selector.best_step == 180
    assert selector.stop_reason == "max_step"


@pytest.mark.parametrize("step", [True, False, 100., "100", None, 90, 99, 101, 105, 201, 210])
def test_invalid_first_step_does_not_mutate_state(step):
    selector = EarlyStopping()
    before = selector.receipt()
    with pytest.raises(ValueError):
        selector.observe(step, .2)
    assert selector.receipt() == before
    assert not selector.observe(100, .2)


@pytest.mark.parametrize("value", [True, False, "0.2", None, float("nan"),
                                    float("inf"), -float("inf"), -.1, [], {}])
def test_invalid_metric_does_not_mutate_state(value):
    selector = EarlyStopping()
    selector.observe(100, .5)
    before = selector.receipt()
    with pytest.raises(ValueError):
        selector.observe(110, value)
    assert selector.receipt() == before
    assert not selector.observe(110, .4)


@pytest.mark.parametrize("step", [100, 120, 130, 200])
def test_duplicate_or_skipped_observation_does_not_mutate_state(step):
    selector = EarlyStopping()
    selector.observe(100, .5)
    before = selector.receipt()
    with pytest.raises(ValueError, match="Expected"):
        selector.observe(step, .1)
    assert selector.receipt() == before
    assert not selector.observe(110, .4)


def test_cannot_start_at_a_later_step():
    selector = EarlyStopping()
    with pytest.raises(ValueError, match="step 100"):
        selector.observe(110, .2)
    assert selector.receipt()["observation_count"] == 0


def test_after_stop_rejects_observations_and_fresh_selector_is_independent():
    selector = EarlyStopping()
    selector.observe(100, .1)
    selector.observe(110, .2)
    assert selector.observe(120, .3)
    before = selector.receipt()
    with pytest.raises(ValueError, match="already stopped"):
        selector.observe(130, 0.)
    assert selector.receipt() == before
    fresh = EarlyStopping()
    assert fresh.best_step is None
    assert not fresh.observe(100, .5)
    assert selector.receipt() == before


def test_receipt_cannot_mutate_selector():
    selector = EarlyStopping()
    selector.observe(100, .5)
    receipt = selector.receipt()
    receipt["policy"]["selector_split"] = "test"
    receipt["observations"][0]["value"] = -100
    receipt["observations"].append({"step": 110, "value": 0.})
    assert selector.receipt()["policy"]["selector_split"] == "train"
    assert selector.receipt()["observations"] == [{"step": 100, "value": .5}]
    assert not selector.observe(110, .4)


def test_no_test_selector_or_configurable_policy_arguments():
    with pytest.raises(TypeError):
        EarlyStopping(selector_split="test")
    with pytest.raises(TypeError):
        EarlyStopping(min_step=0)
