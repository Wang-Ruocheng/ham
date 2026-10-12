import math

import pytest

from recurrent_rg.joint_plateau import COMPONENTS, joint_plateau_reached


def constant_rows(last=4400, *, first=0, cadence=200):
    return [dict(epoch=epoch, component_Q=1., component_C=-2., component_R=.5)
            for epoch in range(first, last + 1, cadence)]


@pytest.mark.parametrize('component', COMPONENTS)
def test_any_continuing_component_improvement_prevents_stopping(component):
    rows = constant_rows()
    for row in rows:
        row[component] -= abs(rows[0][component]) * .01 * (row['epoch'] // 200)
    reached, diagnostic = joint_plateau_reached(rows)
    assert not reached
    assert not diagnostic['checks'][-1]['components'][component]['improved_little']


@pytest.mark.parametrize('component', COMPONENTS)
@pytest.mark.parametrize('confirmation_index', [-3, -2, -1])
def test_rebound_at_any_confirmation_or_component_prevents_stopping(component, confirmation_index):
    rows = constant_rows()
    row = rows[confirmation_index]
    row[component] += .051 * abs(row[component])
    reached, diagnostic = joint_plateau_reached(rows)
    assert not reached
    assert not diagnostic['checks'][confirmation_index]['components'][component]['near_best']


def test_negative_c_uses_signed_minima_and_absolute_relative_denominators():
    rows = constant_rows()
    for row in rows:
        row['component_C'] -= 1e-6 * row['epoch']
    reached, diagnostic = joint_plateau_reached(rows)
    assert reached
    c = diagnostic['checks'][-1]['components']['component_C']
    assert c['best'] < c['window_start_best'] < 0
    assert 0 < c['relative_best_improvement'] < .01
    assert c['rebound'] == 0


def test_all_three_components_need_three_complete_confirmations_after_minimum():
    rows = constant_rows()
    assert not joint_plateau_reached(rows[:-2])[0]
    assert not joint_plateau_reached(rows[:-1])[0]
    reached, diagnostic = joint_plateau_reached(rows)
    assert reached
    assert diagnostic['reason'] == 'plateau'
    assert [check['epoch'] for check in diagnostic['checks']] == [4000, 4200, 4400]
    assert all(item['passed'] for check in diagnostic['checks']
               for item in check['components'].values())


@pytest.mark.parametrize('rows', [[], constant_rows(3000), constant_rows(first=3000),
                                constant_rows()[-3:]])
def test_short_history_cannot_establish_plateau(rows):
    assert not joint_plateau_reached(rows)[0]


@pytest.mark.parametrize('component', COMPONENTS)
@pytest.mark.parametrize('value', [math.nan, math.inf, -math.inf])
def test_nonfinite_component_history_cannot_establish_plateau(component, value):
    rows = constant_rows()
    rows[3][component] = value
    reached, diagnostic = joint_plateau_reached(rows)
    assert not reached
    assert diagnostic['reason'] == 'invalid_history'
    assert diagnostic['invalid_row'] == 3


def test_near_zero_component_uses_loss_floor_for_rebound():
    rows = constant_rows()
    for row in rows:
        row['component_C'] = 0.
    rows[-1]['component_C'] = .049 * 1e-3
    assert joint_plateau_reached(rows)[0]
    rows[-1]['component_C'] = .051 * 1e-3
    assert not joint_plateau_reached(rows)[0]


@pytest.mark.parametrize('component', COMPONENTS)
def test_exact_five_percent_rebound_boundary_is_allowed(component):
    rows = constant_rows()
    rows[-1][component] += .05 * abs(rows[-1][component])
    assert joint_plateau_reached(rows)[0]


def test_near_zero_best_improvement_uses_floor_and_is_strictly_below_one_percent():
    rows = constant_rows()
    for row in rows:
        row['component_C'] = 0.
    rows[-1]['component_C'] = -9e-6
    assert joint_plateau_reached(rows)[0]
    rows[-1]['component_C'] = -1e-5
    assert not joint_plateau_reached(rows)[0]


def test_custom_settings_and_scheduled_check_cadence():
    rows = constant_rows(last=500, cadence=100)
    assert joint_plateau_reached(rows, minimum=400, window=200,
                                 confirmations=2, check_every=100)[0]
    rows.append(dict(rows[-1], epoch=501))
    reached, diagnostic = joint_plateau_reached(rows, minimum=400, window=200,
                                               confirmations=2, check_every=100)
    assert not reached and diagnostic['reason'] == 'not_check_epoch'


def test_missing_scheduled_confirmation_prevents_stopping():
    rows = [row for row in constant_rows() if row['epoch'] != 4200]
    reached, diagnostic = joint_plateau_reached(rows)
    assert not reached and diagnostic['reason'] == 'missing_confirmation'


def test_test_fields_do_not_affect_train_only_decision():
    rows = constant_rows()
    for row in rows:
        row['test'] = dict(component_Q=math.nan, component_C=-math.inf, component_R=math.inf)
    assert joint_plateau_reached(rows)[0]


@pytest.mark.parametrize('kwargs', [dict(window=0), dict(minimum=-1),
                                   dict(confirmations=0), dict(check_every=0),
                                   dict(rtol=math.nan), dict(rebound_rtol=-1.),
                                   dict(loss_floor=0.)])
def test_invalid_settings_fail_explicitly(kwargs):
    with pytest.raises(ValueError):
        joint_plateau_reached(constant_rows(), **kwargs)
