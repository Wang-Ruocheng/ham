"""A hardware-only handoff must not retry work or relax scientific checks."""
from copy import deepcopy
import json
import math
from types import SimpleNamespace

import pytest

from scripts import continue_ising_b0_gpu0 as handoff


GPU0_UUID = 'GPU-5bba87f1-7d32-4dde-fe28-caba4468ab42'


def waiting_job():
    return dict(status='waiting_resources', current='seed92_training',
        workflow_complete=False, error=None, requires_user_action=False,
        stages=[dict(name=name, status='complete', returncode=0)
                for name in ('training_profile', 'seed91_training', 'seed91_report')],
        training={'91': dict(actual_epochs=12000)},
        gpu_elapsed_seconds=4307.6754, cpu_seconds=4594.1452)


def frozen_plan():
    return dict(device='cuda:1',
        gpu_uuid='GPU-18fac97c-11ee-4624-d298-9074e8d7028c',
        resource_policy=dict(gpu_peak_bytes=12 * 1024**3,
            gpu_seconds=14 * 3600, cpu_seconds=30 * 3600,
            max_training_seconds_per_arm=3 * 3600,
            gpu_free_mib=12288, minimum_host_available_gib=36,
            two_gpu_checks_seconds_apart=10, directory_bytes=20 * 1024**3),
        configs={'92': dict(model=dict(theory_dim=5, block_factor=3),
            training=dict(seed=92, epochs=12000),
            loss=dict(closure_scale_weights=None))},
        protocol=dict(labels_used_in_training=False, training_updates=12000,
            seeds=[91, 92, 93, 94, 95], automatic_retry=False, resume=False))


def test_remaining_stages_never_repeat_profile_seed91_or_add_new_science():
    job = waiting_job()
    original = deepcopy(job)
    assert handoff.remaining_stages(job) == [
        'seed92_training', 'seed92_report',
        'seed93_training', 'seed93_report',
        'seed94_training', 'seed94_report',
        'seed95_training', 'seed95_report', 'comparison']
    assert job == original


@pytest.mark.parametrize('field,value', [
    ('status', 'running'),
    ('status', 'failed'),
    ('status', 'budget_hold'),
    ('current', 'seed91_training'),
    ('current', 'seed92_report'),
    ('workflow_complete', True),
    ('error', 'previous failed stage'),
    ('error', ''),
    ('error', False),
    ('requires_user_action', True),
])
def test_only_the_exact_healthy_waiting_boundary_is_eligible(field, value):
    job = waiting_job()
    job[field] = value
    with pytest.raises((ValueError, RuntimeError)):
        handoff.remaining_stages(job)


@pytest.mark.parametrize('index,field,value', [
    (0, 'status', 'running'),
    (1, 'status', 'failed'),
    (2, 'status', 'analysis_budget_hold'),
    (0, 'returncode', None),
    (1, 'returncode', 1),
    (2, 'returncode', 75),
    (1, 'name', 'seed92_training'),
])
def test_partial_or_failed_history_cannot_become_a_retry(index, field, value):
    job = waiting_job()
    job['stages'][index][field] = value
    with pytest.raises((ValueError, RuntimeError)):
        handoff.remaining_stages(job)


@pytest.mark.parametrize('mutation', ['missing_stage', 'extra_stage', 'reordered',
                                     'missing_training', 'already_started_seed92'])
def test_exact_completed_prefix_and_only_seed91_training_are_required(mutation):
    job = waiting_job()
    if mutation == 'missing_stage':
        job['stages'].pop()
    elif mutation == 'extra_stage':
        job['stages'].append(dict(name='seed92_training', status='complete', returncode=0))
    elif mutation == 'reordered':
        job['stages'][0], job['stages'][1] = job['stages'][1], job['stages'][0]
    elif mutation == 'missing_training':
        job['training'] = {}
    else:
        job['training']['92'] = dict(actual_epochs=1)
    with pytest.raises((ValueError, RuntimeError)):
        handoff.remaining_stages(job)


def test_runtime_plan_changes_only_gpu_and_own_peak_cap():
    plan = frozen_plan()
    original = deepcopy(plan)
    expected = deepcopy(plan)
    expected.update(device='cuda:0', gpu_uuid=GPU0_UUID)
    expected['resource_policy']['gpu_peak_bytes'] = None
    runtime = handoff.runtime_plan(plan)
    assert runtime == expected
    assert plan == original
    # Removing the own-allocation cap does not remove admission or time budgets.
    assert runtime['resource_policy']['gpu_free_mib'] == 12288
    assert runtime['resource_policy']['gpu_seconds'] == 14 * 3600
    assert runtime['resource_policy']['cpu_seconds'] == 30 * 3600
    assert runtime['resource_policy']['max_training_seconds_per_arm'] == 3 * 3600
    json.dumps(runtime, allow_nan=False)
    runtime['configs']['92']['training']['epochs'] = 1
    assert plan == original, 'runtime plan must be a deep copy'


@pytest.mark.parametrize('field,value', [
    ('device', 'cuda:2'),
    ('gpu_uuid', GPU0_UUID),
    ('gpu_uuid', 'unverified-device'),
])
def test_runtime_plan_rejects_any_other_hardware_handoff(field, value):
    plan = frozen_plan()
    plan[field] = value
    original = deepcopy(plan)
    with pytest.raises((ValueError, RuntimeError)):
        handoff.runtime_plan(plan)
    assert plan == original


def test_disabled_peak_cap_delegates_all_other_endpoint_and_provenance_checks(tmp_path):
    plan = handoff.runtime_plan(frozen_plan())
    original = deepcopy(plan)
    calls = []

    def validate(root, runtime, seed):
        assert root == tmp_path and seed == 92
        expected = deepcopy(plan)
        expected['resource_policy']['gpu_peak_bytes'] = math.inf
        assert runtime == expected
        calls.append((root, runtime, seed))
        return dict(actual_epochs=12000, manifest_sha256='unchanged',
                    cuda_max_memory_reserved_bytes=19 * 1024**3)

    result = handoff.validate_training(SimpleNamespace(validate_training=validate),
                                       tmp_path, plan, 92)
    assert len(calls) == 1
    assert result['actual_epochs'] == 12000
    assert result['manifest_sha256'] == 'unchanged'
    assert result['gpu_peak_limit_disabled'] is True
    assert plan == original and plan['resource_policy']['gpu_peak_bytes'] is None
    # Infinity is an in-memory compatibility detail, never a persisted plan value.
    json.dumps(plan, allow_nan=False)


def test_finite_peak_cap_is_not_replaced_by_infinity(tmp_path):
    plan = frozen_plan()
    original = deepcopy(plan)

    def validate(root, runtime, seed):
        assert root == tmp_path and seed == 93
        assert runtime == original
        return dict(actual_epochs=12000)

    result = handoff.validate_training(SimpleNamespace(validate_training=validate),
                                       tmp_path, plan, 93)
    assert result['actual_epochs'] == 12000
    assert plan == original


@pytest.mark.parametrize('reason', [
    'Exact fresh label-free endpoint or provenance mismatch',
    'MI cache-hit provenance mismatch',
])
def test_removing_peak_cap_never_swallows_scientific_validation_failure(tmp_path, reason):
    plan = handoff.runtime_plan(frozen_plan())

    def validate(root, runtime, seed):
        raise RuntimeError(reason)

    with pytest.raises(RuntimeError, match=reason):
        handoff.validate_training(SimpleNamespace(validate_training=validate),
                                  tmp_path, plan, 92)
