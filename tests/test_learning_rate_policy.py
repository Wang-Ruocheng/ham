"""Learning rates may change, but Q/C/R physics, Adam order and RNG must not."""
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

import experiments.rsmi_separate_qcr as driver
from recurrent_rg.config import RunConfig, TrainingConfig
from recurrent_rg.cli import main as cli
from recurrent_rg.learning_rate import LearningRatePolicy
import recurrent_rg.runner as runner


DEVICES = ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='Requires CUDA'))]


def policy(**kwargs):
    return LearningRatePolicy(replace(TrainingConfig(), **kwargs))


def test_constant_matches_unscheduled_adam_bitwise_and_has_no_group_writes():
    class GuardedGroup(dict):
        def __setitem__(self, key, value):
            if key == 'lr':
                pytest.fail('Constant must not rewrite Adam learning rates')
            super().__setitem__(key, value)

    torch.manual_seed(517)
    a = torch.nn.Parameter(torch.randn(3, dtype=torch.float64))
    b = torch.nn.Parameter(a.detach().clone())
    historical = torch.optim.Adam([a], lr=.001)
    scheduled = torch.optim.Adam([b], lr=.001)
    scheduled.param_groups[0] = GuardedGroup(scheduled.param_groups[0])
    p = policy()
    rng = torch.get_rng_state().clone()
    for update in range(1, 5):
        historical.zero_grad(); scheduled.zero_grad()
        a.square().sum().backward(); b.square().sum().backward()
        p.apply_update(scheduled, update)
        historical.step(); scheduled.step()
    assert torch.equal(a, b) and torch.equal(rng, torch.get_rng_state())
    for key in historical.state[a]:
        assert torch.equal(historical.state[a][key], scheduled.state[b][key])


def test_multistep_clock_is_completed_updates_not_next_update():
    p = policy(lr_schedule='multistep', lr_milestones=(2, 4), lr_gamma=.5)
    assert [p.rate_for_update(u) for u in range(1, 7)] == [
        .001, .001, .0005, .0005, .00025, .00025]
    state = p.checkpoint_state(2)
    assert state['last_learning_rate'] == .001
    assert state['next_learning_rate'] == .0005


def test_schedule_only_rewrites_lr_preserving_adam_groups_order_and_moments():
    parameters = [torch.nn.Parameter(torch.tensor([float(k)], dtype=torch.float64))
                  for k in (1, 2)]
    optimizer = torch.optim.Adam(parameters, lr=.001)
    sum(p.square().sum() for p in parameters).backward(); optimizer.step()
    group = optimizer.param_groups[0]
    before = {p: {key: value.clone() for key, value in optimizer.state[p].items()}
              for p in parameters}
    rng = torch.get_rng_state().clone()
    p = policy(lr_schedule='multistep', lr_milestones=(1,), lr_gamma=.5)
    p.apply_update(optimizer, 2)
    assert optimizer.param_groups[0] is group and group['lr'] == .0005
    assert all(left is right for left, right in zip(group['params'], parameters))
    for parameter, state in before.items():
        for key, value in state.items():
            assert torch.equal(value, optimizer.state[parameter][key])
    assert torch.equal(rng, torch.get_rng_state())


def test_cosine_explicit_horizon_endpoints_and_rng():
    p = policy(lr_schedule='cosine', lr_cosine_updates=5, lr_min_factor=.1)
    rng = torch.get_rng_state().clone()
    assert p.rate_for_update(1) == .001
    assert p.rate_for_update(3) == pytest.approx(.00055, abs=1e-18)
    assert p.rate_for_update(5) == .0001
    assert p.rate_for_update(500) == .0001
    assert torch.equal(rng, torch.get_rng_state())
    # Increasing the requested total epochs cannot stretch the LR horizon.
    a = replace(TrainingConfig(), lr_schedule='cosine', lr_cosine_updates=5)
    assert LearningRatePolicy(a).description() == LearningRatePolicy(replace(a, epochs=9000)).description()


@pytest.mark.parametrize('settings', [
    {'lr_schedule': 'plateau'}, {'lr_schedule': None},
    {'learning_rate': 0}, {'learning_rate': float('nan')}, {'learning_rate': float('inf')},
    {'learning_rate': True}, {'learning_rate': '0.001'},
    {'lr_gamma': 0}, {'lr_gamma': 1.1}, {'lr_gamma': True},
    {'lr_min_factor': -.1}, {'lr_min_factor': 1.1}, {'lr_min_factor': float('nan')},
    {'lr_schedule': 'multistep'},
    {'lr_schedule': 'multistep', 'lr_milestones': (2, 1)},
    {'lr_schedule': 'multistep', 'lr_milestones': (1, 1)},
    {'lr_schedule': 'multistep', 'lr_milestones': (0, 2)},
    {'lr_schedule': 'multistep', 'lr_milestones': (True, 2)},
    {'lr_schedule': 'multistep', 'lr_milestones': (1., 2)},
    {'lr_schedule': 'multistep', 'lr_milestones': None},
    {'lr_milestones': (2,)}, {'lr_cosine_updates': 2},
    {'lr_schedule': 'cosine'}, {'lr_schedule': 'cosine', 'lr_cosine_updates': 1},
    {'lr_schedule': 'cosine', 'lr_cosine_updates': 2.},
    {'lr_schedule': 'cosine', 'lr_cosine_updates': True},
])
def test_invalid_schedule_fails_closed(settings):
    with pytest.raises(ValueError):
        policy(**settings)


def test_json_config_milestones_canonicalize_for_exact_resume():
    config = json.loads((Path(__file__).resolve().parents[1]/'configs/remote_smoke_225.json').read_text())
    config['training'].update(lr_schedule='multistep', lr_milestones=[2, 4], lr_gamma=.5)
    cfg = RunConfig.from_dict(config)
    cfg.validate()
    assert cfg.training.lr_milestones == (2, 4)
    assert RunConfig.from_dict(json.loads(json.dumps(cfg.to_dict()))).to_dict() == cfg.to_dict()
    with pytest.raises(ValueError, match='strictly increasing'):
        replace(cfg, training=replace(cfg.training, lr_milestones=(4, 2))).validate()


def test_resume_validates_saved_policy_progress_and_adam_learning_rate():
    parameter = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
    optimizer = torch.optim.Adam([parameter], lr=.001)
    p = policy(lr_schedule='multistep', lr_milestones=(2, 4), lr_gamma=.5)
    p.apply_update(optimizer, 3)
    payload = dict(epoch=3, lr_scheduler_state=p.checkpoint_state(3))
    p.validate_resume(payload, optimizer)
    for update in (0, -1, True, 1.):
        with pytest.raises(ValueError):
            p.rate_for_update(update)
    for update in (-1, True, 1., None):
        with pytest.raises(ValueError):
            p.checkpoint_state(update)
    with pytest.raises(ValueError, match='progress differs'):
        p.validate_resume(dict(payload, epoch=4), optimizer)
    with pytest.raises(ValueError, match='progress differs'):
        policy(lr_schedule='multistep', lr_milestones=(2, 5)).validate_resume(payload, optimizer)
    with pytest.raises(ValueError, match='requires saved'):
        p.validate_resume({'epoch': 3}, optimizer)
    optimizer.param_groups[0]['lr'] = .001
    with pytest.raises(ValueError, match='Restored Adam'):
        p.validate_resume(payload, optimizer)
    # Historical checkpoints without policy state are safe for constant only.
    policy().validate_resume({'epoch': 3}, optimizer)


@pytest.mark.parametrize('settings', [
    {}, {'lr_schedule': 'multistep', 'lr_milestones': (2,)},
    {'lr_schedule': 'cosine', 'lr_cosine_updates': 4},
])
def test_schedule_does_not_unlock_recipe_or_cli_legacy_path(monkeypatch, settings):
    cfg = RunConfig.from_dict(json.loads((Path(__file__).resolve().parents[1]/
        'configs/remote_smoke_225.json').read_text()))
    cfg = replace(cfg, training=replace(cfg.training, **settings))
    cfg.validate()
    for flags in (['--color-weight', '0'], ['--no-independent-c'], ['--estimator', 'infonce']):
        args = driver.build_parser().parse_args(['--config', 'unused', '--output', 'unused', *flags])
        with pytest.raises(ValueError, match='Locked default recipe'):
            driver.validate_recipe(args, cfg)
    calls = []
    monkeypatch.setattr(driver, 'main', lambda argv: calls.append(argv))
    monkeypatch.setattr('recurrent_rg.cli.run_training',
        lambda *args, **kwargs: pytest.fail('Scheduler must not dispatch to legacy training'))
    cli(['train', '--config', 'scheduled.json', '--output', 'out', '--resume', 'old.pt'])
    assert calls == [['--config', 'scheduled.json', '--output', 'out',
                      '--resume-checkpoint', 'old.pt']]


@pytest.mark.parametrize('settings', [
    {'lr_schedule': 'multistep', 'lr_milestones': (2,)},
    {'lr_schedule': 'cosine', 'lr_cosine_updates': 4},
])
def test_explicit_legacy_runner_rejects_schedules_before_data_or_training(
        tmp_path, monkeypatch, settings):
    cfg = RunConfig.from_dict(json.loads((Path(__file__).resolve().parents[1]/
        'configs/remote_smoke_225.json').read_text()))
    cfg = replace(cfg, training=replace(cfg.training, **settings))
    monkeypatch.setattr(runner, 'prepared_data',
        lambda *args: pytest.fail('Unsupported legacy schedules must fail before cache reads'))
    monkeypatch.setattr(runner, 'train',
        lambda *args, **kwargs: pytest.fail('No legacy optimization should run'))
    with pytest.raises(ValueError, match='supports only constant'):
        runner.run_training(cfg, tmp_path/'unsupported', allow_legacy_training=True)
    # The added schedule guard is not authorization to enter the legacy path.
    with pytest.raises(ValueError, match='This legacy runner omits'):
        runner.run_training(cfg, tmp_path/'not_authorized')
    assert not (tmp_path/'unsupported').exists()


def test_legacy_raw_config_comparison_canonicalizes_only_missing_schedule_defaults(
        tmp_path, monkeypatch):
    """Exercise historical raw-dict comparison without running legacy training."""
    from recurrent_rg.checkpoints import build_model, checkpoint_payload
    cfg_path, _ = _small_config(tmp_path, {'lr_schedule': 'constant'})
    cfg = RunConfig.load(cfg_path)
    torch.manual_seed(cfg.training.seed)
    model = build_model(cfg.model, cfg.training.max_rg_steps)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    metadata = dict(config=cfg.to_dict(), data_sha256=runner.file_fingerprint(cfg.data.cache))
    metadata['config']['training']['epochs'] = 2
    for field in ('lr_schedule', 'lr_milestones', 'lr_gamma', 'lr_min_factor', 'lr_cosine_updates'):
        metadata['config']['training'].pop(field)
    payload = checkpoint_payload(model, optimizer, torch.Generator(), 2,
                                 metadata=metadata, history=[])
    old = tmp_path/'historical_generic.pt'; torch.save(payload, old)
    calls = []

    def no_training(*args, **kwargs):
        calls.append(kwargs)
        return []

    monkeypatch.setattr(runner, 'train', no_training)
    out = runner.run_training(cfg, tmp_path/'compatible', resume=old, allow_legacy_training=True)
    assert out == tmp_path/'compatible' and len(calls) == 1
    assert calls[0]['resume_state']['epoch'] == 2
    assert calls[0]['learning_rate'] == .001
    # Missing defaults are not filled using requested arbitrary values.
    changed = replace(cfg, training=replace(cfg.training, lr_gamma=.3))
    with pytest.raises(ValueError, match='configuration/data changed'):
        runner.run_training(changed, tmp_path/'different', resume=old, allow_legacy_training=True)
    assert len(calls) == 1


def _small_config(tmp_path, schedule):
    config = json.loads((Path(__file__).resolve().parents[1]/'configs/remote_smoke_225.json').read_text())
    gen = torch.Generator().manual_seed(317)
    train = (2*torch.randint(2, (3, 4, 12, 12, 1), generator=gen)-1).double()
    test = (2*torch.randint(2, (2, 4, 12, 12, 1), generator=gen)-1).double()
    cache = tmp_path/'cache.pt'
    torch.save((train, torch.tensor([.1, .3, .5], dtype=torch.float64),
                test, torch.tensor([.2, .4], dtype=torch.float64)), cache)
    config['data']['cache'] = str(cache)
    config['model']['hidden'] = 4
    config['training'].update(epochs=4, max_rg_steps=1, threads=1, system_batch=2,
        sample_batch=4, checkpoint_every=2, log_every=1, **schedule)
    config['evaluation'].update(max_rg_steps=1, samples=4, systems=2)
    cfg = tmp_path/'config.json'; cfg.write_text(json.dumps(config))
    return cfg, LearningRatePolicy(RunConfig.from_dict(config).training)


@pytest.mark.parametrize('device', DEVICES)
def test_actual_constant_driver_resumes_pre_scheduler_checkpoint(tmp_path, monkeypatch, device):
    """Absence of new JSON fields/state must not reject historical default runs."""
    cfg, p = _small_config(tmp_path, {'lr_schedule': 'constant'})
    monkeypatch.setattr(driver, 'evaluate_run', lambda *args, **kwargs: None)
    common = ['--config', str(cfg), '--device', device]
    if device != 'cpu':
        common += ['--mi-cache-dir', str(tmp_path/'statistics')]
    whole, first, resumed = [tmp_path/name for name in ('whole', 'first', 'resumed')]
    driver.main(common+['--output', str(whole)])
    driver.main(common+['--output', str(first), '--epochs', '2'])
    historical = torch.load(first/'checkpoint_last.pt', weights_only=True, map_location='cpu')
    historical.pop('lr_scheduler_state')
    historical['metadata'].pop('learning_rate_policy')
    for field in ('lr_schedule', 'lr_milestones', 'lr_gamma', 'lr_min_factor', 'lr_cosine_updates'):
        historical['metadata']['config']['training'].pop(field)
    for row in historical['history']:
        row.pop('learning_rate')
    old = tmp_path/'pre_scheduler.pt'; torch.save(historical, old)
    driver.main(common+['--output', str(resumed), '--resume-checkpoint', str(old)])
    a = torch.load(whole/'checkpoint_last.pt', weights_only=True, map_location='cpu')
    b = torch.load(resumed/'checkpoint_last.pt', weights_only=True, map_location='cpu')
    assert b['lr_scheduler_state'] == p.checkpoint_state(4)
    assert [row['epoch'] for row in a['history']] == [row['epoch'] for row in b['history']]
    # Preserve old history verbatim; only new rows acquire the LR audit field.
    assert 'learning_rate' not in b['history'][0] and 'learning_rate' not in b['history'][1]
    assert all(row['learning_rate'] == .001 for row in b['history'][2:])
    atol, rtol = (1e-10, 1e-9) if device != 'cpu' else (0, 0)
    for name, value in a['model'].items():
        torch.testing.assert_close(value, b['model'][name], atol=atol, rtol=rtol)
    for left, right in zip(a['history'], b['history']):
        assert set(left)-{'learning_rate'} == set(right)-{'learning_rate'}
        for key in set(left)-{'learning_rate'}:
            torch.testing.assert_close(torch.tensor(left[key], dtype=torch.float64),
                torch.tensor(right[key], dtype=torch.float64), atol=atol, rtol=rtol)
    for key in ('torch_rng_state', 'batch_generator_state', 'rsmi_generator_state'):
        assert torch.equal(a[key], b[key])
    if device != 'cpu':
        assert torch.equal(a['cuda_rng_state'], b['cuda_rng_state'])


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('schedule', [
    {'lr_schedule': 'constant'},
    {'lr_schedule': 'multistep', 'lr_milestones': [2, 3], 'lr_gamma': .5},
    {'lr_schedule': 'cosine', 'lr_cosine_updates': 4, 'lr_min_factor': .1},
])
def test_real_default_qcr_driver_fresh_resume_preserves_independent_gradients(
        tmp_path, monkeypatch, device, schedule):
    """Four tiny actual updates cover metadata, checkpoint clock and CUDA routing.

    Independent Q/C/R VJPs are checked against their exact original targets
    before every Adam update; no new loss, dtype or sampling convention is used.
    The final phase plotting is omitted, not any training/evaluation objective.
    """
    cfg, p = _small_config(tmp_path, schedule)
    monkeypatch.setattr(driver, 'evaluate_run', lambda *args, **kwargs: None)
    calls = []
    original = driver.independent_component_gradients

    def audited(total, closure, c_objective, q, c, r, critic=()):
        q, c, r, critic = list(q), list(c), list(r), list(critic)
        expected = [torch.autograd.grad(objective, parameters, retain_graph=True, allow_unused=True)
                    for objective, parameters in ((total-closure, q+critic),
                                                   (c_objective, c), (closure, r))]
        original(total, closure, c_objective, q, c, r, critic)
        for parameters, gradients in zip((q+critic, c, r), expected):
            for parameter, gradient in zip(parameters, gradients):
                assert (parameter.grad is None) == (gradient is None)
                if gradient is not None:
                    torch.testing.assert_close(parameter.grad, gradient, atol=1e-11, rtol=1e-10)
        calls.append(1)

    monkeypatch.setattr(driver, 'independent_component_gradients', audited)
    if p.kind == 'constant':
        monkeypatch.setattr(LearningRatePolicy, 'apply_update',
            lambda *args: pytest.fail('Default constant driver must retain original Adam path'))
    common = ['--config', str(cfg), '--device', device]
    if device != 'cpu':
        common += ['--mi-cache-dir', str(tmp_path/'statistics')]
    whole, first, resumed = [tmp_path/name for name in ('whole', 'first', 'resumed')]
    driver.main(common+['--output', str(whole)])
    driver.main(common+['--output', str(first), '--epochs', '2'])
    driver.main(common+['--output', str(resumed), '--resume-checkpoint', str(first/'checkpoint_last.pt')])
    assert len(calls) == 8
    a = torch.load(whole/'checkpoint_last.pt', weights_only=True, map_location='cpu')
    b = torch.load(resumed/'checkpoint_last.pt', weights_only=True, map_location='cpu')
    assert a['epoch'] == b['epoch'] == 4
    assert a['lr_scheduler_state'] == b['lr_scheduler_state'] == p.checkpoint_state(4)
    assert a['metadata']['learning_rate_policy'] == p.description()
    assert a['metadata']['execution_device'] == device
    assert a['metadata']['independent_c'] and not a['metadata']['labels_used_in_training']
    assert list(a['optimizer']['param_groups'][0]['params']) == list(b['optimizer']['param_groups'][0]['params'])
    assert [row['learning_rate'] for row in a['history']] == [p.rate_for_update(u) for u in range(1, 5)]
    for name, value in a['model'].items():
        assert value.dtype == torch.float64
        torch.testing.assert_close(value, b['model'][name], atol=1e-10 if device != 'cpu' else 0,
                                  rtol=1e-9 if device != 'cpu' else 0)
    for left, right in zip(a['history'], b['history']):
        assert left.keys() == right.keys()
        for key in left:
            torch.testing.assert_close(torch.tensor(left[key], dtype=torch.float64),
                torch.tensor(right[key], dtype=torch.float64), atol=1e-10 if device != 'cpu' else 0,
                rtol=1e-9 if device != 'cpu' else 0)
    for key in ('torch_rng_state', 'batch_generator_state', 'rsmi_generator_state'):
        assert torch.equal(a[key], b[key])
    if device != 'cpu':
        assert torch.equal(a['cuda_rng_state'], b['cuda_rng_state'])
    manifest = json.loads((resumed/'manifest.json').read_text())
    summary = json.loads((resumed/'summary.json').read_text())
    assert manifest['learning_rate_policy'] == p.description()
    assert summary['lr_scheduler_state'] == p.checkpoint_state(4)
    # A changed strategy cannot be silently used by an exact continuation.
    changed = json.loads(cfg.read_text())
    changed['training'].update(lr_schedule='constant', lr_milestones=[], lr_cosine_updates=0)
    if p.kind == 'constant':
        changed['training'].update(lr_schedule='cosine', lr_cosine_updates=4)
    cfg.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='identical configuration'):
        driver.main(common+['--output', str(tmp_path/'bad_resume'),
            '--resume-checkpoint', str(first/'checkpoint_last.pt')])
