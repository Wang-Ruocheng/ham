"""Deterministic, cumulative-update learning rates for the same QCR Adam.

No random draws, changes to parameter groups, or loss-dependent scheduling.
The explicit cosine horizon is independent of a run's wall/update cap, so
extending total epochs during an exact resume cannot stretch the schedule.
"""
import bisect
import math


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite real number')
    return float(value)


def validate_learning_rate_config(training):
    if training.lr_schedule not in ('constant', 'multistep', 'cosine'):
        raise ValueError('lr_schedule must be constant, multistep, or cosine')
    if _finite_number(training.learning_rate, 'learning_rate') <= 0:
        raise ValueError('learning_rate must be positive')
    gamma = _finite_number(training.lr_gamma, 'lr_gamma')
    floor = _finite_number(training.lr_min_factor, 'lr_min_factor')
    if not 0 < gamma <= 1:
        raise ValueError('lr_gamma must lie in (0, 1]')
    if not 0 <= floor <= 1:
        raise ValueError('lr_min_factor must lie in [0, 1]')
    milestones = training.lr_milestones
    if (not isinstance(milestones, (tuple, list)) or
            any(isinstance(step, bool) or not isinstance(step, int) or step < 1 for step in milestones)):
        raise ValueError('lr_milestones must contain positive integer updates')
    if tuple(milestones) != tuple(sorted(set(milestones))):
        raise ValueError('lr_milestones must be strictly increasing and unique')
    if training.lr_schedule == 'multistep' and not milestones:
        raise ValueError('multistep requires nonempty lr_milestones')
    if training.lr_schedule != 'multistep' and milestones:
        raise ValueError('lr_milestones applies only to multistep')
    horizon = training.lr_cosine_updates
    if isinstance(horizon, bool) or not isinstance(horizon, int):
        raise ValueError('lr_cosine_updates must be an integer')
    if training.lr_schedule == 'cosine':
        if horizon < 2:
            raise ValueError('cosine requires explicit lr_cosine_updates >= 2')
    elif horizon != 0:
        raise ValueError('lr_cosine_updates applies only to cosine')


class LearningRatePolicy:
    """The LR used by update u, without a mutable PyTorch scheduler clock."""
    def __init__(self, training):
        validate_learning_rate_config(training)
        self.kind = training.lr_schedule
        self.base_rate = float(training.learning_rate)
        self.milestones = tuple(training.lr_milestones)
        self.gamma = float(training.lr_gamma)
        self.minimum_factor = float(training.lr_min_factor)
        self.cosine_updates = training.lr_cosine_updates

    def description(self):
        return dict(version=1, schedule=self.kind, base_learning_rate=self.base_rate,
                    milestones=list(self.milestones), gamma=self.gamma,
                    minimum_factor=self.minimum_factor, cosine_updates=self.cosine_updates,
                    clock='cumulative completed optimizer updates; first update is 1',
                    milestone_convention='after completing m updates, update m+1 uses the decay',
                    cosine_convention='update 1 uses base; update N uses floor; later updates keep floor',
                    parameter_groups='unchanged; all Q/C/R Adam groups share this learning rate')

    def rate_for_update(self, update):
        if isinstance(update, bool) or not isinstance(update, int) or update < 1:
            raise ValueError('learning-rate update must be a positive integer')
        if self.kind == 'constant':
            return self.base_rate
        if self.kind == 'multistep':
            return self.base_rate * self.gamma**bisect.bisect_left(self.milestones, update)
        progress = min((update-1)/(self.cosine_updates-1), 1.)
        factor = self.minimum_factor + (1-self.minimum_factor)*.5*(1+math.cos(math.pi*progress))
        return self.base_rate*factor

    def apply_update(self, optimizer, update):
        """Only update LR fields; keep Adam state, groups, parameter order and RNG."""
        rate = self.rate_for_update(update)
        for group in optimizer.param_groups:
            if self.kind == 'constant':
                if group['lr'] != rate:
                    raise ValueError('Constant policy requires the original Adam learning rate')
            else:
                group['lr'] = rate
        return rate

    def checkpoint_state(self, completed_updates):
        if (isinstance(completed_updates, bool) or not isinstance(completed_updates, int)
                or completed_updates < 0):
            raise ValueError('Completed scheduler updates must be a nonnegative integer')
        return dict(version=1, policy=self.description(), completed_updates=completed_updates,
                    last_learning_rate=(self.rate_for_update(completed_updates)
                                        if completed_updates else self.base_rate),
                    next_learning_rate=self.rate_for_update(completed_updates+1))

    def validate_resume(self, payload, optimizer):
        """Validate saved strategy/progress against the loaded Adam before drawing D."""
        epoch = payload.get('epoch')
        expected = self.checkpoint_state(epoch)
        saved = payload.get('lr_scheduler_state')
        if saved is None:
            if self.kind != 'constant':
                raise ValueError('Scheduled resume requires saved lr_scheduler_state')
        elif saved != expected:
            raise ValueError('Resume learning-rate strategy or cumulative progress differs')
        for group in optimizer.param_groups:
            actual = _finite_number(group['lr'], 'Restored Adam learning rate')
            if actual != expected['last_learning_rate']:
                raise ValueError('Restored Adam learning rate differs from the saved update policy')
        return expected
