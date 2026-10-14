import numpy as np


class RulePolicy:
    def __init__(self, name, seed=0, objective=None):
        if name not in {'stop-immediately', 'defer-immediately', 'random-valid', 'dependency-rule'}:
            raise ValueError('unknown rule')
        self.name, self.rng = name, np.random.default_rng(seed)
        self.objective = objective or {'correct': 1., 'incorrect': 0., 'defer': .25}

    def choose(self, observation):
        if self.name == 'stop-immediately':
            return 'STOP'
        if self.name == 'defer-immediately':
            return 'DEFER'
        if self.name == 'random-valid':
            return str(self.rng.choice(observation['mask']))
        acquired = dict(observation['history'])
        objective = observation.get('objective', self.objective)
        value = objective['incorrect'] + max(observation['diagnoses'].values()) * (objective['correct'] - objective['incorrect'])
        terminal = 'STOP' if value > objective['defer'] else 'DEFER'
        if any(acquired.get(f'status/{r}/v1') == 'OK:STOPPED' for r in ('catalogue-db', 'orders')):
            return terminal
        for resource in ('catalogue-db', 'orders'):
            action = f'status/{resource}/v1'
            if action in observation['mask']:
                return action
        return terminal


class FixedSequencePolicy:
    """Declared fixed acquisition order with a shared terminal utility rule."""
    def __init__(self, sequence, confidence=1.):
        self.sequence = tuple(sequence)
        if len(set(self.sequence)) != len(self.sequence) or any(a in {'STOP', 'DEFER'} for a in self.sequence):
            raise ValueError('invalid fixed sequence')
        if not 0 <= confidence <= 1:
            raise ValueError('invalid stopping confidence')
        self.confidence = confidence

    def choose(self, observation):
        objective = observation['objective']
        probability = max(observation['diagnoses'].values())
        value = objective['incorrect'] + probability * (objective['correct'] - objective['incorrect'])
        terminal = 'STOP' if value > objective['defer'] else 'DEFER'
        if probability >= self.confidence:
            return terminal
        for action in self.sequence:
            if action in observation['mask']:
                return action
        return terminal


class FrozenRuleTransfer:
    """Translate status action versions without adding decisions to the v1 rule."""
    def __init__(self, seed=11, objective=None):
        self.rule = RulePolicy('dependency-rule', seed, objective)

    def choose(self, observation):
        def old(action):
            return action[:-2] + 'v1' if action.startswith('status/') and action.endswith('v2') else action
        mapping = {old(action): action for action in observation['mask']}
        translated = {**observation, 'mask': tuple(mapping),
                      'history': tuple((old(action), response) for action, response in observation['history'])}
        return mapping[self.rule.choose(translated)]


class GeneralizedRulePolicy(FixedSequencePolicy):
    """Status first for declared resource causes, then the registered evidence order."""
    def __init__(self, specs, diagnoses, evidence_order=(), confidence=.95):
        roots = {cause.removeprefix('resource/') for cause in diagnoses if cause.startswith('resource/')}
        status = [s.action_id for s in specs if s.template == 'status' and s.resource_id in roots]
        if set(evidence_order) - {s.action_id for s in specs}:
            raise ValueError('unregistered rule evidence')
        remaining = [s.action_id for s in specs if s.action_id not in status and s.action_id not in evidence_order]
        super().__init__(status + list(evidence_order) + remaining, confidence)
