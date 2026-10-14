"""Controlled decision problems, separate from physical experiment results."""
from itertools import product
from .fixtures import fixture_environment
from .policies.dp import BoundedDP


def three_probe_definition():
    rows = list(product(('ZERO', 'ONE'), repeat=3))
    return {'prices': [1, 1, 1], 'classes': ['A', 'B'],
            'labels': ['A' if sum(v == 'ONE' for v in row) % 2 == 0 else 'B' for row in rows],
            'responses': [list(row) for row in rows]}


def contextual_definition():
    rows = list(product(('ZERO', 'ONE'), repeat=3))
    return {'prices': [1, 1, 1], 'classes': ['A', 'B'],
            'labels': ['A' if row[1 if row[0] == 'ZERO' else 2] == 'ZERO' else 'B' for row in rows],
            'responses': [list(row) for row in rows]}


def validate_adaptive_fixtures():
    definition = three_probe_definition()
    env, _ = fixture_environment(definition)
    one, two, three = (BoundedDP(env.model, env.specs, depth=d) for d in (1, 2, 3))
    actions = [planner.choose(env.observation()) for planner in (one, two, three)]
    values = [max(planner.stats['values'].values()) for planner in (one, two, three)]
    env.close()
    contextual, _ = fixture_environment(contextual_definition())
    planner = BoundedDP(contextual.model, contextual.specs, depth=2)
    first = planner.choose(contextual.observation())
    contextual.step(first)
    second = planner.choose(contextual.observation())
    contextual.close()
    return {'passed': actions[:2] == ['STOP', 'STOP'] and actions[2] not in {'STOP', 'DEFER'} and abs(values[2] - .85) < 1e-10
            and first == 'status/r0/v1' and second == 'status/r1/v1',
            'synthetic_only': True, 'three_probe': {'actions': actions, 'returns': values},
            'contextual': {'first': first, 'next_after_zero': second}}
