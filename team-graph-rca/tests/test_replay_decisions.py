from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pytest
from rca_investigation.contracts import IncidentPrivate
from rca_investigation.environment import score_decision
from rca_investigation.fixtures import fixture_environment
from rca_investigation.policies.dp import BoundedDP
from rca_investigation.belief import fit_model

FIXTURES = json.loads((Path(__file__).parent/'fixtures/decisions.json').read_text())


@pytest.mark.parametrize('name', FIXTURES)
def test_fixture_optima(name):
    env, label = fixture_environment(FIXTURES[name])
    planner = BoundedDP(env.model, env.specs, depth=2)
    action = planner.choose(env.observation())
    assert max(planner.stats['values'].values()) == pytest.approx(FIXTURES[name]['expected_depth2'])
    assert (action not in ('STOP','DEFER')) == (FIXTURES[name]['expected_action'] == 'PROBE')
    if name == 'complementary':
        one = BoundedDP(env.model, env.specs, depth=1)
        assert one.choose(env.observation()) == 'STOP'
    while not env.terminated:
        env.step(planner.choose(env.observation()), policy_hash='fixture-planner')
    assert env.probe_return + score_decision(env.decision, label) == pytest.approx(
        score_decision(env.decision, label) - .05*env.ledger.credits)
    assert not env.observation()['mask']


def test_no_second_charge_and_stopping_after_resolution():
    env, _ = fixture_environment(FIXTURES['waste'])
    spec = env.specs[0]
    env.step(spec.action_id)
    planner = BoundedDP(env.model, env.specs)
    assert planner.choose(env.observation()) == 'STOP'
    before = env.ledger.credits
    assert not env.ledger.acquire(spec, env.backend.execute(spec), env.incident)
    assert env.ledger.credits == before
    with pytest.raises(ValueError):
        env.step(spec.action_id)
    with pytest.raises(ValueError):
        env.ledger.acquire(spec, replace(env.backend.execute(spec), duration_ms=1), env.incident)


def test_private_label_isolation():
    env, label = fixture_environment(FIXTURES['perfect'])
    initial = env.observation()
    # Scorer alone takes labels. The environment has no label parameter.
    private_a = IncidentPrivate(env.incident.incident_id, ('A',), 'container_stopped', '{}', '{}', '{}')
    private_b = replace(private_a, resources=('B',))
    assert initial == env.observation()
    env.step(env.specs[0].action_id)
    decision = env.decision
    assert decision.prediction == 'A'
    assert score_decision(decision, private_a.resources[0]) == 1
    assert score_decision(decision, private_b.resources[0]) == 0
    assert 'B' not in json.dumps([p.to_dict() for p in env.ledger.packets])


def test_failure_cost_and_forced_deferral():
    env, _ = fixture_environment(FIXTURES['timeout'], horizon=1)
    prior = env.observation()['diagnoses']
    _, reward, ended = env.step(env.specs[0].action_id)
    assert reward == -.05 and ended
    assert env.observation()['diagnoses'] == prior
    assert env.ledger.credits == 1
    env, _ = fixture_environment(FIXTURES['exhausted'], horizon=1)
    env.step(env.specs[0].action_id)
    assert env.decision.prediction == 'DEFER'
    assert env.decision.stop_reason == 'horizon'


def test_training_only_and_finite_smoothing():
    env, _ = fixture_environment(FIXTURES['perfect'], smoothing=.1)
    weights = env.model.weights(((env.specs[0].action_id, 'ERROR:UNAVAILABLE'),))
    assert np.isfinite(weights).all() and weights.sum() == pytest.approx(1)
    with pytest.raises(ValueError):
        fit_model([], env.model.classes, env.specs, partition='validation')
    with pytest.raises(ValueError):
        env.model.weights(((env.specs[0].action_id, 'OK:ZERO'),)*2)


def test_search_limit_reports_fallback():
    env, _ = fixture_environment(FIXTURES['complementary'])
    planner = BoundedDP(env.model, env.specs, max_expanded_states=1)
    assert planner.choose(env.observation()) == 'STOP'
    assert planner.stats['fallback']
