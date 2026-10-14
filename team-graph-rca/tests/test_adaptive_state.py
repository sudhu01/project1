from copy import deepcopy
from dataclasses import replace
import pytest
from rca_investigation.contracts import digest
from rca_investigation.policies.expected_sarsa import TabularPolicy
from rca_investigation.policies.rules import RulePolicy, GeneralizedRulePolicy
from rca_investigation.fixtures import fixture_environment


def observation():
    env, _ = fixture_environment({'prices': [1], 'classes': ['A', 'B'], 'labels': ['A', 'B'], 'responses': [['ZERO'], ['ONE']]})
    obs = env.observation()
    obs.update(schema_version=2, task_id='sockshop_adaptive_v2a', topology_hash=env.topology.hash,
               encoding_hash=digest('encoder'), objective_hash=digest('objective'), alert_context='{}')
    return obs


@pytest.mark.parametrize('field,value', [('entry_resource', 'different'), ('encoding_hash', digest('new')),
    ('topology_hash', digest('new')), ('catalogue_hash', digest('new')), ('objective_hash', digest('new')),
    ('alert_context', '{"public":"changed"}'), ('mask', ('STOP', 'DEFER'))])
def test_v2_key_changes_with_public_compatibility(field, value):
    obs = observation()
    policy = TabularPolicy('model', 'task')
    assert policy.key(obs) != policy.key({**obs, field: value})
    assert policy.key(obs) == policy.key({**obs, 'private_mechanism': 'cpu', 'incident_id': 'other'})


def test_unseen_objective_and_changed_mask():
    obs = observation()
    obs['objective'] = {'correct': 2., 'incorrect': .1, 'defer': 1.2}
    policy = TabularPolicy('model', 'task')
    key = policy.initialize(obs)
    assert policy.q[key]['STOP'] == pytest.approx(1.05)
    assert policy.q[key]['DEFER'] == 1.2
    action = next(a for a in obs['mask'] if a not in {'STOP', 'DEFER'})
    assert policy.q[key][action] == 2. - obs['lambda_cost'] * obs['prices'][action]
    changed = {**obs, 'mask': ('STOP', 'DEFER')}
    assert policy.choose(changed) == 'DEFER'


def test_rule_uses_same_terminal_utilities():
    obs = observation()
    obs.update(history=(('status/orders/v1', 'OK:STOPPED'),), diagnoses={'orders': .6, 'NO_FAULT': .4},
               objective={'correct': 1., 'incorrect': 0., 'defer': .8})
    assert RulePolicy('dependency-rule').choose(obs) == 'DEFER'


def test_generalized_rule_covers_new_resource_and_paid_evidence():
    from test_metric_contracts import tool
    specs = (tool('status', 'payment'), tool())
    rule = GeneralizedRulePolicy(specs, ['resource/payment', 'NO_FAULT'])
    obs = observation()
    obs['mask'] = tuple(s.action_id for s in specs) + ('STOP', 'DEFER')
    assert rule.choose(obs) == specs[0].action_id
    obs['mask'] = (specs[1].action_id, 'STOP', 'DEFER')
    assert rule.choose(obs) == specs[1].action_id


def test_frozen_rule_transfer_preserves_status_actions_across_versions():
    from rca_investigation.policies.rules import FrozenRuleTransfer
    obs = observation()
    obs['mask'] = ('status/catalogue-db/v2', 'status/orders/v2', 'STOP', 'DEFER')
    policy = FrozenRuleTransfer()
    assert policy.choose(obs) == 'status/catalogue-db/v2'
    obs.update(history=(('status/catalogue-db/v2', 'OK:RUNNING'),), mask=('status/orders/v2', 'STOP', 'DEFER'))
    assert policy.choose(obs) == 'status/orders/v2'
    obs.update(history=(('status/catalogue-db/v2', 'OK:STOPPED'),), diagnoses={'resource/catalogue-db': .99, 'NO_FAULT': .01})
    assert policy.choose(obs) == 'STOP'


def test_v2_explanation_uses_paid_log_text_and_correct_scope():
    from rca_investigation.explain import explain_result
    from test_metric_contracts import packet, tool
    from rca_investigation.contracts import canonical
    spec = tool('logs', 'orders')
    evidence = {'resource_id': 'orders', 'units': 'log_events', 'sample_count': 1,
                'peers': [], 'event_counts': {'DEPENDENCY_ERROR': 1}, 'truncated': False,
                'lines': ['connection refused'], 'warnings': [], 'max_lines': 20, 'max_bytes': 8192}
    acquired = packet(spec, response_code='DEPENDENCY_ERROR', evidence_json=canonical(evidence))
    result = {'decision': {'prediction': 'edge/orders/payment', 'confidence': .9,
                          'evidence_ids': [acquired.evidence_id], 'stop_reason': 'voluntary'},
              'packets': [acquired.to_dict()], 'probabilities': {'edge/orders/payment': .9, 'NO_FAULT': .1}, 'credits': 2}
    explanation = explain_result(result)
    assert explanation['log_excerpts'][0]['lines'] == ['connection refused']
    assert 'edge/orders/payment' in explanation['scope']
    assert 'stopped dependency' not in explanation['summary']
