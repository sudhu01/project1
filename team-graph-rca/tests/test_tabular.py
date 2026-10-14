import json
import numpy as np
from importlib.resources import files
import pytest
from rca_investigation.fixtures import fixture_environment
from rca_investigation.policies.expected_sarsa import TabularPolicy
from rca_investigation.policies.n_step_sarsa import learn_episode


@pytest.mark.parametrize('method',['expected-sarsa','two-step-sarsa'])
@pytest.mark.parametrize('name',['perfect','useless','waste','expensive','complementary'])
@pytest.mark.parametrize('seed',[11,22,33,44,55])
def test_learns_fixture_useful_probes_and_stopping(name,method,seed):
    definitions=json.loads(files('rca_investigation').joinpath('decision_fixtures.json').read_text())
    definition=definitions[name]
    initial,_=fixture_environment(definition)
    policy=TabularPolicy(initial.model.hash,'fixture',seed=seed)
    best=None
    selection=(-float('inf'),-float('inf'))
    sampler=np.random.default_rng(seed+100000)
    episodes=10000 if name=='expensive' else 2500
    for episode in range(episodes):
        env,label=fixture_environment(definition,row=int(sampler.integers(len(definition['labels']))))
        epsilon=.2-(.15*min(1,episode/8000))
        learn_episode(env,label,policy,epsilon,method=method)
        env.close()
        if (episode+1)%250==0:
            from rca_investigation.environment import score_decision
            returns,credits=[],[]
            for row in range(len(definition['labels'])):
                validation,label=fixture_environment(definition,row=row)
                while not validation.terminated:
                    validation.step(policy.choose(validation.observation()))
                returns.append(validation.probe_return+score_decision(validation.decision,label))
                credits.append(validation.ledger.credits)
            value=(sum(returns)/len(returns),-sum(credits)/len(credits))
            if value>selection:
                selection=value
                best=json.loads(json.dumps(policy.state()))
    policy=TabularPolicy.restore(best)
    env,_=fixture_environment(definition)
    choice=policy.choose(env.observation())
    if name in ('perfect','waste','complementary'):
        assert choice not in ('STOP','DEFER')
        env.step(choice)
        if name in ('perfect','waste') and not env.terminated:
            assert policy.choose(env.observation())=='STOP'
    else:
        assert choice=='STOP'


def test_state_excludes_incident_sequence_and_private_values():
    definition={'prices':[1],'classes':['A','B'],'labels':['A','B'],'responses':[['ZERO'],['ONE']]}
    env,_=fixture_environment(definition)
    policy=TabularPolicy(env.model.hash,'fixture')
    obs=env.observation()
    assert policy.key(obs)==policy.key({**obs,'incident_id':'changed','private_label':'B','capture_sequence':100})
    policy.initialize(obs)
    probabilities=policy.probabilities(obs,.2)
    assert sum(probabilities.values())==pytest.approx(1)
    restored=TabularPolicy.restore(json.loads(json.dumps(policy.state())))
    assert [policy.choose(obs,.2) for _ in range(20)]==[restored.choose(obs,.2) for _ in range(20)]
    assert policy.stats['unseen']
