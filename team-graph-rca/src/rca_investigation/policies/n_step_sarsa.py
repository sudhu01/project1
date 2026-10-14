"""Observed two-reward SARSA updates, with terminal tail flushing."""
from ..environment import score_decision


def learn_episode(env, label, policy, epsilon, *, method='expected-sarsa'):
    if method not in ('expected-sarsa','two-step-sarsa'):
        raise ValueError('unsupported tabular update')
    objective={'defer':env.defer_utility,'correct':env.correct_utility,'incorrect':env.incorrect_utility}
    observation=env.observation()
    policy.initialize(observation,**objective)
    action=policy.choose(observation,epsilon)
    pending=[]
    trace=[]
    while True:
        key=policy.key(observation)
        next_observation,reward,done=env.step(action,policy_hash='training')
        if done:
            reward+=score_decision(env.decision,label,**objective)
            next_action=None
        else:
            policy.initialize(next_observation,**objective)
            # Sample before updating so n-step bootstrapping uses behavior actions.
            next_action=policy.choose(next_observation,epsilon)
        trace.append({'state':key,'action':action,'reward':reward,'epsilon':epsilon,'terminal':done})
        if method=='expected-sarsa':
            target=reward if done else reward+policy.gamma*policy.expected(next_observation,epsilon)
            policy.update(key,action,target)
        else:
            pending.append((key,action,reward))
            if len(pending)==2:
                target=pending[0][2]+policy.gamma*pending[1][2]
                if not done:
                    target+=policy.gamma**2*policy.q[policy.key(next_observation)][next_action]
                state,selected,_=pending.pop(0)
                policy.update(state,selected,target)
            if done:
                while pending:
                    state,selected,_=pending[0]
                    target=sum(policy.gamma**i*row[2] for i,row in enumerate(pending))
                    policy.update(state,selected,target)
                    pending.pop(0)
        if done:
            return {'return':sum(t['reward'] for t in trace),'credits':env.ledger.credits,'trace':trace}
        observation,action=next_observation,next_action
