"""Tabular choices keyed only by public evidence and declared task settings."""
import numpy as np
from ..contracts import canonical, digest


class TabularPolicy:
    def __init__(self, model_hash, task_hash, seed=11, alpha=.1, gamma=1):
        self.model_hash,self.task_hash = model_hash,task_hash
        self.alpha,self.gamma=alpha,gamma
        self.rng=np.random.default_rng(seed)
        self.q,self.visits = {},{}
        self.stats={}

    def key(self, observation):
        if observation.get('schema_version') == 2:
            return canonical({'schema_version': 2, 'task': self.task_hash, 'model': self.model_hash,
                'task_id': observation['task_id'], 'topology': observation['topology_hash'],
                'catalogue': observation['catalogue_hash'], 'encoder': observation['encoding_hash'],
                'objective': observation['objective_hash'], 'entry_resource': observation['entry_resource'],
                'entry_protocol': observation['entry_protocol'], 'alert_context': observation['alert_context'],
                'history': sorted(observation['history']), 'credits': observation['remaining_credits'],
                'probes': observation['remaining_probes'], 'mask': sorted(observation['mask']),
                'lambda_cost': observation['lambda_cost'], 'prices': sorted(observation['prices'].items())})
        return canonical([self.task_hash,self.model_hash,observation['entry_protocol'],
            sorted(observation['history']),observation['remaining_credits'],observation['remaining_probes'],
            observation['lambda_cost'],sorted(observation['prices'].items())])

    def initialize(self, observation, defer=None, correct=None, incorrect=None):
        objective = observation.get('objective', {})
        defer = objective.get('defer', .25) if defer is None else defer
        correct = objective.get('correct', 1) if correct is None else correct
        incorrect = objective.get('incorrect', 0) if incorrect is None else incorrect
        key=self.key(observation)
        stop=incorrect+max(observation['diagnoses'].values())*(correct-incorrect)
        self.q.setdefault(key, {})
        self.visits.setdefault(key, {})
        for action in observation['mask']:
            self.q[key].setdefault(action, stop if action == 'STOP' else defer if action == 'DEFER' else
                                   correct-observation['lambda_cost']*observation['prices'][action])
            self.visits[key].setdefault(action, 0)
        return key

    def probabilities(self, observation, epsilon=0):
        key=self.initialize(observation)
        actions=tuple(observation['mask'])
        best=max(self.q[key][a] for a in actions)
        ties=[a for a in actions if abs(self.q[key][a]-best)<1e-12]
        return {a:epsilon/len(actions)+(1-epsilon)/len(ties) if a in ties else epsilon/len(actions) for a in actions}

    def choose(self, observation, epsilon=0):
        key=self.key(observation)
        count=sum(self.visits.get(key,{}).values())
        self.stats={'unseen':count==0,'lightly_visited':count<5,'fallback':False}
        probabilities=self.probabilities(observation,epsilon)
        actions=tuple(probabilities)
        if epsilon == 0:
            # Stable tie break for evaluation; training samples every greedy tie.
            return max(actions,key=lambda a:probabilities[a])
        return str(self.rng.choice(actions,p=list(probabilities.values())))

    def expected(self, observation, epsilon):
        probabilities=self.probabilities(observation,epsilon)
        return sum(p*self.q[self.key(observation)][a] for a,p in probabilities.items())

    def update(self, state, action, target):
        self.q[state][action]+=self.alpha*(target-self.q[state][action])
        self.visits[state][action]+=1

    @property
    def hash(self): return digest(self.state())

    def state(self):
        return {'model_hash':self.model_hash,'task_hash':self.task_hash,'alpha':self.alpha,'gamma':self.gamma,
                'q':self.q,'visits':self.visits,'rng':self.rng.bit_generator.state}

    @classmethod
    def restore(cls, state):
        policy=cls(state['model_hash'],state['task_hash'],alpha=state['alpha'],gamma=state['gamma'])
        policy.q,policy.visits=state['q'],state['visits']
        policy.rng.bit_generator.state=state['rng']
        return policy


class HybridPolicy:
    def __init__(self, tabular, planner, min_visits=5):
        self.tabular,self.planner,self.min_visits=tabular,planner,min_visits
        self.stats={}
    def choose(self, observation):
        key=self.tabular.key(observation)
        count=sum(self.tabular.visits.get(key,{}).values())
        if count < self.min_visits:
            action=self.planner.choose(observation)
            self.stats={**self.planner.stats,'fallback':True,'unseen':count==0,'lightly_visited':True}
            return action
        action=self.tabular.choose(observation)
        self.stats=self.tabular.stats
        return action
