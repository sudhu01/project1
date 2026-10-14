"""Public investigation transitions. Independent scorers own labels."""
from .catalogue import catalogue_hash, mask
from .contracts import Decision
from .ledger import EvidenceLedger


class InvestigationEnv:
    def __init__(self, topology, specs, model, backend, *, defer_utility=0.25, correct_utility=1, incorrect_utility=0):
        self.topology, self.specs, self.model, self.backend = topology, tuple(specs), model, backend
        if tuple(s.action_id for s in specs) != model.action_ids:
            raise ValueError('model action mapping mismatch')
        catalogue_hash(self.specs)
        self.defer_utility, self.correct_utility, self.incorrect_utility = defer_utility, correct_utility, incorrect_utility
        self.incident = None

    def reset(self, incident):
        incident.validate_context(self.topology, self.specs)
        self.backend.begin(incident)
        self.incident = incident
        self.ledger = EvidenceLedger()
        self.history = ()
        self.probes = 0
        self.terminated = False
        self.decision = None
        self.probe_return = 0.0
        self.capabilities = self.backend.capabilities(self.topology)
        return self.observation()

    def observation(self):
        if self.incident is None:
            raise RuntimeError('reset first')
        remaining = self.incident.budget - self.ledger.credits
        probes_left = self.incident.max_probes - self.probes
        result = {'topology': self.topology.to_dict(), 'entry_resource': self.incident.entry_resource,
                'entry_protocol': self.incident.entry_protocol, 'catalogue_hash': self.incident.catalogue_hash,
                'history': self.history, 'diagnoses': self.model.diagnosis(self.history),
                'prices': {s.action_id: s.credits for s in self.specs}, 'remaining_credits': remaining,
                'remaining_probes': probes_left, 'lambda_cost': self.incident.lambda_cost,
                'mask': mask(self.specs, self.ledger.acquired_actions, remaining, probes_left, self.capabilities, self.terminated)}
        result['objective'] = {'defer': self.defer_utility, 'correct': self.correct_utility, 'incorrect': self.incorrect_utility}
        if getattr(self.incident, 'schema_version', 1) == 2:
            result.update(schema_version=2, task_id=self.incident.task_id, topology_hash=self.topology.hash,
                          encoding_hash=self.incident.encoding_hash, objective_hash=self.incident.objective_hash,
                          alert_context=self.incident.alert_context_json)
        return result

    def terminal_action(self):
        probabilities = self.model.diagnosis(self.history)
        stop = self.incorrect_utility + max(probabilities.values()) * (self.correct_utility-self.incorrect_utility)
        return 'DEFER' if self.defer_utility >= stop else 'STOP'

    def _finish(self, action, reason, policy_hash):
        probabilities = self.model.diagnosis(self.history)
        prediction = max(self.model.classes, key=lambda c: probabilities[c])
        self.decision = Decision('DEFER' if action == 'DEFER' else prediction,
                                 0.0 if action == 'DEFER' else probabilities[prediction],
                                 self.ledger.evidence_ids, policy_hash, self.model.hash, reason)
        self.terminated = True

    def step(self, action, *, policy_hash='unspecified'):
        before = self.observation()
        if action not in before['mask']:
            raise ValueError('invalid, repeated, unaffordable or unsupported action')
        if action in ('STOP', 'DEFER'):
            self._finish(action, 'voluntary', policy_hash)
            return self.observation(), 0.0, True
        spec = next(s for s in self.specs if s.action_id == action)
        packet = self.backend.execute(spec)
        if not self.ledger.acquire(spec, packet, self.incident):
            raise RuntimeError('masked duplicate was dispatched')
        self.probes += 1
        self.history += ((action, packet.symbol),)
        reward = -self.incident.lambda_cost * spec.credits
        self.probe_return += reward
        after = self.observation()
        if not any(a not in ('STOP', 'DEFER') for a in after['mask']):
            reason = 'budget' if after['remaining_credits'] == 0 else 'horizon' if after['remaining_probes'] == 0 else 'no_probe_available'
            self._finish(self.terminal_action(), reason, policy_hash)
            after = self.observation()
        return after, reward, self.terminated

    def close(self):
        self.backend.close()


def score_decision(decision, label, *, correct=1.0, incorrect=0.0, defer=0.25):
    return defer if decision.prediction == 'DEFER' else correct if decision.prediction == label else incorrect
