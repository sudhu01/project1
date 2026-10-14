"""Joint incident prototypes fitted from training records only."""
from dataclasses import dataclass
import numpy as np
from .contracts import canonical, digest, real


def vocabulary(spec):
    return tuple('OK:' + code for code in spec.responses if code != 'UNAVAILABLE') + tuple(
        status + ':UNAVAILABLE' for status in ('NO_DATA', 'UNSUPPORTED', 'TIMEOUT', 'ERROR'))


@dataclass(frozen=True)
class PrototypeModel:
    classes: tuple
    action_ids: tuple
    vocabularies: tuple
    labels: tuple
    responses: tuple
    prior: tuple
    smoothing: float
    training_hashes: tuple = ()

    def __post_init__(self):
        real(self.smoothing, 0, 1)
        if not self.classes or len(set(self.classes)) != len(self.classes) or set(self.labels) != set(self.classes):
            raise ValueError('prototype classes need complete coverage')
        if len(set(self.action_ids)) != len(self.action_ids) or len(self.vocabularies) != len(self.action_ids):
            raise ValueError('invalid action vocabulary')
        if len(self.prior) != len(self.labels) or len(self.responses) != len(self.labels) or any(v <= 0 for v in self.prior) or not np.isclose(sum(self.prior), 1):
            raise ValueError('invalid prototype prior')
        for row in self.responses:
            if len(row) != len(self.action_ids) or any(symbol not in vocab for symbol, vocab in zip(row, self.vocabularies)):
                raise ValueError('prototype response outside vocabulary')

    @property
    def hash(self):
        from dataclasses import asdict
        return digest(asdict(self))

    def likelihood(self, action, symbol):
        index = self.action_ids.index(action)
        vocab = self.vocabularies[index]
        if symbol not in vocab:
            raise ValueError('unknown response symbol')
        return np.array([(1-self.smoothing) * (row[index] == symbol) + self.smoothing/len(vocab) for row in self.responses])

    def weights(self, history):
        logs = np.log(self.prior)
        seen = set()
        for action, symbol in history:
            if action in seen:
                raise ValueError('duplicate evidence in belief history')
            seen.add(action)
            values = self.likelihood(action, symbol)
            with np.errstate(divide='ignore'):
                logs += np.log(values)
        if not np.isfinite(logs.max()):
            raise ValueError('evidence impossible under model')
        weights = np.exp(logs-logs.max())
        return weights/weights.sum()

    def diagnosis(self, history):
        weights = self.weights(history)
        return {c: float(sum(w for w, label in zip(weights, self.labels) if label == c)) for c in self.classes}

    def predictions(self, history, action):
        weights = self.weights(history)
        return {symbol: float(weights @ self.likelihood(action, symbol)) for symbol in self.vocabularies[self.action_ids.index(action)]}


def fit_model(training_records, classes, specs, smoothing=0.1, partition='train'):
    if partition != 'train' or not training_records or any(record['split'] != 'train' for record in training_records):
        raise ValueError('fit-model accepts training records only')
    labels, responses, hashes = [], [], []
    for record in training_records:
        label = record['label']
        packets = record['packets']
        if label not in classes or set(packets) != {s.action_id for s in specs}:
            raise ValueError('incomplete training record')
        row = tuple(packets[s.action_id].symbol for s in specs)
        labels.append(label)
        responses.append(row)
        hashes.append(digest({'incident_id': record['incident_id'], 'label': label,
                              'packets': [packets[s.action_id].evidence_id for s in specs]}))
    prior = tuple(1/len(classes)/labels.count(label) for label in labels)
    return PrototypeModel(tuple(classes), tuple(s.action_id for s in specs), tuple(vocabulary(s) for s in specs),
                          tuple(labels), tuple(responses), prior, smoothing, tuple(hashes))
