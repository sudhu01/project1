"""Synthetic arithmetic checks. They establish no Sock Shop accuracy."""
from .contracts import IncidentPublic, ToolResult, ToolSpec, TopologySnapshot
from .catalogue import catalogue_hash
from .belief import PrototypeModel, vocabulary
from .backends.replay import ReplayBackend
from .environment import InvestigationEnv


def fixture_environment(definition, row=0, *, budget=12, horizon=4, smoothing=0.0):
    resources = tuple(f'r{i}' for i in range(len(definition['prices'])))
    topology = TopologySnapshot('fixture_v1', resources, ('Service',)*len(resources), ())
    specs = tuple(ToolSpec(f'status/{r}/v1', r, 'status', price, ('ZERO', 'ONE', 'UNAVAILABLE'), ('status',))
                  for r, price in zip(resources, definition['prices']))
    symbols = tuple(tuple('TIMEOUT:UNAVAILABLE' if v == 'TIMEOUT' else 'OK:'+v for v in row) for row in definition['responses'])
    labels = tuple(definition['labels'])
    classes = tuple(definition['classes'])
    model = PrototypeModel(classes, tuple(s.action_id for s in specs), tuple(vocabulary(s) for s in specs),
                           labels, symbols, tuple(1/len(classes)/labels.count(c) for c in labels), smoothing)
    incident = IncidentPublic('opaque-incident', topology.hash, catalogue_hash(specs), resources[0], 'manual_frontend',
                              '2026-10-06T10:00:00Z', '2026-10-06T10:01:00Z', budget, horizon, .05)
    packets = tuple(ToolResult(s.action_id, s.resource_id, symbol.split(':')[0], symbol.split(':')[1],
                               incident.window_start, incident.window_end, incident.window_end, 'fixture', 0)
                    for s, symbol in zip(specs, symbols[row]))
    backend = ReplayBackend(packets, specs)
    env = InvestigationEnv(topology, specs, model, backend)
    env.reset(incident)
    return env, labels[row]
