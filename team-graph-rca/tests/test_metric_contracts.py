from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pytest
import yaml
from rca_investigation.contracts import canonical, digest, ToolSpec, ToolResult, IncidentPublic
from rca_investigation.contracts_v2 import *
from rca_investigation.catalogue import build_catalogue, catalogue_hash
from rca_investigation.config import load_config
from rca_investigation.config_v2 import load_v2_config
from rca_investigation.dataset import write_bundle, load_public_bundle, load_private_label
from rca_investigation.experiment import label_name
from rca_investigation.backends.replay import ReplayBackend
from rca_investigation.environment import score_decision
from rca_investigation.topology import sockshop_topology


ENCODING = digest({'test_encoder': 2})
START, END, COLLECTED = '2026-10-07T10:00:00Z', '2026-10-07T10:02:00Z', '2026-10-07T10:02:05Z'


def tool(family='cpu', resource='payment'):
    params = {} if family == 'status' else {'window_seconds': 120}
    if family == 'logs':
        params.update(max_lines=20, max_bytes=8192)
    if family in {'request_errors', 'latency'}:
        params.update(operation='checkout', min_samples=20)
    return ToolSpecV2(f'{family}/{resource}/v2', resource, family, 2, canonical(params),
                      1 if family == 'status' else 2, (family,), VOCABULARIES[family], ENCODING)


def packet(spec=None, **changes):
    spec = spec or tool()
    evidence = {'resource_id': spec.resource_id, 'units': UNITS[spec.template], 'sample_count': 60}
    if spec.template == 'status':
        evidence['state'] = 'running'
    elif spec.template == 'cpu':
        evidence.update(median=.1, p95=.2, high_fraction=0., coverage=1., quota_cores=2.)
    elif spec.template == 'memory':
        evidence.update(median_bytes=100, p95_bytes=120, limit_bytes=1000, limit_fraction=.1,
                        peak_fraction=.12, growth_bytes_per_second=0., growth_span_seconds=118., coverage=1.)
    elif spec.template == 'request_errors':
        evidence.update(operation='checkout', completed=60, attempts=60, server_errors=0, semantic_failures=0,
                        transport_failures=0, error_fraction=0., transport_fraction=0., achieved_rate_per_second=.5)
    elif spec.template == 'latency':
        evidence.update(operation='checkout', median_ms=100., p95_ms=120., timeout_count=0, failed_count=0, coverage=1.)
    elif spec.template == 'logs':
        evidence.update(sample_count=0, peers=[], event_counts={}, truncated=False, lines=[], warnings=[], max_lines=20, max_bytes=8192)
    kwargs = dict(action_id=spec.action_id, resource_id=spec.resource_id, availability='OK',
                  response_code=spec.responses[0], window_start=START, window_end=END,
                  sample_time=END, source='captured-stats-v2', duration_ms=1,
                  collected_at=COLLECTED, encoding_hash=ENCODING, evidence_json=canonical(evidence))
    return ToolResultV2(**{**kwargs, **changes})


def incident(specs):
    topology = sockshop_topology()
    return IncidentPublicV2('opaque-1', topology.hash, catalogue_hash(specs), 'front-end',
                            'manual_frontend', START, END, 10, 5, .05,
                            task_id='sockshop_adaptive_v2b', encoding_hash=ENCODING,
                            objective_hash=digest({'correct': 1, 'incorrect': 0, 'defer': .25}))


@pytest.mark.parametrize('family', list(VOCABULARIES))
def test_all_tools_round_trip_and_strict_parameters(family):
    spec = tool(family)
    assert decode_record(json.loads(spec.to_json()), 'tool') == spec
    assert spec.credits == (1 if family == 'status' else 2)
    with pytest.raises(ValueError):
        replace(spec, parameters_json=canonical({'window_seconds': 120, 'unknown': 1}))
    with pytest.raises(ValueError):
        replace(spec, response_vocabulary=spec.responses + ('NEW_CODE',))


@pytest.mark.parametrize('changes', [
    {'collected_at': START}, {'encoding_hash': 'bad'}, {'schema_version': True},
    {'availability': 'TIMEOUT'}, {'response_code': 'UNAVAILABLE'},
    {'evidence_json': canonical({'units': 'percent', 'resource_id': 'payment', 'sample_count': 60})},
    {'evidence_json': canonical({'units': 'quota_fraction', 'resource_id': 'orders', 'sample_count': 60})},
    {'evidence_json': canonical({'units': 'quota_fraction', 'resource_id': 'payment', 'sample_count': 0})},
    {'evidence_json': canonical({'units': 'quota_fraction', 'resource_id': 'payment', 'sample_count': 60, 'median_bytes': 1})},
    {'raw_json': canonical({'memory': 100})},
])
def test_packet_rejects_invalid_measurement_and_free_evidence(changes):
    with pytest.raises(ValueError):
        packet(**changes)


@pytest.mark.parametrize('availability', ['NO_DATA', 'UNSUPPORTED', 'TIMEOUT', 'ERROR'])
def test_unavailable_is_explicit(availability):
    result = packet(availability=availability, response_code='UNAVAILABLE', reason='insufficient_samples', evidence_json='{}')
    assert result.symbol == f'{availability}:UNAVAILABLE'
    assert decode_record(json.loads(result.to_json()), 'packet') == result


def test_historical_packet_and_encoder_identity():
    spec = tool()
    public = incident((spec,))
    result = packet(spec)
    result.validate_context(spec, public)
    assert result.collected_at > result.sample_time
    with pytest.raises(ValueError):
        replace(result, encoding_hash=digest({'different': 1})).validate_context(spec, public)


def test_edge_cause_replay_and_private_scoring(tmp_path):
    topology, specs = sockshop_topology(), (tool(),)
    public, result = incident(specs), packet(specs[0])
    private = IncidentPrivateV2(2, public.incident_id, 'edge/orders/payment', 'edge', 'dependency_latency',
                                ('orders', 'payment'), '{}', '{}', '{}')
    private.validate_context(topology, public.task_id)
    assert label_name(private) == 'edge/orders/payment'
    path = write_bundle(tmp_path, public, topology, (result,), private,
                        {'status': 'ACCEPTED', 'recovery_confirmed': True})
    loaded, _, packets = load_public_bundle(path, specs)
    assert loaded == public and load_private_label(path) == private
    replay = ReplayBackend(packets, specs)
    replay.begin(loaded)
    assert replay.execute(specs[0]) == result
    before = result.evidence_id
    resource_label = IncidentPrivateV2(2, public.incident_id, 'resource/payment', 'resource', 'cpu_contention',
                                       ('payment',), '{}', '{}', '{}')
    from rca_investigation.contracts import Decision
    decision = Decision(private.cause_id, 1., (before,), 'policy', 'model', 'voluntary')
    assert score_decision(decision, label_name(private)) == 1
    assert score_decision(decision, label_name(resource_label)) == 0
    assert result.evidence_id == before
    with pytest.raises(ValueError):
        private.validate_context(topology, 'sockshop_adaptive_v2a')
    with pytest.raises(ValueError):
        CauseSpec('edge/payment/orders', 'edge', '', 'payment', 'orders').validate_topology(topology)


def test_v1_saved_records_and_result_hashes_unchanged():
    root = Path(__file__).resolve().parents[1]
    result = json.loads((root / 'experiments/stop_v1_result.json').read_text())
    for name, expected in result['artifact_sha256'].items():
        assert hashlib.sha256((root / result['artifacts'][name]).read_bytes()).hexdigest() == expected
    specs = build_catalogue(load_config(root / 'configs/investigation_v1.yaml', sockshop_topology()), sockshop_topology())
    assert all(type(s) is ToolSpec and 'schema_version' not in s.to_dict() for s in specs)
    assert all(decode_record(json.loads(s.to_json()), 'tool').to_json() == s.to_json() for s in specs)
    sample = ToolResult('status/orders/v1', 'orders', 'OK', 'RUNNING', START, END, END, 'docker', 0)
    assert decode_record(json.loads(sample.to_json()), 'packet').evidence_id == sample.evidence_id
    # All saved v1 evaluation bundles retain their checksums and exact serialization.
    from rca_investigation.dataset import verify_bundle
    for path in (root / 'datasets/sockshop_v1').iterdir():
        if not path.is_dir() or path.name.endswith('.pending'):
            continue
        verify_bundle(path)
        for filename, kind in [('public.json', 'public'), ('private_label.json', 'private')]:
            payload = json.loads((path / filename).read_text())
            assert digest(decode_record(payload, kind).to_dict()) == digest(payload)
        for payload in json.loads((path / 'responses.json').read_text()):
            assert digest(decode_record(payload, 'packet').to_dict()) == digest(payload)


def adaptive_config():
    root = Path(__file__).resolve().parents[1]
    value = yaml.safe_load((root / 'configs/investigation_v1.yaml').read_text())
    value.update(schema_version=2, task_id='sockshop_adaptive_v2a', diagnoses=list(PROFILES['sockshop_adaptive_v2a']),
                 encoding_hash=ENCODING,
                 actions=[tool('status', r).to_dict() for r in value['resources']] + [tool().to_dict()],
                 paths={key: f'experiments/v2a/{key}' for key in ('dataset', 'raw_dataset', 'encoder', 'model', 'runs', 'resource_registry')})
    del value['tools']
    value['episode'] = {'observation_window_seconds': 120, 'budget': 10, 'max_probes': 5}
    # from_dict consumes serialized lists, not dataclass tuples.
    return json.loads(canonical(value))


def test_registered_task_contract_and_no_v1_path_fallback(tmp_path):
    value = adaptive_config()
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(value))
    loaded = load_config(path, sockshop_topology())
    assert len(build_catalogue(loaded, sockshop_topology())) == 6
    for changed in (
        {**value, 'diagnoses': ['orders']},
        {**value, 'task_id': 'unregistered'},
        {**value, 'paths': {**value['paths'], 'runs': 'runs/integration_v1'}},
        {**value, 'actions': value['actions'] + [tool('memory').to_dict()]},
    ):
        with pytest.raises(ValueError):
            load_v2_config(json.loads(canonical(changed)), sockshop_topology())


def test_unknown_schema_and_boolean_version_fail():
    for version in (0, 3, True, '2'):
        with pytest.raises(ValueError):
            decode_record({'schema_version': version}, 'packet')
