from types import SimpleNamespace
from datetime import datetime, timedelta, timezone
import json
import pytest
from rca_investigation.backends.container_stats import ContainerStatsCollector, memory_bytes
from rca_investigation.metrics.cpu import summarize
from rca_investigation.encoding import Encoder
from rca_investigation.contracts import canonical, digest
from rca_investigation.capture_store import write_raw, read_raw
from rca_investigation.contracts_v2 import IncidentPrivateV2


CID = 'a' * 64
START = datetime(2026, 10, 7, tzinfo=timezone.utc)
END = START + timedelta(seconds=120)


def samples(cpu=180):
    return [{'resource_id': 'payment', 'container_id': CID, 'cpu_percent': cpu,
             'quota_cores': 2., 'units': 'docker_cpu_percent',
             'sample_time': (START + timedelta(seconds=2 * i)).isoformat()} for i in range(60)]


def test_cpu_multicore_coverage_and_identity():
    result = summarize(samples(), 'payment', CID, 2., START.isoformat(), END.isoformat(), .8)
    assert result[1] == 'HIGH' and result[2]['median'] == .9
    assert result[2]['p95'] == .9
    assert summarize(samples()[:20], 'payment', CID, 2., START.isoformat(), END.isoformat(), .8)[0] == 'NO_DATA'
    assert summarize(samples(250), 'payment', CID, 2., START.isoformat(), END.isoformat(), .8)[2]['median'] == 1.25
    for changed in ({'quota_cores': 1.}, {'resource_id': 'orders'}, {'cpu_percent': float('nan')}, {'units': 'percent'}):
        data = samples()
        data[0].update(changed)
        with pytest.raises(ValueError):
            summarize(data, 'payment', CID, 2., START.isoformat(), END.isoformat(), .8)
    failed = [{**row, 'availability': 'TIMEOUT'} for row in samples()]
    assert summarize(failed, 'payment', CID, 2., START.isoformat(), END.isoformat(), .8)[0] == 'TIMEOUT'


@pytest.mark.parametrize('value,expected', [('1MiB', 1048576), ('1.5 GiB', 1610612736), ('1MB', 1000000), ('0B', 0)])
def test_docker_memory_bytes(value, expected):
    assert memory_bytes(value) == expected


def test_stats_bounded_command_exact_identity_and_missing_container():
    registry = {'payment': {'container_id': CID, 'cpu_quota_cores': 2., 'memory_limit_bytes': 512 * 1024**2}}
    calls = []
    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps({'ID': CID, 'CPUPerc': '180%', 'MemUsage': '100MiB / 512MiB'}))
    collector = ContainerStatsCollector(registry, runner, now=lambda: END.isoformat())
    result = collector.poll()[0]
    assert result['cpu_percent'] == 180 and result['memory_bytes'] == 100 * 1024**2
    assert calls[0][1]['shell'] is False and calls[0][1]['timeout'] == 5
    assert '--no-trunc' in calls[0][0]
    collector.runner = lambda *a, **k: SimpleNamespace(returncode=0, stdout='[]')
    assert collector.poll()[0]['availability'] == 'NO_DATA'
    collector.runner = lambda *a, **k: SimpleNamespace(returncode=0, stdout=json.dumps({'ID': CID, 'CPUPerc': '0%', 'MemUsage': '0B / 0B'}))
    assert collector.poll()[0]['availability'] == 'NO_DATA'
    collector.lock.acquire()
    with pytest.raises(RuntimeError):
        collector.poll()
    collector.lock.release()


def test_encoder_train_only_and_round_trip(tmp_path):
    split = {'group': 'capture_session', 'splits': {'train': ['a', 'b']}, 'hash': digest('split')}
    records = [{'incident_id': 'a', 'private_cause': 'NO_FAULT', 'fitting_values': {'cpu': [.1, .2, .3]}},
               {'incident_id': 'b', 'private_cause': 'resource/payment', 'fitting_values': {'cpu': [.99]}}]
    encoder = Encoder.fit(records, split)
    assert encoder.values['thresholds']['cpu'] < .31
    encoder.save(tmp_path / 'encoder.json')
    assert Encoder.load(tmp_path / 'encoder.json') == encoder
    records[1]['incident_id'] = 'test-leak'
    with pytest.raises(ValueError):
        Encoder.fit(records, split)


def test_raw_immutable_tampering_and_private_boundary(tmp_path):
    public = {'window_start': START.isoformat(), 'window_end': END.isoformat(), 'incident_id': 'raw-1'}
    private = IncidentPrivateV2(2, 'raw-1', 'NO_FAULT', 'healthy', 'healthy', (), '{}', '{}', '{}')
    receipt = {'status': 'ACCEPTED', 'recovery_confirmed': True}
    path = write_raw(tmp_path, 'raw-1', public, {'stats': samples()}, private, receipt)
    payloads, raw_hash = read_raw(path)
    assert 'private_label.json' not in payloads and len(raw_hash) == 64
    with pytest.raises(ValueError):
        write_raw(tmp_path, 'raw-1', public, {}, private, receipt)
    with pytest.raises(ValueError):
        write_raw(tmp_path, 'raw-2', public, {'cause_id': 'payment'}, private, receipt)
    (path / 'measurements.json').write_text('{}')
    with pytest.raises(ValueError):
        read_raw(path)


def test_request_denominator_no_traffic_and_transport():
    from rca_investigation.metrics.requests import summarize, counter_delta
    rows = [{'resource_id': 'front-end', 'operation': 'checkout', 'outcome': 'completed', 'http_status': 500 if i < 2 else 201} for i in range(20)]
    rows.append({'resource_id': 'front-end', 'operation': 'checkout', 'outcome': 'transport_failure', 'http_status': None})
    result = summarize(rows)
    assert result[2]['error_fraction'] == .1 and result[2]['transport_failures'] == 1
    assert summarize([])[0] == 'NO_DATA'
    transport = [{'resource_id': 'front-end', 'operation': 'checkout', 'outcome': 'transport_failure', 'http_status': None}] * 20
    assert summarize(transport)[1] == 'HIGH_TRANSPORT_FAILURES'
    assert 'error_fraction' not in summarize(transport)[2]
    assert summarize(transport[:19])[0] == 'NO_DATA'
    assert summarize(rows + transport)[1] == 'MIXED_FAILURES'
    assert counter_delta([2, 5, 1, 3]) == 6


def test_latency_timeouts_not_zero_and_operation_identity():
    from rca_investigation.metrics.latency import summarize
    rows = [{'resource_id': 'front-end', 'operation': 'checkout', 'outcome': 'completed', 'http_status': 201, 'duration_ms': 600} for i in range(20)]
    rows.append({'resource_id': 'front-end', 'operation': 'checkout', 'outcome': 'transport_failure', 'transport_reason': 'timeout'})
    result = summarize(rows)
    assert result[1] == 'SLOW' and result[2]['median_ms'] == 600 and result[2]['timeout_count'] == 1
    failed = [{'resource_id': 'front-end', 'operation': 'checkout', 'outcome': 'completed', 'http_status': 500, 'duration_ms': 1}] * 50
    assert summarize(failed)[0] == 'NO_DATA'
    assert summarize(rows + failed)[2]['median_ms'] == 600
    assert summarize([])[0] == 'NO_DATA'
    rows[0]['operation'] = 'catalogue'
    with pytest.raises(ValueError):
        summarize(rows)


def test_memory_pressure_growth_and_allocation_release():
    from rca_investigation.metrics.memory import summarize
    data = samples()
    for i, row in enumerate(data):
        row.update(memory_bytes=300 * 1024**2 + i * 3 * 1024**2, memory_limit_bytes=512 * 1024**2,
                   memory_source='docker_cli_stats_linux_v2')
    result = summarize(data, 'payment', CID, 512 * 1024**2, START.isoformat(), END.isoformat(), .7)
    assert result[1] == 'PRESSURE_AND_GROWTH'
    for row in data:
        row['memory_bytes'] = 100 * 1024**2
    assert summarize(data, 'payment', CID, 512 * 1024**2, START.isoformat(), END.isoformat())[1] == 'NORMAL'
    assert summarize(data[:5], 'payment', CID, 512 * 1024**2, START.isoformat(), END.isoformat())[0] == 'NO_DATA'


def test_composite_routes_only_one_family():
    from rca_investigation.backends.composite import CompositeBackend
    from test_metric_contracts import tool, packet, incident
    specs = (tool(), tool('memory'))
    calls = []
    class Handler:
        def begin(self, value): pass
        def close(self): pass
        def execute(self, spec):
            calls.append(spec.template)
            return packet(spec)
    backend = CompositeBackend(specs, {'cpu': Handler(), 'memory': Handler()}, {'payment': ('cpu', 'memory')})
    backend.begin(incident(specs))
    assert backend.execute(specs[0]).action_id == specs[0].action_id
    assert calls == ['cpu']


def test_reencoding_raw_is_reproducible_and_immutable(tmp_path):
    from test_metric_contracts import tool, incident, ENCODING
    from rca_investigation.encoding import derive_bundle, encode_packet
    from rca_investigation.topology import sockshop_topology
    spec, topology = tool(), sockshop_topology()
    public = incident((spec,))
    encoder = Encoder(canonical({'schema_version': 2, 'thresholds': {'cpu': .5}}))
    from dataclasses import replace
    spec = replace(spec, encoding_hash=encoder.hash)
    public = replace(public, encoding_hash=encoder.hash, catalogue_hash=digest([spec.to_dict()]))
    private = IncidentPrivateV2(2, public.incident_id, 'NO_FAULT', 'healthy', 'healthy', (), '{}', '{}', '{}')
    raw_public = {'incident_id': public.incident_id, 'window_start': public.window_start, 'window_end': public.window_end}
    raw_path = write_raw(tmp_path / 'raw', public.incident_id, raw_public, {}, private,
                         {'status': 'ACCEPTED', 'recovery_confirmed': True})
    def project(spec, raw, public, encoder):
        return encode_packet(spec, public, ('NO_DATA', 'UNAVAILABLE', {}, 'missing_samples'), 'captured-stats-v2',
                             collected_at='2026-10-07T10:02:05Z')
    first = derive_bundle(tmp_path / 'derived-a', raw_path, public, topology, (spec,), encoder, project)
    second = derive_bundle(tmp_path / 'derived-b', raw_path, public, topology, (spec,), encoder, project)
    assert (first / 'checksums.json').read_bytes() == (second / 'checksums.json').read_bytes()
    with pytest.raises(ValueError):
        derive_bundle(tmp_path / 'derived-a', raw_path, public, topology, (spec,), encoder, project)
