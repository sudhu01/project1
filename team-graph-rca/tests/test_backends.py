from datetime import datetime, timezone
import json
import subprocess
from types import SimpleNamespace
import pytest
import requests
from rca_investigation.contracts import *
from rca_investigation.catalogue import catalogue_hash
from rca_investigation.backends.docker import DockerBackend, compose_rows, resolve_containers, cpu_sample
from rca_investigation.backends.loki import LokiBackend
from rca_investigation.backends.prometheus import PrometheusBackend


def context(template, resource='orders'):
    responses = {'status': ('RUNNING','STOPPED','OTHER_STATE','UNAVAILABLE'),
                 'cpu': ('NORMAL','HIGH','INSUFFICIENT_DATA','UNAVAILABLE'),
                 'logs': ('NO_RELEVANT_ERROR','DEPENDENCY_ERROR','RESOURCE_ERROR','OTHER_ERROR','UNAVAILABLE')}[template]
    params = {'status': {}, 'cpu': {'window_seconds': 60}, 'logs': {'max_lines': 20}}[template]
    spec = ToolSpec(f'{template}/{resource}/v1', resource, template, 1, responses, (template,), canonical(params))
    topology = TopologySnapshot('v1', (resource,), ('Service',), ())
    incident = IncidentPublic('opaque', topology.hash, catalogue_hash((spec,)), resource, 'manual_frontend',
                              '2026-10-06T10:00:00Z', '2026-10-06T10:01:00Z', 6, 4, .05)
    return spec, incident


def test_compose_json_lines_arrays_and_exact_identity():
    rows = [{'Service': 'catalogue', 'Project': 'rca-lab', 'ID': 'a'*64},
            {'Service': 'catalogue-db', 'Project': 'rca-lab', 'ID': 'b'*64}]
    for output in (json.dumps(rows), '\n'.join(json.dumps(r) for r in rows)):
        parsed = compose_rows(output)
        assert resolve_containers(parsed, ['catalogue'])['catalogue'] == 'a'*64
    with pytest.raises(ValueError):
        resolve_containers(rows[1:], ['catalogue'])
    with pytest.raises(ValueError):
        resolve_containers(rows+rows[:1], ['catalogue'])


@pytest.mark.parametrize('failure,availability', [(None,'OK'), ('timeout','TIMEOUT'), ('failed','ERROR')])
def test_docker_failure_is_not_stopped(failure, availability):
    spec, incident = context('status')
    calls = []
    def runner(argv, **kwargs):
        calls.append(argv)
        assert kwargs['shell'] is False and kwargs['timeout'] == 5
        if argv[1] == 'compose':
            payload = [{'Service':'orders','Project':'rca-lab','ID':'a'*64}]
        else:
            if failure == 'timeout':
                raise subprocess.TimeoutExpired(argv, 5)
            if failure == 'failed':
                return SimpleNamespace(returncode=1, stdout='')
            payload = [{'Id':'a'*64, 'Config':{'Labels':{'com.docker.compose.service':'orders', 'com.docker.compose.project':'rca-lab'}},
                        'State':{'Status':'exited','ExitCode':0}}]
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload))
    backend = DockerBackend((spec,), runner=runner, now=lambda: datetime(2026,10,6,10,0,30,tzinfo=timezone.utc))
    backend.begin(incident)
    packet = backend.execute(spec)
    assert packet.availability == availability
    assert packet.response_code == ('STOPPED' if failure is None else 'UNAVAILABLE')
    assert packet == backend.execute(spec) and len(calls) == 2
    packet.validate_context(spec, incident)


class Response:
    def __init__(self, payload):
        self.payload = payload
    def raise_for_status(self): pass
    def iter_content(self, chunk_size): yield canonical(self.payload).encode()
    def close(self): pass


class Session:
    def __init__(self, payload=None, timeout=False):
        self.payload, self.timeout = payload, timeout
        self.params = None
    def get(self, url, params, **kwargs):
        self.params = params
        assert kwargs['stream'] and kwargs['timeout'] == (3,5)
        if self.timeout:
            raise requests.Timeout()
        return Response(self.payload)
    def close(self): pass


def test_loki_empty_success_and_backend_failure_differ():
    spec, incident = context('logs')
    session = Session({'status':'success','data':{'resultType':'streams','result':[]}})
    backend = LokiBackend((spec,), {'orders':'/rca-lab-orders-1'}, session=session)
    backend.begin(incident)
    packet = backend.execute(spec)
    assert packet.availability == 'OK' and packet.response_code == 'NO_RELEVANT_ERROR'
    assert session.params['query'] == '{container="/rca-lab-orders-1"}'
    assert session.params['limit'] == 20 and 'start' in session.params and 'end' in session.params
    backend = LokiBackend((spec,), {'orders':'/rca-lab-orders-1'}, session=Session(timeout=True))
    backend.begin(incident)
    assert backend.execute(spec).availability == 'TIMEOUT'


def test_prometheus_no_data_not_normal_and_selector_exact():
    spec, incident = context('cpu', 'catalogue')
    session = Session({'status':'success','data':{'resultType':'matrix','result':[]}})
    backend = PrometheusBackend((spec,), {'catalogue':'catalogue:80'}, threshold=80, session=session)
    backend.begin(incident)
    result = backend.execute(spec)
    assert result.availability == 'NO_DATA' and result.response_code == 'UNAVAILABLE'
    assert 'instance="catalogue:80"' in session.params['query'] and '=~' not in session.params['query']
    assert session.params['start'] == incident.window_start


def test_raw_container_cpu_can_exceed_100_percent():
    def runner(argv, **kwargs):
        assert argv[-1] == 'a'*64
        return SimpleNamespace(returncode=0, stdout=json.dumps({'CPUPerc':'220.5%'}))
    sample = cpu_sample('a'*64, runner)
    assert sample['cpu_percent'] == 220.5 and sample['units'] == 'docker_cpu_percent'
