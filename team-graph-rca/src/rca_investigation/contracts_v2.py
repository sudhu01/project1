"""Explicit v2 records. V1 serializers and canonical hashes remain unchanged."""
from dataclasses import dataclass
import json
import re
from .contracts import (JsonRecord, ToolResult, IncidentPublic, canonical, digest,
                        identifier, integer, real, timestamp, unique)


VOCABULARIES = {
    'status': ('RUNNING', 'STOPPED', 'OTHER_STATE', 'UNAVAILABLE'),
    'cpu': ('NORMAL', 'HIGH', 'UNAVAILABLE'),
    'request_errors': ('NORMAL', 'HIGH_HTTP_ERRORS', 'HIGH_TRANSPORT_FAILURES', 'MIXED_FAILURES', 'UNAVAILABLE'),
    'logs': ('NO_RELEVANT_ERROR', 'DEPENDENCY_ERROR', 'RESOURCE_ERROR', 'OTHER_ERROR', 'UNAVAILABLE'),
    'latency': ('NORMAL', 'SLOW', 'UNAVAILABLE'),
    'memory': ('NORMAL', 'PRESSURE', 'GROWTH', 'PRESSURE_AND_GROWTH', 'UNAVAILABLE'),
}
UNITS = {'status': 'container_state', 'cpu': 'quota_fraction',
         'request_errors': 'fraction', 'logs': 'log_events',
         'latency': 'milliseconds', 'memory': 'bytes'}


def sha256(value):
    if type(value) is not str or not re.fullmatch('[a-f0-9]{64}', value):
        raise ValueError('expected SHA256 hash')


def object_json(value, limit=32768):
    parsed = json.loads(value)
    if type(parsed) is not dict or canonical(parsed) != value or len(value.encode()) > limit:
        raise ValueError('expected bounded canonical JSON object')
    return parsed


@dataclass(frozen=True)
class ToolSpecV2(JsonRecord):
    action_id: str
    resource_id: str
    template: str
    schema_version: int
    parameters_json: str
    credits: int
    required_capabilities: tuple
    response_vocabulary: tuple
    encoding_hash: str

    def __post_init__(self):
        identifier(self.resource_id)
        if type(self.schema_version) is not int or self.schema_version != 2:
            raise ValueError('unsupported tool schema')
        if self.template not in VOCABULARIES or self.action_id != f'{self.template}/{self.resource_id}/v2':
            raise ValueError('invalid v2 action')
        integer(self.credits, 1)
        unique(self.required_capabilities)
        if not self.required_capabilities:
            raise ValueError('tool requires a public capability')
        unique(self.response_vocabulary)
        if self.template == 'logs':
            base = VOCABULARIES['logs']
            if not set(base) <= set(self.response_vocabulary) or any(
                code not in base and not code.startswith('DEPENDENCY_ERROR/') for code in self.response_vocabulary):
                raise ValueError('invalid log vocabulary')
        elif self.response_vocabulary != VOCABULARIES[self.template]:
            raise ValueError('unsupported response vocabulary')
        sha256(self.encoding_hash)
        params = object_json(self.parameters_json, 2048)
        expected = set() if self.template == 'status' else {'window_seconds'}
        if self.template == 'logs':
            expected |= {'max_lines', 'max_bytes'}
        if self.template in {'request_errors', 'latency'}:
            expected |= {'operation', 'min_samples'}
        if set(params) != expected:
            raise ValueError('unsupported tool parameters')
        if 'window_seconds' in params and params['window_seconds'] != 120:
            raise ValueError('v2 requires a fixed 120 second window')
        for name in ('window_seconds', 'max_lines', 'max_bytes', 'min_samples'):
            if name in params:
                integer(params[name], 1)
        if self.template == 'logs' and (params['max_lines'] != 20 or params['max_bytes'] != 8192):
            raise ValueError('unsupported log bounds')
        if 'operation' in params and params['operation'] != 'checkout':
            raise ValueError('unregistered workload operation')
        if 'min_samples' in params and params['min_samples'] != 20:
            raise ValueError('v2 requires twenty request samples')

    @property
    def responses(self):
        return self.response_vocabulary

    @property
    def capabilities(self):
        return self.required_capabilities

    def validate_resource(self, topology):
        if self.resource_id not in topology.resources:
            raise ValueError('unknown resource')
        peers = {code.split('/', 1)[1] for code in self.response_vocabulary if '/' in code}
        if any((self.resource_id, peer) not in topology.calls for peer in peers):
            raise ValueError('log peer must be a registered dependency')


@dataclass(frozen=True)
class ToolResultV2(ToolResult):
    schema_version: int = 2
    collected_at: str = ''
    encoding_hash: str = ''
    evidence_json: str = '{}'
    reason: str = ''

    def __post_init__(self):
        super().__post_init__()
        if type(self.schema_version) is not int or self.schema_version != 2:
            raise ValueError('unsupported packet schema')
        if timestamp(self.collected_at) < timestamp(self.sample_time):
            raise ValueError('collection precedes evidence')
        sha256(self.encoding_hash)
        if type(self.reason) is not str or len(self.reason.encode()) > 256:
            raise ValueError('invalid bounded reason')
        if self.availability == 'OK' and self.response_code == 'UNAVAILABLE':
            raise ValueError('OK cannot be unavailable')
        if self.availability != 'OK' and not self.reason:
            raise ValueError('unavailable evidence requires a reason')
        evidence = object_json(self.evidence_json, 8192)
        template = self.action_id.split('/')[0]
        if template not in UNITS or self.action_id != f'{template}/{self.resource_id}/v2':
            raise ValueError('invalid packet action')
        # Raw measurements belong in the raw capture store, never paid packets.
        if self.raw_json != '{}':
            raise ValueError('v2 packets cannot expose raw collector payloads')
        if self.availability == 'OK':
            if evidence.get('units') != UNITS[template]:
                raise ValueError('measurement units mismatch')
            integer(evidence.get('sample_count'), 0 if template == 'logs' else 1)
            if evidence.get('resource_id') != self.resource_id:
                raise ValueError('measurement resource mismatch')
            allowed = {
                'status': {'state'},
                'cpu': {'median', 'p95', 'high_fraction', 'coverage', 'quota_cores'},
                'memory': {'median_bytes', 'p95_bytes', 'limit_bytes', 'limit_fraction', 'peak_fraction', 'growth_bytes_per_second', 'growth_span_seconds', 'coverage'},
                'request_errors': {'operation', 'completed', 'server_errors', 'error_fraction', 'transport_failures', 'attempts', 'semantic_failures', 'transport_fraction', 'achieved_rate_per_second'},
                'latency': {'operation', 'median_ms', 'p95_ms', 'timeout_count', 'failed_count', 'coverage'},
                'logs': {'peers', 'event_counts', 'truncated', 'lines', 'warnings', 'max_lines', 'max_bytes'},
            }[template] | {'units', 'sample_count', 'resource_id'}
            if set(evidence) - allowed:
                raise ValueError('packet reveals undeclared evidence fields')
            required = {
                'status': {'state'},
                'cpu': {'median', 'p95', 'high_fraction', 'coverage', 'quota_cores'},
                'memory': {'median_bytes', 'p95_bytes', 'limit_bytes', 'limit_fraction', 'peak_fraction', 'growth_bytes_per_second', 'growth_span_seconds', 'coverage'},
                'request_errors': {'operation', 'completed', 'server_errors', 'transport_failures', 'attempts', 'semantic_failures', 'achieved_rate_per_second'},
                'latency': {'operation', 'median_ms', 'p95_ms', 'timeout_count', 'failed_count', 'coverage'},
                'logs': {'peers', 'event_counts', 'truncated', 'lines', 'warnings', 'max_lines', 'max_bytes'},
            }[template]
            if not required <= set(evidence):
                raise ValueError('packet omits required measurement fields')
            fractions = {'high_fraction', 'coverage', 'limit_fraction', 'peak_fraction', 'error_fraction', 'transport_fraction'}
            integers = {'median_bytes', 'p95_bytes', 'limit_bytes', 'completed', 'server_errors', 'transport_failures', 'timeout_count', 'attempts', 'semantic_failures', 'failed_count', 'max_lines', 'max_bytes'}
            for name in set(evidence) - {'units', 'sample_count', 'resource_id', 'state', 'operation', 'peers', 'event_counts', 'truncated', 'lines', 'warnings'}:
                if name in integers:
                    integer(evidence[name], 1 if name == 'limit_bytes' else 0)
                else:
                    real(evidence[name], 0, 1 if name in fractions - {'limit_fraction', 'peak_fraction'} else None)
            if 'quota_cores' in evidence and evidence['quota_cores'] == 0:
                raise ValueError('CPU quota must be positive')
            if template in {'cpu', 'memory'} and (evidence['coverage'] < .8 or evidence['sample_count'] < 48):
                raise ValueError('insufficient container measurement coverage')
            if template == 'latency' and evidence['sample_count'] < 20:
                raise ValueError('insufficient successful latency samples')
            if template == 'status':
                state = evidence['state']
                expected = 'RUNNING' if state == 'running' else 'STOPPED' if state in {'exited', 'dead'} else 'OTHER_STATE'
                if type(state) is not str or self.response_code != expected:
                    raise ValueError('state and response disagree')
            if template in {'request_errors', 'latency'} and evidence.get('operation') != 'checkout':
                raise ValueError('measurement operation mismatch')
            if template == 'request_errors' and ('completed' not in evidence or
                    evidence.get('server_errors', 0) > evidence['completed']):
                raise ValueError('invalid completed request denominator')
            if template == 'request_errors':
                attempts = evidence.get('attempts', evidence['completed'] + evidence.get('transport_failures', 0))
                if attempts != evidence['completed'] + evidence.get('transport_failures', 0):
                    raise ValueError('invalid request attempt denominator')
                if 'error_fraction' in evidence and evidence['completed'] < 20:
                    raise ValueError('insufficient HTTP denominator')
                if 'transport_fraction' in evidence and attempts < 20:
                    raise ValueError('insufficient transport denominator')
                if evidence['semantic_failures'] > evidence['completed'] or attempts < 20:
                    raise ValueError('invalid request counts')
                if evidence['completed'] >= 20 and 'error_fraction' not in evidence or 'transport_fraction' not in evidence:
                    raise ValueError('request fraction missing for valid denominator')
                if 'error_fraction' in evidence and abs(evidence['error_fraction'] - evidence['server_errors'] / evidence['completed']) > 1e-9:
                    raise ValueError('HTTP fraction disagrees with counts')
                if abs(evidence['transport_fraction'] - evidence['transport_failures'] / attempts) > 1e-9:
                    raise ValueError('transport fraction disagrees with counts')
            if template == 'logs':
                peers = evidence.get('peers', [])
                if type(peers) is not list or len(peers) != len(set(peers)):
                    raise ValueError('invalid log peers')
                for peer in peers:
                    identifier(peer)
                counts = evidence.get('event_counts', {})
                if type(counts) is not dict:
                    raise ValueError('invalid event counts')
                for key, count in counts.items():
                    identifier(key)
                    integer(count)
                if 'truncated' in evidence and type(evidence['truncated']) is not bool:
                    raise ValueError('invalid truncation flag')
                for field in ('lines', 'warnings'):
                    values = evidence.get(field, [])
                    if type(values) is not list or len(values) > 20 or any(type(v) is not str for v in values):
                        raise ValueError('invalid bounded log text')
                if len(canonical(evidence.get('lines', [])).encode()) > 7000:
                    raise ValueError('log explanation exceeds packet bound')
                if evidence['max_lines'] != 20 or evidence['max_bytes'] != 8192 or len(evidence['lines']) != evidence['sample_count']:
                    raise ValueError('log bounds or sample count changed')
        elif evidence:
            raise ValueError('unavailable packet cannot reveal measurements')

    def validate_context(self, spec, incident):
        super().validate_context(spec, incident)
        if not isinstance(spec, ToolSpecV2) or self.encoding_hash != spec.encoding_hash:
            raise ValueError('packet encoding mismatch')
        if self.availability == 'OK' and spec.template == 'logs':
            if any(f'DEPENDENCY_ERROR/{p}' not in spec.responses for p in json.loads(self.evidence_json).get('peers', [])):
                raise ValueError('unregistered log peer')


@dataclass(frozen=True)
class IncidentPublicV2(IncidentPublic):
    schema_version: int = 2
    task_id: str = ''
    encoding_hash: str = ''
    objective_hash: str = ''
    alert_context_json: str = '{}'

    def __post_init__(self):
        super().__post_init__()
        if type(self.schema_version) is not int or self.schema_version != 2:
            raise ValueError('unsupported incident schema')
        if self.task_id not in PROFILES:
            raise ValueError('unregistered task')
        sha256(self.encoding_hash)
        sha256(self.objective_hash)
        # The first comparison holds entry context constant.
        if object_json(self.alert_context_json) != {}:
            raise ValueError('alert context requires a separate registered experiment')
        if (timestamp(self.window_end) - timestamp(self.window_start)).total_seconds() != 120:
            raise ValueError('v2 observation window must be 120 seconds')

    def validate_context(self, topology, catalogue):
        super().validate_context(topology, catalogue)
        if self.entry_resource != 'front-end' or any(not isinstance(s, ToolSpecV2) or s.encoding_hash != self.encoding_hash for s in catalogue):
            raise ValueError('v2 incident encoding or entry mismatch')


@dataclass(frozen=True)
class CauseSpec(JsonRecord):
    cause_id: str
    cause_kind: str
    resource_id: str
    caller_id: str
    dependency_id: str

    def __post_init__(self):
        identifier(self.cause_id)
        if self.cause_kind == 'resource':
            identifier(self.resource_id)
            valid = self.cause_id == f'resource/{self.resource_id}' and not self.caller_id and not self.dependency_id
        elif self.cause_kind == 'edge':
            identifier(self.caller_id)
            identifier(self.dependency_id)
            valid = self.cause_id == f'edge/{self.caller_id}/{self.dependency_id}' and not self.resource_id
        else:
            valid = self.cause_kind == 'healthy' and self.cause_id == 'NO_FAULT' and not any((self.resource_id, self.caller_id, self.dependency_id))
        if not valid:
            raise ValueError('invalid cause identity')

    def validate_topology(self, topology):
        if self.cause_kind == 'resource' and self.resource_id not in topology.resources:
            raise ValueError('cause resource absent')
        if self.cause_kind == 'edge' and (self.caller_id, self.dependency_id) not in topology.calls:
            raise ValueError('cause path absent')


CAUSES = tuple(CauseSpec(f'resource/{r}', 'resource', r, '', '')
               for r in ('catalogue-db', 'orders', 'payment', 'carts')) + (
    CauseSpec('edge/orders/payment', 'edge', '', 'orders', 'payment'),
    CauseSpec('NO_FAULT', 'healthy', '', '', ''),)
PROFILES = {
    'sockshop_adaptive_v2a': ('resource/catalogue-db', 'resource/orders', 'resource/payment', 'NO_FAULT'),
    'sockshop_adaptive_v2b': ('resource/catalogue-db', 'resource/orders', 'resource/payment', 'edge/orders/payment', 'NO_FAULT'),
    'sockshop_adaptive_v2c': ('resource/catalogue-db', 'resource/orders', 'resource/payment', 'edge/orders/payment', 'resource/carts', 'NO_FAULT'),
}


@dataclass(frozen=True)
class IncidentPrivateV2(JsonRecord):
    schema_version: int
    incident_id: str
    cause_id: str
    cause_kind: str
    mechanism: str
    affected_resources: tuple
    injector_receipt_json: str
    verification_json: str
    recovery_json: str

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 2:
            raise ValueError('unsupported private schema')
        identifier(self.incident_id)
        unique(self.affected_resources)
        cause = next((c for c in CAUSES if c.cause_id == self.cause_id), None)
        if cause is None or cause.cause_kind != self.cause_kind:
            raise ValueError('unregistered cause')
        permitted = {'healthy': {'healthy'}, 'edge': {'dependency_latency'},
                     'resource': {'container_stopped'}}[cause.cause_kind].copy()
        if cause.resource_id == 'payment':
            permitted.add('cpu_contention')
        if cause.resource_id == 'carts':
            permitted.add('memory_pressure')
        if self.mechanism not in permitted:
            raise ValueError('mechanism does not match cause')
        if self.cause_kind == 'healthy' and self.affected_resources:
            raise ValueError('healthy cause cannot have affected resources')
        if self.cause_kind == 'resource' and cause.resource_id not in self.affected_resources:
            raise ValueError('affected resources omit cause resource')
        if self.cause_kind == 'edge' and cause.caller_id not in self.affected_resources:
            raise ValueError('affected resources omit the observed caller')
        for value in (self.injector_receipt_json, self.verification_json, self.recovery_json):
            object_json(value)

    def validate_context(self, topology, task_id):
        cause = next(c for c in CAUSES if c.cause_id == self.cause_id)
        cause.validate_topology(topology)
        if self.cause_id not in PROFILES.get(task_id, ()) or not set(self.affected_resources) <= set(topology.resources):
            raise ValueError('private cause outside profile')
        expected = {'resource/catalogue-db': 'container_stopped', 'resource/orders': 'container_stopped',
                    'resource/payment': 'cpu_contention', 'resource/carts': 'memory_pressure',
                    'edge/orders/payment': 'dependency_latency', 'NO_FAULT': 'healthy'}
        if self.mechanism != expected[self.cause_id]:
            raise ValueError('mechanism outside the registered profile')


def decode_record(payload, kind):
    """Missing schema_version means v1; unknown explicit versions fail closed."""
    from .contracts import ToolSpec, IncidentPrivate
    old, new = {'tool': (ToolSpec, ToolSpecV2), 'packet': (ToolResult, ToolResultV2),
                'public': (IncidentPublic, IncidentPublicV2), 'private': (IncidentPrivate, IncidentPrivateV2)}[kind]
    if type(payload) is not dict:
        raise ValueError('record must be an object')
    version = payload.get('schema_version', 1)
    if type(version) is not int or version not in (1, 2):
        raise ValueError('unsupported record schema')
    return (old if version == 1 else new).from_dict(payload)
