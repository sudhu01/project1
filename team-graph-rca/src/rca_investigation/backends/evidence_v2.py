"""One-family projection of complete frozen raw observations."""
import json
from datetime import datetime, timezone
from ..contracts import timestamp
from ..encoding import encode_packet
from ..metrics import cpu, memory, requests, latency
from ..logs.parser import summarize as logs
from .http import bounded_json


def collect_logs(session, registry, start, end):
    result = {}
    for resource, item in registry.items():
        if not item.get('log_selector'):
            result[resource] = {'availability': 'UNSUPPORTED', 'reason': 'unverified_log_source', 'lines': []}
            continue
        try:
            first, last = (int(timestamp(value).timestamp() * 1e9) for value in (start, end))
            payload = bounded_json(session, 'http://localhost:3100/loki/api/v1/query_range',
                                   {'query': item['log_selector'] + ' |~ "(?i)error|exception|failed|failure|refused|timeout|out of memory"', 'start': str(first), 'end': str(last),
                                    'limit': 20, 'direction': 'backward'})
            if payload['status'] != 'success' or payload['data']['resultType'] != 'streams':
                raise ValueError('Loki query failed')
            lines, clipped = [], False
            for stream in payload['data']['result']:
                if stream['stream'].get('container') != item['container_name']:
                    raise ValueError('Loki identity mismatch')
                for sample, line in stream['values']:
                    if not first <= int(sample) <= last or type(line) is not str:
                        raise ValueError('Loki time or content mismatch')
                    clipped |= len(line) > 1000
                    lines.append(line[:1000])
            if len(lines) > 20:
                raise ValueError('Loki exceeded sample limit')
            result[resource] = {'availability': 'OK', 'lines': lines, 'truncated': clipped or len(lines) == 20,
                                'warnings': payload.get('warnings', []), 'query_limit': 20,
                                'collected_at': datetime.now(timezone.utc).isoformat()}
        except Exception as exc:
            import requests as http_requests
            if not isinstance(exc, (http_requests.RequestException, ValueError, KeyError, TypeError)):
                raise
            result[resource] = {'availability': 'TIMEOUT' if isinstance(exc, http_requests.Timeout) else 'ERROR',
                                'reason': 'log_collection_failed', 'lines': []}
    return result


class RawEvidenceBackend:
    version = 'raw_projection_v2'

    def __init__(self, specs, raw, registry, topology, encoder):
        self.specs = {s.action_id: s for s in specs}
        self.raw, self.registry, self.topology, self.encoder = raw, registry, topology, encoder
        self.incident = None

    def begin(self, incident):
        self.incident = incident

    def execute(self, spec):
        if self.incident is None or self.specs.get(spec.action_id) != spec:
            raise ValueError('unapproved evidence projection')
        item, resource = self.registry[spec.resource_id], spec.resource_id
        start, end = self.incident.window_start, self.incident.window_end
        sample_time = end
        samples = [row for row in self.raw.get('stats', []) if row['resource_id'] == resource]
        thresholds = self.encoder.values['thresholds']
        resource_thresholds = {family: mapping[resource] for family, mapping in self.encoder.values.get('thresholds_by_resource', {}).items() if resource in mapping}
        if spec.template == 'cpu':
            summary = cpu.summarize(samples, resource, item['container_id'], item['cpu_quota_cores'], start, end, resource_thresholds.get('cpu', thresholds['cpu']))
        elif spec.template == 'memory':
            summary = memory.summarize(samples, resource, item['container_id'], item['memory_limit_bytes'], start, end,
                                       pressure=resource_thresholds.get('memory', thresholds['memory']),
                                       growth=resource_thresholds.get('memory_growth', thresholds['memory_growth']))
        elif spec.template in {'request_errors', 'latency'}:
            params = json.loads(spec.parameters_json)
            rows = [row for row in self.raw.get('requests', []) if row['operation'] == params['operation']]
            kwargs = {'transport_threshold': thresholds['transport_errors']} if spec.template == 'request_errors' else {}
            summary = (requests if spec.template == 'request_errors' else latency).summarize(
                rows, resource, params['operation'], thresholds[spec.template], params['min_samples'], **kwargs)
        elif spec.template == 'logs':
            result = self.raw.get('logs', {}).get(resource, {'availability': 'NO_DATA', 'reason': 'missing_log_capture'})
            if result['availability'] != 'OK':
                summary = result['availability'], 'UNAVAILABLE', {}, result['reason']
            else:
                peers = self.encoder.values.get('validated_log_peers', {}).get(resource, [])
                summary = logs(result['lines'], resource, peers, source_truncated=result.get('truncated', False),
                               warnings=result.get('warnings', []))
        elif spec.template == 'status':
            row = self.raw.get('status', {}).get(resource)
            if row is None:
                summary = 'NO_DATA', 'UNAVAILABLE', {}, 'missing_status_capture'
            elif row['container_id'] != item['container_id'] or not timestamp(start) <= timestamp(row['sample_time']) <= timestamp(end):
                raise ValueError('status source or time mismatch')
            else:
                sample_time = row['sample_time']
                code = 'RUNNING' if row['state'] == 'running' else 'STOPPED' if row['state'] in {'exited', 'dead'} else 'OTHER_STATE'
                summary = 'OK', code, {'resource_id': resource, 'units': 'container_state', 'sample_count': 1, 'state': row['state']}, ''
        else:
            raise ValueError('unknown evidence family')
        return encode_packet(spec, self.incident, summary, self.version, collected_at=self.raw['collected_at'], sample_time=sample_time)

    def close(self):
        self.incident = None
