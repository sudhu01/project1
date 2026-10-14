import re
from time import perf_counter
import requests
from ..contracts import ToolResult, canonical, timestamp
from .http import bounded_json


LOG_RULE_VERSION = 'bounded_keywords_v1'


def classify(lines):
    lower = '\n'.join(lines).lower()
    if re.search(r'connection refused|connection reset|dependency|dial tcp', lower):
        return 'DEPENDENCY_ERROR'
    if re.search(r'out of memory|oom|no space|resource exhausted', lower):
        return 'RESOURCE_ERROR'
    if re.search(r'error|exception|fatal', lower):
        return 'OTHER_ERROR'
    return 'NO_RELEVANT_ERROR'


class LokiBackend:
    version = 'loki_logs_v1'

    def __init__(self, specs, container_labels, *, base_url='http://localhost:3100', session=None):
        if base_url != 'http://localhost:3100':
            raise ValueError('unapproved logs endpoint')
        self.specs = {s.action_id: s for s in specs}
        self.container_labels = dict(container_labels)
        for value in self.container_labels.values():
            if not re.fullmatch(r'/?rca-lab-[A-Za-z0-9-]+-1', value):
                raise ValueError('unapproved Loki container label')
        self.base_url, self.session = base_url, session or requests.Session()
        self.incident, self.cache = None, {}

    def capabilities(self, topology):
        return {r: ('logs',) if r in self.container_labels else () for r in topology.resources}

    def begin(self, incident):
        self.incident, self.cache = incident, {}

    def execute(self, spec):
        if self.incident is None or self.specs.get(spec.action_id) != spec:
            raise ValueError('unapproved action or missing window')
        if spec.action_id in self.cache:
            return self.cache[spec.action_id]
        started = perf_counter()
        incident = self.incident
        status, code, raw = 'UNSUPPORTED', 'UNAVAILABLE', {'reason': 'no_verified_log_label'}
        if spec.template == 'logs' and spec.resource_id in self.container_labels:
            try:
                label = self.container_labels[spec.resource_id]
                start, end = (int(timestamp(v).timestamp()*1e9) for v in (incident.window_start, incident.window_end))
                payload = bounded_json(self.session, self.base_url+'/loki/api/v1/query_range',
                                       {'query': '{container='+canonical(label)+'}', 'start': str(start), 'end': str(end), 'limit': 20, 'direction': 'forward'})
                if payload['status'] != 'success' or payload['data']['resultType'] != 'streams':
                    raise ValueError('failed Loki query')
                lines = []
                for stream in payload['data']['result']:
                    if stream['stream'].get('container') != label:
                        raise ValueError('Loki container identity mismatch')
                    for sample_time, line in stream['values']:
                        if not start <= int(sample_time) <= end or type(line) is not str:
                            raise ValueError('invalid log time or line')
                        lines.append(line[:1000])
                if len(lines) > 20:
                    raise ValueError('Loki exceeded fixed line limit')
                status, code = 'OK', classify(lines)
                raw = {'lines': lines, 'container_label': label, 'rule_version': LOG_RULE_VERSION}
            except requests.Timeout:
                status, code, raw = 'TIMEOUT', 'UNAVAILABLE', {'reason': 'loki_timeout'}
            except (requests.RequestException, ValueError, KeyError, TypeError):
                status, code, raw = 'ERROR', 'UNAVAILABLE', {'reason': 'loki_failed'}
        packet = ToolResult(spec.action_id, spec.resource_id, status, code, incident.window_start, incident.window_end,
                            incident.window_end, 'loki', round((perf_counter()-started)*1000), canonical(raw))
        self.cache[spec.action_id] = packet
        return packet

    def close(self):
        self.incident = None
        self.cache.clear()
        self.session.close()
