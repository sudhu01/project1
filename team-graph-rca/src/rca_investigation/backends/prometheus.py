"""Exact historical process-CPU evidence, separate from container CPU."""
import math
from time import perf_counter
import requests
from ..contracts import ToolResult, canonical, timestamp
from .http import bounded_json


class PrometheusBackend:
    version = 'prometheus_process_cpu_v1'

    def __init__(self, specs, instances, *, threshold=None, base_url='http://localhost:9090', session=None):
        if base_url != 'http://localhost:9090':
            raise ValueError('unapproved telemetry endpoint')
        self.specs = {s.action_id: s for s in specs}
        self.instances = dict(instances)
        self.threshold, self.base_url = threshold, base_url
        self.session = session or requests.Session()
        self.incident, self.cache = None, {}
        for resource, instance in self.instances.items():
            if instance not in {f'{resource}:80', f'{resource}:8079'}:
                raise ValueError('unapproved exact Prometheus instance')
        if threshold is not None and (not math.isfinite(threshold) or threshold < 0):
            raise ValueError('invalid fitted CPU threshold')

    def capabilities(self, topology):
        return {r: ('cpu',) if r in self.instances else () for r in topology.resources}

    def begin(self, incident):
        self.incident, self.cache = incident, {}

    def execute(self, spec):
        if self.incident is None or self.specs.get(spec.action_id) != spec:
            raise ValueError('unapproved action or missing window')
        if spec.action_id in self.cache:
            return self.cache[spec.action_id]
        started = perf_counter()
        incident = self.incident
        status, code, raw = 'UNSUPPORTED', 'UNAVAILABLE', {'reason': 'no_verified_selector'}
        if spec.template == 'cpu' and spec.resource_id in self.instances:
            try:
                selector = self.instances[spec.resource_id]
                query = 'rate(process_cpu_seconds_total{job="sockshop",instance=' + canonical(selector) + '}[1m]) * 100'
                payload = bounded_json(self.session, self.base_url+'/api/v1/query_range',
                    {'query': query, 'start': incident.window_start, 'end': incident.window_end, 'step': 15})
                if payload['status'] != 'success' or payload.get('warnings') or payload['data']['resultType'] != 'matrix':
                    raise ValueError('failed or incomplete Prometheus response')
                rows = payload['data']['result']
                samples = []
                if len(rows) > 1:
                    raise ValueError('ambiguous CPU series')
                for row in rows:
                    if row['metric'].get('instance') != selector or row['metric'].get('job') != 'sockshop':
                        raise ValueError('Prometheus selector mismatch')
                    for sample_time, value in row['values']:
                        value = float(value)
                        if not math.isfinite(value) or value < 0 or not timestamp(incident.window_start).timestamp() <= float(sample_time) <= timestamp(incident.window_end).timestamp():
                            raise ValueError('invalid CPU sample')
                        samples.append([float(sample_time), value])
                raw = {'samples': samples[:5], 'units': 'process_cpu_percent', 'selector': selector,
                       'measurement_scope': 'application_process', 'threshold': self.threshold}
                if len(samples) < 4 or self.threshold is None:
                    status, code = 'NO_DATA', 'UNAVAILABLE'
                else:
                    status = 'OK'
                    code = 'HIGH' if sum(v for _, v in samples)/len(samples) > self.threshold else 'NORMAL'
            except requests.Timeout:
                status, code, raw = 'TIMEOUT', 'UNAVAILABLE', {'reason': 'prometheus_timeout'}
            except (requests.RequestException, ValueError, KeyError, TypeError):
                status, code, raw = 'ERROR', 'UNAVAILABLE', {'reason': 'prometheus_failed'}
        packet = ToolResult(spec.action_id, spec.resource_id, status, code, incident.window_start, incident.window_end,
                            incident.window_end, 'prometheus-process-cpu', round((perf_counter()-started)*1000), canonical(raw))
        self.cache[spec.action_id] = packet
        return packet

    def close(self):
        self.incident = None
        self.cache.clear()
        self.session.close()
