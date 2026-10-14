"""Exact Compose identities and read-only bounded Docker evidence."""
from datetime import datetime, timezone
import json
import re
import subprocess
from time import perf_counter
from ..contracts import ToolResult, canonical, timestamp


def compose_rows(output):
    try:
        decoded = json.loads(output)
        rows = decoded if isinstance(decoded, list) else [decoded]
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in output.splitlines() if line.strip()]
    if any(type(row) is not dict for row in rows):
        raise ValueError('invalid Compose JSON')
    return rows


def resolve_containers(rows, resources, project='rca-lab'):
    result = {}
    for resource in resources:
        matches = [row for row in rows if row.get('Service') == resource and row.get('Project') == project]
        if len(matches) != 1 or not re.fullmatch(r'[a-f0-9]{12,64}', str(matches[0].get('ID', ''))):
            raise ValueError(f'exactly one valid container required for {resource}')
        result[resource] = matches[0]['ID']
    return result


class DockerBackend:
    version = 'docker_status_v1'

    def __init__(self, specs, project='rca-lab', runner=subprocess.run, now=None):
        if project != 'rca-lab':
            raise ValueError('the approved experiment project is rca-lab')
        self.specs = {s.action_id: s for s in specs}
        self.project, self.runner = project, runner
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.incident = None
        self.cache = {}

    def _run(self, argv):
        result = self.runner(argv, capture_output=True, text=True, timeout=5, shell=False)
        if result.returncode != 0 or len(result.stdout.encode()) > 65536:
            raise RuntimeError('Docker command failed or exceeded output limit')
        return result.stdout

    def capabilities(self, topology):
        return {r: ('status',) for r in topology.resources}

    def begin(self, incident):
        output = self._run(['docker', 'compose', '-p', self.project, 'ps', '--all', '--no-trunc', '--format', 'json'])
        self.containers = resolve_containers(compose_rows(output), {s.resource_id for s in self.specs.values()}, self.project)
        self.incident, self.cache = incident, {}

    def execute(self, spec):
        if self.incident is None or self.specs.get(spec.action_id) != spec:
            raise ValueError('backend not begun or unapproved action')
        if spec.action_id in self.cache:
            return self.cache[spec.action_id]
        start = perf_counter()
        sample = self.now().isoformat()
        # Docker inspection cannot query historical state. Refuse stale windows.
        if spec.template != 'status' or not timestamp(self.incident.window_start) <= timestamp(sample) <= timestamp(self.incident.window_end):
            availability, code, raw = 'UNSUPPORTED', 'UNAVAILABLE', {'reason': 'status requires a current capture window'}
            sample = self.incident.window_end
        else:
            try:
                container_id = self.containers[spec.resource_id]
                payload = json.loads(self._run(['docker', 'inspect', container_id]))
                if len(payload) != 1 or payload[0].get('Id') != container_id:
                    raise ValueError('inspection identity mismatch')
                labels = payload[0]['Config']['Labels']
                if labels.get('com.docker.compose.service') != spec.resource_id or labels.get('com.docker.compose.project') != self.project:
                    raise ValueError('inspection Compose labels mismatch')
                state = payload[0]['State']['Status']
                code = 'RUNNING' if state == 'running' else 'STOPPED' if state in ('exited', 'dead') else 'OTHER_STATE'
                availability, raw = 'OK', {'state': state, 'container_id': container_id, 'exit_code': payload[0]['State'].get('ExitCode')}
            except subprocess.TimeoutExpired:
                availability, code, raw = 'TIMEOUT', 'UNAVAILABLE', {'reason': 'docker_timeout'}
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                availability, code, raw = 'ERROR', 'UNAVAILABLE', {'reason': 'docker_inspection_failed'}
        packet = ToolResult(spec.action_id, spec.resource_id, availability, code, self.incident.window_start,
                            self.incident.window_end, sample, 'docker-inspect', round((perf_counter()-start)*1000), canonical(raw))
        self.cache[spec.action_id] = packet
        return packet

    def close(self):
        self.incident = None
        self.cache.clear()


def cpu_sample(container_id, runner=subprocess.run):
    """One timestamped raw sample. A capture controller must sample periodically."""
    if not re.fullmatch(r'[a-f0-9]{12,64}', container_id):
        raise ValueError('invalid container ID')
    result = runner(['docker', 'stats', '--no-stream', '--format', '{{json .}}', container_id],
                    capture_output=True, text=True, timeout=5, shell=False)
    if result.returncode != 0 or len(result.stdout) > 4096:
        raise RuntimeError('Docker CPU measurement failed')
    row = json.loads(result.stdout)
    value = float(row['CPUPerc'].removesuffix('%'))
    if not value >= 0 or not value < float('inf'):
        raise ValueError('invalid CPU percentage')
    return {'sample_time': datetime.now(timezone.utc).isoformat(), 'container_id': container_id,
            'cpu_percent': value, 'units': 'docker_cpu_percent'}
