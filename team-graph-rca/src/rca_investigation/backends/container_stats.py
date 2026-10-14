"""Bounded multi-container polling with exact Docker identities."""
from datetime import datetime, timezone
import re
import subprocess
from threading import Lock
from time import perf_counter
from ..contracts import real
from .docker import compose_rows


def memory_bytes(value):
    match = re.fullmatch(r'([0-9]+(?:\.[0-9]+)?)\s*(B|kB|MB|GB|TB|KiB|MiB|GiB|TiB)', value.strip())
    if not match:
        raise ValueError('invalid Docker memory units')
    units = {'B': 1, 'kB': 1000, 'MB': 1000**2, 'GB': 1000**3, 'TB': 1000**4,
             'KiB': 1024, 'MiB': 1024**2, 'GiB': 1024**3, 'TiB': 1024**4}
    return round(float(match[1]) * units[match[2]])


class ContainerStatsCollector:
    version = 'docker_cli_stats_linux_v2'

    def __init__(self, registry, runner=subprocess.run, now=None):
        self.registry, self.runner = registry, runner
        self.now = now or (lambda: datetime.now(timezone.utc).isoformat())
        self.lock = Lock()
        for resource, item in registry.items():
            if not re.fullmatch('[a-f0-9]{64}', item['container_id']):
                raise ValueError('full Docker identity required')
            real(item['cpu_quota_cores'])
            if item['cpu_quota_cores'] == 0 or not item['memory_limit_bytes'] > 0:
                raise ValueError('verified quota and memory limit required')

    def poll(self):
        if not self.lock.acquire(blocking=False):
            raise RuntimeError('overlapping stats collection')
        started = perf_counter()
        try:
            ids = {item['container_id']: resource for resource, item in self.registry.items()}
            try:
                result = self.runner(['docker', 'stats', '--no-stream', '--no-trunc', '--format', '{{json .}}', *ids],
                                     capture_output=True, text=True, timeout=5, shell=False)
            except (subprocess.TimeoutExpired, OSError) as exc:
                return tuple({'resource_id': resource, 'container_id': cid, 'sample_time': self.now(),
                              'availability': 'TIMEOUT' if isinstance(exc, subprocess.TimeoutExpired) else 'ERROR',
                              'units': 'docker_cpu_percent', 'reason': 'stats_collector_failed'} for cid, resource in ids.items())
            if result.returncode or len(result.stdout.encode()) > 65536:
                raise RuntimeError('bounded stats command failed')
            samples = {}
            for row in compose_rows(result.stdout):
                cid = row.get('ID', row.get('Container'))
                if cid not in ids or cid in samples:
                    raise ValueError('stats identity mismatch')
                cpu = float(row['CPUPerc'].removesuffix('%'))
                real(cpu)
                usage, limit = map(memory_bytes, row['MemUsage'].split('/'))
                item = self.registry[ids[cid]]
                if limit == 0 and usage == 0 and cpu == 0:
                    samples[cid] = {'resource_id': ids[cid], 'container_id': cid, 'availability': 'NO_DATA',
                                    'sample_time': self.now(), 'units': 'docker_cpu_percent', 'reason': 'container_not_running'}
                    continue
                if abs(limit - item['memory_limit_bytes']) > max(1024, item['memory_limit_bytes'] * .005):
                    raise ValueError('memory limit changed')
                samples[cid] = {'resource_id': ids[cid], 'container_id': cid, 'availability': 'OK',
                                'sample_time': self.now(), 'cpu_percent': cpu, 'quota_cores': item['cpu_quota_cores'],
                                'memory_bytes': usage, 'memory_limit_bytes': item['memory_limit_bytes'],
                                'memory_usage_original': row['MemUsage'],
                                'memory_source': self.version, 'units': 'docker_cpu_percent',
                                'collection_duration_ms': round((perf_counter() - started) * 1000)}
            for cid, resource in ids.items():
                if cid not in samples:
                    samples[cid] = {'resource_id': resource, 'container_id': cid, 'availability': 'NO_DATA',
                                    'sample_time': self.now(), 'units': 'docker_cpu_percent', 'reason': 'container_not_sampled'}
            return tuple(samples.values())
        finally:
            self.lock.release()
