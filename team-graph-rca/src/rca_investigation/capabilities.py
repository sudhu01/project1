"""Read-only v2 instrumentation inventory. Never derives capability from labels."""
from datetime import datetime, timezone
from pathlib import Path
import json
import shutil
import subprocess
from urllib.parse import urlparse
import requests
import yaml
from .contracts import digest
from .dataset import atomic_json
from .backends.docker import compose_rows, resolve_containers
from .backends.http import bounded_json


RESOURCES = ('front-end', 'catalogue', 'catalogue-db', 'orders', 'payment', 'carts', 'carts-db')
PRICES = {'status': 1, 'cpu': 2, 'request_errors': 2, 'logs': 2, 'latency': 2, 'memory': 2}


def inventory(root, runner=subprocess.run, session=None):
    root = Path(root)
    docker = shutil.which('docker')
    report = {'schema_version': 2, 'collected_at': datetime.now(timezone.utc).isoformat(),
              'project': 'rca-lab', 'passed': False, 'prices': PRICES, 'resources': {}, 'errors': []}
    if not docker:
        report['errors'].append('docker_executable_unavailable')
        return report

    def run(argv):
        result = runner([docker] + argv, cwd=root, capture_output=True, text=True, timeout=15, shell=False)
        if result.returncode != 0 or len(result.stdout.encode()) > 1048576:
            raise RuntimeError('bounded Docker inventory failed')
        return result.stdout

    try:
        rows = compose_rows(run(['compose', '-p', 'rca-lab', 'ps', '--all', '--no-trunc', '--format', 'json']))
        identities = resolve_containers(rows, RESOURCES)
        inspected = json.loads(run(['inspect', *identities.values()]))
        if len(inspected) != len(identities):
            raise ValueError('incomplete inspection')
        by_id = {row['Id']: row for row in inspected}
        for resource, cid in identities.items():
            row = by_id[cid]
            labels = row['Config']['Labels']
            if labels.get('com.docker.compose.project') != 'rca-lab' or labels.get('com.docker.compose.service') != resource:
                raise ValueError('container identity mismatch')
            host = row['HostConfig']
            quota = host.get('NanoCpus', 0) / 1e9
            if not quota and host.get('CpuQuota', 0) > 0 and host.get('CpuPeriod', 0) > 0:
                quota = host['CpuQuota'] / host['CpuPeriod']
            memory = host.get('Memory', 0)
            report['resources'][resource] = {
                'compose_service': resource, 'container_id': cid, 'container_name': row['Name'],
                'running': row['State']['Running'], 'cpu_quota_cores': quota or None,
                'cpu_period_microseconds': host.get('CpuPeriod', 0), 'cpu_quota_microseconds': host.get('CpuQuota', 0),
                'memory_limit_bytes': memory or None, 'capabilities': ['status'],
                'metric_selectors': [], 'log_selector': None, 'workload_operations': [],
                'metric_names': [],
            }
        # A live bounded sample proves stats availability independently of fault state.
        stats = compose_rows(run(['stats', '--no-stream', '--no-trunc', '--format', '{{json .}}', *identities.values()]))
        stats_ids = {row.get('ID', row.get('Container')) for row in stats}
        for item in report['resources'].values():
            if item['running'] and item['container_id'] in stats_ids:
                if item['cpu_quota_cores']:
                    item['capabilities'].append('cpu')
                if item['memory_limit_bytes']:
                    item['capabilities'].append('memory')
    except (OSError, subprocess.TimeoutExpired, RuntimeError, ValueError, KeyError, TypeError) as exc:
        report['errors'].append(f'docker_inventory:{type(exc).__name__}')

    http = session or requests.Session()
    try:
        targets = bounded_json(http, 'http://localhost:9090/api/v1/targets', {})
        for target in targets['data']['activeTargets']:
            labels = target.get('labels', {})
            resource = urlparse(target.get('scrapeUrl', '')).hostname
            if resource not in report['resources'] or target.get('health') != 'up':
                continue
            selector = '{' + ','.join(key + '=' + json.dumps(labels[key]) for key in ('job', 'instance')) + '}'
            names = bounded_json(http, 'http://localhost:9090/api/v1/series', {'match[]': selector})
            item = report['resources'][resource]
            item['metric_selectors'].append(selector)
            item['metric_names'] = sorted({series['__name__'] for series in names['data']})
        # Verify data, not merely the existence of a Loki label name.
        for resource, item in report['resources'].items():
            selector = '{container=' + json.dumps(item['container_name']) + '}'
            payload = bounded_json(http, 'http://localhost:3100/loki/api/v1/query_range',
                                   {'query': selector, 'since': '1h', 'limit': 1, 'direction': 'backward'})
            streams = payload['data']['result']
            if payload.get('status') == 'success' and any(stream.get('values') and stream.get('stream', {}).get('container') == item['container_name'] for stream in streams):
                item['log_selector'] = selector
                item['capabilities'].append('logs')
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        report['errors'].append(f'telemetry_inventory:{type(exc).__name__}')
    finally:
        if session is None:
            http.close()
    report['passed'] = (not report['errors'] and len(report['resources']) == len(RESOURCES)
                        and all(item['running'] and item['cpu_quota_cores'] and item['memory_limit_bytes']
                                for item in report['resources'].values()))
    report['request_metrics_note'] = 'No request or latency action is enabled until matching counters or checkout observations are verified.'
    proof = root / 'runs/adaptive_v2/checkout-verification.json'
    if proof.is_file() and 'front-end' in report['resources']:
        verification = json.loads(proof.read_text(encoding='utf-8'))
        if verification.get('passed') and len(verification.get('requests', [])) >= 10:
            item = report['resources']['front-end']
            item['capabilities'] += ['request_errors', 'latency']
            item['workload_operations'] = ['checkout']
            item['workload_source'] = 'workload_checkout_observations_v2'
            report['checkout_verification_hash'] = digest(verification)
    report['hash'] = digest(report)
    return report


def save_inventory(root, report):
    root = Path(root)
    atomic_json(root / 'runs/adaptive_v2/capabilities.json', report)
    # Keep failed inspection artifacts out of the approved registry.
    if report['passed']:
        (root / 'configs/resources_sockshop_v2.yaml').write_text(yaml.safe_dump(report, sort_keys=True), encoding='utf-8')
