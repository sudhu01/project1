"""Frozen raw split identities and train-only encoder fitting."""
from pathlib import Path
import json
from .capture_store import read_raw
from .contracts import digest, canonical
from .dataset import atomic_json
from .contracts_v2 import decode_record
from .encoding import Encoder


def split_raw(dataset, schedule_state):
    dataset = Path(dataset)
    target = dataset / 'splits.json'
    if target.exists():
        raise ValueError('raw split already frozen')
    state = json.loads(Path(schedule_state).read_text())
    if not state['complete'] or set(state['roles'].values()) != {'train', 'validation', 'test'}:
        raise ValueError('completed independently assigned train/validation/test sessions required')
    splits, hashes, class_counts, events = {role: [] for role in ('train', 'validation', 'test')}, {}, {}, set()
    for identifier in state['accepted'].values():
        payloads, raw_hash = read_raw(dataset / identifier, public_only=False)
        receipt = payloads['capture_receipt.json']
        if receipt['physical_episode'] in events:
            raise ValueError('physical incident reused')
        events.add(receipt['physical_episode'])
        role = state['roles'][receipt['capture_session']]
        splits[role].append(identifier)
        hashes[identifier] = raw_hash
        cause = decode_record(payloads['private_label.json'], 'private').cause_id
        counts = class_counts.setdefault(role, {})
        counts[cause] = counts.get(cause, 0) + 1
    if len({role for role in class_counts}) != 3 or any(set(counts) != set(class_counts['train']) for counts in class_counts.values()):
        raise ValueError('every split requires complete cause coverage')
    manifest = {'schema_version': 2, 'group': 'capture_session', 'assignments': state['roles'], 'splits': splits,
                'bundle_hashes': hashes, 'class_counts': class_counts, 'schedule_hash': state['schedule_hash']}
    manifest['hash'] = digest(manifest)
    atomic_json(target, manifest)
    return manifest


def load_raw_split(dataset, partition):
    dataset = Path(dataset)
    manifest = json.loads((dataset / 'splits.json').read_text())
    claimed = manifest.pop('hash')
    if claimed != digest(manifest) or partition not in {'train', 'validation', 'test'}:
        raise ValueError('raw split identity mismatch')
    manifest['hash'] = claimed
    all_ids = [identifier for ids in manifest['splits'].values() for identifier in ids]
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != set(manifest['bundle_hashes']):
        raise ValueError('raw split incidents overlap')
    rows = []
    for identifier in manifest['splits'][partition]:
        payloads, raw_hash = read_raw(dataset / identifier, public_only=False)
        if raw_hash != manifest['bundle_hashes'][identifier]:
            raise ValueError('frozen raw capture changed')
        rows.append((identifier, payloads))
    return manifest, rows


def fit_raw_encoder(dataset, destination):
    from .metrics.requests import summarize as errors
    manifest, rows = load_raw_split(dataset, 'train')
    records = []
    for identifier, payloads in rows:
        measurements = payloads['measurements.json']
        fitting = {'cpu': [], 'memory': [], 'latency': [], 'request_errors': [], 'transport_errors': [], 'memory_growth': []}
        by_resource = {'cpu': {}, 'memory': {}, 'memory_growth': {}}
        from .metrics.common import covered
        eligible = set()
        for resource in measurements['registry']:
            samples = [row for row in measurements['stats'] if row['resource_id'] == resource and row['availability'] == 'OK']
            _, sufficient = covered(samples, payloads['public.json']['window_start'], payloads['public.json']['window_end'])
            if sufficient:
                eligible.add(resource)
        for row in measurements['stats']:
            if row['availability'] == 'OK' and row['resource_id'] in eligible:
                fitting['cpu'].append(row['cpu_percent'] / (100 * row['quota_cores']))
                fitting['memory'].append(row['memory_bytes'] / row['memory_limit_bytes'])
                by_resource['cpu'].setdefault(row['resource_id'], []).append(row['cpu_percent'] / (100 * row['quota_cores']))
                by_resource['memory'].setdefault(row['resource_id'], []).append(row['memory_bytes'] / row['memory_limit_bytes'])
        checkout = [row for row in measurements['requests'] if row['operation'] == 'checkout']
        timings = [row['duration_ms'] for row in checkout if row['outcome'] == 'completed' and 200 <= row['http_status'] < 300]
        from .metrics.common import percentile
        # Fit a bound on the same window statistic that the paid packet reports.
        fitting['latency'] = [percentile(timings, .95)] if len(timings) >= 20 else []
        summary = errors(checkout)
        if summary[0] == 'OK':
            fitting['request_errors'] = [summary[2]['error_fraction']] if 'error_fraction' in summary[2] else []
            fitting['transport_errors'] = [summary[2]['transport_fraction']] if 'transport_fraction' in summary[2] else []
        from .metrics.memory import robust_growth
        for resource in by_resource['memory']:
            samples = [row for row in measurements['stats'] if row['resource_id'] == resource and row['availability'] == 'OK']
            rate, span = robust_growth(samples)
            if span >= 90:
                fitting['memory_growth'].append(rate)
                by_resource['memory_growth'][resource] = [rate]
        from .logs.parser import summarize as logs
        from .topology import sockshop_topology
        observed = {}
        for resource, capture in measurements.get('logs', {}).items():
            if capture['availability'] == 'OK':
                peers = [peer for caller, peer in sockshop_topology().calls if caller == resource]
                observed[resource] = logs(capture['lines'], resource, peers)[2]['peers']
        records.append({'incident_id': identifier, 'private_cause': decode_record(payloads['private_label.json'], 'private').cause_id,
                        'fitting_values': fitting, 'fitting_values_by_resource': by_resource, 'observed_log_peers': observed})
    encoder = Encoder.fit(records, manifest)
    encoder.save(destination)
    return encoder
