from statistics import median
from ..contracts import real
from .common import covered, percentile, unavailable_summary


def summarize(samples, resource, container_id, quota, start, end, threshold, interval=2):
    real(quota, 0)
    real(threshold, 0)
    if not quota:
        raise ValueError('verified CPU quota required')
    valid = []
    for sample in samples:
        if sample['resource_id'] != resource or sample['container_id'] != container_id or sample.get('units') != 'docker_cpu_percent':
            raise ValueError('CPU source identity or units mismatch')
        if sample.get('availability', 'OK') != 'OK':
            continue
        if sample['quota_cores'] != quota:
            raise ValueError('CPU quota changed within capture')
        real(sample['cpu_percent'])
        valid.append(sample)
    coverage, sufficient = covered(valid, start, end, interval)
    if not sufficient:
        return unavailable_summary(samples, 'cpu')
    values = [s['cpu_percent'] / (100 * quota) for s in valid]
    high = sum(v > threshold for v in values) / len(values)
    return 'OK', 'HIGH' if high >= .5 else 'NORMAL', {
        'units': 'quota_fraction', 'resource_id': resource, 'sample_count': len(values),
        'median': median(values), 'p95': percentile(values), 'high_fraction': high,
        'coverage': coverage, 'quota_cores': quota}, ''
