from statistics import median
from ..contracts import integer, real
from .common import percentile


def summarize(observations, resource='front-end', operation='checkout', threshold=500, minimum=20):
    integer(minimum, 1)
    real(threshold)
    times, timeouts, attempted = [], 0, 0
    for item in observations:
        if item['resource_id'] != resource or item['operation'] != operation:
            raise ValueError('latency observation identity mismatch')
        attempted += 1
        if item['outcome'] == 'transport_failure':
            timeouts += item.get('transport_reason') == 'timeout'
            continue
        real(item['duration_ms'])
        integer(item['http_status'], 100)
        if item['http_status'] > 599 or item['outcome'] not in {'completed', 'business_failure'}:
            raise ValueError('invalid latency outcome')
        if item['outcome'] == 'completed' and 200 <= item['http_status'] < 300:
            times.append(item['duration_ms'])
    # Timed-out requests are censored, not instantaneous successful requests.
    if len(times) < minimum:
        return 'NO_DATA', 'UNAVAILABLE', {}, 'insufficient_latency_samples'
    p95 = percentile(times)
    return 'OK', 'SLOW' if p95 > threshold else 'NORMAL', {
        'resource_id': resource, 'operation': operation, 'units': 'milliseconds',
        'sample_count': len(times), 'median_ms': median(times), 'p95_ms': p95,
        'timeout_count': timeouts, 'failed_count': attempted - len(times),
        'coverage': len(times) / attempted}, ''
