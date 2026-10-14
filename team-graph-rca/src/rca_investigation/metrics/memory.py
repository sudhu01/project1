from statistics import median
from ..contracts import integer, real, timestamp
from .common import covered, percentile, unavailable_summary


def robust_growth(samples):
    """Median long-span pair slopes resist spikes and allocate/release cycles."""
    rows = sorted(samples, key=lambda s: timestamp(s['sample_time']))
    if len(rows) < 2:
        return 0., 0.
    first = timestamp(rows[0]['sample_time'])
    times = [(timestamp(row['sample_time']) - first).total_seconds() for row in rows]
    span = times[-1]
    slopes = [(right['memory_bytes'] - left['memory_bytes']) / (times[j] - times[i])
              for i, left in enumerate(rows) for j, right in enumerate(rows[i + 1:], i + 1)
              if times[j] - times[i] >= max(30, span / 2)]
    return (max(0., median(slopes)) if slopes else 0.), span


def summarize(samples, resource, container_id, limit, start, end, pressure=.8, growth=1048576, interval=2):
    integer(limit, 1)
    real(pressure, 0, 1)
    real(growth)
    valid = []
    for sample in samples:
        if sample['resource_id'] != resource or sample['container_id'] != container_id:
            raise ValueError('memory identity mismatch')
        if sample.get('availability', 'OK') != 'OK':
            continue
        if sample['memory_limit_bytes'] != limit or sample['memory_source'] != 'docker_cli_stats_linux_v2':
            raise ValueError('memory source or limit changed')
        integer(sample['memory_bytes'])
        valid.append(sample)
    coverage, sufficient = covered(valid, start, end, interval)
    if not sufficient:
        return unavailable_summary(samples, 'memory')
    valid.sort(key=lambda s: timestamp(s['sample_time']))
    values = [s['memory_bytes'] for s in valid]
    slope, span = robust_growth(valid)
    pressured = median(values) / limit > pressure
    growing = span >= 90 and slope > growth
    code = 'PRESSURE_AND_GROWTH' if pressured and growing else 'PRESSURE' if pressured else 'GROWTH' if growing else 'NORMAL'
    # Negative growth remains raw information; the public growth measure is positive allocation rate.
    return 'OK', code, {'resource_id': resource, 'units': 'bytes', 'sample_count': len(values),
                         'median_bytes': round(median(values)), 'p95_bytes': round(percentile(values)),
                         'limit_bytes': limit, 'limit_fraction': median(values) / limit,
                         'peak_fraction': max(values) / limit,
                         'growth_bytes_per_second': slope, 'growth_span_seconds': span, 'coverage': coverage}, ''
