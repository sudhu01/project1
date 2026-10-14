from statistics import median
from ..contracts import real, timestamp


def unavailable_summary(samples, family):
    observed = {row.get('availability', 'OK') for row in samples}
    if observed and observed <= {'TIMEOUT', 'ERROR', 'UNSUPPORTED'}:
        status = next(iter(observed)) if len(observed) == 1 else 'ERROR'
        return status, 'UNAVAILABLE', {}, family + '_collector_failed'
    return 'NO_DATA', 'UNAVAILABLE', {}, 'insufficient_' + family + '_coverage'


def percentile(values, fraction=.95):
    values = sorted(values)
    if not values:
        raise ValueError('empty samples')
    index = (len(values) - 1) * fraction
    lo = int(index)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def covered(samples, start, end, interval=2, minimum=.8):
    start, end = timestamp(start), timestamp(end)
    duration = (end - start).total_seconds()
    real(interval, 0)
    if duration <= 0 or interval == 0:
        raise ValueError('invalid sampling window')
    times = sorted(timestamp(s['sample_time']) for s in samples)
    if len(set(times)) != len(times) or any(t < start or t > end for t in times):
        raise ValueError('duplicate or out-of-window sample')
    gaps = [(b - a).total_seconds() for a, b in zip([start] + times, times + [end])]
    coverage = min(1., len(times) * interval / duration)
    return coverage, len(times) >= 2 and coverage >= minimum and max(gaps) <= interval * 3
