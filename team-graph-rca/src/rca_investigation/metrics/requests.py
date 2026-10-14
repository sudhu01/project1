"""Completed-server 5xx fraction; transport and business failures stay separate."""
from ..contracts import integer, real


def summarize(observations, resource='front-end', operation='checkout', threshold=.05, minimum=20,
              transport_threshold=.05, window_seconds=120):
    integer(minimum, 1)
    real(threshold, 0, 1)
    real(transport_threshold, 0, 1)
    real(window_seconds, 1)
    completed, server_errors, transport, semantic = 0, 0, 0, 0
    for item in observations:
        if item['resource_id'] != resource or item['operation'] != operation:
            raise ValueError('request observation identity mismatch')
        if item['outcome'] == 'transport_failure':
            if item.get('http_status') is not None:
                raise ValueError('transport failure cannot have server response')
            transport += 1
            continue
        status = item['http_status']
        integer(status, 100)
        if status > 599 or item['outcome'] not in {'completed', 'business_failure'}:
            raise ValueError('invalid request outcome')
        completed += 1
        server_errors += 500 <= status <= 599
        semantic += item['outcome'] == 'business_failure'
    attempts = completed + transport
    http_valid, transport_valid = completed >= minimum, attempts >= minimum
    high_http = http_valid and server_errors / completed > threshold
    high_transport = transport_valid and transport / attempts > transport_threshold
    if not (high_http or high_transport) and not http_valid:
        return 'NO_DATA', 'UNAVAILABLE', {}, 'insufficient_completed_requests'
    code = ('MIXED_FAILURES' if high_http and high_transport else 'HIGH_HTTP_ERRORS' if high_http
            else 'HIGH_TRANSPORT_FAILURES' if high_transport else 'NORMAL')
    evidence = {
        'resource_id': resource, 'units': 'fraction', 'sample_count': attempts,
        'operation': operation, 'completed': completed, 'server_errors': server_errors,
        'attempts': attempts, 'semantic_failures': semantic, 'transport_failures': transport,
        'achieved_rate_per_second': attempts / window_seconds}
    if http_valid:
        evidence['error_fraction'] = server_errors / completed
    if transport_valid:
        evidence['transport_fraction'] = transport / attempts
    return 'OK', code, evidence, ''


def counter_delta(samples):
    """Account for a reset without treating negative differences as successes."""
    if len(samples) < 2:
        raise ValueError('insufficient counter samples')
    for number in samples:
        real(number)
    return sum(b - a if b >= a else b for a, b in zip(samples, samples[1:]))
