import re
import json
from ..backends.loki import classify


def summarize(lines, resource, peers, max_lines=20, max_bytes=8192, source_truncated=False, warnings=()):
    if type(lines) is not list or any(type(line) is not str for line in lines):
        raise ValueError('invalid log lines')
    # Query time filtering happens in the adapter. Controller entries are excluded.
    lines = [line for line in lines if not re.search(r'rca[-_]fault[-_]controller|rca[-_]helper', line, re.I)]
    selected, used, truncated = [], 0, source_truncated or len(lines) > max_lines
    for line in lines[:max_lines]:
        # Leave room for metadata and JSON escaping inside the packet bound.
        if used + len(json.dumps(line).encode()) > min(max_bytes, 6500):
            truncated = True
            break
        selected.append(line)
        used += len(json.dumps(line).encode())
    found, counts = set(), {}
    for line in selected:
        code = classify([line])
        if code == 'NO_RELEVANT_ERROR':
            continue
        counts[code] = counts.get(code, 0) + 1
        if code == 'DEPENDENCY_ERROR':
            for peer in peers:
                # Exact hostname boundaries prevent payment-db matching payment.
                if re.search(r'(?<![A-Za-z0-9_.-])' + re.escape(peer) + r'(?:\.adaptive-v2)?(?![A-Za-z0-9_.-])', line, re.I):
                    found.add(peer)
    code = classify(selected)
    if len(found) == 1:
        code = 'DEPENDENCY_ERROR/' + next(iter(found))
    return 'OK', code, {'resource_id': resource, 'units': 'log_events',
                        'sample_count': len(selected), 'peers': sorted(found),
                        'event_counts': counts, 'truncated': bool(truncated), 'lines': selected,
                        'max_lines': max_lines, 'max_bytes': max_bytes,
                        'warnings': [str(value)[:200] for value in warnings][:20]}, ''
