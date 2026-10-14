"""Bounded baseline candidates and frozen-method cohort reweighting."""
from statistics import mean


def fixed_candidates(specs, maximum=32, model=None):
    if not specs or maximum < 1:
        raise ValueError('invalid fixed-sequence search budget')
    ids = tuple(s.action_id for s in specs)
    candidates = [ids, tuple(s.action_id for s in sorted(specs, key=lambda s: (s.credits, s.action_id)))]
    candidates.extend(ids[index:] + ids[:index] for index in range(min(len(ids), maximum)))
    if model is not None:
        from math import log2
        from itertools import permutations
        def entropy(probabilities):
            return -sum(p * log2(p) for p in probabilities.values() if p > 0)
        initial = entropy(model.diagnosis(()))
        information = {}
        for spec in specs:
            expected = sum(probability * entropy(model.diagnosis(((spec.action_id, symbol),)))
                           for symbol, probability in model.predictions((), spec.action_id).items() if probability > 0)
            information[spec.action_id] = (initial - expected) / spec.credits
        ranked = tuple(sorted(ids, key=lambda action: (-information[action], action)))
        candidates.insert(0, ranked)
        # Global information ranks can put correlated indicators ahead of the
        # probes needed to separate the remaining classes. Use the fitted joint
        # prototypes to build a second order with conditional information.
        indices = {action: model.action_ids.index(action) for action in ids}
        def conditional_entropy(actions):
            groups = {}
            for row, label, weight in zip(model.responses, model.labels, model.prior):
                key = tuple(row[indices[action]] for action in actions)
                labels = groups.setdefault(key, {})
                labels[label] = labels.get(label, 0.) + weight
            total = 0.
            for labels in groups.values():
                weight = sum(labels.values())
                total += weight * entropy({label: value / weight for label, value in labels.items()})
            return total
        chosen, remaining = [], list(ids)
        while remaining:
            current = conditional_entropy(chosen)
            scores = {action: (current - conditional_entropy(chosen + [action])) /
                      next(spec.credits for spec in specs if spec.action_id == action) for action in remaining}
            action = min(remaining, key=lambda value: (-scores[value], value))
            if scores[action] <= 1e-8:
                break
            chosen.append(action)
            remaining.remove(action)
        greedy = tuple(chosen) + tuple(action for action in ranked if action not in chosen)
        candidates.insert(0, greedy)
        useful = tuple(chosen[:4])
        tail = tuple(action for action in greedy if action not in useful)
        # Interleave first actions so truncating the bounded search does not
        # discard all orders beginning with the fourth useful probe.
        grouped = [[(first,) + order for order in permutations(tuple(a for a in useful if a != first))]
                   for first in useful]
        if grouped:
            candidates.extend(order + tail for batch in zip(*grouped) for order in batch)
    return tuple(dict.fromkeys(candidates))[:maximum]


def evidence_subsets(specs):
    stages = [('status',), ('status', 'cpu'), ('status', 'cpu', 'request_errors'),
              ('status', 'cpu', 'request_errors', 'logs'), ('status', 'cpu', 'request_errors', 'logs', 'latency'),
              ('status', 'cpu', 'request_errors', 'logs', 'latency', 'memory')]
    stages.extend(tuple(f for f in stages[-1] if f != excluded) for excluded in stages[-1])
    # All subsets retain the same diagnoses. Fitting and training must be separate.
    result = {}
    seen = set()
    for families in stages:
        subset = tuple(s for s in specs if s.template in families)
        identity = tuple(s.action_id for s in subset)
        if subset and identity not in seen:
            result['+'.join(sorted({s.template for s in subset}))] = subset
            seen.add(identity)
    return result


def paired_mixture_sweep(first, second, hard_causes, proportions, seed=11, samples=2000):
    """Frozen methods, unchanged priors; resample independent sessions jointly."""
    import numpy as np
    left, right = ({row['incident_id']: row for row in rows} for rows in (first, second))
    if not left or set(left) != set(right):
        raise ValueError('mixture sweep requires identical paired incidents')
    sessions = {}
    for incident, row in left.items():
        other = right[incident]
        if row['capture_session'] != other['capture_session'] or row['truth'] != other['truth']:
            raise ValueError('mixture pairing mismatch')
        sessions.setdefault(row['capture_session'], []).append((row['truth'] in hard_causes,
            row['return_value'] - other['return_value']))
    if any(not 0 <= value <= 1 for value in proportions):
        raise ValueError('invalid hard fraction')
    def estimate(rows, proportion):
        easy = [difference for hard, difference in rows if not hard]
        hard = [difference for is_hard, difference in rows if is_hard]
        return (1 - proportion) * mean(easy) + proportion * mean(hard) if easy and hard else None
    rows = [row for group in sessions.values() for row in group]
    rng, identifiers = np.random.default_rng(seed), sorted(sessions)
    reports = []
    for proportion in proportions:
        value = estimate(rows, proportion)
        if value is None:
            raise ValueError('both easy and hard outcomes required')
        bootstrap = []
        for _ in range(samples):
            sampled = [row for name in rng.choice(identifiers, len(identifiers), replace=True) for row in sessions[name]]
            result = estimate(sampled, proportion)
            if result is not None:
                bootstrap.append(result)
        reports.append({'hard_fraction': proportion, 'paired_return_difference': value,
            'session_bootstrap_95': np.quantile(bootstrap, [.025, .975]).tolist(),
            'independent_sessions': len(identifiers), 'resamples': len(bootstrap), 'seed': seed,
            'retrained': False, 'priors_changed': False})
    return reports


def mixture_sweep(easy_differences, hard_differences, proportions):
    if not easy_differences or not hard_differences or any(not 0 <= p <= 1 for p in proportions):
        raise ValueError('paired easy and hard outcomes required')
    easy, hard = mean(easy_differences), mean(hard_differences)
    return [{'hard_fraction': p, 'paired_return_difference': (1-p)*easy+p*hard} for p in proportions]
